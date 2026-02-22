"""
AWS SageMaker training script for futures price prediction models.

This is a self-contained mirror of the local training pipeline, structured
for SageMaker's training job conventions:
  - Input data: /opt/ml/input/data/training/
  - Hyperparams: /opt/ml/input/config/hyperparameters.json
  - Model output: /opt/ml/model/

The script is self-contained (no imports from src/) because SageMaker training
jobs run in isolated Docker containers.

--- HOW TO LAUNCH FROM A NOTEBOOK OR LOCAL SCRIPT ---

    import sagemaker
    from sagemaker.pytorch import PyTorch

    sess = sagemaker.Session()
    role = sagemaker.get_execution_role()
    bucket = sess.default_bucket()

    # 1. Upload dataset CSVs to S3
    s3_data_uri = sess.upload_data(
        path='files/dataset',
        bucket=bucket,
        key_prefix='AAI_540_group_8/dataset'
    )

    # 2. Configure the estimator
    estimator = PyTorch(
        entry_point='scripts/sagemaker_training.py',
        role=role,
        instance_count=1,
        instance_type='ml.m5.large',      # CPU, ~$0.12/hr
        framework_version='2.1',
        py_version='py310',
        hyperparameters={
            'epochs': '200',
            'batch_size': '32',
            'lookback': '20',
            'learning_rate': '0.001',
            'weight_decay': '0.00001',
            'grad_clip_norm': '1.0',
            'early_stopping_patience': '15',
            'lr_scheduler_patience': '7',
            'lr_scheduler_factor': '0.5',
            'lr_min': '0.000001',
            'models': 'lstm,transformer,bilstm_attention',
            'tickers': 'CL=F,GC=F',
            'lstm_hidden_size': '64',
            'lstm_num_layers': '2',
            'lstm_dropout': '0.2',
            'transformer_d_model': '64',
            'transformer_nhead': '4',
            'transformer_num_layers': '2',
            'transformer_dim_ff': '128',
            'transformer_dropout': '0.1',
            'bilstm_hidden_size': '64',
            'bilstm_num_layers': '2',
            'bilstm_dropout': '0.2',
        }
    )

    # 3. Launch training job
    estimator.fit({'training': s3_data_uri})

    # 4. Download trained models
    # Model artifacts are saved to s3://{bucket}/{estimator._current_job_name}/output/model.tar.gz

--- ALTERNATIVELY: Upload pre-trained local models to S3 ---

    import boto3

    s3 = boto3.client('s3')
    model_dir = Path('files/models')
    for model_file in model_dir.rglob('*'):
        if model_file.is_file():
            s3_key = f'AAI_540_group_8/models/{model_file.relative_to(model_dir)}'
            s3.upload_file(str(model_file), bucket, s3_key)
"""
import os
import sys
import json
import logging
import time
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('sagemaker-training')

# ============================================================================
# Data Loading (self-contained copy)
# ============================================================================

COLUMNS_TO_DROP = ['Unnamed: 0', 'Date', 'adj close', 'repaired?']
FEATURE_COLUMNS = [
    'high', 'low', 'open', 'volume',
    'MA', 'EMA', 'KAMA', 'WMA', 'MidPrice',
    'BOP', 'CMO', 'MFI', 'ROC', 'WILLR',
    'AD', 'OBV', 'NATR', 'ATR', 'TRANGE', 'TSF'
]


class FuturesDataset(Dataset):
    def __init__(self, features, targets, lookback):
        self.features = features
        self.targets = targets
        self.lookback = lookback

    def __len__(self):
        return len(self.features) - self.lookback

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.lookback]
        y = self.targets[idx + self.lookback]
        return torch.FloatTensor(x), torch.FloatTensor([y])


def load_ticker_data(ticker, split, dataset_path):
    # NOTE: Inverted naming convention from the original data pipeline:
    # y_*.csv = features, x_*.csv = target
    features_df = pd.read_csv(dataset_path / f"{ticker}y_{split}.csv")
    target_df = pd.read_csv(dataset_path / f"{ticker}x_{split}.csv")

    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in features_df.columns]
    features_df = features_df.drop(columns=cols_to_drop)
    available = [c for c in FEATURE_COLUMNS if c in features_df.columns]
    features_df = features_df[available]

    target_series = target_df['close'] if 'close' in target_df.columns else target_df.iloc[:, -1]
    return features_df, target_series


def create_data_loaders(ticker, config, dataset_path):
    lookback = int(config.get('lookback', 20))
    batch_size = int(config.get('batch_size', 32))

    train_feat, train_tgt = load_ticker_data(ticker, 'dev', dataset_path)
    test_feat, test_tgt = load_ticker_data(ticker, 'test', dataset_path)
    val_feat, val_tgt = load_ticker_data(ticker, 'val', dataset_path)

    num_features = train_feat.shape[1]

    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()

    train_f = scaler_features.fit_transform(train_feat.values)
    train_t = scaler_target.fit_transform(train_tgt.values.reshape(-1, 1)).flatten()
    test_f = scaler_features.transform(test_feat.values)
    test_t = scaler_target.transform(test_tgt.values.reshape(-1, 1)).flatten()
    val_f = scaler_features.transform(val_feat.values)
    val_t = scaler_target.transform(val_tgt.values.reshape(-1, 1)).flatten()

    train_loader = DataLoader(FuturesDataset(train_f, train_t, lookback),
                              batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(FuturesDataset(test_f, test_t, lookback),
                             batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(FuturesDataset(val_f, val_t, lookback),
                            batch_size=batch_size, shuffle=False)

    return {
        'train': train_loader, 'val': val_loader, 'test': test_loader,
        'scaler_features': scaler_features, 'scaler_target': scaler_target,
        'num_features': num_features,
    }


# ============================================================================
# Model Architectures (self-contained copies)
# ============================================================================

class LSTMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        nf = int(config.get('num_features', 20))
        hs = int(config.get('lstm_hidden_size', 64))
        nl = int(config.get('lstm_num_layers', 2))
        do = float(config.get('lstm_dropout', 0.2))
        self.lstm = nn.LSTM(nf, hs, nl, batch_first=True,
                            dropout=do if nl > 1 else 0.0)
        self.dropout = nn.Dropout(do)
        self.fc = nn.Linear(hs, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        nf = int(config.get('num_features', 20))
        dm = int(config.get('transformer_d_model', 64))
        nh = int(config.get('transformer_nhead', 4))
        nl = int(config.get('transformer_num_layers', 2))
        ff = int(config.get('transformer_dim_ff', 128))
        do = float(config.get('transformer_dropout', 0.1))
        lb = int(config.get('lookback', 20))
        self.proj = nn.Linear(nf, dm)
        self.pos = nn.Parameter(torch.randn(1, lb, dm) * 0.1)
        layer = nn.TransformerEncoderLayer(dm, nh, ff, do, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, nl)
        self.fc = nn.Linear(dm, 1)

    def forward(self, x):
        x = self.proj(x) + self.pos
        return self.fc(self.enc(x).mean(dim=1))


class BiLSTMAttentionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        nf = int(config.get('num_features', 20))
        hs = int(config.get('bilstm_hidden_size', 64))
        nl = int(config.get('bilstm_num_layers', 2))
        do = float(config.get('bilstm_dropout', 0.2))
        self.bilstm = nn.LSTM(nf, hs, nl, batch_first=True,
                              bidirectional=True,
                              dropout=do if nl > 1 else 0.0)
        self.attn = nn.Linear(hs * 2, 1)
        self.dropout = nn.Dropout(do)
        self.fc = nn.Linear(hs * 2, 1)

    def forward(self, x):
        out, _ = self.bilstm(x)
        w = torch.softmax(self.attn(out), dim=1)
        ctx = (out * w).sum(dim=1)
        return self.fc(self.dropout(ctx))


MODEL_REGISTRY = {
    'lstm': LSTMModel,
    'transformer': TransformerModel,
    'bilstm_attention': BiLSTMAttentionModel,
}


# ============================================================================
# Training Loop (self-contained copy)
# ============================================================================

class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-6):
        self.patience, self.min_delta = patience, min_delta
        self.counter, self.best_loss = 0, None

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_model(model, train_loader, val_loader, config, save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    epochs = int(config.get('epochs', 200))
    lr = float(config.get('learning_rate', 0.001))
    wd = float(config.get('weight_decay', 1e-5))
    gc = float(config.get('grad_clip_norm', 1.0))

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=int(config.get('lr_scheduler_patience', 7)),
        factor=float(config.get('lr_scheduler_factor', 0.5)),
        min_lr=float(config.get('lr_min', 1e-6))
    )
    es = EarlyStopping(patience=int(config.get('early_stopping_patience', 15)))
    history = {'train_loss': [], 'val_loss': []}
    best_val = float('inf')

    for epoch in range(epochs):
        model.train()
        tl = 0.0
        tb = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gc)
            optimizer.step()
            tl += loss.item()
            tb += 1

        model.eval()
        vl = 0.0
        vb = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vl += criterion(model(xb), yb).item()
                vb += 1

        avg_tl = tl / max(tb, 1)
        avg_vl = vl / max(vb, 1)
        history['train_loss'].append(avg_tl)
        history['val_loss'].append(avg_vl)

        if avg_vl < best_val:
            best_val = avg_vl
            torch.save(model.state_dict(), save_path)

        scheduler.step(avg_vl)
        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - train: {avg_tl:.6f}, val: {avg_vl:.6f}")

        if es(avg_vl):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(torch.load(save_path, weights_only=True))
    return history


# ============================================================================
# SageMaker Entry Point
# ============================================================================

def main():
    # SageMaker environment paths
    input_dir = Path(os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
    model_dir = Path(os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    model_dir.mkdir(parents=True, exist_ok=True)

    # Read hyperparameters
    hp_path = Path('/opt/ml/input/config/hyperparameters.json')
    if hp_path.exists():
        with open(hp_path) as f:
            config = json.load(f)
    else:
        config = {}

    tickers = config.get('tickers', 'CL=F,GC=F').split(',')
    model_names = config.get('models', 'lstm,transformer,bilstm_attention').split(',')

    for ticker in tickers:
        ticker_dir = model_dir / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)

        data = create_data_loaders(ticker, config, input_dir)
        config['num_features'] = str(data['num_features'])

        for mname in model_names:
            logger.info(f"Training {mname} for {ticker}")
            model = MODEL_REGISTRY[mname](config)
            save_path = ticker_dir / f"{mname}_{ticker}.pth"

            history = train_model(model, data['train'], data['val'], config, save_path)

            # Save scalers
            joblib.dump(data['scaler_features'],
                        ticker_dir / f"{mname}_{ticker}_feature_scaler.pkl")
            joblib.dump(data['scaler_target'],
                        ticker_dir / f"{mname}_{ticker}_target_scaler.pkl")

            # Save metadata
            metadata = {
                'model_name': mname, 'ticker': ticker,
                'epochs_trained': len(history['train_loss']),
                'best_val_loss': min(history['val_loss']),
            }
            with open(ticker_dir / f"{mname}_{ticker}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Completed {mname} for {ticker}")


if __name__ == '__main__':
    main()
