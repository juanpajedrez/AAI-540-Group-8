"""
SageMaker Pipeline Evaluation Step - self-contained container script.

Runs inside a SageMaker Processing container as part of the CI/CD pipeline.
Loads a trained model, runs inference on the test set, computes regression
metrics (MSE, RMSE, MAE), and writes evaluation.json for the ConditionStep.

Input:
    /opt/ml/processing/input/model/   - model.tar.gz contents (extracted by SageMaker)
    /opt/ml/processing/input/test/    - preprocessed test data + scalers

Output:
    /opt/ml/processing/output/evaluation.json  - metrics in SageMaker PropertyFile format

Environment variables:
    TICKER      - ticker symbol (default: "CL=F")
    MODEL_NAME  - model architecture name (default: "lstm")
"""
import json
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("sagemaker-pipeline-evaluation")


# ============================================================================
# Constants
# ============================================================================

MODEL_DIR = Path("/opt/ml/processing/input/model")
TEST_DIR = Path("/opt/ml/processing/input/test")
OUTPUT_DIR = Path("/opt/ml/processing/output")


# ============================================================================
# Model Architectures (self-contained copies matching saved state_dict keys)
# Copied from sagemaker_backtest_processing.py lines 122-187 (CORRECT names)
# ============================================================================

class LSTMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        nf = int(config.get("num_features", 20))
        hs = int(config.get("lstm_hidden_size", 64))
        nl = int(config.get("lstm_num_layers", 2))
        do = float(config.get("lstm_dropout", 0.2))
        self.lstm = nn.LSTM(nf, hs, nl, batch_first=True,
                            dropout=do if nl > 1 else 0.0)
        self.dropout_layer = nn.Dropout(do)
        self.fc = nn.Linear(hs, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout_layer(out[:, -1, :]))


class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        nf = int(config.get("num_features", 20))
        dm = int(config.get("transformer_d_model", 64))
        nh = int(config.get("transformer_nhead", 4))
        nl = int(config.get("transformer_num_layers", 2))
        ff = int(config.get("transformer_dim_ff", 128))
        do = float(config.get("transformer_dropout", 0.1))
        lb = int(config.get("lookback", 20))
        # Attribute names match the saved state_dict from architectures.py
        self.input_projection = nn.Linear(nf, dm)
        self.pos_encoding = nn.Parameter(torch.randn(1, lb, dm) * 0.1)
        layer = nn.TransformerEncoderLayer(dm, nh, ff, do, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, nl)
        self.fc = nn.Linear(dm, 1)

    def forward(self, x):
        x = self.input_projection(x) + self.pos_encoding
        return self.fc(self.transformer_encoder(x).mean(dim=1))


class BiLSTMAttentionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        nf = int(config.get("num_features", 20))
        hs = int(config.get("bilstm_hidden_size", 64))
        nl = int(config.get("bilstm_num_layers", 2))
        do = float(config.get("bilstm_dropout", 0.2))
        self.bilstm = nn.LSTM(nf, hs, nl, batch_first=True,
                              bidirectional=True,
                              dropout=do if nl > 1 else 0.0)
        # Attribute name matches saved state_dict from architectures.py
        self.attention_fc = nn.Linear(hs * 2, 1)
        self.dropout_layer = nn.Dropout(do)
        self.fc = nn.Linear(hs * 2, 1)

    def forward(self, x):
        out, _ = self.bilstm(x)
        w = torch.softmax(self.attention_fc(out), dim=1)
        ctx = (out * w).sum(dim=1)
        return self.fc(self.dropout_layer(ctx))


MODEL_REGISTRY = {
    "lstm": LSTMModel,
    "transformer": TransformerModel,
    "bilstm_attention": BiLSTMAttentionModel,
}


# ============================================================================
# Dataset
# ============================================================================

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


# ============================================================================
# Evaluation
# ============================================================================

def find_model_file(model_dir, ticker, model_name):
    """Find the .pth model file in the model directory.

    SageMaker training output structure: model.tar.gz is extracted into
    MODEL_DIR. The internal structure may be:
        {ticker}/{model_name}_{ticker}.pth   (from sagemaker_training.py)
    or just:
        {model_name}_{ticker}.pth            (flat)
    """
    candidates = [
        model_dir / ticker / f"{model_name}_{ticker}.pth",
        model_dir / f"{model_name}_{ticker}.pth",
    ]
    for path in candidates:
        if path.exists():
            return path

    # Fallback: search recursively
    pth_files = list(model_dir.rglob(f"*{model_name}*{ticker}*.pth"))
    if pth_files:
        return pth_files[0]

    raise FileNotFoundError(
        f"Model file not found for {model_name}/{ticker} in {model_dir}. "
        f"Files present: {list(model_dir.rglob('*'))}"
    )


def load_test_data(test_dir, ticker, lookback=20):
    """Load preprocessed test features and targets.

    The preprocessing step saves:
        {ticker}/  {ticker}y_test.csv (features), {ticker}x_test.csv (target)
        {ticker}/  feature_scaler.pkl, target_scaler.pkl
    """
    ticker_dir = test_dir / ticker

    # Try ticker subdirectory first, then flat
    if ticker_dir.exists():
        base = ticker_dir
    else:
        base = test_dir

    features_path = base / f"{ticker}y_test.csv"
    target_path = base / f"{ticker}x_test.csv"

    if not features_path.exists():
        raise FileNotFoundError(f"Test features not found: {features_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"Test target not found: {target_path}")

    import pandas as pd
    features_df = pd.read_csv(features_path)
    target_df = pd.read_csv(target_path)

    features = features_df.values
    targets = (
        target_df["close"].values if "close" in target_df.columns
        else target_df.iloc[:, -1].values
    )

    return features, targets


def main():
    logger.info("=" * 60)
    logger.info("SageMaker Pipeline: Evaluation Step")
    logger.info("=" * 60)

    ticker = os.environ.get("TICKER", "CL=F")
    model_name = os.environ.get("MODEL_NAME", "lstm")
    lookback = int(os.environ.get("LOOKBACK", "20"))

    logger.info(f"Ticker:     {ticker}")
    logger.info(f"Model:      {model_name}")
    logger.info(f"Lookback:   {lookback}")
    logger.info(f"Model dir:  {MODEL_DIR}")
    logger.info(f"Test dir:   {TEST_DIR}")

    # List input files for debugging
    for d, label in [(MODEL_DIR, "Model"), (TEST_DIR, "Test")]:
        if d.exists():
            files = [f for f in d.rglob("*") if f.is_file()]
            logger.info(f"{label} files ({len(files)}):")
            for f in files:
                logger.info(f"  {f}")

    # Model config (defaults matching config_training.yaml)
    config = {
        "num_features": 20,
        "lookback": lookback,
        "lstm_hidden_size": 64,
        "lstm_num_layers": 2,
        "lstm_dropout": 0.2,
        "transformer_d_model": 64,
        "transformer_nhead": 4,
        "transformer_num_layers": 2,
        "transformer_dim_ff": 128,
        "transformer_dropout": 0.1,
        "bilstm_hidden_size": 64,
        "bilstm_num_layers": 2,
        "bilstm_dropout": 0.2,
    }

    # 1. Load model
    if model_name not in MODEL_REGISTRY:
        logger.error(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
        sys.exit(1)

    model_path = find_model_file(MODEL_DIR, ticker, model_name)
    logger.info(f"Loading model from: {model_path}")

    model = MODEL_REGISTRY[model_name](config)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))
    model.eval()
    logger.info("Model loaded successfully.")

    # 2. Load test data (already scaled by preprocessing step)
    features, targets = load_test_data(TEST_DIR, ticker, lookback)
    logger.info(f"Test data: {features.shape[0]} samples, {features.shape[1]} features")

    # 3. Create data loader
    dataset = FuturesDataset(features, targets, lookback)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 4. Run inference
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb)
            all_preds.append(preds.numpy())
            all_targets.append(yb.numpy())

    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()
    logger.info(f"Inference complete: {len(all_preds)} predictions")

    # 5. Compute metrics (on scaled values, matching training loss)
    mse = float(mean_squared_error(all_targets, all_preds))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(all_targets, all_preds))

    # Compute standard deviation of per-sample squared errors
    squared_errors = (all_targets - all_preds) ** 2
    mse_std = float(np.std(squared_errors))

    logger.info(f"Metrics:")
    logger.info(f"  MSE:  {mse:.6f} (std: {mse_std:.6f})")
    logger.info(f"  RMSE: {rmse:.6f}")
    logger.info(f"  MAE:  {mae:.6f}")

    # 6. Write evaluation.json in SageMaker PropertyFile format
    # This format is required for JsonGet in the pipeline ConditionStep
    evaluation = {
        "regression_metrics": {
            "mse": {
                "value": mse,
                "standard_deviation": mse_std,
            },
            "rmse": {
                "value": rmse,
            },
            "mae": {
                "value": mae,
            },
        }
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    eval_path = OUTPUT_DIR / "evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(evaluation, f, indent=2)
    logger.info(f"Evaluation written to: {eval_path}")

    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
