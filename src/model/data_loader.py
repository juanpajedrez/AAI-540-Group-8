import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple, Dict, Optional

from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger('aws')

from src.misc.logger_utils import log_function_call

# NOTE: The dataset naming convention is INVERTED from standard ML convention.
# In dataset_local_operator.py, labels are passed as the first arg to train_test_split,
# so x_*.csv files contain the TARGET (single 'close' column) and
# y_*.csv files contain the FEATURES (24 columns with OHLCV + technical indicators).
COLUMNS_TO_DROP = ['Unnamed: 0', 'Date', 'adj close', 'repaired?']

FEATURE_COLUMNS = [
    'high', 'low', 'open', 'volume',
    'MA', 'EMA', 'KAMA', 'WMA', 'MidPrice',
    'BOP', 'CMO', 'MFI', 'ROC', 'WILLR',
    'AD', 'OBV',
    'NATR', 'ATR', 'TRANGE', 'TSF'
]


@log_function_call
def load_ticker_data(ticker: str, split: str,
                     dataset_path: Optional[Path] = None
                     ) -> Tuple[pd.DataFrame, pd.Series]:
    """Load features and target for a given ticker and split.

    Because of the inverted naming convention:
      - y_*.csv -> features (24 cols)
      - x_*.csv -> target (close price)
    """
    try:
        if dataset_path is None:
            dataset_path = Path.cwd() / 'files' / 'dataset'

        features_file = dataset_path / f"{ticker}y_{split}.csv"
        target_file = dataset_path / f"{ticker}x_{split}.csv"

        features_df = pd.read_csv(features_file)
        target_df = pd.read_csv(target_file)

        # Drop non-feature columns
        cols_to_drop = [c for c in COLUMNS_TO_DROP if c in features_df.columns]
        features_df = features_df.drop(columns=cols_to_drop)

        # Ensure only known feature columns remain (in case of extra columns)
        available_features = [c for c in FEATURE_COLUMNS if c in features_df.columns]
        features_df = features_df[available_features]

        # Target is the 'close' column from x files
        if 'close' in target_df.columns:
            target_series = target_df['close']
        else:
            # Fallback: if 'close' not found, use the second column (first is index)
            target_series = target_df.iloc[:, -1]

        return features_df, target_series

    except Exception as e:
        logger.error(e)
        raise


class FuturesDataset(Dataset):
    """PyTorch Dataset that creates sliding window sequences from time series data."""

    def __init__(self, features: np.ndarray, targets: np.ndarray, lookback: int):
        """
        Args:
            features: Scaled feature array, shape [num_samples, num_features]
            targets: Scaled target array, shape [num_samples]
            lookback: Number of past timesteps per input window
        """
        self.features = features
        self.targets = targets
        self.lookback = lookback

    def __len__(self):
        return len(self.features) - self.lookback

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.lookback]  # [lookback, num_features]
        y = self.targets[idx + self.lookback]        # scalar
        return torch.FloatTensor(x), torch.FloatTensor([y])


@log_function_call
def create_data_loaders(ticker: str, config: dict) -> Dict:
    """Create train/val/test DataLoaders with scalers fitted on training data only.

    Returns:
        Dict with keys: 'train', 'val', 'test' (DataLoaders),
        'scaler_features', 'scaler_target' (fitted MinMaxScalers),
        'num_features' (int)
    """
    try:
        dataset_path = Path.cwd() / config.get('dataset_path', 'files/dataset')
        lookback = config.get('lookback', 20)
        batch_size = config.get('batch_size', 32)

        # Load all splits
        train_features, train_target = load_ticker_data(ticker, 'dev', dataset_path)
        test_features, test_target = load_ticker_data(ticker, 'test', dataset_path)
        val_features, val_target = load_ticker_data(ticker, 'val', dataset_path)

        num_features = train_features.shape[1]

        # Fit scalers on training data only
        scaler_features = MinMaxScaler()
        scaler_target = MinMaxScaler()

        train_feat_scaled = scaler_features.fit_transform(train_features.values)
        train_tgt_scaled = scaler_target.fit_transform(
            train_target.values.reshape(-1, 1)
        ).flatten()

        # Transform test and val using training scalers
        test_feat_scaled = scaler_features.transform(test_features.values)
        test_tgt_scaled = scaler_target.transform(
            test_target.values.reshape(-1, 1)
        ).flatten()

        val_feat_scaled = scaler_features.transform(val_features.values)
        val_tgt_scaled = scaler_target.transform(
            val_target.values.reshape(-1, 1)
        ).flatten()

        # Create datasets
        train_dataset = FuturesDataset(train_feat_scaled, train_tgt_scaled, lookback)
        test_dataset = FuturesDataset(test_feat_scaled, test_tgt_scaled, lookback)
        val_dataset = FuturesDataset(val_feat_scaled, val_tgt_scaled, lookback)

        # Create data loaders (shuffle=False for time series)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=False, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, drop_last=False)

        logger.info(f"[{ticker}] Data loaders created: "
                     f"train={len(train_dataset)}, "
                     f"val={len(val_dataset)}, "
                     f"test={len(test_dataset)} samples, "
                     f"features={num_features}")

        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'scaler_features': scaler_features,
            'scaler_target': scaler_target,
            'num_features': num_features,
        }

    except Exception as e:
        logger.error(e)
        raise
