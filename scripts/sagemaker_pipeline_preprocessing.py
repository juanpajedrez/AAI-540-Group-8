"""
SageMaker Pipeline Preprocessing Step - self-contained container script.

Runs inside a SageMaker Processing container as part of the CI/CD pipeline.
Installs TA-Lib, computes 20 technical indicator features from raw OHLCV data,
splits into train/val/test, applies MinMaxScaler, and saves processed CSVs + scalers.

Input:  /opt/ml/processing/input/data/   (raw CSV files from S3)
Output: /opt/ml/processing/output/        (processed CSVs + scaler .pkl files)

Environment variables:
    TICKER  - ticker symbol to process (default: "CL=F")
"""
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("sagemaker-pipeline-preprocessing")


# ============================================================================
# Dependency Installation (runs before any ML imports)
# ============================================================================

def install_dependencies():
    """Install TA-Lib C library and Python packages inside the container."""
    logger.info("Installing system dependencies...")
    subprocess.check_call(
        ["apt-get", "update", "-qq"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    subprocess.check_call(
        ["apt-get", "install", "-y", "-qq", "build-essential", "wget"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    logger.info("Downloading and compiling TA-Lib C library...")
    talib_url = "https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz"
    subprocess.check_call(["wget", "-q", talib_url, "-O", "/tmp/ta-lib.tar.gz"])
    subprocess.check_call(["tar", "-xzf", "/tmp/ta-lib.tar.gz", "-C", "/tmp"])
    subprocess.check_call(["./configure", "--prefix=/usr"], cwd="/tmp/ta-lib-0.6.4")
    subprocess.check_call(["make", "-j2"], cwd="/tmp/ta-lib-0.6.4")
    subprocess.check_call(["make", "install"], cwd="/tmp/ta-lib-0.6.4")

    logger.info("Installing Python packages...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "TA-Lib",
    ])
    logger.info("All dependencies installed.")


install_dependencies()

# --- Now safe to import ML / TA-Lib libs ---
import numpy as np
import pandas as pd
import joblib
import talib
from sklearn.preprocessing import MinMaxScaler


# ============================================================================
# Constants
# ============================================================================

INPUT_DIR = Path("/opt/ml/processing/input/data")
OUTPUT_DIR = Path("/opt/ml/processing/output")

# Columns to drop from raw feature CSVs
COLUMNS_TO_DROP = ["Unnamed: 0", "Date", "adj close", "repaired?"]

# 20 TA-Lib features (same as feature_local_talib.py and sagemaker_training.py)
FEATURE_COLUMNS = [
    "high", "low", "open", "volume",
    "MA", "EMA", "KAMA", "WMA", "MidPrice",
    "BOP", "CMO", "MFI", "ROC", "WILLR",
    "AD", "OBV", "NATR", "ATR", "TRANGE", "TSF",
]


# ============================================================================
# Feature Engineering (replicates feature_local_talib.py)
# ============================================================================

def compute_talib_features(df):
    """Compute 16 TA-Lib indicators from OHLCV data (adds to existing 4 OHLV columns)."""
    df = df.copy().sort_index()

    # Overlap Studies
    df["MA"] = talib.MA(df["close"], timeperiod=10)
    df["EMA"] = talib.EMA(df["close"], timeperiod=10)
    df["KAMA"] = talib.KAMA(df["close"], timeperiod=10)
    df["WMA"] = talib.WMA(df["close"], timeperiod=10)
    df["MidPrice"] = talib.MIDPRICE(df["high"], df["low"], timeperiod=10)

    # Momentum Indicators
    df["BOP"] = talib.BOP(df["open"], df["high"], df["low"], df["close"])
    df["CMO"] = talib.CMO(df["close"], timeperiod=10)
    df["MFI"] = talib.MFI(df["high"], df["low"], df["close"], df["volume"])
    df["ROC"] = talib.ROC(df["close"], timeperiod=10)
    df["WILLR"] = talib.WILLR(df["high"], df["low"], df["close"], timeperiod=14)

    # Volume Indicators
    df["AD"] = talib.AD(df["high"], df["low"], df["close"], df["volume"])
    df["OBV"] = talib.OBV(df["close"], df["volume"])

    # Volatility Indicators
    df["NATR"] = talib.NATR(df["high"], df["low"], df["close"], timeperiod=14)
    df["ATR"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)
    df["TRANGE"] = talib.TRANGE(df["high"], df["low"], df["close"])

    # Miscellaneous
    df["TSF"] = talib.TSF(df["close"], timeperiod=14)

    # Drop first 15 rows (NaN from windowed indicators)
    df = df.iloc[15:]
    return df


# ============================================================================
# Data Loading
# ============================================================================

def load_raw_data(ticker, input_dir):
    """
    Load raw CSV data for a ticker.

    Dataset naming convention (INVERTED):
        {ticker}y_*.csv = FEATURES (OHLCV + indicators)
        {ticker}x_*.csv = TARGET (close price)

    We look for dev/val/test splits if they exist. If only raw unsplit data
    exists, we compute features and split ourselves.
    """
    # Check if pre-split data exists (from sagemaker_training.py pattern)
    splits = ["dev", "val", "test"]
    has_splits = all(
        (input_dir / f"{ticker}y_{s}.csv").exists() for s in splits
    )

    if has_splits:
        logger.info(f"Found pre-split data for {ticker}")
        return _load_presplit_data(ticker, input_dir, splits)
    else:
        logger.info(f"No pre-split data found for {ticker}, loading raw and splitting")
        return _load_and_split_raw(ticker, input_dir)


def _load_presplit_data(ticker, input_dir, splits):
    """Load data that's already split into dev/val/test."""
    data = {}
    for split in splits:
        features_path = input_dir / f"{ticker}y_{split}.csv"
        target_path = input_dir / f"{ticker}x_{split}.csv"

        features_df = pd.read_csv(features_path)
        target_df = pd.read_csv(target_path)

        # Drop metadata columns
        cols_to_drop = [c for c in COLUMNS_TO_DROP if c in features_df.columns]
        features_df = features_df.drop(columns=cols_to_drop)

        # Keep only known feature columns
        available = [c for c in FEATURE_COLUMNS if c in features_df.columns]
        features_df = features_df[available]

        # Extract target
        target_series = (
            target_df["close"] if "close" in target_df.columns
            else target_df.iloc[:, -1]
        )

        # Map split names: dev -> train
        split_name = "train" if split == "dev" else split
        data[split_name] = (features_df, target_series)

    return data


def _load_and_split_raw(ticker, input_dir):
    """Load raw OHLCV CSV, compute features, and split 60/20/20."""
    # Try multiple filename patterns
    candidates = [
        input_dir / f"{ticker}x_prod.csv",
        input_dir / f"{ticker}.csv",
        input_dir / f"{ticker}x_raw.csv",
    ]

    raw_df = None
    for path in candidates:
        if path.exists():
            raw_df = pd.read_csv(path)
            logger.info(f"Loaded raw data from {path}: {len(raw_df)} rows")
            break

    if raw_df is None:
        raise FileNotFoundError(
            f"No raw data found for {ticker} in {input_dir}. "
            f"Tried: {[str(c) for c in candidates]}"
        )

    # Ensure date column is set as index
    if "Date" in raw_df.columns:
        raw_df["Date"] = pd.to_datetime(raw_df["Date"])
        raw_df = raw_df.set_index("Date").sort_index()

    # Ensure required OHLCV columns exist
    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in raw_df.columns:
            raise ValueError(f"Missing required column '{col}' in {ticker} data")

    # Compute TA-Lib features
    featured_df = compute_talib_features(raw_df)

    # Extract target (close) and features
    target = featured_df["close"].copy()
    features = featured_df[FEATURE_COLUMNS].copy()

    # Split: 60% train, 20% val, 20% test (chronological, no shuffle)
    n = len(features)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    data = {
        "train": (features.iloc[:train_end], target.iloc[:train_end]),
        "val": (features.iloc[train_end:val_end], target.iloc[train_end:val_end]),
        "test": (features.iloc[val_end:], target.iloc[val_end:]),
    }

    return data


# ============================================================================
# Main Processing
# ============================================================================

def main():
    logger.info("=" * 60)
    logger.info("SageMaker Pipeline: Preprocessing Step")
    logger.info("=" * 60)

    ticker = os.environ.get("TICKER", "CL=F")
    logger.info(f"Processing ticker: {ticker}")
    logger.info(f"Input directory: {INPUT_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # List input files for debugging
    if INPUT_DIR.exists():
        files = list(INPUT_DIR.rglob("*"))
        logger.info(f"Input files ({len(files)}):")
        for f in files:
            if f.is_file():
                logger.info(f"  {f}")
    else:
        logger.error(f"Input directory does not exist: {INPUT_DIR}")
        sys.exit(1)

    # Load data (handles both pre-split and raw formats)
    data = load_raw_data(ticker, INPUT_DIR)

    # Scale features and target (fit on train only)
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()

    train_feat, train_tgt = data["train"]
    val_feat, val_tgt = data["val"]
    test_feat, test_tgt = data["test"]

    logger.info(f"Data shapes - train: {train_feat.shape}, val: {val_feat.shape}, test: {test_feat.shape}")

    # Fit scalers on training data
    train_feat_scaled = scaler_features.fit_transform(train_feat.values)
    train_tgt_scaled = scaler_target.fit_transform(
        train_tgt.values.reshape(-1, 1)
    ).flatten()

    # Transform val and test
    val_feat_scaled = scaler_features.transform(val_feat.values)
    val_tgt_scaled = scaler_target.transform(
        val_tgt.values.reshape(-1, 1)
    ).flatten()

    test_feat_scaled = scaler_features.transform(test_feat.values)
    test_tgt_scaled = scaler_target.transform(
        test_tgt.values.reshape(-1, 1)
    ).flatten()

    # Save processed CSVs to output directory
    ticker_output = OUTPUT_DIR / ticker
    ticker_output.mkdir(parents=True, exist_ok=True)

    # Save scaled features (using the INVERTED naming convention: y = features, x = target)
    pd.DataFrame(train_feat_scaled, columns=FEATURE_COLUMNS).to_csv(
        ticker_output / f"{ticker}y_dev.csv", index=False
    )
    pd.DataFrame(val_feat_scaled, columns=FEATURE_COLUMNS).to_csv(
        ticker_output / f"{ticker}y_val.csv", index=False
    )
    pd.DataFrame(test_feat_scaled, columns=FEATURE_COLUMNS).to_csv(
        ticker_output / f"{ticker}y_test.csv", index=False
    )

    # Save scaled targets
    pd.DataFrame({"close": train_tgt_scaled}).to_csv(
        ticker_output / f"{ticker}x_dev.csv", index=False
    )
    pd.DataFrame({"close": val_tgt_scaled}).to_csv(
        ticker_output / f"{ticker}x_val.csv", index=False
    )
    pd.DataFrame({"close": test_tgt_scaled}).to_csv(
        ticker_output / f"{ticker}x_test.csv", index=False
    )

    # Save scalers for later use in evaluation and inference
    joblib.dump(scaler_features, ticker_output / "feature_scaler.pkl")
    joblib.dump(scaler_target, ticker_output / "target_scaler.pkl")

    # Save preprocessing metadata
    metadata = {
        "ticker": ticker,
        "num_features": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "train_samples": len(train_feat),
        "val_samples": len(val_feat),
        "test_samples": len(test_feat),
        "scaler_type": "MinMaxScaler",
    }
    with open(ticker_output / "preprocessing_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info(f"  Ticker: {ticker}")
    logger.info(f"  Train samples: {len(train_feat)}")
    logger.info(f"  Val samples:   {len(val_feat)}")
    logger.info(f"  Test samples:  {len(test_feat)}")
    logger.info(f"  Features:      {len(FEATURE_COLUMNS)}")
    logger.info(f"  Output dir:    {ticker_output}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
