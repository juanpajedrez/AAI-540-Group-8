"""
Compute baseline distribution statistics from training data.

Generates a JSON file per ticker with per-feature stats (mean, std, min, max,
quartiles) grouped into 4 monitoring buckets. These baselines are used by the
monitoring health check to detect data drift.

Usage:
    python scripts/compute_baseline_stats.py
    python scripts/compute_baseline_stats.py --dataset-path files/dataset --output-dir files/models

Output:
    files/models/{ticker}/baseline_stats.json
"""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TICKERS = ["CL=F", "GC=F"]

# Columns to drop before computing stats (metadata, not features)
COLUMNS_TO_DROP = ["Unnamed: 0", "Date", "adj close", "repaired?"]

# 4 monitoring alarm buckets -- each groups related features
FEATURE_BUCKETS = {
    "price_trend": ["high", "low", "open", "volume", "close",
                    "MA", "EMA", "KAMA", "WMA", "MidPrice", "TSF"],
    "momentum":    ["BOP", "CMO", "MFI", "ROC", "WILLR"],
    "volume":      ["AD", "OBV"],
    "volatility":  ["NATR", "ATR", "TRANGE"],
}

# Flat list of all monitored features (order doesn't matter)
ALL_FEATURES = [f for bucket in FEATURE_BUCKETS.values() for f in bucket]


def compute_feature_stats(series: pd.Series) -> dict:
    """Compute distribution statistics for a single feature column."""
    return {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
        "q25": float(series.quantile(0.25)),
        "q50": float(series.quantile(0.50)),
        "q75": float(series.quantile(0.75)),
        "count": int(series.count()),
    }


def compute_baseline(ticker: str, dataset_path: Path) -> dict:
    """Load training features for a ticker and compute baseline stats."""
    # NOTE: Inverted naming -- y_*.csv contains FEATURES, x_*.csv contains TARGET
    features_file = dataset_path / f"{ticker}y_dev.csv"
    target_file = dataset_path / f"{ticker}x_dev.csv"

    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_file}")

    logger.info(f"Loading training features from {features_file}")
    df = pd.read_csv(features_file)

    # Drop metadata columns
    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Also load target (close price) for completeness in price_trend bucket
    if target_file.exists():
        target_df = pd.read_csv(target_file)
        if "close" in target_df.columns:
            df["close"] = target_df["close"].values

    # Compute per-feature stats
    feature_stats = {}
    for col in df.columns:
        if col in ALL_FEATURES:
            feature_stats[col] = compute_feature_stats(df[col].dropna())

    # Organize into buckets
    bucket_stats = {}
    for bucket_name, bucket_features in FEATURE_BUCKETS.items():
        available = [f for f in bucket_features if f in feature_stats]
        bucket_stats[bucket_name] = {
            "features": {f: feature_stats[f] for f in available},
            "feature_count": len(available),
        }

    baseline = {
        "ticker": ticker,
        "training_split": "dev",
        "source_file": str(features_file.name),
        "total_features": len(feature_stats),
        "total_samples": len(df),
        "feature_buckets": bucket_stats,
        "all_features": feature_stats,
    }

    return baseline


def main():
    parser = argparse.ArgumentParser(description="Compute baseline stats from training data")
    parser.add_argument(
        "--dataset-path", default=str(PROJECT_ROOT / "files" / "dataset"),
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--output-dir", default=str(PROJECT_ROOT / "files" / "models"),
        help="Path to output directory (saves under {ticker}/baseline_stats.json)",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)

    for ticker in TICKERS:
        logger.info(f"Computing baseline for {ticker}...")
        baseline = compute_baseline(ticker, dataset_path)

        ticker_dir = output_dir / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        output_path = ticker_dir / "baseline_stats.json"

        with open(output_path, "w") as f:
            json.dump(baseline, f, indent=2)

        logger.info(f"Saved baseline to {output_path}")
        logger.info(
            f"  {baseline['total_features']} features, "
            f"{baseline['total_samples']} samples"
        )

    logger.info("Baseline computation complete.")


if __name__ == "__main__":
    main()
