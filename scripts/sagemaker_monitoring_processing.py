"""
SageMaker Processing Job script for ML model monitoring health checks.

Self-contained (no src/ imports). Runs inside a SageMaker Processing container:
  - Installs TA-Lib C library + Python dependencies (needed for feature computation)
  - Loads baseline stats, production data, and trained models
  - Runs 4 health checks: data drift, structural break, model disagreement, prediction anomaly
  - Publishes metrics to CloudWatch and saves a JSON report to /opt/ml/processing/output/

Health Checks:
  1. Data Drift (KS test): per-feature KS statistic vs baseline, aggregated into bucket scores
  2. Structural Break (CUSUM): detects mean shifts in the close price series
  3. Model Disagreement: coefficient of variation across 3 model predictions
  4. Prediction Anomaly: flags predictions outside 2 std of baseline close price range
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
logger = logging.getLogger("sagemaker-monitoring")


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
        sys.executable, "-m", "pip", "install", "-q",
        "TA-Lib", "scipy",
    ])
    logger.info("All dependencies installed.")


install_dependencies()

# --- Now safe to import ML / science libs ---

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import talib
import boto3
from datetime import datetime, timezone
from scipy import stats


# ============================================================================
# SageMaker Processing paths
# ============================================================================

INPUT_BASELINE = Path("/opt/ml/processing/input/baseline")
INPUT_DATA = Path("/opt/ml/processing/input/data")
INPUT_MODELS = Path("/opt/ml/processing/input/models")
OUTPUT_DIR = Path("/opt/ml/processing/output")

TICKERS = ["CL=F", "GC=F"]
MODELS = ["lstm", "transformer", "bilstm_attention"]

# Feature columns matching the training pipeline (from sagemaker_backtest_processing.py)
FEATURE_COLUMNS = [
    "high", "low", "open", "volume",
    "MA", "EMA", "KAMA", "WMA", "MidPrice",
    "BOP", "CMO", "MFI", "ROC", "WILLR",
    "AD", "OBV", "NATR", "ATR", "TRANGE", "TSF",
]

# Feature bucket mapping (matches baseline_stats.json structure)
FEATURE_BUCKETS = {
    "price_trend": [
        "high", "low", "open", "volume", "close",
        "MA", "EMA", "KAMA", "WMA", "MidPrice", "TSF",
    ],
    "momentum": ["BOP", "CMO", "MFI", "ROC", "WILLR"],
    "volume": ["AD", "OBV"],
    "volatility": ["NATR", "ATR", "TRANGE"],
}

# KS test drift alarm threshold
KS_THRESHOLD = 0.15

# Model config (same defaults as training)
DEFAULT_CONFIG = {
    "lookback": 20, "num_features": 20,
    "lstm_hidden_size": 64, "lstm_num_layers": 2, "lstm_dropout": 0.2,
    "transformer_d_model": 64, "transformer_nhead": 4,
    "transformer_num_layers": 2, "transformer_dim_ff": 128,
    "transformer_dropout": 0.1,
    "bilstm_hidden_size": 64, "bilstm_num_layers": 2,
    "bilstm_dropout": 0.2,
}


# ============================================================================
# Model Architectures (self-contained copies -- must match saved state_dicts)
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
        self.dropout = nn.Dropout(do)
        self.fc = nn.Linear(hs, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


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
        self.attention_fc = nn.Linear(hs * 2, 1)
        self.dropout = nn.Dropout(do)
        self.fc = nn.Linear(hs * 2, 1)

    def forward(self, x):
        out, _ = self.bilstm(x)
        w = torch.softmax(self.attention_fc(out), dim=1)
        ctx = (out * w).sum(dim=1)
        return self.fc(self.dropout(ctx))


MODEL_REGISTRY = {
    "lstm": LSTMModel,
    "transformer": TransformerModel,
    "bilstm_attention": BiLSTMAttentionModel,
}


# ============================================================================
# Feature Engineering (same as sagemaker_backtest_processing.py)
# ============================================================================

def compute_talib_features(ohlcv_df):
    """Replicate feature engineering from the training pipeline."""
    df = ohlcv_df.copy().sort_index()

    df["MA"] = talib.MA(df["close"], timeperiod=10)
    df["EMA"] = talib.EMA(df["close"], timeperiod=10)
    df["KAMA"] = talib.KAMA(df["close"], timeperiod=10)
    df["WMA"] = talib.WMA(df["close"], timeperiod=10)
    df["MidPrice"] = talib.MIDPRICE(df["high"], df["low"], timeperiod=10)

    df["BOP"] = talib.BOP(df["open"], df["high"], df["low"], df["close"])
    df["CMO"] = talib.CMO(df["close"], timeperiod=10)
    df["MFI"] = talib.MFI(df["high"], df["low"], df["close"], df["volume"])
    df["ROC"] = talib.ROC(df["close"], timeperiod=10)
    df["WILLR"] = talib.WILLR(df["high"], df["low"], df["close"], timeperiod=14)

    df["AD"] = talib.AD(df["high"], df["low"], df["close"], df["volume"])
    df["OBV"] = talib.OBV(df["close"], df["volume"])

    df["NATR"] = talib.NATR(df["high"], df["low"], df["close"], timeperiod=14)
    df["ATR"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)
    df["TRANGE"] = talib.TRANGE(df["high"], df["low"], df["close"])

    df["TSF"] = talib.TSF(df["close"], timeperiod=14)

    # Drop warmup rows (NaN from TA-Lib indicators)
    df = df.iloc[15:]
    return df


# ============================================================================
# Check 1: Data Drift (KS Test)
# ============================================================================

def check_data_drift(ticker, baseline, featured_df):
    """
    Compare current production feature distributions against baseline using
    the two-sample Kolmogorov-Smirnov test.

    Returns per-bucket KS scores and a global drift score.
    """
    logger.info(f"[{ticker}] Running data drift check (KS test)...")

    bucket_scores = {}
    per_feature_results = {}

    for bucket_name, bucket_features in FEATURE_BUCKETS.items():
        ks_values = []
        for feat in bucket_features:
            # Get baseline stats for this feature
            baseline_feat = baseline.get("all_features", {}).get(feat)
            if baseline_feat is None:
                logger.warning(f"  Skipping {feat}: not in baseline")
                continue

            if feat not in featured_df.columns:
                logger.warning(f"  Skipping {feat}: not in prod data")
                continue

            prod_values = featured_df[feat].dropna().values
            if len(prod_values) < 10:
                logger.warning(f"  Skipping {feat}: too few samples ({len(prod_values)})")
                continue

            # Synthesize baseline samples from stored statistics for KS test.
            # Use a normal distribution with the baseline mean/std.
            baseline_mean = baseline_feat["mean"]
            baseline_std = baseline_feat["std"]
            n_baseline = int(baseline_feat["count"])

            # Generate synthetic baseline samples from the stored distribution
            np.random.seed(42)
            baseline_samples = np.random.normal(baseline_mean, baseline_std, n_baseline)

            # Two-sample KS test
            ks_stat, p_value = stats.ks_2samp(baseline_samples, prod_values)

            per_feature_results[feat] = {
                "ks_statistic": round(float(ks_stat), 4),
                "p_value": round(float(p_value), 4),
                "bucket": bucket_name,
                "drifted": ks_stat > KS_THRESHOLD,
            }
            ks_values.append(ks_stat)

        # Bucket score = mean KS statistic across features in the bucket
        if ks_values:
            bucket_scores[bucket_name] = round(float(np.mean(ks_values)), 4)
        else:
            bucket_scores[bucket_name] = 0.0

        logger.info(f"  Bucket '{bucket_name}': KS={bucket_scores[bucket_name]:.4f}")

    # Global drift score = mean across all buckets
    global_score = round(float(np.mean(list(bucket_scores.values()))), 4) if bucket_scores else 0.0
    logger.info(f"  Global drift score: {global_score:.4f}")

    return {
        "bucket_scores": bucket_scores,
        "global_score": global_score,
        "per_feature": per_feature_results,
    }


# ============================================================================
# Check 2: Structural Break (CUSUM)
# ============================================================================

def check_structural_break(ticker, close_series):
    """
    Run CUSUM (cumulative sum) test on the close price series to detect
    mean shifts (structural breaks).

    Uses a simple CUSUM approach: accumulate deviations from the overall mean.
    A break is detected if the CUSUM range exceeds a threshold based on
    the series standard deviation.
    """
    logger.info(f"[{ticker}] Running structural break check (CUSUM)...")

    values = close_series.dropna().values
    if len(values) < 20:
        logger.warning(f"  Too few data points for CUSUM ({len(values)})")
        return {"break_detected": False, "cusum_range": 0.0, "threshold": 0.0}

    mean_val = np.mean(values)
    std_val = np.std(values)

    # Cumulative sum of deviations from the mean
    cusum = np.cumsum(values - mean_val)

    # CUSUM range (max - min) as the test statistic
    cusum_range = float(np.max(cusum) - np.min(cusum))

    # Threshold: scale by std * sqrt(n) -- a simple heuristic
    # Values above this suggest a structural break
    threshold = float(std_val * np.sqrt(len(values)))

    break_detected = cusum_range > threshold
    logger.info(f"  CUSUM range: {cusum_range:.2f}, threshold: {threshold:.2f}, break: {break_detected}")

    return {
        "break_detected": break_detected,
        "cusum_range": round(cusum_range, 4),
        "threshold": round(threshold, 4),
    }


# ============================================================================
# Check 3: Model Disagreement
# ============================================================================

def check_model_disagreement(ticker, models_dict, featured_df, lookback):
    """
    Get predictions from all 3 models on the latest production features.
    Compute disagreement as the coefficient of variation (std / |mean|).
    High disagreement means models are confused -- data may be out-of-distribution.
    """
    logger.info(f"[{ticker}] Running model disagreement check...")

    if len(featured_df) < lookback:
        logger.warning(f"  Not enough data for lookback={lookback}")
        return {"disagreement_score": 0.0, "predictions": {}}

    # Get the last `lookback` rows of features
    features = featured_df[FEATURE_COLUMNS].values[-lookback:]

    predictions = {}
    for mname, m in models_dict.items():
        model = m["model"]
        scaler_features = m["scaler_features"]
        scaler_target = m["scaler_target"]

        model.eval()
        scaled = scaler_features.transform(features)
        x = torch.FloatTensor(scaled).unsqueeze(0)
        with torch.no_grad():
            pred = model(x).numpy().flatten()
        pred_price = float(scaler_target.inverse_transform(pred.reshape(-1, 1)).flatten()[0])
        predictions[mname] = round(pred_price, 4)
        logger.info(f"  {mname}: predicted ${pred_price:.2f}")

    pred_values = list(predictions.values())
    if len(pred_values) >= 2 and abs(np.mean(pred_values)) > 0.001:
        # Coefficient of variation
        disagreement = float(np.std(pred_values) / abs(np.mean(pred_values)))
    else:
        disagreement = 0.0

    logger.info(f"  Disagreement (CV): {disagreement:.4f}")

    return {
        "disagreement_score": round(disagreement, 4),
        "predictions": predictions,
    }


# ============================================================================
# Check 4: Prediction Distribution Anomaly
# ============================================================================

def check_prediction_anomaly(ticker, predictions, baseline):
    """
    Compare model predictions against the baseline close price distribution.
    Flag if any prediction falls outside 2 std from the baseline close mean.
    """
    logger.info(f"[{ticker}] Running prediction anomaly check...")

    baseline_close = baseline.get("all_features", {}).get("close", {})
    if not baseline_close:
        logger.warning(f"  No baseline close stats found")
        return {"anomaly_detected": False, "anomaly_count": 0, "details": {}}

    baseline_mean = baseline_close["mean"]
    baseline_std = baseline_close["std"]
    lower_bound = baseline_mean - 2 * baseline_std
    upper_bound = baseline_mean + 2 * baseline_std

    logger.info(f"  Baseline close: mean={baseline_mean:.2f}, std={baseline_std:.2f}")
    logger.info(f"  Expected range: [{lower_bound:.2f}, {upper_bound:.2f}]")

    anomaly_count = 0
    details = {}
    for mname, pred_price in predictions.items():
        is_anomaly = pred_price < lower_bound or pred_price > upper_bound
        details[mname] = {
            "prediction": pred_price,
            "in_range": not is_anomaly,
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
        }
        if is_anomaly:
            anomaly_count += 1
            logger.info(f"  ANOMALY: {mname} predicted ${pred_price:.2f} (outside expected range)")

    anomaly_detected = anomaly_count > 0
    logger.info(f"  Anomalies: {anomaly_count}/{len(predictions)}")

    return {
        "anomaly_detected": anomaly_detected,
        "anomaly_count": anomaly_count,
        "total_models": len(predictions),
        "details": details,
    }


# ============================================================================
# CloudWatch Metrics Publishing
# ============================================================================

def publish_metrics(report):
    """Push all monitoring metrics to CloudWatch."""
    logger.info("Publishing metrics to CloudWatch...")
    cw = boto3.client("cloudwatch", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))

    namespace = "FuturesMLOps/Monitoring"
    metrics = []

    for ticker, ticker_report in report["tickers"].items():
        # Data drift bucket scores
        drift = ticker_report.get("data_drift", {})
        for bucket_name, score in drift.get("bucket_scores", {}).items():
            metric_name = f"DataDrift/{bucket_name.title().replace('_', '')}"
            metrics.append({
                "MetricName": metric_name,
                "Dimensions": [{"Name": "Ticker", "Value": ticker}],
                "Value": score,
                "Unit": "None",
            })

        # Global drift score
        metrics.append({
            "MetricName": "DataDrift/Global",
            "Dimensions": [{"Name": "Ticker", "Value": ticker}],
            "Value": drift.get("global_score", 0.0),
            "Unit": "None",
        })

        # Structural break (1.0 = break detected, 0.0 = no break)
        sbreak = ticker_report.get("structural_break", {})
        metrics.append({
            "MetricName": f"StructuralBreak/{ticker}",
            "Value": 1.0 if sbreak.get("break_detected", False) else 0.0,
            "Unit": "None",
        })

        # Model disagreement
        disagree = ticker_report.get("model_disagreement", {})
        metrics.append({
            "MetricName": f"ModelDisagreement/{ticker}",
            "Value": disagree.get("disagreement_score", 0.0),
            "Unit": "None",
        })

        # Prediction anomaly (count of anomalous predictions)
        anomaly = ticker_report.get("prediction_anomaly", {})
        metrics.append({
            "MetricName": f"PredictionAnomaly/{ticker}",
            "Value": float(anomaly.get("anomaly_count", 0)),
            "Unit": "Count",
        })

    # CloudWatch accepts max 1000 metrics per PutMetricData call, batch by 20
    for i in range(0, len(metrics), 20):
        batch = metrics[i:i + 20]
        cw.put_metric_data(Namespace=namespace, MetricData=batch)
        logger.info(f"  Published {len(batch)} metrics (batch {i // 20 + 1})")

    logger.info(f"Published {len(metrics)} total metrics to {namespace}")


# ============================================================================
# Input Verification
# ============================================================================

def verify_inputs():
    """Check that all expected input files exist."""
    missing = []

    for ticker in TICKERS:
        # Baseline stats
        baseline_path = INPUT_BASELINE / ticker / "baseline_stats.json"
        if not baseline_path.exists():
            missing.append(str(baseline_path))

        # Production OHLCV data
        data_path = INPUT_DATA / f"{ticker}x_prod.csv"
        if not data_path.exists():
            missing.append(str(data_path))

        # Models and scalers
        for mname in MODELS:
            pth = INPUT_MODELS / ticker / f"{mname}_{ticker}.pth"
            feat_scl = INPUT_MODELS / ticker / f"{mname}_{ticker}_feature_scaler.pkl"
            tgt_scl = INPUT_MODELS / ticker / f"{mname}_{ticker}_target_scaler.pkl"
            for f in [pth, feat_scl, tgt_scl]:
                if not f.exists():
                    missing.append(str(f))

    if missing:
        logger.error("Missing input files:\n  " + "\n  ".join(missing))
        sys.exit(1)

    logger.info("All input files verified.")


# ============================================================================
# Main Processing Pipeline
# ============================================================================

def main():
    logger.info("=" * 60)
    logger.info("SageMaker Processing: Monitoring Health Checks")
    logger.info("=" * 60)

    # 1. Verify inputs
    verify_inputs()

    config = DEFAULT_CONFIG
    lookback = int(config["lookback"])

    # Full report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tickers": {},
    }

    for ticker in TICKERS:
        logger.info(f"\n{'=' * 40}")
        logger.info(f"Processing ticker: {ticker}")
        logger.info(f"{'=' * 40}")

        # Load baseline stats
        baseline_path = INPUT_BASELINE / ticker / "baseline_stats.json"
        with open(baseline_path) as f:
            baseline = json.load(f)
        logger.info(f"Loaded baseline: {baseline.get('total_samples', '?')} samples")

        # Load production OHLCV data and compute features
        data_path = INPUT_DATA / f"{ticker}x_prod.csv"
        raw_df = pd.read_csv(data_path)
        raw_df["Date"] = pd.to_datetime(raw_df["Date"])
        raw_df = raw_df.set_index("Date").sort_index()
        logger.info(f"Loaded prod data: {len(raw_df)} rows")

        # Compute TA-Lib features from OHLCV
        featured_df = compute_talib_features(raw_df)
        logger.info(f"Computed features: {len(featured_df)} rows after warmup")

        # Load models
        models_dict = {}
        for mname in MODELS:
            pth = INPUT_MODELS / ticker / f"{mname}_{ticker}.pth"
            feat_scl = INPUT_MODELS / ticker / f"{mname}_{ticker}_feature_scaler.pkl"
            tgt_scl = INPUT_MODELS / ticker / f"{mname}_{ticker}_target_scaler.pkl"

            model = MODEL_REGISTRY[mname](config)
            model.load_state_dict(torch.load(pth, weights_only=True))
            model.eval()

            models_dict[mname] = {
                "model": model,
                "scaler_features": joblib.load(feat_scl),
                "scaler_target": joblib.load(tgt_scl),
            }
        logger.info(f"Loaded {len(models_dict)} models")

        # --- Run 4 health checks ---

        # Check 1: Data Drift
        drift_result = check_data_drift(ticker, baseline, featured_df)

        # Check 2: Structural Break
        break_result = check_structural_break(ticker, raw_df["close"])

        # Check 3: Model Disagreement
        disagree_result = check_model_disagreement(ticker, models_dict, featured_df, lookback)

        # Check 4: Prediction Anomaly
        anomaly_result = check_prediction_anomaly(
            ticker, disagree_result["predictions"], baseline,
        )

        report["tickers"][ticker] = {
            "data_drift": drift_result,
            "structural_break": break_result,
            "model_disagreement": disagree_result,
            "prediction_anomaly": anomaly_result,
        }

    # 2. Publish metrics to CloudWatch
    publish_metrics(report)

    # 3. Save JSON report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "monitoring_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to {report_path}")

    # 4. Print summary
    logger.info("\n" + "=" * 60)
    logger.info("MONITORING SUMMARY")
    logger.info("=" * 60)
    for ticker, tr in report["tickers"].items():
        drift = tr["data_drift"]
        logger.info(f"\n  {ticker}:")
        logger.info(f"    Drift buckets: {drift['bucket_scores']}")
        logger.info(f"    Global drift:  {drift['global_score']}")
        logger.info(f"    Structural break: {tr['structural_break']['break_detected']}")
        logger.info(f"    Model disagreement: {tr['model_disagreement']['disagreement_score']}")
        logger.info(f"    Prediction anomalies: {tr['prediction_anomaly']['anomaly_count']}/{tr['prediction_anomaly']['total_models']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
