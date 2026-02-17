"""
Register trained models as SageMaker Model Packages for visualization in
the SageMaker console (Model Registry).

Creates individual model.tar.gz archives for each of the 6 models
(3 architectures x 2 tickers), uploads to S3, and registers them
in SageMaker Model Package Groups.

Groups:
  - futures-clf-models  (CL=F: lstm, transformer, bilstm-attention)
  - futures-gcf-models  (GC=F: lstm, transformer, bilstm-attention)

Usage:
    python scripts/deploy_sagemaker_model_packages.py
    python scripts/deploy_sagemaker_model_packages.py --role arn:aws:iam::806081623304:role/LabRole
"""
import argparse
import io
import json
import yaml
import logging
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from sagemaker.core.helper.session_helper import get_execution_role, Session
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "files" / "models"

# ===================================================================
import re
from typing import Dict, Any, Tuple

# Allowed regex from SageMaker docs
# matches INVALID chars
_INVALID_PATTERN = re.compile(r"[^\w\s_.:/=+\-@]")

def sanitize_sagemaker_metadata(
    metadata: Dict[str, Any],
    max_len: int = 256,
    verbose: bool = True,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Validates and sanitizes SageMaker customerMetadataProperties.

    - Converts values to strings
    - Removes invalid characters
    - Truncates values > max_len
    - Drops empty results

    Returns:
        cleaned_metadata: dict ready for SageMaker
        report: dict of modifications
    """
    cleaned = {}
    report = {}

    for key, value in metadata.items():
        original = value

        # --- Convert to string safely ---
        if isinstance(value, (dict, list, tuple)):
            value = str(value)
        else:
            value = str(value)

        # --- Remove invalid characters ---
        sanitized = _INVALID_PATTERN.sub("", value)

        # --- Trim whitespace ---
        sanitized = sanitized.strip()

        # --- Enforce length ---
        if len(sanitized) > max_len:
            sanitized = sanitized[:max_len]
            report[key] = f"TRUNCATED to {max_len} chars"

        # --- Remove empty values ---
        if len(sanitized) == 0:
            report[key] = "REMOVED (empty after sanitization)"
            continue

        # --- Track modifications ---
        if sanitized != str(original):
            report[key] = f"SANITIZED: '{original}' -> '{sanitized}'"

        cleaned[key] = sanitized

    if verbose and report:
        print("🔍 SageMaker Metadata Sanitization Report:")
        for k, v in report.items():
            print(f"  - {k}: {v}")

    return cleaned, report

# ===================================================================
# Let's get the role, sess, region, and bucket name from sagemaker AWS
role = get_execution_role()
sess = Session()
region = sess.boto_region_name
bucket = sess.default_bucket()

# Local path to configs
config_path = Path().cwd().parent / 'configs'

# NOTE: Work in progress, adding argparser in the future
config_file = 'config_training.yaml'
config_selected = config_path / config_file

with open(config_selected, 'r') as f:
    config = yaml.safe_load(f)

# =======================================================
BUCKET = bucket
PREFIX = config["data_handler"]["prefix"]
FALLBACK_ROLE_ARN = role
# ==============================================================


TICKERS = ["CL=F", "GC=F"]
MODEL_NAMES = ["lstm", "transformer", "bilstm_attention"]

# PyTorch inference container (used as framework image for model packages)
INFERENCE_IMAGE = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "pytorch-inference:2.1.0-cpu-py310-ubuntu20.04-sagemaker"
)

# Friendly names for display
TICKER_LABELS = {"CL=F": "Crude Oil Futures (CL=F)", "GC=F": "Gold Futures (GC=F)"}
MODEL_LABELS = {
    "lstm": "LSTM",
    "transformer": "Transformer",
    "bilstm_attention": "BiLSTM-Attention",
}

ARCHITECTURE_DESCRIPTIONS = {
    "lstm": (
        "Stacked LSTM (2 layers, hidden=64, dropout=0.2) -> Linear head. "
        "Inspired by Rogendo/forex-lstm-models."
    ),
    "transformer": (
        "Input projection -> Positional encoding -> 2-layer TransformerEncoder "
        "(d=64, 4 heads, ff=128) -> Avg pool -> Linear. "
        "Inspired by SatyamSinghal/financial-ttm."
    ),
    "bilstm_attention": (
        "2-layer BiLSTM (hidden=64) -> Self-attention scoring -> "
        "Weighted sum -> Linear. Inspired by JonusNattapong/xauusd-trading-ai."
    ),
}

FEATURE_LIST = (
    "high, low, open, volume, MA, EMA, KAMA, WMA, MidPrice, "
    "BOP, CMO, MFI, ROC, WILLR, AD, OBV, NATR, ATR, TRANGE, TSF"
)

INFERENCE_SCRIPT = PROJECT_ROOT / "src" / "model" / "inference.py"


def discover_role(iam_client, cli_role: str | None) -> str:
    """Resolve IAM role ARN: CLI override > IAM lookup > fallback."""
    if cli_role:
        logger.info(f"Using CLI-provided role: {cli_role}")
        return cli_role
    try:
        resp = iam_client.get_role(RoleName="LabRole")
        arn = resp["Role"]["Arn"]
        logger.info(f"Discovered IAM role: {arn}")
        return arn
    except ClientError:
        logger.warning(f"Could not discover LabRole, using fallback: {FALLBACK_ROLE_ARN}")
        return FALLBACK_ROLE_ARN


def create_model_tarball(ticker: str, model_name: str) -> io.BytesIO:
    """Create a model.tar.gz containing the .pth, scalers, and metadata for one model."""
    ticker_dir = MODELS_DIR / ticker
    prefix = f"{model_name}_{ticker}"

    files_to_include = [
        f"{prefix}.pth",
        f"{prefix}_feature_scaler.pkl",
        f"{prefix}_target_scaler.pkl",
        f"{prefix}_metadata.json",
    ]

    # Optional files
    for optional in [f"{prefix}_history.json", f"{prefix}_training_curves.png",
                     f"{prefix}_test_predictions.png"]:
        if (ticker_dir / optional).exists():
            files_to_include.append(optional)

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for fname in files_to_include:
            fpath = ticker_dir / fname
            if not fpath.exists():
                logger.warning(f"Missing file: {fpath}")
                continue
            tar.add(str(fpath), arcname=fname)
            logger.info(f"  Added: {fname}")

        # Bundle inference.py into code/ directory (required by PyTorch serving container)
        if INFERENCE_SCRIPT.exists():
            tar.add(str(INFERENCE_SCRIPT), arcname="code/inference.py")
            logger.info("  Added: code/inference.py")
        else:
            logger.warning(f"Inference script not found: {INFERENCE_SCRIPT}")

    buf.seek(0)
    return buf


def upload_model_tarball(s3_client, buf: io.BytesIO, bucket: str,
                         prefix: str, ticker: str, model_name: str) -> str:
    """Upload model.tar.gz to S3 and return the S3 URI."""
    ticker_safe = ticker.replace("=", "").replace("F", "").lower()
    s3_key = f"{prefix}/model-packages/{ticker_safe}/{model_name}/model.tar.gz"
    s3_client.upload_fileobj(buf, bucket, s3_key)
    s3_uri = f"s3://{bucket}/{s3_key}"
    logger.info(f"Uploaded to {s3_uri}")
    return s3_uri


def ensure_model_package_group(sm_client, group_name: str, description: str) -> str:
    """Create a Model Package Group if it doesn't exist. Returns the ARN."""
    try:
        resp = sm_client.describe_model_package_group(
            ModelPackageGroupName=group_name,
        )
        arn = resp["ModelPackageGroupArn"]
        logger.info(f"Model Package Group already exists: {group_name}")
        return arn
    except ClientError as e:
        if "does not exist" not in str(e) and "ValidationException" not in str(e):
            raise

    resp = sm_client.create_model_package_group(
        ModelPackageGroupName=group_name,
        ModelPackageGroupDescription=description,
    )
    arn = resp["ModelPackageGroupArn"]
    logger.info(f"Created Model Package Group: {group_name} ({arn})")
    return arn


def register_model_package(sm_client, group_name: str, model_s3_uri: str,
                           ticker: str, model_name: str, image_uri: str) -> str:
    """Register a model as a versioned Model Package."""
    # Load metadata if available
    ticker_dir = MODELS_DIR / ticker
    meta_path = ticker_dir / f"{model_name}_{ticker}_metadata.json"
    metadata = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    model_label = MODEL_LABELS.get(model_name, model_name)
    ticker_label = TICKER_LABELS.get(ticker, ticker)
    arch_desc = ARCHITECTURE_DESCRIPTIONS.get(model_name, "")
    config = metadata.get("config", {})

    best_val = metadata.get("best_val_loss")
    best_val_str = f"{best_val:.6f}" if isinstance(best_val, (int, float)) else "N/A"

    description = (
        f"{model_label} model for {ticker_label}. "
        f"{arch_desc} "
        f"Epochs trained: {metadata.get('epochs_trained', 'N/A')}, "
        f"Best val loss: {best_val_str}. "
        f"Intended use: ML momentum trading strategy with majority vote ensemble. "
        f"Training data: {config.get('dataset_path', 'files/dataset')} "
        f"(2010-01-04 to 2026-02-05)."
    )
    
    # Build customer metadata (all values must be strings, limit 50 keys)
    customer_metadata = {
        "model_architecture": model_name,
        "architecture_description":arch_desc[:256],
        "ticker": ticker,
        "framework": "pytorch",
        "framework_version": "2.1.0",
        "intended_use": "ML momentum trading with ensemble majority vote",
        "strategy_weights": "70% model_vote + 15% EMA + 15% McClellan",
        "training_data_range": "2010-01-04 to 2026-02-05",
        "feature_list": FEATURE_LIST,
        "num_features": str(config.get("num_features", 20)),
        "lookback": str(config.get("lookback", 20)),
        "batch_size": str(config.get("batch_size", 32)),
        "learning_rate": str(config.get("learning_rate", 0.001)),
        "weight_decay": str(config.get("weight_decay", 1e-5)),
        "grad_clip_norm": str(config.get("grad_clip_norm", 1.0)),
        "early_stopping_patience": str(config.get("early_stopping_patience", 15)),
        "lr_scheduler_patience": str(config.get("lr_scheduler_patience", 7)),
        "lr_scheduler_factor": str(config.get("lr_scheduler_factor", 0.5)),
        "lr_min": str(config.get("lr_min", 1e-6)),
        "max_epochs": str(config.get("epochs", 200)),
    }
    if metadata.get("epochs_trained"):
        customer_metadata["epochs_trained"] = str(metadata["epochs_trained"])
    if metadata.get("best_val_loss"):
        customer_metadata["best_val_loss"] = str(round(metadata["best_val_loss"], 6))
    if metadata.get("final_train_loss"):
        customer_metadata["final_train_loss"] = str(round(metadata["final_train_loss"], 6))
    if metadata.get("final_val_loss"):
        customer_metadata["final_val_loss"] = str(round(metadata["final_val_loss"], 6))

    # Add architecture-specific hyperparams
    if model_name == "lstm":
        customer_metadata["lstm_hidden_size"] = str(config.get("lstm_hidden_size", 64))
        customer_metadata["lstm_num_layers"] = str(config.get("lstm_num_layers", 2))
        customer_metadata["lstm_dropout"] = str(config.get("lstm_dropout", 0.2))
    elif model_name == "transformer":
        customer_metadata["transformer_d_model"] = str(config.get("transformer_d_model", 64))
        customer_metadata["transformer_nhead"] = str(config.get("transformer_nhead", 4))
        customer_metadata["transformer_num_layers"] = str(config.get("transformer_num_layers", 2))
        customer_metadata["transformer_dim_ff"] = str(config.get("transformer_dim_ff", 128))
        customer_metadata["transformer_dropout"] = str(config.get("transformer_dropout", 0.1))
    elif model_name == "bilstm_attention":
        customer_metadata["bilstm_hidden_size"] = str(config.get("bilstm_hidden_size", 64))
        customer_metadata["bilstm_num_layers"] = str(config.get("bilstm_num_layers", 2))
        customer_metadata["bilstm_dropout"] = str(config.get("bilstm_dropout", 0.2))

    # Let's clean the customerMetadata from not possible characters
    customer_metadata, __ = sanitize_sagemaker_metadata(customer_metadata)
    
    resp = sm_client.create_model_package(
        ModelPackageGroupName=group_name,
        ModelPackageDescription=description,
        InferenceSpecification={
            "Containers": [
                {
                    "Image": image_uri,
                    "ModelDataUrl": model_s3_uri,
                    "Framework": "PYTORCH",
                    "FrameworkVersion": "2.1.0",
                    "NearestModelName": f"{model_name}-{ticker}",
                },
            ],
            "SupportedTransformInstanceTypes": ["ml.m5.large"],
            "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.large", "ml.m5.xlarge"],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
        },
        ModelApprovalStatus="Approved",
        CustomerMetadataProperties=customer_metadata,
    )

    arn = resp["ModelPackageArn"]
    logger.info(f"Registered Model Package: {arn}")
    return arn


def main():
    parser = argparse.ArgumentParser(
        description="Register models as SageMaker Model Packages",
    )
    parser.add_argument("--role", default=None, help="IAM role ARN")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--bucket", default=BUCKET, help="S3 bucket")
    parser.add_argument("--prefix", default=PREFIX, help="S3 key prefix")
    args = parser.parse_args()

    boto_session = boto3.Session(region_name=args.region)
    iam = boto_session.client("iam")
    s3 = boto_session.client("s3")
    sm = boto_session.client("sagemaker")

    # 1. Discover role (needed for validation)
    role_arn = discover_role(iam, args.role)

    # 2. Verify local model files exist
    for ticker in TICKERS:
        for mname in MODEL_NAMES:
            pth = MODELS_DIR / ticker / f"{mname}_{ticker}.pth"
            if not pth.exists():
                logger.error(f"Missing model file: {pth}")
                sys.exit(1)
    logger.info("All model files verified.")

    # 3. Create Model Package Groups (one per ticker)
    groups = {}
    for ticker in TICKERS:
        ticker_safe = ticker.replace("=", "").replace("F", "").lower()
        group_name = f"futures-{ticker_safe}-models"
        description = (
            f"ML momentum trading models for {TICKER_LABELS[ticker]}. "
            f"Architectures: LSTM, Transformer, BiLSTM-Attention. "
            f"Trained on production data for price prediction."
        )
        groups[ticker] = ensure_model_package_group(sm, group_name, description)

    # 4. Package, upload, and register each model
    registered = []
    for ticker in TICKERS:
        ticker_safe = ticker.replace("=", "").replace("F", "").lower()
        group_name = f"futures-{ticker_safe}-models"

        for mname in MODEL_NAMES:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing: {MODEL_LABELS[mname]} for {ticker}")
            logger.info(f"{'='*50}")

            # Create tar.gz
            buf = create_model_tarball(ticker, mname)

            # Upload to S3
            model_s3_uri = upload_model_tarball(s3, buf, args.bucket, args.prefix, ticker, mname)

            # Register as Model Package
            pkg_arn = register_model_package(
                sm, group_name, model_s3_uri, ticker, mname, INFERENCE_IMAGE,
            )
            registered.append({
                "ticker": ticker,
                "model": mname,
                "group": group_name,
                "arn": pkg_arn,
                "s3_uri": model_s3_uri,
            })

    # 5. Summary
    logger.info("\n" + "=" * 60)
    logger.info("MODEL REGISTRATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Registered {len(registered)} model packages:\n")

    for r in registered:
        logger.info(f"  {MODEL_LABELS[r['model']]:20s} | {r['ticker']} | {r['group']}")
        logger.info(f"    ARN: {r['arn']}")
        logger.info(f"    S3:  {r['s3_uri']}")

    logger.info("\nView in SageMaker Console:")
    logger.info("  SageMaker > Model Registry > futures-cl-models")
    logger.info("  SageMaker > Model Registry > futures-gc-models")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
