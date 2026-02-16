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
import logging
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "files" / "models"

BUCKET = "labs-usd-01"
PREFIX = "AAI_540_group_8"
FALLBACK_ROLE_ARN = "arn:aws:iam::806081623304:role/LabRole"

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

    description = (
        f"{model_label} model for {ticker_label}. "
        f"Epochs trained: {metadata.get('epochs_trained', 'N/A')}, "
        f"Best val loss: {metadata.get('best_val_loss', 'N/A')}"
    )

    # Build customer metadata (all values must be strings)
    customer_metadata = {
        "model_architecture": model_name,
        "ticker": ticker,
        "framework": "pytorch",
        "framework_version": "2.1.0",
    }
    if metadata.get("epochs_trained"):
        customer_metadata["epochs_trained"] = str(metadata["epochs_trained"])
    if metadata.get("best_val_loss"):
        customer_metadata["best_val_loss"] = str(round(metadata["best_val_loss"], 6))

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
