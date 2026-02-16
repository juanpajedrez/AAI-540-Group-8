"""
Launch a SageMaker PyTorch training job for futures price prediction.

Uses boto3 directly (no sagemaker SDK) to avoid version compatibility issues.
Discovers the IAM role, packages source code, launches the job, polls status,
and reports results.

Usage:
    python scripts/deploy_sagemaker_training.py
    python scripts/deploy_sagemaker_training.py --role arn:aws:iam::806081623304:role/LabRole
    python scripts/deploy_sagemaker_training.py --epochs 50 --instance-type ml.m5.large
"""
import argparse
import io
import json
import logging
import os
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
import yaml
from botocore.exceptions import ClientError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config_training.yaml"

BUCKET = "labs-usd-01"
PREFIX = "AAI_540_group_8"
FALLBACK_ROLE_ARN = "arn:aws:iam::806081623304:role/LabRole"

# PyTorch 2.1.0, Python 3.10, CPU training container for us-east-1
TRAINING_IMAGE = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "pytorch-training:2.1.0-cpu-py310-ubuntu20.04-sagemaker"
)

# ml.m5.large on-demand price in us-east-1 (USD/hr)
INSTANCE_PRICE_PER_HOUR = 0.115


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


def load_hyperparameters(config_path: Path, epoch_override: int | None) -> dict[str, str]:
    """Load hyperparameters from config_training.yaml, all as strings for SageMaker."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    mh = cfg.get("model_handler", {})

    return {
        "epochs": str(epoch_override or mh.get("epochs", 50)),
        "batch_size": str(mh.get("batch_size", 32)),
        "lookback": str(mh.get("lookback", 20)),
        "learning_rate": str(mh.get("learning_rate", 0.001)),
        "weight_decay": str(mh.get("weight_decay", 1e-5)),
        "grad_clip_norm": str(mh.get("grad_clip_norm", 1.0)),
        "early_stopping_patience": str(mh.get("early_stopping_patience", 15)),
        "lr_scheduler_patience": str(mh.get("lr_scheduler_patience", 7)),
        "lr_scheduler_factor": str(mh.get("lr_scheduler_factor", 0.5)),
        "lr_min": str(mh.get("lr_min", 1e-6)),
        "models": ",".join(mh.get("models", ["lstm", "transformer", "bilstm_attention"])),
        "tickers": ",".join(mh.get("tickers", ["CL=F", "GC=F"])),
        "lstm_hidden_size": str(mh.get("lstm_hidden_size", 64)),
        "lstm_num_layers": str(mh.get("lstm_num_layers", 2)),
        "lstm_dropout": str(mh.get("lstm_dropout", 0.2)),
        "transformer_d_model": str(mh.get("transformer_d_model", 64)),
        "transformer_nhead": str(mh.get("transformer_nhead", 4)),
        "transformer_num_layers": str(mh.get("transformer_num_layers", 2)),
        "transformer_dim_ff": str(mh.get("transformer_dim_ff", 128)),
        "transformer_dropout": str(mh.get("transformer_dropout", 0.1)),
        "bilstm_hidden_size": str(mh.get("bilstm_hidden_size", 64)),
        "bilstm_num_layers": str(mh.get("bilstm_num_layers", 2)),
        "bilstm_dropout": str(mh.get("bilstm_dropout", 0.2)),
    }


def upload_source_tarball(s3_client, bucket: str, prefix: str) -> str:
    """Package sagemaker_training.py into a tar.gz and upload to S3."""
    source_file = PROJECT_ROOT / "scripts" / "sagemaker_training.py"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(str(source_file), arcname="sagemaker_training.py")
    buf.seek(0)

    s3_key = f"{prefix}/sagemaker-source/sourcedir.tar.gz"
    s3_client.upload_fileobj(buf, bucket, s3_key)
    s3_uri = f"s3://{bucket}/{s3_key}"
    logger.info(f"Uploaded source tarball to {s3_uri}")
    return s3_uri


def poll_training_job(sm_client, job_name: str, interval: int = 30):
    """Poll SageMaker training job status until terminal state."""
    logger.info(f"Polling job {job_name} every {interval}s...")
    terminal = {"Completed", "Failed", "Stopped"}

    while True:
        resp = sm_client.describe_training_job(TrainingJobName=job_name)
        status = resp["TrainingJobStatus"]
        secondary = resp.get("SecondaryStatus", "")
        logger.info(f"  Status: {status} ({secondary})")

        if status in terminal:
            return resp

        time.sleep(interval)


def report_results(job_desc: dict):
    """Print job results: duration, cost estimate, and model artifact location."""
    status = job_desc["TrainingJobStatus"]
    job_name = job_desc["TrainingJobName"]

    if status == "Completed":
        billable = job_desc.get("BillableTimeInSeconds", 0)
        cost = (billable / 3600) * INSTANCE_PRICE_PER_HOUR
        model_s3 = job_desc.get("ModelArtifacts", {}).get("S3ModelArtifacts", "N/A")

        logger.info("=" * 60)
        logger.info(f"Job {job_name} COMPLETED")
        logger.info(f"  Billable seconds: {billable}")
        logger.info(f"  Estimated cost:   ${cost:.4f}")
        logger.info(f"  Model artifacts:  {model_s3}")
        logger.info("=" * 60)
        logger.info(
            f"\nTo download: aws s3 cp {model_s3} model.tar.gz"
            f"\nTo extract:  tar xzf model.tar.gz"
        )
    elif status == "Failed":
        reason = job_desc.get("FailureReason", "Unknown")
        logger.error(f"Job {job_name} FAILED: {reason}")
        sys.exit(1)
    else:
        logger.warning(f"Job {job_name} ended with status: {status}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Launch SageMaker training job")
    parser.add_argument("--role", default=None, help="IAM role ARN (auto-discovered if omitted)")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--bucket", default=BUCKET, help="S3 bucket")
    parser.add_argument("--prefix", default=PREFIX, help="S3 key prefix")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs (default: 50)")
    parser.add_argument(
        "--instance-type", default="ml.m5.large", help="SageMaker instance type"
    )
    parser.add_argument("--max-run", type=int, default=3600, help="Max runtime in seconds")
    parser.add_argument("--poll-interval", type=int, default=30, help="Status poll interval (s)")
    args = parser.parse_args()

    boto_session = boto3.Session(region_name=args.region)
    iam = boto_session.client("iam")
    s3 = boto_session.client("s3")
    sm = boto_session.client("sagemaker")

    # 1. Discover IAM role
    role_arn = discover_role(iam, args.role)

    # 2. Load hyperparameters
    hp = load_hyperparameters(CONFIG_PATH, args.epochs)
    logger.info(
        f"Hyperparameters: epochs={hp['epochs']}, "
        f"tickers={hp['tickers']}, models={hp['models']}"
    )

    # 3. Package and upload source code
    source_s3_uri = upload_source_tarball(s3, args.bucket, args.prefix)

    # 4. Create training job
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
    job_name = f"futures-pytorch-{timestamp}"
    s3_output = f"s3://{args.bucket}/{args.prefix}/sagemaker-output"
    s3_data = f"s3://{args.bucket}/{args.prefix}/dataset"

    logger.info(f"Launching training job: {job_name}")
    logger.info(f"  Input data: {s3_data}")
    logger.info(f"  Output:     {s3_output}")
    logger.info(f"  Instance:   {args.instance_type}")
    logger.info(f"  Image:      {TRAINING_IMAGE}")

    sm.create_training_job(
        TrainingJobName=job_name,
        RoleArn=role_arn,
        AlgorithmSpecification={
            "TrainingImage": TRAINING_IMAGE,
            "TrainingInputMode": "File",
        },
        HyperParameters={
            **hp,
            "sagemaker_program": "sagemaker_training.py",
            "sagemaker_submit_directory": source_s3_uri,
        },
        InputDataConfig=[
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": s3_data,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "text/csv",
            }
        ],
        OutputDataConfig={
            "S3OutputPath": s3_output,
        },
        ResourceConfig={
            "InstanceType": args.instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": 10,
        },
        StoppingCondition={
            "MaxRuntimeInSeconds": args.max_run,
        },
    )

    logger.info(f"Training job created: {job_name}")

    # 5. Poll until completion
    job_desc = poll_training_job(sm, job_name, args.poll_interval)

    # 6. Report results
    report_results(job_desc)


if __name__ == "__main__":
    main()
