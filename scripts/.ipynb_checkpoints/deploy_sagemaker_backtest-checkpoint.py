"""
Launch a SageMaker Processing Job to run the ML momentum backtest.

Uploads the processing script, configures S3 input/output channels for models
and production data, launches the job, polls until completion, and downloads
results locally.

Usage:
    python scripts/deploy_sagemaker_backtest.py
    python scripts/deploy_sagemaker_backtest.py --role arn:aws:iam::806081623304:role/LabRole
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
import boto3
from botocore.exceptions import ClientError
from sagemaker.core.helper.session_helper import get_execution_role, Session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ======================================================================
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

BUCKET = bucket
PREFIX = config["data_handler"]["prefix"]
FALLBACK_ROLE_ARN = role
# ======================================================================

# PyTorch 2.1.0, Python 3.10, CPU container for us-east-1
PROCESSING_IMAGE = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "pytorch-training:2.1.0-cpu-py310-ubuntu20.04-sagemaker"
)

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


def upload_processing_script(s3_client, bucket: str, prefix: str) -> str:
    """Upload the processing script to S3."""
    source_file = PROJECT_ROOT / "scripts" / "sagemaker_backtest_processing.py"
    s3_key = f"{prefix}/sagemaker-processing/sagemaker_backtest_processing.py"
    s3_client.upload_file(str(source_file), bucket, s3_key)
    s3_uri = f"s3://{bucket}/{s3_key}"
    logger.info(f"Uploaded processing script to {s3_uri}")
    return s3_uri


def poll_processing_job(sm_client, job_name: str, interval: int = 30) -> dict:
    """Poll SageMaker processing job status until terminal state."""
    logger.info(f"Polling job {job_name} every {interval}s...")
    terminal = {"Completed", "Failed", "Stopped"}

    while True:
        resp = sm_client.describe_processing_job(ProcessingJobName=job_name)
        status = resp["ProcessingJobStatus"]
        logger.info(f"  Status: {status}")

        if status in terminal:
            return resp

        time.sleep(interval)


def download_results(s3_client, bucket: str, s3_prefix: str, local_dir: Path):
    """Download all output files from the processing job's S3 output."""
    local_dir.mkdir(parents=True, exist_ok=True)

    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=s3_prefix)

    count = 0
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = key.split("/")[-1]
            if not filename:
                continue
            local_path = local_dir / filename
            s3_client.download_file(bucket, key, str(local_path))
            logger.info(f"  Downloaded: {filename}")
            count += 1

    logger.info(f"Downloaded {count} files to {local_dir}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Launch SageMaker backtest processing job")
    parser.add_argument("--role", default=None, help="IAM role ARN (auto-discovered if omitted)")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--bucket", default=BUCKET, help="S3 bucket")
    parser.add_argument("--prefix", default=PREFIX, help="S3 key prefix")
    parser.add_argument(
        "--instance-type", default="ml.m5.large", help="SageMaker instance type",
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

    # 2. Upload processing script to S3
    script_s3_uri = upload_processing_script(s3, args.bucket, args.prefix)

    # 3. Create processing job
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
    job_name = f"futures-backtest-{timestamp}"
    output_s3_prefix = f"{args.prefix}/sagemaker-backtest/{job_name}"
    output_s3_uri = f"s3://{args.bucket}/{output_s3_prefix}"

    models_s3_uri = f"s3://{args.bucket}/{args.prefix}/models"
    data_s3_uri = f"s3://{args.bucket}/{args.prefix}/dataset/prod"

    logger.info(f"Launching processing job: {job_name}")
    logger.info(f"  Models input:  {models_s3_uri}")
    logger.info(f"  Data input:    {data_s3_uri}")
    logger.info(f"  Output:        {output_s3_uri}")
    logger.info(f"  Instance:      {args.instance_type}")

    sm.create_processing_job(
        ProcessingJobName=job_name,
        RoleArn=role_arn,
        AppSpecification={
            "ImageUri": PROCESSING_IMAGE,
            "ContainerEntrypoint": [
                "python3",
                "/opt/ml/processing/input/code/sagemaker_backtest_processing.py",
            ],
        },
        ProcessingInputs=[
            {
                "InputName": "code",
                "S3Input": {
                    "S3Uri": f"s3://{args.bucket}/{args.prefix}/sagemaker-processing",
                    "LocalPath": "/opt/ml/processing/input/code",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                },
            },
            {
                "InputName": "models",
                "S3Input": {
                    "S3Uri": models_s3_uri,
                    "LocalPath": "/opt/ml/processing/input/models",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                },
            },
            {
                "InputName": "data",
                "S3Input": {
                    "S3Uri": data_s3_uri,
                    "LocalPath": "/opt/ml/processing/input/data",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                },
            },
        ],
        ProcessingOutputConfig={
            "Outputs": [
                {
                    "OutputName": "results",
                    "S3Output": {
                        "S3Uri": output_s3_uri,
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                },
            ],
        },
        ProcessingResources={
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": args.instance_type,
                "VolumeSizeInGB": 20,
            },
        },
        StoppingCondition={
            "MaxRuntimeInSeconds": args.max_run,
        },
    )

    logger.info(f"Processing job created: {job_name}")

    # 4. Poll until completion
    job_desc = poll_processing_job(sm, job_name, args.poll_interval)
    status = job_desc["ProcessingJobStatus"]

    if status == "Completed":
        billable = job_desc.get("ProcessingEndTime", datetime.now(timezone.utc))
        start_time = job_desc.get("ProcessingStartTime", datetime.now(timezone.utc))
        duration_s = (billable - start_time).total_seconds()
        cost = (duration_s / 3600) * INSTANCE_PRICE_PER_HOUR

        logger.info("=" * 60)
        logger.info(f"Job {job_name} COMPLETED")
        logger.info(f"  Duration:       {duration_s:.0f}s")
        logger.info(f"  Estimated cost: ${cost:.4f}")
        logger.info("=" * 60)

        # 5. Download results locally
        local_results = PROJECT_ROOT / "files" / "backtest" / "sagemaker_results"
        download_results(s3, args.bucket, output_s3_prefix, local_results)

        # Print summary if available
        summary_path = local_results / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            print("\n" + "=" * 60)
            print("BACKTEST RESULTS")
            print("=" * 60)
            for k, v in summary.items():
                print(f"  {k}: {v}")
            print("=" * 60)

        report_path = local_results / "report.html"
        if report_path.exists():
            print(f"\nHTML report: {report_path}")
            print(f"Open with: xdg-open {report_path}")

    elif status == "Failed":
        reason = job_desc.get("FailureReason", "Unknown")
        logger.error(f"Job {job_name} FAILED: {reason}")
        # Try to get CloudWatch logs
        logger.error("Check CloudWatch logs for details:")
        logger.error(
            f"  /aws/sagemaker/ProcessingJobs/{job_name}"
        )
        sys.exit(1)
    else:
        logger.warning(f"Job {job_name} ended with status: {status}")
        sys.exit(1)


if __name__ == "__main__":
    main()
