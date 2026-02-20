"""
Launch a SageMaker Processing Job to run ML monitoring health checks.

Uploads the monitoring script, baseline stats, and models to S3, then launches
a Processing Job that runs data drift, structural break, model disagreement,
and prediction anomaly checks. Creates CloudWatch alarms for drift metrics.

Usage:
    python scripts/deploy_sagemaker_monitoring.py
    python scripts/deploy_sagemaker_monitoring.py --role arn:aws:iam::806081623304:role/LabRole
    python scripts/deploy_sagemaker_monitoring.py --cleanup
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

# CloudWatch alarm definitions for drift metrics
DRIFT_ALARMS = {
    "futures-drift-price-trend": {
        "metric": "DataDrift/PriceTrend",
        "threshold": 0.15,
        "description": "Data drift alarm for price trend features (KS > 0.15)",
    },
    "futures-drift-momentum": {
        "metric": "DataDrift/Momentum",
        "threshold": 0.15,
        "description": "Data drift alarm for momentum features (KS > 0.15)",
    },
    "futures-drift-volume": {
        "metric": "DataDrift/Volume",
        "threshold": 0.15,
        "description": "Data drift alarm for volume features (KS > 0.15)",
    },
    "futures-drift-volatility": {
        "metric": "DataDrift/Volatility",
        "threshold": 0.15,
        "description": "Data drift alarm for volatility features (KS > 0.15)",
    },
    "futures-drift-global": {
        "metric": "DataDrift/Global",
        "threshold": 0.15,
        "description": "Data drift alarm for global drift score (KS > 0.15)",
    },
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


def upload_processing_script(s3_client, bucket: str, prefix: str) -> str:
    """Upload the monitoring processing script to S3."""
    source_file = PROJECT_ROOT / "scripts" / "sagemaker_monitoring_processing.py"
    s3_key = f"{prefix}/sagemaker-monitoring/sagemaker_monitoring_processing.py"
    s3_client.upload_file(str(source_file), bucket, s3_key)
    s3_uri = f"s3://{bucket}/{s3_key}"
    logger.info(f"Uploaded monitoring script to {s3_uri}")
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


def create_cloudwatch_alarms(cw_client):
    """Create CloudWatch alarms for drift metrics."""
    logger.info("Creating CloudWatch alarms...")
    namespace = "FuturesMLOps/Monitoring"

    for alarm_name, alarm_def in DRIFT_ALARMS.items():
        cw_client.put_metric_alarm(
            AlarmName=alarm_name,
            AlarmDescription=alarm_def["description"],
            Namespace=namespace,
            MetricName=alarm_def["metric"],
            Statistic="Maximum",
            Period=86400,  # 24 hours -- monitoring runs daily at most
            EvaluationPeriods=1,
            Threshold=alarm_def["threshold"],
            ComparisonOperator="GreaterThanThreshold",
            TreatMissingData="notBreaching",
        )
        logger.info(f"  Created alarm: {alarm_name} (threshold: {alarm_def['threshold']})")

    logger.info(f"Created {len(DRIFT_ALARMS)} CloudWatch alarms")


def cleanup_cloudwatch_alarms(cw_client):
    """Delete all monitoring CloudWatch alarms."""
    logger.info("Cleaning up CloudWatch alarms...")
    alarm_names = list(DRIFT_ALARMS.keys())
    cw_client.delete_alarms(AlarmNames=alarm_names)
    logger.info(f"Deleted {len(alarm_names)} alarms: {', '.join(alarm_names)}")


def main():
    parser = argparse.ArgumentParser(description="Launch SageMaker monitoring processing job")
    parser.add_argument("--role", default=None, help="IAM role ARN (auto-discovered if omitted)")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--bucket", default=BUCKET, help="S3 bucket")
    parser.add_argument("--prefix", default=PREFIX, help="S3 key prefix")
    parser.add_argument(
        "--instance-type", default="ml.m5.large", help="SageMaker instance type",
    )
    parser.add_argument("--max-run", type=int, default=3600, help="Max runtime in seconds")
    parser.add_argument("--poll-interval", type=int, default=30, help="Status poll interval (s)")
    parser.add_argument(
        "--cleanup", action="store_true",
        help="Delete CloudWatch alarms after validation (for cost control)",
    )
    args = parser.parse_args()

    boto_session = boto3.Session(region_name=args.region)
    iam = boto_session.client("iam")
    s3 = boto_session.client("s3")
    sm = boto_session.client("sagemaker")
    cw = boto_session.client("cloudwatch")

    # Handle cleanup-only mode
    if args.cleanup:
        cleanup_cloudwatch_alarms(cw)
        return

    # 1. Discover IAM role
    role_arn = discover_role(iam, args.role)

    # 2. Upload monitoring processing script to S3
    upload_processing_script(s3, args.bucket, args.prefix)

    # 3. Create CloudWatch alarms (before the job, so they are ready to receive metrics)
    create_cloudwatch_alarms(cw)

    # 4. Create processing job
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
    job_name = f"futures-monitoring-{timestamp}"
    output_s3_prefix = f"{args.prefix}/sagemaker-monitoring/{job_name}"
    output_s3_uri = f"s3://{args.bucket}/{output_s3_prefix}"

    models_s3_uri = f"s3://{args.bucket}/{args.prefix}/models"
    data_s3_uri = f"s3://{args.bucket}/{args.prefix}/dataset/prod"
    baseline_s3_uri = f"s3://{args.bucket}/{args.prefix}/models"
    code_s3_uri = f"s3://{args.bucket}/{args.prefix}/sagemaker-monitoring"

    logger.info(f"Launching processing job: {job_name}")
    logger.info(f"  Code input:     {code_s3_uri}")
    logger.info(f"  Models input:   {models_s3_uri}")
    logger.info(f"  Data input:     {data_s3_uri}")
    logger.info(f"  Baseline input: {baseline_s3_uri}")
    logger.info(f"  Output:         {output_s3_uri}")
    logger.info(f"  Instance:       {args.instance_type}")

    sm.create_processing_job(
        ProcessingJobName=job_name,
        RoleArn=role_arn,
        AppSpecification={
            "ImageUri": PROCESSING_IMAGE,
            "ContainerEntrypoint": [
                "python3",
                "/opt/ml/processing/input/code/sagemaker_monitoring_processing.py",
            ],
        },
        ProcessingInputs=[
            {
                "InputName": "code",
                "S3Input": {
                    "S3Uri": code_s3_uri,
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
            {
                # Baseline stats are stored alongside models in S3
                # (models/{ticker}/baseline_stats.json)
                # So we reuse the models S3 prefix mapped to a separate local path
                "InputName": "baseline",
                "S3Input": {
                    "S3Uri": baseline_s3_uri,
                    "LocalPath": "/opt/ml/processing/input/baseline",
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

    # 5. Poll until completion
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

        # 6. Download results locally
        local_results = PROJECT_ROOT / "files" / "monitoring" / "sagemaker_results"
        download_results(s3, args.bucket, output_s3_prefix, local_results)

        # Print monitoring report if available
        report_path = local_results / "monitoring_report.json"
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
            print("\n" + "=" * 60)
            print("MONITORING REPORT")
            print("=" * 60)
            for ticker, tr in report.get("tickers", {}).items():
                drift = tr.get("data_drift", {})
                print(f"\n  {ticker}:")
                print(f"    Drift buckets:      {drift.get('bucket_scores', {})}")
                print(f"    Global drift:       {drift.get('global_score', 'N/A')}")
                print(f"    Structural break:   {tr.get('structural_break', {}).get('break_detected', 'N/A')}")
                print(f"    Model disagreement: {tr.get('model_disagreement', {}).get('disagreement_score', 'N/A')}")
                anomaly = tr.get("prediction_anomaly", {})
                print(f"    Pred. anomalies:    {anomaly.get('anomaly_count', 0)}/{anomaly.get('total_models', 0)}")
            print("=" * 60)

    elif status == "Failed":
        reason = job_desc.get("FailureReason", "Unknown")
        logger.error(f"Job {job_name} FAILED: {reason}")
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
