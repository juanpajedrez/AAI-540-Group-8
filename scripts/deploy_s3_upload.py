"""
Upload all project artifacts to S3 bucket labs-usd-01.

Uploads datasets, pre-trained models, and backtest data using boto3
(no SageMaker session required).

Usage:
    python scripts/deploy_s3_upload.py
    python scripts/deploy_s3_upload.py --bucket my-bucket --prefix my_prefix
    python scripts/deploy_s3_upload.py --dry-run
"""
import argparse
import logging
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

UPLOAD_MANIFEST = [
    # (local_dir, s3_sub_prefix, glob_pattern, description)
    ("files/dataset", "dataset", "*.csv", "train/val/test CSVs"),
    ("files/dataset/prod", "dataset/prod", "*.csv", "production CSVs"),
    ("files/models/CL=F", "models/CL=F", "*", "CL=F model artifacts"),
    ("files/models/GC=F", "models/GC=F", "*", "GC=F model artifacts"),
    ("files/backtest", "backtest", "**/*", "backtest data & results"),
]

SKIP_PATTERNS = {".ipynb_checkpoints", "__pycache__"}


def upload_directory(
    s3_client,
    local_dir: Path,
    bucket: str,
    s3_prefix: str,
    glob_pattern: str,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Upload files matching glob_pattern from local_dir to s3://bucket/s3_prefix/."""
    uploaded, skipped = 0, 0
    for filepath in sorted(local_dir.glob(glob_pattern)):
        if not filepath.is_file():
            continue
        if any(part in SKIP_PATTERNS for part in filepath.parts):
            skipped += 1
            continue

        relative = filepath.relative_to(local_dir)
        s3_key = f"{s3_prefix}/{relative}"

        if dry_run:
            logger.info(f"  [DRY RUN] {filepath} -> s3://{bucket}/{s3_key}")
        else:
            s3_client.upload_file(str(filepath), bucket, s3_key)
            logger.info(f"  Uploaded {filepath.name} -> s3://{bucket}/{s3_key}")
        uploaded += 1

    return uploaded, skipped


def main():
    parser = argparse.ArgumentParser(description="Upload project artifacts to S3")
    parser.add_argument("--bucket", default="labs-usd-01", help="S3 bucket name")
    parser.add_argument("--prefix", default="AAI_540_group_8", help="S3 key prefix")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--dry-run", action="store_true", help="List files without uploading")
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=args.region)

    # Verify bucket access
    if not args.dry_run:
        try:
            s3.head_bucket(Bucket=args.bucket)
            logger.info(f"Bucket s3://{args.bucket} is accessible")
        except ClientError as e:
            code = e.response["Error"]["Code"]
            logger.error(f"Cannot access bucket s3://{args.bucket}: {code}")
            sys.exit(1)

    total_uploaded, total_skipped = 0, 0
    for local_rel, s3_sub, pattern, desc in UPLOAD_MANIFEST:
        local_dir = PROJECT_ROOT / local_rel
        if not local_dir.exists():
            logger.warning(f"Skipping {desc}: {local_dir} does not exist")
            continue

        s3_prefix = f"{args.prefix}/{s3_sub}"
        logger.info(f"Uploading {desc} from {local_dir} -> s3://{args.bucket}/{s3_prefix}/")

        uploaded, skipped = upload_directory(
            s3, local_dir, args.bucket, s3_prefix, pattern, args.dry_run
        )
        total_uploaded += uploaded
        total_skipped += skipped

    logger.info(f"Done: {total_uploaded} files uploaded, {total_skipped} skipped")

    if not args.dry_run:
        logger.info(f"\nVerify with: aws s3 ls --recursive s3://{args.bucket}/{args.prefix}/")


if __name__ == "__main__":
    main()
