"""
Create and optionally execute a SageMaker Pipeline for futures model CI/CD.

Pipeline structure:
    Preprocess -> Train -> Evaluate -> Condition (MSE check) -> Register or Fail

We build the pipeline definition as a JSON document and submit it via boto3
instead of using the sagemaker.workflow.* SDK classes. This keeps us consistent
with all our other deploy scripts (deploy_sagemaker_training.py, etc.) which
also use raw boto3. It avoids SDK version incompatibilities (v2 vs v3 import
paths changed) and makes the code easier for teammates to read and run.

Usage:
    # Validate pipeline (upsert only, no execution):
    python scripts/deploy_sagemaker_pipeline.py

    # Run validation pipeline (cheap: 5 epochs, single ticker, lenient threshold):
    python scripts/deploy_sagemaker_pipeline.py --execute

    # Run production pipeline:
    python scripts/deploy_sagemaker_pipeline.py --execute --epochs 200 --mse-threshold 0.005

    # Clean up after validation:
    python scripts/deploy_sagemaker_pipeline.py --cleanup
"""
import argparse
import io
import json
import logging
import sys
import tarfile
import time
from pathlib import Path

import boto3
import yaml
from botocore.exceptions import ClientError
from sagemaker.core.helper.session_helper import get_execution_role, Session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config_training.yaml"

# PyTorch 2.1.0 CPU training container for us-east-1
TRAINING_IMAGE = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "pytorch-training:2.1.0-cpu-py310-ubuntu20.04-sagemaker"
)

# PyTorch inference container (for model registration)
INFERENCE_IMAGE = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "pytorch-inference:2.1.0-cpu-py310-ubuntu20.04-sagemaker"
)

# ml.m5.large on-demand price in us-east-1 (USD/hr)
INSTANCE_PRICE_PER_HOUR = 0.115

PIPELINE_NAME = "futures-ml-pipeline"


# ============================================================================
# AWS Discovery (same pattern as deploy_sagemaker_training.py)
# ============================================================================

# Let's get the role, sess, region, and bucket name from sagemaker AWS
role = get_execution_role()
sess = Session()
region = sess.boto_region_name
bucket = sess.default_bucket()

# Local path to configs
config_path = Path().cwd().parent / "configs"
config_file = "config_training.yaml"
config_selected = config_path / config_file

with open(config_selected, "r") as f:
    config = yaml.safe_load(f)

BUCKET = bucket
PREFIX = config["data_handler"]["prefix"]
FALLBACK_ROLE_ARN = role


def discover_role(iam_client, cli_role):
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
        logger.warning(
            f"Could not discover LabRole, using fallback: {FALLBACK_ROLE_ARN}"
        )
        return FALLBACK_ROLE_ARN


# ============================================================================
# S3 Upload Helpers
# ============================================================================


def upload_pipeline_scripts(s3_client, bucket, prefix):
    """Upload preprocessing and evaluation scripts to S3 for use in processing steps."""
    scripts = {
        "sagemaker_pipeline_preprocessing.py": (
            PROJECT_ROOT / "scripts" / "sagemaker_pipeline_preprocessing.py"
        ),
        "sagemaker_pipeline_evaluation.py": (
            PROJECT_ROOT / "scripts" / "sagemaker_pipeline_evaluation.py"
        ),
    }

    s3_uris = {}
    for name, local_path in scripts.items():
        s3_key = f"{prefix}/pipeline-scripts/{name}"
        s3_client.upload_file(str(local_path), bucket, s3_key)
        s3_uri = f"s3://{bucket}/{s3_key}"
        logger.info(f"Uploaded {name} to {s3_uri}")
        s3_uris[name] = s3_uri

    return s3_uris


def upload_training_source(s3_client, bucket, prefix):
    """Package sagemaker_training.py into tar.gz and upload to S3."""
    source_file = PROJECT_ROOT / "scripts" / "sagemaker_training.py"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(str(source_file), arcname="sagemaker_training.py")
    buf.seek(0)

    s3_key = f"{prefix}/pipeline-source/sourcedir.tar.gz"
    s3_client.upload_fileobj(buf, bucket, s3_key)
    s3_uri = f"s3://{bucket}/{s3_key}"
    logger.info(f"Uploaded training source to {s3_uri}")
    return s3_uri


# ============================================================================
# Pipeline Definition Builder (raw JSON, no sagemaker.workflow.* SDK)
# ============================================================================


def build_pipeline_definition(args, role_arn, bucket, prefix, config):
    """Build the SageMaker Pipeline definition as a JSON-serializable dict.

    We construct the pipeline definition JSON directly instead of using
    sagemaker.workflow.* classes. This avoids SDK version issues (v2 vs v3
    moved all workflow imports) and stays consistent with the rest of our
    deploy scripts which all use raw boto3.

    Pipeline JSON schema reference:
        https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-pipeline-definition.html
    """
    mh = config.get("model_handler", {})

    # S3 paths
    s3_data = f"s3://{bucket}/{prefix}/dataset"
    s3_pipeline_out = f"s3://{bucket}/{prefix}/pipeline-output"
    s3_preprocess_out = f"{s3_pipeline_out}/preprocessed"
    s3_training_out = f"{s3_pipeline_out}/training"
    s3_evaluation_out = f"{s3_pipeline_out}/evaluation"
    s3_preprocess_script = (
        f"s3://{bucket}/{prefix}/pipeline-scripts/"
        "sagemaker_pipeline_preprocessing.py"
    )
    s3_eval_script = (
        f"s3://{bucket}/{prefix}/pipeline-scripts/"
        "sagemaker_pipeline_evaluation.py"
    )
    s3_training_source = f"s3://{bucket}/{prefix}/pipeline-source/sourcedir.tar.gz"

    # Determine model package group name from ticker
    ticker_safe = args.ticker.replace("=", "").replace("F", "").lower()
    model_package_group = f"futures-{ticker_safe}-models"

    # Hyperparameters (all strings, as SageMaker requires)
    hyperparameters = {
        "epochs": str(args.epochs),
        "batch_size": str(mh.get("batch_size", 32)),
        "lookback": str(mh.get("lookback", 20)),
        "learning_rate": str(mh.get("learning_rate", 0.001)),
        "weight_decay": str(mh.get("weight_decay", 1e-5)),
        "grad_clip_norm": str(mh.get("grad_clip_norm", 1.0)),
        "early_stopping_patience": str(mh.get("early_stopping_patience", 15)),
        "lr_scheduler_patience": str(mh.get("lr_scheduler_patience", 7)),
        "lr_scheduler_factor": str(mh.get("lr_scheduler_factor", 0.5)),
        "lr_min": str(mh.get("lr_min", 1e-6)),
        "models": args.model,
        "tickers": args.ticker,
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
        "sagemaker_program": "sagemaker_training.py",
        "sagemaker_submit_directory": s3_training_source,
    }

    # ==================================================================
    # Step 1: Preprocess (Processing Job)
    # ==================================================================
    step_preprocess = {
        "Name": "Preprocess",
        "Type": "Processing",
        "Arguments": {
            "AppSpecification": {
                "ImageUri": TRAINING_IMAGE,
                "ContainerEntrypoint": ["python3", "/opt/ml/processing/input/code/sagemaker_pipeline_preprocessing.py"],
            },
            "RoleArn": role_arn,
            "ProcessingInputs": [
                {
                    "InputName": "data",
                    "S3Input": {
                        "S3Uri": s3_data,
                        "LocalPath": "/opt/ml/processing/input/data",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                        "S3DataDistributionType": "FullyReplicated",
                    },
                },
                {
                    "InputName": "code",
                    "S3Input": {
                        "S3Uri": s3_preprocess_script,
                        "LocalPath": "/opt/ml/processing/input/code",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                        "S3DataDistributionType": "FullyReplicated",
                    },
                },
            ],
            "ProcessingOutputConfig": {
                "Outputs": [
                    {
                        "OutputName": "processed",
                        "S3Output": {
                            "S3Uri": s3_preprocess_out,
                            "LocalPath": "/opt/ml/processing/output",
                            "S3UploadMode": "EndOfJob",
                        },
                    }
                ]
            },
            "ProcessingResources": {
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": args.instance_type,
                    "VolumeSizeInGB": 10,
                }
            },
            "Environment": {
                "TICKER": args.ticker,
            },
        },
    }

    # ==================================================================
    # Step 2: Train (Training Job)
    # ==================================================================
    step_train = {
        "Name": "Train",
        "Type": "Training",
        "Arguments": {
            "AlgorithmSpecification": {
                "TrainingImage": TRAINING_IMAGE,
                "TrainingInputMode": "File",
            },
            "RoleArn": role_arn,
            "HyperParameters": hyperparameters,
            "InputDataConfig": [
                {
                    "ChannelName": "training",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": s3_preprocess_out,
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                    "ContentType": "text/csv",
                }
            ],
            "OutputDataConfig": {
                "S3OutputPath": s3_training_out,
            },
            "ResourceConfig": {
                "InstanceCount": 1,
                "InstanceType": args.instance_type,
                "VolumeSizeInGB": 10,
            },
            "StoppingCondition": {
                "MaxRuntimeInSeconds": 3600,
            },
        },
        "DependsOn": ["Preprocess"],
    }

    # ==================================================================
    # Step 3: Evaluate (Processing Job)
    # ==================================================================
    step_evaluate = {
        "Name": "Evaluate",
        "Type": "Processing",
        "Arguments": {
            "AppSpecification": {
                "ImageUri": TRAINING_IMAGE,
                "ContainerEntrypoint": ["python3", "/opt/ml/processing/input/code/sagemaker_pipeline_evaluation.py"],
            },
            "RoleArn": role_arn,
            "ProcessingInputs": [
                {
                    "InputName": "model",
                    "S3Input": {
                        "S3Uri": {
                            "Get": "Steps.Train.ModelArtifacts.S3ModelArtifacts"
                        },
                        "LocalPath": "/opt/ml/processing/input/model",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                        "S3DataDistributionType": "FullyReplicated",
                    },
                },
                {
                    "InputName": "test",
                    "S3Input": {
                        "S3Uri": s3_preprocess_out,
                        "LocalPath": "/opt/ml/processing/input/test",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                        "S3DataDistributionType": "FullyReplicated",
                    },
                },
                {
                    "InputName": "code",
                    "S3Input": {
                        "S3Uri": s3_eval_script,
                        "LocalPath": "/opt/ml/processing/input/code",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                        "S3DataDistributionType": "FullyReplicated",
                    },
                },
            ],
            "ProcessingOutputConfig": {
                "Outputs": [
                    {
                        "OutputName": "evaluation",
                        "S3Output": {
                            "S3Uri": s3_evaluation_out,
                            "LocalPath": "/opt/ml/processing/output",
                            "S3UploadMode": "EndOfJob",
                        },
                    }
                ]
            },
            "ProcessingResources": {
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": args.instance_type,
                    "VolumeSizeInGB": 10,
                }
            },
            "Environment": {
                "TICKER": args.ticker,
                "MODEL_NAME": args.model,
                "LOOKBACK": str(mh.get("lookback", 20)),
            },
        },
        "DependsOn": ["Train"],
        "PropertyFiles": [
            {
                "PropertyFileName": "EvaluationReport",
                "OutputName": "evaluation",
                "FilePath": "evaluation.json",
            }
        ],
    }

    # ==================================================================
    # Step 4a: Register Model (if MSE <= threshold)
    # ==================================================================
    step_register = {
        "Name": "RegisterModel",
        "Type": "RegisterModel",
        "Arguments": {
            "ModelPackageGroupName": model_package_group,
            "ModelApprovalStatus": "PendingManualApproval",
            "InferenceSpecification": {
                "Containers": [
                    {
                        "Image": INFERENCE_IMAGE,
                        "ModelDataUrl": {
                            "Get": "Steps.Train.ModelArtifacts.S3ModelArtifacts"
                        },
                    }
                ],
                "SupportedContentTypes": ["application/json"],
                "SupportedResponseMIMETypes": ["application/json"],
                "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.large"],
                "SupportedTransformInstanceTypes": ["ml.m5.large"],
            },
        },
    }

    # ==================================================================
    # Step 4b: Fail (if MSE > threshold)
    # ==================================================================
    step_fail = {
        "Name": "FailModelQuality",
        "Type": "Fail",
        "Arguments": {
            "ErrorMessage": (
                f"Model quality check failed. "
                f"MSE exceeded threshold of {args.mse_threshold}"
            ),
        },
    }

    # ==================================================================
    # Step 4: Condition (quality gate)
    # ==================================================================
    step_condition = {
        "Name": "CheckModelQuality",
        "Type": "Condition",
        "Arguments": {
            "Conditions": [
                {
                    "Type": "LessThanOrEqualTo",
                    "LeftValue": {
                        "Std:JsonGet": {
                            "PropertyFile": {
                                "Get": "Steps.Evaluate.PropertyFiles.EvaluationReport"
                            },
                            "Path": "regression_metrics.mse.value",
                        }
                    },
                    "RightValue": args.mse_threshold,
                }
            ],
            "IfSteps": [step_register],
            "ElseSteps": [step_fail],
        },
        "DependsOn": ["Evaluate"],
    }

    # ==================================================================
    # Full Pipeline Definition
    # ==================================================================
    pipeline_definition = {
        "Version": "2020-12-01",
        "Metadata": {},
        "Parameters": [
            {
                "Name": "Epochs",
                "Type": "Integer",
                "DefaultValue": args.epochs,
            },
            {
                "Name": "Ticker",
                "Type": "String",
                "DefaultValue": args.ticker,
            },
            {
                "Name": "ModelName",
                "Type": "String",
                "DefaultValue": args.model,
            },
            {
                "Name": "InstanceType",
                "Type": "String",
                "DefaultValue": args.instance_type,
            },
            {
                "Name": "MseThreshold",
                "Type": "Float",
                "DefaultValue": args.mse_threshold,
            },
        ],
        "PipelineExperimentConfig": {
            "ExperimentName": {"Get": "Execution.PipelineName"},
            "TrialName": {"Get": "Execution.PipelineExecutionId"},
        },
        "Steps": [step_preprocess, step_train, step_evaluate, step_condition],
    }

    return pipeline_definition


# ============================================================================
# Pipeline Execution Polling
# ============================================================================


def poll_pipeline_execution(sm_client, execution_arn, interval=30):
    """Poll pipeline execution until terminal state."""
    logger.info(f"Polling execution every {interval}s...")
    terminal = {"Succeeded", "Failed", "Stopped"}

    while True:
        resp = sm_client.describe_pipeline_execution(
            PipelineExecutionArn=execution_arn
        )
        status = resp["PipelineExecutionStatus"]
        logger.info(f"  Pipeline status: {status}")

        if status in terminal:
            return resp

        time.sleep(interval)


def report_pipeline_results(sm_client, execution_arn, exec_desc):
    """Print pipeline execution results."""
    status = exec_desc["PipelineExecutionStatus"]

    if status == "Succeeded":
        logger.info("=" * 60)
        logger.info("PIPELINE EXECUTION SUCCEEDED")
        logger.info(f"  Execution ARN: {execution_arn}")
        logger.info("=" * 60)

        # List step details
        try:
            steps = sm_client.list_pipeline_execution_steps(
                PipelineExecutionArn=execution_arn
            )
            for step in steps.get("PipelineExecutionSteps", []):
                name = step.get("StepName", "?")
                step_status = step.get("StepStatus", "?")
                logger.info(f"  Step {name}: {step_status}")
        except ClientError:
            pass

    elif status == "Failed":
        reason = exec_desc.get("FailureReason", "Unknown")
        logger.error(f"PIPELINE EXECUTION FAILED: {reason}")

        # Show which step failed
        try:
            steps = sm_client.list_pipeline_execution_steps(
                PipelineExecutionArn=execution_arn
            )
            for step in steps.get("PipelineExecutionSteps", []):
                name = step.get("StepName", "?")
                step_status = step.get("StepStatus", "?")
                step_reason = step.get("FailureReason", "")
                logger.info(f"  Step {name}: {step_status} {step_reason}")
        except ClientError:
            pass

        sys.exit(1)
    else:
        logger.warning(f"Pipeline ended with status: {status}")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Create and run a SageMaker Pipeline for futures model CI/CD"
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Training epochs (default: 5 for validation, 200 for production)",
    )
    parser.add_argument(
        "--ticker", default="CL=F",
        help="Ticker to train (default: CL=F, single ticker to save cost)",
    )
    parser.add_argument(
        "--model", default="lstm",
        help="Model architecture: lstm, transformer, bilstm_attention (default: lstm)",
    )
    parser.add_argument(
        "--mse-threshold", type=float, default=0.05,
        help="MSE threshold for quality gate (default: 0.05 for validation, 0.005 for production)",
    )
    parser.add_argument(
        "--instance-type", default="ml.m5.large",
        help="SageMaker instance type (default: ml.m5.large)",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Execute the pipeline after upserting (default: upsert only)",
    )
    parser.add_argument(
        "--cleanup", action="store_true",
        help="Delete the pipeline (for cleanup after validation)",
    )
    parser.add_argument(
        "--role", default=None,
        help="IAM role ARN (auto-discovered if omitted)",
    )
    parser.add_argument(
        "--bucket", default=BUCKET,
        help="S3 bucket (default: auto-discovered)",
    )
    parser.add_argument(
        "--prefix", default=PREFIX,
        help="S3 prefix (default: from config_training.yaml)",
    )
    parser.add_argument(
        "--region", default="us-east-1",
        help="AWS region (default: us-east-1)",
    )
    parser.add_argument(
        "--poll-interval", type=int, default=30,
        help="Status poll interval in seconds (default: 30)",
    )
    args = parser.parse_args()

    boto_session = boto3.Session(region_name=args.region)
    iam = boto_session.client("iam")
    s3 = boto_session.client("s3")
    sm = boto_session.client("sagemaker")

    # 1. Discover IAM role
    role_arn = discover_role(iam, args.role)

    logger.info(f"Role:     {role_arn}")
    logger.info(f"Bucket:   {args.bucket}")
    logger.info(f"Prefix:   {args.prefix}")
    logger.info(f"Pipeline: {PIPELINE_NAME}")

    # Handle cleanup
    if args.cleanup:
        try:
            sm.delete_pipeline(PipelineName=PIPELINE_NAME)
            logger.info(f"Deleted pipeline: {PIPELINE_NAME}")
        except ClientError as e:
            logger.warning(f"Could not delete pipeline: {e}")
        return

    # 2. Upload scripts to S3
    upload_pipeline_scripts(s3, args.bucket, args.prefix)
    upload_training_source(s3, args.bucket, args.prefix)

    # 3. Build pipeline definition JSON
    logger.info("Building pipeline definition...")
    logger.info(f"  Epochs:        {args.epochs}")
    logger.info(f"  Ticker:        {args.ticker}")
    logger.info(f"  Model:         {args.model}")
    logger.info(f"  MSE threshold: {args.mse_threshold}")
    logger.info(f"  Instance:      {args.instance_type}")

    pipeline_def = build_pipeline_definition(
        args, role_arn, args.bucket, args.prefix, config
    )
    pipeline_json = json.dumps(pipeline_def)

    # 4. Upsert pipeline (try update first, create if not exists)
    try:
        sm.update_pipeline(
            PipelineName=PIPELINE_NAME,
            PipelineDefinition=pipeline_json,
            PipelineDescription="Futures ML CI/CD: Preprocess -> Train -> Evaluate -> Register/Fail",
            RoleArn=role_arn,
        )
        logger.info(f"Pipeline updated: {PIPELINE_NAME}")
    except ClientError as e:
        if "ResourceNotFound" in str(e) or "ValidationException" in str(e):
            sm.create_pipeline(
                PipelineName=PIPELINE_NAME,
                PipelineDefinition=pipeline_json,
                PipelineDescription="Futures ML CI/CD: Preprocess -> Train -> Evaluate -> Register/Fail",
                RoleArn=role_arn,
            )
            logger.info(f"Pipeline created: {PIPELINE_NAME}")
        else:
            raise

    # 5. Optionally execute
    if args.execute:
        logger.info("Starting pipeline execution...")
        exec_resp = sm.start_pipeline_execution(
            PipelineName=PIPELINE_NAME,
            PipelineExecutionDisplayName=f"{args.model}-{args.ticker.replace('=','')}-{args.epochs}ep",
        )
        execution_arn = exec_resp["PipelineExecutionArn"]
        logger.info(f"Pipeline execution started: {execution_arn}")
        logger.info(
            f"Monitor via:\n"
            f"  aws sagemaker describe-pipeline-execution "
            f"--pipeline-execution-arn {execution_arn}"
        )

        # Poll until completion
        exec_desc = poll_pipeline_execution(sm, execution_arn, args.poll_interval)
        report_pipeline_results(sm, execution_arn, exec_desc)
    else:
        logger.info(
            "Pipeline upserted but NOT executed. "
            "Pass --execute to start a run."
        )

    logger.info("=" * 60)
    logger.info("PIPELINE SETUP COMPLETE")
    logger.info(f"  Name:          {PIPELINE_NAME}")
    logger.info(f"  Ticker:        {args.ticker}")
    logger.info(f"  Model:         {args.model}")
    logger.info(f"  Epochs:        {args.epochs}")
    logger.info(f"  MSE threshold: {args.mse_threshold}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
