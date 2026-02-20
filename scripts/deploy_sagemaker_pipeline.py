"""
Create and optionally execute a SageMaker Pipeline for futures model CI/CD.

Pipeline structure:
    Preprocess -> Train -> Evaluate -> Condition (MSE check) -> Register or Fail

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
import logging
import sys
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

PIPELINE_NAME = "futures-ml-pipeline"


def discover_role_and_session():
    """Get SageMaker role, session, bucket, and config from existing project patterns."""
    from sagemaker.core.helper.session_helper import get_execution_role, Session

    role = get_execution_role()
    sess = Session()
    bucket = sess.default_bucket()

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    prefix = config["data_handler"]["prefix"]
    return role, sess, bucket, prefix, config


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


def build_pipeline(args, role, bucket, prefix, config):
    """Build and return a SageMaker Pipeline object."""
    from sagemaker.workflow.pipeline_context import PipelineSession
    from sagemaker.workflow.parameters import (
        ParameterInteger,
        ParameterString,
        ParameterFloat,
    )
    from sagemaker.workflow.steps import ProcessingStep, TrainingStep
    from sagemaker.workflow.step_collections import RegisterModel
    from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
    from sagemaker.workflow.condition_step import ConditionStep
    from sagemaker.workflow.fail_step import FailStep
    from sagemaker.workflow.properties import PropertyFile
    from sagemaker.workflow.functions import JsonGet, Join
    from sagemaker.workflow.pipeline import Pipeline
    from sagemaker.processing import (
        ScriptProcessor,
        ProcessingInput,
        ProcessingOutput,
    )
    from sagemaker.pytorch import PyTorch
    from sagemaker.inputs import TrainingInput
    from sagemaker.model_metrics import MetricsSource, ModelMetrics

    pipeline_session = PipelineSession()

    # ======================================================================
    # Pipeline Parameters
    # ======================================================================
    param_epochs = ParameterInteger(name="Epochs", default_value=args.epochs)
    param_ticker = ParameterString(name="Ticker", default_value=args.ticker)
    param_model = ParameterString(name="ModelName", default_value=args.model)
    param_instance_type = ParameterString(
        name="InstanceType", default_value=args.instance_type
    )
    param_mse_threshold = ParameterFloat(
        name="MseThreshold", default_value=args.mse_threshold
    )

    # S3 paths
    s3_data = f"s3://{bucket}/{prefix}/dataset"
    s3_pipeline_output = f"s3://{bucket}/{prefix}/pipeline-output"

    # ======================================================================
    # Step 1: Preprocess
    # ======================================================================
    preprocess_processor = ScriptProcessor(
        image_uri=TRAINING_IMAGE,
        role=role,
        instance_count=1,
        instance_type=args.instance_type,
        command=["python3"],
        sagemaker_session=pipeline_session,
        env={"TICKER": args.ticker},
    )

    step_preprocess = ProcessingStep(
        name="Preprocess",
        processor=preprocess_processor,
        inputs=[
            ProcessingInput(
                source=s3_data,
                destination="/opt/ml/processing/input/data",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="processed",
                source="/opt/ml/processing/output",
                destination=f"{s3_pipeline_output}/preprocessed",
            ),
        ],
        code=str(
            PROJECT_ROOT / "scripts" / "sagemaker_pipeline_preprocessing.py"
        ),
    )

    # ======================================================================
    # Step 2: Train
    # ======================================================================
    mh = config.get("model_handler", {})
    hyperparameters = {
        "epochs": args.epochs,
        "batch_size": mh.get("batch_size", 32),
        "lookback": mh.get("lookback", 20),
        "learning_rate": mh.get("learning_rate", 0.001),
        "weight_decay": mh.get("weight_decay", 1e-5),
        "grad_clip_norm": mh.get("grad_clip_norm", 1.0),
        "early_stopping_patience": mh.get("early_stopping_patience", 15),
        "lr_scheduler_patience": mh.get("lr_scheduler_patience", 7),
        "lr_scheduler_factor": mh.get("lr_scheduler_factor", 0.5),
        "lr_min": mh.get("lr_min", 1e-6),
        "models": args.model,
        "tickers": args.ticker,
        "lstm_hidden_size": mh.get("lstm_hidden_size", 64),
        "lstm_num_layers": mh.get("lstm_num_layers", 2),
        "lstm_dropout": mh.get("lstm_dropout", 0.2),
        "transformer_d_model": mh.get("transformer_d_model", 64),
        "transformer_nhead": mh.get("transformer_nhead", 4),
        "transformer_num_layers": mh.get("transformer_num_layers", 2),
        "transformer_dim_ff": mh.get("transformer_dim_ff", 128),
        "transformer_dropout": mh.get("transformer_dropout", 0.1),
        "bilstm_hidden_size": mh.get("bilstm_hidden_size", 64),
        "bilstm_num_layers": mh.get("bilstm_num_layers", 2),
        "bilstm_dropout": mh.get("bilstm_dropout", 0.2),
    }

    estimator = PyTorch(
        entry_point="sagemaker_training.py",
        source_dir=str(PROJECT_ROOT / "scripts"),
        role=role,
        instance_count=1,
        instance_type=args.instance_type,
        framework_version="2.1",
        py_version="py310",
        hyperparameters=hyperparameters,
        output_path=f"{s3_pipeline_output}/training",
        sagemaker_session=pipeline_session,
        max_run=3600,
    )

    step_train = TrainingStep(
        name="Train",
        estimator=estimator,
        inputs={
            "training": TrainingInput(
                s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs[
                    "processed"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )

    # ======================================================================
    # Step 3: Evaluate
    # ======================================================================
    eval_processor = ScriptProcessor(
        image_uri=TRAINING_IMAGE,
        role=role,
        instance_count=1,
        instance_type=args.instance_type,
        command=["python3"],
        sagemaker_session=pipeline_session,
        env={
            "TICKER": args.ticker,
            "MODEL_NAME": args.model,
            "LOOKBACK": str(mh.get("lookback", 20)),
        },
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    step_evaluate = ProcessingStep(
        name="Evaluate",
        processor=eval_processor,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/input/model",
            ),
            ProcessingInput(
                source=step_preprocess.properties.ProcessingOutputConfig.Outputs[
                    "processed"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/output",
                destination=f"{s3_pipeline_output}/evaluation",
            ),
        ],
        code=str(
            PROJECT_ROOT / "scripts" / "sagemaker_pipeline_evaluation.py"
        ),
        property_files=[evaluation_report],
    )

    # ======================================================================
    # Step 4: Condition (quality gate)
    # ======================================================================
    cond_mse = ConditionLessThanOrEqualTo(
        left=JsonGet(
            step_name=step_evaluate.name,
            property_file=evaluation_report,
            json_path="regression_metrics.mse.value",
        ),
        right=param_mse_threshold,
    )

    # ======================================================================
    # Step 5a: Register Model (if MSE <= threshold)
    # ======================================================================
    # Determine model package group name from ticker
    ticker_safe = args.ticker.replace("=", "").replace("F", "").lower()
    model_package_group = f"futures-{ticker_safe}-models"

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(
                on="/",
                values=[
                    step_evaluate.properties.ProcessingOutputConfig.Outputs[
                        "evaluation"
                    ].S3Output.S3Uri,
                    "evaluation.json",
                ],
            ),
            content_type="application/json",
        ),
    )

    step_register = RegisterModel(
        name="RegisterModel",
        estimator=estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group,
        approval_status="PendingManualApproval",
        model_metrics=model_metrics,
    )

    # ======================================================================
    # Step 5b: Fail (if MSE > threshold)
    # ======================================================================
    step_fail = FailStep(
        name="FailModelQuality",
        error_message=Join(
            on=" ",
            values=[
                "Model quality check failed.",
                "MSE exceeded threshold of",
                str(args.mse_threshold),
            ],
        ),
    )

    # ======================================================================
    # Step 4 (continued): Condition Step wiring
    # ======================================================================
    step_condition = ConditionStep(
        name="CheckModelQuality",
        conditions=[cond_mse],
        if_steps=[step_register],
        else_steps=[step_fail],
    )

    # ======================================================================
    # Build Pipeline
    # ======================================================================
    pipeline = Pipeline(
        name=PIPELINE_NAME,
        parameters=[
            param_epochs,
            param_ticker,
            param_model,
            param_instance_type,
            param_mse_threshold,
        ],
        steps=[step_preprocess, step_train, step_evaluate, step_condition],
        sagemaker_session=pipeline_session,
    )

    return pipeline


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
    args = parser.parse_args()

    # Discover AWS resources
    role, sess, bucket, prefix, config = discover_role_and_session()

    if args.role:
        role = args.role

    logger.info(f"Role:     {role}")
    logger.info(f"Bucket:   {bucket}")
    logger.info(f"Prefix:   {prefix}")
    logger.info(f"Pipeline: {PIPELINE_NAME}")

    # Handle cleanup
    if args.cleanup:
        try:
            sm = boto3.client("sagemaker", region_name="us-east-1")
            sm.delete_pipeline(PipelineName=PIPELINE_NAME)
            logger.info(f"Deleted pipeline: {PIPELINE_NAME}")
        except ClientError as e:
            logger.warning(f"Could not delete pipeline: {e}")
        return

    # Upload processing scripts to S3
    s3 = boto3.client("s3", region_name="us-east-1")
    upload_pipeline_scripts(s3, bucket, prefix)

    # Build the pipeline
    logger.info("Building pipeline...")
    logger.info(f"  Epochs:        {args.epochs}")
    logger.info(f"  Ticker:        {args.ticker}")
    logger.info(f"  Model:         {args.model}")
    logger.info(f"  MSE threshold: {args.mse_threshold}")
    logger.info(f"  Instance:      {args.instance_type}")

    pipeline = build_pipeline(args, role, bucket, prefix, config)

    # Upsert (create or update) the pipeline
    logger.info("Upserting pipeline definition...")
    pipeline.upsert(role_arn=role)
    logger.info(f"Pipeline upserted: {PIPELINE_NAME}")

    # Optionally execute
    if args.execute:
        logger.info("Starting pipeline execution...")
        execution = pipeline.start()
        logger.info(f"Pipeline execution started: {execution.arn}")
        logger.info(
            f"Monitor in SageMaker Studio or via:\n"
            f"  aws sagemaker describe-pipeline-execution "
            f"--pipeline-execution-arn {execution.arn}"
        )
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
