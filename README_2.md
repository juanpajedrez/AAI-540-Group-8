# AAI-540 Group 8 — ML Momentum Trading on AWS

Step-by-step guide to deploying the full MLOps pipeline: S3 data upload, SageMaker training, Model Registry, backtesting, and daily predictions.

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Environment Setup](#2-environment-setup)
3. [AWS Credentials](#3-aws-credentials)
4. [S3 Data Upload](#4-s3-data-upload)
5. [Model Training on SageMaker](#5-model-training-on-sagemaker)
6. [Model Registration](#6-model-registration)
7. [Running Backtests](#7-running-backtests)
8. [Daily Predictions](#8-daily-predictions)
9. [Project Structure](#9-project-structure)
10. [Cost Estimates](#10-cost-estimates)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Prerequisites

- **AWS Account** with access to SageMaker, S3, and IAM
- **IAM Role** with SageMaker full access (default: `LabRole`)
- **Python 3.10+**
- **TA-Lib C library** (required for technical indicators)
- **pip** package manager

## 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/juanpajedrez/AAI-540-Group-8.git
cd AAI-540-Group-8

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install TA-Lib C library (Ubuntu/Debian)
sudo apt-get install -y build-essential wget
wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz
tar -xzf ta-lib-0.6.4-src.tar.gz
cd ta-lib-0.6.4 && ./configure --prefix=/usr && make -j2 && sudo make install && cd ..
rm -rf ta-lib-0.6.4 ta-lib-0.6.4-src.tar.gz

# Install Python dependencies
pip install -e .
```

## 3. AWS Credentials

Configure credentials with a session token (required for AWS Academy / temporary access):

```bash
aws configure
# AWS Access Key ID: <your-key>
# AWS Secret Access Key: <your-secret>
# Default region name: us-east-1
# Default output format: json
```

For session tokens, add to `~/.aws/credentials`:

```ini
[default]
aws_access_key_id = YOUR_KEY
aws_secret_access_key = YOUR_SECRET
aws_session_token = YOUR_SESSION_TOKEN
```

Verify access:

```bash
aws sts get-caller-identity
aws s3 ls s3://labs-usd-01/AAI_540_group_8/
```

## 4. S3 Data Upload

Upload datasets, pre-trained models, and backtest data to S3:

```bash
.venv/bin/python scripts/deploy_s3_upload.py
```

What gets uploaded:
- `files/dataset/*.csv` — train/val/test splits for CL=F and GC=F
- `files/dataset/prod/*.csv` — production data (2019-2026)
- `files/models/CL=F/*` and `files/models/GC=F/*` — trained model artifacts (.pth, scalers, metadata)
- `files/backtest/*` — backtest configuration data

Dry run (list files without uploading):

```bash
.venv/bin/python scripts/deploy_s3_upload.py --dry-run
```

## 5. Model Training on SageMaker

Launch a SageMaker training job that trains all 6 models (3 architectures x 2 tickers):

```bash
.venv/bin/python scripts/deploy_sagemaker_training.py
```

Options:

```bash
.venv/bin/python scripts/deploy_sagemaker_training.py \
    --epochs 50 \
    --instance-type ml.m5.large \
    --role arn:aws:iam::806081623304:role/LabRole
```

The script will:
1. Package `scripts/sagemaker_training.py` and upload to S3
2. Launch a training job on `ml.m5.large`
3. Poll status every 30 seconds until completion
4. Report cost estimate and model artifact S3 location

**Models trained:**

| Model | Architecture | Inspiration |
|-------|-------------|-------------|
| LSTM | 2-layer stacked LSTM (hidden=64, dropout=0.2) | Rogendo/forex-lstm-models |
| Transformer | 2-layer TransformerEncoder (d=64, 4 heads, ff=128) | SatyamSinghal/financial-ttm |
| BiLSTM-Attention | 2-layer BiLSTM (hidden=64) + self-attention | JonusNattapong/xauusd-trading-ai |

## 6. Model Registration

Register trained models in SageMaker Model Registry with enriched model cards:

```bash
.venv/bin/python scripts/deploy_sagemaker_model_packages.py
```

This creates:
- **Model Package Groups**: `futures-cl-models` (CL=F) and `futures-gc-models` (GC=F)
- **3 versioned packages per group**: LSTM (v1), Transformer (v2), BiLSTM-Attention (v3)
- Each package includes the inference script (`code/inference.py`) for endpoint deployment
- Enriched metadata: architecture description, hyperparameters, training metrics, feature list

View in the SageMaker Console:
- SageMaker > Model Registry > `futures-cl-models`
- SageMaker > Model Registry > `futures-gc-models`

## 7. Running Backtests

Three options for running the ML momentum backtest:

### Option A: Local Backtest (zipline)

Requires zipline-reloaded and the prod data bundle:

```bash
.venv/bin/python scripts/prepare_backtest_data.py
.venv/bin/python scripts/run_backtest.py
```

### Option B: SageMaker Processing Job

Runs the backtest inside a SageMaker container (no local zipline needed):

```bash
.venv/bin/python scripts/deploy_sagemaker_backtest.py
```

The script:
1. Uploads `scripts/sagemaker_backtest_processing.py` to S3
2. Launches a Processing Job with model and data inputs from S3
3. Polls until completion
4. Downloads results (HTML report, equity curve, trade log) to `files/backtest/sagemaker_results/`

### Option C: Endpoint Backtest (notebook)

Deploys 6 SageMaker endpoints and runs the backtest by calling them day-by-day:

1. Open `notebooks/sagemaker_backtest_endpoint.ipynb` in Jupyter
2. Run all cells — deploys endpoints, iterates through prod data, generates equity curve
3. **Run the cleanup cell** when done to delete endpoints and stop charges

## 8. Daily Predictions

Generate live BUY/SELL signals using the latest market data:

1. Open `notebooks/sagemaker_daily_prediction.ipynb` in Jupyter
2. Run all cells:
   - Deploys (or reuses) 6 SageMaker endpoints
   - Fetches ~120 days of market data via `yfinance`
   - Computes TA-Lib features
   - Calls all 6 endpoints for predictions
   - Generates trading signals per ticker
3. **Run the cleanup cell** when done to delete endpoints

Output for each ticker:
- Current price and 3 model predictions with direction
- Signal components: Model Vote (70%), EMA (15%), McClellan (15%)
- Final recommendation: BUY / SELL / HOLD

## 9. Project Structure

```
AAI-540-Group-8/
|-- configs/
|   |-- config_training.yaml          # Hyperparameters and training config
|
|-- files/
|   |-- dataset/                      # Train/val/test CSVs
|   |   |-- prod/                     # Production data (2019-2026)
|   |-- models/
|   |   |-- CL=F/                     # Crude Oil model artifacts
|   |   |-- GC=F/                     # Gold model artifacts
|   |-- backtest/                     # Backtest results
|
|-- notebooks/
|   |-- sagemaker_backtest_endpoint.ipynb   # Endpoint-based backtest
|   |-- sagemaker_daily_prediction.ipynb   # Daily live predictions
|   |-- dataset_upload_s3_athena.ipynb     # S3/Athena data upload
|   |-- yfinance_dataset.ipynb             # Dataset exploration
|
|-- scripts/
|   |-- deploy_s3_upload.py                # Upload artifacts to S3
|   |-- deploy_sagemaker_training.py       # Launch SageMaker training job
|   |-- deploy_sagemaker_model_packages.py # Register models in Model Registry
|   |-- deploy_sagemaker_backtest.py       # Launch SageMaker Processing backtest
|   |-- sagemaker_training.py              # Runs INSIDE SageMaker training container
|   |-- sagemaker_backtest_processing.py   # Runs INSIDE SageMaker Processing container
|   |-- prepare_backtest_data.py           # Prepare local zipline bundle
|   |-- run_backtest.py                    # Run local zipline backtest
|
|-- src/
|   |-- model/
|   |   |-- architectures.py          # LSTM, Transformer, BiLSTM-Attention
|   |   |-- inference.py              # SageMaker endpoint inference handler
|   |   |-- data_loader.py            # Dataset loading and scaling
|   |   |-- training.py               # Training orchestrator
|   |   |-- train_pipeline.py         # Training loop and artifact saving
|   |   |-- deployment.py             # Local model loading for inference
|   |   |-- evaluation.py             # Model evaluation metrics
|   |
|   |-- trader/
|   |   |-- momentum_algo.py          # Zipline trading algorithm
|   |
|   |-- data/
|   |   |-- data_local_handler.py     # Data download and processing
|   |   |-- feature_local_talib.py    # TA-Lib feature engineering
|   |   |-- dataset_local_operator.py # Train/test splitting
|   |
|   |-- misc/
|       |-- aws_utils.py              # AWS helper functions
|       |-- logger.py                 # Logging setup
```

## 10. Cost Estimates

All costs are for `us-east-1` region using `ml.m5.large` ($0.115/hr):

| Step | Resource | Duration | Est. Cost |
|------|----------|----------|-----------|
| S3 Upload | S3 storage + PUT requests | instant | < $0.01 |
| Training | ml.m5.large x 1 | ~15 min | ~$0.03 |
| Registration | API calls only | instant | $0.00 |
| Processing Backtest | ml.m5.large x 1 | ~15 min | ~$0.03 |
| Endpoint Backtest | ml.m5.large x 6 | ~1 hr | ~$0.69 |
| Daily Prediction | ml.m5.large x 6 | ~10 min | ~$0.12 |

**Total for full pipeline run: ~$0.87**

**Warning:** SageMaker endpoints incur charges for every minute they are running. Always run the cleanup cell in the notebooks to delete endpoints when done.

## 11. Troubleshooting

### AWS Credentials Expired

```
botocore.exceptions.ClientError: ExpiredTokenException
```

**Fix:** Refresh your session token in `~/.aws/credentials`. AWS Academy tokens expire every ~4 hours.

### Endpoint Timeout / ModelError

```
ModelError: Received server error (503) from model
```

**Fix:** The endpoint may still be starting up (takes 5-8 minutes). Wait and retry. If persistent, check CloudWatch logs:

```bash
aws logs get-log-events \
    --log-group-name /aws/sagemaker/Endpoints/futures-cl-lstm \
    --log-stream-name AllTraffic/i-xxxxx
```

### Model Architecture Mismatch

```
RuntimeError: Error(s) in loading state_dict: Missing key(s)
```

**Fix:** The model architecture class attribute names must match the saved `.pth` state_dict keys exactly:
- `LSTMModel`: uses `dropout_layer` (not `dropout`)
- `TransformerModel`: uses `input_projection`, `pos_encoding`, `transformer_encoder`
- `BiLSTMAttentionModel`: uses `attention_fc`, `dropout_layer`

The inference script (`src/model/inference.py`) has the correct attribute names.

### TA-Lib Import Error

```
ImportError: libta_lib.so.0: cannot open shared object file
```

**Fix:** Install the TA-Lib C library before the Python package:

```bash
sudo apt-get install -y build-essential wget
wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz
tar -xzf ta-lib-0.6.4-src.tar.gz
cd ta-lib-0.6.4 && ./configure --prefix=/usr && make && sudo make install
pip install TA-Lib
```

### SageMaker Processing Job Fails

Check CloudWatch logs for the processing job:

```bash
aws logs get-log-events \
    --log-group-name /aws/sagemaker/ProcessingJobs \
    --log-stream-name futures-backtest-YYYY-MM-DD-HH-MM-SS/algo-1
```

Common causes:
- Missing input data in S3 (run `deploy_s3_upload.py` first)
- TA-Lib compilation failure (network issues downloading source)

### Endpoint Not Found During Cleanup

If endpoints were already deleted or never created, the cleanup cell will print skip messages. This is safe to ignore.
