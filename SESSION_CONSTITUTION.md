# Session Constitution

Governing principles for this development session. All code, commits, and architectural decisions must align with these rules.

---

## 1. Mission

Build the remaining MLOps infrastructure (monitoring, CI/CD pipeline, UI) for the futures trading project so that a **teammate with a fresh AWS Learner Lab account can run a small number of deploy scripts and have the full system operational**.

---

## 2. Delivery Contract

### Who We're Building For
- **Alberto** (us): designs architecture, writes code, validates locally
- **Teammate (deployer)**: receives the repo, runs scripts, gets a working AWS environment
- The teammate should NOT need to understand internals -- scripts must be self-documenting and idempotent

### What "Done" Looks Like
The teammate clones the repo, sets credentials, and runs:
1. `deploy_s3_upload.py` -- data and models land in S3
2. `deploy_sagemaker_training.py` -- trains models (or uses existing)
3. `deploy_sagemaker_model_packages.py` -- registers in Model Registry
4. **NEW**: a monitoring deploy script -- sets up Model Quality Monitor + CloudWatch alarms
5. **NEW**: a CI/CD pipeline deploy script -- creates SageMaker Pipeline (preprocess -> train -> evaluate -> conditional register)
6. **NEW**: a UI launch script -- starts the dashboard locally or deploys it

---

## 3. Development Rules

### 3.1 Commit Discipline
- **Commit on every significant development** (new script, major refactor, feature complete)
- **Never push** -- Alberto reviews and pushes manually
- Commit messages: imperative mood, concise, explain the "why"
- Branch: `claude_athena` (current working branch)

### 3.2 Code Quality
- All deploy scripts must be **runnable standalone** with `.venv/bin/python scripts/<name>.py`
- Scripts that run INSIDE SageMaker containers must be **self-contained** (no imports from `src/`)
- Always verify model state_dict keys before loading (known architecture mismatch risk)
- No hardcoded credentials -- read from `credentials.txt`, env vars, or boto3 default chain
- Config-driven where possible (use `configs/config_training.yaml`)

### 3.3 AWS Constraints
- **Budget**: AWS Learner Lab has a $50 limit -- be cost-conscious
- **Instance types**: prefer `ml.m5.large` (CPU, cheapest) unless GPU is required
- **Region**: `us-east-1` only
- **IAM Role**: `arn:aws:iam::806081623304:role/LabRole` (Learner Lab fixed role)
- **Bucket**: `labs-usd-01`, prefix `AAI_540_group_8`
- **Idempotency**: all deploy scripts must handle "already exists" gracefully (no crashes on re-run)
- **Cleanup**: include teardown/cleanup functions or flags where AWS resources are created

### 3.4 Compatibility
- Python 3.10+ (SageMaker containers use 3.10)
- PyTorch 2.1.0 (matches SageMaker container image)
- Existing dataset naming convention (inverted: x=target, y=features) must be preserved
- New code must not break existing scripts

---

## 4. Architecture Alignment with Course Labs

Our implementations must map to the course lab patterns:

| Course Lab | Our Equivalent | Key Adaptation |
|------------|---------------|----------------|
| Lab 5.1: Model Monitoring (XGBoost endpoint + CloudWatch) | `deploy_sagemaker_monitoring.py` | PyTorch models, regression (not classification), futures domain |
| Lab 6.1: CI/CD Pipeline (SageMaker Pipelines) | `deploy_sagemaker_pipeline.py` | Multi-model (3 arch x 2 tickers), custom training container, our feature engineering |

### Lab 5.1 Mapping (Monitoring)
- **Reference**: XGBoost churn model on real-time endpoint with Model Quality Monitor
- **Our version**: PyTorch futures models, monitoring prediction quality over time
- **Must include**: baseline generation, monitoring schedule, CloudWatch alarm, data capture
- **Adaptation**: regression metrics (MSE, RMSE) instead of binary classification (F2, precision)

### Lab 6.1 Mapping (CI/CD Pipeline)
- **Reference**: Abalone regression with SageMaker Pipelines (preprocess -> train -> eval -> condition -> register/fail)
- **Our version**: Futures data with TA-Lib features, PyTorch training, multi-model evaluation
- **Must include**: pipeline parameters, processing step, training step, evaluation step, conditional model registration, fail step
- **Adaptation**: PyTorch estimator instead of built-in XGBoost, our feature engineering pipeline, multi-ticker handling

---

## 5. What We Are NOT Doing

- NOT changing the existing model architectures (LSTM, Transformer, BiLSTM-Attention)
- NOT retraining models unless the pipeline requires it
- NOT modifying the trading algorithm logic
- NOT deploying to production endpoints that cost money to keep running
- NOT pushing to remote -- local commits only

---

## 6. Agent Team Operating Rules (if using multi-agent)

- **Team lead** (main agent): orchestrates, reviews, commits
- **Specialists**: write code for their domain, report back
- All agents must read this constitution before starting work
- All code goes through team lead for commit
- Agents must not duplicate work -- check task list before starting
- Prefer parallel work on independent features (monitoring, pipeline, UI can be developed concurrently)

---

## 7. File Organization for New Features

```
scripts/
  deploy_sagemaker_monitoring.py     # NEW: deploy monitoring + CloudWatch
  deploy_sagemaker_pipeline.py       # NEW: deploy CI/CD pipeline
  sagemaker_pipeline_processing.py   # NEW: preprocessing step (runs in container)
  sagemaker_pipeline_evaluation.py   # NEW: evaluation step (runs in container)

src/
  ui/
    app.py                           # NEW: dashboard application
    backend_service.py               # NEW: API/backend for UI

configs/
  config_training.yaml               # EXISTING: may add pipeline params
```

---

## 8. Accountability Checkpoints

Commit triggers (at minimum):
1. Session constitution created (this file)
2. Each new deploy script reaches "runnable" state
3. Each container script (self-contained) is complete
4. UI reaches minimum viable state
5. Integration between components verified
6. Config updates that affect behavior

---

*Created: 2026-02-20 | Branch: claude_athena | Session: Claude + Alberto*
