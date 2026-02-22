"""
SageMaker PyTorch Inference Script for futures price prediction models.

Runs inside the pytorch-inference:2.1.0-cpu-py310-ubuntu20.04-sagemaker container.
Loaded automatically when the model.tar.gz contains code/inference.py.

Functions:
    model_fn    - Load model, scalers, and metadata from model_dir
    input_fn    - Parse JSON request body into feature tensor
    predict_fn  - Run inference with scaling/inverse-scaling
    output_fn   - Serialize prediction to JSON response
"""
import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ============================================================================
# Model Architectures (self-contained copies matching src/model/architectures.py)
# ============================================================================

class LSTMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        nf = int(config.get("num_features", 20))
        hs = int(config.get("lstm_hidden_size", 64))
        nl = int(config.get("lstm_num_layers", 2))
        do = float(config.get("lstm_dropout", 0.2))
        self.lstm = nn.LSTM(nf, hs, nl, batch_first=True,
                            dropout=do if nl > 1 else 0.0)
        self.dropout_layer = nn.Dropout(do)
        self.fc = nn.Linear(hs, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout_layer(out[:, -1, :]))


class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        nf = int(config.get("num_features", 20))
        dm = int(config.get("transformer_d_model", 64))
        nh = int(config.get("transformer_nhead", 4))
        nl = int(config.get("transformer_num_layers", 2))
        ff = int(config.get("transformer_dim_ff", 128))
        do = float(config.get("transformer_dropout", 0.1))
        lb = int(config.get("lookback", 20))
        self.input_projection = nn.Linear(nf, dm)
        self.pos_encoding = nn.Parameter(torch.randn(1, lb, dm) * 0.1)
        layer = nn.TransformerEncoderLayer(dm, nh, ff, do, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, nl)
        self.fc = nn.Linear(dm, 1)

    def forward(self, x):
        x = self.input_projection(x) + self.pos_encoding
        return self.fc(self.transformer_encoder(x).mean(dim=1))


class BiLSTMAttentionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        nf = int(config.get("num_features", 20))
        hs = int(config.get("bilstm_hidden_size", 64))
        nl = int(config.get("bilstm_num_layers", 2))
        do = float(config.get("bilstm_dropout", 0.2))
        self.bilstm = nn.LSTM(nf, hs, nl, batch_first=True,
                              bidirectional=True,
                              dropout=do if nl > 1 else 0.0)
        self.attention_fc = nn.Linear(hs * 2, 1)
        self.dropout_layer = nn.Dropout(do)
        self.fc = nn.Linear(hs * 2, 1)

    def forward(self, x):
        out, _ = self.bilstm(x)
        w = torch.softmax(self.attention_fc(out), dim=1)
        ctx = (out * w).sum(dim=1)
        return self.fc(self.dropout_layer(ctx))


MODEL_REGISTRY = {
    "lstm": LSTMModel,
    "transformer": TransformerModel,
    "bilstm_attention": BiLSTMAttentionModel,
}


# ============================================================================
# SageMaker Inference Functions
# ============================================================================

def model_fn(model_dir):
    """Load model, scalers, and metadata from the model directory.

    The model.tar.gz is extracted to model_dir with this structure:
        <model_name>_<ticker>.pth
        <model_name>_<ticker>_feature_scaler.pkl
        <model_name>_<ticker>_target_scaler.pkl
        <model_name>_<ticker>_metadata.json
        code/inference.py  (this file)

    Returns:
        dict with keys: model, scaler_features, scaler_target, model_name, ticker, config
    """
    model_dir = Path(model_dir)
    logger.info(f"Loading model from {model_dir}")
    logger.info(f"Contents: {list(model_dir.iterdir())}")

    # Find metadata file to discover model_name and ticker
    metadata_files = list(model_dir.glob("*_metadata.json"))
    if not metadata_files:
        raise FileNotFoundError(f"No *_metadata.json found in {model_dir}")

    meta_path = metadata_files[0]
    with open(meta_path) as f:
        metadata = json.load(f)

    model_name = metadata["model_name"]
    ticker = metadata["ticker"]
    config = metadata.get("config", {})
    logger.info(f"Discovered model: {model_name} for {ticker}")

    # Load model architecture and weights
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    model = MODEL_REGISTRY[model_name](config)
    pth_path = model_dir / f"{model_name}_{ticker}.pth"
    model.load_state_dict(torch.load(pth_path, map_location="cpu", weights_only=True))
    model.eval()
    logger.info(f"Loaded weights from {pth_path}")

    # Load scalers
    scaler_features = joblib.load(model_dir / f"{model_name}_{ticker}_feature_scaler.pkl")
    scaler_target = joblib.load(model_dir / f"{model_name}_{ticker}_target_scaler.pkl")
    logger.info("Loaded feature and target scalers")

    return {
        "model": model,
        "scaler_features": scaler_features,
        "scaler_target": scaler_target,
        "model_name": model_name,
        "ticker": ticker,
        "config": config,
    }


def input_fn(request_body, content_type="application/json"):
    """Parse JSON request body into a numpy array of features.

    Expected JSON format:
        {"features": [[f1, f2, ...], [f1, f2, ...], ...]}

    where the outer list has `lookback` rows and each inner list has `num_features` values.
    Features should be RAW (unscaled) values in FEATURE_COLUMNS order:
        high, low, open, volume, MA, EMA, KAMA, WMA, MidPrice,
        BOP, CMO, MFI, ROC, WILLR, AD, OBV, NATR, ATR, TRANGE, TSF
    """
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}. Use application/json.")

    data = json.loads(request_body)
    features = np.array(data["features"], dtype=np.float32)
    logger.info(f"Input features shape: {features.shape}")
    return features


def predict_fn(features, model_dict):
    """Scale features, run inference, inverse-transform prediction.

    Args:
        features: numpy array of shape [lookback, num_features], raw/unscaled
        model_dict: dict returned by model_fn

    Returns:
        dict with predicted_price, model_name, ticker
    """
    model = model_dict["model"]
    scaler_features = model_dict["scaler_features"]
    scaler_target = model_dict["scaler_target"]

    # Scale features using the training scaler
    scaled = scaler_features.transform(features)
    x = torch.FloatTensor(scaled).unsqueeze(0)  # [1, lookback, num_features]

    # Run inference
    with torch.no_grad():
        pred_scaled = model(x).numpy().flatten()

    # Inverse-transform to get actual price
    pred_price = scaler_target.inverse_transform(
        pred_scaled.reshape(-1, 1)
    ).flatten()[0]

    return {
        "predicted_price": float(pred_price),
        "model_name": model_dict["model_name"],
        "ticker": model_dict["ticker"],
    }


def output_fn(prediction, accept="application/json"):
    """Serialize prediction dict to JSON."""
    if accept != "application/json":
        raise ValueError(f"Unsupported accept type: {accept}. Use application/json.")
    return json.dumps(prediction), "application/json"
