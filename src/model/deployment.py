import logging
import torch
import joblib
import numpy as np
from pathlib import Path

logger = logging.getLogger('aws')

from src.misc.logger_utils import log_function_call
from src.model.architectures import get_model


@log_function_call
def load_model_for_inference(model_name: str, ticker: str, config: dict):
    """Load a trained model and its scalers for inference.

    Returns:
        Tuple of (model, scaler_features, scaler_target)
    """
    try:
        model_dir = Path.cwd() / 'files' / 'models' / ticker

        model = get_model(model_name, config)
        model.load_state_dict(
            torch.load(model_dir / f"{model_name}_{ticker}.pth",
                        weights_only=True)
        )
        model.eval()

        scaler_features = joblib.load(
            model_dir / f"{model_name}_{ticker}_feature_scaler.pkl"
        )
        scaler_target = joblib.load(
            model_dir / f"{model_name}_{ticker}_target_scaler.pkl"
        )

        return model, scaler_features, scaler_target

    except Exception as e:
        logger.error(e)
        raise


@log_function_call
def predict(model, features: np.ndarray,
            scaler_features, scaler_target) -> np.ndarray:
    """Run inference on new feature data.

    Args:
        model: Trained PyTorch model
        features: Raw feature array, shape [lookback, num_features]
        scaler_features: Fitted MinMaxScaler for features
        scaler_target: Fitted MinMaxScaler for target

    Returns:
        Predicted closing price(s) in original scale.
    """
    try:
        model.eval()

        # Scale features
        scaled = scaler_features.transform(features)

        # Add batch dimension: [1, lookback, num_features]
        x = torch.FloatTensor(scaled).unsqueeze(0)

        with torch.no_grad():
            pred_scaled = model(x).numpy().flatten()

        # Inverse transform to original price scale
        pred_original = scaler_target.inverse_transform(
            pred_scaled.reshape(-1, 1)
        ).flatten()

        return pred_original

    except Exception as e:
        logger.error(e)
        raise
