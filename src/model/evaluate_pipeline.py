import logging
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger('aws')

from src.misc.logger_utils import log_function_call


@log_function_call
def evaluate_model(model: torch.nn.Module,
                   data_loader,
                   scaler_target,
                   device: str = 'cpu'
                   ) -> Dict:
    """Evaluate model and return metrics in original price scale.

    Returns:
        Dict with mse, rmse, mae, mape, and actual/predicted arrays.
    """
    try:
        model.eval()
        model = model.to(device)
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(device)
                predictions = model(x_batch)
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(y_batch.numpy())

        preds = np.concatenate(all_preds, axis=0).flatten()
        targets = np.concatenate(all_targets, axis=0).flatten()

        # Inverse transform to original price scale
        preds_original = scaler_target.inverse_transform(
            preds.reshape(-1, 1)
        ).flatten()
        targets_original = scaler_target.inverse_transform(
            targets.reshape(-1, 1)
        ).flatten()

        mse = mean_squared_error(targets_original, preds_original)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets_original, preds_original)
        mape = np.mean(
            np.abs((targets_original - preds_original) /
                   np.clip(np.abs(targets_original), 1e-8, None))
        ) * 100

        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'predictions': preds_original,
            'actuals': targets_original,
        }

    except Exception as e:
        logger.error(e)
        raise


@log_function_call
def plot_predictions(actuals: np.ndarray,
                     predictions: np.ndarray,
                     ticker: str,
                     model_name: str,
                     split_name: str,
                     save_dir: Path):
    """Plot actual vs predicted closing prices and save as PNG."""
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(actuals, label='Actual', alpha=0.8)
        ax.plot(predictions, label='Predicted', alpha=0.8)
        ax.set_title(f'{model_name} - {ticker} ({split_name})')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Close Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = f"{model_name}_{ticker}_{split_name}_predictions.png"
        fig.savefig(save_dir / filename, dpi=100)
        plt.close(fig)
        logger.info(f"Saved prediction plot: {filename}")

    except Exception as e:
        logger.error(e)


@log_function_call
def plot_training_curves(history: Dict,
                         ticker: str,
                         model_name: str,
                         save_dir: Path):
    """Plot training and validation loss curves."""
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history['train_loss'], label='Train Loss')
        ax.plot(history['val_loss'], label='Val Loss')
        ax.set_title(f'{model_name} - {ticker} Training Curves')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = f"{model_name}_{ticker}_training_curves.png"
        fig.savefig(save_dir / filename, dpi=100)
        plt.close(fig)
        logger.info(f"Saved training curves: {filename}")

    except Exception as e:
        logger.error(e)


@log_function_call
def compare_models(all_results: Dict) -> str:
    """Print a formatted comparison table of all model-ticker results.

    Args:
        all_results: {model_name: {ticker: {'test': metrics, 'val': metrics}}}

    Returns:
        Formatted string of the comparison table.
    """
    try:
        header = (f"{'Model':<22} {'Ticker':<8} {'Split':<6} "
                  f"{'MSE':>10} {'RMSE':>10} {'MAE':>10} {'MAPE%':>10}")
        separator = '-' * len(header)
        lines = [separator, header, separator]

        for model_name, tickers in all_results.items():
            for ticker, splits in tickers.items():
                for split_name, metrics in splits.items():
                    line = (f"{model_name:<22} {ticker:<8} {split_name:<6} "
                            f"{metrics['mse']:>10.4f} "
                            f"{metrics['rmse']:>10.4f} "
                            f"{metrics['mae']:>10.4f} "
                            f"{metrics['mape']:>10.4f}")
                    lines.append(line)

        lines.append(separator)
        table = '\n'.join(lines)
        logger.info(f"\nModel Comparison:\n{table}")
        print(table)
        return table

    except Exception as e:
        logger.error(e)
        return ""
