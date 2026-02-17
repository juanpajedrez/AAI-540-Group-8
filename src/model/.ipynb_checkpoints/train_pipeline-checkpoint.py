import logging
import json
import time
import torch
import torch.nn as nn
import joblib
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger('aws')

from src.misc.logger_utils import log_function_call


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


@log_function_call
def train_model(model: nn.Module,
                train_loader,
                val_loader,
                config: dict,
                save_dir: Path,
                model_name: str,
                ticker: str) -> Dict:
    """Train a model with early stopping and LR scheduling.

    Returns:
        Training history dict with per-epoch train_loss and val_loss.
    """
    try:
        device = torch.device(config.get('device', 'cpu'))
        model = model.to(device)

        epochs = config.get('epochs', 200)
        lr = config.get('learning_rate', 0.001)
        weight_decay = config.get('weight_decay', 1e-5)
        grad_clip = config.get('grad_clip_norm', 1.0)
        es_patience = config.get('early_stopping_patience', 15)
        lr_patience = config.get('lr_scheduler_patience', 7)
        lr_factor = config.get('lr_scheduler_factor', 0.5)
        lr_min = config.get('lr_min', 1e-6)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=lr_patience,
            factor=lr_factor, min_lr=lr_min
        )
        early_stopping = EarlyStopping(patience=es_patience)

        history = {'train_loss': [], 'val_loss': [], 'lr': []}
        best_val_loss = float('inf')
        best_model_path = save_dir / f"{model_name}_{ticker}.pth"

        logger.info(f"[{model_name}][{ticker}] Starting training for up to {epochs} epochs")
        start_time = time.time()

        for epoch in range(epochs):
            # --- Train phase ---
            model.train()
            train_loss_sum = 0.0
            train_batches = 0

            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                predictions = model(x_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                train_loss_sum += loss.item()
                train_batches += 1

            avg_train_loss = train_loss_sum / max(train_batches, 1)

            # --- Validation phase ---
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0

            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    predictions = model(x_batch)
                    loss = criterion(predictions, y_batch)
                    val_loss_sum += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss_sum / max(val_batches, 1)

            current_lr = optimizer.param_groups[0]['lr']
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['lr'].append(current_lr)

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), best_model_path)

            scheduler.step(avg_val_loss)

            if (epoch + 1) % 20 == 0 or epoch == 0:
                logger.info(
                    f"[{model_name}][{ticker}] Epoch {epoch+1}/{epochs} - "
                    f"train_loss: {avg_train_loss:.6f}, "
                    f"val_loss: {avg_val_loss:.6f}, "
                    f"lr: {current_lr:.2e}"
                )

            if early_stopping(avg_val_loss):
                logger.info(
                    f"[{model_name}][{ticker}] Early stopping at epoch {epoch+1}"
                )
                break

        elapsed = time.time() - start_time
        logger.info(
            f"[{model_name}][{ticker}] Training complete. "
            f"Best val_loss: {best_val_loss:.6f}, "
            f"Epochs: {epoch+1}, "
            f"Time: {elapsed:.1f}s"
        )

        # Load best weights back into model
        model.load_state_dict(torch.load(best_model_path, weights_only=True))

        return history

    except Exception as e:
        logger.error(e)
        raise


@log_function_call
def save_training_artifacts(model: nn.Module,
                            scalers: Dict,
                            config: dict,
                            history: Dict,
                            save_dir: Path,
                            model_name: str,
                            ticker: str):
    """Save model weights, scalers, metadata, and training history."""
    try:
        save_dir.mkdir(parents=True, exist_ok=True)

        # Scalers
        joblib.dump(
            scalers['features'],
            save_dir / f"{model_name}_{ticker}_feature_scaler.pkl"
        )
        joblib.dump(
            scalers['target'],
            save_dir / f"{model_name}_{ticker}_target_scaler.pkl"
        )

        # Training history
        with open(save_dir / f"{model_name}_{ticker}_history.json", 'w') as f:
            json.dump(history, f, indent=2)

        # Metadata
        metadata = {
            'model_name': model_name,
            'ticker': ticker,
            'epochs_trained': len(history['train_loss']),
            'best_val_loss': min(history['val_loss']),
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'config': {k: v for k, v in config.items()
                       if isinstance(v, (int, float, str, bool))},
        }
        with open(save_dir / f"{model_name}_{ticker}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            f"[{model_name}][{ticker}] Artifacts saved to {save_dir}"
        )

    except Exception as e:
        logger.error(e)
        raise
