import logging
import torch
from pathlib import Path

logger = logging.getLogger('aws')

from src.misc.logger_utils import log_function_call
from src.model.data_loader import create_data_loaders
from src.model.architectures import get_model
from src.model.train_pipeline import train_model, save_training_artifacts


@log_function_call
def model_training_main(config: dict):
    """Main training orchestrator. Trains all specified models for all tickers."""
    try:
        tickers = config.get('tickers', ['CL=F', 'GC=F'])
        model_names = config.get('models',
                                 ['lstm', 'transformer', 'bilstm_attention'])
        save_base = Path.cwd() / 'files' / 'models'
        save_base.mkdir(parents=True, exist_ok=True)

        for ticker in tickers:
            ticker_dir = save_base / ticker
            ticker_dir.mkdir(parents=True, exist_ok=True)

            # Create data loaders (shared across models for same ticker)
            data = create_data_loaders(ticker, config)

            # Update config with actual num_features from data
            config['num_features'] = data['num_features']

            for model_name in model_names:
                logger.info(f"Training {model_name} for {ticker}")
                model = get_model(model_name, config)

                param_count = sum(p.numel() for p in model.parameters())
                logger.info(f"[{model_name}] Parameters: {param_count:,}")

                history = train_model(
                    model, data['train'], data['val'],
                    config, ticker_dir, model_name, ticker
                )

                save_training_artifacts(
                    model,
                    {'features': data['scaler_features'],
                     'target': data['scaler_target']},
                    config, history, ticker_dir, model_name, ticker
                )

                logger.info(f"Completed {model_name} for {ticker}")

    except Exception as e:
        logger.error(e)
        raise
