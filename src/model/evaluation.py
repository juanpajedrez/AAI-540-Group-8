import logging
import json
import torch
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger('aws')

from src.misc.logger_utils import log_function_call
from src.model.data_loader import create_data_loaders
from src.model.architectures import get_model
from src.model.evaluate_pipeline import (
    evaluate_model, plot_predictions, plot_training_curves, compare_models
)


@log_function_call
def model_evaluation_main(config: dict):
    """Main evaluation orchestrator. Loads trained models and evaluates."""
    try:
        tickers = config.get('tickers', ['CL=F', 'GC=F'])
        model_names = config.get('models',
                                 ['lstm', 'transformer', 'bilstm_attention'])
        save_base = Path.cwd() / 'files' / 'models'
        device = config.get('device', 'cpu')

        all_results = defaultdict(dict)

        for ticker in tickers:
            ticker_dir = save_base / ticker
            data = create_data_loaders(ticker, config)
            config['num_features'] = data['num_features']

            for model_name in model_names:
                model_path = ticker_dir / f"{model_name}_{ticker}.pth"
                if not model_path.exists():
                    logger.warning(
                        f"Model file not found: {model_path}, skipping"
                    )
                    continue

                model = get_model(model_name, config)
                model.load_state_dict(
                    torch.load(model_path, weights_only=True)
                )

                # Evaluate on test and validation splits
                test_metrics = evaluate_model(
                    model, data['test'], data['scaler_target'], device
                )
                val_metrics = evaluate_model(
                    model, data['val'], data['scaler_target'], device
                )

                all_results[model_name][ticker] = {
                    'test': {k: v for k, v in test_metrics.items()
                             if k not in ('predictions', 'actuals')},
                    'val': {k: v for k, v in val_metrics.items()
                            if k not in ('predictions', 'actuals')},
                }

                logger.info(
                    f"[{model_name}][{ticker}] Test RMSE: "
                    f"{test_metrics['rmse']:.4f}, "
                    f"MAE: {test_metrics['mae']:.4f}"
                )

                # Generate plots
                plot_predictions(
                    test_metrics['actuals'], test_metrics['predictions'],
                    ticker, model_name, 'test', ticker_dir
                )

                # Load and plot training curves
                history_path = ticker_dir / f"{model_name}_{ticker}_history.json"
                if history_path.exists():
                    with open(history_path) as f:
                        history = json.load(f)
                    plot_training_curves(history, ticker, model_name, ticker_dir)

        # Print comparison table
        compare_models(dict(all_results))

    except Exception as e:
        logger.error(e)
        raise
