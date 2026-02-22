"""
Standalone training script for futures price prediction models.

Usage:
    python -m scripts.run_training
    python -m scripts.run_training --config configs/config_training.yaml
    python -m scripts.run_training --evaluate
"""
import sys
import yaml
import argparse
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.misc.logger import set_logger
logger = set_logger('aws')

from src.model.training import model_training_main
from src.model.evaluation import model_evaluation_main


def main():
    parser = argparse.ArgumentParser(
        description='Train futures price prediction models'
    )
    parser.add_argument(
        '--config', type=str,
        default='configs/config_training.yaml',
        help='Path to training config YAML'
    )
    parser.add_argument(
        '--evaluate', action='store_true',
        help='Run evaluation after training'
    )
    parser.add_argument(
        '--eval-only', action='store_true',
        help='Only run evaluation (skip training)'
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model_config = config['model_handler']

    if not args.eval_only:
        model_training_main(model_config)

    if args.evaluate or args.eval_only:
        model_evaluation_main(model_config)


if __name__ == '__main__':
    main()
