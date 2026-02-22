import os
import yaml
from src.misc.logger import set_logger
from pathlib import Path

logger = set_logger('aws')

import yfinance as yf
from src.data.data_local_handler import data_handler_main
from src.model.training import model_training_main
from src.model.evaluation import model_evaluation_main

def main ():
    config_path = Path().cwd() / 'configs'

    # NOTE: Work in progress, adding argparser in the future
    config_file = 'config_training.yaml'
    config_selected = config_path / config_file

    with open(config_selected, 'r') as f:
        config = yaml.safe_load(f)

    # this will act as a glorified cli function caller (which are one off scripts of functions run sequentially)
    if config.get('data_handler', {}).get('full_exec', False):
        data_handler_main(full_exec=config['data_handler']['full_exec'],
                         start_date = config['data_handler']['start_date'],
                         end_date = config['data_handler']['end_date'],
                         prefix = config['data_handler']['prefix'],
                         upload_backtest = config['data_handler']['upload_backtest'],
                         upload_dataset = config['data_handler']['upload_dataset'],
                         upload_production = config['data_handler']['upload_production'])

    # Model training and evaluation pipeline
    model_config = config.get('model_handler', {})
    if model_config.get('run_training', False):
        model_training_main(model_config)
    if model_config.get('run_evaluation', False):
        model_evaluation_main(model_config)

if __name__ == '__main__':
    main()
