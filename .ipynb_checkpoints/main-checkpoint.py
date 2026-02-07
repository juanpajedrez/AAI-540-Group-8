import os
import yaml
from src.misc.logger import set_logger
from pathlib import Path

logger = set_logger('aws')

import yfinance as yf
from src.data.data_local_handler import data_handler_main

def main (): 
    config_path = Path().cwd() / 'configs'

    # NOTE: Work in progress, adding argparser in the future
    config_file = 'config_example.yaml'
    config_selected = config_path / config_file

    with open(config_selected, 'r') as f:
        config = yaml.safe_load(f)

    # this will act as a glorified cli function caller (which are one off scripts of functions run sequentially)
    data_handler_main(full_exec=config['data_handler']['full_exec'],
                     start_date = config['data_handler']['start_date'],
                     end_date = config['data_handler']['end_date'],
                     prefix = config['data_handler']['prefix'],
                     upload_backtest = config['data_handler']['upload_backtest'],
                     upload_dataset = config['data_handler']['upload_dataset'],
                     upload_production = config['data_handler']['upload_production'])

if __name__ == '__main__':
    main()
