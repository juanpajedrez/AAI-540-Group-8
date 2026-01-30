import yfinance as yf
import pandas as pd
from datetime import datetime
import os

import logging
logger = logging.getLogger('aws')

# Import from src the get_stocks_data_yahoo
from src.data.data_utils import get_stocks_data_yahoo
from src.misc.logger_utils import log_function_call

@log_function_call
def data_handler_main():
    # Let's create the different tickets for NYMEX oil and COMEX gold
    future_tickers = ["CL=F","GC=F"]

    # West Texas Intermediate Barrel
    list_dfs = get_stocks_data_yahoo(
        tickers=future_tickers, start_date="2010-01-01", end_date="2026-01-29")

    for df in list_dfs:
        print(df.head())
        print(df.info())