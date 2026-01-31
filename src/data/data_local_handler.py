import yfinance as yf
import pandas as pd
from datetime import date
import os

import logging
logger = logging.getLogger('aws')

# Import from src the get_stocks_data_yahoo
from src.data.data_utils import get_stocks_data_yahoo, get_stocks_data_local
from src.misc.logger_utils import log_function_call

@log_function_call
def data_handler_main(download=False):
    # Let's create the different tickets for NYMEX oil and COMEX gold
    future_tickers = ["CL=F","GC=F"]

    if download:
        # West Texas Intermediate Barrel
        list_dfs = get_stocks_data_yahoo(
            tickers=future_tickers, start_date="2010-01-04", end_date=f"{date.today()}",
        ) #2000-01-01 as far as we could go
    else:
        list_dfs = get_stocks_data_local(future_tickers)

    for df, ft in zip(list_dfs, future_tickers):
        print(f"\t\t\t<-------{ft}------->\n")
        print(f"{df.head()}\n")
        print(f"\t\tRows:\t{len(df)}\tColumns:\t{len(df.columns)}\t\t\n")

        if download:
            df.to_csv(f'{ft}.csv')