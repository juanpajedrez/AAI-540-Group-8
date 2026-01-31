import yfinance as yf
import pandas as pd
from datetime import date, timedelta
import os

import logging
logger = logging.getLogger('aws')

# Import from src the get_stocks_data_yahoo
from src.data.data_utils import get_stocks_data_yahoo, get_stocks_data_local
from src.misc.logger_utils import log_function_call
from src.data.data_eda import ticker_eda_profile

@log_function_call
def data_handler_main(full_exec=False):
    # Let's create the different tickets for NYMEX oil and COMEX gold
    future_tickers = ["CL=F","GC=F"]

    if full_exec:
        # West Texas Intermediate Barrel
        list_dfs = get_stocks_data_yahoo(
            tickers=future_tickers, start_date="2010-01-04", end_date=f"{date.today()}",
        ) #2000-01-01 as far as we could go
    else:
        list_dfs = get_stocks_data_local(future_tickers)

    ## Printout
    for df, ft in zip(list_dfs, future_tickers):
        if full_exec:
            ## EDA
            df.to_csv(f'files/{ft}.csv')
            ticker_eda_profile(df, ft)


    ## missing (doing this on JAN 31ST):

    ## feature engineering WITH TA-LIB
    ## Train test validation split
    # Reserve data pool

    ## same stuff but in aws:
    ## raw data set store in an S3 Datalake.
    ## Set up Athena tables to enable cataloging and querying of your data.
    ## Perform exploratory data analysis on your data in a Sagemaker notebook.
    ## Perform feature engineering on raw data and store it in a Feature Store.
    ## Split your feature data into training (~40%), test (~10%) validation (~10%) datasets.
    ## Reserve some data for “production data” (~40%).