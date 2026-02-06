import yfinance as yf
import pandas as pd
from datetime import date, timedelta
import os

import logging

logger = logging.getLogger('aws')

# Import from src the get_stocks_data_yahoo
from src.data.data_utils import get_stocks_data_local, get_zipline_stocks, setup_local_files_dirs
from src.misc.logger_utils import log_function_call
from src.data.data_eda import ticker_eda_profile
from src.data.feature_local_talib import feature_talib_engineering
from src.data.dataset_local_operator import dataset_operations, prod_dev_split

# Import from AWS utils
from src.misc.aws_utils import setup_aws_sagemaker_resources

@log_function_call
def data_handler_main(prefix:str,
    start_date:str = '2010-01-04',
    end_date:str = '2026-01-04',
    full_exec=False):
    
    try:

        # Let's create the different tickets for NYMEX oil and COMEX gold
        future_tickers = ["CL=F","GC=F"]

        if full_exec:
            # West Texas Intermediate Barrel
            list_dfs = get_zipline_stocks(
                tickers=future_tickers, start_date=start_date, end_date=end_date, #date.today()
            ) #2000-01-01 as far as we could go
        else:
            list_dfs = get_stocks_data_local(future_tickers)

        ## Setup local dir paths (To avoid errors)
        setup_local_files_dirs()
        
        ## Printout
        for df, ft in zip(list_dfs, future_tickers):

            df = prod_dev_split(df, ft, full_exec)

            # RUNNING EDAs AND FEATURE ENG. ONLY ON THE DEV DATA (60%)
            df.sort_index()

            ## EDA - No Feature
            ticker_eda_profile(df, f"{ft}_B", full_exec)

            # Add Technical Analysis Features
            df = feature_talib_engineering(df)

            # Non-OHLCV Columns, keeping just the Close column
            columns = ["open", "low", "high", "volume", "adj close"]
            df_features = df.copy()
            df_features = df_features.drop(columns=columns)

            ## Preliminary EDA - Features
            ticker_eda_profile(df_features, f"{ft}_F", full_exec)

            if full_exec:
                dataset_operations(df, ft, "close")

        ## same stuff but in aws:

        # Obtain the required sagemaker resources
        #sess, region, bucket = setup_aws_sagemaker_resources()
        
        ## raw data set store in an S3 Datalake.
        '''
        NOTE: We will be adding the following structure to the s3 bucket
        in order to ensure proper athena queries when used, also TECHNICALLY 
        the flow of MLOPs should be:
        
        [Raw data -> AWS s3 upload -> AWS athena queries -> feature engineering -> AWS feature Store]

        Instead we will be doing this, since we want to lower costs from using AWS feature store service
        [Raw data -> AWS S3 upload -> AWS Athena queries -> feature engineering -> AWS s3 upload -> AWS Athena queries]

        The file structure in AWS S3 bucket is as follows
        AAI-540-Group-8/
        ├── dataset/
        │   |── raw/
        |   |     └── (raw .csv files)
        |   |── prod/
        |   |     └── (processed prod.csv files)
        |   |── (processed.csv files)
        ├── backtest/
        │   └── (backtesting.csv files)
        '''
        
        ## Set up Athena tables to enable cataloging and querying of your data.
        ## Perform exploratory data analysis on your data in a Sagemaker notebook.
        ## Perform feature engineering on raw data and store it in a Feature Store.
        ## Split your feature data into training (~40%), test (~10%) validation (~10%) datasets.
        ## Reserve some data for “production data” (~40%).

    except Exception as e:
        logger.error(e)