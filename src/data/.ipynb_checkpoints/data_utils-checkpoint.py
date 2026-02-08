# Import necessary modules
import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf

from typing import List, Union, Dict

import logging
from pathlib import Path
from pandas import DataFrame

logger = logging.getLogger('aws')

from src.misc.logger_utils import log_function_call

@log_function_call
def assert_yyyy_mm_dd(ticker_date:str) -> Union[None, ValueError]:
    '''
    ChatGPT generated: Function that will check wether a ticker_date is
    in the format yyyy_mm_dd or not.
    Args:
        ticker_date (str): String containing the ticker_date in format
        YYYY-MM-DD.
    '''
    try:
        datetime.strptime(ticker_date, "%Y-%m-%d")
    except ValueError as e:
        logger.error(e)
        raise AssertionError(f"Invalid date format, expected YYYY-MM-DD: {ticker_date}")

@log_function_call
def get_stocks_data_yahoo(
    tickers: List[str],
    start_date: str,
    end_date: str) -> List[pd.DataFrame]:

    '''
    Function to download the date from multiple desired tickers using the yahoo
    API through yfinance module in python:
    Args:
        - tickers (List[str]) : A list of strings containing all of the tickers necessary
        - start_date (str): A string in the format of YYYY-MM-DD to get a range from.
        - end_date (str): A string in the format of YYYY-MM-DD to get a range from.
    Returns:
        List[pd.DataFrame]: List of pd.DataFrames contianing the following columns per
        DataFrame:

    '''
    try:

        # Assert the input type is gonna be list
        assert isinstance(tickers, list), f"Please use a list of proper tickers, current type is: {type(tickers)}"

        # Make sure is the correct datetime format.
        assert_yyyy_mm_dd(start_date)
        assert_yyyy_mm_dd(end_date)

        # Returns a pd.Series containing a multi->index label of the tickers
        data = yf.download(tickers=tickers, start=start_date, end=end_date, auto_adjust=False)

        # Let's create a holder to df_tickers_splitted
        df_tickers_splitted = []

        for ticker in tickers:

            # Create a list to hold the ticker data
            ticker_data = {}

            #Add the datetime index orginally
            ticker_data["date"] = data.index
            total_rows = len(ticker_data["date"])

            # Iterate across each of the columns
            for col in data.columns:
                # If string ticker in the column, use it
                if ticker in col:
                    # Let's get the col_name and ticker_name
                    col_name, ticker_name = col[0], col[1]

                    # Assign the proper ticker_data to col_name
                    ticker_data[col_name] = data[col].to_list()

            # At the very end of the loop, add a new name called 'Ticker'
            ticker_data["Ticker"] = [ticker_name]*total_rows

            # Get the dataframe and set the index to be the datetime
            df_to_pass = pd.DataFrame(ticker_data)
            df_to_pass.set_index("Date")

            # Cast the 'Ticker' column to string and append to resulting list
            df_to_pass['Ticker'] = df_to_pass['Ticker'].astype(str).astype("string")
            df_tickers_splitted.append(df_to_pass)
        return df_tickers_splitted

    except Exception as e:
        logger.error(e)


@log_function_call
def setup_local_files_dirs():
    try:
        local_files_path = Path().cwd() / 'files'
        local_files_path.mkdir(parents = True, exist_ok = True)
        local_backtest_path = local_files_path / 'backtest'
        local_backtest_path.mkdir(parents = True, exist_ok = True)
        local_dataset_path = local_files_path / 'dataset'
        local_dataset_path.mkdir(parents = True, exist_ok = True)
    except Exception as e:
        logger.error(e)


@log_function_call
def backtest_dataset_creation(df: DataFrame, ticker: str):
    try:
        backtest_path = Path().cwd() / 'files' / 'backtest' / 'Daily'
        backtest_path.mkdir(parents = True, exist_ok = True)
        ticker_path = backtest_path / f"{ticker}.csv"
        if not ticker_path.exists():
            df.ffill()
            df[df.select_dtypes(np.float64).columns] = df.select_dtypes(np.float64).round().astype(int)
            df.to_csv(ticker_path)  # backtest needs this specific format
    except Exception as e:
        logger.error(e)


@log_function_call
def get_zipline_stocks(tickers: List[str],
    start_date: str,
    end_date: str) -> list[DataFrame] | None:
    try:
        dfs = []
        for ticker in tickers:
            df = yf.download(tickers=ticker, start=start_date, end=end_date, auto_adjust=False, repair=True
                             ) #multi_level_index=False
            df.columns = [x.lower() for x, _ in df.columns]
            backtest_dataset_creation(df, ticker)
            df.reset_index(inplace=True)
            dfs.append(df)
        return dfs

    except Exception as e:
        logger.error(e)


@log_function_call
def get_stocks_data_local(tickers: List[str]):
    try:
        df_tickers = []
        [df_tickers.append(pd.read_csv(f"files/backtest/Daily/{ticker}.csv")) for ticker in tickers]
        return df_tickers
    except Exception as e:
        logger.error(e)

@log_function_call
def get_backtest_file_paths() -> List[str, str]:
    try:
        backtest_path = Path().cwd() / 'files' / 'backtest' 
        backtest_daily_path = backtest_path / 'Daily'
        return backtest_path, backtest_daily_path
    except Exception as e:
        logger.error(e)

@log_function_call
def get_dataset_file_paths() -> List[str, str]:
    try:
        dataset_path = Path().cwd() / 'files' / 'dataset' 
        prod_path = dataset_path / 'prod'
        return dataset_path, prod_path
    except Exception as e:
        logger.error(e)