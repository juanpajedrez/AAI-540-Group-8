# Import necessary modules
import pandas as pd
from datetime import datetime
import yfinance as yf

from typing import List, Union, Dict

import logging

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
def get_zipline_stocks(tickers: List[str],
    start_date: str,
    end_date: str) -> list[DataFrame] | None:
    try:
        dfs = []
        for ticker in tickers:
            df = yf.download(tickers=ticker, start=start_date, end=end_date, auto_adjust=False,
                             multi_level_index=False)
            df.columns = [x.lower() for x in df.columns]
            df["date"] = df.index

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