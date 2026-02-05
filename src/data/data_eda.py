import logging
from pathlib import Path
logger = logging.getLogger('aws')

from ydata_profiling import ProfileReport

from src.misc.logger_utils import log_function_call

@log_function_call
def ticker_eda_profile(data_frame, ticker_symbol, visualization=True):
    try:
        if visualization:
            #https://docs.profiling.ydata.ai/latest/features/time_series_datasets/
            profile_rep_path = Path().cwd() / "files" / f"{ticker_symbol}.html"
            if not profile_rep_path.exists():
                profile = ProfileReport(data_frame, tsmode=True, sortby="Date", title=f'{ticker_symbol} Report')
                profile.to_file(profile_rep_path)
        else:
            print(f"\t\t\t<-------{ticker_symbol}------->\n")
            print(f"{data_frame.head()}\n")
            print(f"{data_frame.describe()}\n")
            print(f"{data_frame.info()}\n")
            print(f"\t\tRows:\t{len(data_frame)}\tColumns:\t{len(data_frame.columns)}\t\t\n")
            print(f"\t\tDuplicates:\t{data_frame.duplicated().sum()}\tMissing values:\t{len(data_frame.isna().sum())}\t\t\n")

    except Exception as e:
        logger.error(e)