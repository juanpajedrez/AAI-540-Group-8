import logging
logger = logging.getLogger('aws')

from src.misc.logger_utils import log_function_call

import talib
# https://github.com/carlosOrtizM/aai-520-group7-final-project/blob/main/final-project-ipynbs/price_history_provider.ipynb

@log_function_call
def feature_talib_engineering(df):
    try:
        df = df.sort_index() # making sure it is sorted
        # Overlap Studies
        df['MA'] = talib.MA(df['close'], timeperiod=10) #windowed TAs produce lots of null values in demo...
        df['EMA'] = talib.EMA(df['close'], timeperiod=10)
        df['KAMA'] = talib.KAMA(df['close'], timeperiod=10)
        df['WMA'] = talib.WMA(df['close'], timeperiod=10)
        df['MidPrice'] = talib.MIDPRICE(df['high'], df['low'], timeperiod=10)

        # Momentum Indicator
        df['BOP'] = talib.BOP(df['open'], df['high'], df['low'],df['close'])
        df['CMO'] = talib.CMO(df['close'], timeperiod=10)
        df['MFI'] = talib.MFI(df['high'], df['low'], df['close'],df['volume'])
        df['ROC'] = talib.ROC(df['close'], timeperiod=10)
        df['WILLR'] = talib.WILLR(df['high'], df['low'], df['close'],timeperiod=14)

        ## volume
        df['AD'] = talib.AD(df['high'], df['low'], df['close'],df['volume'])
        df['OBV'] = talib.OBV(df['close'], df['volume'])

        ## Volatility
        df['NATR'] = talib.NATR(df['high'], df['low'], df['close'],timeperiod=14)
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'],timeperiod=14)
        df['TRANGE'] = talib.TRANGE(df['high'], df['low'], df['close'])

        ## Misc.
        df['TSF'] = talib.TSF(df['close'], timeperiod=14)

        # cutting the df short of the first two weeks, as there are null values for the TAs
        df = df.iloc[15:]
        return df
    except Exception as e:
            logger.error(f"TA computing error {e}")