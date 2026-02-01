import logging
logger = logging.getLogger('aws')

from src.misc.logger_utils import log_function_call

import talib
# https://github.com/carlosOrtizM/aai-520-group7-final-project/blob/main/final-project-ipynbs/price_history_provider.ipynb

@log_function_call
def feature_talib_engineering(df):
    try:
        # Overlap Studies
        df['MA'] = talib.MA(df['Close'], timeperiod=10) #windowed TAs produce lots of null values in demo...
        df['EMA'] = talib.EMA(df['Close'], timeperiod=10)
        df['KAMA'] = talib.KAMA(df['Close'], timeperiod=10)
        df['WMA'] = talib.WMA(df['Close'], timeperiod=10)
        df['MidPrice'] = talib.MIDPRICE(df['High'], df['Low'], timeperiod=10)

        # Momentum Indicator
        df['BOP'] = talib.BOP(df['Open'], df['High'], df['Low'],df['Close'])
        df['CMO'] = talib.CMO(df['Close'], timeperiod=10)
        df['MFI'] = talib.MFI(df['High'], df['Low'], df['Close'],df['Volume'])
        df['ROC'] = talib.ROC(df['Close'], timeperiod=10)
        df['WILLR'] = talib.WILLR(df['High'], df['Low'], df['Close'],timeperiod=14)

        ## Volume
        df['AD'] = talib.AD(df['High'], df['Low'], df['Close'],df['Volume'])
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])

        ## Volatility
        df['NATR'] = talib.NATR(df['High'], df['Low'], df['Close'],timeperiod=14)
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'],timeperiod=14)
        df['TRANGE'] = talib.TRANGE(df['High'], df['Low'], df['Close'])

        ## Misc.
        df['TSF'] = talib.TSF(df['Close'], timeperiod=14)

        # cutting the df short of the first two weeks, as there are null values for the TAs
        df = df.iloc[15:]
        return df
    except Exception as e:
            logger.error(f"TA computing error {e}")