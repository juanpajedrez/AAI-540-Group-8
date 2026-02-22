"""
ML Momentum Trading Algorithm for Zipline.

Uses 6 deep learning models (LSTM, Transformer, BiLSTM+Attention for CL=F and
GC=F) combined with EMA and McClellan-style oscillator signals to generate
trading decisions.

Signal composition: 70% model majority vote + 15% EMA + 15% McClellan oscillator
"""
import logging
import numpy as np
import pandas as pd
import talib
import yaml
from pathlib import Path

from zipline.api import order_target, record, symbol

from src.model.deployment import load_model_for_inference, predict
from src.model.data_loader import FEATURE_COLUMNS

logger = logging.getLogger('aws')

# Warmup: 39 (McClellan) + 15 (TA-Lib NaN) + 6 (safety) = 60 days
WARMUP_DAYS = 60
POSITION_SIZE = 100
EMA_SPAN = 10
MODELS = ['lstm', 'transformer', 'bilstm_attention']


def initialize(context):
    """Load config, models, and initialize trading state."""
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / 'configs' / 'config_training.yaml'

    with open(config_path) as f:
        full_config = yaml.safe_load(f)
    context.config = full_config['model_handler']

    context.tickers = context.config['tickers']  # ['CL=F', 'GC=F']
    context.lookback = context.config.get('lookback', 20)

    # Resolve zipline symbols
    context.assets = {}
    for ticker in context.tickers:
        context.assets[ticker] = symbol(ticker)

    # Load all 6 models (3 per ticker)
    context.models = {}
    for ticker in context.tickers:
        context.models[ticker] = {}
        for model_name in MODELS:
            model, scaler_feat, scaler_tgt = load_model_for_inference(
                model_name, ticker, context.config
            )
            context.models[ticker][model_name] = {
                'model': model,
                'scaler_features': scaler_feat,
                'scaler_target': scaler_tgt,
            }

    # EMA state per ticker
    context.ema_prev = {t: None for t in context.tickers}

    # McClellan oscillator buffers (19-day and 39-day EMA of daily changes)
    context.mcclel_fast = {t: None for t in context.tickers}  # 19-day EMA
    context.mcclel_slow = {t: None for t in context.tickers}  # 39-day EMA
    context.prev_close = {t: None for t in context.tickers}

    context.day_count = 0


def _compute_talib_features(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """Replicate TA-Lib feature engineering from feature_local_talib.py.

    Args:
        ohlcv_df: DataFrame with columns [open, high, low, close, volume],
                  at least 60 rows to produce 20+ valid rows after NaN drop.

    Returns:
        DataFrame with FEATURE_COLUMNS computed, NaN rows dropped.
    """
    df = ohlcv_df.copy()
    df = df.sort_index()

    # Overlap Studies
    df['MA'] = talib.MA(df['close'], timeperiod=10)
    df['EMA'] = talib.EMA(df['close'], timeperiod=10)
    df['KAMA'] = talib.KAMA(df['close'], timeperiod=10)
    df['WMA'] = talib.WMA(df['close'], timeperiod=10)
    df['MidPrice'] = talib.MIDPRICE(df['high'], df['low'], timeperiod=10)

    # Momentum Indicators
    df['BOP'] = talib.BOP(df['open'], df['high'], df['low'], df['close'])
    df['CMO'] = talib.CMO(df['close'], timeperiod=10)
    df['MFI'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'])
    df['ROC'] = talib.ROC(df['close'], timeperiod=10)
    df['WILLR'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)

    # Volume Indicators
    df['AD'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
    df['OBV'] = talib.OBV(df['close'], df['volume'])

    # Volatility Indicators
    df['NATR'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['TRANGE'] = talib.TRANGE(df['high'], df['low'], df['close'])

    # Miscellaneous
    df['TSF'] = talib.TSF(df['close'], timeperiod=14)

    # Drop NaN rows from windowed indicators
    df = df.iloc[15:]
    return df


def _ema_update(prev_ema, value, span):
    """Incremental EMA update."""
    alpha = 2.0 / (span + 1)
    if prev_ema is None:
        return value
    return alpha * value + (1 - alpha) * prev_ema


def handle_data(context, data):
    """Called on each trading bar. Computes signals and places orders."""
    context.day_count += 1

    if context.day_count < WARMUP_DAYS:
        return

    for ticker in context.tickers:
        asset = context.assets[ticker]

        if not data.can_trade(asset):
            continue

        # Get 60 days of OHLCV history
        try:
            hist_close = data.history(asset, 'close', bar_count=60, frequency='1d')
            hist_open = data.history(asset, 'open', bar_count=60, frequency='1d')
            hist_high = data.history(asset, 'high', bar_count=60, frequency='1d')
            hist_low = data.history(asset, 'low', bar_count=60, frequency='1d')
            hist_vol = data.history(asset, 'volume', bar_count=60, frequency='1d')
        except Exception:
            continue

        # Build OHLCV DataFrame for TA-Lib
        ohlcv = pd.DataFrame({
            'open': hist_open.values,
            'high': hist_high.values,
            'low': hist_low.values,
            'close': hist_close.values,
            'volume': hist_vol.values.astype(float),
        })

        # Compute TA-Lib features
        featured = _compute_talib_features(ohlcv)
        if len(featured) < context.lookback:
            continue

        # Extract last `lookback` rows of features in correct column order
        features = featured[FEATURE_COLUMNS].values[-context.lookback:]

        # Get yesterday's close for direction comparison
        yesterday_close = hist_close.iloc[-2]
        today_close = hist_close.iloc[-1]

        # --- Model majority vote (70% weight) ---
        votes = 0
        predictions = {}
        for model_name in MODELS:
            m = context.models[ticker][model_name]
            pred = predict(
                m['model'], features,
                m['scaler_features'], m['scaler_target']
            )
            pred_price = float(pred[0])
            predictions[model_name] = pred_price
            if pred_price > yesterday_close:
                votes += 1

        # Majority: 2+ of 3 models predict up -> +1, else -1
        model_signal = 1.0 if votes >= 2 else -1.0

        # --- EMA signal (15% weight) ---
        context.ema_prev[ticker] = _ema_update(
            context.ema_prev[ticker], today_close, EMA_SPAN
        )
        ema_current = context.ema_prev[ticker]
        # Need previous EMA to detect direction; on first signal day, neutral
        # We approximate by recomputing: EMA is rising if current > yesterday's
        # Since we update incrementally, compare current price vs EMA
        # Rising EMA: current close above EMA -> bullish
        ema_signal = 1.0 if today_close > ema_current else -1.0

        # --- McClellan-style oscillator (15% weight) ---
        daily_change = 0.0
        if context.prev_close[ticker] is not None:
            daily_change = today_close - context.prev_close[ticker]
        context.prev_close[ticker] = today_close

        context.mcclel_fast[ticker] = _ema_update(
            context.mcclel_fast[ticker], daily_change, 19
        )
        context.mcclel_slow[ticker] = _ema_update(
            context.mcclel_slow[ticker], daily_change, 39
        )
        mcclellan = context.mcclel_fast[ticker] - context.mcclel_slow[ticker]
        mcclellan_signal = 1.0 if mcclellan > 0 else -1.0

        # --- Combined signal ---
        combined = (0.70 * model_signal
                    + 0.15 * ema_signal
                    + 0.15 * mcclellan_signal)

        # --- Execute trades ---
        if combined > 0:
            order_target(asset, POSITION_SIZE)
        elif combined < 0:
            order_target(asset, 0)

        # --- Record values ---
        prefix = ticker.replace('=', '').replace('F', '')
        record(**{
            f'{prefix}_price': today_close,
            f'{prefix}_signal': combined,
            f'{prefix}_model_sig': model_signal,
            f'{prefix}_ema_sig': ema_signal,
            f'{prefix}_mccl_sig': mcclellan_signal,
            f'{prefix}_mccl_val': mcclellan,
        })
