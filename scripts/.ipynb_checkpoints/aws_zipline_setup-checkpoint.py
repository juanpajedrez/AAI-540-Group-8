"""
AWS deployment script for ML momentum backtesting pipeline.

Self-contained script (no src/ imports) that downloads models and data from S3,
sets up the zipline environment, and runs the backtest on an EC2 or SageMaker
instance.

--- EC2 / SageMaker Setup Instructions ---

    # 1. Launch an EC2 instance (e.g., t3.xlarge, Ubuntu 22.04)

    # 2. Install system dependencies
    sudo apt-get update
    sudo apt-get install -y python3.12 python3.12-venv python3-pip wget build-essential

    # 3. Install TA-Lib C library
    wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz
    tar -xzf ta-lib-0.6.4-src.tar.gz
    cd ta-lib-0.6.4
    ./configure --prefix=/usr
    make
    sudo make install
    cd ..

    # 4. Create virtualenv and install Python deps
    python3.12 -m venv .venv
    source .venv/bin/activate
    pip install boto3 pandas numpy torch scikit-learn joblib pyyaml \\
                TA-Lib zipline-reloaded exchange-calendars matplotlib

    # 5. Run this script
    python scripts/aws_zipline_setup.py \\
        --bucket YOUR_BUCKET \\
        --prefix AAI_540_group_8 \\
        --run-backtest

Usage:
    python scripts/aws_zipline_setup.py --bucket BUCKET --prefix PREFIX [--run-backtest]
"""
import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import boto3
import joblib
import numpy as np
import pandas as pd
import talib
import torch
import torch.nn as nn
import yaml
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('aws-zipline')


# ============================================================================
# Model Architectures (self-contained copies)
# ============================================================================

class LSTMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        nf = int(config.get('num_features', 20))
        hs = int(config.get('lstm_hidden_size', 64))
        nl = int(config.get('lstm_num_layers', 2))
        do = float(config.get('lstm_dropout', 0.2))
        self.lstm = nn.LSTM(nf, hs, nl, batch_first=True,
                            dropout=do if nl > 1 else 0.0)
        self.dropout = nn.Dropout(do)
        self.fc = nn.Linear(hs, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        nf = int(config.get('num_features', 20))
        dm = int(config.get('transformer_d_model', 64))
        nh = int(config.get('transformer_nhead', 4))
        nl = int(config.get('transformer_num_layers', 2))
        ff = int(config.get('transformer_dim_ff', 128))
        do = float(config.get('transformer_dropout', 0.1))
        lb = int(config.get('lookback', 20))
        # Attribute names must match the saved state_dict from training
        self.input_projection = nn.Linear(nf, dm)
        self.pos_encoding = nn.Parameter(torch.randn(1, lb, dm) * 0.1)
        layer = nn.TransformerEncoderLayer(dm, nh, ff, do, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, nl)
        self.fc = nn.Linear(dm, 1)

    def forward(self, x):
        x = self.input_projection(x) + self.pos_encoding
        return self.fc(self.transformer_encoder(x).mean(dim=1))


class BiLSTMAttentionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        nf = int(config.get('num_features', 20))
        hs = int(config.get('bilstm_hidden_size', 64))
        nl = int(config.get('bilstm_num_layers', 2))
        do = float(config.get('bilstm_dropout', 0.2))
        self.bilstm = nn.LSTM(nf, hs, nl, batch_first=True,
                              bidirectional=True,
                              dropout=do if nl > 1 else 0.0)
        # Attribute name must match saved state_dict from training
        self.attention_fc = nn.Linear(hs * 2, 1)
        self.dropout = nn.Dropout(do)
        self.fc = nn.Linear(hs * 2, 1)

    def forward(self, x):
        out, _ = self.bilstm(x)
        w = torch.softmax(self.attention_fc(out), dim=1)
        ctx = (out * w).sum(dim=1)
        return self.fc(self.dropout(ctx))


MODEL_REGISTRY = {
    'lstm': LSTMModel,
    'transformer': TransformerModel,
    'bilstm_attention': BiLSTMAttentionModel,
}


# ============================================================================
# Feature Columns (matching training pipeline)
# ============================================================================

FEATURE_COLUMNS = [
    'high', 'low', 'open', 'volume',
    'MA', 'EMA', 'KAMA', 'WMA', 'MidPrice',
    'BOP', 'CMO', 'MFI', 'ROC', 'WILLR',
    'AD', 'OBV', 'NATR', 'ATR', 'TRANGE', 'TSF'
]

TICKERS = ['CL=F', 'GC=F']
MODELS = ['lstm', 'transformer', 'bilstm_attention']


# ============================================================================
# Prediction logic (self-contained)
# ============================================================================

def predict_price(model, features, scaler_features, scaler_target):
    """Run inference. features: [lookback, num_features] raw (unscaled)."""
    model.eval()
    scaled = scaler_features.transform(features)
    x = torch.FloatTensor(scaled).unsqueeze(0)
    with torch.no_grad():
        pred = model(x).numpy().flatten()
    return scaler_target.inverse_transform(pred.reshape(-1, 1)).flatten()


# ============================================================================
# S3 Download Functions
# ============================================================================

def download_models_from_s3(bucket: str, prefix: str, local_dir: Path):
    """Download model artifacts (.pth, .pkl) from S3."""
    s3 = boto3.client('s3')
    local_dir.mkdir(parents=True, exist_ok=True)

    for ticker in TICKERS:
        ticker_dir = local_dir / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)

        for model_name in MODELS:
            for suffix in ['.pth', '_feature_scaler.pkl', '_target_scaler.pkl']:
                key = f"{prefix}/models/{ticker}/{model_name}_{ticker}{suffix}"
                local_path = ticker_dir / f"{model_name}_{ticker}{suffix}"
                logger.info(f"Downloading s3://{bucket}/{key}")
                s3.download_file(bucket, key, str(local_path))

    logger.info(f"Models downloaded to {local_dir}")


def download_prod_data_from_s3(bucket: str, prefix: str, local_dir: Path):
    """Download production CSVs from S3."""
    s3 = boto3.client('s3')
    local_dir.mkdir(parents=True, exist_ok=True)

    for ticker in TICKERS:
        key = f"{prefix}/dataset/prod/{ticker}x_prod.csv"
        local_path = local_dir / f"{ticker}x_prod.csv"
        logger.info(f"Downloading s3://{bucket}/{key}")
        s3.download_file(bucket, key, str(local_path))

    logger.info(f"Prod data downloaded to {local_dir}")


# ============================================================================
# Zipline Setup
# ============================================================================

def setup_zipline_extension(backtest_dir: Path):
    """Write ~/.zipline/extension.py for the prod bundle."""
    zipline_dir = Path.home() / '.zipline'
    zipline_dir.mkdir(parents=True, exist_ok=True)

    ext_content = f"""import pandas as pd
from zipline.data.bundles import register
from zipline.data.bundles.csvdir import csvdir_equities

register(
    'futures-prod-bundle',
    csvdir_equities(
        ['prod_daily'],
        '{backtest_dir}',
    ),
    calendar_name='NYSE',
    start_session=pd.Timestamp('2019-09-03'),
    end_session=pd.Timestamp('2026-02-05'),
)
"""
    ext_path = zipline_dir / 'extension.py'
    ext_path.write_text(ext_content)
    logger.info(f"Wrote zipline extension to {ext_path}")


def prepare_bundle_data(prod_dir: Path, output_dir: Path):
    """Convert prod CSVs to zipline csvdir format.

    Self-contained version (no src/ imports) of prepare_backtest_data.py.
    """
    from exchange_calendars import get_calendar

    cal = get_calendar('XNYS')
    nyse_days = cal.sessions_in_range('2019-09-03', '2026-02-05')
    output_dir.mkdir(parents=True, exist_ok=True)

    for ticker in TICKERS:
        input_path = prod_dir / f"{ticker}x_prod.csv"
        df = pd.read_csv(input_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        df = df[df.index.isin(nyse_days)]

        # Zipline stores OHLC as uint32 (price * 1000); clip negatives
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].clip(lower=0.01)

        out = pd.DataFrame({
            'Date': df.index.strftime('%Y-%m-%d'),
            'open': df['open'].round(2),
            'high': df['high'].round(2),
            'low': df['low'].round(2),
            'close': df['close'].round(2),
            'volume': df['volume'].astype(int),
            'dividend': 0,
            'split': 1.0,
        })

        output_path = output_dir / f"{ticker}.csv"
        out.to_csv(output_path, index=False)
        logger.info(f"Wrote {len(out)} rows to {output_path}")


# ============================================================================
# TA-Lib Feature Computation (self-contained)
# ============================================================================

def compute_talib_features(ohlcv_df):
    """Replicate feature_local_talib.py feature engineering."""
    df = ohlcv_df.copy().sort_index()

    df['MA'] = talib.MA(df['close'], timeperiod=10)
    df['EMA'] = talib.EMA(df['close'], timeperiod=10)
    df['KAMA'] = talib.KAMA(df['close'], timeperiod=10)
    df['WMA'] = talib.WMA(df['close'], timeperiod=10)
    df['MidPrice'] = talib.MIDPRICE(df['high'], df['low'], timeperiod=10)

    df['BOP'] = talib.BOP(df['open'], df['high'], df['low'], df['close'])
    df['CMO'] = talib.CMO(df['close'], timeperiod=10)
    df['MFI'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'])
    df['ROC'] = talib.ROC(df['close'], timeperiod=10)
    df['WILLR'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)

    df['AD'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
    df['OBV'] = talib.OBV(df['close'], df['volume'])

    df['NATR'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['TRANGE'] = talib.TRANGE(df['high'], df['low'], df['close'])

    df['TSF'] = talib.TSF(df['close'], timeperiod=14)

    df = df.iloc[15:]
    return df


# ============================================================================
# Trading Algorithm (self-contained for AWS)
# ============================================================================

def ema_update(prev, value, span):
    alpha = 2.0 / (span + 1)
    return value if prev is None else alpha * value + (1 - alpha) * prev


def make_algo(config, model_dir):
    """Build initialize/handle_data callables with models loaded from model_dir.

    Returns (initialize_fn, handle_data_fn) for use with zipline.run_algorithm.
    """
    from zipline.api import order_target, record, symbol as zsymbol

    WARMUP = 60
    POSITION_SIZE = 100
    EMA_SPAN = 10

    def initialize(context):
        context.config = config
        context.tickers = TICKERS
        context.lookback = int(config.get('lookback', 20))

        context.assets = {t: zsymbol(t) for t in TICKERS}

        context.models = {}
        for ticker in TICKERS:
            context.models[ticker] = {}
            for mname in MODELS:
                pth = model_dir / ticker / f"{mname}_{ticker}.pth"
                feat_scl = model_dir / ticker / f"{mname}_{ticker}_feature_scaler.pkl"
                tgt_scl = model_dir / ticker / f"{mname}_{ticker}_target_scaler.pkl"

                model = MODEL_REGISTRY[mname](config)
                model.load_state_dict(torch.load(pth, weights_only=True))
                model.eval()

                context.models[ticker][mname] = {
                    'model': model,
                    'scaler_features': joblib.load(feat_scl),
                    'scaler_target': joblib.load(tgt_scl),
                }

        context.ema_prev = {t: None for t in TICKERS}
        context.mcclel_fast = {t: None for t in TICKERS}
        context.mcclel_slow = {t: None for t in TICKERS}
        context.prev_close = {t: None for t in TICKERS}
        context.day_count = 0

    def handle_data(context, data):
        context.day_count += 1
        if context.day_count < WARMUP:
            return

        for ticker in context.tickers:
            asset = context.assets[ticker]
            if not data.can_trade(asset):
                continue

            try:
                hist_close = data.history(asset, 'close', 60, '1d')
                hist_open = data.history(asset, 'open', 60, '1d')
                hist_high = data.history(asset, 'high', 60, '1d')
                hist_low = data.history(asset, 'low', 60, '1d')
                hist_vol = data.history(asset, 'volume', 60, '1d')
            except Exception:
                continue

            ohlcv = pd.DataFrame({
                'open': hist_open.values,
                'high': hist_high.values,
                'low': hist_low.values,
                'close': hist_close.values,
                'volume': hist_vol.values.astype(float),
            })

            featured = compute_talib_features(ohlcv)
            if len(featured) < context.lookback:
                continue

            features = featured[FEATURE_COLUMNS].values[-context.lookback:]
            yesterday_close = hist_close.iloc[-2]
            today_close = hist_close.iloc[-1]

            # Model majority vote (70%)
            votes = 0
            for mname in MODELS:
                m = context.models[ticker][mname]
                pred = predict_price(
                    m['model'], features,
                    m['scaler_features'], m['scaler_target']
                )
                if float(pred[0]) > yesterday_close:
                    votes += 1
            model_signal = 1.0 if votes >= 2 else -1.0

            # EMA signal (15%)
            context.ema_prev[ticker] = ema_update(
                context.ema_prev[ticker], today_close, EMA_SPAN
            )
            ema_signal = 1.0 if today_close > context.ema_prev[ticker] else -1.0

            # McClellan oscillator (15%)
            daily_change = 0.0
            if context.prev_close[ticker] is not None:
                daily_change = today_close - context.prev_close[ticker]
            context.prev_close[ticker] = today_close

            context.mcclel_fast[ticker] = ema_update(
                context.mcclel_fast[ticker], daily_change, 19
            )
            context.mcclel_slow[ticker] = ema_update(
                context.mcclel_slow[ticker], daily_change, 39
            )
            mcclellan = context.mcclel_fast[ticker] - context.mcclel_slow[ticker]
            mcclellan_signal = 1.0 if mcclellan > 0 else -1.0

            combined = (0.70 * model_signal
                        + 0.15 * ema_signal
                        + 0.15 * mcclellan_signal)

            if combined > 0:
                order_target(asset, POSITION_SIZE)
            elif combined < 0:
                order_target(asset, 0)

            prefix = ticker.replace('=', '').replace('F', '')
            record(**{
                f'{prefix}_price': today_close,
                f'{prefix}_signal': combined,
            })

    return initialize, handle_data


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='AWS setup and backtest for ML momentum trading')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--prefix', default='AAI_540_group_8',
                        help='S3 key prefix')
    parser.add_argument('--work-dir', default='/tmp/zipline_backtest',
                        help='Local working directory')
    parser.add_argument('--run-backtest', action='store_true',
                        help='Run the backtest after setup')
    args = parser.parse_args()

    work = Path(args.work_dir)
    model_dir = work / 'models'
    prod_dir = work / 'prod_data'
    backtest_dir = work / 'backtest'
    bundle_dir = backtest_dir / 'prod_daily'
    results_dir = work / 'results'

    # Step 1: Download from S3
    download_models_from_s3(args.bucket, args.prefix, model_dir)
    download_prod_data_from_s3(args.bucket, args.prefix, prod_dir)

    # Step 2: Prepare bundle data
    prepare_bundle_data(prod_dir, bundle_dir)

    # Step 3: Setup zipline extension
    setup_zipline_extension(backtest_dir)

    # Step 4: Ingest bundle
    logger.info("Ingesting zipline bundle...")
    result = subprocess.run(
        ['zipline', 'ingest', '-b', 'futures-prod-bundle'],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        logger.error(f"Ingest failed: {result.stderr}")
        sys.exit(1)
    logger.info("Bundle ingested.")

    if not args.run_backtest:
        logger.info("Setup complete. Use --run-backtest to run the backtest.")
        return

    # Step 5: Run backtest
    logger.info("Running backtest...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from zipline import run_algorithm

    config_path = work / 'config.yaml'
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)['model_handler']
    else:
        # Default config matching training
        config = {
            'lookback': 20, 'num_features': 20,
            'lstm_hidden_size': 64, 'lstm_num_layers': 2, 'lstm_dropout': 0.2,
            'transformer_d_model': 64, 'transformer_nhead': 4,
            'transformer_num_layers': 2, 'transformer_dim_ff': 128,
            'transformer_dropout': 0.1,
            'bilstm_hidden_size': 64, 'bilstm_num_layers': 2,
            'bilstm_dropout': 0.2,
        }

    init_fn, handle_fn = make_algo(config, model_dir)

    perf = run_algorithm(
        start=pd.Timestamp('2019-09-03'),
        end=pd.Timestamp('2026-02-05'),
        initialize=init_fn,
        handle_data=handle_fn,
        capital_base=10_000.0,
        bundle='futures-prod-bundle',
    )

    # Save results
    results_dir.mkdir(parents=True, exist_ok=True)
    perf.to_csv(results_dir / 'performance.csv')

    fig, ax = plt.subplots(figsize=(14, 6))
    perf['portfolio_value'].plot(ax=ax, title='Portfolio Value Over Time')
    ax.set_ylabel('Portfolio Value ($)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / 'equity_curve.png', dpi=150)
    plt.close(fig)

    total_ret = (perf['portfolio_value'].iloc[-1] / 10_000) - 1
    daily_rets = perf['portfolio_value'].pct_change().dropna()
    sharpe = (daily_rets.mean() / daily_rets.std() * (252 ** 0.5)
              if daily_rets.std() > 0 else 0.0)
    cummax = perf['portfolio_value'].cummax()
    max_dd = ((perf['portfolio_value'] - cummax) / cummax).min()

    summary = {
        'total_return_pct': round(total_ret * 100, 2),
        'sharpe_ratio': round(float(sharpe), 4),
        'max_drawdown_pct': round(float(max_dd) * 100, 2),
        'final_value': round(float(perf['portfolio_value'].iloc[-1]), 2),
    }
    with open(results_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\nBacktest complete. Results saved to:", results_dir)
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
