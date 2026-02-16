"""
SageMaker Processing Job script for ML momentum backtest.

Self-contained (no src/ imports). Runs inside a SageMaker Processing container:
  - Installs TA-Lib C library + Python dependencies
  - Loads models and prod data from /opt/ml/processing/input/
  - Runs zipline backtest with 70% model vote + 15% EMA + 15% McClellan
  - Generates HTML report with embedded charts
  - Saves outputs to /opt/ml/processing/output/
"""
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("sagemaker-backtest")


# ============================================================================
# Dependency Installation (runs before any ML imports)
# ============================================================================

def install_dependencies():
    """Install TA-Lib C library and Python packages inside the container."""
    logger.info("Installing system dependencies...")
    subprocess.check_call(
        ["apt-get", "update", "-qq"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    subprocess.check_call(
        ["apt-get", "install", "-y", "-qq", "build-essential", "wget"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    logger.info("Downloading and compiling TA-Lib C library...")
    talib_url = "https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz"
    subprocess.check_call(["wget", "-q", talib_url, "-O", "/tmp/ta-lib.tar.gz"])
    subprocess.check_call(["tar", "-xzf", "/tmp/ta-lib.tar.gz", "-C", "/tmp"])
    subprocess.check_call(["./configure", "--prefix=/usr"], cwd="/tmp/ta-lib-0.6.4")
    subprocess.check_call(["make", "-j2"], cwd="/tmp/ta-lib-0.6.4")
    subprocess.check_call(["make", "install"], cwd="/tmp/ta-lib-0.6.4")

    logger.info("Installing Python packages...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "TA-Lib", "zipline-reloaded", "exchange-calendars", "matplotlib",
    ])
    logger.info("All dependencies installed.")


install_dependencies()

# --- Now safe to import ML / zipline libs ---

import numpy as np

# Patch numpy 2.x removals used by empyrical
if not hasattr(np, "NINF"):
    np.NINF = -np.inf
if not hasattr(np, "PINF"):
    np.PINF = np.inf

import base64
import io
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import talib
import torch
import torch.nn as nn

# ============================================================================
# SageMaker Processing paths
# ============================================================================

INPUT_MODELS = Path("/opt/ml/processing/input/models")
INPUT_DATA = Path("/opt/ml/processing/input/data")
OUTPUT_DIR = Path("/opt/ml/processing/output")

WORK_DIR = Path("/tmp/zipline_backtest")
BUNDLE_DIR = WORK_DIR / "prod_daily"

START_DATE = pd.Timestamp("2019-09-03")
END_DATE = pd.Timestamp("2026-02-05")
CAPITAL = 10_000.0

TICKERS = ["CL=F", "GC=F"]
MODELS = ["lstm", "transformer", "bilstm_attention"]

FEATURE_COLUMNS = [
    "high", "low", "open", "volume",
    "MA", "EMA", "KAMA", "WMA", "MidPrice",
    "BOP", "CMO", "MFI", "ROC", "WILLR",
    "AD", "OBV", "NATR", "ATR", "TRANGE", "TSF",
]

DEFAULT_CONFIG = {
    "lookback": 20, "num_features": 20,
    "lstm_hidden_size": 64, "lstm_num_layers": 2, "lstm_dropout": 0.2,
    "transformer_d_model": 64, "transformer_nhead": 4,
    "transformer_num_layers": 2, "transformer_dim_ff": 128,
    "transformer_dropout": 0.1,
    "bilstm_hidden_size": 64, "bilstm_num_layers": 2,
    "bilstm_dropout": 0.2,
}


# ============================================================================
# Model Architectures (self-contained copies)
# ============================================================================

class LSTMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        nf = int(config.get("num_features", 20))
        hs = int(config.get("lstm_hidden_size", 64))
        nl = int(config.get("lstm_num_layers", 2))
        do = float(config.get("lstm_dropout", 0.2))
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
        nf = int(config.get("num_features", 20))
        dm = int(config.get("transformer_d_model", 64))
        nh = int(config.get("transformer_nhead", 4))
        nl = int(config.get("transformer_num_layers", 2))
        ff = int(config.get("transformer_dim_ff", 128))
        do = float(config.get("transformer_dropout", 0.1))
        lb = int(config.get("lookback", 20))
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
        nf = int(config.get("num_features", 20))
        hs = int(config.get("bilstm_hidden_size", 64))
        nl = int(config.get("bilstm_num_layers", 2))
        do = float(config.get("bilstm_dropout", 0.2))
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
    "lstm": LSTMModel,
    "transformer": TransformerModel,
    "bilstm_attention": BiLSTMAttentionModel,
}


# ============================================================================
# Prediction / Feature Logic
# ============================================================================

def predict_price(model, features, scaler_features, scaler_target):
    """Run inference. features: [lookback, num_features] raw (unscaled)."""
    model.eval()
    scaled = scaler_features.transform(features)
    x = torch.FloatTensor(scaled).unsqueeze(0)
    with torch.no_grad():
        pred = model(x).numpy().flatten()
    return scaler_target.inverse_transform(pred.reshape(-1, 1)).flatten()


def compute_talib_features(ohlcv_df):
    """Replicate feature_local_talib.py feature engineering."""
    df = ohlcv_df.copy().sort_index()

    df["MA"] = talib.MA(df["close"], timeperiod=10)
    df["EMA"] = talib.EMA(df["close"], timeperiod=10)
    df["KAMA"] = talib.KAMA(df["close"], timeperiod=10)
    df["WMA"] = talib.WMA(df["close"], timeperiod=10)
    df["MidPrice"] = talib.MIDPRICE(df["high"], df["low"], timeperiod=10)

    df["BOP"] = talib.BOP(df["open"], df["high"], df["low"], df["close"])
    df["CMO"] = talib.CMO(df["close"], timeperiod=10)
    df["MFI"] = talib.MFI(df["high"], df["low"], df["close"], df["volume"])
    df["ROC"] = talib.ROC(df["close"], timeperiod=10)
    df["WILLR"] = talib.WILLR(df["high"], df["low"], df["close"], timeperiod=14)

    df["AD"] = talib.AD(df["high"], df["low"], df["close"], df["volume"])
    df["OBV"] = talib.OBV(df["close"], df["volume"])

    df["NATR"] = talib.NATR(df["high"], df["low"], df["close"], timeperiod=14)
    df["ATR"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)
    df["TRANGE"] = talib.TRANGE(df["high"], df["low"], df["close"])

    df["TSF"] = talib.TSF(df["close"], timeperiod=14)

    df = df.iloc[15:]
    return df


# ============================================================================
# Zipline Setup
# ============================================================================

def setup_zipline_extension(backtest_dir: Path):
    """Write ~/.zipline/extension.py for the prod bundle."""
    zipline_dir = Path.home() / ".zipline"
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
    ext_path = zipline_dir / "extension.py"
    ext_path.write_text(ext_content)
    logger.info(f"Wrote zipline extension to {ext_path}")


def prepare_bundle_data(prod_dir: Path, output_dir: Path):
    """Convert prod CSVs to zipline csvdir format."""
    from exchange_calendars import get_calendar

    cal = get_calendar("XNYS")
    nyse_days = cal.sessions_in_range("2019-09-03", "2026-02-05")
    output_dir.mkdir(parents=True, exist_ok=True)

    for ticker in TICKERS:
        input_path = prod_dir / f"{ticker}x_prod.csv"
        df = pd.read_csv(input_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        df = df[df.index.isin(nyse_days)]

        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].clip(lower=0.01)

        out = pd.DataFrame({
            "Date": df.index.strftime("%Y-%m-%d"),
            "open": df["open"].round(2),
            "high": df["high"].round(2),
            "low": df["low"].round(2),
            "close": df["close"].round(2),
            "volume": df["volume"].astype(int),
            "dividend": 0,
            "split": 1.0,
        })

        output_path = output_dir / f"{ticker}.csv"
        out.to_csv(output_path, index=False)
        logger.info(f"Wrote {len(out)} rows to {output_path}")


# ============================================================================
# Trading Algorithm
# ============================================================================

def ema_update(prev, value, span):
    alpha = 2.0 / (span + 1)
    return value if prev is None else alpha * value + (1 - alpha) * prev


def make_algo(config, model_dir):
    """Build initialize/handle_data callables with models loaded from model_dir."""
    from zipline.api import order_target, record, symbol as zsymbol

    WARMUP = 60
    POSITION_SIZE = 100
    EMA_SPAN = 10

    def initialize(context):
        context.config = config
        context.tickers = TICKERS
        context.lookback = int(config.get("lookback", 20))

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
                    "model": model,
                    "scaler_features": joblib.load(feat_scl),
                    "scaler_target": joblib.load(tgt_scl),
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
                hist_close = data.history(asset, "close", 60, "1d")
                hist_open = data.history(asset, "open", 60, "1d")
                hist_high = data.history(asset, "high", 60, "1d")
                hist_low = data.history(asset, "low", 60, "1d")
                hist_vol = data.history(asset, "volume", 60, "1d")
            except Exception:
                continue

            ohlcv = pd.DataFrame({
                "open": hist_open.values,
                "high": hist_high.values,
                "low": hist_low.values,
                "close": hist_close.values,
                "volume": hist_vol.values.astype(float),
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
                    m["model"], features,
                    m["scaler_features"], m["scaler_target"],
                )
                if float(pred[0]) > yesterday_close:
                    votes += 1
            model_signal = 1.0 if votes >= 2 else -1.0

            # EMA signal (15%)
            context.ema_prev[ticker] = ema_update(
                context.ema_prev[ticker], today_close, EMA_SPAN,
            )
            ema_signal = 1.0 if today_close > context.ema_prev[ticker] else -1.0

            # McClellan oscillator (15%)
            daily_change = 0.0
            if context.prev_close[ticker] is not None:
                daily_change = today_close - context.prev_close[ticker]
            context.prev_close[ticker] = today_close

            context.mcclel_fast[ticker] = ema_update(
                context.mcclel_fast[ticker], daily_change, 19,
            )
            context.mcclel_slow[ticker] = ema_update(
                context.mcclel_slow[ticker], daily_change, 39,
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

            prefix = ticker.replace("=", "").replace("F", "")
            record(**{
                f"{prefix}_price": today_close,
                f"{prefix}_signal": combined,
            })

    return initialize, handle_data


# ============================================================================
# HTML Report Generation
# ============================================================================

def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_html_report(perf: pd.DataFrame, summary: dict, output_dir: Path):
    """Generate a standalone HTML report with embedded charts and tables."""
    # --- Equity curve chart ---
    fig_eq, ax_eq = plt.subplots(figsize=(14, 5))
    perf["portfolio_value"].plot(ax=ax_eq, color="#2563eb", linewidth=1.5)
    ax_eq.set_title("Equity Curve", fontsize=14, fontweight="bold")
    ax_eq.set_ylabel("Portfolio Value ($)")
    ax_eq.set_xlabel("")
    ax_eq.grid(True, alpha=0.3)
    ax_eq.axhline(y=CAPITAL, color="#94a3b8", linestyle="--", alpha=0.6, label="Initial Capital")
    ax_eq.legend()
    fig_eq.tight_layout()
    equity_b64 = fig_to_base64(fig_eq)
    fig_eq.savefig(output_dir / "equity_curve.png", dpi=150)
    plt.close(fig_eq)

    # --- Drawdown chart ---
    cummax = perf["portfolio_value"].cummax()
    drawdown = (perf["portfolio_value"] - cummax) / cummax * 100
    fig_dd, ax_dd = plt.subplots(figsize=(14, 4))
    ax_dd.fill_between(drawdown.index, drawdown.values, 0, color="#ef4444", alpha=0.3)
    drawdown.plot(ax=ax_dd, color="#dc2626", linewidth=1)
    ax_dd.set_title("Drawdown", fontsize=14, fontweight="bold")
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.set_xlabel("")
    ax_dd.grid(True, alpha=0.3)
    fig_dd.tight_layout()
    drawdown_b64 = fig_to_base64(fig_dd)
    fig_dd.savefig(output_dir / "drawdown.png", dpi=150)
    plt.close(fig_dd)

    # --- Trade log ---
    all_txns = []
    for dt, txn_list in perf["transactions"].items():
        for txn in txn_list:
            row = dict(txn)
            row["date"] = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)
            all_txns.append(row)

    trade_rows_html = ""
    if all_txns:
        trades_df = pd.DataFrame(all_txns)
        trades_df.to_csv(output_dir / "trade_log.csv", index=False)
        # Show last 50 trades in the report
        display_trades = trades_df.tail(50)
        for _, row in display_trades.iterrows():
            sid_val = row.get("sid", "")
            # Extract symbol name from sid if it's a zipline Equity object
            symbol_str = str(sid_val)
            if hasattr(sid_val, "symbol"):
                symbol_str = sid_val.symbol
            trade_rows_html += (
                f"<tr><td>{row.get('date', '')}</td>"
                f"<td>{symbol_str}</td>"
                f"<td>{row.get('amount', '')}</td>"
                f"<td>${row.get('price', 0):.2f}</td>"
                f"<td>${row.get('cost', 0):,.2f}</td></tr>\n"
            )
        total_trades = len(trades_df)
    else:
        total_trades = 0

    # --- Metrics table rows ---
    metrics_rows = ""
    metric_labels = {
        "start_date": "Start Date",
        "end_date": "End Date",
        "trading_days": "Trading Days",
        "initial_capital": "Initial Capital",
        "final_value": "Final Value",
        "total_return_pct": "Total Return (%)",
        "annualized_return_pct": "Annualized Return (%)",
        "sharpe_ratio": "Sharpe Ratio",
        "max_drawdown_pct": "Max Drawdown (%)",
        "total_transactions": "Total Transactions",
    }
    for key, label in metric_labels.items():
        val = summary.get(key, "N/A")
        if key in ("initial_capital", "final_value"):
            val = f"${val:,.2f}" if isinstance(val, (int, float)) else val
        metrics_rows += f"<tr><td>{label}</td><td><strong>{val}</strong></td></tr>\n"

    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ML Momentum Backtest Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #f8fafc; color: #1e293b; line-height: 1.6; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 28px; color: #0f172a; margin-bottom: 8px; }}
  h2 {{ font-size: 20px; color: #334155; margin: 32px 0 16px; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; }}
  .subtitle {{ color: #64748b; font-size: 14px; margin-bottom: 24px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 32px; }}
  .kpi {{ background: #fff; border-radius: 12px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
          border-left: 4px solid #2563eb; }}
  .kpi-label {{ font-size: 13px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }}
  .kpi-value {{ font-size: 28px; font-weight: 700; color: #0f172a; margin-top: 4px; }}
  .chart {{ background: #fff; border-radius: 12px; padding: 16px; margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }}
  .chart img {{ max-width: 100%; height: auto; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 12px;
           overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  th {{ background: #1e293b; color: #fff; padding: 12px 16px; text-align: left; font-size: 13px;
       text-transform: uppercase; letter-spacing: 0.5px; }}
  td {{ padding: 10px 16px; border-bottom: 1px solid #e2e8f0; font-size: 14px; }}
  tr:hover td {{ background: #f1f5f9; }}
  .footer {{ text-align: center; color: #94a3b8; font-size: 12px; margin-top: 40px; padding-top: 16px;
             border-top: 1px solid #e2e8f0; }}
  .overview {{ background: #fff; border-radius: 12px; padding: 20px; margin-bottom: 24px;
               box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .overview-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }}
  .overview-item {{ font-size: 14px; }}
  .overview-item span {{ font-weight: 600; }}
</style>
</head>
<body>
<div class="container">

<h1>ML Momentum Backtest Report</h1>
<p class="subtitle">Generated {generated_at} | SageMaker Processing Job</p>

<h2>Strategy Overview</h2>
<div class="overview">
  <div class="overview-grid">
    <div class="overview-item">Tickers: <span>CL=F, GC=F</span></div>
    <div class="overview-item">Models: <span>LSTM, Transformer, BiLSTM-Attention</span></div>
    <div class="overview-item">Signal Weights: <span>70% Model Vote, 15% EMA, 15% McClellan</span></div>
    <div class="overview-item">Period: <span>{summary.get('start_date', 'N/A')} to {summary.get('end_date', 'N/A')}</span></div>
    <div class="overview-item">Initial Capital: <span>${CAPITAL:,.0f}</span></div>
    <div class="overview-item">Position Size: <span>100 shares per signal</span></div>
  </div>
</div>

<h2>Key Metrics</h2>
<div class="kpi-grid">
  <div class="kpi">
    <div class="kpi-label">Final Portfolio Value</div>
    <div class="kpi-value">${summary.get('final_value', 0):,.2f}</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Total Return</div>
    <div class="kpi-value">{summary.get('total_return_pct', 0):.2f}%</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Sharpe Ratio</div>
    <div class="kpi-value">{summary.get('sharpe_ratio', 0):.4f}</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Max Drawdown</div>
    <div class="kpi-value">{summary.get('max_drawdown_pct', 0):.2f}%</div>
  </div>
</div>

<h2>Performance Metrics</h2>
<table>
{metrics_rows}
</table>

<h2>Equity Curve</h2>
<div class="chart">
  <img src="data:image/png;base64,{equity_b64}" alt="Equity Curve">
</div>

<h2>Drawdown</h2>
<div class="chart">
  <img src="data:image/png;base64,{drawdown_b64}" alt="Drawdown Chart">
</div>

<h2>Trade Log (Last {min(total_trades, 50)} of {total_trades} Trades)</h2>
<table>
  <tr><th>Date</th><th>Symbol</th><th>Amount</th><th>Price</th><th>Cost</th></tr>
  {trade_rows_html}
</table>

<div class="footer">
  AAI-540 Group 8 | ML Momentum Trading Strategy | Powered by zipline-reloaded
</div>

</div>
</body>
</html>"""

    report_path = output_dir / "report.html"
    report_path.write_text(html)
    logger.info(f"HTML report saved to {report_path}")


# ============================================================================
# Main Processing Pipeline
# ============================================================================

def verify_inputs():
    """Check that all expected input files exist."""
    missing = []
    for ticker in TICKERS:
        for mname in MODELS:
            pth = INPUT_MODELS / ticker / f"{mname}_{ticker}.pth"
            feat = INPUT_MODELS / ticker / f"{mname}_{ticker}_feature_scaler.pkl"
            tgt = INPUT_MODELS / ticker / f"{mname}_{ticker}_target_scaler.pkl"
            for f in [pth, feat, tgt]:
                if not f.exists():
                    missing.append(str(f))
        csv_path = INPUT_DATA / f"{ticker}x_prod.csv"
        if not csv_path.exists():
            missing.append(str(csv_path))

    if missing:
        logger.error("Missing input files:\n  " + "\n  ".join(missing))
        sys.exit(1)

    logger.info("All input files verified.")


def main():
    logger.info("=" * 60)
    logger.info("SageMaker Processing: ML Momentum Backtest")
    logger.info("=" * 60)

    # 1. Verify inputs
    verify_inputs()

    # 2. Prepare zipline bundle data
    logger.info("Preparing zipline bundle data...")
    prepare_bundle_data(INPUT_DATA, BUNDLE_DIR)

    # 3. Setup zipline extension
    setup_zipline_extension(WORK_DIR)

    # 4. Ingest bundle
    logger.info("Ingesting zipline bundle...")
    result = subprocess.run(
        [sys.executable, "-m", "zipline", "ingest", "-b", "futures-prod-bundle"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.error(f"Bundle ingest failed:\n{result.stderr}")
        sys.exit(1)
    logger.info("Bundle ingested successfully.")

    # 5. Run backtest
    logger.info("Running zipline backtest...")
    from zipline import run_algorithm

    config = DEFAULT_CONFIG
    init_fn, handle_fn = make_algo(config, INPUT_MODELS)

    perf = run_algorithm(
        start=START_DATE,
        end=END_DATE,
        initialize=init_fn,
        handle_data=handle_fn,
        capital_base=CAPITAL,
        bundle="futures-prod-bundle",
    )

    # 6. Compute summary statistics
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_return = (perf["portfolio_value"].iloc[-1] / CAPITAL) - 1
    days = (perf.index[-1] - perf.index[0]).days
    annual_return = (1 + total_return) ** (365.25 / max(days, 1)) - 1

    daily_returns = perf["portfolio_value"].pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std() * (252 ** 0.5)
              if daily_returns.std() > 0 else 0.0)

    cummax = perf["portfolio_value"].cummax()
    max_drawdown = ((perf["portfolio_value"] - cummax) / cummax).min()

    summary = {
        "start_date": str(perf.index[0].date()),
        "end_date": str(perf.index[-1].date()),
        "trading_days": len(perf),
        "initial_capital": CAPITAL,
        "final_value": round(float(perf["portfolio_value"].iloc[-1]), 2),
        "total_return_pct": round(total_return * 100, 2),
        "annualized_return_pct": round(annual_return * 100, 2),
        "sharpe_ratio": round(float(sharpe), 4),
        "max_drawdown_pct": round(float(max_drawdown) * 100, 2),
        "total_transactions": int(perf["transactions"].apply(len).sum()),
    }

    # 7. Save outputs
    perf.to_csv(OUTPUT_DIR / "performance.csv")
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 8. Generate HTML report
    generate_html_report(perf, summary, OUTPUT_DIR)

    logger.info("=" * 60)
    logger.info("BACKTEST COMPLETE")
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
