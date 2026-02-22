"""
Backtesting pipeline for ML momentum trading algorithm.

Runs the momentum_algo against production (unseen) data using zipline-reloaded.

Usage:
    python -m scripts.run_backtest                    # Run backtest only
    python -m scripts.run_backtest --prepare-data     # Also prepare bundle data
    python -m scripts.run_backtest --ingest           # Also re-ingest bundle
"""
import argparse
import json
import subprocess
import sys
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
# Patch numpy 2.x removals used by empyrical
if not hasattr(np, 'NINF'):
    np.NINF = -np.inf
if not hasattr(np, 'PINF'):
    np.PINF = np.inf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from zipline import run_algorithm
from src.trader.momentum_algo import initialize, handle_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('backtest')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_NAME = 'futures-prod-bundle'
START_DATE = pd.Timestamp('2019-09-03')
END_DATE = pd.Timestamp('2026-02-05')
CAPITAL = 10_000.0


def prepare_data():
    """Run the data preparation script."""
    logger.info("Preparing backtest data from prod CSVs...")
    from scripts.prepare_backtest_data import main as prep_main
    prep_main()


def ingest_bundle():
    """Ingest the zipline bundle."""
    logger.info(f"Ingesting bundle '{BUNDLE_NAME}'...")
    venv_zipline = PROJECT_ROOT / '.venv' / 'bin' / 'zipline'
    cmd = [str(venv_zipline), 'ingest', '-b', BUNDLE_NAME]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Ingest failed:\n{result.stderr}")
        sys.exit(1)
    logger.info("Bundle ingested successfully.")


def save_results(perf: pd.DataFrame, results_dir: Path):
    """Save backtest results: performance CSV, equity curve, summary, trades."""
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Full performance DataFrame
    perf_path = results_dir / 'performance.csv'
    perf.to_csv(perf_path)
    logger.info(f"Saved performance to {perf_path}")

    # 2. Equity curve
    fig, ax = plt.subplots(figsize=(14, 6))
    perf['portfolio_value'].plot(ax=ax, title='Portfolio Value Over Time')
    ax.set_ylabel('Portfolio Value ($)')
    ax.set_xlabel('Date')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = results_dir / 'equity_curve.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved equity curve to {fig_path}")

    # 3. Summary statistics
    total_return = (perf['portfolio_value'].iloc[-1] / CAPITAL) - 1
    days = (perf.index[-1] - perf.index[0]).days
    annual_return = (1 + total_return) ** (365.25 / max(days, 1)) - 1

    daily_returns = perf['portfolio_value'].pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std() * (252 ** 0.5)
              if daily_returns.std() > 0 else 0.0)

    cummax = perf['portfolio_value'].cummax()
    drawdown = (perf['portfolio_value'] - cummax) / cummax
    max_drawdown = drawdown.min()

    summary = {
        'start_date': str(perf.index[0].date()),
        'end_date': str(perf.index[-1].date()),
        'trading_days': len(perf),
        'initial_capital': CAPITAL,
        'final_value': round(float(perf['portfolio_value'].iloc[-1]), 2),
        'total_return_pct': round(total_return * 100, 2),
        'annualized_return_pct': round(annual_return * 100, 2),
        'sharpe_ratio': round(float(sharpe), 4),
        'max_drawdown_pct': round(float(max_drawdown) * 100, 2),
        'total_transactions': int(perf['transactions'].apply(len).sum()),
    }

    summary_path = results_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_path}")

    # 4. Trade log
    all_txns = []
    for dt, txn_list in perf['transactions'].items():
        for txn in txn_list:
            row = dict(txn)
            row['date'] = dt
            all_txns.append(row)

    if all_txns:
        trades_df = pd.DataFrame(all_txns)
        trades_path = results_dir / 'trade_log.csv'
        trades_df.to_csv(trades_path, index=False)
        logger.info(f"Saved {len(trades_df)} transactions to {trades_path}")
    else:
        logger.info("No transactions recorded.")

    # Print summary
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Run ML momentum backtest')
    parser.add_argument('--prepare-data', action='store_true',
                        help='Prepare bundle data from prod CSVs')
    parser.add_argument('--ingest', action='store_true',
                        help='Re-ingest the zipline bundle')
    args = parser.parse_args()

    if args.prepare_data:
        prepare_data()

    if args.ingest:
        ingest_bundle()

    logger.info(f"Running backtest: {START_DATE.date()} to {END_DATE.date()}, "
                f"capital=${CAPITAL:,.0f}, bundle='{BUNDLE_NAME}'")

    perf = run_algorithm(
        start=START_DATE,
        end=END_DATE,
        initialize=initialize,
        handle_data=handle_data,
        capital_base=CAPITAL,
        bundle=BUNDLE_NAME,
    )

    results_dir = PROJECT_ROOT / 'files' / 'backtest' / 'results'
    save_results(perf, results_dir)


if __name__ == '__main__':
    main()
