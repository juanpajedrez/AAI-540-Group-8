"""
Prepare clean backtest data from production (unseen) CSVs.

Reads prod data (2019-09-03 to 2026-02-05) and converts to zipline csvdir
format, filtered to NYSE trading days only.

Usage:
    python -m scripts.prepare_backtest_data
"""
import pandas as pd
from pathlib import Path
from pandas.tseries.offsets import CustomBusinessDay
from exchange_calendars import get_calendar


def get_nyse_trading_days(start: str, end: str) -> pd.DatetimeIndex:
    """Return NYSE trading days between start and end (inclusive)."""
    cal = get_calendar('XNYS')
    sessions = cal.sessions_in_range(start, end)
    return sessions


def convert_prod_csv_to_zipline(input_path: Path, output_path: Path,
                                nyse_days: pd.DatetimeIndex) -> None:
    """Convert a prod CSV to zipline csvdir equity format.

    Input format:  index, Date, adj close, close, high, low, open, repaired?, volume
    Output format: Date, open, high, low, close, volume, dividend, split
    """
    df = pd.read_csv(input_path)

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()

    # Filter to NYSE trading days only
    df = df[df.index.isin(nyse_days)]

    # Zipline stores OHLC as uint32 (price * 1000), so negative prices
    # (e.g., CL=F on 2020-04-20) must be floored at 0.01.
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].clip(lower=0.01)

    # Build zipline-format DataFrame
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"  Wrote {len(out)} rows to {output_path}")


def main():
    project_root = Path(__file__).resolve().parent.parent
    prod_dir = project_root / 'files' / 'dataset' / 'prod'
    output_dir = project_root / 'files' / 'backtest' / 'prod_daily'

    tickers = {
        'CL=F': prod_dir / 'CL=Fx_prod.csv',
        'GC=F': prod_dir / 'GC=Fx_prod.csv',
    }

    # Get NYSE trading calendar for full prod range
    nyse_days = get_nyse_trading_days('2019-09-03', '2026-02-05')
    print(f"NYSE trading days in range: {len(nyse_days)}")

    for ticker, input_path in tickers.items():
        print(f"Processing {ticker}...")
        output_path = output_dir / f"{ticker}.csv"
        convert_prod_csv_to_zipline(input_path, output_path, nyse_days)

    print("Done. Output directory:", output_dir)


if __name__ == '__main__':
    main()
