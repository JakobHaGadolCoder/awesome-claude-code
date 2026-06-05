"""
Export OHLCV bars from a running MetaTrader 5 terminal to CSV.

RUN THIS ON WINDOWS (the MetaTrader5 Python package is Windows-only; Linux+Wine
also works). The MT5 *Android* app cannot export history — use the desktop
terminal. The terminal must be installed, logged in, and running.

Setup (once):
    pip install MetaTrader5 pandas

Usage:
    # default: XAUUSD, M15/H1/H4/D1, 5000 bars each, into ./data
    python mt5_export_csv.py

    # custom symbol / timeframes / depth / output dir
    python mt5_export_csv.py --symbol XAUUSD --timeframes M15,H1,H4,D1 --bars 8000 --out ./data

Writes one file per timeframe named '<SYMBOL>_<TF>.csv' with columns:
    time,open,high,low,close,volume
where `time` is ISO-8601 UTC and `volume` is MT5 tick volume. This is exactly
the format options_trader.utils.csv_loader (and examples/analyze_csv.py) expect.
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--timeframes", default="M15,H1,H4,D1")
    ap.add_argument("--bars", type=int, default=5000)
    ap.add_argument("--out", default="./data")
    # optional explicit login (usually unnecessary if the terminal is already
    # logged in and running)
    ap.add_argument("--login", type=int)
    ap.add_argument("--password")
    ap.add_argument("--server")
    args = ap.parse_args()

    try:
        import MetaTrader5 as mt5
    except ImportError:
        sys.exit("MetaTrader5 not installed. Run: pip install MetaTrader5  (Windows only)")
    import pandas as pd

    tf_map = {
        "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1,
    }
    timeframes = [t.strip().upper() for t in args.timeframes.split(",") if t.strip()]
    bad = [t for t in timeframes if t not in tf_map]
    if bad:
        sys.exit(f"Unknown timeframes: {bad}. Choose from {list(tf_map)}")

    init_kwargs = {}
    if args.login:
        init_kwargs = dict(login=args.login, password=args.password, server=args.server)
    if not mt5.initialize(**init_kwargs):
        sys.exit(f"mt5.initialize() failed: {mt5.last_error()}")

    os.makedirs(args.out, exist_ok=True)
    try:
        if not mt5.symbol_select(args.symbol, True):
            sys.exit(f"symbol_select({args.symbol}) failed: {mt5.last_error()}")

        for tf in timeframes:
            rates = mt5.copy_rates_from_pos(args.symbol, tf_map[tf], 0, args.bars)
            if rates is None or len(rates) == 0:
                print(f"  WARN {tf}: no data ({mt5.last_error()})")
                continue
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df.rename(columns={"tick_volume": "volume"})
            df = df[["time", "open", "high", "low", "close", "volume"]]
            path = os.path.join(args.out, f"{args.symbol}_{tf}.csv")
            df.to_csv(path, index=False)
            print(f"  wrote {path}  ({len(df)} bars  {df['time'].iloc[0]} -> {df['time'].iloc[-1]})")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
