"""
Backtest the CFD strategy on a real MT5 M15 CSV.

Walk-forward replays an M15 history (exported via mt5_export_csv.py) through the
full pipeline; higher timeframes for the MTF gate are resampled on the fly from
the M15 window (no look-ahead). Prints R-multiple stats for each filter config.

Usage:
    python -m options_trader.examples.backtest_csv --m15 ./data/XAUUSD_M15.csv
    python -m options_trader.examples.backtest_csv --dir ./data --symbol XAUUSD
"""
from __future__ import annotations

import argparse

from options_trader.core.config import TradingConfig
from options_trader.backtesting.cfd_backtester import CFDBacktester, CFDBacktestResult
from options_trader.utils.csv_loader import load_ohlcv_csv, discover_mtf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m15", help="path to the M15 CSV")
    ap.add_argument("--dir", help="directory with <SYMBOL>_M15.csv")
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--warmup", type=int, default=80)
    args = ap.parse_args()

    if args.m15:
        m15 = load_ohlcv_csv(args.m15)
    elif args.dir:
        m15 = discover_mtf(args.dir, args.symbol, timeframes=["M15"])["M15"]
    else:
        ap.error("provide --m15 <path> or --dir <folder>")

    print(f"  loaded M15: {len(m15)} bars  {m15.index[0]} -> {m15.index[-1]}\n")

    configs = [
        ("MR=ON  ",      dict(enable_mean_reversion_overrides=True,  range_filter_mode="off")),
        ("MR=OFF ",      dict(enable_mean_reversion_overrides=False, range_filter_mode="off")),
        ("OFF+block",    dict(enable_mean_reversion_overrides=False, range_filter_mode="block_all")),
        ("OFF+reversal", dict(enable_mean_reversion_overrides=False, range_filter_mode="reversal_only")),
    ]
    print(CFDBacktestResult.header())
    print("-" * 92)
    for tag, kw in configs:
        cfg = TradingConfig()
        for k, v in kw.items():
            setattr(cfg, k, v)
        res = CFDBacktester(cfg).run(m15, label=f"{args.symbol} {tag}",
                                     symbol=args.symbol, warmup=args.warmup)
        print(res.row())


if __name__ == "__main__":
    main()
