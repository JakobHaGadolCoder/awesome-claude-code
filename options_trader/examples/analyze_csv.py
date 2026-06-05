"""
Point-in-time analysis from real MT5 CSV data.

Loads native M15/H1/H4/D1 CSVs (as written by examples/mt5_export_csv.py) and
runs the full pipeline + trade-param math, printing the same report used for
the chart screenshots — but on real data, with a real multi-timeframe gate.

Usage:
    # directory containing XAUUSD_M15.csv, XAUUSD_H1.csv, XAUUSD_H4.csv, XAUUSD_D1.csv
    python -m options_trader.examples.analyze_csv --dir ./data --symbol XAUUSD

    # or pass files explicitly
    python -m options_trader.examples.analyze_csv \
        --m15 XAUUSD_M15.csv --h1 XAUUSD_H1.csv --h4 XAUUSD_H4.csv --d1 XAUUSD_D1.csv

Add --range-filter block_all|reversal_only to apply the regime stand-aside.
"""
from __future__ import annotations

import argparse

from options_trader.core.config import TradingConfig
from options_trader.reporting import analyze_and_print
from options_trader.utils.csv_loader import discover_mtf, load_mtf_csvs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", help="directory with <SYMBOL>_<TF>.csv files")
    ap.add_argument("--symbol", default="XAUUSD")
    for tf in ("m15", "h1", "h4", "d1", "w1"):
        ap.add_argument(f"--{tf}", help=f"explicit {tf.upper()} csv path")
    ap.add_argument("--range-filter", default="off",
                    choices=["off", "block_all", "reversal_only"])
    args = ap.parse_args()

    if args.dir:
        frames = discover_mtf(args.dir, args.symbol)
    else:
        paths = {tf.upper(): getattr(args, tf)
                 for tf in ("m15", "h1", "h4", "d1", "w1") if getattr(args, tf)}
        if "M15" not in paths:
            ap.error("provide --dir, or at least --m15 (plus optional --h1/--h4/--d1)")
        frames = load_mtf_csvs(paths)

    for tf, df in frames.items():
        print(f"  loaded {tf}: {len(df)} bars  {df.index[0]} -> {df.index[-1]}")

    cfg = TradingConfig()
    cfg.range_filter_mode = args.range_filter
    analyze_and_print(args.symbol, frames, config=cfg)


if __name__ == "__main__":
    main()
