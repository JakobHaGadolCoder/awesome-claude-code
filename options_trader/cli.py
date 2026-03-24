"""
CLI Entry Point
Usage:
  python -m options_trader scan                  # scan all configured symbols
  python -m options_trader analyze SPY           # analyze a single symbol
  python -m options_trader backtest SPY 252      # backtest with N days of history
  python -m options_trader greeks SPY --strike 500 --dte 30 --iv 0.20 --type call
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

from options_trader.core.config import TradingConfig
from options_trader.engine import TradingEngine


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Options Algo Trading Bot — Order Flow, Indicators, S/R, Events"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- scan ---
    scan_p = sub.add_parser("scan", help="Scan all configured symbols")
    scan_p.add_argument("--symbols", nargs="+", default=None, help="Override symbol list")
    scan_p.add_argument("--capital", type=float, default=100_000, help="Paper trading capital")

    # --- analyze ---
    ana_p = sub.add_parser("analyze", help="Analyze a single symbol")
    ana_p.add_argument("symbol", type=str)
    ana_p.add_argument("--capital", type=float, default=100_000)

    # --- backtest ---
    bt_p = sub.add_parser("backtest", help="Run backtest on historical data")
    bt_p.add_argument("symbol", type=str)
    bt_p.add_argument("days", type=int, nargs="?", default=252, help="Days of history (default 252)")
    bt_p.add_argument("--capital", type=float, default=100_000)

    # --- greeks ---
    gr_p = sub.add_parser("greeks", help="Compute Black-Scholes Greeks for a contract")
    gr_p.add_argument("symbol", type=str, help="Underlying ticker")
    gr_p.add_argument("--price", type=float, required=True, help="Underlying price")
    gr_p.add_argument("--strike", type=float, required=True, help="Strike price")
    gr_p.add_argument("--dte", type=int, required=True, help="Days to expiration")
    gr_p.add_argument("--iv", type=float, required=True, help="Implied volatility (e.g. 0.20)")
    gr_p.add_argument("--type", choices=["call", "put"], default="call")
    gr_p.add_argument("--rate", type=float, default=0.05, help="Risk-free rate")

    return parser.parse_args(argv)


def cmd_scan(args):
    config = TradingConfig(paper_trading_capital=args.capital)
    if args.symbols:
        config.symbols = args.symbols
    engine = TradingEngine(config)
    results = engine.scan_universe()
    signals = {sym: sig for sym, sig in results.items() if sig is not None}
    if not signals:
        print("\nNo valid trade signals found in current market conditions.")
    else:
        print(f"\nFound {len(signals)} trade signal(s):")
        for sym in signals:
            print(f"  • {sym}: {signals[sym].option_type.value} {signals[sym].strike} "
                  f"exp={signals[sym].expiration.date()} conf={signals[sym].confidence*100:.0f}%")


def cmd_analyze(args):
    config = TradingConfig(paper_trading_capital=args.capital)
    engine = TradingEngine(config)
    engine.analyze_symbol(args.symbol)


def cmd_backtest(args):
    config = TradingConfig(paper_trading_capital=args.capital)
    engine = TradingEngine(config)
    engine.backtest(args.symbol, days=args.days)


def cmd_greeks(args):
    from options_trader.analyzers.options_greeks import BlackScholes
    from options_trader.core.models import OptionType

    opt_type = OptionType.CALL if args.type == "call" else OptionType.PUT
    T = args.dte / 365.0
    result = BlackScholes.greeks(args.price, args.strike, T, args.rate, args.iv, opt_type)

    print(f"\nBlack-Scholes Greeks: {args.symbol} {opt_type.value} ${args.strike} {args.dte}DTE IV={args.iv*100:.0f}%")
    print("-" * 55)
    print(f"  Theoretical Price: ${result.price:.4f}")
    print(f"  Intrinsic Value:   ${result.intrinsic_value:.4f}")
    print(f"  Time Value:        ${result.time_value:.4f}")
    print(f"  Delta:             {result.delta:+.4f}")
    print(f"  Gamma:             {result.gamma:.6f}")
    print(f"  Theta:             {result.theta:+.4f}  (per day)")
    print(f"  Vega:              {result.vega:.4f}   (per 1% IV)")
    print(f"  Rho:               {result.rho:+.4f}   (per 1% rate)")
    print(f"  Vanna:             {result.vanna:+.6f}")
    print(f"  Charm:             {result.charm:+.6f}")
    print()


def main(argv=None):
    args = parse_args(argv)
    dispatch = {
        "scan": cmd_scan,
        "analyze": cmd_analyze,
        "backtest": cmd_backtest,
        "greeks": cmd_greeks,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
