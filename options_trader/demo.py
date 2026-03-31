"""
Demo script — run the full pipeline against synthetic/historical data.
No live broker needed; uses yfinance for price data and Black-Scholes for pricing.

Usage:
    python -m options_trader.demo
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from options_trader.core.config import TradingConfig
from options_trader.core.models import (
    MarketEvent,
    OptionContract,
    OptionType,
)
from options_trader.analyzers.order_flow import OrderFlowAnalyzer
from options_trader.analyzers.technical import TechnicalAnalyzer
from options_trader.analyzers.support_resistance import SupportResistanceAnalyzer
from options_trader.analyzers.events import EventsAnalyzer
from options_trader.analyzers.options_greeks import BlackScholes, OptionsGreeksCalculator
from options_trader.analyzers.price_action import PriceActionAnalyzer
from options_trader.analyzers.vwap import VWAPAnalyzer
from options_trader.strategies.signal_aggregator import SignalAggregator
from options_trader.strategies.trade_selector import TradeSelector
from options_trader.utils.logger import setup_logger

setup_logger("options_trader", "INFO")
logger = logging.getLogger("demo")


def make_synthetic_ohlcv(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for demo purposes."""
    rng = np.random.default_rng(seed)
    price = 500.0
    prices = [price]
    for _ in range(n - 1):
        price *= np.exp(rng.normal(0.0003, 0.012))
        prices.append(price)
    prices = np.array(prices)
    highs = prices * (1 + rng.uniform(0, 0.01, n))
    lows = prices * (1 - rng.uniform(0, 0.01, n))
    opens = prices * (1 + rng.normal(0, 0.003, n))
    volumes = rng.integers(5_000_000, 30_000_000, n).astype(float)

    idx = pd.date_range(end=datetime.utcnow().date(), periods=n, freq="B")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": prices, "volume": volumes},
        index=idx,
    )


def make_synthetic_chain(
    symbol: str, current_price: float, iv: float = 0.20
) -> list[OptionContract]:
    """Build a synthetic options chain around current_price."""
    contracts = []
    expirations = [
        datetime.utcnow() + timedelta(days=d) for d in (14, 30, 45)
    ]
    strikes = np.arange(current_price * 0.90, current_price * 1.10, 5)

    for exp in expirations:
        T = max(0.001, (exp - datetime.utcnow()).total_seconds() / (365.25 * 86400))
        for K in strikes:
            for opt_type in (OptionType.CALL, OptionType.PUT):
                price = BlackScholes.price(current_price, K, T, 0.05, iv, opt_type)
                greeks = BlackScholes.greeks(current_price, K, T, 0.05, iv, opt_type)
                volume = max(10, int(abs(greeks.delta) * 2000))
                oi = volume * 5
                contracts.append(
                    OptionContract(
                        symbol=symbol,
                        option_type=opt_type,
                        strike=round(K, 2),
                        expiration=exp,
                        bid=max(0.01, price * 0.97),
                        ask=price * 1.03,
                        last=price,
                        volume=volume,
                        open_interest=oi,
                        implied_volatility=iv,
                        delta=greeks.delta,
                        gamma=greeks.gamma,
                        theta=greeks.theta,
                        vega=greeks.vega,
                        underlying_price=current_price,
                    )
                )
    return contracts


def run_demo():
    print("\n" + "=" * 65)
    print("  OPTIONS ALGO TRADING BOT — DEMO")
    print("=" * 65)

    config = TradingConfig(
        symbols=["SPY"],
        paper_trading_capital=100_000.0,
        min_confidence=0.50,
        log_level="WARNING",  # suppress verbose logs for demo
    )
    config.validate()

    symbol = "SPY"
    ohlcv = make_synthetic_ohlcv(n=120, seed=7)
    current_price = float(ohlcv["close"].iloc[-1])
    print(f"\nSynthetic {symbol} data: {len(ohlcv)} bars, current price ${current_price:.2f}")

    # --- Technical Analysis ---
    print("\n[1] Technical Analysis")
    tech = TechnicalAnalyzer(config)
    signals, regime, score = tech.analyze(symbol, ohlcv)
    print(f"    Regime: {regime.value}  |  Composite score: {score:+.3f}")
    for s in signals:
        print(f"    {s.indicator:20s} → {s.signal.name:12s}  {s.description}")

    # --- Support / Resistance ---
    print("\n[2] Support & Resistance")
    chain = make_synthetic_chain(symbol, current_price, iv=0.20)
    sr = SupportResistanceAnalyzer(config)
    levels, sr_signal = sr.analyze(symbol, ohlcv, current_price, chain)
    print(f"    Detected {len(levels)} levels  |  Signal: {sr_signal.signal.name}")
    print(f"    Nearest: {sr_signal.description}")

    # --- Order Flow ---
    print("\n[3] Order Flow Analysis")
    flow_analyzer = OrderFlowAnalyzer(config)
    flow_data, flow_signal = flow_analyzer.analyze(symbol, chain)
    print(f"    P/C Ratio: {flow_data.put_call_ratio:.2f}")
    print(f"    Net Premium Flow: ${flow_data.net_premium_flow:,.0f}")
    print(f"    Unusual Call: {flow_data.unusual_call_activity}  |  Unusual Put: {flow_data.unusual_put_activity}")
    print(f"    Signal: {flow_signal.signal.name}  —  {flow_signal.description}")

    # --- Events ---
    print("\n[4] Events & News Analysis")
    ev_analyzer = EventsAnalyzer(config)
    earnings = MarketEvent(
        name=f"{symbol} Quarterly Earnings",
        event_type="earnings",
        scheduled_date=datetime.utcnow() + timedelta(days=10),
        importance="high",
        expected_impact=0.05,
        symbol=symbol,
    )
    news_headlines = [
        {"headline": "SPY rallies on strong economic data, bulls take charge", "sentiment": 0.6, "date": datetime.utcnow()},
        {"headline": "Fed signals patience on rate cuts", "sentiment": -0.1, "date": datetime.utcnow() - timedelta(hours=4)},
    ]
    upcoming, ev_signal = ev_analyzer.analyze(symbol, events=[earnings], news_headlines=news_headlines)
    print(f"    Upcoming events: {[e.name for e in upcoming]}")
    print(f"    Signal: {ev_signal.signal.name}  —  {ev_signal.description}")

    # --- Price Action Analysis ---
    print("\n[5] Price Action Analysis")
    pa_analyzer = PriceActionAnalyzer(config)
    pa_result = pa_analyzer.analyze(symbol, ohlcv, current_price)
    struct = pa_result.structure
    print(f"    Market Structure: {struct.trend.upper()}"
          + (" HH+HL" if struct.higher_highs and struct.higher_lows else "")
          + (" LH+LL" if struct.lower_highs and struct.lower_lows else ""))
    if struct.last_bos:
        print(f"    Break of Structure (BOS): {struct.last_bos:.2f}")
    if struct.last_choch:
        print(f"    Change of Character (CHoCH): {struct.last_choch:.2f}  ⚠ structure flipping")
    active_obs = [ob for ob in pa_result.order_blocks if not ob.mitigated]
    active_fvgs = [f for f in pa_result.fvgs if not f.filled]
    print(f"    Active Order Blocks: {len(active_obs)}")
    for ob in active_obs[-3:]:
        print(f"      {ob.direction.upper():8s} OB  {ob.price_low:.2f}–{ob.price_high:.2f}  {ob.description}")
    print(f"    Active FVGs: {len(active_fvgs)}")
    for fvg in active_fvgs[-3:]:
        print(f"      {fvg.direction.upper():8s} FVG  {fvg.price_low:.2f}–{fvg.price_high:.2f}  (size={fvg.size:.2f})")
    recent_pats = pa_result.patterns[-5:]
    if recent_pats:
        print(f"    Recent Candle Patterns:")
        for p in recent_pats:
            print(f"      [{p.direction.upper():7s}] {p.name:22s}  strength={p.strength:.2f}  — {p.description}")
    print(f"    PA Signal: {pa_result.signal.signal.name}  —  {pa_result.signal.description}")

    # --- VWAP Analysis ---
    print("\n[6] VWAP Analysis")
    vwap_analyzer = VWAPAnalyzer(config)
    vwap_bands, vwap_ctx, vwap_signal = vwap_analyzer.analyze(symbol, ohlcv, session_reset=True)
    dev_sign = "+" if vwap_ctx.deviation_pct >= 0 else ""
    print(f"    VWAP:       {vwap_bands.vwap:.2f}  (price {dev_sign}{vwap_ctx.deviation_pct*100:.2f}% from VWAP)")
    print(f"    +1σ / -1σ:  {vwap_bands.upper_1:.2f}  /  {vwap_bands.lower_1:.2f}")
    print(f"    +2σ / -2σ:  {vwap_bands.upper_2:.2f}  /  {vwap_bands.lower_2:.2f}")
    print(f"    Band pos:   {vwap_ctx.band_position}")
    print(f"    VWAP trend: {vwap_ctx.vwap_trend}  (slope={vwap_bands.slope:+.3f})")
    if vwap_ctx.is_reclaim:
        print(f"    *** VWAP RECLAIM DETECTED — bullish high-conviction event ***")
    if vwap_ctx.is_rejection:
        print(f"    *** VWAP REJECTION DETECTED — bearish high-conviction event ***")
    if vwap_ctx.is_extended:
        print(f"    ⚠ Extended {abs(vwap_ctx.deviation_pct)*100:.1f}% from VWAP — mean reversion candidate")
    print(f"    VWAP Signal: {vwap_signal.signal.name}  —  {vwap_signal.description}")

    # --- Greeks Demo ---
    print("\n[7] Black-Scholes Greeks")
    calc = OptionsGreeksCalculator()
    atm_call = next(
        (c for c in chain if c.option_type == OptionType.CALL and abs(c.strike - current_price) < 10 and (c.expiration - datetime.utcnow()).days > 25),
        chain[0]
    )
    g = calc.compute_greeks(atm_call)
    print(f"    {symbol} CALL ${atm_call.strike:.0f} {atm_call.expiration.date()}")
    print(f"    Price=${g.price:.2f}  Delta={g.delta:+.3f}  Gamma={g.gamma:.5f}  Theta={g.theta:+.4f}/day  Vega={g.vega:.4f}/1%IV")

    # --- Signal Aggregation ---
    print("\n[8] Signal Aggregation (All Sources)")
    aggregator = SignalAggregator(config)
    agg = aggregator.aggregate(
        symbol=symbol,
        order_flow_signal=flow_signal,
        technical_score=score,
        sr_signal=sr_signal,
        event_signal=ev_signal,
        regime=regime,
        price_action_signal=pa_result.signal,
        vwap_signal=vwap_signal,
        additional_signals=signals,
    )
    print(f"    Composite Score:  {agg.composite_score:+.3f}")
    print(f"    Direction:        {agg.direction.name}")
    print(f"    Confidence:       {agg.confidence*100:.0f}%")
    print(f"    Regime:           {agg.regime.value}")
    print(f"    Signal Sources:")
    for k, v in agg.component_signals.items():
        print(f"      {k:15s} → {v.signal.name:12s}  {v.description[:60]}")

    # --- Trade Selection ---
    print("\n[9] Trade Selection")
    selector = TradeSelector(config)
    trade = selector.select_trade(agg, chain, config.paper_trading_capital)
    if trade:
        dte = (trade.expiration - datetime.utcnow()).days
        print(f"    ACTION:    {trade.action} {trade.option_type.value}")
        print(f"    Strike:    ${trade.strike:.0f}  ({dte} DTE)")
        print(f"    Entry:     ${trade.entry_price:.2f}")
        print(f"    Target:    ${trade.target_price:.2f}  (+{(trade.target_price/trade.entry_price-1)*100:.0f}%)")
        print(f"    Stop:      ${trade.stop_loss:.2f}  (-{(1-trade.stop_loss/trade.entry_price)*100:.0f}%)")
        print(f"    R/R:       {trade.risk_reward:.1f}x")
        print(f"    Confidence:{trade.confidence*100:.0f}%")
        print(f"    Strength:  {trade.strength.name}")
    else:
        print("    No valid trade signal (thresholds not met)")

    # --- Backtest (quick) ---
    print("\n[10] Backtesting (synthetic 120-day run)")
    from options_trader.backtesting.backtester import Backtester
    bt_config = TradingConfig(paper_trading_capital=100_000, log_level="WARNING")
    backtester = Backtester(bt_config)
    iv_series = pd.Series(0.20, index=ohlcv.index)
    result = backtester.run(symbol, ohlcv, historical_iv=iv_series, warmup_bars=60)
    print(result.summary())

    print("=" * 65)
    print("  Demo complete.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_demo()
