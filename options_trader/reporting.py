"""
Point-in-time analysis report.

Runs the full analysis pipeline on a {timeframe: OHLCV} mapping and prints the
same human-readable read used for the chart screenshots — driven by either
synthetic reconstruction or real MT5 CSV data. Returns the AggregatedSignal.
"""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from options_trader.core.config import TradingConfig
from options_trader.core.models import SignalStrength, TechnicalSignal
from options_trader.analyzers.technical import TechnicalAnalyzer
from options_trader.analyzers.support_resistance import SupportResistanceAnalyzer
from options_trader.analyzers.events import EventsAnalyzer
from options_trader.analyzers.price_action import PriceActionAnalyzer
from options_trader.analyzers.vwap import VWAPAnalyzer
from options_trader.analyzers.multi_timeframe import MultiTimeframeAnalyzer
from options_trader.analyzers.divergence import DivergenceDetector
from options_trader.strategies.signal_aggregator import SignalAggregator, AggregatedSignal
from options_trader.strategies.regime_filter import range_stand_aside


def atr14(df: pd.DataFrame) -> float:
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    if len(df) < 2:
        return 0.0
    trs = [max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])) for i in range(1, len(df))]
    k = min(14, len(trs))
    return float(sum(trs[-k:]) / k)


def _late_entry_mult(df, is_buy, atr, lookback=10, full=2.5, mn=1.2):
    d = df.tail(lookback)
    if len(d) < 3 or atr == 0:
        return full, 0.0
    entry = float(d["close"].iloc[-1])
    move = (entry - float(d["low"].min())) if is_buy else (float(d["high"].max()) - entry)
    r = max(0.0, move) / atr
    if r <= 1.5:
        return full, r
    return round(max(mn, full - min(0.5, (r - 1.5) * 0.15) * full), 1), r


def _cap_tp(entry, tp, is_buy, levels, buf=0.001):
    if is_buy:
        block = [lv for lv in levels if lv.level_type == "resistance"
                 and entry < lv.price < tp and lv.strength >= 0.75]
        if block:
            return round(min(block, key=lambda x: x.price).price * (1 - buf), 2), \
                   min(block, key=lambda x: x.price)
    else:
        block = [lv for lv in levels if lv.level_type == "support"
                 and tp < lv.price < entry and lv.strength >= 0.75]
        if block:
            return round(max(block, key=lambda x: x.price).price * (1 + buf), 2), \
                   max(block, key=lambda x: x.price)
    return tp, None


def _params(df, is_buy, entry, levels, cfg):
    atr = atr14(df)
    tp_mult, ratio = _late_entry_mult(df, is_buy, atr)
    if is_buy:
        sl = round(entry - 1.5 * atr, 2); tp1 = round(entry + tp_mult * atr, 2)
    else:
        sl = round(entry + 1.5 * atr, 2); tp1 = round(entry - tp_mult * atr, 2)
    capped, lv = _cap_tp(entry, tp1, is_buy, levels)
    risk, reward = abs(entry - sl), abs(entry - capped)
    rr = reward / risk if risk else 0.0
    return dict(atr=atr, tp_mult=tp_mult, ratio=ratio, sl=sl, tp1=capped, cap=lv, rr=rr)


def analyze_and_print(symbol: str, ohlcv_by_tf: Dict[str, pd.DataFrame],
                      config: Optional[TradingConfig] = None,
                      base_tf: str = "M15") -> AggregatedSignal:
    cfg = config or TradingConfig()
    base = ohlcv_by_tf[base_tf]
    price = float(base["close"].iloc[-1])

    tech = TechnicalAnalyzer(cfg); sr = SupportResistanceAnalyzer(cfg); ev = EventsAnalyzer(cfg)
    pa = PriceActionAnalyzer(cfg); vw = VWAPAnalyzer(cfg); mtf = MultiTimeframeAnalyzer(cfg)
    dv = DivergenceDetector(cfg); agg = SignalAggregator(cfg)

    tech_signals, regime, tech_score = tech.analyze(symbol, base)
    pa_res = pa.analyze(symbol, base, price)
    _, vwap_ctx, vwap_sig = vw.analyze(symbol, base, session_reset=True)
    mtf_res = mtf.analyze(symbol, ohlcv_by_tf)
    div_res = dv.analyze(symbol, base)
    sr_levels, sr_sig = sr.analyze(symbol, base, price)
    _, ev_sig = ev.analyze(symbol)

    flow_sig = TechnicalSignal(indicator="OrderFlow(CFD-n/a)", value=0.0,
                               signal=SignalStrength.NEUTRAL, description="N/A for spot CFD")
    agg_sig = agg.aggregate(
        symbol=symbol, order_flow_signal=flow_sig, technical_score=tech_score,
        sr_signal=sr_sig, event_signal=ev_sig, regime=regime,
        price_action_signal=pa_res.signal, vwap_signal=vwap_sig,
        mtf_signal=mtf_res.signal, divergence_signal=div_res.combined_signal,
        correlation_signal=None, session_signal=None, additional_signals=tech_signals,
    )
    parabolic = pa._detect_parabolic_extension(base)
    post_imp = pa._detect_post_impulse_correction(base)

    h1_atr = atr14(ohlcv_by_tf["H1"]) if "H1" in ohlcv_by_tf else float("nan")
    print("=" * 72)
    print(f"  {symbol} ENGINE RUN  |  price={price:.2f}  |  regime={regime.value}  "
          f"|  {base_tf} ATR={atr14(base):.2f}  H1 ATR={h1_atr:.2f}")
    print("=" * 72)
    print(f"Technical composite : {tech_score:+.2f}")
    for s in tech_signals:
        print(f"   {s.indicator:14s} {s.signal.name:12s} {s.description}")
    print("-" * 72)
    print(f"PriceAction         : {pa_res.signal.signal.name}  (score={pa_res.signal.value:+.2f})  "
          f"structure={pa_res.structure.trend}")
    print(f"   exhaustion   : {pa_res.exhaustion.description if pa_res.exhaustion else 'none'}")
    print(f"   parabolic    : {parabolic['description'] if parabolic else 'none'}")
    print(f"   post_impulse : {post_imp['description'] if post_imp else 'none'}")
    print("-" * 72)
    print(f"VWAP                : {vwap_sig.signal.name}  vwap={vwap_ctx.vwap:.2f}  "
          f"dev={vwap_ctx.deviation_pct*100:+.2f}%  band={vwap_ctx.band_position}  "
          f"trend={vwap_ctx.vwap_trend}  reclaim={vwap_ctx.is_reclaim}  reject={vwap_ctx.is_rejection}")
    print("-" * 72)
    print(f"MTF                 : htf_bias={mtf_res.htf_bias}  ltf_aligned={mtf_res.ltf_aligned}  "
          f"confluence={mtf_res.confluence_score:.0%}  can_trade={mtf_res.can_trade}")
    for tf in ("W1", "D1", "H4", "H1", "M15"):
        b = mtf_res.timeframe_biases.get(tf)
        if b:
            print(f"   {tf:4s} {b.trend:8s} score={b.score:+.2f} rsi={b.rsi:.0f} ema_aligned={b.ema_aligned}")
    print("-" * 72)
    print(f"Divergence          : {div_res.combined_signal.signal.name}  "
          f"RSI_divs={len(div_res.rsi_divergences)} MACD_divs={len(div_res.macd_divergences)}")
    nearest = min(sr_levels, key=lambda lv: abs(lv.price - price)) if sr_levels else None
    print(f"S/R                 : {sr_sig.signal.name} — {sr_sig.description[:66]}")
    if nearest:
        print(f"   nearest level: {nearest.level_type} @ {nearest.price:.2f} "
              f"(strength={nearest.strength:.2f}) {nearest.description}")
    print("=" * 72)
    blocked = range_stand_aside(regime, cfg.range_filter_mode,
                                agg_sig.direction in (SignalStrength.BUY, SignalStrength.STRONG_BUY),
                                vwap_ctx.band_position, sr_sig.signal)
    print(f"  AGGREGATED   : {agg_sig.direction.name}   score={agg_sig.composite_score:+.3f}   "
          f"confidence={agg_sig.confidence*100:.1f}%  (min={cfg.min_confidence*100:.0f}%)")
    print(f"  GATES: not_neutral={agg_sig.direction != SignalStrength.NEUTRAL} | "
          f"conf>=min={agg_sig.confidence >= cfg.min_confidence} | mtf_aligned={mtf_res.ltf_aligned} | "
          f"range_filter_block={blocked}")
    print("=" * 72)
    print("\n  TRADE-PARAM MATH (1.5xATR stop, directional late-entry TP scaling, SR cap, min RR=1.5)")
    tfs = [(base_tf, base)] + ([("H1", ohlcv_by_tf["H1"])] if "H1" in ohlcv_by_tf else [])
    for is_buy, label in [(True, "HYPOTHETICAL LONG"), (False, "HYPOTHETICAL SHORT")]:
        for tf_name, df in tfs:
            p = _params(df, is_buy, price, sr_levels, cfg)
            ok = p["rr"] >= cfg.min_risk_reward
            cap = f" cap@{p['cap'].price:.2f}" if p["cap"] else ""
            print(f"  {label:18s} [{tf_name}] ATR={p['atr']:.2f} dirMove/ATR={p['ratio']:.2f}  "
                  f"TPx={p['tp_mult']}  SL={p['sl']:.2f} TP1={p['tp1']:.2f}{cap}  "
                  f"R:R={p['rr']:.2f} -> {'ACCEPT' if ok else 'REJECT (<1.5)'}")
    print("=" * 72)
    return agg_sig
