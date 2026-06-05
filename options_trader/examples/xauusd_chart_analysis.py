"""
Run the live-bot analysis pipeline on the XAUUSD M15 + H1 screenshots.

Reconstructs approximate OHLCV from the chart candles, then executes the SAME
analyzers the MT5 live loop uses (technical, price action, VWAP, MTF,
divergence, S/R, aggregator) plus the _calculate_trade_params SL/TP math —
without needing an MT5 terminal.

Reconstruction notes
--------------------
* The two screenshots only show the recent M15 and H1 windows. Higher
  timeframes (H4/D1/W1) are NOT visible, so a *plausible* multi-day gold
  context is synthesised and the H1 base series is resampled up to H4/D1/W1.
  Treat every HTF-derived number as ASSUMED CONTEXT, not read-from-chart.
* Lead-in bars use gold-realistic volatility so ATR(14) lands in a sensible
  range (M15 ~3-5, H1 ~10-15) and the regime classifier behaves correctly.
  Swap in real OHLC (e.g. exported from MT5) for a production-grade read.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from datetime import datetime, timezone

from options_trader.core.config import TradingConfig
from options_trader.core.models import SignalStrength, TechnicalSignal
from options_trader.analyzers.technical import TechnicalAnalyzer
from options_trader.analyzers.support_resistance import SupportResistanceAnalyzer
from options_trader.analyzers.events import EventsAnalyzer
from options_trader.analyzers.price_action import PriceActionAnalyzer
from options_trader.analyzers.vwap import VWAPAnalyzer
from options_trader.analyzers.multi_timeframe import MultiTimeframeAnalyzer
from options_trader.analyzers.divergence import DivergenceDetector
from options_trader.strategies.signal_aggregator import SignalAggregator

SYMBOL = "XAUUSD"

# ---------------------------------------------------------------------------
# Visible candles read off the screenshots: (open, high, low, close)
# ---------------------------------------------------------------------------
# M15 window (left->right), ends at current ~4459 (update 12:21 — another leg
# down from the 4484 lower-high to a ~4456 low, now a small 2-candle bounce).
M15_VIS = [
    (4500,4502,4498,4499),(4499,4505,4498,4504),(4504,4513,4503,4512),
    (4512,4513,4503,4504),(4504,4505,4495,4496),(4496,4497,4487,4488),
    (4488,4489,4479,4480),(4480,4481,4472,4473),(4473,4474,4456,4465),
    (4465,4471,4463,4470),(4470,4471,4466,4467),(4467,4477,4466,4476),
    (4476,4481,4475,4480),(4480,4484,4479,4482),(4482,4483,4477,4478),
    (4478,4480,4475,4476),(4476,4479,4475,4478),(4478,4480,4476,4477),
    (4477,4480,4475,4479),(4479,4481,4477,4478),(4478,4482,4477,4481),
    (4481,4482,4476,4477),(4477,4478,4472,4473),(4473,4474,4469,4470),
    (4470,4472,4469,4471),(4471,4472,4466,4467),(4467,4468,4463,4464),
    (4464,4465,4460,4461),(4461,4462,4457,4458),(4458,4459,4453,4456),
    (4456,4459,4455,4458),(4458,4460,4457,4459),
]
# H1 window, ends at current ~4461 (includes the parabolic spike to ~4515).
H1_VIS = [
    (4458,4460,4452,4454),(4454,4456,4447,4449),(4449,4451,4443,4445),
    (4445,4447,4427,4444),(4444,4456,4443,4455),(4455,4465,4453,4463),
    (4463,4466,4456,4458),(4458,4470,4457,4468),(4468,4477,4466,4475),
    (4475,4481,4473,4480),(4480,4482,4474,4476),(4476,4478,4461,4463),
    (4463,4515,4462,4511),(4511,4513,4502,4503),(4503,4504,4493,4495),
    (4495,4497,4478,4480),(4480,4484,4478,4483),(4483,4484,4472,4471),
    (4471,4475,4469,4473),(4473,4474,4460,4461),
]

IS_BUY = True  # set by params(); used by late_entry_mult()


def _walk(n, start, step_std, wick_std, drift, seed):
    """Generate n realistic OHLC candles via a drifting random walk."""
    rng = np.random.default_rng(seed)
    rows, px = [], start
    for _ in range(n):
        o = px
        c = o + drift + rng.normal(0, step_std)
        hi = max(o, c) + abs(rng.normal(0, wick_std))
        lo = min(o, c) - abs(rng.normal(0, wick_std))
        rows.append((o, hi, lo, c)); px = c
    return rows


def build_frame(visible, lead_n, freq_min, end_dt, step_std, wick_std,
                lead_drift, lead_seed):
    """Prepend a gold-realistic lead-in so analyzers have >=50 bars."""
    start_px = visible[0][0] - lead_drift * lead_n  # connect into the window
    lead = _walk(lead_n, start_px, step_std, wick_std, lead_drift, lead_seed)
    rows = lead + visible
    n = len(rows)
    idx = pd.date_range(end=end_dt, periods=n, freq=f"{freq_min}min")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    rng_ = (df["high"] - df["low"]).abs()
    noise = np.random.default_rng(lead_seed + 1).uniform(0.7, 1.3, size=n)
    df["volume"] = (rng_ * 220 * noise + 600).round().astype(int)
    return df


def atr14(df):
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    trs = [max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])) for i in range(1, len(df))]
    return float(sum(trs[-14:]) / 14)


def resample(df, rule):
    return df.resample(rule).agg({"open": "first", "high": "max", "low": "min",
                                  "close": "last", "volume": "sum"}).dropna()


def late_entry_mult(df, atr, is_buy, lookback=10, full=2.5, mn=1.2):
    d = df.tail(lookback); entry = float(d["close"].iloc[-1])
    pm = (entry - float(d["low"].min())) if is_buy else (float(d["high"].max()) - entry)
    pm = max(0.0, pm); r = pm / atr if atr else 0.0
    if r <= 1.5:
        return full, r
    red = min(0.5, (r - 1.5) * 0.15)
    return round(max(mn, full - red * full), 1), r


def cap_tp(entry, tp, is_buy, levels, buf=0.001):
    if is_buy:
        block = [lv for lv in levels if lv.level_type == "resistance"
                 and entry < lv.price < tp and lv.strength >= 0.75]
        if block:
            nr = min(block, key=lambda x: x.price); return round(nr.price*(1-buf), 2), nr
    else:
        block = [lv for lv in levels if lv.level_type == "support"
                 and tp < lv.price < entry and lv.strength >= 0.75]
        if block:
            nr = max(block, key=lambda x: x.price); return round(nr.price*(1+buf), 2), nr
    return tp, None


def params(df, is_buy, entry, levels):
    atr = atr14(df); tp_mult, ratio = late_entry_mult(df, atr, is_buy)
    if is_buy:
        sl = round(entry-1.5*atr, 2); tp1 = round(entry+tp_mult*atr, 2)
    else:
        sl = round(entry+1.5*atr, 2); tp1 = round(entry-tp_mult*atr, 2)
    capped, lv = cap_tp(entry, tp1, is_buy, levels)
    risk = abs(entry-sl); rew = abs(entry-capped); rr = rew/risk if risk else 0.0
    return dict(atr=atr, tp_mult=tp_mult, ratio=ratio, sl=sl, tp1=capped, cap=lv, rr=rr)


def main():
    # M15: dedicated intraday series ending in the visible window.
    m15 = build_frame(M15_VIS, lead_n=60, freq_min=15,
                      end_dt=datetime(2026, 6, 5, 3, 0, tzinfo=timezone.utc),
                      step_std=2.6, wick_std=1.6, lead_drift=0.0, lead_seed=11)
    # H1 base: long multi-day history -> resampled to H4/D1/W1.
    # Plausible context: multi-day grind higher into the spike, then rollover.
    # lead_n large enough that the resampled D1 frame has >=20 bars (MTF needs
    # >=20 bars per TF): ~500 H1 bars ~= 21 calendar days ~= 21 D1 candles.
    h1 = build_frame(H1_VIS, lead_n=480, freq_min=60,
                     end_dt=datetime(2026, 6, 4, 20, 0, tzinfo=timezone.utc),
                     step_std=6.0, wick_std=3.2, lead_drift=0.035, lead_seed=23)
    ohlcv_by_tf = {"M15": m15, "H1": h1, "H4": resample(h1, "4h"),
                   "D1": resample(h1, "1D"), "W1": resample(h1, "1W")}
    price = float(m15["close"].iloc[-1])

    cfg = TradingConfig()
    tech, sr, ev = TechnicalAnalyzer(cfg), SupportResistanceAnalyzer(cfg), EventsAnalyzer(cfg)
    pa, vw, mtf = PriceActionAnalyzer(cfg), VWAPAnalyzer(cfg), MultiTimeframeAnalyzer(cfg)
    dv, agg = DivergenceDetector(cfg), SignalAggregator(cfg)

    tech_signals, regime, tech_score = tech.analyze(SYMBOL, m15)
    pa_res = pa.analyze(SYMBOL, m15, price)
    _, vwap_ctx, vwap_sig = vw.analyze(SYMBOL, m15, session_reset=True)
    mtf_res = mtf.analyze(SYMBOL, ohlcv_by_tf)
    div_res = dv.analyze(SYMBOL, m15)
    sr_levels, sr_sig = sr.analyze(SYMBOL, m15, price)
    _, ev_sig = ev.analyze(SYMBOL)

    # Order flow is N/A for a spot CFD -> NEUTRAL (do not double-count RSI).
    flow_sig = TechnicalSignal(indicator="OrderFlow(CFD-n/a)", value=0.0,
                               signal=SignalStrength.NEUTRAL, description="N/A for spot CFD")

    agg_sig = agg.aggregate(
        symbol=SYMBOL, order_flow_signal=flow_sig, technical_score=tech_score,
        sr_signal=sr_sig, event_signal=ev_sig, regime=regime,
        price_action_signal=pa_res.signal, vwap_signal=vwap_sig,
        mtf_signal=mtf_res.signal, divergence_signal=div_res.combined_signal,
        correlation_signal=None, session_signal=None, additional_signals=tech_signals,
    )
    parabolic = pa._detect_parabolic_extension(m15)
    post_imp = pa._detect_post_impulse_correction(m15)

    print("="*72)
    print(f"  XAUUSD ENGINE RUN  |  price={price:.2f}  |  regime={regime.value}  "
          f"|  M15 ATR={atr14(m15):.2f}  H1 ATR={atr14(h1):.2f}")
    print("="*72)
    print(f"Technical composite : {tech_score:+.2f}")
    for s in tech_signals:
        print(f"   {s.indicator:14s} {s.signal.name:12s} {s.description}")
    print("-"*72)
    print(f"PriceAction         : {pa_res.signal.signal.name}  (score={pa_res.signal.value:+.2f})  "
          f"structure={pa_res.structure.trend}")
    print(f"   exhaustion   : {pa_res.exhaustion.description if pa_res.exhaustion else 'none'}")
    print(f"   fvg_fill     : {pa_res.fvg_fill_bias}")
    print(f"   parabolic    : {parabolic['description'] if parabolic else 'none'}")
    print(f"   post_impulse : {post_imp['description'] if post_imp else 'none'}")
    print("-"*72)
    print(f"VWAP                : {vwap_sig.signal.name}  vwap={vwap_ctx.vwap:.2f}  "
          f"dev={vwap_ctx.deviation_pct*100:+.2f}%  band={vwap_ctx.band_position}  "
          f"trend={vwap_ctx.vwap_trend}  reclaim={vwap_ctx.is_reclaim}  reject={vwap_ctx.is_rejection}")
    print("-"*72)
    print(f"MTF                 : htf_bias={mtf_res.htf_bias}  ltf_aligned={mtf_res.ltf_aligned}  "
          f"confluence={mtf_res.confluence_score:.0%}  can_trade={mtf_res.can_trade}")
    for tf in ("W1", "D1", "H4", "H1", "M15"):
        b = mtf_res.timeframe_biases.get(tf)
        if b:
            print(f"   {tf:4s} {b.trend:8s} score={b.score:+.2f} rsi={b.rsi:.0f} ema_aligned={b.ema_aligned}")
    print(f"   suppression: {mtf_res.suppression_reason or 'none'}")
    print("-"*72)
    print(f"Divergence          : {div_res.combined_signal.signal.name}  "
          f"RSI_divs={len(div_res.rsi_divergences)} MACD_divs={len(div_res.macd_divergences)}")
    nearest = min(sr_levels, key=lambda lv: abs(lv.price - price)) if sr_levels else None
    print(f"S/R                 : {sr_sig.signal.name} — {sr_sig.description[:66]}")
    if nearest:
        print(f"   nearest level: {nearest.level_type} @ {nearest.price:.2f} "
              f"(strength={nearest.strength:.2f}) {nearest.description}")
    print("="*72)
    print(f"  AGGREGATED   : {agg_sig.direction.name}   score={agg_sig.composite_score:+.3f}   "
          f"confidence={agg_sig.confidence*100:.1f}%  (min={cfg.min_confidence*100:.0f}%)")
    neutral = agg_sig.direction == SignalStrength.NEUTRAL
    print(f"  GATES: signal!=neutral={not neutral} | conf>=min={agg_sig.confidence >= cfg.min_confidence} "
          f"| mtf_aligned={mtf_res.ltf_aligned} | mtf_can_trade={mtf_res.can_trade}")
    print("="*72)
    print("\n  TRADE-PARAM MATH (1.5xATR stop, directional late-entry TP scaling, SR cap, min RR=1.5)")
    for is_buy, label in [(True, "HYPOTHETICAL LONG"), (False, "HYPOTHETICAL SHORT")]:
        for tf_name, df in [("M15", m15), ("H1", h1)]:
            p = params(df, is_buy, price, sr_levels)
            ok = p["rr"] >= cfg.min_risk_reward
            cap = f" capped@{p['cap'].price:.2f}({p['cap'].description[:22]})" if p['cap'] else ""
            print(f"  {label:18s} [{tf_name}] ATR={p['atr']:.2f} dirMove/ATR={p['ratio']:.2f}  "
                  f"TPx={p['tp_mult']}  SL={p['sl']:.2f} TP1={p['tp1']:.2f}{cap}  "
                  f"R:R={p['rr']:.2f} -> {'ACCEPT' if ok else 'REJECT (<1.5)'}")
    print("="*72)


if __name__ == "__main__":
    main()
