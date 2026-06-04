"""
Run the live-bot analysis pipeline on the XAUUSD M15 + H1 screenshots.
Reconstructs approximate OHLCV from the chart candles, then executes the SAME
analyzers the MT5 live loop uses, plus the _calculate_trade_params math.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone

from options_trader.core.config import TradingConfig
from options_trader.core.models import SignalStrength
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
# M15 (most recent window, left->right). Ends at current ~4462.
M15_VIS = [
    (4495,4496,4476,4477),(4477,4484,4476,4483),(4483,4484,4477,4478),
    (4478,4481,4476,4480),(4480,4481,4475,4476),(4476,4479,4474,4478),
    (4478,4480,4476,4477),(4477,4479,4471,4472),(4472,4474,4470,4473),
    (4473,4481,4472,4480),(4480,4482,4478,4479),(4479,4480,4473,4474),
    (4474,4476,4471,4472),(4472,4473,4467,4468),(4468,4470,4466,4470),
    (4470,4472,4468,4469),(4469,4470,4463,4464),(4464,4466,4461,4462),
    (4462,4464,4459,4463),(4463,4464,4460,4462),
]
# H1 window. Ends at current ~4461.
H1_VIS = [
    (4458,4460,4452,4454),(4454,4456,4447,4449),(4449,4451,4443,4445),
    (4445,4447,4427,4444),(4444,4456,4443,4455),(4455,4465,4453,4463),
    (4463,4466,4456,4458),(4458,4470,4457,4468),(4468,4477,4466,4475),
    (4475,4481,4473,4480),(4480,4482,4474,4476),(4476,4478,4461,4463),
    (4463,4515,4462,4511),(4511,4513,4502,4503),(4503,4504,4493,4495),
    (4495,4497,4478,4480),(4480,4484,4478,4483),(4483,4484,4472,4471),
    (4471,4475,4469,4473),(4473,4474,4460,4461),
]

def build_frame(visible, periods_lead, freq_min, end_dt):
    """Prepend a deterministic mild lead-in so analyzers have >=50 bars."""
    rng = np.random.default_rng(7)
    start_px = visible[0][0]
    lead = []
    px = start_px - 6  # start a touch lower and drift in
    for _ in range(periods_lead):
        o = px
        c = px + rng.normal(0, 1.2)
        hi = max(o, c) + abs(rng.normal(0, 0.8))
        lo = min(o, c) - abs(rng.normal(0, 0.8))
        lead.append((o, hi, lo, c)); px = c
    rows = lead + visible
    n = len(rows)
    idx = pd.date_range(end=end_dt, periods=n, freq=f"{freq_min}min")
    df = pd.DataFrame(rows, columns=["open","high","low","close"], index=idx)
    # volume: proportional to range, bigger on impulse candles
    rng2 = (df["high"]-df["low"]).abs()
    df["volume"] = (rng2*350 + 800).round().astype(int)
    return df

end = datetime(2026,6,5,2,30, tzinfo=timezone.utc)
m15 = build_frame(M15_VIS, 45, 15, end)
h1  = build_frame(H1_VIS, 45, 60, datetime(2026,6,4,20,0, tzinfo=timezone.utc))
price = float(m15["close"].iloc[-1])

cfg = TradingConfig()
tech = TechnicalAnalyzer(cfg); sr = SupportResistanceAnalyzer(cfg)
ev = EventsAnalyzer(cfg); pa = PriceActionAnalyzer(cfg)
vw = VWAPAnalyzer(cfg); mtf = MultiTimeframeAnalyzer(cfg)
dv = DivergenceDetector(cfg); agg = SignalAggregator(cfg)

# --- run modules (mirrors MT5LiveBot._run_analysis) ---
tech_signals, regime, tech_score = tech.analyze(SYMBOL, m15)
pa_res = pa.analyze(SYMBOL, m15, price)
_, vwap_ctx, vwap_sig = vw.analyze(SYMBOL, m15, session_reset=True)
mtf_res = mtf.analyze(SYMBOL, {"M15": m15, "H1": h1})
div_res = dv.analyze(SYMBOL, m15)
sr_levels, sr_sig = sr.analyze(SYMBOL, m15, price)
_, ev_sig = ev.analyze(SYMBOL)
flow_sig = tech_signals[0]
agg_sig = agg.aggregate(
    symbol=SYMBOL, order_flow_signal=flow_sig, technical_score=tech_score,
    sr_signal=sr_sig, event_signal=ev_sig, regime=regime,
    price_action_signal=pa_res.signal, vwap_signal=vwap_sig,
    mtf_signal=mtf_res.signal, divergence_signal=div_res.combined_signal,
    correlation_signal=None, session_signal=None, additional_signals=tech_signals,
)

# extra PA flags (private but informative)
parabolic = pa._detect_parabolic_extension(m15)
post_imp  = pa._detect_post_impulse_correction(m15)

# --- trade params (replicates MT5LiveBot._calculate_trade_params) ---
def atr14(df):
    h,l,c = df["high"].values, df["low"].values, df["close"].values
    trs=[max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1])) for i in range(1,len(df))]
    return float(sum(trs[-14:])/14)

def late_entry_mult(df, atr, lookback=10, full=2.5, mn=1.2):
    d=df.tail(lookback); pm=float(d["high"].max())-float(d["low"].min())
    r=pm/atr
    if r<=1.5: return full
    red=min(0.5,(r-1.5)*0.15)
    return round(max(mn, full-red*full),1), r

def cap_tp(entry,tp,is_buy,levels,buf=0.001):
    if is_buy:
        block=[lv for lv in levels if lv.level_type=="resistance" and entry<lv.price<tp and lv.strength>=0.75]
        if block:
            nr=min(block,key=lambda x:x.price); return round(nr.price*(1-buf),2), nr
    else:
        block=[lv for lv in levels if lv.level_type=="support" and tp<lv.price<entry and lv.strength>=0.75]
        if block:
            nr=max(block,key=lambda x:x.price); return round(nr.price*(1+buf),2), nr
    return tp, None

def params(df, is_buy, entry, levels):
    atr=atr14(df); lm=late_entry_mult(df,atr)
    tp_mult, ratio = (lm if isinstance(lm,tuple) else (lm,None))
    if is_buy:
        sl=round(entry-1.5*atr,2); tp1=round(entry+tp_mult*atr,2)
    else:
        sl=round(entry+1.5*atr,2); tp1=round(entry-tp_mult*atr,2)
    capped,lv=cap_tp(entry,tp1,is_buy,levels)
    risk=abs(entry-sl); rew=abs(entry-capped); rr=rew/risk if risk else 0
    return dict(atr=atr,tp_mult=tp_mult,prior_ratio=ratio,sl=sl,tp1=tp1,
                tp1_capped=capped,cap_lvl=lv,risk=risk,reward=rew,rr=rr)

print("="*70)
print(f"  XAUUSD ENGINE RUN  |  price={price:.2f}  |  regime={regime.value}")
print("="*70)
print(f"Technical composite : {tech_score:+.2f}")
for s in tech_signals: print(f"   {s.indicator:14s} {s.signal.name:12s} {s.description}")
print("-"*70)
print(f"PriceAction sig     : {pa_res.signal.signal.name}  (score={pa_res.signal.value:+.2f})")
print(f"   structure        : {pa_res.structure.trend}")
print(f"   exhaustion       : {pa_res.exhaustion.description if pa_res.exhaustion else 'none'}")
print(f"   fvg_fill_bias    : {pa_res.fvg_fill_bias}")
print(f"   parabolic        : {parabolic['description'] if parabolic else 'none'}")
print(f"   post_impulse     : {post_imp['description'] if post_imp else 'none'}")
print("-"*70)
print(f"VWAP                : {vwap_sig.signal.name}  vwap={vwap_ctx.vwap:.2f}  "
      f"dev={vwap_ctx.deviation_pct*100:+.2f}%  band={vwap_ctx.band_position}  "
      f"trend={vwap_ctx.vwap_trend}  reclaim={vwap_ctx.is_reclaim}  reject={vwap_ctx.is_rejection}")
print(f"   {vwap_sig.description}")
print("-"*70)
print(f"MTF                 : htf_bias={mtf_res.htf_bias}  ltf_aligned={mtf_res.ltf_aligned}  "
      f"confluence={mtf_res.confluence_score:.0%}  can_trade={mtf_res.can_trade}")
for tf,b in mtf_res.timeframe_biases.items():
    print(f"   {tf:4s} {b.trend:8s} score={b.score:+.2f} rsi={b.rsi:.0f} ema_aligned={b.ema_aligned}")
print(f"   suppression: {mtf_res.suppression_reason or 'none'}")
print("-"*70)
print(f"Divergence          : {div_res.combined_signal.signal.name}  "
      f"RSI_divs={len(div_res.rsi_divergences)} MACD_divs={len(div_res.macd_divergences)}")
print(f"S/R signal          : {sr_sig.signal.name} — {sr_sig.description[:80]}")
print("="*70)
print(f"  AGGREGATED         : {agg_sig.direction.name}")
print(f"  composite score    : {agg_sig.composite_score:+.3f}")
print(f"  confidence         : {agg_sig.confidence*100:.1f}%   (min to trade = {cfg.min_confidence*100:.0f}%)")
print(f"  regime             : {agg_sig.regime.value}")
neutral = agg_sig.direction == SignalStrength.NEUTRAL
gate_conf = agg_sig.confidence >= cfg.min_confidence
mtf_gate = mtf_res.ltf_aligned
print(f"  GATES: signal!=neutral={not neutral} | conf>=min={gate_conf} | mtf_aligned={mtf_gate}")
print("="*70)

print("\n  TRADE-PARAM MATH (1.5xATR stop, late-entry TP scaling, SR cap, min RR=1.5)")
for is_buy,label in [(True,"HYPOTHETICAL LONG"),(False,"HYPOTHETICAL SHORT")]:
    for tf_name,df in [("M15",m15),("H1",h1)]:
        p=params(df,is_buy,price,sr_levels)
        ok = p["rr"]>=cfg.min_risk_reward
        cap = f" capped@{p['tp1_capped']:.2f}({p['cap_lvl'].description[:30]})" if p['cap_lvl'] else ""
        print(f"  {label:18s} [{tf_name}] ATR={p['atr']:.2f} priorMove/ATR={p['prior_ratio']}  "
              f"TPx={p['tp_mult']}  SL={p['sl']:.2f} TP1={p['tp1_capped']:.2f}{cap}  "
              f"R:R={p['rr']:.2f}  -> {'ACCEPT' if ok else 'REJECT (<1.5)'}")
print("="*70)
