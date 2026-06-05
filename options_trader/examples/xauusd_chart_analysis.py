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
from options_trader.reporting import analyze_and_print

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


def resample(df, rule):
    return df.resample(rule).agg({"open": "first", "high": "max", "low": "min",
                                  "close": "last", "volume": "sum"}).dropna()


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

    # Delegate the pipeline + report to the shared reporting module (same code
    # path as the real-CSV runner, examples/analyze_csv.py).
    analyze_and_print(SYMBOL, ohlcv_by_tf, config=TradingConfig())


if __name__ == "__main__":
    main()
