"""
A/B backtest: mean-reversion overrides + the RANGING stand-aside filter
=======================================================================
Runs the CFD backtester over several synthetic but regime-labelled gold
histories and prints an R-multiple comparison across four configs:
    MR=ON         overrides on,  no range filter (original behaviour)
    MR=OFF        overrides off, no range filter
    OFF+block     overrides off + stand aside entirely while RANGING
    OFF+reversal  overrides off + allow only mean-reversion fades in a range

The hypothesis under test (from the accuracy audit): the "fade-the-extreme"
overrides (parabolic dampener, exhaustion reversal, VWAP band flip, FVG-fill)
should HELP in ranging / spike-and-revert regimes but HURT in strong trends
(where extended/parabolic is normal and price keeps going).

NB: synthetic data validates the *harness and the directional behaviour of the
overrides across regimes* — not a real-money edge. Feed real MT5 history for
that. Seeds are fixed for reproducibility.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timezone

from options_trader.core.config import TradingConfig
from options_trader.backtesting.cfd_backtester import CFDBacktester, CFDBacktestResult


def _ohlc_from_close(closes, wick_std, seed):
    rng = np.random.default_rng(seed)
    o = np.r_[closes[0], closes[:-1]]
    hi = np.maximum(o, closes) + np.abs(rng.normal(0, wick_std, len(closes)))
    lo = np.minimum(o, closes) - np.abs(rng.normal(0, wick_std, len(closes)))
    vol = ((hi - lo) * 200 * rng.uniform(0.7, 1.3, len(closes)) + 600).round()
    idx = pd.date_range(end=datetime(2026, 6, 1, tzinfo=timezone.utc),
                        periods=len(closes), freq="15min")
    return pd.DataFrame({"open": o, "high": hi, "low": lo,
                         "close": closes, "volume": vol}, index=idx)


def gen_trend(n, drift, seed, start=4400.0, step=2.2):
    rng = np.random.default_rng(seed)
    closes = start + np.cumsum(drift + rng.normal(0, step, n))
    return _ohlc_from_close(closes, wick_std=1.4, seed=seed + 1)


def gen_range(n, seed, level=4500.0, theta=0.05, step=2.6):
    rng = np.random.default_rng(seed)
    closes = np.empty(n); px = level
    for i in range(n):
        px += theta * (level - px) + rng.normal(0, step)
        closes[i] = px
    return _ohlc_from_close(closes, wick_std=1.5, seed=seed + 1)


def gen_spike_revert(n, seed, level=4500.0, step=2.4):
    """Range with periodic parabolic spikes that fully mean-revert."""
    rng = np.random.default_rng(seed)
    closes = np.empty(n); px = level; i = 0
    while i < n:
        if i > 0 and i % 90 == 0:
            direction = rng.choice([-1, 1])
            run = rng.integers(5, 8)            # parabolic impulse
            for _ in range(run):
                if i >= n: break
                px += direction * rng.uniform(5, 8); closes[i] = px; i += 1
            back = rng.integers(8, 14)          # revert to the mean
            for _ in range(back):
                if i >= n: break
                px += (level - px) * 0.25 + rng.normal(0, step); closes[i] = px; i += 1
        else:
            px += 0.05 * (level - px) + rng.normal(0, step); closes[i] = px; i += 1
    return _ohlc_from_close(closes, wick_std=1.6, seed=seed + 1)


def main():
    n = 480
    scenarios = {
        "strong_uptrend":  gen_trend(n, drift=+0.9, seed=1, start=4350),
        "strong_downtrend": gen_trend(n, drift=-0.9, seed=2, start=4650),
        "ranging":         gen_range(n, seed=3),
        "spike_and_revert": gen_spike_revert(n, seed=4),
    }

    # Three configs:
    #   MR=ON   overrides on, no range filter (original post-mortem behaviour)
    #   MR=OFF  overrides off, no range filter (best from the prior A/B)
    #   OFF+RF  overrides off + RANGING stand-aside filter (the prototype)
    configs = [
        ("MR=ON  ",        dict(enable_mean_reversion_overrides=True,  range_filter_mode="off")),
        ("MR=OFF ",        dict(enable_mean_reversion_overrides=False, range_filter_mode="off")),
        ("OFF+block",      dict(enable_mean_reversion_overrides=False, range_filter_mode="block_all")),
        ("OFF+reversal",   dict(enable_mean_reversion_overrides=False, range_filter_mode="reversal_only")),
    ]

    print(CFDBacktestResult.header())
    print("-" * 92)
    agg = {tag: [] for tag, _ in configs}
    for name, df in scenarios.items():
        for tag, kw in configs:
            cfg = TradingConfig()
            for k, v in kw.items():
                setattr(cfg, k, v)
            res = CFDBacktester(cfg).run(df, label=f"{name:<17} {tag}")
            agg[tag].append(res)
            print(res.row())
        print("-" * 92)

    # Portfolio roll-up across all scenarios
    print("\nAGGREGATE ACROSS ALL REGIMES")
    print(CFDBacktestResult.header())
    print("-" * 92)
    for tag, _ in configs:
        merged = CFDBacktestResult(label=f"{'ALL':<17} {tag}")
        for r in agg[tag]:
            merged.trades.extend(r.trades)
        print(merged.row())


if __name__ == "__main__":
    main()
