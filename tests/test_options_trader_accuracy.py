"""
Regression tests for the options_trader analysis accuracy fixes.

These lock in the behaviour changes from the accuracy audit:
  * regime detection is timeframe-agnostic (intraday trends are detected)
  * divergence detection uses strict pivots + consecutive pairs (no phantoms)
  * S/R touch-counting counts approaches, not every containing bar
  * S/R periodic highs/lows use the real calendar index, not raw bar offsets

Skipped automatically if the scientific stack isn't installed.
"""
import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from options_trader.core.config import TradingConfig
from options_trader.core.models import MarketRegime, SignalStrength
from options_trader.analyzers.technical import TechnicalAnalyzer
from options_trader.analyzers.divergence import DivergenceDetector
from options_trader.analyzers.support_resistance import SupportResistanceAnalyzer


def _frame(closes, freq="15min", start="2026-06-01"):
    closes = np.asarray(closes, dtype=float)
    idx = pd.date_range(start=start, periods=len(closes), freq=freq)
    high = closes + 1.0
    low = closes - 1.0
    openp = np.r_[closes[0], closes[:-1]]
    vol = np.full(len(closes), 1000)
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": closes, "volume": vol},
        index=idx,
    )


def test_regime_detects_intraday_downtrend():
    """A steady intraday decline must read as TRENDING_DOWN, not LOW_VOLATILITY."""
    cfg = TradingConfig()
    closes = np.linspace(4500, 4400, 120) + np.random.default_rng(0).normal(0, 1.5, 120)
    _, regime, _ = TechnicalAnalyzer(cfg).analyze("XAUUSD", _frame(closes))
    assert regime == MarketRegime.TRENDING_DOWN


def test_divergence_no_phantoms_on_flat_series():
    """A flat (no real swings) series must not manufacture divergences."""
    cfg = TradingConfig()
    closes = 4500 + np.random.default_rng(1).normal(0, 0.2, 120)  # essentially flat
    res = DivergenceDetector(cfg).analyze("XAUUSD", _frame(closes))
    assert len(res.rsi_divergences) + len(res.macd_divergences) <= 1
    assert res.combined_signal.signal == SignalStrength.NEUTRAL


def test_touch_count_is_bounded():
    """A mid-range level must not report a touch for every containing bar."""
    cfg = TradingConfig()
    closes = 4500 + np.sin(np.linspace(0, 12, 200)) * 20  # oscillates across 4500
    sr = SupportResistanceAnalyzer(cfg)
    touches = sr._count_touches(_frame(closes), 4500.0)
    # Old behaviour returned ~len(df); a real approach count must be far smaller.
    assert touches < 40, f"touch count {touches} looks like the old per-bar count"


def test_periodic_levels_use_calendar_not_bar_offsets():
    """On intraday data, 'previous day' must come from a real calendar resample."""
    cfg = TradingConfig()
    closes = 4500 + np.random.default_rng(2).normal(0, 5, 300)
    df = _frame(closes, freq="15min")  # ~3 days of M15
    sr = SupportResistanceAnalyzer(cfg)
    levels = sr._periodic_highs_lows(df)
    daily = [lv for lv in levels if "Previous Day" in lv.description]
    assert daily, "expected a calendar-derived Previous Day level"
    # The PDH must equal the prior calendar day's high, not bar[-2]'s high.
    prev_day_high = df.resample("D").agg({"high": "max"}).dropna().iloc[-2]["high"]
    assert any(abs(lv.price - float(prev_day_high)) < 1e-6
               for lv in daily if lv.level_type == "resistance")
