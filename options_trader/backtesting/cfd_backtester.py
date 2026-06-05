"""
CFD / spot backtester
=====================
Replays an OHLCV history bar-by-bar through the SAME analysis pipeline the
MT5 live loop uses (technical, price action, VWAP, MTF, divergence, S/R,
aggregator) and the same ATR-based SL/TP logic, then reports R-multiple
performance statistics.

Unlike the options Backtester, this models a spot CFD position (long/short,
1.5xATR stop, directional late-entry-scaled TP capped at S/R) so it can be
used to A/B the mean-reversion overrides via
``config.enable_mean_reversion_overrides``.

Results are reported in R (risk multiples), which is capital-agnostic:
one losing trade = -1R, a trade that hits a 2R target = +2R.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from options_trader.core.config import TradingConfig
from options_trader.core.models import SignalStrength, TechnicalSignal
from options_trader.analyzers.technical import TechnicalAnalyzer
from options_trader.analyzers.support_resistance import SupportResistanceAnalyzer
from options_trader.analyzers.price_action import PriceActionAnalyzer
from options_trader.analyzers.vwap import VWAPAnalyzer
from options_trader.analyzers.multi_timeframe import MultiTimeframeAnalyzer
from options_trader.analyzers.divergence import DivergenceDetector
from options_trader.analyzers.events import EventsAnalyzer
from options_trader.strategies.signal_aggregator import SignalAggregator
from options_trader.strategies.regime_filter import range_stand_aside

logger = logging.getLogger(__name__)

_BUY = (SignalStrength.BUY, SignalStrength.STRONG_BUY)
_SELL = (SignalStrength.SELL, SignalStrength.STRONG_SELL)


@dataclass
class CFDTrade:
    direction: str            # "BUY" | "SELL"
    entry_index: int
    entry_price: float
    sl: float
    tp: float
    risk: float               # |entry - sl| in price units
    exit_index: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""     # "tp" | "sl" | "timeout"
    confidence: float = 0.0

    @property
    def r_multiple(self) -> float:
        if self.exit_price is None or self.risk == 0:
            return 0.0
        sign = 1.0 if self.direction == "BUY" else -1.0
        return sign * (self.exit_price - self.entry_price) / self.risk


@dataclass
class CFDBacktestResult:
    label: str
    trades: List[CFDTrade] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.r_multiple > 0)

    @property
    def win_rate(self) -> float:
        return self.wins / self.n if self.n else 0.0

    @property
    def expectancy_r(self) -> float:
        return float(np.mean([t.r_multiple for t in self.trades])) if self.n else 0.0

    @property
    def total_r(self) -> float:
        return float(sum(t.r_multiple for t in self.trades))

    @property
    def avg_win_r(self) -> float:
        w = [t.r_multiple for t in self.trades if t.r_multiple > 0]
        return float(np.mean(w)) if w else 0.0

    @property
    def avg_loss_r(self) -> float:
        l = [t.r_multiple for t in self.trades if t.r_multiple <= 0]
        return float(np.mean(l)) if l else 0.0

    @property
    def profit_factor(self) -> float:
        gross_win = sum(t.r_multiple for t in self.trades if t.r_multiple > 0)
        gross_loss = abs(sum(t.r_multiple for t in self.trades if t.r_multiple < 0))
        return gross_win / gross_loss if gross_loss else float("inf")

    @property
    def max_drawdown_r(self) -> float:
        eq = np.cumsum([t.r_multiple for t in self.trades]) if self.n else np.array([0.0])
        peak = np.maximum.accumulate(eq)
        return float(np.max(peak - eq)) if self.n else 0.0

    def row(self) -> str:
        return (f"{self.label:<26} {self.n:>4}  {self.win_rate*100:>5.1f}%  "
                f"{self.expectancy_r:>+6.2f}R  {self.total_r:>+7.1f}R  "
                f"{self.profit_factor:>5.2f}  {self.max_drawdown_r:>5.1f}R  "
                f"{self.avg_win_r:>+5.2f}/{self.avg_loss_r:>+5.2f}")

    @staticmethod
    def header() -> str:
        return (f"{'scenario / config':<26} {'#':>4}  {'win':>6}  "
                f"{'expect':>6}  {'totR':>7}  {'PF':>5}  {'maxDD':>6}  {'avgW/L':>11}")


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    if len(df) < 2:
        return 0.0
    trs = [max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])) for i in range(1, len(df))]
    k = min(period, len(trs))
    return float(sum(trs[-k:]) / k)


def _late_entry_mult(window: pd.DataFrame, is_buy: bool, atr: float,
                     lookback: int = 10, full: float = 2.5, mn: float = 1.2) -> float:
    d = window.tail(lookback)
    if len(d) < 3 or atr == 0:
        return full
    entry = float(d["close"].iloc[-1])
    move = (entry - float(d["low"].min())) if is_buy else (float(d["high"].max()) - entry)
    ratio = max(0.0, move) / atr
    if ratio <= 1.5:
        return full
    reduction = min(0.5, (ratio - 1.5) * 0.15)
    return round(max(mn, full - reduction * full), 1)


def _cap_tp_at_sr(entry, tp, is_buy, sr_levels, buf=0.001):
    if is_buy:
        block = [lv for lv in sr_levels if lv.level_type == "resistance"
                 and entry < lv.price < tp and lv.strength >= 0.75]
        if block:
            return round(min(block, key=lambda x: x.price).price * (1 - buf), 2)
    else:
        block = [lv for lv in sr_levels if lv.level_type == "support"
                 and tp < lv.price < entry and lv.strength >= 0.75]
        if block:
            return round(max(block, key=lambda x: x.price).price * (1 + buf), 2)
    return tp


class CFDBacktester:
    """Bar-by-bar spot CFD backtester over the full analysis pipeline."""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.tech = TechnicalAnalyzer(config)
        self.sr = SupportResistanceAnalyzer(config)
        self.pa = PriceActionAnalyzer(config)
        self.vw = VWAPAnalyzer(config)
        self.mtf = MultiTimeframeAnalyzer(config)
        self.dv = DivergenceDetector(config)
        self.ev = EventsAnalyzer(config)
        self.agg = SignalAggregator(config)

    # ------------------------------------------------------------------
    def run(self, df: pd.DataFrame, label: str = "", symbol: str = "XAUUSD",
            warmup: int = 80, max_hold: int = 48) -> CFDBacktestResult:
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        result = CFDBacktestResult(label=label)
        open_trade: Optional[CFDTrade] = None
        n = len(df)

        for i in range(warmup, n):
            bar = df.iloc[i]

            # 1) Manage an open position (intrabar SL/TP; SL assumed first if both hit)
            if open_trade is not None:
                hit = self._check_exit(open_trade, bar, i)
                if hit or (i - open_trade.entry_index) >= max_hold:
                    if not hit:
                        open_trade.exit_index = i
                        open_trade.exit_price = float(bar["close"])
                        open_trade.exit_reason = "timeout"
                    result.trades.append(open_trade)
                    open_trade = None
                continue

            # 2) Flat -> look for an entry on the closed window up to bar i
            window = df.iloc[: i + 1]
            open_trade = self._maybe_enter(symbol, window, i)

        # close any trade still open at the end
        if open_trade is not None:
            open_trade.exit_index = n - 1
            open_trade.exit_price = float(df.iloc[-1]["close"])
            open_trade.exit_reason = "timeout"
            result.trades.append(open_trade)
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _check_exit(trade: CFDTrade, bar, i: int) -> bool:
        hi, lo = float(bar["high"]), float(bar["low"])
        if trade.direction == "BUY":
            if lo <= trade.sl:
                trade.exit_index, trade.exit_price, trade.exit_reason = i, trade.sl, "sl"
                return True
            if hi >= trade.tp:
                trade.exit_index, trade.exit_price, trade.exit_reason = i, trade.tp, "tp"
                return True
        else:
            if hi >= trade.sl:
                trade.exit_index, trade.exit_price, trade.exit_reason = i, trade.sl, "sl"
                return True
            if lo <= trade.tp:
                trade.exit_index, trade.exit_price, trade.exit_reason = i, trade.tp, "tp"
                return True
        return False

    # ------------------------------------------------------------------
    def _maybe_enter(self, symbol: str, window: pd.DataFrame, i: int) -> Optional[CFDTrade]:
        price = float(window["close"].iloc[-1])
        try:
            tech_signals, regime, tech_score = self.tech.analyze(symbol, window)
            pa_res = self.pa.analyze(symbol, window, price)
            _, vwap_ctx, vwap_sig = self.vw.analyze(symbol, window, session_reset=True)
            mtf_res = self.mtf.analyze(symbol, self._mtf_frames(window))
            div_res = self.dv.analyze(symbol, window)
            sr_levels, sr_sig = self.sr.analyze(symbol, window, price)
            _, ev_sig = self.ev.analyze(symbol)
        except Exception as exc:
            logger.debug("analysis error at %d: %s", i, exc)
            return None

        flow_sig = TechnicalSignal(indicator="OrderFlow", value=0.0,
                                   signal=SignalStrength.NEUTRAL, description="N/A (CFD)")
        agg = self.agg.aggregate(
            symbol=symbol, order_flow_signal=flow_sig, technical_score=tech_score,
            sr_signal=sr_sig, event_signal=ev_sig, regime=regime,
            price_action_signal=pa_res.signal, vwap_signal=vwap_sig,
            mtf_signal=mtf_res.signal, divergence_signal=div_res.combined_signal,
            correlation_signal=None, session_signal=None, additional_signals=tech_signals,
        )

        # gates: direction, confidence, MTF alignment
        if agg.direction == SignalStrength.NEUTRAL:
            return None
        if agg.confidence < self.config.min_confidence:
            return None
        if not mtf_res.ltf_aligned:
            return None

        is_buy = agg.direction in _BUY

        # Regime stand-aside filter (configurable; see TradingConfig).
        if range_stand_aside(
            regime, self.config.range_filter_mode, is_buy,
            vwap_ctx.band_position, sr_sig.signal,
        ):
            return None
        atr = _atr(window)
        if atr == 0:
            return None
        tp_mult = _late_entry_mult(window, is_buy, atr)
        if is_buy:
            sl = round(price - 1.5 * atr, 2)
            tp = round(price + tp_mult * atr, 2)
        else:
            sl = round(price + 1.5 * atr, 2)
            tp = round(price - tp_mult * atr, 2)
        tp = _cap_tp_at_sr(price, tp, is_buy, sr_levels)

        risk = abs(price - sl)
        reward = abs(price - tp)
        if risk == 0 or reward / risk < self.config.min_risk_reward:
            return None

        return CFDTrade(
            direction="BUY" if is_buy else "SELL",
            entry_index=i, entry_price=price, sl=sl, tp=tp, risk=risk,
            confidence=agg.confidence,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _mtf_frames(window: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Resample the base (M15) window up to H1/H4 for the MTF gate."""
        frames = {"M15": window}
        if isinstance(window.index, pd.DatetimeIndex):
            for label, rule in (("H1", "1h"), ("H4", "4h")):
                rs = window.resample(rule).agg(
                    {"open": "first", "high": "max", "low": "min",
                     "close": "last", "volume": "sum"}).dropna()
                if len(rs) >= 20:
                    frames[label] = rs
        return frames
