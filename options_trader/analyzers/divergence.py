"""
RSI + MACD Divergence Detector
Provides:
- Bullish regular divergence: price makes lower low, RSI makes higher low → LONG
- Bearish regular divergence: price makes higher high, RSI makes lower high → SHORT
- Bullish hidden divergence: price makes higher low, RSI makes lower low → trend continuation LONG
- Bearish hidden divergence: price makes lower high, RSI makes higher high → trend continuation SHORT
- MACD histogram divergence (same logic applied to histogram)
- Combined RSI + MACD signal for higher confidence
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from options_trader.core.models import SignalStrength, TechnicalSignal
from options_trader.core.config import TradingConfig

logger = logging.getLogger(__name__)


@dataclass
class DivergenceEvent:
    """A single detected divergence between price and an oscillator."""
    divergence_type: str          # "bullish_regular", "bearish_regular", "bullish_hidden", "bearish_hidden"
    indicator: str                # "RSI" or "MACD"
    price_point_1: float          # first pivot price
    price_point_2: float          # second pivot price (more recent)
    osc_point_1: float            # oscillator at first pivot
    osc_point_2: float            # oscillator at second pivot
    bar_index_1: int
    bar_index_2: int
    confidence: float             # 0.0 – 1.0
    description: str = ""

    @property
    def is_bullish(self) -> bool:
        return "bullish" in self.divergence_type

    @property
    def bars_apart(self) -> int:
        return abs(self.bar_index_2 - self.bar_index_1)


@dataclass
class DivergenceResult:
    """Full divergence analysis result."""
    rsi_divergences: List[DivergenceEvent]
    macd_divergences: List[DivergenceEvent]
    combined_signal: TechnicalSignal
    highest_confidence: float
    rsi_value: float
    macd_histogram: float


class DivergenceDetector:
    """
    Detects price/oscillator divergence using local pivot points.

    Regular divergence = counter-trend (reversal signal)
    Hidden divergence  = trend continuation
    """

    def __init__(self, config: TradingConfig, pivot_order: int = 5):
        self.config = config
        self.pivot_order = pivot_order   # bars on each side of a pivot

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyze(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        lookback: int = 60,
    ) -> DivergenceResult:
        """
        Args:
            ohlcv: OHLCV DataFrame with DatetimeIndex
            lookback: number of most recent bars to scan

        Returns:
            DivergenceResult with all detected events and a combined signal.
        """
        df = ohlcv.copy().tail(max(lookback, 50))
        df.columns = [c.lower() for c in df.columns]
        close = df["close"].reset_index(drop=True)

        # Compute oscillators
        rsi = self._rsi(close)
        macd_hist = self._macd_histogram(close)

        rsi_divs   = self._detect_divergences(close, rsi, "RSI")
        macd_divs  = self._detect_divergences(close, macd_hist, "MACD")

        # Filter to recent events only (within last 20 bars)
        n = len(close)
        rsi_divs  = [d for d in rsi_divs  if d.bar_index_2 >= n - 20]
        macd_divs = [d for d in macd_divs if d.bar_index_2 >= n - 20]

        signal = self._build_signal(rsi_divs, macd_divs)
        highest_conf = max(
            [d.confidence for d in rsi_divs + macd_divs], default=0.0
        )

        logger.info(
            "Divergence %s | RSI_divs=%d | MACD_divs=%d | signal=%s | conf=%.2f",
            symbol,
            len(rsi_divs), len(macd_divs),
            signal.signal.name, highest_conf,
        )

        return DivergenceResult(
            rsi_divergences=rsi_divs,
            macd_divergences=macd_divs,
            combined_signal=signal,
            highest_confidence=highest_conf,
            rsi_value=float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0,
            macd_histogram=float(macd_hist.iloc[-1]) if not np.isnan(macd_hist.iloc[-1]) else 0.0,
        )

    # ------------------------------------------------------------------
    # Core divergence detection
    # ------------------------------------------------------------------

    def _detect_divergences(
        self,
        price: pd.Series,
        oscillator: pd.Series,
        indicator_name: str,
    ) -> List[DivergenceEvent]:
        events: List[DivergenceEvent] = []

        price_arr = price.values
        osc_arr = oscillator.values

        # Find local lows and highs
        order = self.pivot_order
        lows_idx  = argrelextrema(price_arr, np.less_equal,    order=order)[0]
        highs_idx = argrelextrema(price_arr, np.greater_equal, order=order)[0]

        # --- Regular Bullish: price lower-low, osc higher-low ---
        events += self._scan_pairs(
            price_arr, osc_arr, lows_idx,
            price_lower=True, osc_lower=False,
            div_type="bullish_regular",
            indicator=indicator_name,
        )

        # --- Regular Bearish: price higher-high, osc lower-high ---
        events += self._scan_pairs(
            price_arr, osc_arr, highs_idx,
            price_lower=False, osc_lower=True,
            div_type="bearish_regular",
            indicator=indicator_name,
        )

        # --- Hidden Bullish: price higher-low, osc lower-low ---
        events += self._scan_pairs(
            price_arr, osc_arr, lows_idx,
            price_lower=False, osc_lower=True,
            div_type="bullish_hidden",
            indicator=indicator_name,
        )

        # --- Hidden Bearish: price lower-high, osc higher-high ---
        events += self._scan_pairs(
            price_arr, osc_arr, highs_idx,
            price_lower=True, osc_lower=False,
            div_type="bearish_hidden",
            indicator=indicator_name,
        )

        return events

    def _scan_pairs(
        self,
        price: np.ndarray,
        osc: np.ndarray,
        pivot_indices: np.ndarray,
        price_lower: bool,
        osc_lower: bool,
        div_type: str,
        indicator: str,
        max_gap: int = 40,
        min_gap: int = 5,
    ) -> List[DivergenceEvent]:
        events = []
        n = len(pivot_indices)
        for i in range(n - 1):
            for j in range(i + 1, n):
                idx1, idx2 = pivot_indices[i], pivot_indices[j]
                gap = idx2 - idx1
                if gap < min_gap or gap > max_gap:
                    continue

                p1, p2 = price[idx1], price[idx2]
                o1, o2 = osc[idx1], osc[idx2]

                if np.isnan(o1) or np.isnan(o2):
                    continue

                price_cond = (p2 < p1) if price_lower else (p2 > p1)
                osc_cond   = (o2 > o1) if not osc_lower else (o2 < o1)

                if price_cond and osc_cond:
                    price_diff = abs(p2 - p1) / (abs(p1) + 1e-9)
                    osc_diff   = abs(o2 - o1) / (max(abs(o1), abs(o2)) + 1e-9)
                    confidence = min(0.95, 0.50 + 0.25 * min(price_diff / 0.01, 1.0) + 0.25 * min(osc_diff / 0.2, 1.0))

                    desc = (
                        f"{indicator} {div_type.replace('_', ' ').title()}: "
                        f"price {'↓' if price_lower else '↑'} {p1:.2f}→{p2:.2f}, "
                        f"{indicator} {'↑' if not osc_lower else '↓'} {o1:.2f}→{o2:.2f}"
                    )
                    events.append(DivergenceEvent(
                        divergence_type=div_type,
                        indicator=indicator,
                        price_point_1=float(p1),
                        price_point_2=float(p2),
                        osc_point_1=float(o1),
                        osc_point_2=float(o2),
                        bar_index_1=int(idx1),
                        bar_index_2=int(idx2),
                        confidence=confidence,
                        description=desc,
                    ))
        return events

    # ------------------------------------------------------------------
    # Signal synthesis
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        rsi_divs: List[DivergenceEvent],
        macd_divs: List[DivergenceEvent],
    ) -> TechnicalSignal:
        all_divs = rsi_divs + macd_divs
        if not all_divs:
            return TechnicalSignal(
                indicator="Divergence",
                value=0.0,
                signal=SignalStrength.NEUTRAL,
                description="No divergence detected",
            )

        score = 0.0
        reasons = []

        for div in all_divs:
            weight = 1.0 if "regular" in div.divergence_type else 0.6  # regular > hidden
            if div.is_bullish:
                score += div.confidence * weight
                reasons.append(f"✚ {div.description}")
            else:
                score -= div.confidence * weight
                reasons.append(f"✖ {div.description}")

        # Bonus: RSI + MACD both agree
        rsi_bull  = sum(1 for d in rsi_divs  if d.is_bullish)
        rsi_bear  = sum(1 for d in rsi_divs  if not d.is_bullish)
        macd_bull = sum(1 for d in macd_divs if d.is_bullish)
        macd_bear = sum(1 for d in macd_divs if not d.is_bullish)

        if rsi_bull and macd_bull:
            score += 0.5
            reasons.append("RSI + MACD both show bullish divergence — HIGH CONVICTION")
        if rsi_bear and macd_bear:
            score -= 0.5
            reasons.append("RSI + MACD both show bearish divergence — HIGH CONVICTION")

        if score >= 1.5:
            sig = SignalStrength.STRONG_BUY
        elif score >= 0.4:
            sig = SignalStrength.BUY
        elif score <= -1.5:
            sig = SignalStrength.STRONG_SELL
        elif score <= -0.4:
            sig = SignalStrength.SELL
        else:
            sig = SignalStrength.NEUTRAL

        desc = " | ".join(reasons[:4]) if reasons else "Divergence detected"
        return TechnicalSignal(indicator="Divergence", value=score, signal=sig, description=desc)

    # ------------------------------------------------------------------
    # Indicator helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _macd_histogram(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line - signal_line
