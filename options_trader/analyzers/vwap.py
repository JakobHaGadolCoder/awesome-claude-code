"""
Enhanced VWAP Analyzer
Provides:
- Session VWAP (resets each day)
- VWAP Standard Deviation Bands (±1σ, ±2σ, ±3σ)
- VWAP Slope (rising/flat/falling)
- Price vs VWAP position analysis
- VWAP Reclaim / Rejection detection
- VWAP as dynamic support/resistance
- Volume-weighted price distribution context
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from options_trader.core.models import SignalStrength, TechnicalSignal
from options_trader.core.config import TradingConfig

logger = logging.getLogger(__name__)


@dataclass
class VWAPBands:
    vwap: float
    upper_1: float     # VWAP + 1σ
    lower_1: float     # VWAP - 1σ
    upper_2: float     # VWAP + 2σ
    lower_2: float     # VWAP - 2σ
    upper_3: float     # VWAP + 3σ (extreme / blow-off)
    lower_3: float     # VWAP - 3σ (extreme / capitulation)
    slope: float       # rate of change of VWAP (positive = rising)
    slope_pct: float   # slope as % of VWAP price


@dataclass
class VWAPContext:
    """Summary of where price is relative to VWAP."""
    current_price: float
    vwap: float
    deviation_pct: float        # how far price is from VWAP in %
    band_position: str          # "above_2s", "above_1s", "at_vwap", "below_1s", "below_2s", "below_3s", etc.
    vwap_trend: str             # "rising", "falling", "flat"
    is_reclaim: bool            # price just crossed back above VWAP
    is_rejection: bool          # price rejected VWAP from above
    is_extended: bool           # price > 2σ from VWAP (mean reversion candidate)
    distance_to_vwap: float     # absolute pip distance


class VWAPAnalyzer:
    """
    Session-based VWAP with full band analysis and price action signals.
    Designed to work with both intraday (M1–H1) and daily data.
    """

    def __init__(self, config: TradingConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyze(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        session_reset: bool = True,
    ) -> Tuple[VWAPBands, VWAPContext, TechnicalSignal]:
        """
        Args:
            ohlcv: DataFrame [open, high, low, close, volume] with DatetimeIndex
            session_reset: if True, compute session VWAP (reset each day)

        Returns:
            bands     – VWAPBands for the current bar
            context   – VWAPContext summarising price vs VWAP
            signal    – TechnicalSignal with directional bias
        """
        ohlcv = ohlcv.copy()
        ohlcv.columns = [c.lower() for c in ohlcv.columns]

        if session_reset:
            vwap_series, upper1, lower1, upper2, lower2, upper3, lower3 = (
                self._session_vwap_bands(ohlcv)
            )
        else:
            vwap_series, upper1, lower1, upper2, lower2, upper3, lower3 = (
                self._rolling_vwap_bands(ohlcv)
            )

        current_price = float(ohlcv["close"].iloc[-1])
        vwap_now = float(vwap_series.iloc[-1])

        slope = self._compute_slope(vwap_series)
        bands = VWAPBands(
            vwap=vwap_now,
            upper_1=float(upper1.iloc[-1]),
            lower_1=float(lower1.iloc[-1]),
            upper_2=float(upper2.iloc[-1]),
            lower_2=float(lower2.iloc[-1]),
            upper_3=float(upper3.iloc[-1]),
            lower_3=float(lower3.iloc[-1]),
            slope=slope,
            slope_pct=slope / vwap_now if vwap_now != 0 else 0.0,
        )

        context = self._build_context(current_price, bands, vwap_series, ohlcv)
        signal = self._build_signal(context, bands, ohlcv)

        logger.info(
            "VWAP %s | vwap=%.2f | price=%.2f | dev=%.2f%% | band=%s | trend=%s | signal=%s",
            symbol,
            vwap_now,
            current_price,
            context.deviation_pct * 100,
            context.band_position,
            context.vwap_trend,
            signal.signal.name,
        )

        return bands, context, signal

    # ------------------------------------------------------------------
    # Session VWAP (resets at midnight / session open)
    # ------------------------------------------------------------------

    def _session_vwap_bands(
        self, ohlcv: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        tp = (ohlcv["high"] + ohlcv["low"] + ohlcv["close"]) / 3
        vol = ohlcv["volume"]

        # Group by date to reset
        try:
            dates = ohlcv.index.normalize()
        except AttributeError:
            dates = pd.Series(range(len(ohlcv)), index=ohlcv.index)

        cum_tpv = (tp * vol).groupby(dates).cumsum()
        cum_vol = vol.groupby(dates).cumsum()
        vwap = cum_tpv / cum_vol.replace(0, np.nan)

        # Variance for bands
        cum_tp2v = ((tp ** 2) * vol).groupby(dates).cumsum()
        variance = (cum_tp2v / cum_vol.replace(0, np.nan)) - vwap ** 2
        variance = variance.clip(lower=0)
        std = variance.apply(np.sqrt)

        return (
            vwap,
            vwap + std,
            vwap - std,
            vwap + 2 * std,
            vwap - 2 * std,
            vwap + 3 * std,
            vwap - 3 * std,
        )

    # ------------------------------------------------------------------
    # Rolling VWAP (no session reset — useful for longer timeframes)
    # ------------------------------------------------------------------

    def _rolling_vwap_bands(
        self, ohlcv: pd.DataFrame, window: int = 20
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        tp = (ohlcv["high"] + ohlcv["low"] + ohlcv["close"]) / 3
        vol = ohlcv["volume"]

        cum_tpv = (tp * vol).rolling(window).sum()
        cum_vol = vol.rolling(window).sum()
        vwap = cum_tpv / cum_vol.replace(0, np.nan)

        cum_tp2v = ((tp ** 2) * vol).rolling(window).sum()
        variance = (cum_tp2v / cum_vol.replace(0, np.nan)) - vwap ** 2
        variance = variance.clip(lower=0)
        std = variance.apply(np.sqrt)

        return (
            vwap,
            vwap + std,
            vwap - std,
            vwap + 2 * std,
            vwap - 2 * std,
            vwap + 3 * std,
            vwap - 3 * std,
        )

    # ------------------------------------------------------------------
    # Slope
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_slope(vwap_series: pd.Series, lookback: int = 5) -> float:
        """Rate of change of VWAP over last N bars (price units per bar)."""
        clean = vwap_series.dropna()
        if len(clean) < lookback:
            return 0.0
        tail = clean.iloc[-lookback:]
        return float(tail.iloc[-1] - tail.iloc[0]) / lookback

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    def _build_context(
        self,
        current_price: float,
        bands: VWAPBands,
        vwap_series: pd.Series,
        ohlcv: pd.DataFrame,
    ) -> VWAPContext:
        deviation_pct = (current_price - bands.vwap) / bands.vwap if bands.vwap != 0 else 0.0

        # Band position
        if current_price >= bands.upper_3:
            band_pos = "above_3s"
        elif current_price >= bands.upper_2:
            band_pos = "above_2s"
        elif current_price >= bands.upper_1:
            band_pos = "above_1s"
        elif current_price >= bands.vwap:
            band_pos = "above_vwap"
        elif current_price >= bands.lower_1:
            band_pos = "below_vwap"
        elif current_price >= bands.lower_2:
            band_pos = "below_1s"
        elif current_price >= bands.lower_3:
            band_pos = "below_2s"
        else:
            band_pos = "below_3s"

        # VWAP trend
        slope_threshold = bands.vwap * 0.0001
        if bands.slope > slope_threshold:
            vwap_trend = "rising"
        elif bands.slope < -slope_threshold:
            vwap_trend = "falling"
        else:
            vwap_trend = "flat"

        # Reclaim detection: price was below VWAP and crossed back above
        closes = ohlcv["close"].values
        vwap_vals = vwap_series.ffill().values
        is_reclaim = False
        is_rejection = False
        if len(closes) >= 3:
            was_below = closes[-3] < vwap_vals[-3]
            now_above = closes[-1] > vwap_vals[-1]
            was_above = closes[-3] > vwap_vals[-3]
            now_below = closes[-1] < vwap_vals[-1]
            is_reclaim = was_below and now_above
            is_rejection = was_above and now_below

        is_extended = abs(deviation_pct) > 0.02  # >2% from VWAP

        return VWAPContext(
            current_price=current_price,
            vwap=bands.vwap,
            deviation_pct=deviation_pct,
            band_position=band_pos,
            vwap_trend=vwap_trend,
            is_reclaim=is_reclaim,
            is_rejection=is_rejection,
            is_extended=is_extended,
            distance_to_vwap=abs(current_price - bands.vwap),
        )

    # ------------------------------------------------------------------
    # Signal synthesis
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        ctx: VWAPContext,
        bands: VWAPBands,
        ohlcv: pd.DataFrame,
    ) -> TechnicalSignal:
        score = 0.0
        reasons: List[str] = []

        # --- VWAP reclaim / rejection (high priority) ---
        if ctx.is_reclaim:
            score += 1.5
            reasons.append(f"VWAP RECLAIM — price crossed back above {bands.vwap:.2f} (bullish)")
        elif ctx.is_rejection:
            score -= 1.5
            reasons.append(f"VWAP REJECTION — price broke below {bands.vwap:.2f} (bearish)")

        # ================================================================
        # BAND POSITION SCORING — post-mortem fix 31 Mar
        #
        # Previous error: "below_1s" was scored -0.5 (bearish momentum)
        # This caused the system to CONFIRM bearish bias when price was
        # already extended below VWAP — exactly the wrong call at extremes.
        #
        # Fix: At -1σ or lower with an extended move, flip to MEAN REVERSION.
        # The score below -1σ is now BULLISH (snap-back potential).
        # Only score bearish momentum when price first breaks below VWAP
        # (above_vwap → below_vwap transition), not at extremes.
        # ================================================================
        band_scores = {
            "above_3s":  (-1.5, f"Extreme +3σ ({bands.upper_3:.2f}) — STRONG mean-reversion SHORT"),
            "above_2s":  (-1.0, f"Above +2σ ({bands.upper_2:.2f}) — overbought, fade the extension"),
            "above_1s":  (0.5,  f"Above +1σ ({bands.upper_1:.2f}) — bullish momentum"),
            "above_vwap":(0.3,  f"Above VWAP ({bands.vwap:.2f}) — buyers in control"),
            "below_vwap":(-0.3, f"Below VWAP ({bands.vwap:.2f}) — sellers in control"),
            # FIXED: below -1σ is now BULLISH mean-reversion, not bearish momentum
            "below_1s":  (0.5,  f"Below -1σ ({bands.lower_1:.2f}) — oversold vs VWAP, snap-back candidate"),
            "below_2s":  (1.0,  f"Below -2σ ({bands.lower_2:.2f}) — STRONG mean-reversion LONG"),
            "below_3s":  (1.5,  f"Below -3σ ({bands.lower_3:.2f}) — extreme capitulation, HIGH-PROB reversal"),
        }

        if ctx.band_position in band_scores and not ctx.is_reclaim and not ctx.is_rejection:
            s, desc = band_scores[ctx.band_position]
            score += s
            reasons.append(desc)

        # --- VWAP slope (reduced weight when price is extended) ---
        # Post-mortem fix: don't let a falling VWAP override an extreme extension signal
        slope_weight = 0.15 if ctx.is_extended else 0.3
        if ctx.vwap_trend == "rising":
            score += slope_weight
            reasons.append(f"VWAP rising (slope={bands.slope:+.3f}) — bullish drift")
        elif ctx.vwap_trend == "falling":
            score -= slope_weight
            reasons.append(f"VWAP falling (slope={bands.slope:+.3f}) — bearish drift")

        # --- Extension warning (now generates an actionable signal) ---
        if ctx.is_extended:
            sign = "above" if ctx.deviation_pct > 0 else "below"
            ext_pct = abs(ctx.deviation_pct) * 100
            if ctx.deviation_pct < 0:
                # Extended BELOW VWAP → mean reversion LONG signal
                mr_score = min(1.0, ext_pct / 1.5)   # stronger the extension, stronger the signal
                score += mr_score
                reasons.append(
                    f"⚡ MEAN REVERSION SIGNAL: {ext_pct:.1f}% below VWAP "
                    f"(+{mr_score:.2f} score) — snap-back to {bands.vwap:.2f} expected"
                )
            else:
                # Extended ABOVE VWAP → mean reversion SHORT signal
                mr_score = min(1.0, ext_pct / 1.5)
                score -= mr_score
                reasons.append(
                    f"⚡ MEAN REVERSION SIGNAL: {ext_pct:.1f}% above VWAP "
                    f"(-{mr_score:.2f} score) — pullback to {bands.vwap:.2f} expected"
                )

        signal = self._score_to_signal(score)
        desc_str = " | ".join(reasons) if reasons else f"Price at VWAP ({bands.vwap:.2f})"

        return TechnicalSignal(
            indicator="VWAP",
            value=score,
            signal=signal,
            description=desc_str,
        )

    @staticmethod
    def _score_to_signal(score: float) -> SignalStrength:
        if score >= 1.5:
            return SignalStrength.STRONG_BUY
        elif score >= 0.4:
            return SignalStrength.BUY
        elif score <= -1.5:
            return SignalStrength.STRONG_SELL
        elif score <= -0.4:
            return SignalStrength.SELL
        return SignalStrength.NEUTRAL

    # ------------------------------------------------------------------
    # Utility: get VWAP levels as S/R for the current session
    # ------------------------------------------------------------------

    def get_vwap_levels(self, bands: VWAPBands) -> Dict[str, float]:
        """Return all VWAP levels as a dict for use in S/R analysis."""
        return {
            "VWAP":       bands.vwap,
            "VWAP+1σ":    bands.upper_1,
            "VWAP-1σ":    bands.lower_1,
            "VWAP+2σ":    bands.upper_2,
            "VWAP-2σ":    bands.lower_2,
            "VWAP+3σ":    bands.upper_3,
            "VWAP-3σ":    bands.lower_3,
        }
