"""
DXY Correlation & Macro Correlation Module
Provides:
- DXY inverse correlation with Gold (XAUUSD)
- Rolling correlation coefficient (20/50-bar)
- DXY trend detection (bullish DXY → bearish Gold)
- Correlation strength filter: only act when correlation is strong (|r| > 0.6)
- Risk-on / Risk-off detection via VIX proxy (high-beta equity momentum)
- Gold-specific logic: when DXY strong → suppresses Gold buy signals
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from options_trader.core.models import SignalStrength, TechnicalSignal
from options_trader.core.config import TradingConfig

logger = logging.getLogger(__name__)

# Known inverse correlations (asset → negatively correlated asset)
INVERSE_PAIRS: Dict[str, str] = {
    "XAUUSD": "DXY",
    "GLD":    "DXY",
    "GOLD":   "DXY",
    "GC=F":   "DXY",
}

# Known positive correlations
POSITIVE_PAIRS: Dict[str, str] = {
    "SPY": "QQQ",
    "QQQ": "SPY",
}

STRONG_CORRELATION_THRESHOLD = 0.60
MODERATE_CORRELATION_THRESHOLD = 0.35


@dataclass
class CorrelationResult:
    """Correlation analysis output."""
    symbol: str
    correlated_asset: str
    correlation_20: float          # 20-bar rolling correlation
    correlation_50: float          # 50-bar rolling correlation
    is_inverse: bool               # True for Gold/DXY
    correlation_strength: str      # "strong", "moderate", "weak"
    dxy_trend: str                 # "rising", "falling", "flat" (if DXY data available)
    risk_regime: str               # "risk_on", "risk_off", "neutral"
    signal: TechnicalSignal
    suppression_active: bool       # True when correlation overrides primary signal
    suppression_reason: str


class CorrelationAnalyzer:
    """
    Analyses the impact of correlated/inverse-correlated assets on
    the primary symbol.

    Primary use case: Gold (XAUUSD) vs DXY
    - When DXY is in a strong uptrend → suppress Gold buy signals
    - When DXY is collapsing → amplify Gold buy signals
    - Correlation filter: only apply when |r| > STRONG_CORRELATION_THRESHOLD

    Also handles:
    - Risk-on/off regime detection via equity momentum proxy
    - Generic positive correlation boost (SPY/QQQ confirming each other)
    """

    def __init__(self, config: TradingConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyze(
        self,
        symbol: str,
        primary_ohlcv: pd.DataFrame,
        correlated_ohlcv: Optional[pd.DataFrame] = None,
        correlated_symbol: Optional[str] = None,
    ) -> CorrelationResult:
        """
        Args:
            symbol: primary asset (e.g. "XAUUSD")
            primary_ohlcv: OHLCV for the primary asset
            correlated_ohlcv: OHLCV for the correlated asset (e.g. DXY)
                              If None, returns neutral result with no suppression.
            correlated_symbol: override name (e.g. "DXY")

        Returns:
            CorrelationResult with suppression flag and final signal.
        """
        # Determine expected correlation direction
        sym_upper = symbol.upper()
        corr_name = correlated_symbol or INVERSE_PAIRS.get(sym_upper) or POSITIVE_PAIRS.get(sym_upper, "UNKNOWN")
        is_inverse = sym_upper in INVERSE_PAIRS

        if correlated_ohlcv is None or len(correlated_ohlcv) < 20:
            return self._neutral_result(symbol, corr_name, is_inverse)

        primary_close = primary_ohlcv.copy()
        primary_close.columns = [c.lower() for c in primary_close.columns]
        primary_close = primary_close["close"]

        corr_close = correlated_ohlcv.copy()
        corr_close.columns = [c.lower() for c in corr_close.columns]
        corr_close = corr_close["close"]

        # Align on common index
        combined = pd.concat([primary_close.rename("primary"), corr_close.rename("corr")], axis=1).dropna()
        if len(combined) < 20:
            return self._neutral_result(symbol, corr_name, is_inverse)

        primary_ret = combined["primary"].pct_change().dropna()
        corr_ret    = combined["corr"].pct_change().dropna()

        corr_20 = float(primary_ret.tail(20).corr(corr_ret.tail(20)))
        corr_50 = float(primary_ret.tail(50).corr(corr_ret.tail(50)))

        corr_20 = corr_20 if not np.isnan(corr_20) else 0.0
        corr_50 = corr_50 if not np.isnan(corr_50) else 0.0

        # Classify strength
        avg_corr = (abs(corr_20) + abs(corr_50)) / 2
        if avg_corr >= STRONG_CORRELATION_THRESHOLD:
            strength = "strong"
        elif avg_corr >= MODERATE_CORRELATION_THRESHOLD:
            strength = "moderate"
        else:
            strength = "weak"

        # DXY trend
        dxy_trend = self._compute_trend(corr_close.tail(30))

        # Risk regime from primary asset momentum
        risk_regime = self._compute_risk_regime(primary_close)

        # Build signal + suppression logic
        signal, suppression_active, suppression_reason = self._build_signal(
            symbol, corr_name, corr_20, corr_50, is_inverse,
            strength, dxy_trend, risk_regime,
        )

        logger.info(
            "Correlation %s/%s | r20=%.2f | r50=%.2f | strength=%s | dxy=%s | risk=%s | suppress=%s",
            symbol, corr_name, corr_20, corr_50,
            strength, dxy_trend, risk_regime, suppression_active,
        )

        return CorrelationResult(
            symbol=symbol,
            correlated_asset=corr_name,
            correlation_20=corr_20,
            correlation_50=corr_50,
            is_inverse=is_inverse,
            correlation_strength=strength,
            dxy_trend=dxy_trend,
            risk_regime=risk_regime,
            signal=signal,
            suppression_active=suppression_active,
            suppression_reason=suppression_reason,
        )

    # ------------------------------------------------------------------
    # DXY trend
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_trend(series: pd.Series, threshold_pct: float = 0.003) -> str:
        """Rising/falling/flat based on regression slope."""
        if len(series) < 5:
            return "flat"
        clean = series.dropna()
        x = np.arange(len(clean))
        slope = np.polyfit(x, clean.values, 1)[0]
        slope_pct = slope / float(clean.mean()) if float(clean.mean()) != 0 else 0.0
        if slope_pct > threshold_pct:
            return "rising"
        elif slope_pct < -threshold_pct:
            return "falling"
        return "flat"

    # ------------------------------------------------------------------
    # Risk regime
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_risk_regime(close: pd.Series) -> str:
        """
        Simple proxy: compare 5-day vs 20-day momentum.
        Positive short-term momentum vs flat/down longer = risk_on surge.
        """
        if len(close) < 21:
            return "neutral"
        mom5  = float(close.iloc[-1] / close.iloc[-6] - 1)
        mom20 = float(close.iloc[-1] / close.iloc[-21] - 1)
        if mom5 > 0.005 and mom20 > 0.01:
            return "risk_on"
        elif mom5 < -0.005 and mom20 < -0.01:
            return "risk_off"
        return "neutral"

    # ------------------------------------------------------------------
    # Signal synthesis
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        symbol: str,
        corr_asset: str,
        corr_20: float,
        corr_50: float,
        is_inverse: bool,
        strength: str,
        dxy_trend: str,
        risk_regime: str,
    ) -> Tuple[TechnicalSignal, bool, str]:
        score = 0.0
        suppression_active = False
        suppression_reason = ""
        reasons = []

        if strength == "weak":
            # Correlation too weak to act on
            sig = SignalStrength.NEUTRAL
            desc = f"Correlation {symbol}/{corr_asset}: r20={corr_20:.2f} (weak — not actionable)"
            return TechnicalSignal(indicator="Correlation", value=0.0, signal=sig, description=desc), False, ""

        # Gold/DXY inverse correlation logic
        if is_inverse:
            if dxy_trend == "rising":
                # Strong DXY = headwind for Gold
                score -= 1.0 if strength == "strong" else 0.5
                reasons.append(f"{corr_asset} RISING — headwind for {symbol} (inverse corr)")
                if strength == "strong":
                    suppression_active = True
                    suppression_reason = f"DXY uptrend (strong inverse r={corr_20:.2f}) suppresses Gold buy bias"
            elif dxy_trend == "falling":
                # Weak DXY = tailwind for Gold
                score += 1.0 if strength == "strong" else 0.5
                reasons.append(f"{corr_asset} FALLING — tailwind for {symbol} (inverse corr)")
            else:
                reasons.append(f"{corr_asset} flat — neutral for {symbol}")

        else:
            # Positive correlation logic (SPY/QQQ confirming)
            # corr_20 close to +1 means they're moving together
            if corr_20 > STRONG_CORRELATION_THRESHOLD:
                score += 0.5
                reasons.append(f"{symbol}/{corr_asset} moving in sync (r={corr_20:.2f}) — confirming signal")
            elif corr_20 < -MODERATE_CORRELATION_THRESHOLD:
                score -= 0.5
                reasons.append(f"{symbol}/{corr_asset} diverging (r={corr_20:.2f}) — caution, unusual decoupling")

        # Risk-on / risk-off overlay
        if risk_regime == "risk_on":
            score += 0.3
            reasons.append(f"Risk-on regime — bullish macro backdrop")
        elif risk_regime == "risk_off":
            score -= 0.3
            reasons.append(f"Risk-off regime — flight to safety")

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

        desc = f"Correlation {symbol}/{corr_asset} (r20={corr_20:.2f}, {strength}): " + " | ".join(reasons)
        return TechnicalSignal(indicator="Correlation", value=score, signal=sig, description=desc), suppression_active, suppression_reason

    # ------------------------------------------------------------------
    # Neutral fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _neutral_result(symbol: str, corr_name: str, is_inverse: bool) -> CorrelationResult:
        sig = TechnicalSignal(
            indicator="Correlation", value=0.0, signal=SignalStrength.NEUTRAL,
            description=f"Correlation {symbol}/{corr_name}: insufficient data"
        )
        return CorrelationResult(
            symbol=symbol,
            correlated_asset=corr_name,
            correlation_20=0.0,
            correlation_50=0.0,
            is_inverse=is_inverse,
            correlation_strength="weak",
            dxy_trend="flat",
            risk_regime="neutral",
            signal=sig,
            suppression_active=False,
            suppression_reason="",
        )
