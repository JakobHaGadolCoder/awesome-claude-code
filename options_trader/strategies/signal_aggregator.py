"""
Signal Aggregator
Combines signals from all analyzers into a single composite directional
score and confidence level using configurable weights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from options_trader.core.models import (
    MarketRegime,
    SignalStrength,
    TechnicalSignal,
)
from options_trader.core.config import TradingConfig

logger = logging.getLogger(__name__)


@dataclass
class AggregatedSignal:
    symbol: str
    composite_score: float      # weighted average of all signals, -2 to +2
    confidence: float           # 0.0 - 1.0
    direction: SignalStrength
    regime: MarketRegime
    component_signals: Dict[str, TechnicalSignal]
    rationale: str


class SignalAggregator:
    """
    Combines:
    - Order flow signal (weight: config.weight_order_flow)
    - Technical indicators composite (weight: config.weight_technical)
    - Support/Resistance signal (weight: config.weight_sr_levels)
    - Events signal (weight: config.weight_events)

    Applies regime-based adjustments to weights and overall confidence.
    """

    def __init__(self, config: TradingConfig):
        self.config = config

    def aggregate(
        self,
        symbol: str,
        order_flow_signal: TechnicalSignal,
        technical_score: float,
        sr_signal: TechnicalSignal,
        event_signal: TechnicalSignal,
        regime: MarketRegime,
        additional_signals: Optional[List[TechnicalSignal]] = None,
    ) -> AggregatedSignal:
        """
        Returns a single AggregatedSignal with direction and confidence.
        """
        weights = self._get_regime_adjusted_weights(regime)

        components = {
            "order_flow": order_flow_signal,
            "sr_levels": sr_signal,
            "events": event_signal,
        }

        weighted_score = (
            weights["order_flow"] * order_flow_signal.signal.value
            + weights["technical"] * technical_score
            + weights["sr_levels"] * sr_signal.signal.value
            + weights["events"] * event_signal.signal.value
        )

        # Additional signals averaged in with small weight
        if additional_signals:
            extra_score = sum(s.signal.value for s in additional_signals) / len(additional_signals)
            weighted_score = weighted_score * 0.9 + extra_score * 0.1

        confidence = self._compute_confidence(
            order_flow_signal, technical_score, sr_signal, event_signal, regime
        )
        direction = self._score_to_signal(weighted_score)
        rationale = self._build_rationale(
            components, technical_score, weighted_score, confidence, regime
        )

        logger.info(
            "Aggregated %s | score=%.3f | confidence=%.2f | direction=%s | regime=%s",
            symbol,
            weighted_score,
            confidence,
            direction.name,
            regime.value,
        )
        return AggregatedSignal(
            symbol=symbol,
            composite_score=weighted_score,
            confidence=confidence,
            direction=direction,
            regime=regime,
            component_signals=components,
            rationale=rationale,
        )

    # ------------------------------------------------------------------
    # Regime-adjusted weights
    # ------------------------------------------------------------------

    def _get_regime_adjusted_weights(
        self, regime: MarketRegime
    ) -> Dict[str, float]:
        base = {
            "order_flow": self.config.weight_order_flow,
            "technical": self.config.weight_technical,
            "sr_levels": self.config.weight_sr_levels,
            "events": self.config.weight_events,
        }

        if regime == MarketRegime.HIGH_VOLATILITY:
            # In high-vol: order flow more important, technical less reliable
            base["order_flow"] = min(0.45, base["order_flow"] * 1.3)
            base["technical"] = max(0.15, base["technical"] * 0.7)
        elif regime == MarketRegime.TRENDING_UP or regime == MarketRegime.TRENDING_DOWN:
            # In trends: technical momentum + EMA signals are more reliable
            base["technical"] = min(0.45, base["technical"] * 1.2)
            base["sr_levels"] = max(0.10, base["sr_levels"] * 0.9)
        elif regime == MarketRegime.RANGING:
            # In ranges: S/R levels are king
            base["sr_levels"] = min(0.40, base["sr_levels"] * 1.5)
            base["technical"] = max(0.15, base["technical"] * 0.8)

        # Normalise weights to sum to 1.0
        total = sum(base.values())
        return {k: v / total for k, v in base.items()}

    # ------------------------------------------------------------------
    # Confidence calculation
    # ------------------------------------------------------------------

    def _compute_confidence(
        self,
        order_flow: TechnicalSignal,
        technical_score: float,
        sr: TechnicalSignal,
        events: TechnicalSignal,
        regime: MarketRegime,
    ) -> float:
        """
        Confidence is higher when:
        - Multiple signals agree on direction
        - Event risk is low
        - Regime is trending (not high-vol or ranging)
        """
        signals_numeric = [
            order_flow.signal.value,
            technical_score,
            sr.signal.value,
            events.signal.value,
        ]

        # Count agreeing signals (same sign)
        bullish = sum(1 for s in signals_numeric if s > 0.25)
        bearish = sum(1 for s in signals_numeric if s < -0.25)
        agreement = max(bullish, bearish) / len(signals_numeric)

        base_confidence = 0.3 + 0.5 * agreement

        # Event risk penalty
        if abs(events.signal.value) >= 1.0:
            base_confidence *= 0.6  # binary event upcoming
        elif abs(events.signal.value) >= 0.3:
            base_confidence *= 0.85

        # Regime adjustments
        if regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
            base_confidence = min(1.0, base_confidence * 1.1)
        elif regime == MarketRegime.HIGH_VOLATILITY:
            base_confidence = min(1.0, base_confidence * 0.85)

        # Penalise contradicting strong signals
        if bullish >= 2 and bearish >= 2:
            base_confidence *= 0.7

        return round(min(1.0, max(0.0, base_confidence)), 3)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_to_signal(score: float) -> SignalStrength:
        if score >= 1.0:
            return SignalStrength.STRONG_BUY
        elif score >= 0.3:
            return SignalStrength.BUY
        elif score <= -1.0:
            return SignalStrength.STRONG_SELL
        elif score <= -0.3:
            return SignalStrength.SELL
        return SignalStrength.NEUTRAL

    def _build_rationale(
        self,
        components: Dict[str, TechnicalSignal],
        technical_score: float,
        composite: float,
        confidence: float,
        regime: MarketRegime,
    ) -> str:
        parts = [
            f"Regime: {regime.value}",
            f"Composite score: {composite:.2f}",
            f"Confidence: {confidence*100:.0f}%",
            f"Order Flow: {components['order_flow'].signal.name} — {components['order_flow'].description}",
            f"Technical: score={technical_score:.2f}",
            f"S/R: {components['sr_levels'].signal.name} — {components['sr_levels'].description}",
            f"Events: {components['events'].signal.name} — {components['events'].description}",
        ]
        return " | ".join(parts)
