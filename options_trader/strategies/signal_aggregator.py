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
    - Price action signal (15% of adjusted weight)
    - VWAP signal (10% of adjusted weight)
    - MTF confluence signal (12% of adjusted weight — HTF gate)
    - Divergence signal (8% of adjusted weight)
    - Correlation signal (7% of adjusted weight)
    - Session signal (modifier only — not scored, used in confidence)

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
        price_action_signal: Optional[TechnicalSignal] = None,
        vwap_signal: Optional[TechnicalSignal] = None,
        mtf_signal: Optional[TechnicalSignal] = None,
        divergence_signal: Optional[TechnicalSignal] = None,
        correlation_signal: Optional[TechnicalSignal] = None,
        session_signal: Optional[TechnicalSignal] = None,
        additional_signals: Optional[List[TechnicalSignal]] = None,
    ) -> AggregatedSignal:
        """
        Returns a single AggregatedSignal with direction and confidence.
        """
        weights = self._get_regime_adjusted_weights(
            regime,
            has_price_action=price_action_signal is not None,
            has_vwap=vwap_signal is not None,
            has_mtf=mtf_signal is not None,
            has_divergence=divergence_signal is not None,
            has_correlation=correlation_signal is not None,
        )

        components: Dict[str, TechnicalSignal] = {
            "order_flow": order_flow_signal,
            "sr_levels":  sr_signal,
            "events":     event_signal,
        }
        if price_action_signal:
            components["price_action"] = price_action_signal
        if vwap_signal:
            components["vwap"] = vwap_signal
        if mtf_signal:
            components["mtf"] = mtf_signal
        if divergence_signal:
            components["divergence"] = divergence_signal
        if correlation_signal:
            components["correlation"] = correlation_signal
        if session_signal:
            components["session"] = session_signal

        weighted_score = (
            weights["order_flow"] * order_flow_signal.signal.value
            + weights["technical"]  * technical_score
            + weights["sr_levels"]  * sr_signal.signal.value
            + weights["events"]     * event_signal.signal.value
        )

        # Price action: high-weight confluence signal
        if price_action_signal:
            weighted_score += weights.get("price_action", 0.15) * price_action_signal.signal.value

        # VWAP: reclaim/rejection events are very high conviction
        if vwap_signal:
            vwap_weight = weights.get("vwap", 0.10)
            if "RECLAIM" in vwap_signal.description or "REJECTION" in vwap_signal.description:
                vwap_weight *= 2.0
            weighted_score += vwap_weight * vwap_signal.signal.value

        # MTF: HTF bias alignment — apply full weight; neutral MTF = no boost/penalty
        if mtf_signal:
            mtf_weight = weights.get("mtf", 0.12)
            weighted_score += mtf_weight * mtf_signal.value

        # Divergence: counter-trend or continuation signal
        if divergence_signal and divergence_signal.signal != SignalStrength.NEUTRAL:
            div_weight = weights.get("divergence", 0.08)
            weighted_score += div_weight * divergence_signal.signal.value

        # Correlation: DXY headwind/tailwind
        if correlation_signal and correlation_signal.signal != SignalStrength.NEUTRAL:
            corr_weight = weights.get("correlation", 0.07)
            weighted_score += corr_weight * correlation_signal.signal.value

        # Additional signals averaged in with small weight
        if additional_signals:
            extra_score = sum(s.signal.value for s in additional_signals) / len(additional_signals)
            weighted_score = weighted_score * 0.92 + extra_score * 0.08

        confidence = self._compute_confidence(
            order_flow_signal, technical_score, sr_signal, event_signal, regime,
            price_action_signal=price_action_signal,
            vwap_signal=vwap_signal,
            mtf_signal=mtf_signal,
            divergence_signal=divergence_signal,
            correlation_signal=correlation_signal,
            session_signal=session_signal,
        )
        direction = self._score_to_signal(weighted_score)
        rationale = self._build_rationale(
            components, technical_score, weighted_score, confidence, regime
        )

        logger.info(
            "Aggregated %s | score=%.3f | confidence=%.2f | direction=%s | regime=%s | "
            "PA=%s | VWAP=%s | MTF=%s | Div=%s | Corr=%s",
            symbol,
            weighted_score,
            confidence,
            direction.name,
            regime.value,
            price_action_signal.signal.name if price_action_signal else "N/A",
            vwap_signal.signal.name if vwap_signal else "N/A",
            mtf_signal.signal.name if mtf_signal else "N/A",
            divergence_signal.signal.name if divergence_signal else "N/A",
            correlation_signal.signal.name if correlation_signal else "N/A",
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
        self,
        regime: MarketRegime,
        has_price_action: bool = False,
        has_vwap: bool = False,
        has_mtf: bool = False,
        has_divergence: bool = False,
        has_correlation: bool = False,
    ) -> Dict[str, float]:
        base = {
            "order_flow": self.config.weight_order_flow,
            "technical":  self.config.weight_technical,
            "sr_levels":  self.config.weight_sr_levels,
            "events":     self.config.weight_events,
        }

        # Carve out budget for each new signal source
        allocations = []
        if has_price_action:
            allocations.append(("price_action", 0.15))
        if has_vwap:
            allocations.append(("vwap", 0.10))
        if has_mtf:
            allocations.append(("mtf", 0.12))
        if has_divergence:
            allocations.append(("divergence", 0.08))
        if has_correlation:
            allocations.append(("correlation", 0.07))

        for name, w in allocations:
            base = {k: v * (1 - w) for k, v in base.items()}
            base[name] = w

        # Regime adjustments
        if regime == MarketRegime.HIGH_VOLATILITY:
            base["order_flow"] = min(0.45, base["order_flow"] * 1.3)
            base["technical"]  = max(0.10, base["technical"]  * 0.7)
            if "vwap" in base:
                base["vwap"] = min(0.20, base["vwap"] * 1.5)
        elif regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
            base["technical"] = min(0.40, base["technical"] * 1.2)
            base["sr_levels"] = max(0.08, base["sr_levels"] * 0.9)
            if "price_action" in base:
                base["price_action"] = min(0.25, base["price_action"] * 1.3)
            if "mtf" in base:
                # MTF more reliable in trends
                base["mtf"] = min(0.20, base["mtf"] * 1.25)
        elif regime == MarketRegime.RANGING:
            base["sr_levels"] = min(0.35, base["sr_levels"] * 1.5)
            base["technical"] = max(0.10, base["technical"] * 0.8)
            if "vwap" in base:
                base["vwap"] = min(0.20, base["vwap"] * 1.4)
            if "divergence" in base:
                # Divergence especially useful in ranges (mean-reversion signal)
                base["divergence"] = min(0.15, base["divergence"] * 1.5)

        # Normalise to sum to 1.0
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
        price_action_signal: Optional[TechnicalSignal] = None,
        vwap_signal: Optional[TechnicalSignal] = None,
        mtf_signal: Optional[TechnicalSignal] = None,
        divergence_signal: Optional[TechnicalSignal] = None,
        correlation_signal: Optional[TechnicalSignal] = None,
        session_signal: Optional[TechnicalSignal] = None,
    ) -> float:
        signals_numeric = [
            order_flow.signal.value,
            technical_score,
            sr.signal.value,
            events.signal.value,
        ]
        if price_action_signal:
            signals_numeric.append(price_action_signal.signal.value)
        if vwap_signal:
            signals_numeric.append(vwap_signal.signal.value)
        if mtf_signal:
            signals_numeric.append(mtf_signal.value)
        if divergence_signal and divergence_signal.signal != SignalStrength.NEUTRAL:
            signals_numeric.append(divergence_signal.signal.value)
        if correlation_signal and correlation_signal.signal != SignalStrength.NEUTRAL:
            signals_numeric.append(correlation_signal.signal.value)

        bullish = sum(1 for s in signals_numeric if s > 0.25)
        bearish = sum(1 for s in signals_numeric if s < -0.25)
        agreement = max(bullish, bearish) / len(signals_numeric)

        # VWAP reclaim/rejection = very high conviction
        if vwap_signal and (
            "RECLAIM" in vwap_signal.description or "REJECTION" in vwap_signal.description
        ):
            agreement = min(1.0, agreement + 0.15)

        # MTF confluence alignment boosts confidence
        if mtf_signal and mtf_signal.signal not in (SignalStrength.NEUTRAL,):
            if abs(mtf_signal.value) >= 1.0:
                agreement = min(1.0, agreement + 0.10)

        # RSI + MACD dual divergence = very high conviction
        if divergence_signal and "RSI + MACD both" in divergence_signal.description:
            agreement = min(1.0, agreement + 0.12)

        base_confidence = 0.3 + 0.5 * agreement

        # Session quality modifier
        if session_signal:
            liq = session_signal.value / 2.0  # normalise back to 0–1
            if liq < 0.35:
                base_confidence *= 0.7   # low liquidity = lower confidence
            elif liq >= 0.90:
                base_confidence = min(1.0, base_confidence * 1.08)

        # Event risk penalty
        if abs(events.signal.value) >= 1.0:
            base_confidence *= 0.6
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
        for key in ("price_action", "vwap", "mtf", "divergence", "correlation"):
            if key in components:
                sig = components[key]
                parts.append(f"{key.upper()}: {sig.signal.name} — {sig.description[:80]}")
        return " | ".join(parts)
