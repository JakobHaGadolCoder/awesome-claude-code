"""
Order Flow Analyzer
Detects unusual options activity, dark pool prints, put/call ratio,
premium flow imbalances, and IV skew to gauge institutional intent.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from options_trader.core.models import (
    OptionContract,
    OptionType,
    OrderFlowData,
    SignalStrength,
    TechnicalSignal,
)
from options_trader.core.config import TradingConfig

logger = logging.getLogger(__name__)


class OrderFlowAnalyzer:
    """
    Analyses options order flow to detect:
    - Unusual volume spikes (vs. rolling average OI and volume)
    - Large block / sweeping orders
    - Put/Call ratio divergences
    - Net premium flow direction
    - IV skew changes (term structure + put/call skew)
    - Dark pool prints (large off-exchange prints flagged separately)
    """

    def __init__(self, config: TradingConfig):
        self.config = config
        self._volume_history: Dict[str, List[int]] = {}   # symbol -> [daily volumes]
        self._flow_history: Dict[str, List[OrderFlowData]] = {}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyze(
        self,
        symbol: str,
        contracts: List[OptionContract],
        dark_pool_prints: Optional[List[Dict]] = None,
    ) -> Tuple[OrderFlowData, TechnicalSignal]:
        """
        Aggregate contract-level data into an OrderFlowData snapshot and
        return a TechnicalSignal summarising the directional bias.
        """
        if not contracts:
            logger.warning("No contracts provided for %s", symbol)
            return self._empty_flow(symbol), self._neutral_signal()

        calls = [c for c in contracts if c.option_type == OptionType.CALL]
        puts = [c for c in contracts if c.option_type == OptionType.PUT]

        call_volume = sum(c.volume for c in calls)
        put_volume = sum(c.volume for c in puts)
        call_premium = sum(c.volume * c.mid_price * 100 for c in calls)
        put_premium = sum(c.volume * c.mid_price * 100 for c in puts)

        unusual_call = self._detect_unusual_volume(symbol, "call", call_volume)
        unusual_put = self._detect_unusual_volume(symbol, "put", put_volume)

        iv_skew = self._compute_iv_skew(calls, puts)
        term_slope = self._compute_term_structure(contracts)

        dp_volume = 0
        dp_bullish = False
        if dark_pool_prints:
            dp_volume = sum(p.get("size", 0) for p in dark_pool_prints)
            bullish_prints = sum(
                p.get("size", 0)
                for p in dark_pool_prints
                if p.get("above_ask", False)
            )
            dp_bullish = bullish_prints > dp_volume * 0.5

        flow = OrderFlowData(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            total_volume=call_volume + put_volume,
            call_volume=call_volume,
            put_volume=put_volume,
            call_premium_spent=call_premium,
            put_premium_spent=put_premium,
            unusual_call_activity=unusual_call,
            unusual_put_activity=unusual_put,
            dark_pool_volume=dp_volume,
            dark_pool_bullish=dp_bullish,
            iv_skew=iv_skew,
            term_structure_slope=term_slope,
        )

        self._store_flow(symbol, flow)

        signal = self._build_signal(flow)
        logger.info(
            "OrderFlow %s | PCR=%.2f | NetPremium=$%.0f | Bias=%s",
            symbol,
            flow.put_call_ratio,
            flow.net_premium_flow,
            signal.signal.name,
        )
        return flow, signal

    # ------------------------------------------------------------------
    # Unusual volume detection
    # ------------------------------------------------------------------

    def _detect_unusual_volume(
        self, symbol: str, side: str, current_volume: int
    ) -> bool:
        key = f"{symbol}_{side}"
        history = self._volume_history.get(key, [])
        if len(history) < 5:
            self._volume_history.setdefault(key, []).append(current_volume)
            return False
        avg = np.mean(history[-20:]) if len(history) >= 20 else np.mean(history)
        self._volume_history[key].append(current_volume)
        return current_volume > avg * self.config.unusual_volume_multiplier

    # ------------------------------------------------------------------
    # Block trade detection
    # ------------------------------------------------------------------

    def detect_block_trades(
        self, contracts: List[OptionContract]
    ) -> List[OptionContract]:
        """Return contracts that qualify as large block trades."""
        blocks = []
        for c in contracts:
            premium = c.volume * c.mid_price * 100
            if (
                c.volume >= self.config.min_block_trade_size
                and premium >= self.config.min_block_premium
            ):
                blocks.append(c)
                logger.info(
                    "BLOCK TRADE detected: %s %s $%.0f strike=%s exp=%s premium=$%.0f",
                    c.symbol,
                    c.option_type.value,
                    c.mid_price,
                    c.strike,
                    c.expiration.date(),
                    premium,
                )
        return blocks

    # ------------------------------------------------------------------
    # IV skew
    # ------------------------------------------------------------------

    def _compute_iv_skew(
        self, calls: List[OptionContract], puts: List[OptionContract]
    ) -> float:
        """
        Put/call IV skew: ATM put IV minus ATM call IV.
        Positive = put skew (bearish fear premium).
        Negative = call skew (bullish greed / upside demand).
        """
        atm_calls = self._get_atm_contracts(calls)
        atm_puts = self._get_atm_contracts(puts)
        if not atm_calls or not atm_puts:
            return 0.0
        avg_call_iv = np.mean([c.implied_volatility for c in atm_calls])
        avg_put_iv = np.mean([c.implied_volatility for c in atm_puts])
        return float(avg_put_iv - avg_call_iv)

    def _get_atm_contracts(
        self, contracts: List[OptionContract], atm_range: float = 0.03
    ) -> List[OptionContract]:
        """Return contracts within atm_range (3%) of the underlying price."""
        result = []
        for c in contracts:
            if c.underlying_price > 0:
                pct_diff = abs(c.strike - c.underlying_price) / c.underlying_price
                if pct_diff <= atm_range:
                    result.append(c)
        return result

    def _compute_term_structure(self, contracts: List[OptionContract]) -> float:
        """
        Term structure slope: near-term IV vs far-term IV.
        Positive = contango (normal). Negative = backwardation (fear spike).
        """
        by_expiry: Dict[datetime, List[float]] = {}
        for c in contracts:
            by_expiry.setdefault(c.expiration, []).append(c.implied_volatility)
        if len(by_expiry) < 2:
            return 0.0
        sorted_expiries = sorted(by_expiry.keys())
        near_iv = np.mean(by_expiry[sorted_expiries[0]])
        far_iv = np.mean(by_expiry[sorted_expiries[-1]])
        return float(far_iv - near_iv)

    # ------------------------------------------------------------------
    # Signal synthesis
    # ------------------------------------------------------------------

    def _build_signal(self, flow: OrderFlowData) -> TechnicalSignal:
        score = 0.0
        reasons = []

        # PCR signal
        pcr = flow.put_call_ratio
        if pcr < self.config.pcr_bullish_threshold:
            score += 1.0
            reasons.append(f"Bullish PCR={pcr:.2f}")
        elif pcr > self.config.pcr_bearish_threshold:
            score -= 1.0
            reasons.append(f"Bearish PCR={pcr:.2f}")

        # Net premium flow
        if flow.net_premium_flow > self.config.min_block_premium:
            score += 1.0
            reasons.append(f"Net call premium +${flow.net_premium_flow:,.0f}")
        elif flow.net_premium_flow < -self.config.min_block_premium:
            score -= 1.0
            reasons.append(f"Net put premium -${abs(flow.net_premium_flow):,.0f}")

        # Unusual activity
        if flow.unusual_call_activity and not flow.unusual_put_activity:
            score += 0.5
            reasons.append("Unusual call volume")
        elif flow.unusual_put_activity and not flow.unusual_call_activity:
            score -= 0.5
            reasons.append("Unusual put volume")

        # Dark pool
        if flow.dark_pool_volume > 0:
            if flow.dark_pool_bullish:
                score += 0.5
                reasons.append("Bullish dark pool prints")
            else:
                score -= 0.5
                reasons.append("Bearish dark pool prints")

        # IV skew
        if flow.iv_skew < -0.05:
            score += 0.5
            reasons.append(f"Call skew (IV skew={flow.iv_skew:.3f})")
        elif flow.iv_skew > 0.10:
            score -= 0.5
            reasons.append(f"Put skew (IV skew={flow.iv_skew:.3f})")

        # Term structure backwardation = fear
        if flow.term_structure_slope < -0.05:
            score -= 0.25
            reasons.append("IV backwardation (fear)")

        signal = self._score_to_signal(score)
        return TechnicalSignal(
            indicator="OrderFlow",
            value=score,
            signal=signal,
            description=" | ".join(reasons) if reasons else "No strong bias",
        )

    @staticmethod
    def _score_to_signal(score: float) -> SignalStrength:
        if score >= 1.5:
            return SignalStrength.STRONG_BUY
        elif score >= 0.5:
            return SignalStrength.BUY
        elif score <= -1.5:
            return SignalStrength.STRONG_SELL
        elif score <= -0.5:
            return SignalStrength.SELL
        return SignalStrength.NEUTRAL

    def _store_flow(self, symbol: str, flow: OrderFlowData) -> None:
        self._flow_history.setdefault(symbol, []).append(flow)
        # Keep last 100 snapshots
        if len(self._flow_history[symbol]) > 100:
            self._flow_history[symbol] = self._flow_history[symbol][-100:]

    def get_flow_history(self, symbol: str) -> List[OrderFlowData]:
        return self._flow_history.get(symbol, [])

    @staticmethod
    def _empty_flow(symbol: str) -> OrderFlowData:
        return OrderFlowData(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            total_volume=0,
            call_volume=0,
            put_volume=0,
            call_premium_spent=0.0,
            put_premium_spent=0.0,
        )

    @staticmethod
    def _neutral_signal() -> TechnicalSignal:
        return TechnicalSignal(
            indicator="OrderFlow",
            value=0.0,
            signal=SignalStrength.NEUTRAL,
            description="Insufficient data",
        )
