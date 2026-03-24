"""
Trade Selector
Translates aggregated signals into specific options trade recommendations.
Selects optimal strike, expiration, and strategy type.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np

from options_trader.core.models import (
    OptionContract,
    OptionType,
    SignalStrength,
    TradeSignal,
)
from options_trader.core.config import TradingConfig
from options_trader.analyzers.options_greeks import OptionsGreeksCalculator
from options_trader.strategies.signal_aggregator import AggregatedSignal

logger = logging.getLogger(__name__)


class TradeSelector:
    """
    Given an AggregatedSignal and an options chain, selects the best
    contract(s) and constructs a TradeSignal with entry/target/stop.
    """

    def __init__(self, config: TradingConfig):
        self.config = config
        self.greeks_calc = OptionsGreeksCalculator()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def select_trade(
        self,
        agg_signal: AggregatedSignal,
        chain: List[OptionContract],
        portfolio_value: float,
    ) -> Optional[TradeSignal]:
        """
        Returns a TradeSignal if a qualifying trade is found, else None.
        """
        if agg_signal.direction == SignalStrength.NEUTRAL:
            logger.info("No trade for %s: neutral signal", agg_signal.symbol)
            return None

        if agg_signal.confidence < self.config.min_confidence:
            logger.info(
                "No trade for %s: confidence %.2f < %.2f",
                agg_signal.symbol,
                agg_signal.confidence,
                self.config.min_confidence,
            )
            return None

        # Filter chain to valid contracts
        filtered = self._filter_chain(chain)
        if not filtered:
            logger.warning("No valid contracts after filtering for %s", agg_signal.symbol)
            return None

        # Determine option type from signal direction
        if agg_signal.direction in (SignalStrength.STRONG_BUY, SignalStrength.BUY):
            option_type = OptionType.CALL
            strategy = "long_call" if agg_signal.direction == SignalStrength.STRONG_BUY else "bull_call_spread"
        else:
            option_type = OptionType.PUT
            strategy = "long_put" if agg_signal.direction == SignalStrength.STRONG_SELL else "bear_put_spread"

        candidates = [c for c in filtered if c.option_type == option_type]
        if not candidates:
            logger.warning("No %s contracts available for %s", option_type.value, agg_signal.symbol)
            return None

        contract = self._select_best_contract(candidates, agg_signal)
        if contract is None:
            return None

        trade_signal = self._build_trade_signal(
            agg_signal, contract, strategy, portfolio_value
        )

        if not trade_signal.is_valid:
            logger.info(
                "Trade rejected for %s: RR=%.2f confidence=%.2f",
                agg_signal.symbol,
                trade_signal.risk_reward,
                trade_signal.confidence,
            )
            return None

        logger.info(
            "TRADE SIGNAL: %s %s %s strike=%.0f exp=%s entry=%.2f target=%.2f stop=%.2f RR=%.1f conf=%.0f%%",
            trade_signal.action,
            trade_signal.symbol,
            trade_signal.option_type.value,
            trade_signal.strike,
            trade_signal.expiration.date(),
            trade_signal.entry_price,
            trade_signal.target_price,
            trade_signal.stop_loss,
            trade_signal.risk_reward,
            trade_signal.confidence * 100,
        )
        return trade_signal

    # ------------------------------------------------------------------
    # Chain filtering
    # ------------------------------------------------------------------

    def _filter_chain(
        self, chain: List[OptionContract]
    ) -> List[OptionContract]:
        now = datetime.utcnow()
        result = []
        for c in chain:
            dte = (c.expiration - now).days
            if dte < self.config.min_dte or dte > self.config.max_dte:
                continue
            if c.open_interest < self.config.min_open_interest:
                continue
            if c.volume < self.config.min_volume:
                continue
            if c.spread_pct > self.config.max_spread_pct:
                continue
            result.append(c)
        return result

    # ------------------------------------------------------------------
    # Contract selection
    # ------------------------------------------------------------------

    def _select_best_contract(
        self,
        candidates: List[OptionContract],
        agg_signal: AggregatedSignal,
    ) -> Optional[OptionContract]:
        """
        Score each candidate by:
        1. Delta within target range (primary filter)
        2. Liquidity score (OI + volume)
        3. DTE target (30-45 DTE optimal for single-leg; 7-21 for spreads)
        4. Spread cost (lower is better)
        """
        scored: List[Tuple[float, OptionContract]] = []

        for c in candidates:
            delta_abs = abs(c.delta)
            delta_min, delta_max = self.config.target_delta_range

            # Prefer delta in target range
            if delta_abs < delta_min or delta_abs > delta_max:
                # Allow slightly OTM for strong signals
                if abs(delta_abs - 0.35) > 0.2:
                    continue

            # Liquidity score
            liq_score = min(1.0, (c.volume / 500 + c.open_interest / 5000) / 2)

            # DTE score (prefer ~30-45 DTE)
            dte = (c.expiration - datetime.utcnow()).days
            dte_target = 35
            dte_score = 1.0 - abs(dte - dte_target) / dte_target

            # Spread cost score (lower spread = higher score)
            spread_score = 1.0 - c.spread_pct / self.config.max_spread_pct

            # Delta score (prefer closer to middle of range)
            delta_mid = (delta_min + delta_max) / 2
            delta_score = 1.0 - abs(delta_abs - delta_mid) / delta_mid

            total = (
                0.35 * liq_score
                + 0.25 * dte_score
                + 0.25 * delta_score
                + 0.15 * spread_score
            )
            scored.append((total, c))

        if not scored:
            return None

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    # ------------------------------------------------------------------
    # Trade signal construction
    # ------------------------------------------------------------------

    def _build_trade_signal(
        self,
        agg_signal: AggregatedSignal,
        contract: OptionContract,
        strategy: str,
        portfolio_value: float,
    ) -> TradeSignal:
        entry = contract.mid_price
        underlying = contract.underlying_price

        # Position sizing
        max_risk_dollar = portfolio_value * self.config.max_position_size_pct
        contracts_qty = max(1, int(max_risk_dollar / (entry * 100)))

        # Target and stop based on delta and signal strength
        multiplier = 1.5 if agg_signal.direction in (SignalStrength.STRONG_BUY, SignalStrength.STRONG_SELL) else 1.0
        target_pct = 0.50 * multiplier   # target 50% gain (strong) or 50% (moderate)
        stop_pct = 0.30                   # stop at 30% loss

        target = round(entry * (1 + target_pct), 2)
        stop = round(entry * (1 - stop_pct), 2)

        max_loss = entry * stop_pct * contracts_qty * 100
        max_gain = entry * target_pct * contracts_qty * 100
        risk_reward = max_gain / max_loss if max_loss > 0 else 0.0

        is_bullish = agg_signal.direction in (SignalStrength.STRONG_BUY, SignalStrength.BUY)

        return TradeSignal(
            symbol=agg_signal.symbol,
            option_type=contract.option_type,
            action="BUY",
            strike=contract.strike,
            expiration=contract.expiration,
            strength=agg_signal.direction,
            confidence=agg_signal.confidence,
            entry_price=entry,
            target_price=target,
            stop_loss=stop,
            max_loss=max_loss,
            max_gain=max_gain,
            risk_reward=risk_reward,
            rationale=agg_signal.rationale,
            order_flow_score=agg_signal.component_signals["order_flow"].value,
            technical_score=agg_signal.composite_score,
            event_score=agg_signal.component_signals["events"].value,
            sr_score=agg_signal.component_signals["sr_levels"].value,
        )
