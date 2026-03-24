"""
Main Trading Engine
Orchestrates all analyzers and generates live trade signals.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from options_trader.core.config import TradingConfig
from options_trader.core.models import (
    MarketEvent,
    OptionContract,
    TradeSignal,
)
from options_trader.analyzers.order_flow import OrderFlowAnalyzer
from options_trader.analyzers.technical import TechnicalAnalyzer
from options_trader.analyzers.support_resistance import SupportResistanceAnalyzer
from options_trader.analyzers.events import EventsAnalyzer
from options_trader.analyzers.options_greeks import OptionsGreeksCalculator
from options_trader.strategies.signal_aggregator import SignalAggregator
from options_trader.strategies.trade_selector import TradeSelector
from options_trader.utils.data_fetcher import DataFetcher
from options_trader.utils.logger import setup_logger

logger = logging.getLogger(__name__)


class TradingEngine:
    """
    Full analysis pipeline:
    1. Fetch OHLCV + options chain + news
    2. Run order flow analysis
    3. Run technical analysis
    4. Detect S/R levels
    5. Parse events/news
    6. Aggregate signals
    7. Select trade (if valid)
    """

    def __init__(self, config: Optional[TradingConfig] = None):
        self.config = config or TradingConfig()
        self.config.validate()

        setup_logger("options_trader", self.config.log_level, self.config.log_file)

        self.data = DataFetcher(self.config)
        self.flow_analyzer = OrderFlowAnalyzer(self.config)
        self.tech_analyzer = TechnicalAnalyzer(self.config)
        self.sr_analyzer = SupportResistanceAnalyzer(self.config)
        self.events_analyzer = EventsAnalyzer(self.config)
        self.greeks_calc = OptionsGreeksCalculator()
        self.aggregator = SignalAggregator(self.config)
        self.selector = TradeSelector(self.config)

    # ------------------------------------------------------------------
    # Single symbol analysis
    # ------------------------------------------------------------------

    def analyze_symbol(
        self,
        symbol: str,
        events: Optional[List[MarketEvent]] = None,
    ) -> Optional[TradeSignal]:
        """
        Full pipeline for a single symbol. Returns a TradeSignal or None.
        """
        logger.info("=" * 60)
        logger.info("Analyzing %s", symbol)
        logger.info("=" * 60)

        # 1. Fetch data
        try:
            ohlcv = self.data.fetch_ohlcv(symbol, days=120)
            chain = self.data.fetch_options_chain(symbol)
            news = self.data.fetch_news(symbol)
        except Exception as exc:
            logger.error("Data fetch error for %s: %s", symbol, exc)
            return None

        current_price = float(ohlcv["close"].iloc[-1])
        logger.info("Current price: $%.2f | Chain size: %d contracts", current_price, len(chain))

        # 2. Order flow analysis
        flow_data, flow_signal = self.flow_analyzer.analyze(symbol, chain)
        block_trades = self.flow_analyzer.detect_block_trades(chain)
        if block_trades:
            logger.info("BLOCK TRADES: %d detected for %s", len(block_trades), symbol)

        # 3. Technical analysis
        tech_signals, regime, tech_score = self.tech_analyzer.analyze(symbol, ohlcv)

        # 4. Support / Resistance
        sr_levels, sr_signal = self.sr_analyzer.analyze(symbol, ohlcv, current_price, chain)

        # 5. Events + news
        upcoming_events, event_signal = self.events_analyzer.analyze(
            symbol, events=events, news_headlines=news
        )

        # 6. Greeks on chain
        greeks_map = self.greeks_calc.analyze_chain(chain)
        logger.debug("Computed Greeks for %d contracts", len(greeks_map))

        # 7. Aggregate
        agg = self.aggregator.aggregate(
            symbol=symbol,
            order_flow_signal=flow_signal,
            technical_score=tech_score,
            sr_signal=sr_signal,
            event_signal=event_signal,
            regime=regime,
            additional_signals=tech_signals,
        )

        # 8. Select trade
        trade = self.selector.select_trade(
            agg, chain, self.config.paper_trading_capital
        )

        if trade:
            self._print_trade_signal(trade, current_price)
        else:
            logger.info("No valid trade signal for %s", symbol)

        return trade

    # ------------------------------------------------------------------
    # Multi-symbol scan
    # ------------------------------------------------------------------

    def scan_universe(
        self, events: Optional[List[MarketEvent]] = None
    ) -> Dict[str, Optional[TradeSignal]]:
        """Run analysis on all configured symbols and return results."""
        results = {}
        for symbol in self.config.symbols:
            results[symbol] = self.analyze_symbol(symbol, events=events)
        return results

    # ------------------------------------------------------------------
    # Backtesting
    # ------------------------------------------------------------------

    def backtest(
        self,
        symbol: str,
        days: int = 252,
    ):
        """Run backtesting on historical data."""
        from options_trader.backtesting.backtester import Backtester
        ohlcv = self.data.fetch_ohlcv(symbol, days=days)
        iv_series = self.data.fetch_historical_iv(symbol, ohlcv)
        backtester = Backtester(self.config)
        result = backtester.run(symbol, ohlcv, historical_iv=iv_series)
        print(result.summary())
        return result

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _print_trade_signal(self, trade: TradeSignal, current_price: float) -> None:
        dte = (trade.expiration - datetime.utcnow()).days
        print("\n" + "=" * 60)
        print(f"  TRADE SIGNAL: {trade.action} {trade.symbol} {trade.option_type.value}")
        print("=" * 60)
        print(f"  Strike:      ${trade.strike:.0f}  ({trade.option_type.value})")
        print(f"  Expiration:  {trade.expiration.date()}  ({dte} DTE)")
        print(f"  Entry:       ${trade.entry_price:.2f} per contract")
        print(f"  Target:      ${trade.target_price:.2f}  (+{(trade.target_price/trade.entry_price-1)*100:.0f}%)")
        print(f"  Stop Loss:   ${trade.stop_loss:.2f}  (-{(1-trade.stop_loss/trade.entry_price)*100:.0f}%)")
        print(f"  Max Loss:    ${trade.max_loss:,.0f}")
        print(f"  Max Gain:    ${trade.max_gain:,.0f}")
        print(f"  Risk/Reward: {trade.risk_reward:.1f}x")
        print(f"  Confidence:  {trade.confidence*100:.0f}%")
        print(f"  Strength:    {trade.strength.name}")
        print(f"  Underlying:  ${current_price:.2f}")
        print("-" * 60)
        print(f"  Rationale:")
        for part in trade.rationale.split(" | "):
            print(f"    • {part}")
        print("=" * 60 + "\n")
