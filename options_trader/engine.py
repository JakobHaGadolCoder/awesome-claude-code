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
from options_trader.analyzers.price_action import PriceActionAnalyzer
from options_trader.analyzers.vwap import VWAPAnalyzer
from options_trader.analyzers.multi_timeframe import MultiTimeframeAnalyzer
from options_trader.analyzers.divergence import DivergenceDetector
from options_trader.analyzers.correlation import CorrelationAnalyzer
from options_trader.strategies.signal_aggregator import SignalAggregator
from options_trader.strategies.trade_selector import TradeSelector
from options_trader.utils.data_fetcher import DataFetcher
from options_trader.utils.logger import setup_logger
from options_trader.utils.session_filter import SessionFilter

logger = logging.getLogger(__name__)


class TradingEngine:
    """
    Full analysis pipeline:
    1.  Fetch OHLCV + options chain + news
    2.  Session filter (is it worth trading right now?)
    3.  Run order flow analysis
    4.  Run technical analysis
    5.  Price action analysis (structure, patterns, OBs, FVGs)
    6.  VWAP analysis (session VWAP, bands, reclaim/rejection)
    7.  Multi-timeframe confluence (HTF bias gate)
    8.  RSI + MACD divergence detection
    9.  DXY / macro correlation
    10. Detect S/R levels
    11. Parse events/news
    12. Aggregate all signals
    13. Select trade (if valid)
    """

    def __init__(self, config: Optional[TradingConfig] = None):
        self.config = config or TradingConfig()
        self.config.validate()

        setup_logger("options_trader", self.config.log_level, self.config.log_file)

        self.data            = DataFetcher(self.config)
        self.flow_analyzer   = OrderFlowAnalyzer(self.config)
        self.tech_analyzer   = TechnicalAnalyzer(self.config)
        self.sr_analyzer     = SupportResistanceAnalyzer(self.config)
        self.events_analyzer = EventsAnalyzer(self.config)
        self.greeks_calc     = OptionsGreeksCalculator()
        self.pa_analyzer     = PriceActionAnalyzer(self.config)
        self.vwap_analyzer   = VWAPAnalyzer(self.config)
        self.mtf_analyzer    = MultiTimeframeAnalyzer(self.config)
        self.div_detector    = DivergenceDetector(self.config)
        self.corr_analyzer   = CorrelationAnalyzer(self.config)
        self.session_filter  = SessionFilter(self.config)
        self.aggregator      = SignalAggregator(self.config)
        self.selector        = TradeSelector(self.config)

    # ------------------------------------------------------------------
    # Single symbol analysis
    # ------------------------------------------------------------------

    def analyze_symbol(
        self,
        symbol: str,
        events: Optional[List[MarketEvent]] = None,
        ohlcv_by_tf: Optional[Dict[str, pd.DataFrame]] = None,
        correlated_ohlcv: Optional[pd.DataFrame] = None,
    ) -> Optional[TradeSignal]:
        """
        Full pipeline for a single symbol. Returns a TradeSignal or None.

        Args:
            ohlcv_by_tf: dict of {timeframe: DataFrame} for MTF analysis
                         e.g. {"M15": df_m15, "H1": df_h1, "H4": df_h4, "D1": df_d1}
            correlated_ohlcv: OHLCV of correlated asset (e.g. DXY for Gold)
        """
        logger.info("=" * 60)
        logger.info("Analyzing %s", symbol)
        logger.info("=" * 60)

        # 1. Fetch data
        try:
            ohlcv = self.data.fetch_ohlcv(symbol, days=120)
            chain = self.data.fetch_options_chain(symbol)
            news  = self.data.fetch_news(symbol)
        except Exception as exc:
            logger.error("Data fetch error for %s: %s", symbol, exc)
            return None

        current_price = float(ohlcv["close"].iloc[-1])
        logger.info("Current price: $%.2f | Chain size: %d contracts", current_price, len(chain))

        # 2. Session filter
        session_info = self.session_filter.analyze(symbol)
        if not session_info.is_tradeable:
            logger.warning(
                "Session filter BLOCKED %s — %s (liquidity=%.2f)",
                symbol, session_info.session_name, session_info.liquidity_score,
            )
            # Still continue but note the suppression

        # 3. Order flow analysis
        flow_data, flow_signal = self.flow_analyzer.analyze(symbol, chain)
        block_trades = self.flow_analyzer.detect_block_trades(chain)
        if block_trades:
            logger.info("BLOCK TRADES: %d detected for %s", len(block_trades), symbol)

        # 4. Technical analysis
        tech_signals, regime, tech_score = self.tech_analyzer.analyze(symbol, ohlcv)

        # 5. Price action analysis
        pa_result  = self.pa_analyzer.analyze(symbol, ohlcv, current_price)
        pa_signal  = pa_result.signal
        logger.info(
            "PriceAction %s | trend=%s | OBs=%d active | FVGs=%d active | nearest_ob=%s",
            symbol,
            pa_result.structure.trend,
            sum(1 for ob in pa_result.order_blocks if not ob.mitigated),
            sum(1 for fvg in pa_result.fvgs if not fvg.filled),
            pa_result.nearest_ob.description if pa_result.nearest_ob else "none",
        )

        # 6. VWAP analysis
        vwap_bands, vwap_ctx, vwap_signal = self.vwap_analyzer.analyze(symbol, ohlcv)
        logger.info(
            "VWAP %s | vwap=%.2f | band=%s | trend=%s | reclaim=%s | rejection=%s",
            symbol,
            vwap_bands.vwap,
            vwap_ctx.band_position,
            vwap_ctx.vwap_trend,
            vwap_ctx.is_reclaim,
            vwap_ctx.is_rejection,
        )

        # 7. Multi-timeframe confluence
        if ohlcv_by_tf is None:
            # Build MTF from base OHLCV by resampling
            ohlcv_by_tf = self._build_mtf_from_ohlcv(ohlcv)
        mtf_result = self.mtf_analyzer.analyze(symbol, ohlcv_by_tf)
        logger.info(
            "MTF %s | htf_bias=%s | confluence=%.2f | can_trade=%s | ltf_aligned=%s",
            symbol,
            mtf_result.htf_bias,
            mtf_result.confluence_score,
            mtf_result.can_trade,
            mtf_result.ltf_aligned,
        )

        # 8. Divergence detection
        div_result = self.div_detector.analyze(symbol, ohlcv)
        logger.info(
            "Divergence %s | RSI=%d | MACD=%d | signal=%s",
            symbol,
            len(div_result.rsi_divergences),
            len(div_result.macd_divergences),
            div_result.combined_signal.signal.name,
        )

        # 9. DXY / Macro correlation
        corr_result = self.corr_analyzer.analyze(
            symbol, ohlcv, correlated_ohlcv=correlated_ohlcv
        )
        if corr_result.suppression_active:
            logger.warning(
                "CORRELATION SUPPRESSION for %s: %s",
                symbol, corr_result.suppression_reason,
            )

        # 10. Support / Resistance (augment with VWAP levels)
        sr_levels, sr_signal = self.sr_analyzer.analyze(symbol, ohlcv, current_price, chain)

        # 11. Events + news
        upcoming_events, event_signal = self.events_analyzer.analyze(
            symbol, events=events, news_headlines=news
        )

        # 12. Greeks on chain
        greeks_map = self.greeks_calc.analyze_chain(chain)
        logger.debug("Computed Greeks for %d contracts", len(greeks_map))

        # 13. Aggregate all signals (with MTF gate)
        # If MTF is suppressed (LTF contradicts HTF), override to neutral
        if not mtf_result.ltf_aligned and mtf_result.confluence_score > 0:
            logger.warning("MTF suppression active for %s: %s", symbol, mtf_result.suppression_reason)

        agg = self.aggregator.aggregate(
            symbol=symbol,
            order_flow_signal=flow_signal,
            technical_score=tech_score,
            sr_signal=sr_signal,
            event_signal=event_signal,
            regime=regime,
            price_action_signal=pa_signal,
            vwap_signal=vwap_signal,
            mtf_signal=mtf_result.signal,
            divergence_signal=div_result.combined_signal,
            correlation_signal=corr_result.signal,
            session_signal=session_info.signal,
            additional_signals=tech_signals,
        )

        # Hard gates: MTF suppression or correlation suppression → no trade
        if not mtf_result.ltf_aligned or corr_result.suppression_active:
            logger.info(
                "Hard gate triggered for %s — MTF=%s, Corr=%s",
                symbol, mtf_result.suppression_reason, corr_result.suppression_reason,
            )
            return None

        # 14. Select trade
        trade = self.selector.select_trade(
            agg, chain, self.config.paper_trading_capital
        )

        if trade:
            self._print_trade_signal(
                trade, current_price, vwap_ctx, pa_result,
                mtf_result=mtf_result, div_result=div_result,
                corr_result=corr_result, session_info=session_info,
            )
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

    def backtest(self, symbol: str, days: int = 252):
        """Run backtesting on historical data."""
        from options_trader.backtesting.backtester import Backtester
        ohlcv = self.data.fetch_ohlcv(symbol, days=days)
        iv_series = self.data.fetch_historical_iv(symbol, ohlcv)
        backtester = Backtester(self.config)
        result = backtester.run(symbol, ohlcv, historical_iv=iv_series)
        print(result.summary())
        return result

    # ------------------------------------------------------------------
    # MTF builder from single OHLCV
    # ------------------------------------------------------------------

    @staticmethod
    def _build_mtf_from_ohlcv(ohlcv: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Resample daily OHLCV into pseudo-HTF frames using rolling windows.
        This is a best-effort approximation when only one TF is available.
        """
        mtf = {"D1": ohlcv}
        try:
            # Treat last 20 bars as "H4 equivalent" (recent close-up view)
            if len(ohlcv) >= 20:
                mtf["H4"] = ohlcv.tail(60)
            if len(ohlcv) >= 10:
                mtf["H1"] = ohlcv.tail(30)
        except Exception:
            pass
        return mtf

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _print_trade_signal(
        self,
        trade: TradeSignal,
        current_price: float,
        vwap_ctx=None,
        pa_result=None,
        mtf_result=None,
        div_result=None,
        corr_result=None,
        session_info=None,
    ) -> None:
        dte = (trade.expiration - datetime.utcnow()).days
        print("\n" + "=" * 65)
        print(f"  TRADE SIGNAL: {trade.action} {trade.symbol} {trade.option_type.value}")
        print("=" * 65)
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

        # Session context
        if session_info:
            print("-" * 65)
            print(f"  Session: {session_info.session_name.upper()} | "
                  f"Liquidity={session_info.liquidity_score:.0%} | "
                  f"Overlap={'YES' if session_info.is_overlap else 'no'}")
            print(f"    {session_info.notes}")

        # MTF context
        if mtf_result:
            print("-" * 65)
            print(f"  MTF Confluence: {mtf_result.htf_bias.upper()} | "
                  f"score={mtf_result.confluence_score:.0%} | "
                  f"trend={mtf_result.dominant_trend}")
            tf_lines = [
                f"    {tf}: {b.trend.upper()} (RSI={b.rsi:.0f}, EMA={'aligned' if b.ema_aligned else 'mixed'})"
                for tf, b in sorted(mtf_result.timeframe_biases.items())
            ]
            for line in tf_lines:
                print(line)

        # Divergence context
        if div_result:
            all_divs = div_result.rsi_divergences + div_result.macd_divergences
            if all_divs:
                print("-" * 65)
                print(f"  Divergence: {len(all_divs)} event(s) | "
                      f"RSI={div_result.rsi_value:.0f} | "
                      f"signal={div_result.combined_signal.signal.name}")
                for d in all_divs[:3]:
                    print(f"    {d.description}")

        # Correlation context
        if corr_result and corr_result.correlation_strength != "weak":
            print("-" * 65)
            print(f"  Correlation: {corr_result.symbol}/{corr_result.correlated_asset} | "
                  f"r20={corr_result.correlation_20:.2f} ({corr_result.correlation_strength}) | "
                  f"DXY={corr_result.dxy_trend} | risk={corr_result.risk_regime}")

        # VWAP context block
        if vwap_ctx:
            print("-" * 65)
            dev_sign = "+" if vwap_ctx.deviation_pct >= 0 else ""
            print(f"  VWAP Context:")
            print(f"    VWAP:          {vwap_ctx.vwap:.2f}  (price {dev_sign}{vwap_ctx.deviation_pct*100:.2f}% away)")
            print(f"    Band position: {vwap_ctx.band_position}")
            print(f"    VWAP trend:    {vwap_ctx.vwap_trend}")
            if vwap_ctx.is_reclaim:
                print(f"    *** VWAP RECLAIM — high-conviction bullish event ***")
            if vwap_ctx.is_rejection:
                print(f"    *** VWAP REJECTION — high-conviction bearish event ***")
            if vwap_ctx.is_extended:
                print(f"    ⚠ Price extended {abs(vwap_ctx.deviation_pct)*100:.1f}% from VWAP")

        # Price action context block
        if pa_result:
            print("-" * 65)
            print(f"  Price Action:")
            print(f"    Structure:     {pa_result.structure.trend.upper()}"
                  + (" (HH+HL)" if pa_result.structure.higher_highs and pa_result.structure.higher_lows else "")
                  + (" (LH+LL)" if pa_result.structure.lower_highs and pa_result.structure.lower_lows else ""))
            if pa_result.structure.last_bos:
                print(f"    BOS:           {pa_result.structure.last_bos:.2f}")
            if pa_result.structure.last_choch:
                print(f"    CHoCH:         {pa_result.structure.last_choch:.2f} ⚠ structure flipping")
            if pa_result.nearest_ob:
                ob = pa_result.nearest_ob
                print(f"    Nearest OB:    {ob.direction.upper()} {ob.price_low:.2f}–{ob.price_high:.2f}")
            if pa_result.nearest_fvg:
                fvg = pa_result.nearest_fvg
                print(f"    Nearest FVG:   {fvg.direction.upper()} {fvg.price_low:.2f}–{fvg.price_high:.2f}")
            recent_pats = [p for p in pa_result.patterns[-5:]]
            if recent_pats:
                print(f"    Recent pats:   {', '.join(p.name for p in recent_pats)}")

        print("-" * 65)
        print(f"  Rationale:")
        for part in trade.rationale.split(" | "):
            print(f"    • {part}")
        print("=" * 65 + "\n")
