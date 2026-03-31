"""
MT5 Live Trading Loop
======================
Replaces the yfinance-based engine.analyze_symbol() with a real-time
polling loop connected to a live MT5 terminal.

Architecture:
  MT5Connector  →  fetches OHLCV + tick data every N seconds
  Analysis pipeline  →  same 12-module engine (PA, VWAP, MTF, Divergence, etc.)
  Signal gate  →  checks for new candle close on primary timeframe
  MT5Executor  →  places / manages orders based on signals
  TradeJournal  →  logs every entry/exit to SQLite

Usage (paper mode — no real capital):
    python -m options_trader.mt5_live

Usage (live mode — REAL CAPITAL):
    python -m options_trader.mt5_live --live --login 12345678 --password "xxx" --server "ICMarkets-Live"

Workflow per poll cycle:
  1. Check if a new M15 candle has closed since last cycle
  2. Fetch updated OHLCV for all timeframes
  3. Run full analysis pipeline
  4. If signal strong enough and no existing position: open order
  5. If open position: check SL/TP, apply trailing stop, log state
  6. Sleep until next poll interval
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Optional

import pandas as pd

from options_trader.core.config import TradingConfig
from options_trader.analyzers.order_flow import OrderFlowAnalyzer
from options_trader.analyzers.technical import TechnicalAnalyzer
from options_trader.analyzers.support_resistance import SupportResistanceAnalyzer
from options_trader.analyzers.events import EventsAnalyzer
from options_trader.analyzers.price_action import PriceActionAnalyzer
from options_trader.analyzers.vwap import VWAPAnalyzer
from options_trader.analyzers.multi_timeframe import MultiTimeframeAnalyzer
from options_trader.analyzers.divergence import DivergenceDetector
from options_trader.analyzers.correlation import CorrelationAnalyzer
from options_trader.strategies.signal_aggregator import SignalAggregator, AggregatedSignal
from options_trader.utils.logger import setup_logger
from options_trader.utils.mt5_connector import MT5Connector
from options_trader.utils.mt5_executor import MT5Executor, MT5Position, OrderResult
from options_trader.utils.session_filter import SessionFilter
from options_trader.core.models import SignalStrength

logger = logging.getLogger(__name__)

# Minimum signal strength to open a trade
MIN_TRADE_STRENGTH  = {SignalStrength.BUY, SignalStrength.STRONG_BUY,
                       SignalStrength.SELL, SignalStrength.STRONG_SELL}

# Trailing stop activation (pips in profit before trail activates)
TRAIL_ACTIVATION_PIPS = 15.0
TRAIL_DISTANCE_PIPS   = 10.0


class MT5LiveBot:
    """
    Full live/paper trading loop powered by the MT5 terminal.

    The bot:
    - Polls every config.mt5_poll_interval seconds
    - Only generates new entry signals on confirmed candle closes
      (avoids acting on mid-candle noise)
    - Enforces one position per symbol at a time
    - Applies trailing stops and breakeven moves automatically
    - Logs every trade to the SQLite journal
    """

    def __init__(self, config: TradingConfig):
        self.config = config
        self._running = False
        self._last_candle_time: Dict[str, datetime] = {}  # symbol → last processed candle

        # MT5 connection layer
        self.connector = MT5Connector(config)
        self.executor  = MT5Executor(config, connector=self.connector)

        # Analysis modules
        self.tech_analyzer   = TechnicalAnalyzer(config)
        self.sr_analyzer     = SupportResistanceAnalyzer(config)
        self.events_analyzer = EventsAnalyzer(config)
        self.pa_analyzer     = PriceActionAnalyzer(config)
        self.vwap_analyzer   = VWAPAnalyzer(config)
        self.mtf_analyzer    = MultiTimeframeAnalyzer(config)
        self.div_detector    = DivergenceDetector(config)
        self.corr_analyzer   = CorrelationAnalyzer(config)
        self.flow_analyzer   = OrderFlowAnalyzer(config)
        self.session_filter  = SessionFilter(config)
        self.aggregator      = SignalAggregator(config)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the live trading loop. Blocks until interrupted."""
        logger.info("=" * 60)
        logger.info("MT5 Live Bot starting | paper=%s | symbols=%s",
                    self.config.mt5_paper, self.config.mt5_symbols)
        logger.info("Poll interval: %ds | Primary TF: %s",
                    self.config.mt5_poll_interval, self.config.mt5_timeframes[0])
        logger.info("=" * 60)

        if not self.connector.connect():
            if self.config.mt5_paper:
                logger.warning("MT5 terminal not available — running in full simulation mode")
            else:
                logger.error("MT5 connection failed and mt5_paper=False. Exiting.")
                return

        # Graceful shutdown on Ctrl+C / SIGTERM
        self._running = True
        signal.signal(signal.SIGINT,  self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        try:
            self._loop()
        finally:
            self.connector.disconnect()
            if self.config.mt5_paper:
                print(self.executor.paper_summary())
            logger.info("MT5 Live Bot stopped")

    def _loop(self) -> None:
        while self._running:
            cycle_start = time.time()

            for symbol in self.config.mt5_symbols:
                try:
                    self._process_symbol(symbol)
                except Exception as exc:
                    logger.error("Error processing %s: %s", symbol, exc, exc_info=True)

            # Manage existing open positions (trailing stops, SL/TP checks)
            self._manage_positions()

            elapsed = time.time() - cycle_start
            sleep_for = max(0, self.config.mt5_poll_interval - elapsed)
            logger.debug("Cycle complete in %.1fs — sleeping %.1fs", elapsed, sleep_for)
            time.sleep(sleep_for)

    # ------------------------------------------------------------------
    # Per-symbol analysis
    # ------------------------------------------------------------------

    def _process_symbol(self, symbol: str) -> None:
        # 1. Fetch primary timeframe OHLCV
        primary_tf = self.config.mt5_timeframes[0]  # e.g. "M15"
        ohlcv = self.connector.fetch_ohlcv_mt5(symbol, primary_tf, bars=300)

        if len(ohlcv) < 50:
            logger.warning("%s: insufficient bars (%d)", symbol, len(ohlcv))
            return

        current_price = float(ohlcv["close"].iloc[-1])
        last_candle_ts = ohlcv.index[-1]

        # 2. Only act on confirmed (closed) candles
        # The current forming candle is unreliable — wait for a new close
        if not self._is_new_candle(symbol, last_candle_ts):
            logger.debug("%s: no new candle close — skipping analysis", symbol)
            return

        self._last_candle_time[symbol] = last_candle_ts
        logger.info("-" * 50)
        logger.info("%s new %s candle @ %.5f  [%s UTC]",
                    symbol, primary_tf, current_price,
                    last_candle_ts.strftime("%Y-%m-%d %H:%M"))

        # 3. Session filter
        session_info = self.session_filter.analyze(symbol)
        if not session_info.is_tradeable:
            logger.info("%s: session filter — %s (liq=%.0f%%) — skip",
                        symbol, session_info.session_name, session_info.liquidity_score * 100)
            return

        # 4. Check if already in a position for this symbol
        existing = self.executor.get_positions(symbol)
        if existing:
            logger.info("%s: already in position (ticket=%d) — skip entry",
                        symbol, existing[0].ticket)
            return

        # 5. Full analysis pipeline
        agg, meta = self._run_analysis(symbol, ohlcv)
        if agg is None:
            return

        # 6. Signal gate
        direction = agg.direction
        if direction not in (SignalStrength.BUY, SignalStrength.STRONG_BUY,
                             SignalStrength.SELL, SignalStrength.STRONG_SELL):
            logger.info("%s: signal NEUTRAL — no trade", symbol)
            return

        if agg.confidence < self.config.min_confidence:
            logger.info("%s: confidence %.0f%% < %.0f%% threshold — skip",
                        symbol, agg.confidence * 100, self.config.min_confidence * 100)
            return

        # MTF hard gate
        if meta.get("mtf_suppressed"):
            logger.info("%s: MTF suppression active — skip entry", symbol)
            return

        # Correlation suppression gate
        if meta.get("corr_suppressed"):
            logger.info("%s: correlation suppression — skip entry", symbol)
            return

        # 7. Calculate SL/TP from ATR
        sl, tp1, tp2, lot_size = self._calculate_trade_params(
            symbol, ohlcv, direction, current_price
        )
        if sl is None:
            return

        # 8. Open order
        trade_dir = "BUY" if direction in (SignalStrength.BUY, SignalStrength.STRONG_BUY) else "SELL"
        comment = f"bot_{primary_tf}_{agg.confidence*100:.0f}pct"

        result = (
            self.executor.open_buy(symbol, lot_size, sl, tp1, comment)
            if trade_dir == "BUY"
            else self.executor.open_sell(symbol, lot_size, sl, tp1, comment)
        )

        if result.success:
            self._log_trade_open(symbol, trade_dir, result, agg, ohlcv, current_price)
        else:
            logger.error("%s: order failed — %s", symbol, result.error_message)

    # ------------------------------------------------------------------
    # Analysis pipeline (CFD mode — no options chain)
    # ------------------------------------------------------------------

    def _run_analysis(self, symbol: str, ohlcv: pd.DataFrame):
        """
        Runs the full 9-module analysis pipeline.
        Returns (AggregatedSignal, meta_dict) or (None, {}) on failure.
        """
        current_price = float(ohlcv["close"].iloc[-1])
        meta: dict = {}

        try:
            # Technical
            tech_signals, regime, tech_score = self.tech_analyzer.analyze(symbol, ohlcv)

            # Price action
            pa_result = self.pa_analyzer.analyze(symbol, ohlcv, current_price)

            # VWAP
            _, vwap_ctx, vwap_signal = self.vwap_analyzer.analyze(symbol, ohlcv, session_reset=True)

            # MTF
            ohlcv_by_tf = self.connector.fetch_ohlcv_multi_tf(symbol)
            if not ohlcv_by_tf:
                ohlcv_by_tf = {self.config.mt5_timeframes[0]: ohlcv}
            mtf_result = self.mtf_analyzer.analyze(symbol, ohlcv_by_tf)
            meta["mtf_suppressed"] = not mtf_result.ltf_aligned

            # Divergence
            div_result = self.div_detector.analyze(symbol, ohlcv)

            # DXY correlation (attempt to fetch DXY; skip if unavailable)
            corr_result = self._run_correlation(symbol, ohlcv)
            meta["corr_suppressed"] = corr_result.suppression_active if corr_result else False

            # S/R
            _, sr_signal = self.sr_analyzer.analyze(symbol, ohlcv, current_price)

            # Events (stub for CFD — no options events)
            _, event_signal = self.events_analyzer.analyze(symbol)

            # Order flow (stub for CFD)
            flow_signal = tech_signals[0] if tech_signals else self._neutral_signal("OrderFlow")

            # Session
            session_info = self.session_filter.analyze(symbol)

            # Aggregate
            agg = self.aggregator.aggregate(
                symbol=symbol,
                order_flow_signal=flow_signal,
                technical_score=tech_score,
                sr_signal=sr_signal,
                event_signal=event_signal,
                regime=regime,
                price_action_signal=pa_result.signal,
                vwap_signal=vwap_signal,
                mtf_signal=mtf_result.signal,
                divergence_signal=div_result.combined_signal,
                correlation_signal=corr_result.signal if corr_result else None,
                session_signal=session_info.signal,
                additional_signals=tech_signals,
            )

            logger.info(
                "%s | score=%+.3f | conf=%.0f%% | dir=%s | MTF=%s | PA=%s | VWAP=%s | Div=%s",
                symbol, agg.composite_score, agg.confidence * 100,
                agg.direction.name, mtf_result.htf_bias.upper(),
                pa_result.signal.signal.name, vwap_signal.signal.name,
                div_result.combined_signal.signal.name,
            )
            return agg, meta

        except Exception as exc:
            logger.error("Analysis pipeline error for %s: %s", symbol, exc, exc_info=True)
            return None, {}

    def _run_correlation(self, symbol: str, primary_ohlcv: pd.DataFrame):
        """Attempt DXY fetch; return None silently if unavailable."""
        try:
            # Try to fetch DXY from MT5 (may not be available on all brokers)
            dxy_names = ["DXY", "USDX", "USDIndex", "DX-Y.NYB"]
            for name in dxy_names:
                dxy_ohlcv = self.connector.fetch_ohlcv_mt5(name, "D1", bars=100)
                if len(dxy_ohlcv) > 20:
                    return self.corr_analyzer.analyze(symbol, primary_ohlcv,
                                                      correlated_ohlcv=dxy_ohlcv,
                                                      correlated_symbol="DXY")
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Trade parameter calculation
    # ------------------------------------------------------------------

    def _calculate_trade_params(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        direction: SignalStrength,
        current_price: float,
    ):
        """
        Calculate SL, TP1, TP2, and lot size using ATR-based levels.

        SL  = 1.5× ATR below entry (BUY) or above entry (SELL)
        TP1 = 2.5× ATR from entry (R:R ~1.67:1)
        TP2 = 4.0× ATR from entry (R:R ~2.67:1)
        """
        try:
            h = ohlcv["high"].values
            l = ohlcv["low"].values
            c = ohlcv["close"].values
            n = len(ohlcv)

            # ATR(14)
            trs = [max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
                   for i in range(1, n)]
            atr = float(sum(trs[-14:]) / 14) if len(trs) >= 14 else float(sum(trs) / len(trs))

            sym_info = self.connector.symbol_info(symbol)
            is_buy = direction in (SignalStrength.BUY, SignalStrength.STRONG_BUY)

            if is_buy:
                sl  = round(current_price - 1.5 * atr, sym_info.get("digits", 2))
                tp1 = round(current_price + 2.5 * atr, sym_info.get("digits", 2))
                tp2 = round(current_price + 4.0 * atr, sym_info.get("digits", 2))
            else:
                sl  = round(current_price + 1.5 * atr, sym_info.get("digits", 2))
                tp1 = round(current_price - 2.5 * atr, sym_info.get("digits", 2))
                tp2 = round(current_price - 4.0 * atr, sym_info.get("digits", 2))

            # Minimum R:R check
            risk  = abs(current_price - sl)
            reward = abs(current_price - tp1)
            if risk == 0 or (reward / risk) < self.config.min_risk_reward:
                logger.warning("%s: R:R %.2f < min %.2f — skip",
                               symbol, reward / risk if risk else 0, self.config.min_risk_reward)
                return None, None, None, None

            # Lot size from risk %
            sl_pips = abs(current_price - sl) / (sym_info.get("point", 0.01) * 10)
            lot_size = self.connector.calculate_lot_size(symbol, sl_pips)

            logger.info(
                "%s params | price=%.2f | SL=%.2f (%+.1f) | TP1=%.2f | TP2=%.2f | lot=%.2f | ATR=%.2f",
                symbol, current_price, sl, sl - current_price, tp1, tp2, lot_size, atr,
            )
            return sl, tp1, tp2, lot_size

        except Exception as exc:
            logger.error("Trade param calculation failed for %s: %s", symbol, exc)
            return None, None, None, None

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _manage_positions(self) -> None:
        """
        Called every cycle. Handles:
        - Paper mode SL/TP simulation
        - Trailing stop updates
        - Breakeven moves
        """
        # Paper mode: check SL/TP hits against live ticks
        if self.config.mt5_paper:
            closed = self.executor.check_sl_tp()
            if closed:
                logger.info("Paper positions closed by SL/TP: %s", closed)

        # Update trailing stops for all open positions
        for symbol in self.config.mt5_symbols:
            positions = self.executor.get_positions(symbol)
            for pos in positions:
                # Move to breakeven once 15 pips in profit
                sym_info = self.connector.symbol_info(pos.symbol)
                point = sym_info.get("point", 0.01)
                profit_pips = pos.pips
                if profit_pips >= TRAIL_ACTIVATION_PIPS:
                    # Apply trailing stop
                    self.executor.apply_trailing_stop(
                        pos.ticket,
                        trail_pips=TRAIL_DISTANCE_PIPS,
                        activation_pips=TRAIL_ACTIVATION_PIPS,
                    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_new_candle(self, symbol: str, candle_ts: datetime) -> bool:
        """
        Returns True if this candle timestamp is newer than the last
        one we processed for this symbol.
        First call always returns True.
        """
        last = self._last_candle_time.get(symbol)
        if last is None:
            return True  # first run
        if candle_ts.tzinfo is None:
            candle_ts = candle_ts.replace(tzinfo=timezone.utc)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return candle_ts > last

    def _log_trade_open(
        self,
        symbol: str,
        direction: str,
        result: OrderResult,
        agg: AggregatedSignal,
        ohlcv: pd.DataFrame,
        current_price: float,
    ) -> None:
        mode = "[PAPER]" if self.config.mt5_paper else "[LIVE]"
        print(f"\n{'='*55}")
        print(f"  {mode} TRADE OPENED")
        print(f"{'='*55}")
        print(f"  Symbol    : {symbol}")
        print(f"  Direction : {direction}")
        print(f"  Ticket    : {result.ticket}")
        print(f"  Price     : {result.price:.5f}")
        print(f"  SL        : {result.sl:.5f}  ({abs(result.price-result.sl):.2f} pts)")
        print(f"  TP        : {result.tp:.5f}  ({abs(result.tp-result.price):.2f} pts)")
        print(f"  Lot Size  : {result.volume:.2f}")
        print(f"  Confidence: {agg.confidence*100:.0f}%")
        print(f"  Direction : {agg.direction.name}")
        print(f"  Score     : {agg.composite_score:+.3f}")
        print(f"  Rationale :")
        for part in agg.rationale.split(" | ")[:5]:
            print(f"    • {part}")
        print(f"{'='*55}\n")

    @staticmethod
    def _neutral_signal(name: str):
        from options_trader.core.models import TechnicalSignal, SignalStrength
        return TechnicalSignal(
            indicator=name, value=0.0, signal=SignalStrength.NEUTRAL, description="N/A"
        )

    def _handle_shutdown(self, signum, frame) -> None:
        logger.info("Shutdown signal received — stopping bot...")
        self._running = False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_config_from_args(args: argparse.Namespace) -> TradingConfig:
    """Build TradingConfig from CLI arguments."""
    cfg = TradingConfig(
        mt5_login    = args.login,
        mt5_password = args.password,
        mt5_server   = args.server,
        mt5_paper    = not args.live,
        mt5_symbols  = args.symbols.split(","),
        mt5_timeframes = args.timeframes.split(","),
        mt5_poll_interval = args.interval,
        mt5_risk_pct = args.risk / 100.0,
        min_confidence = args.confidence / 100.0,
        log_level    = "DEBUG" if args.debug else "INFO",
    )
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MT5 Live/Paper Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Paper mode (safe — no real capital):
  python -m options_trader.mt5_live

  # Paper mode with explicit symbols:
  python -m options_trader.mt5_live --symbols XAUUSD,EURUSD

  # Live mode (REAL CAPITAL — use with extreme caution):
  python -m options_trader.mt5_live --live \\
    --login 12345678 --password "xxx" --server "ICMarkets-Live" \\
    --symbols XAUUSD --risk 1.0 --confidence 65

Requirements:
  - pip install MetaTrader5
  - MT5 terminal running on Windows (or Linux + Wine)
  - Account credentials for your broker
        """,
    )
    parser.add_argument("--live",       action="store_true",  help="Enable live execution (default: paper)")
    parser.add_argument("--login",      type=int,   default=None,    help="MT5 account number")
    parser.add_argument("--password",   type=str,   default=None,    help="MT5 password")
    parser.add_argument("--server",     type=str,   default=None,    help="MT5 broker server")
    parser.add_argument("--symbols",    type=str,   default="XAUUSD", help="Comma-separated symbols")
    parser.add_argument("--timeframes", type=str,   default="M15,H1,H4,D1", help="Comma-separated TFs")
    parser.add_argument("--interval",   type=int,   default=60,      help="Poll interval in seconds")
    parser.add_argument("--risk",       type=float, default=1.0,     help="Risk % per trade (default: 1.0)")
    parser.add_argument("--confidence", type=float, default=60.0,    help="Min confidence % (default: 60)")
    parser.add_argument("--debug",      action="store_true",         help="Verbose debug logging")
    args = parser.parse_args()

    config = build_config_from_args(args)
    setup_logger("options_trader", config.log_level, "mt5_live.log")

    if args.live:
        print("\n" + "!" * 60)
        print("  WARNING: LIVE TRADING MODE — REAL CAPITAL AT RISK")
        print("  Ensure you have tested thoroughly in paper mode first.")
        print("!" * 60)
        confirm = input("\n  Type 'CONFIRM' to proceed with live trading: ").strip()
        if confirm != "CONFIRM":
            print("Aborted.")
            sys.exit(0)

    bot = MT5LiveBot(config)
    bot.run()


if __name__ == "__main__":
    main()
