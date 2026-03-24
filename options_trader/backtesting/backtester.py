"""
Backtesting Framework
Replays historical OHLCV data through the full analysis pipeline and
tracks simulated options trades with realistic P&L accounting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from options_trader.core.config import TradingConfig
from options_trader.core.models import (
    OptionContract,
    OptionType,
    SignalStrength,
    TradeSignal,
)
from options_trader.analyzers.options_greeks import BlackScholes, OptionsGreeksCalculator
from options_trader.analyzers.technical import TechnicalAnalyzer
from options_trader.analyzers.support_resistance import SupportResistanceAnalyzer
from options_trader.analyzers.order_flow import OrderFlowAnalyzer
from options_trader.analyzers.events import EventsAnalyzer
from options_trader.strategies.signal_aggregator import SignalAggregator
from options_trader.strategies.trade_selector import TradeSelector

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    symbol: str
    option_type: OptionType
    strike: float
    expiration: datetime
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""       # "target", "stop", "expiry", "signal_exit"
    contracts: int = 1
    pnl: float = 0.0
    pnl_pct: float = 0.0
    signal_confidence: float = 0.0
    signal_direction: Optional[SignalStrength] = None


@dataclass
class BacktestResult:
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    trades: List[BacktestTrade] = field(default_factory=list)

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        return sum(1 for t in self.trades if t.pnl > 0)

    @property
    def losing_trades(self) -> int:
        return sum(1 for t in self.trades if t.pnl <= 0)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return self.winning_trades / len(self.trades)

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def avg_pnl(self) -> float:
        if not self.trades:
            return 0.0
        return self.total_pnl / len(self.trades)

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    @property
    def max_drawdown(self) -> float:
        if not self.trades:
            return 0.0
        cumulative = np.cumsum([t.pnl for t in self.trades])
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        return float(drawdowns.max()) if len(drawdowns) > 0 else 0.0

    @property
    def sharpe_ratio(self) -> float:
        if len(self.trades) < 2:
            return 0.0
        pnls = [t.pnl for t in self.trades]
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls)
        if std_pnl == 0:
            return 0.0
        return float(mean_pnl / std_pnl * np.sqrt(252))

    @property
    def total_return_pct(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return (self.final_capital - self.initial_capital) / self.initial_capital * 100

    def summary(self) -> str:
        return (
            f"=== Backtest Results: {self.symbol} ===\n"
            f"Period:        {self.start_date.date()} → {self.end_date.date()}\n"
            f"Capital:       ${self.initial_capital:,.0f} → ${self.final_capital:,.0f}\n"
            f"Total Return:  {self.total_return_pct:+.1f}%\n"
            f"Total Trades:  {self.total_trades}\n"
            f"Win Rate:      {self.win_rate*100:.1f}%\n"
            f"Profit Factor: {self.profit_factor:.2f}\n"
            f"Avg P&L/Trade: ${self.avg_pnl:,.0f}\n"
            f"Max Drawdown:  ${self.max_drawdown:,.0f}\n"
            f"Sharpe Ratio:  {self.sharpe_ratio:.2f}\n"
        )


class Backtester:
    """
    Runs the trading bot against historical OHLCV price data.
    Options are synthetically priced using Black-Scholes.
    """

    def __init__(self, config: TradingConfig):
        self.config = config
        self.tech_analyzer = TechnicalAnalyzer(config)
        self.sr_analyzer = SupportResistanceAnalyzer(config)
        self.flow_analyzer = OrderFlowAnalyzer(config)
        self.events_analyzer = EventsAnalyzer(config)
        self.aggregator = SignalAggregator(config)
        self.selector = TradeSelector(config)
        self.greeks_calc = OptionsGreeksCalculator()

    def run(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        historical_iv: Optional[pd.Series] = None,
        warmup_bars: int = 60,
    ) -> BacktestResult:
        """
        Args:
            ohlcv: DataFrame with columns [open, high, low, close, volume], DatetimeIndex
            historical_iv: optional Series of daily IV indexed by date (for realistic pricing)
            warmup_bars: bars to skip at start for indicator warmup
        """
        ohlcv = ohlcv.copy()
        ohlcv.columns = [c.lower() for c in ohlcv.columns]

        capital = self.config.paper_trading_capital
        trades: List[BacktestTrade] = []
        open_trade: Optional[BacktestTrade] = None

        start_date = ohlcv.index[warmup_bars]
        end_date = ohlcv.index[-1]

        logger.info("Starting backtest for %s | %s → %s", symbol, start_date.date(), end_date.date())

        for i in range(warmup_bars, len(ohlcv)):
            bar = ohlcv.iloc[i]
            current_date = ohlcv.index[i]
            current_price = float(bar["close"])
            hist_slice = ohlcv.iloc[: i + 1]

            iv = float(historical_iv.iloc[i]) if historical_iv is not None else 0.20

            # --- Manage open trade ---
            if open_trade is not None:
                open_trade, closed = self._manage_open_trade(
                    open_trade, current_price, current_date, iv
                )
                if closed:
                    pnl = (open_trade.exit_price - open_trade.entry_price) * open_trade.contracts * 100
                    open_trade.pnl = pnl
                    open_trade.pnl_pct = pnl / (open_trade.entry_price * open_trade.contracts * 100)
                    capital += pnl
                    trades.append(open_trade)
                    logger.info(
                        "CLOSED %s | exit=%s | P&L=$%.0f (%.1f%%) | reason=%s",
                        symbol,
                        open_trade.exit_date.date() if open_trade.exit_date else "?",
                        pnl,
                        open_trade.pnl_pct * 100,
                        open_trade.exit_reason,
                    )
                    open_trade = None

            # --- Generate signal ---
            if open_trade is None and i % 1 == 0:  # check every bar
                signal = self._generate_signal(symbol, hist_slice, current_price, iv)
                if signal and signal.is_valid:
                    contracts_qty = max(1, int(capital * self.config.max_position_size_pct / (signal.entry_price * 100)))
                    cost = signal.entry_price * contracts_qty * 100
                    if cost <= capital * self.config.max_position_size_pct * 2:
                        expiry = current_date + timedelta(days=30)
                        option_type = signal.option_type
                        open_trade = BacktestTrade(
                            symbol=symbol,
                            option_type=option_type,
                            strike=signal.strike,
                            expiration=expiry,
                            entry_date=current_date,
                            entry_price=signal.entry_price,
                            contracts=contracts_qty,
                            signal_confidence=signal.confidence,
                            signal_direction=signal.strength,
                        )
                        logger.info(
                            "OPEN %s %s %s strike=%.0f | entry=%.2f | qty=%d | conf=%.0f%%",
                            symbol,
                            option_type.value,
                            current_date.date(),
                            signal.strike,
                            signal.entry_price,
                            contracts_qty,
                            signal.confidence * 100,
                        )

        # Close any remaining open trade at last price
        if open_trade is not None:
            last_price = float(ohlcv["close"].iloc[-1])
            last_iv = float(historical_iv.iloc[-1]) if historical_iv is not None else 0.20
            synth_exit = self._synthetic_option_price(
                open_trade, last_price, last_iv, ohlcv.index[-1]
            )
            open_trade.exit_price = synth_exit
            open_trade.exit_date = ohlcv.index[-1]
            open_trade.exit_reason = "end_of_data"
            pnl = (synth_exit - open_trade.entry_price) * open_trade.contracts * 100
            open_trade.pnl = pnl
            open_trade.pnl_pct = pnl / (open_trade.entry_price * open_trade.contracts * 100)
            capital += pnl
            trades.append(open_trade)

        result = BacktestResult(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.config.paper_trading_capital,
            final_capital=capital,
            trades=trades,
        )
        logger.info("\n%s", result.summary())
        return result

    # ------------------------------------------------------------------
    # Signal generation (simplified for backtesting)
    # ------------------------------------------------------------------

    def _generate_signal(
        self,
        symbol: str,
        ohlcv_slice: pd.DataFrame,
        current_price: float,
        iv: float,
    ) -> Optional[TradeSignal]:
        try:
            tech_signals, regime, tech_score = self.tech_analyzer.analyze(symbol, ohlcv_slice)
            sr_levels, sr_signal = self.sr_analyzer.analyze(symbol, ohlcv_slice, current_price)

            # Synthetic order flow from price action
            flow_signal = self._synthetic_flow_signal(ohlcv_slice)

            # No events in backtest (simplified)
            from options_trader.core.models import TechnicalSignal, SignalStrength
            event_signal = TechnicalSignal(
                indicator="Events",
                value=0.0,
                signal=SignalStrength.NEUTRAL,
                description="No events data in backtest",
            )

            agg = self.aggregator.aggregate(
                symbol=symbol,
                order_flow_signal=flow_signal,
                technical_score=tech_score,
                sr_signal=sr_signal,
                event_signal=event_signal,
                regime=regime,
            )

            # Build a synthetic contract at ATM
            strike = round(current_price / 5) * 5  # round to nearest $5
            expiry = datetime.utcnow() + timedelta(days=30)
            option_type = OptionType.CALL if agg.direction.value > 0 else OptionType.PUT

            T = 30 / 365.0
            bs_price = BlackScholes.price(
                current_price, strike, T, 0.05, iv, option_type
            )

            synthetic_contract = OptionContract(
                symbol=symbol,
                option_type=option_type,
                strike=strike,
                expiration=expiry,
                bid=bs_price * 0.97,
                ask=bs_price * 1.03,
                last=bs_price,
                volume=1000,
                open_interest=5000,
                implied_volatility=iv,
                delta=0.35 if option_type == OptionType.CALL else -0.35,
                underlying_price=current_price,
            )

            return self.selector.select_trade(agg, [synthetic_contract], self.config.paper_trading_capital)
        except Exception as exc:
            logger.debug("Signal generation error: %s", exc)
            return None

    def _synthetic_flow_signal(self, ohlcv: pd.DataFrame):
        """Approximate order flow from price/volume action."""
        from options_trader.core.models import TechnicalSignal, SignalStrength
        if len(ohlcv) < 5:
            return TechnicalSignal(
                indicator="OrderFlow", value=0.0,
                signal=SignalStrength.NEUTRAL, description="Insufficient data"
            )
        vol_avg = ohlcv["volume"].rolling(20).mean().iloc[-1]
        last_vol = float(ohlcv["volume"].iloc[-1])
        last_close = float(ohlcv["close"].iloc[-1])
        prev_close = float(ohlcv["close"].iloc[-2])
        vol_ratio = last_vol / vol_avg if vol_avg > 0 else 1.0
        direction = 1 if last_close > prev_close else -1
        score = direction * min(1.0, vol_ratio - 1.0)
        signal = SignalStrength.NEUTRAL
        if score > 0.5:
            signal = SignalStrength.BUY
        elif score < -0.5:
            signal = SignalStrength.SELL
        return TechnicalSignal(
            indicator="OrderFlow", value=score,
            signal=signal, description=f"Synthetic flow (vol_ratio={vol_ratio:.1f})"
        )

    # ------------------------------------------------------------------
    # Trade management
    # ------------------------------------------------------------------

    def _manage_open_trade(
        self,
        trade: BacktestTrade,
        current_price: float,
        current_date: datetime,
        current_iv: float,
    ) -> Tuple[BacktestTrade, bool]:
        """Check stop/target/expiry. Returns (trade, was_closed)."""
        current_option_price = self._synthetic_option_price(
            trade, current_price, current_iv, current_date
        )

        pnl_pct = (current_option_price - trade.entry_price) / trade.entry_price

        # Target hit
        if pnl_pct >= 0.50:
            trade.exit_price = current_option_price
            trade.exit_date = current_date
            trade.exit_reason = "target"
            return trade, True

        # Stop hit
        if pnl_pct <= -0.30:
            trade.exit_price = current_option_price
            trade.exit_date = current_date
            trade.exit_reason = "stop"
            return trade, True

        # Expiry
        if current_date >= trade.expiration:
            intrinsic = max(0.0, current_price - trade.strike) if trade.option_type == OptionType.CALL \
                else max(0.0, trade.strike - current_price)
            trade.exit_price = intrinsic
            trade.exit_date = current_date
            trade.exit_reason = "expiry"
            return trade, True

        return trade, False

    def _synthetic_option_price(
        self,
        trade: BacktestTrade,
        underlying_price: float,
        iv: float,
        current_date: datetime,
    ) -> float:
        T = max(0.0, (trade.expiration - current_date).total_seconds() / (365.25 * 86400))
        return max(
            0.01,
            BlackScholes.price(underlying_price, trade.strike, T, 0.05, iv, trade.option_type),
        )
