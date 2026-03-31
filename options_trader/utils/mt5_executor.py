"""
MetaTrader 5 Order Executor
============================
Handles all trade execution via the MT5 Python API.

Supports:
  - Market orders (BUY / SELL) with automatic SL/TP
  - Pending orders (BUY_LIMIT / SELL_LIMIT / BUY_STOP / SELL_STOP)
  - Position modification (move SL/TP, breakeven)
  - Position closing (full or partial)
  - Trailing stop management
  - Lot size calculation from risk %

Paper mode (config.mt5_paper = True):
  All orders are simulated locally. No real orders are sent to the broker.
  Paper positions are tracked in-memory with P&L calculated from live ticks.

IMPORTANT: For live trading, always test on a DEMO account first.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from options_trader.core.config import TradingConfig

logger = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None  # type: ignore
    MT5_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MT5Position:
    """Represents an open position (live or paper)."""
    ticket: int
    symbol: str
    direction: str          # "BUY" | "SELL"
    lot_size: float
    open_price: float
    stop_loss: float
    take_profit: float
    open_time: datetime
    magic: int
    comment: str = ""
    current_price: float = 0.0
    pnl: float = 0.0        # unrealised P&L in account currency
    pips: float = 0.0       # unrealised pips
    closed: bool = False
    close_price: float = 0.0
    close_time: Optional[datetime] = None
    close_reason: str = ""  # "sl", "tp", "manual", "trailing"


@dataclass
class OrderResult:
    """Result of an order send attempt."""
    success: bool
    ticket: int = 0
    order_type: str = ""
    symbol: str = ""
    volume: float = 0.0
    price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    error_code: int = 0
    error_message: str = ""
    paper: bool = False


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class MT5Executor:
    """
    Executes and manages trades on MetaTrader 5.

    In paper mode (config.mt5_paper = True) all execution is simulated:
    orders are tracked in-memory, P&L is computed from live ticks pulled
    via the connector. This allows full end-to-end testing without risking
    real capital.
    """

    def __init__(self, config: TradingConfig, connector=None):
        self.config = config
        self.connector = connector          # MT5Connector instance
        self._paper_positions: Dict[int, MT5Position] = {}
        self._next_paper_ticket = 100_001
        self._paper_balance = config.paper_trading_capital

    # ------------------------------------------------------------------
    # Market orders
    # ------------------------------------------------------------------

    def open_buy(
        self,
        symbol: str,
        lot_size: float,
        stop_loss: float,
        take_profit: float,
        comment: str = "bot_buy",
    ) -> OrderResult:
        """Open a BUY (long) market order."""
        return self._send_market_order(
            symbol, "BUY", lot_size, stop_loss, take_profit, comment
        )

    def open_sell(
        self,
        symbol: str,
        lot_size: float,
        stop_loss: float,
        take_profit: float,
        comment: str = "bot_sell",
    ) -> OrderResult:
        """Open a SELL (short) market order."""
        return self._send_market_order(
            symbol, "SELL", lot_size, stop_loss, take_profit, comment
        )

    def _send_market_order(
        self,
        symbol: str,
        direction: str,
        lot_size: float,
        stop_loss: float,
        take_profit: float,
        comment: str,
    ) -> OrderResult:
        if self.config.mt5_paper:
            return self._paper_open(symbol, direction, lot_size, stop_loss, take_profit, comment)

        if not MT5_AVAILABLE:
            return OrderResult(success=False, error_message="MT5 package not installed")

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return OrderResult(success=False, symbol=symbol,
                               error_message=f"Cannot get tick for {symbol}")

        order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
        price = tick.ask if direction == "BUY" else tick.bid

        request = {
            "action":      mt5.TRADE_ACTION_DEAL,
            "symbol":      symbol,
            "volume":      lot_size,
            "type":        order_type,
            "price":       price,
            "sl":          stop_loss,
            "tp":          take_profit,
            "deviation":   self.config.mt5_deviation,
            "magic":       self.config.mt5_magic,
            "comment":     comment,
            "type_time":   mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        logger.info(
            "MT5 SEND %s %s %.2f lots | price=%.5f | SL=%.5f | TP=%.5f",
            direction, symbol, lot_size, price, stop_loss, take_profit,
        )

        result = mt5.order_send(request)
        if result is None:
            err = mt5.last_error()
            logger.error("MT5 order_send returned None: %s", err)
            return OrderResult(success=False, symbol=symbol,
                               error_code=err[0], error_message=str(err))

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(
                "MT5 order failed: retcode=%d comment=%s", result.retcode, result.comment
            )
            return OrderResult(
                success=False, symbol=symbol,
                error_code=result.retcode, error_message=result.comment,
            )

        logger.info(
            "MT5 ORDER FILLED | ticket=%d | %s %s %.2f lots @ %.5f",
            result.order, direction, symbol, lot_size, result.price,
        )
        return OrderResult(
            success=True,
            ticket=result.order,
            order_type=direction,
            symbol=symbol,
            volume=lot_size,
            price=result.price,
            sl=stop_loss,
            tp=take_profit,
        )

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def close_position(
        self,
        ticket: int,
        reason: str = "manual",
        partial_lot: Optional[float] = None,
    ) -> OrderResult:
        """Close a position by ticket number (full or partial)."""
        if self.config.mt5_paper:
            return self._paper_close(ticket, reason)

        if not MT5_AVAILABLE:
            return OrderResult(success=False, error_message="MT5 not available")

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return OrderResult(success=False, error_message=f"Ticket {ticket} not found")

        pos = positions[0]
        volume = partial_lot if partial_lot else pos.volume
        order_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(pos.symbol)
        price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask

        request = {
            "action":      mt5.TRADE_ACTION_DEAL,
            "symbol":      pos.symbol,
            "volume":      volume,
            "type":        order_type,
            "position":    ticket,
            "price":       price,
            "deviation":   self.config.mt5_deviation,
            "magic":       self.config.mt5_magic,
            "comment":     f"close_{reason}",
            "type_time":   mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            err = mt5.last_error() if result is None else (result.retcode, result.comment)
            logger.error("MT5 close failed: %s", err)
            return OrderResult(success=False, error_message=str(err))

        logger.info("MT5 CLOSED ticket=%d | reason=%s | price=%.5f", ticket, reason, result.price)
        return OrderResult(success=True, ticket=ticket, price=result.price)

    def modify_position(
        self,
        ticket: int,
        new_sl: Optional[float] = None,
        new_tp: Optional[float] = None,
    ) -> OrderResult:
        """Modify SL/TP on an existing position."""
        if self.config.mt5_paper:
            return self._paper_modify(ticket, new_sl, new_tp)

        if not MT5_AVAILABLE:
            return OrderResult(success=False, error_message="MT5 not available")

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return OrderResult(success=False, error_message=f"Ticket {ticket} not found")

        pos = positions[0]
        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "symbol":   pos.symbol,
            "position": ticket,
            "sl":       new_sl if new_sl is not None else pos.sl,
            "tp":       new_tp if new_tp is not None else pos.tp,
        }
        result = mt5.order_send(request)
        success = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        return OrderResult(success=success, ticket=ticket,
                           error_message="" if success else str(result))

    def move_to_breakeven(self, ticket: int, buffer_pips: float = 2.0) -> OrderResult:
        """Move SL to entry + buffer_pips to lock in breakeven."""
        if self.config.mt5_paper:
            pos = self._paper_positions.get(ticket)
            if not pos:
                return OrderResult(success=False, error_message=f"Paper ticket {ticket} not found")
            sym_info = self.connector.symbol_info(pos.symbol) if self.connector else {}
            point = sym_info.get("point", 0.01)
            buf = buffer_pips * point * 10
            new_sl = pos.open_price + buf if pos.direction == "BUY" else pos.open_price - buf
            return self._paper_modify(ticket, new_sl=new_sl)

        if not MT5_AVAILABLE:
            return OrderResult(success=False)

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return OrderResult(success=False)

        pos = positions[0]
        sym = mt5.symbol_info(pos.symbol)
        buf = buffer_pips * sym.point * 10
        new_sl = pos.price_open + buf if pos.type == 0 else pos.price_open - buf
        return self.modify_position(ticket, new_sl=new_sl)

    def get_positions(self, symbol: Optional[str] = None) -> List[MT5Position]:
        """Return all open positions, optionally filtered by symbol."""
        if self.config.mt5_paper:
            positions = [p for p in self._paper_positions.values() if not p.closed]
            if symbol:
                positions = [p for p in positions if p.symbol == symbol]
            self._update_paper_pnl(positions)
            return positions

        if not MT5_AVAILABLE:
            return []

        raw = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if raw is None:
            return []

        result = []
        for p in raw:
            result.append(MT5Position(
                ticket=p.ticket,
                symbol=p.symbol,
                direction="BUY" if p.type == 0 else "SELL",
                lot_size=p.volume,
                open_price=p.price_open,
                stop_loss=p.sl,
                take_profit=p.tp,
                open_time=datetime.utcfromtimestamp(p.time),
                magic=p.magic,
                comment=p.comment,
                current_price=p.price_current,
                pnl=p.profit,
            ))
        return result

    # ------------------------------------------------------------------
    # Trailing stop
    # ------------------------------------------------------------------

    def apply_trailing_stop(
        self,
        ticket: int,
        trail_pips: float,
        activation_pips: float = 0.0,
    ) -> OrderResult:
        """
        Moves SL by `trail_pips` whenever price moves `trail_pips` in profit
        beyond the activation threshold. Call this on every tick/bar update.
        """
        positions = self.get_positions()
        pos = next((p for p in positions if p.ticket == ticket), None)
        if not pos:
            return OrderResult(success=False, error_message=f"Position {ticket} not found")

        if self.connector:
            tick = self.connector.fetch_tick(pos.symbol)
            current = tick.get("bid" if pos.direction == "BUY" else "ask", pos.current_price)
        else:
            current = pos.current_price

        sym_info = self.connector.symbol_info(pos.symbol) if self.connector else {}
        point = sym_info.get("point", 0.01)
        trail_dist = trail_pips * point * 10
        act_dist   = activation_pips * point * 10

        if pos.direction == "BUY":
            profit_pips = (current - pos.open_price) / (point * 10)
            if profit_pips < activation_pips:
                return OrderResult(success=True, ticket=ticket)  # not activated yet
            new_sl = current - trail_dist
            if new_sl <= pos.stop_loss:
                return OrderResult(success=True, ticket=ticket)  # no improvement
        else:
            profit_pips = (pos.open_price - current) / (point * 10)
            if profit_pips < activation_pips:
                return OrderResult(success=True, ticket=ticket)
            new_sl = current + trail_dist
            if new_sl >= pos.stop_loss:
                return OrderResult(success=True, ticket=ticket)

        logger.info(
            "TRAILING STOP ticket=%d | %s %s | old_sl=%.5f → new_sl=%.5f",
            ticket, pos.direction, pos.symbol, pos.stop_loss, new_sl,
        )
        return self.modify_position(ticket, new_sl=new_sl)

    # ------------------------------------------------------------------
    # Paper trading simulation
    # ------------------------------------------------------------------

    def _paper_open(
        self,
        symbol: str,
        direction: str,
        lot_size: float,
        stop_loss: float,
        take_profit: float,
        comment: str,
    ) -> OrderResult:
        ticket = self._next_paper_ticket
        self._next_paper_ticket += 1

        # Get current price from connector or use synthetic fallback
        if self.connector:
            tick = self.connector.fetch_tick(symbol)
            price = tick.get("ask" if direction == "BUY" else "bid", 0.0)
        else:
            price = 0.0

        pos = MT5Position(
            ticket=ticket,
            symbol=symbol,
            direction=direction,
            lot_size=lot_size,
            open_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            open_time=datetime.utcnow(),
            magic=self.config.mt5_magic,
            comment=comment,
            current_price=price,
        )
        self._paper_positions[ticket] = pos

        logger.info(
            "[PAPER] OPEN ticket=%d | %s %s %.2f lots @ %.5f | SL=%.5f TP=%.5f",
            ticket, direction, symbol, lot_size, price, stop_loss, take_profit,
        )
        return OrderResult(
            success=True, ticket=ticket, order_type=direction,
            symbol=symbol, volume=lot_size, price=price,
            sl=stop_loss, tp=take_profit, paper=True,
        )

    def _paper_close(self, ticket: int, reason: str) -> OrderResult:
        pos = self._paper_positions.get(ticket)
        if not pos or pos.closed:
            return OrderResult(success=False, error_message=f"Paper ticket {ticket} not found")

        if self.connector:
            tick = self.connector.fetch_tick(pos.symbol)
            close_price = tick.get("bid" if pos.direction == "BUY" else "ask", pos.current_price)
        else:
            close_price = pos.current_price

        sym_info = self.connector.symbol_info(pos.symbol) if self.connector else {}
        contract_size = sym_info.get("trade_contract_size", 100.0)

        if pos.direction == "BUY":
            pnl = (close_price - pos.open_price) * pos.lot_size * contract_size
        else:
            pnl = (pos.open_price - close_price) * pos.lot_size * contract_size

        pos.closed = True
        pos.close_price = close_price
        pos.close_time = datetime.utcnow()
        pos.close_reason = reason
        pos.pnl = pnl
        self._paper_balance += pnl

        logger.info(
            "[PAPER] CLOSE ticket=%d | %s %s | open=%.5f close=%.5f | P&L=%.2f | reason=%s | balance=%.2f",
            ticket, pos.direction, pos.symbol,
            pos.open_price, close_price, pnl, reason, self._paper_balance,
        )
        return OrderResult(success=True, ticket=ticket, price=close_price, paper=True)

    def _paper_modify(
        self,
        ticket: int,
        new_sl: Optional[float] = None,
        new_tp: Optional[float] = None,
    ) -> OrderResult:
        pos = self._paper_positions.get(ticket)
        if not pos:
            return OrderResult(success=False, error_message=f"Paper ticket {ticket} not found")
        if new_sl is not None:
            pos.stop_loss = new_sl
        if new_tp is not None:
            pos.take_profit = new_tp
        return OrderResult(success=True, ticket=ticket, paper=True)

    def _update_paper_pnl(self, positions: List[MT5Position]) -> None:
        """Update unrealised P&L for paper positions using live tick data."""
        for pos in positions:
            if not self.connector:
                continue
            try:
                tick = self.connector.fetch_tick(pos.symbol)
                sym_info = self.connector.symbol_info(pos.symbol)
                contract_size = sym_info.get("trade_contract_size", 100.0)
                current = tick.get("bid" if pos.direction == "BUY" else "ask", pos.open_price)
                pos.current_price = current
                if pos.direction == "BUY":
                    pos.pnl   = (current - pos.open_price) * pos.lot_size * contract_size
                    pos.pips  = (current - pos.open_price) / sym_info.get("point", 0.01) / 10
                else:
                    pos.pnl   = (pos.open_price - current) * pos.lot_size * contract_size
                    pos.pips  = (pos.open_price - current) / sym_info.get("point", 0.01) / 10
            except Exception as exc:
                logger.debug("P&L update failed for ticket %d: %s", pos.ticket, exc)

    def check_sl_tp(self, positions: Optional[List[MT5Position]] = None) -> List[int]:
        """
        Check paper positions for SL/TP hits. Returns list of closed tickets.
        Call this on every bar/tick update in paper mode.
        """
        if not self.config.mt5_paper:
            return []

        closed_tickets = []
        open_positions = positions or [p for p in self._paper_positions.values() if not p.closed]

        for pos in open_positions:
            if not self.connector:
                continue
            tick = self.connector.fetch_tick(pos.symbol)
            bid = tick.get("bid", pos.current_price)
            ask = tick.get("ask", pos.current_price)
            pos.current_price = bid if pos.direction == "BUY" else ask

            if pos.direction == "BUY":
                if pos.stop_loss > 0 and bid <= pos.stop_loss:
                    self._paper_close(pos.ticket, "sl")
                    closed_tickets.append(pos.ticket)
                elif pos.take_profit > 0 and bid >= pos.take_profit:
                    self._paper_close(pos.ticket, "tp")
                    closed_tickets.append(pos.ticket)
            else:
                if pos.stop_loss > 0 and ask >= pos.stop_loss:
                    self._paper_close(pos.ticket, "sl")
                    closed_tickets.append(pos.ticket)
                elif pos.take_profit > 0 and ask <= pos.take_profit:
                    self._paper_close(pos.ticket, "tp")
                    closed_tickets.append(pos.ticket)

        return closed_tickets

    @property
    def paper_balance(self) -> float:
        return self._paper_balance

    def paper_summary(self) -> str:
        """Print a summary of all paper trades."""
        all_pos = list(self._paper_positions.values())
        closed = [p for p in all_pos if p.closed]
        open_  = [p for p in all_pos if not p.closed]
        wins   = [p for p in closed if p.pnl > 0]
        losses = [p for p in closed if p.pnl <= 0]
        total_pnl = sum(p.pnl for p in closed)
        win_rate  = len(wins) / len(closed) * 100 if closed else 0

        lines = [
            "=" * 55,
            "  PAPER TRADING SUMMARY",
            "=" * 55,
            f"  Starting Capital : ${self.config.paper_trading_capital:,.2f}",
            f"  Current Balance  : ${self._paper_balance:,.2f}",
            f"  Total P&L        : ${total_pnl:+,.2f}",
            f"  Total Trades     : {len(closed)}",
            f"  Win Rate         : {win_rate:.1f}%",
            f"  Wins / Losses    : {len(wins)} / {len(losses)}",
            f"  Open Positions   : {len(open_)}",
            "=" * 55,
        ]
        if closed:
            lines.append("  Recent trades:")
            for p in sorted(closed, key=lambda x: x.close_time or datetime.min)[-5:]:
                lines.append(
                    f"    [{p.close_time.strftime('%m-%d %H:%M') if p.close_time else '?'}] "
                    f"{p.direction} {p.symbol} {p.lot_size}lot | "
                    f"P&L={p.pnl:+.2f} | reason={p.close_reason}"
                )
        return "\n".join(lines)
