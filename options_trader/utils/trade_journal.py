"""
Trade Journal & Performance Tracker
====================================
Logs every signal and trade outcome to a SQLite database.
Tracks performance by:
- Signal source (which analyzer fired)
- Pattern type (exhaustion, VWAP reclaim, divergence, etc.)
- Session (London, NY, Asian)
- Timeframe
- Day of week / time of day

After every N trades, auto-recalibrates signal weights based on
which signals have the best win rate on actual trade history.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DB_PATH = Path("options_trader_journal.db")
RECALIBRATE_EVERY = 20   # re-tune weights every N closed trades


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class JournalEntry:
    """A single paper/live trade record."""
    id: Optional[int]
    symbol: str
    direction: str              # "LONG" | "SHORT"
    timeframe: str              # "M15" | "H1" etc.
    session: str                # "london" | "ny" | "asian" | "overlap"
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    confidence: float
    entry_time: datetime

    # Signal attribution (which signals fired)
    signals_fired: Dict[str, str]   # {"order_flow": "BUY", "vwap": "STRONG_BUY", ...}
    patterns: List[str]             # ["BullishEngulfing", "VWAPReclaim"]
    exhaustion_detected: bool = False
    divergence_detected: bool = False
    mtf_aligned: bool = False
    session_quality: str = "normal"  # "prime" | "normal" | "avoid"

    # Outcome (filled after close)
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None   # "target1" | "target2" | "stop" | "manual"
    pnl_pips: Optional[float] = None
    pnl_rr: Optional[float] = None      # R:R achieved
    outcome: Optional[str] = None       # "win" | "loss" | "breakeven"
    notes: str = ""


@dataclass
class PerformanceStats:
    symbol: str
    period: str
    total_trades: int
    wins: int
    losses: int
    breakevens: int
    win_rate: float
    avg_win_rr: float
    avg_loss_rr: float
    profit_factor: float
    expectancy: float            # avg R per trade
    max_consecutive_losses: int
    max_consecutive_wins: int
    best_session: str
    worst_session: str
    best_pattern: str
    signal_accuracy: Dict[str, float]   # per-signal win rate
    suggested_weight_adjustments: Dict[str, float]


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------

class TradeJournal:
    """
    SQLite-backed trade journal with performance analytics
    and automatic signal weight recalibration.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    # ------------------------------------------------------------------
    # Database setup
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol          TEXT NOT NULL,
                    direction       TEXT NOT NULL,
                    timeframe       TEXT,
                    session         TEXT,
                    entry_price     REAL,
                    stop_loss       REAL,
                    target_1        REAL,
                    target_2        REAL,
                    confidence      REAL,
                    entry_time      TEXT,
                    signals_fired   TEXT,
                    patterns        TEXT,
                    exhaustion      INTEGER DEFAULT 0,
                    divergence      INTEGER DEFAULT 0,
                    mtf_aligned     INTEGER DEFAULT 0,
                    session_quality TEXT,
                    exit_price      REAL,
                    exit_time       TEXT,
                    exit_reason     TEXT,
                    pnl_pips        REAL,
                    pnl_rr          REAL,
                    outcome         TEXT,
                    notes           TEXT DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS weight_history (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT,
                    weights     TEXT,
                    based_on_n  INTEGER,
                    notes       TEXT
                )
            """)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    # ------------------------------------------------------------------
    # Logging trades
    # ------------------------------------------------------------------

    def log_entry(self, entry: JournalEntry) -> int:
        """Log a new trade entry. Returns the trade ID."""
        with self._conn() as conn:
            cursor = conn.execute("""
                INSERT INTO trades (
                    symbol, direction, timeframe, session,
                    entry_price, stop_loss, target_1, target_2, confidence,
                    entry_time, signals_fired, patterns,
                    exhaustion, divergence, mtf_aligned, session_quality, notes
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                entry.symbol, entry.direction, entry.timeframe, entry.session,
                entry.entry_price, entry.stop_loss, entry.target_1, entry.target_2,
                entry.confidence,
                entry.entry_time.isoformat(),
                json.dumps(entry.signals_fired),
                json.dumps(entry.patterns),
                int(entry.exhaustion_detected),
                int(entry.divergence_detected),
                int(entry.mtf_aligned),
                entry.session_quality,
                entry.notes,
            ))
            trade_id = cursor.lastrowid
            logger.info("Journal: logged entry #%d %s %s @ %.2f",
                        trade_id, entry.direction, entry.symbol, entry.entry_price)
            return trade_id

    def log_exit(
        self,
        trade_id: int,
        exit_price: float,
        exit_reason: str,
        notes: str = "",
    ) -> Optional[PerformanceStats]:
        """
        Close a trade, compute P&L, and trigger recalibration if needed.
        Returns updated PerformanceStats if recalibration threshold reached.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM trades WHERE id=?", (trade_id,)
            ).fetchone()
            if not row:
                logger.warning("Journal: trade #%d not found", trade_id)
                return None

            cols = [d[0] for d in conn.execute("PRAGMA table_info(trades)").fetchall()]
            trade = dict(zip(cols, row))

            entry_price = trade["entry_price"]
            stop_loss = trade["stop_loss"]
            direction = trade["direction"]

            risk = abs(entry_price - stop_loss)
            if direction == "LONG":
                pnl_pips = exit_price - entry_price
            else:
                pnl_pips = entry_price - exit_price

            pnl_rr = pnl_pips / risk if risk > 0 else 0.0
            outcome = "win" if pnl_pips > 0.1 * risk else ("loss" if pnl_pips < -0.1 * risk else "breakeven")

            conn.execute("""
                UPDATE trades SET
                    exit_price=?, exit_time=?, exit_reason=?,
                    pnl_pips=?, pnl_rr=?, outcome=?, notes=?
                WHERE id=?
            """, (
                exit_price,
                datetime.utcnow().isoformat(),
                exit_reason,
                round(pnl_pips, 2),
                round(pnl_rr, 3),
                outcome,
                notes,
                trade_id,
            ))

            logger.info(
                "Journal: closed #%d | outcome=%s | pips=%+.1f | R=%+.2f | reason=%s",
                trade_id, outcome, pnl_pips, pnl_rr, exit_reason
            )

        # Check if recalibration threshold reached
        closed = self._count_closed(trade["symbol"])
        if closed > 0 and closed % RECALIBRATE_EVERY == 0:
            return self.recalibrate_weights(trade["symbol"])
        return None

    # ------------------------------------------------------------------
    # Performance analytics
    # ------------------------------------------------------------------

    def get_stats(
        self,
        symbol: Optional[str] = None,
        days: int = 30,
    ) -> PerformanceStats:
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        with self._conn() as conn:
            q = "SELECT * FROM trades WHERE outcome IS NOT NULL AND entry_time > ?"
            params: list = [since]
            if symbol:
                q += " AND symbol=?"
                params.append(symbol)
            rows = conn.execute(q, params).fetchall()
            cols = [d[0] for d in conn.execute("PRAGMA table_info(trades)").fetchall()]

        trades = [dict(zip(cols, r)) for r in rows]
        if not trades:
            return self._empty_stats(symbol or "ALL", f"{days}d")

        wins = [t for t in trades if t["outcome"] == "win"]
        losses = [t for t in trades if t["outcome"] == "loss"]
        bes = [t for t in trades if t["outcome"] == "breakeven"]

        win_rrs = [t["pnl_rr"] for t in wins if t["pnl_rr"]]
        loss_rrs = [abs(t["pnl_rr"]) for t in losses if t["pnl_rr"]]

        avg_win = float(np.mean(win_rrs)) if win_rrs else 0.0
        avg_loss = float(np.mean(loss_rrs)) if loss_rrs else 1.0
        profit_factor = (len(wins) * avg_win) / (len(losses) * avg_loss) if losses else float("inf")
        win_rate = len(wins) / len(trades)
        expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

        # Consecutive runs
        max_consec_loss = self._max_consecutive(trades, "loss")
        max_consec_win = self._max_consecutive(trades, "win")

        # Session performance
        session_wr = self._win_rate_by(trades, "session")
        best_session = max(session_wr, key=session_wr.get) if session_wr else "unknown"
        worst_session = min(session_wr, key=session_wr.get) if session_wr else "unknown"

        # Pattern performance
        pattern_wr = self._pattern_win_rates(trades)
        best_pattern = max(pattern_wr, key=pattern_wr.get) if pattern_wr else "unknown"

        # Signal accuracy
        signal_accuracy = self._signal_win_rates(trades)

        # Weight suggestions
        weight_adj = self._suggest_weight_adjustments(signal_accuracy)

        return PerformanceStats(
            symbol=symbol or "ALL",
            period=f"{days}d",
            total_trades=len(trades),
            wins=len(wins),
            losses=len(losses),
            breakevens=len(bes),
            win_rate=round(win_rate, 3),
            avg_win_rr=round(avg_win, 2),
            avg_loss_rr=round(avg_loss, 2),
            profit_factor=round(profit_factor, 2),
            expectancy=round(expectancy, 3),
            max_consecutive_losses=max_consec_loss,
            max_consecutive_wins=max_consec_win,
            best_session=best_session,
            worst_session=worst_session,
            best_pattern=best_pattern,
            signal_accuracy=signal_accuracy,
            suggested_weight_adjustments=weight_adj,
        )

    def print_stats(self, symbol: Optional[str] = None, days: int = 30) -> None:
        s = self.get_stats(symbol, days)
        print(f"\n{'='*60}")
        print(f"  TRADE JOURNAL — {s.symbol} | Last {s.period}")
        print(f"{'='*60}")
        print(f"  Trades:          {s.total_trades} ({s.wins}W / {s.losses}L / {s.breakevens}BE)")
        print(f"  Win Rate:        {s.win_rate*100:.1f}%")
        print(f"  Avg Win:         +{s.avg_win_rr:.2f}R")
        print(f"  Avg Loss:        -{s.avg_loss_rr:.2f}R")
        print(f"  Profit Factor:   {s.profit_factor:.2f}")
        print(f"  Expectancy:      {s.expectancy:+.3f}R per trade")
        print(f"  Max Consec Loss: {s.max_consecutive_losses}")
        print(f"  Best Session:    {s.best_session}")
        print(f"  Best Pattern:    {s.best_pattern}")
        if s.signal_accuracy:
            print(f"  Signal Win Rates:")
            for sig, wr in sorted(s.signal_accuracy.items(), key=lambda x: -x[1]):
                print(f"    {sig:20s} {wr*100:.0f}%")
        if s.suggested_weight_adjustments:
            print(f"  Suggested Weight Adjustments:")
            for k, v in s.suggested_weight_adjustments.items():
                arrow = "↑" if v > 0 else "↓"
                print(f"    {k:20s} {arrow} {abs(v)*100:.0f}%")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Auto weight recalibration
    # ------------------------------------------------------------------

    def recalibrate_weights(
        self, symbol: str, min_trades_per_signal: int = 5
    ) -> Optional[PerformanceStats]:
        """
        After RECALIBRATE_EVERY trades, compute per-signal win rates
        and suggest adjusted weights. Logs the adjustment to DB.
        """
        stats = self.get_stats(symbol, days=180)
        if stats.total_trades < RECALIBRATE_EVERY:
            return None

        adj = stats.suggested_weight_adjustments
        if adj:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO weight_history (timestamp, weights, based_on_n, notes) VALUES (?,?,?,?)",
                    (
                        datetime.utcnow().isoformat(),
                        json.dumps(adj),
                        stats.total_trades,
                        f"Auto-recalibration after {stats.total_trades} trades",
                    ),
                )
            logger.info(
                "Journal: weight recalibration after %d trades: %s",
                stats.total_trades, adj
            )

        return stats

    def get_weight_history(self) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT timestamp, weights, based_on_n FROM weight_history ORDER BY id DESC LIMIT 10"
            ).fetchall()
        return [{"timestamp": r[0], "weights": json.loads(r[1]), "based_on_n": r[2]} for r in rows]

    # ------------------------------------------------------------------
    # Helper analytics
    # ------------------------------------------------------------------

    @staticmethod
    def _max_consecutive(trades: List[Dict], outcome: str) -> int:
        max_run = current = 0
        for t in trades:
            if t["outcome"] == outcome:
                current += 1
                max_run = max(max_run, current)
            else:
                current = 0
        return max_run

    @staticmethod
    def _win_rate_by(trades: List[Dict], field: str) -> Dict[str, float]:
        groups: Dict[str, list] = {}
        for t in trades:
            key = t.get(field, "unknown")
            groups.setdefault(key, []).append(t["outcome"] == "win")
        return {k: float(np.mean(v)) for k, v in groups.items() if len(v) >= 3}

    @staticmethod
    def _pattern_win_rates(trades: List[Dict]) -> Dict[str, float]:
        pattern_wins: Dict[str, list] = {}
        for t in trades:
            try:
                pats = json.loads(t.get("patterns", "[]"))
            except Exception:
                pats = []
            for p in pats:
                pattern_wins.setdefault(p, []).append(t["outcome"] == "win")
        return {k: float(np.mean(v)) for k, v in pattern_wins.items() if len(v) >= 3}

    @staticmethod
    def _signal_win_rates(trades: List[Dict]) -> Dict[str, float]:
        signal_wins: Dict[str, list] = {}
        for t in trades:
            try:
                sigs = json.loads(t.get("signals_fired", "{}"))
            except Exception:
                sigs = {}
            won = t["outcome"] == "win"
            for sig_name, sig_val in sigs.items():
                if sig_val in ("BUY", "STRONG_BUY", "SELL", "STRONG_SELL"):
                    signal_wins.setdefault(sig_name, []).append(won)
        return {k: float(np.mean(v)) for k, v in signal_wins.items() if len(v) >= 3}

    @staticmethod
    def _suggest_weight_adjustments(
        signal_accuracy: Dict[str, float],
        baseline_wr: float = 0.50,
    ) -> Dict[str, float]:
        """
        If a signal has >60% win rate when it fires → increase its weight.
        If <40% win rate → decrease its weight.
        Returns delta adjustments (e.g. +0.05 means increase weight by 5%).
        """
        adjustments = {}
        for sig, wr in signal_accuracy.items():
            if wr > 0.60:
                adjustments[sig] = round(min(0.10, (wr - baseline_wr) * 0.3), 3)
            elif wr < 0.40:
                adjustments[sig] = round(max(-0.10, (wr - baseline_wr) * 0.3), 3)
        return adjustments

    def _count_closed(self, symbol: str) -> int:
        with self._conn() as conn:
            r = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE symbol=? AND outcome IS NOT NULL", (symbol,)
            ).fetchone()
            return r[0] if r else 0

    @staticmethod
    def _empty_stats(symbol: str, period: str) -> PerformanceStats:
        return PerformanceStats(
            symbol=symbol, period=period, total_trades=0,
            wins=0, losses=0, breakevens=0, win_rate=0.0,
            avg_win_rr=0.0, avg_loss_rr=0.0, profit_factor=0.0,
            expectancy=0.0, max_consecutive_losses=0, max_consecutive_wins=0,
            best_session="N/A", worst_session="N/A", best_pattern="N/A",
            signal_accuracy={}, suggested_weight_adjustments={},
        )
