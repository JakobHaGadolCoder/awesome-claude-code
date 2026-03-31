"""
Session Filter
Provides:
- Detects current trading session: Asian / London / New York / London-NY Overlap
- Session-specific characteristics for XAUUSD / SPY / FX
- Filters out low-liquidity periods (late Asian, pre-London)
- Recommends trade bias adjustments per session
- Overlap detection (London/NY = highest liquidity, widest range)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, time
from typing import Optional, Tuple

from options_trader.core.models import SignalStrength, TechnicalSignal
from options_trader.core.config import TradingConfig

logger = logging.getLogger(__name__)


# Session boundaries in UTC
SESSION_HOURS = {
    "asian":     (time(0, 0),  time(8, 59)),    # 00:00–08:59 UTC
    "london":    (time(7, 0),  time(15, 59)),    # 07:00–15:59 UTC (pre-open + main)
    "new_york":  (time(13, 0), time(21, 59)),    # 13:00–21:59 UTC
    "overlap":   (time(13, 0), time(16, 59)),    # 13:00–16:59 UTC (London/NY overlap)
    "after_hours":(time(22, 0), time(23, 59)),   # 22:00–23:59 UTC
}

# Session characteristics per instrument type
SESSION_PROFILES = {
    "xauusd": {
        "asian":      {"liquidity": 0.35, "trend_reliability": 0.40, "range_pct": 0.003,  "notes": "Low vol, false breakouts common, range-bound"},
        "london":     {"liquidity": 0.75, "trend_reliability": 0.72, "range_pct": 0.007,  "notes": "London open sets daily bias; strong directional moves"},
        "new_york":   {"liquidity": 0.80, "trend_reliability": 0.70, "range_pct": 0.008,  "notes": "NY drives Gold on USD/risk sentiment; key economic releases"},
        "overlap":    {"liquidity": 1.00, "trend_reliability": 0.82, "range_pct": 0.012,  "notes": "HIGHEST LIQUIDITY — best entries, cleanest signals"},
        "after_hours":{"liquidity": 0.20, "trend_reliability": 0.25, "range_pct": 0.002,  "notes": "Avoid: very thin, spike risk"},
    },
    "equities": {
        "asian":      {"liquidity": 0.15, "trend_reliability": 0.30, "range_pct": 0.002,  "notes": "Asian session has minimal US equity flow"},
        "london":     {"liquidity": 0.55, "trend_reliability": 0.60, "range_pct": 0.005,  "notes": "Pre-market; European sentiment leads"},
        "new_york":   {"liquidity": 0.95, "trend_reliability": 0.80, "range_pct": 0.010,  "notes": "Primary session; all major catalyst events"},
        "overlap":    {"liquidity": 1.00, "trend_reliability": 0.85, "range_pct": 0.012,  "notes": "Best liquidity and trend clarity"},
        "after_hours":{"liquidity": 0.10, "trend_reliability": 0.20, "range_pct": 0.003,  "notes": "Avoid: earnings gaps but unreliable"},
    },
    "default": {
        "asian":      {"liquidity": 0.40, "trend_reliability": 0.45, "range_pct": 0.004,  "notes": "Moderate liquidity"},
        "london":     {"liquidity": 0.70, "trend_reliability": 0.68, "range_pct": 0.007,  "notes": "Active European session"},
        "new_york":   {"liquidity": 0.85, "trend_reliability": 0.75, "range_pct": 0.009,  "notes": "Primary US session"},
        "overlap":    {"liquidity": 1.00, "trend_reliability": 0.82, "range_pct": 0.011,  "notes": "Peak hours"},
        "after_hours":{"liquidity": 0.15, "trend_reliability": 0.20, "range_pct": 0.002,  "notes": "Thin"},
    },
}

# Minimum liquidity score to allow a new entry
MIN_LIQUIDITY_THRESHOLD = 0.35


@dataclass
class SessionInfo:
    """Current session state and characteristics."""
    session_name: str              # "asian", "london", "new_york", "overlap", "after_hours"
    utc_time: datetime
    is_overlap: bool               # London/NY overlap
    is_tradeable: bool             # Above minimum liquidity threshold
    liquidity_score: float         # 0.0 – 1.0
    trend_reliability: float       # how reliable are breakouts in this session
    expected_range_pct: float      # expected % range for the session
    notes: str
    signal: TechnicalSignal


class SessionFilter:
    """
    Determines the current trading session and whether conditions
    are suitable for new entries.

    XAUUSD-aware: knows that London and NY overlap produces the
    cleanest, highest-probability Gold moves.
    """

    def __init__(self, config: TradingConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyze(
        self,
        symbol: str,
        dt: Optional[datetime] = None,
    ) -> SessionInfo:
        """
        Args:
            symbol: ticker symbol (used to select session profile)
            dt: datetime in UTC; if None, uses datetime.utcnow()

        Returns:
            SessionInfo with session name, liquidity, and trade filter signal.
        """
        if dt is None:
            dt = datetime.utcnow().replace(tzinfo=timezone.utc)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        current_time = dt.time()
        session = self._identify_session(current_time)
        profile_key = self._get_profile_key(symbol)
        profile = SESSION_PROFILES[profile_key][session]

        is_overlap = self._is_overlap(current_time)
        is_tradeable = profile["liquidity"] >= MIN_LIQUIDITY_THRESHOLD

        signal = self._build_signal(session, profile, is_overlap, is_tradeable, symbol)

        logger.info(
            "Session %s | %s | session=%s | liquidity=%.2f | tradeable=%s | overlap=%s",
            symbol, dt.strftime("%H:%M UTC"), session,
            profile["liquidity"], is_tradeable, is_overlap,
        )

        return SessionInfo(
            session_name=session,
            utc_time=dt,
            is_overlap=is_overlap,
            is_tradeable=is_tradeable,
            liquidity_score=profile["liquidity"],
            trend_reliability=profile["trend_reliability"],
            expected_range_pct=profile["range_pct"],
            notes=profile["notes"],
            signal=signal,
        )

    # ------------------------------------------------------------------
    # Session identification
    # ------------------------------------------------------------------

    @staticmethod
    def _identify_session(t: time) -> str:
        """Return session name for a given UTC time."""
        # Overlap takes priority
        ol_start, ol_end = SESSION_HOURS["overlap"]
        if ol_start <= t <= ol_end:
            return "overlap"

        ny_start, ny_end = SESSION_HOURS["new_york"]
        if ny_start <= t <= ny_end:
            return "new_york"

        lon_start, lon_end = SESSION_HOURS["london"]
        if lon_start <= t <= lon_end:
            return "london"

        asi_start, asi_end = SESSION_HOURS["asian"]
        if asi_start <= t <= asi_end:
            return "asian"

        return "after_hours"

    @staticmethod
    def _is_overlap(t: time) -> bool:
        ol_start, ol_end = SESSION_HOURS["overlap"]
        return ol_start <= t <= ol_end

    @staticmethod
    def _get_profile_key(symbol: str) -> str:
        sym = symbol.upper()
        if "XAU" in sym or "GOLD" in sym:
            return "xauusd"
        if sym in ("SPY", "QQQ", "IWM", "DIA") or sym.endswith("US"):
            return "equities"
        return "default"

    # ------------------------------------------------------------------
    # Signal synthesis
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        session: str,
        profile: dict,
        is_overlap: bool,
        is_tradeable: bool,
        symbol: str,
    ) -> TechnicalSignal:
        liquidity = profile["liquidity"]
        reliability = profile["trend_reliability"]

        if not is_tradeable:
            sig = SignalStrength.NEUTRAL
            desc = (
                f"SESSION FILTER: {session.upper()} — LOW LIQUIDITY ({liquidity:.0%}). "
                f"Avoid new entries. {profile['notes']}"
            )
            return TechnicalSignal(indicator="Session", value=0.0, signal=sig, description=desc)

        # Score based on liquidity + reliability
        score = (liquidity * 0.6 + reliability * 0.4) * 2.0  # 0–2 range

        if is_overlap:
            score = min(2.0, score * 1.15)
            sig = SignalStrength.STRONG_BUY if score >= 1.5 else SignalStrength.BUY
            desc = (
                f"SESSION: LONDON/NY OVERLAP — PEAK CONDITIONS | "
                f"liquidity={liquidity:.0%} | reliability={reliability:.0%} | {profile['notes']}"
            )
        elif session in ("london", "new_york"):
            sig = SignalStrength.BUY if score >= 1.0 else SignalStrength.NEUTRAL
            desc = (
                f"SESSION: {session.upper()} | "
                f"liquidity={liquidity:.0%} | reliability={reliability:.0%} | {profile['notes']}"
            )
        else:
            sig = SignalStrength.NEUTRAL
            desc = (
                f"SESSION: {session.upper()} | "
                f"liquidity={liquidity:.0%} | reliability={reliability:.0%} | {profile['notes']}"
            )

        return TechnicalSignal(indicator="Session", value=score, signal=sig, description=desc)

    # ------------------------------------------------------------------
    # Utility: is current time a high-value window?
    # ------------------------------------------------------------------

    @staticmethod
    def is_high_value_window(dt: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Returns (True, reason) if within a high-value entry window:
        - London open (07:00–08:30 UTC)
        - NY open (13:30–15:00 UTC)
        - London/NY overlap (13:00–17:00 UTC)
        """
        if dt is None:
            dt = datetime.utcnow()
        t = dt.time()

        if time(7, 0) <= t <= time(8, 30):
            return True, "London Open (07:00–08:30 UTC) — high momentum window"
        if time(13, 0) <= t <= time(16, 59):
            return True, "London/NY Overlap (13:00–17:00 UTC) — peak liquidity window"
        if time(13, 30) <= t <= time(14, 30):
            return True, "NY Open (13:30–14:30 UTC) — institutional order flow window"
        return False, f"Current time {t.strftime('%H:%M')} UTC is outside high-value windows"
