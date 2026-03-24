"""
Real-World Events Analyzer
Tracks and scores impact of:
- Earnings announcements
- Fed/FOMC meetings
- Economic data releases (CPI, NFP, GDP, etc.)
- Geopolitical events
- Sector-specific news
- Analyst upgrades/downgrades
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from options_trader.core.models import (
    MarketEvent,
    SignalStrength,
    TechnicalSignal,
)
from options_trader.core.config import TradingConfig

logger = logging.getLogger(__name__)


# Approximate historical IV expansion multipliers by event type
EVENT_IV_MULTIPLIERS: Dict[str, float] = {
    "earnings": 0.15,          # +15% IV on avg into earnings
    "fed_meeting": 0.05,
    "cpi": 0.04,
    "nfp": 0.03,
    "gdp": 0.02,
    "geopolitical": 0.08,
    "analyst_rating": 0.03,
    "product_launch": 0.04,
    "fda_approval": 0.20,
    "merger_acquisition": 0.25,
    "news": 0.02,
}

# Known macro events calendar (simplified static calendar for demo)
MACRO_EVENTS_TEMPLATE: List[Dict] = [
    {"name": "FOMC Meeting", "event_type": "fed_meeting", "importance": "high"},
    {"name": "CPI Release", "event_type": "cpi", "importance": "high"},
    {"name": "Non-Farm Payrolls", "event_type": "nfp", "importance": "high"},
    {"name": "GDP Release", "event_type": "gdp", "importance": "medium"},
    {"name": "PCE Inflation", "event_type": "pceinflation", "importance": "high"},
    {"name": "Initial Jobless Claims", "event_type": "jobless_claims", "importance": "medium"},
    {"name": "Retail Sales", "event_type": "retail_sales", "importance": "medium"},
]


class EventsAnalyzer:
    """
    Analyses upcoming and recent real-world events to:
    1. Flag high-impact events within the next N days
    2. Score the expected directional bias post-event
    3. Assess whether to avoid or exploit IV expansion
    4. Parse news sentiment for directional bias
    """

    def __init__(self, config: TradingConfig):
        self.config = config
        self._events: List[MarketEvent] = []
        self._news_cache: Dict[str, List[Dict]] = {}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyze(
        self,
        symbol: str,
        events: Optional[List[MarketEvent]] = None,
        news_headlines: Optional[List[Dict]] = None,
    ) -> Tuple[List[MarketEvent], TechnicalSignal]:
        """
        Args:
            symbol: ticker symbol
            events: pre-populated list of known upcoming events
            news_headlines: list of {"headline": str, "sentiment": float (-1 to 1), "date": datetime}

        Returns:
            relevant_events: events relevant to this symbol in the window
            signal: TechnicalSignal based on event risk/opportunity
        """
        all_events = list(events or []) + self._events
        relevant = self._filter_relevant(symbol, all_events)
        upcoming = self._filter_upcoming(relevant)

        news_signal = self._analyze_news(symbol, news_headlines or [])
        event_signal = self._build_event_signal(symbol, upcoming, news_signal)

        logger.info(
            "Events %s | upcoming=%d | news_bias=%.2f | signal=%s",
            symbol,
            len(upcoming),
            news_signal,
            event_signal.signal.name,
        )
        return upcoming, event_signal

    # ------------------------------------------------------------------
    # Event registration
    # ------------------------------------------------------------------

    def add_event(self, event: MarketEvent) -> None:
        self._events.append(event)
        logger.debug("Event added: %s on %s", event.name, event.scheduled_date.date())

    def add_earnings(
        self,
        symbol: str,
        date: datetime,
        expected_move_pct: float = 0.05,
        consensus_direction: str = "neutral",
    ) -> MarketEvent:
        """Convenience method to register an earnings event."""
        event = MarketEvent(
            name=f"{symbol} Earnings",
            event_type="earnings",
            scheduled_date=date,
            importance="high",
            expected_impact=expected_move_pct,
            description=f"Earnings release - consensus: {consensus_direction}",
            symbol=symbol,
        )
        self.add_event(event)
        return event

    def add_macro_events_for_month(
        self, year: int, month: int, dates: Dict[str, datetime]
    ) -> None:
        """
        Register standard monthly macro events.
        dates: {"FOMC Meeting": datetime(...), "CPI Release": datetime(...), ...}
        """
        for template in MACRO_EVENTS_TEMPLATE:
            if template["name"] in dates:
                event = MarketEvent(
                    name=template["name"],
                    event_type=template["event_type"],
                    scheduled_date=dates[template["name"]],
                    importance=template["importance"],
                    expected_impact=EVENT_IV_MULTIPLIERS.get(template["event_type"], 0.02),
                    description=f"Scheduled macro release",
                    symbol=None,  # Affects all
                )
                self.add_event(event)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _filter_relevant(
        self, symbol: str, events: List[MarketEvent]
    ) -> List[MarketEvent]:
        """Return events that apply to this symbol (or global macro events)."""
        return [e for e in events if e.symbol is None or e.symbol == symbol]

    def _filter_upcoming(
        self, events: List[MarketEvent]
    ) -> List[MarketEvent]:
        """Return events scheduled within the high_impact_days_window."""
        now = datetime.utcnow()
        cutoff = now + timedelta(days=self.config.high_impact_days_window)
        return [e for e in events if now <= e.scheduled_date <= cutoff]

    # ------------------------------------------------------------------
    # News sentiment analysis
    # ------------------------------------------------------------------

    def _analyze_news(
        self, symbol: str, headlines: List[Dict]
    ) -> float:
        """
        Returns a sentiment score -1.0 (very bearish) to +1.0 (very bullish).
        Each headline dict: {"headline": str, "sentiment": float, "date": datetime}
        Uses recency weighting (more recent = higher weight).
        """
        if not headlines:
            return 0.0

        now = datetime.utcnow()
        weighted_sum = 0.0
        weight_total = 0.0

        for h in headlines:
            sentiment = h.get("sentiment", 0.0)
            date = h.get("date", now)
            hours_old = max(1, (now - date).total_seconds() / 3600)
            # Exponential decay: half-life ~24 hours
            weight = 2 ** (-hours_old / 24)
            weighted_sum += sentiment * weight
            weight_total += weight

        return weighted_sum / weight_total if weight_total > 0 else 0.0

    # ------------------------------------------------------------------
    # Signal synthesis
    # ------------------------------------------------------------------

    def _build_event_signal(
        self,
        symbol: str,
        upcoming_events: List[MarketEvent],
        news_sentiment: float,
    ) -> TechnicalSignal:
        score = 0.0
        reasons = []

        # --- High-impact upcoming events ---
        high_impact = [e for e in upcoming_events if e.is_high_impact]
        if high_impact:
            # High-impact events increase uncertainty; mute directional signals
            # unless we have a strong news/sentiment lean
            days_out = min(e.days_until for e in high_impact)
            iv_expansion = max(e.expected_impact for e in high_impact)
            event_names = ", ".join(e.name for e in high_impact[:3])
            reasons.append(
                f"High-impact event in {days_out}d: {event_names} "
                f"(IV expansion ~{iv_expansion*100:.0f}%)"
            )
            # Penalise directional trades near binary events
            score -= 0.3

        # --- Earnings-specific logic ---
        earnings = [e for e in upcoming_events if e.event_type == "earnings"]
        if earnings:
            e = earnings[0]
            days = e.days_until
            if days <= 1:
                reasons.append(f"Earnings TOMORROW - avoid directional, prefer straddle/strangle")
                score = 0.0  # Override to neutral for binary events
            elif days <= 3:
                reasons.append(f"Earnings in {days}d - elevated IV risk")
                score -= 0.2
            else:
                reasons.append(f"Earnings in {days}d - monitor IV buildup")

        # --- News sentiment ---
        if news_sentiment > 0.3:
            score += news_sentiment
            reasons.append(f"Positive news sentiment ({news_sentiment:.2f})")
        elif news_sentiment < -0.3:
            score += news_sentiment  # negative adds negative
            reasons.append(f"Negative news sentiment ({news_sentiment:.2f})")
        elif abs(news_sentiment) > 0.1:
            reasons.append(f"Mild news sentiment ({news_sentiment:.2f})")

        # --- Fed / macro days ---
        fed_days = [e for e in upcoming_events if e.event_type == "fed_meeting"]
        if fed_days:
            reasons.append(f"FOMC meeting in {fed_days[0].days_until}d - heightened vol")
            score -= 0.1

        signal = self._score_to_signal(score)
        return TechnicalSignal(
            indicator="Events",
            value=score,
            signal=signal,
            description=" | ".join(reasons) if reasons else "No major events detected",
        )

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

    # ------------------------------------------------------------------
    # Post-event analysis (after surprise reveal)
    # ------------------------------------------------------------------

    def process_event_surprise(
        self,
        event: MarketEvent,
        actual_surprise: float,  # positive = beat, negative = miss
        price_move_pct: float,
    ) -> TechnicalSignal:
        """
        Analyse post-event reaction. Large moves after earnings =
        momentum signal for options plays.
        """
        event.actual_surprise = actual_surprise
        reasons = []

        if actual_surprise > 0.10:
            score = 1.5
            reasons.append(f"Strong beat ({actual_surprise*100:.1f}%), price +{price_move_pct*100:.1f}%")
        elif actual_surprise > 0:
            score = 0.5
            reasons.append(f"Modest beat ({actual_surprise*100:.1f}%), price {price_move_pct*100:+.1f}%")
        elif actual_surprise < -0.10:
            score = -1.5
            reasons.append(f"Big miss ({actual_surprise*100:.1f}%), price {price_move_pct*100:.1f}%")
        else:
            score = -0.5
            reasons.append(f"Slight miss ({actual_surprise*100:.1f}%)")

        # Sell-the-news if big gap and price already moved
        if abs(price_move_pct) > 0.08:
            reasons.append("Large gap - expect IV crush, consider selling premium")

        signal = self._score_to_signal(score)
        return TechnicalSignal(
            indicator="EventSurprise",
            value=score,
            signal=signal,
            description=" | ".join(reasons),
        )
