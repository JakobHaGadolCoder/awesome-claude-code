"""Core data models for the options trading bot."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class OptionType(Enum):
    CALL = "CALL"
    PUT = "PUT"


class SignalStrength(Enum):
    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class OptionContract:
    symbol: str
    option_type: OptionType
    strike: float
    expiration: datetime
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    underlying_price: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread_pct(self) -> float:
        if self.mid_price == 0:
            return 0.0
        return (self.ask - self.bid) / self.mid_price

    @property
    def moneyness(self) -> str:
        if self.underlying_price == 0:
            return "unknown"
        ratio = self.underlying_price / self.strike
        if self.option_type == OptionType.CALL:
            if ratio > 1.02:
                return "ITM"
            elif ratio < 0.98:
                return "OTM"
            return "ATM"
        else:
            if ratio < 0.98:
                return "ITM"
            elif ratio > 1.02:
                return "OTM"
            return "ATM"


@dataclass
class OrderFlowData:
    """Represents aggregated order flow metrics."""
    symbol: str
    timestamp: datetime
    # Volume metrics
    total_volume: int
    call_volume: int
    put_volume: int
    # Large trade (block) activity
    call_premium_spent: float  # total $ spent on calls
    put_premium_spent: float   # total $ spent on puts
    # Unusual activity flags
    unusual_call_activity: bool = False
    unusual_put_activity: bool = False
    # Dark pool / institutional prints
    dark_pool_volume: int = 0
    dark_pool_bullish: bool = False
    # Skew metrics
    iv_skew: float = 0.0        # put IV - call IV (positive = put skew / bearish fear)
    term_structure_slope: float = 0.0  # near-term IV vs far-term IV

    @property
    def put_call_ratio(self) -> float:
        if self.call_volume == 0:
            return float("inf")
        return self.put_volume / self.call_volume

    @property
    def net_premium_flow(self) -> float:
        """Positive = net bullish (more $ in calls), Negative = net bearish."""
        return self.call_premium_spent - self.put_premium_spent

    @property
    def flow_bias(self) -> SignalStrength:
        pcr = self.put_call_ratio
        net = self.net_premium_flow
        if pcr < 0.5 and net > 0:
            return SignalStrength.STRONG_BUY
        elif pcr < 0.7 and net > 0:
            return SignalStrength.BUY
        elif pcr > 1.5 and net < 0:
            return SignalStrength.STRONG_SELL
        elif pcr > 1.2 and net < 0:
            return SignalStrength.SELL
        return SignalStrength.NEUTRAL


@dataclass
class MarketEvent:
    """A real-world event that may impact the underlying."""
    name: str
    event_type: str           # "earnings", "fed_meeting", "economic_data", "geopolitical", "news"
    scheduled_date: datetime
    importance: str           # "high", "medium", "low"
    expected_impact: float    # estimated IV expansion (e.g. 0.05 = 5% IV bump)
    actual_surprise: Optional[float] = None   # post-event: beat/miss magnitude
    description: str = ""
    symbol: Optional[str] = None   # None = macro (affects all)

    @property
    def days_until(self) -> int:
        delta = self.scheduled_date - datetime.utcnow()
        return max(0, delta.days)

    @property
    def is_high_impact(self) -> bool:
        return self.importance == "high"


@dataclass
class TechnicalSignal:
    """Output from a single technical indicator."""
    indicator: str
    value: float
    signal: SignalStrength
    description: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SupportResistanceLevel:
    """A key price level in the underlying."""
    price: float
    level_type: str       # "support" or "resistance"
    strength: float       # 0.0 - 1.0, how many times tested/respected
    timeframe: str        # "intraday", "daily", "weekly", "monthly"
    last_tested: Optional[datetime] = None
    description: str = ""

    @property
    def distance_from(self) -> float:
        """Returns raw price distance (caller provides current price)."""
        return self.price

    def distance_pct(self, current_price: float) -> float:
        if current_price == 0:
            return 0.0
        return abs(current_price - self.price) / current_price


@dataclass
class TradeSignal:
    """A fully composed options trade recommendation."""
    symbol: str
    option_type: OptionType
    action: str                   # "BUY" or "SELL"
    strike: float
    expiration: datetime
    strength: SignalStrength
    confidence: float             # 0.0 - 1.0
    entry_price: float
    target_price: float
    stop_loss: float
    max_loss: float
    max_gain: float
    risk_reward: float
    rationale: str
    # Contributing signals
    order_flow_score: float = 0.0
    technical_score: float = 0.0
    event_score: float = 0.0
    sr_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_valid(self) -> bool:
        return (
            self.confidence >= 0.5
            and self.risk_reward >= 1.5
            and self.strength != SignalStrength.NEUTRAL
        )
