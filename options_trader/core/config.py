"""Trading bot configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TradingConfig:
    # --- Universe ---
    symbols: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "AAPL", "TSLA"])
    data_provider: str = "yfinance"        # "yfinance" | "alpaca" | "tradier" | "polygon"
    news_provider: str = "newsapi"         # "newsapi" | "benzinga" | "finnhub"

    # --- Risk limits ---
    max_position_size_pct: float = 0.05    # max 5% of portfolio per trade
    max_portfolio_risk_pct: float = 0.20   # max 20% total risk exposure
    min_risk_reward: float = 1.5
    min_confidence: float = 0.55
    max_spread_pct: float = 0.10           # reject contracts with >10% bid/ask spread

    # --- Options filters ---
    min_dte: int = 7                       # minimum days to expiration
    max_dte: int = 45                      # maximum days to expiration
    target_delta_range: tuple = (0.25, 0.45)  # preferred delta range
    min_open_interest: int = 100
    min_volume: int = 10

    # --- Signal weights (must sum to 1.0) ---
    weight_order_flow: float = 0.30
    weight_technical: float = 0.30
    weight_sr_levels: float = 0.20
    weight_events: float = 0.20

    # --- Order flow thresholds ---
    unusual_volume_multiplier: float = 2.0   # X times average = unusual
    pcr_bullish_threshold: float = 0.7
    pcr_bearish_threshold: float = 1.3
    min_block_trade_size: int = 100          # contracts for a "block trade"
    min_block_premium: float = 50_000        # $50k minimum for unusual premium flag

    # --- Technical indicator settings ---
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    vwap_session: str = "daily"
    atr_period: int = 14
    ema_short: int = 9
    ema_medium: int = 21
    ema_long: int = 50

    # --- Support/Resistance ---
    sr_lookback_days: int = 90
    sr_min_touches: int = 2
    sr_zone_pct: float = 0.005             # 0.5% zone around a level

    # --- Events ---
    earnings_iv_expansion_threshold: float = 0.10
    high_impact_days_window: int = 3       # flag events within 3 days

    # --- Broker / execution ---
    broker: str = "paper"                  # "paper" | "alpaca" | "tradier" | "ibkr"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    paper_trading_capital: float = 100_000.0

    # --- Logging ---
    log_level: str = "INFO"
    log_file: Optional[str] = "options_trader.log"

    def validate(self) -> None:
        total_weight = (
            self.weight_order_flow
            + self.weight_technical
            + self.weight_sr_levels
            + self.weight_events
        )
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Signal weights must sum to 1.0, got {total_weight:.4f}")
        if self.min_dte >= self.max_dte:
            raise ValueError("min_dte must be less than max_dte")
        if not (0 < self.min_confidence < 1):
            raise ValueError("min_confidence must be between 0 and 1")
