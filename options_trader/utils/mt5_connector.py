"""
MetaTrader 5 Data Connector
============================
Drop-in replacement for DataFetcher that sources all price data directly
from a running MT5 terminal instead of yfinance.

Requirements:
  pip install MetaTrader5
  MT5 terminal must be running on the same machine (Windows) or via Wine (Linux).

Implements the same public interface as DataFetcher:
  fetch_ohlcv(symbol, days, interval) -> pd.DataFrame
  fetch_options_chain(symbol) -> List[OptionContract]   (stub — MT5 has no options)
  fetch_news(symbol) -> List[dict]                      (stub)
  fetch_historical_iv(symbol, ohlcv) -> pd.Series

Additional MT5-specific methods:
  connect() / disconnect()
  fetch_ohlcv_mt5(symbol, timeframe_str, bars) -> pd.DataFrame
  fetch_ohlcv_multi_tf(symbol) -> Dict[str, pd.DataFrame]
  fetch_tick(symbol) -> dict   {bid, ask, last, time}
  fetch_account_info() -> dict
  symbol_info(symbol) -> dict
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from options_trader.core.config import TradingConfig
from options_trader.core.models import OptionContract

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MT5 import — graceful fallback if package not installed or terminal offline
# ---------------------------------------------------------------------------
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None  # type: ignore
    MT5_AVAILABLE = False
    logger.warning(
        "MetaTrader5 package not found. Install with: pip install MetaTrader5\n"
        "MT5Connector will run in SIMULATION mode using synthetic data."
    )

# ---------------------------------------------------------------------------
# Timeframe string → MT5 constant mapping
# ---------------------------------------------------------------------------
TF_MAP: Dict[str, int] = {}
if MT5_AVAILABLE:
    TF_MAP = {
        "M1":  mt5.TIMEFRAME_M1,
        "M2":  mt5.TIMEFRAME_M2,
        "M3":  mt5.TIMEFRAME_M3,
        "M4":  mt5.TIMEFRAME_M4,
        "M5":  mt5.TIMEFRAME_M5,
        "M6":  mt5.TIMEFRAME_M6,
        "M10": mt5.TIMEFRAME_M10,
        "M12": mt5.TIMEFRAME_M12,
        "M15": mt5.TIMEFRAME_M15,
        "M20": mt5.TIMEFRAME_M20,
        "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,
        "H2":  mt5.TIMEFRAME_H2,
        "H3":  mt5.TIMEFRAME_H3,
        "H4":  mt5.TIMEFRAME_H4,
        "H6":  mt5.TIMEFRAME_H6,
        "H8":  mt5.TIMEFRAME_H8,
        "H12": mt5.TIMEFRAME_H12,
        "D1":  mt5.TIMEFRAME_D1,
        "W1":  mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1,
    }

# Approx minutes per timeframe (for days → bars conversion)
TF_MINUTES: Dict[str, int] = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D1": 1440, "W1": 10080,
}


class MT5Connector:
    """
    Connects to a live MT5 terminal and fetches real-time OHLCV data,
    tick prices, and account information.

    Usage:
        config = TradingConfig(
            mt5_login=12345678,
            mt5_password="yourpassword",
            mt5_server="ICMarkets-Live",
        )
        conn = MT5Connector(config)
        conn.connect()
        df = conn.fetch_ohlcv("XAUUSD", days=5, interval="M15")
        tick = conn.fetch_tick("XAUUSD")
        conn.disconnect()
    """

    def __init__(self, config: TradingConfig):
        self.config = config
        self._connected = False

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """
        Initialise the MT5 connection.
        Returns True on success, False on failure.
        On Linux/Mac without MT5 installed, returns False and falls back
        to simulation mode.
        """
        if not MT5_AVAILABLE:
            logger.warning("MT5 not available — running in simulation mode")
            self._connected = False
            return False

        kwargs: dict = {}
        if self.config.mt5_terminal_path:
            kwargs["path"] = self.config.mt5_terminal_path
        if self.config.mt5_login:
            kwargs["login"] = self.config.mt5_login
        if self.config.mt5_password:
            kwargs["password"] = self.config.mt5_password
        if self.config.mt5_server:
            kwargs["server"] = self.config.mt5_server

        if not mt5.initialize(**kwargs):
            err = mt5.last_error()
            logger.error("MT5 initialize failed: %s", err)
            self._connected = False
            return False

        info = mt5.terminal_info()
        acct  = mt5.account_info()
        logger.info(
            "MT5 connected | terminal=%s | broker=%s | account=%s | balance=%.2f %s",
            getattr(info, "name", "?"),
            getattr(acct, "company", "?"),
            getattr(acct, "login", "?"),
            getattr(acct, "balance", 0),
            getattr(acct, "currency", "?"),
        )
        self._connected = True
        return True

    def disconnect(self) -> None:
        if MT5_AVAILABLE and self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("MT5 disconnected")

    @property
    def is_connected(self) -> bool:
        if not MT5_AVAILABLE or not self._connected:
            return False
        try:
            return mt5.terminal_info() is not None
        except Exception:
            return False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()

    # ------------------------------------------------------------------
    # DataFetcher-compatible interface
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        days: int = 120,
        interval: str = "D1",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars for `symbol` over the last `days` calendar days.
        `interval` uses MT5 timeframe strings: M1/M5/M15/M30/H1/H4/D1/W1.

        Falls back to synthetic data if MT5 is not connected.
        """
        if not self.is_connected:
            logger.warning("MT5 not connected — returning synthetic data for %s", symbol)
            return self._synthetic_ohlcv(symbol, days)

        tf_str = interval.upper()
        tf_const = TF_MAP.get(tf_str)
        if tf_const is None:
            raise ValueError(f"Unknown MT5 timeframe: {interval}. Valid: {list(TF_MAP)}")

        tf_mins = TF_MINUTES.get(tf_str, 60)
        bars_needed = int(days * 1440 / tf_mins) + 50  # extra buffer

        rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, bars_needed)
        if rates is None or len(rates) == 0:
            err = mt5.last_error()
            logger.error("MT5 copy_rates_from_pos failed for %s/%s: %s", symbol, tf_str, err)
            return self._synthetic_ohlcv(symbol, days)

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time").rename(columns={
            "open": "open", "high": "high", "low": "low",
            "close": "close", "tick_volume": "volume",
        })[["open", "high", "low", "close", "volume"]]

        # Filter to requested days
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        df = df[df.index >= cutoff]

        logger.info(
            "MT5 fetch_ohlcv %s/%s | bars=%d | from=%s to=%s",
            symbol, tf_str, len(df),
            df.index[0].strftime("%Y-%m-%d %H:%M") if len(df) else "N/A",
            df.index[-1].strftime("%Y-%m-%d %H:%M") if len(df) else "N/A",
        )
        return df

    def fetch_ohlcv_mt5(
        self,
        symbol: str,
        timeframe_str: str,
        bars: int = 500,
    ) -> pd.DataFrame:
        """
        Fetch a fixed number of bars for a given timeframe.
        More direct than fetch_ohlcv when you know exactly how many bars you need.
        """
        if not self.is_connected:
            return self._synthetic_ohlcv(symbol, bars // 24)

        tf_const = TF_MAP.get(timeframe_str.upper())
        if tf_const is None:
            raise ValueError(f"Unknown timeframe: {timeframe_str}")

        rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, bars)
        if rates is None or len(rates) == 0:
            logger.error("MT5 failed to fetch %s %s bars=%d", symbol, timeframe_str, bars)
            return self._synthetic_ohlcv(symbol, 30)

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time").rename(columns={"tick_volume": "volume"})[
            ["open", "high", "low", "close", "volume"]
        ]
        return df

    def fetch_ohlcv_multi_tf(
        self,
        symbol: str,
        timeframes: Optional[List[str]] = None,
        bars_per_tf: int = 300,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV for multiple timeframes in one call.
        Returns dict: {"M15": df_m15, "H1": df_h1, "H4": df_h4, "D1": df_d1}

        This feeds directly into MultiTimeframeAnalyzer.
        """
        tfs = timeframes or self.config.mt5_timeframes
        result: Dict[str, pd.DataFrame] = {}
        for tf in tfs:
            try:
                result[tf] = self.fetch_ohlcv_mt5(symbol, tf, bars=bars_per_tf)
                logger.debug("MT5 MTF %s/%s: %d bars", symbol, tf, len(result[tf]))
            except Exception as exc:
                logger.warning("MT5 MTF fetch failed for %s/%s: %s", symbol, tf, exc)
        return result

    def fetch_tick(self, symbol: str) -> dict:
        """
        Returns current bid/ask/last for symbol.
        dict keys: bid, ask, last, spread, time
        """
        if not self.is_connected:
            return {"bid": 0.0, "ask": 0.0, "last": 0.0, "spread": 0.0, "time": datetime.utcnow()}

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error("MT5 symbol_info_tick failed for %s", symbol)
            return {"bid": 0.0, "ask": 0.0, "last": 0.0, "spread": 0.0, "time": datetime.utcnow()}

        return {
            "bid":    tick.bid,
            "ask":    tick.ask,
            "last":   tick.last,
            "spread": round(tick.ask - tick.bid, 5),
            "time":   datetime.utcfromtimestamp(tick.time),
        }

    def fetch_account_info(self) -> dict:
        """Returns account balance, equity, margin, free_margin, currency."""
        if not self.is_connected:
            return {
                "balance": self.config.paper_trading_capital,
                "equity":  self.config.paper_trading_capital,
                "margin": 0.0, "free_margin": self.config.paper_trading_capital,
                "currency": "USD", "leverage": 100,
            }
        acct = mt5.account_info()
        if acct is None:
            return {}
        return {
            "balance":     acct.balance,
            "equity":      acct.equity,
            "margin":      acct.margin,
            "free_margin": acct.margin_free,
            "currency":    acct.currency,
            "leverage":    acct.leverage,
            "login":       acct.login,
            "server":      acct.server,
            "company":     acct.company,
        }

    def symbol_info(self, symbol: str) -> dict:
        """
        Returns symbol properties:
        point, digits, trade_contract_size, volume_min, volume_step, volume_max
        """
        if not self.is_connected:
            # Sensible defaults for XAUUSD
            return {
                "point": 0.01, "digits": 2,
                "trade_contract_size": 100.0,
                "volume_min": 0.01, "volume_step": 0.01, "volume_max": 500.0,
                "description": symbol,
            }
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.error("MT5 symbol_info failed for %s", symbol)
            return {}
        return {
            "point":               info.point,
            "digits":              info.digits,
            "trade_contract_size": info.trade_contract_size,
            "volume_min":          info.volume_min,
            "volume_step":         info.volume_step,
            "volume_max":          info.volume_max,
            "description":         info.description,
            "currency_base":       info.currency_base,
            "currency_profit":     info.currency_profit,
            "spread":              info.spread,
        }

    def calculate_lot_size(
        self,
        symbol: str,
        stop_loss_pips: float,
        risk_pct: Optional[float] = None,
    ) -> float:
        """
        Calculate lot size based on account balance and risk %.

        Formula:
            lot = (balance × risk_pct) / (stop_loss_pips × pip_value_per_lot)

        For XAUUSD: 1 lot = 100 oz, pip = $0.01, pip value ≈ $1/lot/pip
        """
        rp = risk_pct or self.config.mt5_risk_pct
        acct = self.fetch_account_info()
        balance = acct.get("balance", self.config.paper_trading_capital)
        risk_amount = balance * rp

        sym = self.symbol_info(symbol)
        contract_size = sym.get("trade_contract_size", 100.0)
        point = sym.get("point", 0.01)

        # pip value per lot = contract_size × point
        pip_value_per_lot = contract_size * point
        if pip_value_per_lot == 0 or stop_loss_pips == 0:
            return self.config.mt5_min_lot

        lot = risk_amount / (stop_loss_pips * pip_value_per_lot)
        lot = round(lot / sym.get("volume_step", 0.01)) * sym.get("volume_step", 0.01)
        lot = max(self.config.mt5_min_lot, min(self.config.mt5_max_lot, lot))
        return round(lot, 2)

    # ------------------------------------------------------------------
    # DataFetcher stub methods (MT5 has no options chain or news API)
    # ------------------------------------------------------------------

    def fetch_options_chain(self, symbol: str, max_expirations: int = 3) -> list:
        """MT5 doesn't provide options chains. Returns empty list."""
        logger.debug("fetch_options_chain: MT5 is a CFD platform — no options chain available")
        return []

    def fetch_news(self, symbol: str, max_articles: int = 20) -> List[dict]:
        """MT5 doesn't have a news API. Returns empty list."""
        return []

    def fetch_historical_iv(
        self, symbol: str, ohlcv: pd.DataFrame, window: int = 20
    ) -> pd.Series:
        """Approximate IV from realised volatility (same as DataFetcher)."""
        log_ret = np.log(ohlcv["close"] / ohlcv["close"].shift(1))
        realised_vol = log_ret.rolling(window).std() * np.sqrt(252)
        return (realised_vol * 1.15).rename("iv")

    # ------------------------------------------------------------------
    # Simulation fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _synthetic_ohlcv(symbol: str, days: int, seed: int = 42) -> pd.DataFrame:
        """
        Generates realistic-looking synthetic OHLCV for testing when
        MT5 terminal is not available.
        """
        rng = np.random.default_rng(seed)
        base = 4500.0 if "XAU" in symbol.upper() or "GOLD" in symbol.upper() else 100.0
        n = max(days, 30)
        price = base
        prices = [price]
        for _ in range(n - 1):
            price *= np.exp(rng.normal(0.0001, 0.008))
            prices.append(price)

        prices = np.array(prices)
        highs  = prices * (1 + rng.uniform(0.001, 0.004, n))
        lows   = prices * (1 - rng.uniform(0.001, 0.004, n))
        opens  = prices * (1 + rng.normal(0, 0.001, n))
        vols   = rng.integers(500, 5000, n).astype(float)

        idx = pd.date_range(end=datetime.utcnow(), periods=n, freq="h", tz="UTC")
        return pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": prices, "volume": vols},
            index=idx,
        )
