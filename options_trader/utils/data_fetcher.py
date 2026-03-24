"""
Data Fetcher
Provides a unified interface to fetch OHLCV, options chain, and news data
from multiple providers (yfinance, Alpaca, Polygon, NewsAPI).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from options_trader.core.config import TradingConfig
from options_trader.core.models import MarketEvent, OptionContract, OptionType

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Abstracts data acquisition. Falls back gracefully if optional
    third-party packages are not installed.
    """

    def __init__(self, config: TradingConfig):
        self.config = config

    # ------------------------------------------------------------------
    # OHLCV price data
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        days: int = 120,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data. Returns DataFrame with columns:
        [Open, High, Low, Close, Volume] indexed by datetime.
        """
        if self.config.data_provider == "yfinance":
            return self._fetch_yfinance(symbol, days, interval)
        raise NotImplementedError(f"Provider {self.config.data_provider!r} not yet implemented")

    def _fetch_yfinance(
        self, symbol: str, days: int, interval: str
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
            end = datetime.utcnow()
            start = end - timedelta(days=days)
            df = yf.download(symbol, start=start, end=end, interval=interval, progress=False, auto_adjust=True)
            if df.empty:
                raise ValueError(f"No data returned for {symbol}")
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df.columns = [c.lower() for c in df.columns]
            logger.info("Fetched %d bars for %s from yfinance", len(df), symbol)
            return df
        except ImportError:
            logger.error("yfinance not installed. Run: pip install yfinance")
            raise

    # ------------------------------------------------------------------
    # Options chain
    # ------------------------------------------------------------------

    def fetch_options_chain(
        self, symbol: str, max_expirations: int = 3
    ) -> List[OptionContract]:
        """Fetch options chain contracts for the nearest expirations."""
        if self.config.data_provider == "yfinance":
            return self._fetch_chain_yfinance(symbol, max_expirations)
        raise NotImplementedError(f"Provider {self.config.data_provider!r} not yet implemented")

    def _fetch_chain_yfinance(
        self, symbol: str, max_expirations: int
    ) -> List[OptionContract]:
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            spot = ticker.fast_info.get("lastPrice") or ticker.fast_info.get("regularMarketPrice") or 0.0
            expirations = ticker.options[:max_expirations]
            contracts: List[OptionContract] = []

            for exp_str in expirations:
                chain = ticker.option_chain(exp_str)
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d")

                for _, row in chain.calls.iterrows():
                    contracts.append(self._row_to_contract(row, symbol, OptionType.CALL, exp_date, spot))
                for _, row in chain.puts.iterrows():
                    contracts.append(self._row_to_contract(row, symbol, OptionType.PUT, exp_date, spot))

            logger.info("Fetched %d option contracts for %s", len(contracts), symbol)
            return contracts
        except ImportError:
            logger.error("yfinance not installed.")
            raise

    @staticmethod
    def _row_to_contract(
        row: pd.Series,
        symbol: str,
        option_type: OptionType,
        exp_date: datetime,
        spot: float,
    ) -> OptionContract:
        return OptionContract(
            symbol=symbol,
            option_type=option_type,
            strike=float(row.get("strike", 0)),
            expiration=exp_date,
            bid=float(row.get("bid", 0)),
            ask=float(row.get("ask", 0)),
            last=float(row.get("lastPrice", 0)),
            volume=int(row.get("volume", 0) or 0),
            open_interest=int(row.get("openInterest", 0) or 0),
            implied_volatility=float(row.get("impliedVolatility", 0.20)),
            delta=float(row.get("delta", 0.0) or 0.0),
            gamma=float(row.get("gamma", 0.0) or 0.0),
            theta=float(row.get("theta", 0.0) or 0.0),
            vega=float(row.get("vega", 0.0) or 0.0),
            underlying_price=spot,
        )

    # ------------------------------------------------------------------
    # News
    # ------------------------------------------------------------------

    def fetch_news(
        self, symbol: str, max_articles: int = 20
    ) -> List[Dict]:
        """
        Returns list of {"headline": str, "sentiment": float, "date": datetime}.
        Sentiment is approximated from title keywords if no NLP library available.
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            news = ticker.news or []
            results = []
            for item in news[:max_articles]:
                headline = item.get("title", "")
                pub_ts = item.get("providerPublishTime", 0)
                date = datetime.utcfromtimestamp(pub_ts) if pub_ts else datetime.utcnow()
                sentiment = self._simple_sentiment(headline)
                results.append({"headline": headline, "sentiment": sentiment, "date": date})
            return results
        except Exception as exc:
            logger.warning("News fetch error for %s: %s", symbol, exc)
            return []

    @staticmethod
    def _simple_sentiment(text: str) -> float:
        """Very simple keyword sentiment score."""
        text_lower = text.lower()
        bullish_words = ["beat", "surge", "rally", "record", "strong", "growth", "upgrade",
                         "buy", "gain", "profit", "positive", "bullish", "rise", "up"]
        bearish_words = ["miss", "fall", "decline", "loss", "weak", "cut", "downgrade",
                         "sell", "drop", "bearish", "crash", "down", "concern", "risk"]
        score = 0.0
        for w in bullish_words:
            if w in text_lower:
                score += 0.1
        for w in bearish_words:
            if w in text_lower:
                score -= 0.1
        return max(-1.0, min(1.0, score))

    # ------------------------------------------------------------------
    # IV history (approximated from realised vol)
    # ------------------------------------------------------------------

    def fetch_historical_iv(
        self, symbol: str, ohlcv: pd.DataFrame, window: int = 20
    ) -> pd.Series:
        """
        Approximate historical IV from realised volatility (annualised).
        In production, replace with actual IV surface data.
        """
        log_returns = ohlcv["close"].apply(lambda x: x).pct_change().apply(
            lambda r: r if abs(r) < 1 else 0
        )
        realised_vol = log_returns.rolling(window).std() * (252 ** 0.5)
        # Add a small premium above realised vol to simulate IV risk premium
        iv_series = realised_vol * 1.15
        iv_series = iv_series.fillna(0.20).clip(0.05, 2.0)
        return iv_series
