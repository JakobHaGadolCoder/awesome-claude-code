"""
Technical Indicators Analyzer
Computes RSI, MACD, Bollinger Bands, VWAP, ATR, EMA confluence,
volume profile, and market-regime detection from OHLCV price data.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from options_trader.core.models import (
    MarketRegime,
    SignalStrength,
    TechnicalSignal,
)
from options_trader.core.config import TradingConfig

logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """
    Computes a suite of technical indicators and aggregates them into
    a composite directional score + market regime classification.

    Expected input: pandas DataFrame with columns
        ['open', 'high', 'low', 'close', 'volume']
    with a DatetimeIndex (daily or intraday OHLCV).
    """

    def __init__(self, config: TradingConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyze(
        self, symbol: str, ohlcv: pd.DataFrame
    ) -> Tuple[List[TechnicalSignal], MarketRegime, float]:
        """
        Returns:
            signals    – list of individual TechnicalSignal objects
            regime     – detected MarketRegime
            composite  – weighted composite score (-2 to +2)
        """
        if len(ohlcv) < max(self.config.macd_slow + self.config.macd_signal, 50):
            logger.warning("Insufficient data for %s (%d bars)", symbol, len(ohlcv))
            return [], MarketRegime.RANGING, 0.0

        ohlcv = ohlcv.copy()
        ohlcv.columns = [c.lower() for c in ohlcv.columns]

        signals: List[TechnicalSignal] = []

        signals.append(self._rsi_signal(ohlcv))
        signals.append(self._macd_signal(ohlcv))
        signals.append(self._bollinger_signal(ohlcv))
        signals.append(self._ema_confluence_signal(ohlcv))
        signals.append(self._vwap_signal(ohlcv))
        signals.append(self._volume_signal(ohlcv))
        signals.append(self._atr_regime_signal(ohlcv))

        regime = self._detect_regime(ohlcv, signals)
        composite = self._composite_score(signals)

        logger.info(
            "Technical %s | regime=%s | composite=%.2f | signals=%s",
            symbol,
            regime.value,
            composite,
            [(s.indicator, s.signal.name) for s in signals],
        )
        return signals, regime, composite

    # ------------------------------------------------------------------
    # RSI
    # ------------------------------------------------------------------

    def _rsi_signal(self, ohlcv: pd.DataFrame) -> TechnicalSignal:
        rsi = self._rsi(ohlcv["close"], self.config.rsi_period)
        val = float(rsi.iloc[-1])

        if val <= self.config.rsi_oversold:
            sig = SignalStrength.STRONG_BUY
            desc = f"RSI oversold ({val:.1f})"
        elif val <= 40:
            sig = SignalStrength.BUY
            desc = f"RSI low ({val:.1f})"
        elif val >= self.config.rsi_overbought:
            sig = SignalStrength.STRONG_SELL
            desc = f"RSI overbought ({val:.1f})"
        elif val >= 60:
            sig = SignalStrength.SELL
            desc = f"RSI elevated ({val:.1f})"
        else:
            sig = SignalStrength.NEUTRAL
            desc = f"RSI neutral ({val:.1f})"

        return TechnicalSignal(indicator="RSI", value=val, signal=sig, description=desc)

    # ------------------------------------------------------------------
    # MACD
    # ------------------------------------------------------------------

    def _macd_signal(self, ohlcv: pd.DataFrame) -> TechnicalSignal:
        macd_line, signal_line, histogram = self._macd(
            ohlcv["close"],
            self.config.macd_fast,
            self.config.macd_slow,
            self.config.macd_signal,
        )
        hist_now = float(histogram.iloc[-1])
        hist_prev = float(histogram.iloc[-2])
        macd_val = float(macd_line.iloc[-1])
        sig_val = float(signal_line.iloc[-1])

        cross_up = hist_prev < 0 <= hist_now
        cross_down = hist_prev > 0 >= hist_now
        momentum_up = hist_now > hist_prev > 0
        momentum_down = hist_now < hist_prev < 0

        if cross_up and macd_val > 0:
            sig = SignalStrength.STRONG_BUY
            desc = "MACD bullish crossover above zero"
        elif cross_up:
            sig = SignalStrength.BUY
            desc = "MACD bullish crossover below zero"
        elif momentum_up:
            sig = SignalStrength.BUY
            desc = f"MACD positive momentum (hist={hist_now:.4f})"
        elif cross_down and macd_val < 0:
            sig = SignalStrength.STRONG_SELL
            desc = "MACD bearish crossover below zero"
        elif cross_down:
            sig = SignalStrength.SELL
            desc = "MACD bearish crossover above zero"
        elif momentum_down:
            sig = SignalStrength.SELL
            desc = f"MACD negative momentum (hist={hist_now:.4f})"
        else:
            sig = SignalStrength.NEUTRAL
            desc = f"MACD neutral (hist={hist_now:.4f})"

        return TechnicalSignal(
            indicator="MACD", value=hist_now, signal=sig, description=desc
        )

    # ------------------------------------------------------------------
    # Bollinger Bands
    # ------------------------------------------------------------------

    def _bollinger_signal(self, ohlcv: pd.DataFrame) -> TechnicalSignal:
        upper, middle, lower = self._bollinger_bands(
            ohlcv["close"], self.config.bb_period, self.config.bb_std
        )
        close = float(ohlcv["close"].iloc[-1])
        upper_val = float(upper.iloc[-1])
        lower_val = float(lower.iloc[-1])
        mid_val = float(middle.iloc[-1])
        bandwidth = (upper_val - lower_val) / mid_val if mid_val else 0.0

        if close <= lower_val:
            sig = SignalStrength.STRONG_BUY
            desc = f"Price at/below lower BB (close={close:.2f}, lower={lower_val:.2f})"
        elif close < mid_val:
            sig = SignalStrength.BUY
            desc = f"Price below BB midline"
        elif close >= upper_val:
            sig = SignalStrength.STRONG_SELL
            desc = f"Price at/above upper BB (close={close:.2f}, upper={upper_val:.2f})"
        elif close > mid_val:
            sig = SignalStrength.SELL
            desc = "Price above BB midline"
        else:
            sig = SignalStrength.NEUTRAL
            desc = "Price at BB midline"

        return TechnicalSignal(
            indicator="BollingerBands",
            value=bandwidth,
            signal=sig,
            description=desc,
        )

    # ------------------------------------------------------------------
    # EMA Confluence
    # ------------------------------------------------------------------

    def _ema_confluence_signal(self, ohlcv: pd.DataFrame) -> TechnicalSignal:
        close = ohlcv["close"]
        ema9 = self._ema(close, self.config.ema_short).iloc[-1]
        ema21 = self._ema(close, self.config.ema_medium).iloc[-1]
        ema50 = self._ema(close, self.config.ema_long).iloc[-1]
        price = float(close.iloc[-1])

        bullish_stack = ema9 > ema21 > ema50
        bearish_stack = ema9 < ema21 < ema50
        price_above_all = price > ema50
        price_below_all = price < ema50

        if bullish_stack and price_above_all:
            sig = SignalStrength.STRONG_BUY
            desc = f"Bullish EMA stack: 9>{21}>{50}, price above all"
        elif bullish_stack:
            sig = SignalStrength.BUY
            desc = "Bullish EMA stack"
        elif bearish_stack and price_below_all:
            sig = SignalStrength.STRONG_SELL
            desc = f"Bearish EMA stack: 9<{21}<{50}, price below all"
        elif bearish_stack:
            sig = SignalStrength.SELL
            desc = "Bearish EMA stack"
        else:
            sig = SignalStrength.NEUTRAL
            desc = f"Mixed EMAs (9={ema9:.2f}, 21={ema21:.2f}, 50={ema50:.2f})"

        return TechnicalSignal(
            indicator="EMA_Confluence",
            value=float(ema9 - ema50),
            signal=sig,
            description=desc,
        )

    # ------------------------------------------------------------------
    # VWAP
    # ------------------------------------------------------------------

    def _vwap_signal(self, ohlcv: pd.DataFrame) -> TechnicalSignal:
        vwap = self._vwap(ohlcv)
        close = float(ohlcv["close"].iloc[-1])
        vwap_val = float(vwap.iloc[-1])
        dev_pct = (close - vwap_val) / vwap_val if vwap_val else 0.0

        if dev_pct > 0.02:
            sig = SignalStrength.SELL
            desc = f"Price {dev_pct*100:.1f}% above VWAP (extended)"
        elif dev_pct < -0.02:
            sig = SignalStrength.BUY
            desc = f"Price {abs(dev_pct)*100:.1f}% below VWAP (discount)"
        elif close > vwap_val:
            sig = SignalStrength.BUY
            desc = f"Price above VWAP ({close:.2f} vs {vwap_val:.2f})"
        elif close < vwap_val:
            sig = SignalStrength.SELL
            desc = f"Price below VWAP ({close:.2f} vs {vwap_val:.2f})"
        else:
            sig = SignalStrength.NEUTRAL
            desc = "Price at VWAP"

        return TechnicalSignal(
            indicator="VWAP", value=vwap_val, signal=sig, description=desc
        )

    # ------------------------------------------------------------------
    # Volume
    # ------------------------------------------------------------------

    def _volume_signal(self, ohlcv: pd.DataFrame) -> TechnicalSignal:
        vol = ohlcv["volume"]
        vol_avg = vol.rolling(20).mean()
        vol_ratio = float(vol.iloc[-1] / vol_avg.iloc[-1]) if vol_avg.iloc[-1] else 1.0
        close_change = float(
            ohlcv["close"].iloc[-1] - ohlcv["close"].iloc[-2]
        )

        # High volume confirms direction
        if vol_ratio > 1.5 and close_change > 0:
            sig = SignalStrength.STRONG_BUY
            desc = f"High volume up day ({vol_ratio:.1f}x avg)"
        elif vol_ratio > 1.5 and close_change < 0:
            sig = SignalStrength.STRONG_SELL
            desc = f"High volume down day ({vol_ratio:.1f}x avg)"
        elif vol_ratio > 1.2 and close_change > 0:
            sig = SignalStrength.BUY
            desc = f"Above-avg volume up day ({vol_ratio:.1f}x)"
        elif vol_ratio > 1.2 and close_change < 0:
            sig = SignalStrength.SELL
            desc = f"Above-avg volume down day ({vol_ratio:.1f}x)"
        elif vol_ratio < 0.7:
            sig = SignalStrength.NEUTRAL
            desc = f"Low volume ({vol_ratio:.1f}x avg), weak conviction"
        else:
            sig = SignalStrength.NEUTRAL
            desc = f"Normal volume ({vol_ratio:.1f}x avg)"

        return TechnicalSignal(
            indicator="Volume", value=vol_ratio, signal=sig, description=desc
        )

    # ------------------------------------------------------------------
    # ATR / Regime
    # ------------------------------------------------------------------

    def _atr_regime_signal(self, ohlcv: pd.DataFrame) -> TechnicalSignal:
        atr = self._atr(ohlcv, self.config.atr_period)
        atr_val = float(atr.iloc[-1])
        atr_pct = atr_val / float(ohlcv["close"].iloc[-1]) if ohlcv["close"].iloc[-1] else 0
        atr_avg = float(atr.rolling(20).mean().iloc[-1])
        atr_ratio = atr_val / atr_avg if atr_avg else 1.0

        if atr_ratio > 1.5:
            sig = SignalStrength.NEUTRAL  # High vol = caution not direction
            desc = f"High ATR ({atr_pct*100:.1f}% of price, {atr_ratio:.1f}x avg) - elevated risk"
        elif atr_ratio < 0.6:
            sig = SignalStrength.NEUTRAL
            desc = f"Low ATR ({atr_pct*100:.1f}% of price) - consolidating"
        else:
            sig = SignalStrength.NEUTRAL
            desc = f"Normal ATR ({atr_pct*100:.1f}% of price)"

        return TechnicalSignal(
            indicator="ATR", value=atr_pct, signal=sig, description=desc
        )

    # ------------------------------------------------------------------
    # Market Regime Detection
    # ------------------------------------------------------------------

    def _detect_regime(
        self, ohlcv: pd.DataFrame, signals: List[TechnicalSignal]
    ) -> MarketRegime:
        atr_sig = next((s for s in signals if s.indicator == "ATR"), None)
        ema_sig = next((s for s in signals if s.indicator == "EMA_Confluence"), None)

        if atr_sig and atr_sig.value > 0.025:
            return MarketRegime.HIGH_VOLATILITY

        close = ohlcv["close"]
        ema50 = self._ema(close, 50)
        slope = float(ema50.iloc[-1] - ema50.iloc[-10]) / float(ema50.iloc[-10]) if float(ema50.iloc[-10]) else 0

        if slope > 0.01:
            return MarketRegime.TRENDING_UP
        elif slope < -0.01:
            return MarketRegime.TRENDING_DOWN

        if atr_sig and atr_sig.value < 0.008:
            return MarketRegime.LOW_VOLATILITY

        return MarketRegime.RANGING

    # ------------------------------------------------------------------
    # Composite score
    # ------------------------------------------------------------------

    def _composite_score(self, signals: List[TechnicalSignal]) -> float:
        # Exclude ATR from directional composite (regime only)
        directional = [s for s in signals if s.indicator != "ATR"]
        if not directional:
            return 0.0
        scores = [s.signal.value for s in directional]
        return float(np.mean(scores))

    # ------------------------------------------------------------------
    # Low-level indicator computations
    # ------------------------------------------------------------------

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def _macd(
        series: pd.Series, fast: int, slow: int, signal_period: int
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def _bollinger_bands(
        series: pd.Series, period: int, num_std: float
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        middle = series.rolling(period).mean()
        std = series.rolling(period).std()
        upper = middle + num_std * std
        lower = middle - num_std * std
        return upper, middle, lower

    @staticmethod
    def _vwap(ohlcv: pd.DataFrame) -> pd.Series:
        typical_price = (ohlcv["high"] + ohlcv["low"] + ohlcv["close"]) / 3
        cumulative_tpv = (typical_price * ohlcv["volume"]).cumsum()
        cumulative_vol = ohlcv["volume"].cumsum()
        return cumulative_tpv / cumulative_vol.replace(0, np.nan)

    @staticmethod
    def _atr(ohlcv: pd.DataFrame, period: int) -> pd.Series:
        high = ohlcv["high"]
        low = ohlcv["low"]
        prev_close = ohlcv["close"].shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        return tr.ewm(com=period - 1, min_periods=period).mean()
