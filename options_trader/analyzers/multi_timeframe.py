"""
Multi-Timeframe Confluence Engine
Provides:
- HTF bias (Daily/H4) always overrides LTF signals
- H4/Daily  → determines primary trend bias
- H1        → structure & entry zone
- M15       → precise entry trigger
- Confluence score: higher = more timeframes aligned
- MTF signal suppression: LTF against HTF bias = no trade
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from options_trader.core.models import SignalStrength, TechnicalSignal
from options_trader.core.config import TradingConfig

logger = logging.getLogger(__name__)

# Timeframe hierarchy (higher index = higher timeframe)
TF_HIERARCHY = ["M5", "M15", "M30", "H1", "H4", "D1", "W1"]
TF_WEIGHTS = {"M5": 0.05, "M15": 0.10, "M30": 0.12, "H1": 0.20, "H4": 0.28, "D1": 0.35, "W1": 0.40}


@dataclass
class TimeframeBias:
    """Directional bias for a single timeframe."""
    timeframe: str
    trend: str            # "bullish", "bearish", "neutral"
    score: float          # -2.0 to +2.0
    ema_aligned: bool     # EMA 9 > 21 > 50 (bull) or 9 < 21 < 50 (bear)
    above_vwap: bool
    rsi: float
    atr_pct: float        # ATR as % of price (volatility proxy)
    description: str = ""


@dataclass
class MTFResult:
    """Full multi-timeframe confluence result."""
    htf_bias: str                   # "bullish" / "bearish" / "neutral" (D1/H4 driven)
    ltf_aligned: bool               # True if H1/M15 agree with HTF
    confluence_score: float         # 0.0–1.0 (1 = all TFs perfectly aligned)
    timeframe_biases: Dict[str, TimeframeBias]
    dominant_trend: str             # overall label
    can_trade: bool                 # False if LTF contradicts HTF
    suppression_reason: str         # why trade is suppressed (if any)
    signal: TechnicalSignal         # final MTF signal


class MultiTimeframeAnalyzer:
    """
    Accepts OHLCV data at multiple timeframes and synthesizes them
    into a single HTF-dominant confluence score.

    The engine enforces the rule:
        HTF (H4/D1) bias  →  determines the ONLY valid trade direction.
        LTF (M15/H1)      →  provides entry timing.
        If LTF contradicts HTF → SUPPRESSED (no trade).
    """

    def __init__(self, config: TradingConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyze(
        self,
        symbol: str,
        ohlcv_by_tf: Dict[str, pd.DataFrame],
    ) -> MTFResult:
        """
        Args:
            ohlcv_by_tf: dict mapping timeframe label → OHLCV DataFrame
                         e.g. {"M15": df_m15, "H1": df_h1, "H4": df_h4, "D1": df_d1}

        Returns:
            MTFResult with confluence score and HTF-driven directional bias.
        """
        if not ohlcv_by_tf:
            return self._neutral_result(symbol)

        biases: Dict[str, TimeframeBias] = {}
        for tf, ohlcv in ohlcv_by_tf.items():
            if ohlcv is not None and len(ohlcv) >= 20:
                biases[tf] = self._analyze_single_tf(tf, ohlcv)

        if not biases:
            return self._neutral_result(symbol)

        htf_bias = self._compute_htf_bias(biases)
        ltf_aligned, suppression_reason = self._check_ltf_alignment(biases, htf_bias)
        confluence_score = self._compute_confluence(biases, htf_bias)
        dominant_trend = self._label_trend(htf_bias, confluence_score)
        can_trade = ltf_aligned and confluence_score >= 0.40

        signal = self._build_signal(htf_bias, confluence_score, ltf_aligned, biases)

        logger.info(
            "MTF %s | htf=%s | ltf_aligned=%s | confluence=%.2f | can_trade=%s",
            symbol, htf_bias, ltf_aligned, confluence_score, can_trade,
        )

        return MTFResult(
            htf_bias=htf_bias,
            ltf_aligned=ltf_aligned,
            confluence_score=confluence_score,
            timeframe_biases=biases,
            dominant_trend=dominant_trend,
            can_trade=can_trade,
            suppression_reason=suppression_reason,
            signal=signal,
        )

    # ------------------------------------------------------------------
    # Single timeframe analysis
    # ------------------------------------------------------------------

    def _analyze_single_tf(self, tf: str, ohlcv: pd.DataFrame) -> TimeframeBias:
        df = ohlcv.copy()
        df.columns = [c.lower() for c in df.columns]
        close = df["close"]
        high = df["high"]
        low = df["low"]

        n = len(df)

        # EMAs
        ema9  = close.ewm(span=9,  adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        ema50 = close.ewm(span=min(50, n - 1), adjust=False).mean()

        ema_bull = float(ema9.iloc[-1]) > float(ema21.iloc[-1]) > float(ema50.iloc[-1])
        ema_bear = float(ema9.iloc[-1]) < float(ema21.iloc[-1]) < float(ema50.iloc[-1])

        # VWAP (rolling 20 for non-session TFs)
        tp = (high + low + close) / 3
        vol = df["volume"]
        cum_tpv = (tp * vol).rolling(20).sum()
        cum_vol = vol.rolling(20).sum()
        vwap = cum_tpv / cum_vol.replace(0, np.nan)
        above_vwap = float(close.iloc[-1]) > float(vwap.iloc[-1]) if not np.isnan(vwap.iloc[-1]) else False

        # RSI
        rsi = self._rsi(close, period=14)
        rsi_val = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0

        # ATR %
        atr = self._atr(high, low, close, period=14)
        price = float(close.iloc[-1])
        atr_pct = float(atr.iloc[-1]) / price if price > 0 else 0.0

        # Score
        score = 0.0
        reasons = []

        if ema_bull:
            score += 1.0
            reasons.append("EMA bullish stack")
        elif ema_bear:
            score -= 1.0
            reasons.append("EMA bearish stack")
        else:
            reasons.append("EMA mixed")

        if above_vwap:
            score += 0.5
            reasons.append(f"above VWAP")
        else:
            score -= 0.5
            reasons.append(f"below VWAP")

        if rsi_val > 55:
            score += 0.4
            reasons.append(f"RSI {rsi_val:.0f} bullish")
        elif rsi_val < 45:
            score -= 0.4
            reasons.append(f"RSI {rsi_val:.0f} bearish")
        else:
            reasons.append(f"RSI {rsi_val:.0f} neutral")

        # Price vs 50-period high/low (trend proxy)
        lookback = min(50, n)
        period_high = float(high.iloc[-lookback:].max())
        period_low  = float(low.iloc[-lookback:].min())
        mid = (period_high + period_low) / 2
        if price > mid:
            score += 0.3
        else:
            score -= 0.3

        if score > 0.5:
            trend = "bullish"
        elif score < -0.5:
            trend = "bearish"
        else:
            trend = "neutral"

        return TimeframeBias(
            timeframe=tf,
            trend=trend,
            score=score,
            ema_aligned=ema_bull or ema_bear,
            above_vwap=above_vwap,
            rsi=rsi_val,
            atr_pct=atr_pct,
            description=f"[{tf}] {trend.upper()} | " + " | ".join(reasons),
        )

    # ------------------------------------------------------------------
    # HTF bias (D1 > H4 > H1)
    # ------------------------------------------------------------------

    def _compute_htf_bias(self, biases: Dict[str, TimeframeBias]) -> str:
        """Weight biases by TF hierarchy; highest TFs dominate."""
        score = 0.0
        total_weight = 0.0

        # Prefer higher timeframes
        htf_priority = ["D1", "W1", "H4", "H1"]
        for tf in htf_priority:
            if tf in biases:
                w = TF_WEIGHTS.get(tf, 0.20)
                score += biases[tf].score * w
                total_weight += w

        if total_weight == 0:
            # Fall back to whatever we have
            for tf, b in biases.items():
                w = TF_WEIGHTS.get(tf, 0.10)
                score += b.score * w
                total_weight += w

        if total_weight == 0:
            return "neutral"

        norm = score / total_weight
        if norm > 0.4:
            return "bullish"
        elif norm < -0.4:
            return "bearish"
        return "neutral"

    # ------------------------------------------------------------------
    # LTF alignment check
    # ------------------------------------------------------------------

    def _check_ltf_alignment(
        self, biases: Dict[str, TimeframeBias], htf_bias: str
    ) -> Tuple[bool, str]:
        """
        Returns (aligned, suppression_reason).
        SUPPRESSED if LTF (M15/H1) directly contradicts HTF.
        """
        if htf_bias == "neutral":
            return True, ""  # can't conflict with neutral

        ltf_frames = ["M15", "M5", "M30"]
        contradictions = []
        for tf in ltf_frames:
            if tf in biases:
                b = biases[tf]
                if htf_bias == "bullish" and b.trend == "bearish":
                    contradictions.append(f"{tf} bearish vs HTF bullish")
                elif htf_bias == "bearish" and b.trend == "bullish":
                    contradictions.append(f"{tf} bullish vs HTF bearish")

        # H1 contradiction is more severe
        if "H1" in biases:
            b = biases["H1"]
            if htf_bias == "bullish" and b.trend == "bearish":
                contradictions.append(f"H1 bearish contradicts HTF bullish bias")
            elif htf_bias == "bearish" and b.trend == "bullish":
                contradictions.append(f"H1 bullish contradicts HTF bearish bias")

        if contradictions:
            return False, "LTF/HTF conflict: " + "; ".join(contradictions)
        return True, ""

    # ------------------------------------------------------------------
    # Confluence score
    # ------------------------------------------------------------------

    def _compute_confluence(
        self, biases: Dict[str, TimeframeBias], htf_bias: str
    ) -> float:
        """
        Score from 0.0–1.0.
        1.0 = every available timeframe agrees with HTF bias.
        """
        if htf_bias == "neutral":
            return 0.3

        agreeing_weight = 0.0
        total_weight = 0.0

        for tf, b in biases.items():
            w = TF_WEIGHTS.get(tf, 0.10)
            total_weight += w
            if b.trend == htf_bias:
                agreeing_weight += w
            elif b.trend == "neutral":
                agreeing_weight += w * 0.4  # partial credit

        if total_weight == 0:
            return 0.3
        return min(1.0, agreeing_weight / total_weight)

    # ------------------------------------------------------------------
    # Signal synthesis
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        htf_bias: str,
        confluence: float,
        ltf_aligned: bool,
        biases: Dict[str, TimeframeBias],
    ) -> TechnicalSignal:
        if not ltf_aligned:
            signal = SignalStrength.NEUTRAL
            desc = f"MTF SUPPRESSED — LTF contradicts HTF {htf_bias} bias"
            return TechnicalSignal(indicator="MTF", value=0.0, signal=signal, description=desc)

        score = confluence * 2.0 if htf_bias == "bullish" else -confluence * 2.0 if htf_bias == "bearish" else 0.0

        if score >= 1.5:
            sig = SignalStrength.STRONG_BUY
        elif score >= 0.5:
            sig = SignalStrength.BUY
        elif score <= -1.5:
            sig = SignalStrength.STRONG_SELL
        elif score <= -0.5:
            sig = SignalStrength.SELL
        else:
            sig = SignalStrength.NEUTRAL

        tf_summary = ", ".join(
            f"{tf}:{b.trend[0].upper()}" for tf, b in sorted(biases.items(), key=lambda x: TF_HIERARCHY.index(x[0]) if x[0] in TF_HIERARCHY else 99)
        )
        desc = (
            f"MTF {htf_bias.upper()} | confluence={confluence:.0%} | aligned={ltf_aligned} | [{tf_summary}]"
        )
        return TechnicalSignal(indicator="MTF", value=score, signal=sig, description=desc)

    @staticmethod
    def _label_trend(htf_bias: str, confluence: float) -> str:
        if htf_bias == "neutral":
            return "ranging"
        strength = "strong" if confluence >= 0.7 else "weak"
        return f"{strength}_{htf_bias}_trend"

    # ------------------------------------------------------------------
    # Neutral fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _neutral_result(symbol: str) -> MTFResult:
        sig = TechnicalSignal(
            indicator="MTF", value=0.0, signal=SignalStrength.NEUTRAL,
            description="MTF: insufficient data"
        )
        return MTFResult(
            htf_bias="neutral",
            ltf_aligned=True,
            confluence_score=0.0,
            timeframe_biases={},
            dominant_trend="unknown",
            can_trade=False,
            suppression_reason="Insufficient multi-timeframe data",
            signal=sig,
        )

    # ------------------------------------------------------------------
    # Resample helper — build lower TF df into higher TF
    # ------------------------------------------------------------------

    @staticmethod
    def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        """
        Resample a higher-frequency OHLCV DataFrame to a lower frequency.
        rule examples: "4H", "1D", "1H", "15min"
        """
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        resampled = df.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()
        return resampled

    # ------------------------------------------------------------------
    # Indicator helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(com=period - 1, adjust=False).mean()
