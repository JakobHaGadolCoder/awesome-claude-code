"""
Price Action Analyzer
Detects:
- Candlestick patterns: hammer, shooting star, doji, engulfing, pin bar, inside bar
- Market structure: Higher Highs/Higher Lows (bullish) vs Lower Highs/Lower Lows (bearish)
- Break of Structure (BOS) and Change of Character (CHoCH)
- Order Blocks (last bullish/bearish candle before a strong impulsive move)
- Fair Value Gaps / Imbalances (FVG)
- Swing point identification
- Momentum / impulsive vs corrective move classification
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from options_trader.core.models import SignalStrength, TechnicalSignal
from options_trader.core.config import TradingConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CandlePattern:
    name: str
    candle_index: int          # position in the OHLCV DataFrame
    direction: str             # "bullish" | "bearish" | "neutral"
    strength: float            # 0.0 – 1.0
    description: str = ""


@dataclass
class SwingPoint:
    price: float
    index: int
    kind: str                  # "high" | "low"
    timestamp: Optional[datetime] = None


@dataclass
class OrderBlock:
    price_high: float
    price_low: float
    direction: str             # "bullish" (demand) | "bearish" (supply)
    index: int
    mitigated: bool = False
    description: str = ""

    @property
    def midpoint(self) -> float:
        return (self.price_high + self.price_low) / 2


@dataclass
class FairValueGap:
    """3-candle imbalance: gap between candle[i-2] high and candle[i] low (bullish FVG)."""
    price_high: float
    price_low: float
    direction: str             # "bullish" | "bearish"
    index: int
    filled: bool = False

    @property
    def size(self) -> float:
        return abs(self.price_high - self.price_low)

    @property
    def midpoint(self) -> float:
        return (self.price_high + self.price_low) / 2


@dataclass
class MarketStructure:
    trend: str                 # "bullish" | "bearish" | "ranging"
    last_bos: Optional[float] = None       # price level of last BOS
    last_choch: Optional[float] = None     # price level of last CHoCH
    swing_highs: List[SwingPoint] = field(default_factory=list)
    swing_lows: List[SwingPoint] = field(default_factory=list)
    higher_highs: bool = False
    higher_lows: bool = False
    lower_highs: bool = False
    lower_lows: bool = False


@dataclass
class PriceActionResult:
    patterns: List[CandlePattern]
    structure: MarketStructure
    order_blocks: List[OrderBlock]
    fvgs: List[FairValueGap]
    signal: TechnicalSignal
    nearest_ob: Optional[OrderBlock] = None
    nearest_fvg: Optional[FairValueGap] = None


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class PriceActionAnalyzer:
    """
    Full price action analysis suite.
    Input: OHLCV DataFrame with lowercase columns [open, high, low, close, volume].
    """

    def __init__(self, config: TradingConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyze(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        current_price: Optional[float] = None,
    ) -> PriceActionResult:
        ohlcv = ohlcv.copy()
        ohlcv.columns = [c.lower() for c in ohlcv.columns]

        cp = current_price if current_price is not None else float(ohlcv["close"].iloc[-1])

        patterns = self._detect_patterns(ohlcv)
        swing_highs, swing_lows = self._find_swings(ohlcv)
        structure = self._analyze_structure(ohlcv, swing_highs, swing_lows)
        order_blocks = self._find_order_blocks(ohlcv)
        fvgs = self._find_fvgs(ohlcv)

        # Mark mitigated OBs and filled FVGs
        self._mark_mitigated(order_blocks, ohlcv)
        self._mark_filled_fvgs(fvgs, ohlcv)

        nearest_ob = self._nearest_active_ob(order_blocks, cp)
        nearest_fvg = self._nearest_active_fvg(fvgs, cp)

        signal = self._build_signal(
            patterns, structure, order_blocks, fvgs, cp
        )

        logger.info(
            "PriceAction %s | trend=%s | BOS=%.2f | patterns=%s | OBs=%d | FVGs=%d | signal=%s",
            symbol,
            structure.trend,
            structure.last_bos or 0,
            [p.name for p in patterns[-3:]],
            len([o for o in order_blocks if not o.mitigated]),
            len([f for f in fvgs if not f.filled]),
            signal.signal.name,
        )

        return PriceActionResult(
            patterns=patterns,
            structure=structure,
            order_blocks=order_blocks,
            fvgs=fvgs,
            signal=signal,
            nearest_ob=nearest_ob,
            nearest_fvg=nearest_fvg,
        )

    # ------------------------------------------------------------------
    # Candlestick pattern detection
    # ------------------------------------------------------------------

    def _detect_patterns(self, ohlcv: pd.DataFrame) -> List[CandlePattern]:
        patterns: List[CandlePattern] = []
        n = len(ohlcv)
        if n < 3:
            return patterns

        o = ohlcv["open"].values
        h = ohlcv["high"].values
        l = ohlcv["low"].values
        c = ohlcv["close"].values

        for i in range(2, n):
            body = abs(c[i] - o[i])
            full_range = h[i] - l[i]
            if full_range == 0:
                continue

            upper_wick = h[i] - max(o[i], c[i])
            lower_wick = min(o[i], c[i]) - l[i]
            body_pct = body / full_range
            is_bull = c[i] > o[i]

            # --- Doji ---
            if body_pct < 0.10 and full_range > 0:
                patterns.append(CandlePattern(
                    name="Doji", candle_index=i, direction="neutral",
                    strength=0.5,
                    description="Indecision — potential reversal or continuation pause"
                ))

            # --- Hammer (bullish) ---
            elif (lower_wick > body * 2 and upper_wick < body * 0.5
                  and is_bull and body_pct > 0.1):
                patterns.append(CandlePattern(
                    name="Hammer", candle_index=i, direction="bullish",
                    strength=0.75,
                    description="Bullish hammer — strong rejection of lows"
                ))

            # --- Shooting Star (bearish) ---
            elif (upper_wick > body * 2 and lower_wick < body * 0.5
                  and not is_bull and body_pct > 0.1):
                patterns.append(CandlePattern(
                    name="ShootingStar", candle_index=i, direction="bearish",
                    strength=0.75,
                    description="Shooting star — strong rejection of highs"
                ))

            # --- Pin Bar (both) ---
            elif lower_wick > full_range * 0.6 and body_pct < 0.25:
                patterns.append(CandlePattern(
                    name="BullishPinBar", candle_index=i, direction="bullish",
                    strength=0.8,
                    description="Bullish pin bar — heavy wick rejection at lows"
                ))
            elif upper_wick > full_range * 0.6 and body_pct < 0.25:
                patterns.append(CandlePattern(
                    name="BearishPinBar", candle_index=i, direction="bearish",
                    strength=0.8,
                    description="Bearish pin bar — heavy wick rejection at highs"
                ))

            # --- Bullish Engulfing ---
            if (i >= 1 and not is_bull is False):
                prev_bull = c[i - 1] > o[i - 1]
                if (is_bull and not prev_bull
                        and c[i] > o[i - 1] and o[i] < c[i - 1]):
                    patterns.append(CandlePattern(
                        name="BullishEngulfing", candle_index=i, direction="bullish",
                        strength=0.85,
                        description="Bullish engulfing — buyers overwhelmed sellers"
                    ))
                elif (not is_bull and prev_bull
                      and c[i] < o[i - 1] and o[i] > c[i - 1]):
                    patterns.append(CandlePattern(
                        name="BearishEngulfing", candle_index=i, direction="bearish",
                        strength=0.85,
                        description="Bearish engulfing — sellers overwhelmed buyers"
                    ))

            # --- Inside Bar ---
            if (h[i] < h[i - 1] and l[i] > l[i - 1]):
                patterns.append(CandlePattern(
                    name="InsideBar", candle_index=i, direction="neutral",
                    strength=0.6,
                    description="Inside bar — consolidation / coiled spring before breakout"
                ))

            # --- Marubozu (strong momentum candle) ---
            if body_pct > 0.85:
                direction = "bullish" if is_bull else "bearish"
                patterns.append(CandlePattern(
                    name=f"{'Bullish' if is_bull else 'Bearish'}Marubozu",
                    candle_index=i,
                    direction=direction,
                    strength=0.9,
                    description=f"{'Bullish' if is_bull else 'Bearish'} marubozu — strong {'buying' if is_bull else 'selling'} momentum, no hesitation"
                ))

        return patterns

    # ------------------------------------------------------------------
    # Swing point detection
    # ------------------------------------------------------------------

    def _find_swings(
        self, ohlcv: pd.DataFrame, order: int = 3
    ) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        highs = ohlcv["high"].values
        lows = ohlcv["low"].values
        swing_highs, swing_lows = [], []

        for i in range(order, len(ohlcv) - order):
            # Swing high: highest in surrounding bars
            if highs[i] == max(highs[i - order: i + order + 1]):
                ts = ohlcv.index[i] if hasattr(ohlcv.index[i], 'to_pydatetime') else None
                swing_highs.append(SwingPoint(price=float(highs[i]), index=i, kind="high", timestamp=ts))
            # Swing low: lowest in surrounding bars
            if lows[i] == min(lows[i - order: i + order + 1]):
                ts = ohlcv.index[i] if hasattr(ohlcv.index[i], 'to_pydatetime') else None
                swing_lows.append(SwingPoint(price=float(lows[i]), index=i, kind="low", timestamp=ts))

        return swing_highs, swing_lows

    # ------------------------------------------------------------------
    # Market Structure: HH/HL, LH/LL, BOS, CHoCH
    # ------------------------------------------------------------------

    def _analyze_structure(
        self,
        ohlcv: pd.DataFrame,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
    ) -> MarketStructure:
        structure = MarketStructure(trend="ranging", swing_highs=swing_highs, swing_lows=swing_lows)

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return structure

        # Use last 4 swings for structure
        recent_highs = sorted(swing_highs, key=lambda x: x.index)[-4:]
        recent_lows = sorted(swing_lows, key=lambda x: x.index)[-4:]

        # HH / HL detection
        hh = all(recent_highs[i].price > recent_highs[i - 1].price for i in range(1, len(recent_highs)))
        hl = all(recent_lows[i].price > recent_lows[i - 1].price for i in range(1, len(recent_lows)))
        lh = all(recent_highs[i].price < recent_highs[i - 1].price for i in range(1, len(recent_highs)))
        ll = all(recent_lows[i].price < recent_lows[i - 1].price for i in range(1, len(recent_lows)))

        structure.higher_highs = hh
        structure.higher_lows = hl
        structure.lower_highs = lh
        structure.lower_lows = ll

        if hh and hl:
            structure.trend = "bullish"
        elif lh and ll:
            structure.trend = "bearish"
        else:
            structure.trend = "ranging"

        # Break of Structure (BOS): price closes beyond last swing
        closes = ohlcv["close"].values
        last_high = recent_highs[-1].price if recent_highs else None
        last_low = recent_lows[-1].price if recent_lows else None
        last_close = float(closes[-1])

        if last_high and last_close > last_high:
            structure.last_bos = last_high
        elif last_low and last_close < last_low:
            structure.last_bos = last_low

        # Change of Character (CHoCH): in an uptrend, first LL; in downtrend, first HH
        if structure.trend == "bullish" and len(recent_lows) >= 2:
            if recent_lows[-1].price < recent_lows[-2].price:
                structure.last_choch = recent_lows[-1].price
        elif structure.trend == "bearish" and len(recent_highs) >= 2:
            if recent_highs[-1].price > recent_highs[-2].price:
                structure.last_choch = recent_highs[-1].price

        return structure

    # ------------------------------------------------------------------
    # Order Blocks
    # ------------------------------------------------------------------

    def _find_order_blocks(self, ohlcv: pd.DataFrame, lookback: int = 30) -> List[OrderBlock]:
        """
        An Order Block is the last opposite-coloured candle before a strong
        impulsive move. Bullish OB = last red candle before a strong up move.
        Bearish OB = last green candle before a strong down move.
        """
        obs: List[OrderBlock] = []
        o = ohlcv["open"].values
        h = ohlcv["high"].values
        l = ohlcv["low"].values
        c = ohlcv["close"].values
        n = len(ohlcv)
        start = max(0, n - lookback)

        for i in range(start + 2, n):
            move_size = abs(c[i] - o[i])
            avg_body = np.mean([abs(c[j] - o[j]) for j in range(max(0, i - 10), i)])
            if avg_body == 0:
                continue
            is_impulse = move_size > avg_body * 2.0

            if not is_impulse:
                continue

            is_bull_impulse = c[i] > o[i]
            # For bullish impulse: look for the last bearish candle before it
            for j in range(i - 1, max(start, i - 5), -1):
                if is_bull_impulse and c[j] < o[j]:  # bearish candle = demand OB
                    obs.append(OrderBlock(
                        price_high=float(o[j]),   # open of the bearish candle
                        price_low=float(c[j]),    # close of the bearish candle
                        direction="bullish",
                        index=j,
                        description=f"Demand OB @ {c[j]:.2f}–{o[j]:.2f} (before impulse up)"
                    ))
                    break
                elif not is_bull_impulse and c[j] > o[j]:  # bullish candle = supply OB
                    obs.append(OrderBlock(
                        price_high=float(c[j]),
                        price_low=float(o[j]),
                        direction="bearish",
                        index=j,
                        description=f"Supply OB @ {o[j]:.2f}–{c[j]:.2f} (before impulse down)"
                    ))
                    break

        # Deduplicate by proximity
        seen: List[float] = []
        deduped: List[OrderBlock] = []
        for ob in obs:
            if not any(abs(ob.midpoint - s) < ob.midpoint * 0.002 for s in seen):
                seen.append(ob.midpoint)
                deduped.append(ob)

        return deduped[-10:]  # keep last 10

    def _mark_mitigated(self, order_blocks: List[OrderBlock], ohlcv: pd.DataFrame) -> None:
        """Mark OBs that have been traded through (mitigated)."""
        lows = ohlcv["low"].values
        highs = ohlcv["high"].values
        for ob in order_blocks:
            for i in range(ob.index + 1, len(ohlcv)):
                if ob.direction == "bullish" and lows[i] < ob.price_low:
                    ob.mitigated = True
                    break
                elif ob.direction == "bearish" and highs[i] > ob.price_high:
                    ob.mitigated = True
                    break

    # ------------------------------------------------------------------
    # Fair Value Gaps
    # ------------------------------------------------------------------

    def _find_fvgs(self, ohlcv: pd.DataFrame, lookback: int = 30) -> List[FairValueGap]:
        """
        Bullish FVG: candle[i].low > candle[i-2].high  → gap between them
        Bearish FVG: candle[i].high < candle[i-2].low  → gap between them
        """
        fvgs: List[FairValueGap] = []
        h = ohlcv["high"].values
        l = ohlcv["low"].values
        n = len(ohlcv)
        start = max(2, n - lookback)

        for i in range(start, n):
            # Bullish FVG
            if l[i] > h[i - 2]:
                fvgs.append(FairValueGap(
                    price_high=float(l[i]),
                    price_low=float(h[i - 2]),
                    direction="bullish",
                    index=i,
                ))
            # Bearish FVG
            elif h[i] < l[i - 2]:
                fvgs.append(FairValueGap(
                    price_high=float(l[i - 2]),
                    price_low=float(h[i]),
                    direction="bearish",
                    index=i,
                ))

        return fvgs[-10:]

    def _mark_filled_fvgs(self, fvgs: List[FairValueGap], ohlcv: pd.DataFrame) -> None:
        lows = ohlcv["low"].values
        highs = ohlcv["high"].values
        for fvg in fvgs:
            for i in range(fvg.index + 1, len(ohlcv)):
                if fvg.direction == "bullish" and lows[i] <= fvg.price_low:
                    fvg.filled = True
                    break
                elif fvg.direction == "bearish" and highs[i] >= fvg.price_high:
                    fvg.filled = True
                    break

    # ------------------------------------------------------------------
    # Nearest active OB / FVG
    # ------------------------------------------------------------------

    def _nearest_active_ob(
        self, obs: List[OrderBlock], current_price: float
    ) -> Optional[OrderBlock]:
        active = [ob for ob in obs if not ob.mitigated]
        if not active:
            return None
        return min(active, key=lambda ob: abs(ob.midpoint - current_price))

    def _nearest_active_fvg(
        self, fvgs: List[FairValueGap], current_price: float
    ) -> Optional[FairValueGap]:
        active = [f for f in fvgs if not f.filled]
        if not active:
            return None
        return min(active, key=lambda f: abs(f.midpoint - current_price))

    # ------------------------------------------------------------------
    # Signal synthesis
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        patterns: List[CandlePattern],
        structure: MarketStructure,
        obs: List[OrderBlock],
        fvgs: List[FairValueGap],
        current_price: float,
    ) -> TechnicalSignal:
        score = 0.0
        reasons: List[str] = []

        # --- Market structure (most weight) ---
        if structure.trend == "bullish":
            score += 1.0
            hh_hl = "HH+HL" if structure.higher_highs and structure.higher_lows else "partial"
            reasons.append(f"Bullish structure ({hh_hl})")
        elif structure.trend == "bearish":
            score -= 1.0
            lh_ll = "LH+LL" if structure.lower_highs and structure.lower_lows else "partial"
            reasons.append(f"Bearish structure ({lh_ll})")
        else:
            reasons.append("Ranging/no clear structure")

        # --- BOS / CHoCH ---
        if structure.last_bos:
            if structure.last_bos < current_price:
                score += 0.5
                reasons.append(f"Bullish BOS above {structure.last_bos:.2f}")
            else:
                score -= 0.5
                reasons.append(f"Bearish BOS below {structure.last_bos:.2f}")

        if structure.last_choch:
            # CHoCH = warning sign of structure flip
            reasons.append(f"CHoCH at {structure.last_choch:.2f} — structure may be flipping")
            score *= 0.7  # reduce confidence

        # --- Recent candlestick patterns (last 3 candles) ---
        recent = [p for p in patterns if p.candle_index >= len(patterns) - 5]
        for pat in recent[-3:]:
            if pat.direction == "bullish":
                score += pat.strength * 0.4
                reasons.append(f"{pat.name} (bullish)")
            elif pat.direction == "bearish":
                score -= pat.strength * 0.4
                reasons.append(f"{pat.name} (bearish)")

        # --- Order block proximity ---
        active_obs = [ob for ob in obs if not ob.mitigated]
        zone_pct = 0.005
        for ob in active_obs:
            dist = abs(current_price - ob.midpoint) / current_price
            if dist < zone_pct:
                if ob.direction == "bullish":
                    score += 0.6
                    reasons.append(f"Price in demand OB ({ob.description})")
                else:
                    score -= 0.6
                    reasons.append(f"Price in supply OB ({ob.description})")

        # --- FVG proximity ---
        active_fvgs = [f for f in fvgs if not f.filled]
        for fvg in active_fvgs:
            dist = abs(current_price - fvg.midpoint) / current_price
            if dist < zone_pct:
                if fvg.direction == "bullish":
                    score += 0.4
                    reasons.append(f"Price entering bullish FVG ({fvg.price_low:.2f}–{fvg.price_high:.2f})")
                else:
                    score -= 0.4
                    reasons.append(f"Price in bearish FVG ({fvg.price_low:.2f}–{fvg.price_high:.2f})")

        signal = self._score_to_signal(score)
        return TechnicalSignal(
            indicator="PriceAction",
            value=score,
            signal=signal,
            description=" | ".join(reasons) if reasons else "No clear price action signal",
        )

    @staticmethod
    def _score_to_signal(score: float) -> SignalStrength:
        if score >= 1.5:
            return SignalStrength.STRONG_BUY
        elif score >= 0.5:
            return SignalStrength.BUY
        elif score <= -1.5:
            return SignalStrength.STRONG_SELL
        elif score <= -0.5:
            return SignalStrength.SELL
        return SignalStrength.NEUTRAL
