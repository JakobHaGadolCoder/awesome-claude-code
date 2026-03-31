"""
Support & Resistance Analyzer
Detects key price levels using:
- Swing high/low pivot points
- Volume profile (Point of Control / Value Area)
- Round number / psychological levels
- Previous day/week/month high-low
- Options max-pain and gamma exposure (GEX) levels
- Fibonacci retracement levels of the most recent significant impulse
- Broken-level role-reversal detection (old resistance = new support)

Post-mortem amendment (31 Mar):
  The 50% Fibonacci retracement of the 4482→4618 impulse landed at 4550.
  Price reversed from exactly 4549. The S/R module had no Fibonacci logic,
  so this critical support was invisible to the system. Fixed by adding
  Fibonacci retracement detection on the most recent impulse move.
  Also added role-reversal check: a broken resistance level flips to support
  and should SUPPRESS bearish signals when price pulls back to test it.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from options_trader.core.models import (
    OptionContract,
    OptionType,
    SignalStrength,
    SupportResistanceLevel,
    TechnicalSignal,
)
from options_trader.core.config import TradingConfig

logger = logging.getLogger(__name__)


class SupportResistanceAnalyzer:
    """
    Identifies key support and resistance levels across multiple methods
    and assesses how close the current price is to each level.
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
        current_price: float,
        options_chain: Optional[List[OptionContract]] = None,
    ) -> Tuple[List[SupportResistanceLevel], TechnicalSignal]:
        """
        Returns all detected S/R levels and a signal based on where
        current price sits relative to them.
        """
        levels: List[SupportResistanceLevel] = []

        levels.extend(self._pivot_levels(ohlcv))
        levels.extend(self._volume_profile_levels(ohlcv))
        levels.extend(self._psychological_levels(current_price))
        levels.extend(self._periodic_highs_lows(ohlcv))
        if options_chain:
            levels.extend(self._max_pain_and_gex(options_chain, current_price))

        levels.extend(self._fibonacci_levels(ohlcv, current_price))
        levels = self._merge_nearby_levels(levels, current_price)
        signal = self._build_signal(levels, current_price, ohlcv)

        logger.info(
            "S/R %s | current=%.2f | levels=%d | nearest=%s",
            symbol,
            current_price,
            len(levels),
            self._nearest_level(levels, current_price),
        )
        return levels, signal

    # ------------------------------------------------------------------
    # Pivot points (swing highs/lows)
    # ------------------------------------------------------------------

    def _pivot_levels(self, ohlcv: pd.DataFrame) -> List[SupportResistanceLevel]:
        levels = []
        closes = ohlcv["close"].values
        highs = ohlcv["high"].values
        lows = ohlcv["low"].values
        order = 5  # look 5 bars either side

        high_indices = argrelextrema(highs, np.greater_equal, order=order)[0]
        low_indices = argrelextrema(lows, np.less_equal, order=order)[0]

        for idx in high_indices[-20:]:  # Keep last 20 pivots
            price = float(highs[idx])
            touches = self._count_touches(ohlcv, price)
            strength = min(1.0, touches / 5.0)
            levels.append(
                SupportResistanceLevel(
                    price=price,
                    level_type="resistance",
                    strength=strength,
                    timeframe="daily",
                    description=f"Swing high pivot (touched {touches}x)",
                )
            )

        for idx in low_indices[-20:]:
            price = float(lows[idx])
            touches = self._count_touches(ohlcv, price)
            strength = min(1.0, touches / 5.0)
            levels.append(
                SupportResistanceLevel(
                    price=price,
                    level_type="support",
                    strength=strength,
                    timeframe="daily",
                    description=f"Swing low pivot (touched {touches}x)",
                )
            )

        return levels

    # ------------------------------------------------------------------
    # Volume Profile (POC + Value Area)
    # ------------------------------------------------------------------

    def _volume_profile_levels(
        self, ohlcv: pd.DataFrame, bins: int = 50
    ) -> List[SupportResistanceLevel]:
        prices = ohlcv["close"].values
        volumes = ohlcv["volume"].values

        if len(prices) < 2:
            return []

        price_min, price_max = prices.min(), prices.max()
        price_range = price_max - price_min
        if price_range == 0:
            return []

        bin_edges = np.linspace(price_min, price_max, bins + 1)
        bin_indices = np.digitize(prices, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, bins - 1)

        vol_by_bin = np.zeros(bins)
        for i, vol in zip(bin_indices, volumes):
            vol_by_bin[i] += vol

        # Point of Control = highest volume bin
        poc_bin = int(np.argmax(vol_by_bin))
        poc_price = float((bin_edges[poc_bin] + bin_edges[poc_bin + 1]) / 2)

        # Value Area: bins covering 70% of volume
        total_vol = vol_by_bin.sum()
        target = total_vol * 0.70
        sorted_bins = np.argsort(vol_by_bin)[::-1]
        cumvol = 0.0
        va_bins = []
        for b in sorted_bins:
            cumvol += vol_by_bin[b]
            va_bins.append(b)
            if cumvol >= target:
                break

        va_high_bin = max(va_bins)
        va_low_bin = min(va_bins)
        va_high = float((bin_edges[va_high_bin] + bin_edges[va_high_bin + 1]) / 2)
        va_low = float((bin_edges[va_low_bin] + bin_edges[va_low_bin + 1]) / 2)

        return [
            SupportResistanceLevel(
                price=poc_price,
                level_type="support" if poc_price > va_low else "resistance",
                strength=0.9,
                timeframe="daily",
                description=f"Volume Profile POC (highest volume node)",
            ),
            SupportResistanceLevel(
                price=va_high,
                level_type="resistance",
                strength=0.7,
                timeframe="daily",
                description="Volume Profile Value Area High (VAH)",
            ),
            SupportResistanceLevel(
                price=va_low,
                level_type="support",
                strength=0.7,
                timeframe="daily",
                description="Volume Profile Value Area Low (VAL)",
            ),
        ]

    # ------------------------------------------------------------------
    # Psychological / round number levels
    # ------------------------------------------------------------------

    def _psychological_levels(
        self, current_price: float, range_pct: float = 0.05
    ) -> List[SupportResistanceLevel]:
        levels = []
        price_min = current_price * (1 - range_pct)
        price_max = current_price * (1 + range_pct)

        # Determine rounding increment based on price magnitude
        if current_price >= 1000:
            increment = 50.0
        elif current_price >= 500:
            increment = 25.0
        elif current_price >= 100:
            increment = 10.0
        elif current_price >= 50:
            increment = 5.0
        else:
            increment = 1.0

        start = (price_min // increment) * increment
        price = start
        while price <= price_max:
            if price >= price_min:
                levels.append(
                    SupportResistanceLevel(
                        price=round(price, 2),
                        level_type="support" if price < current_price else "resistance",
                        strength=0.5,
                        timeframe="daily",
                        description=f"Psychological round number level",
                    )
                )
            price += increment

        return levels

    # ------------------------------------------------------------------
    # Prior period highs and lows
    # ------------------------------------------------------------------

    def _periodic_highs_lows(
        self, ohlcv: pd.DataFrame
    ) -> List[SupportResistanceLevel]:
        levels = []
        if len(ohlcv) < 2:
            return levels

        # Previous day
        pdh = float(ohlcv["high"].iloc[-2])
        pdl = float(ohlcv["low"].iloc[-2])
        levels.append(
            SupportResistanceLevel(
                price=pdh,
                level_type="resistance",
                strength=0.75,
                timeframe="daily",
                description="Previous Day High (PDH)",
            )
        )
        levels.append(
            SupportResistanceLevel(
                price=pdl,
                level_type="support",
                strength=0.75,
                timeframe="daily",
                description="Previous Day Low (PDL)",
            )
        )

        # Weekly high/low (last 5 bars)
        if len(ohlcv) >= 5:
            wh = float(ohlcv["high"].iloc[-5:].max())
            wl = float(ohlcv["low"].iloc[-5:].min())
            levels.append(
                SupportResistanceLevel(
                    price=wh,
                    level_type="resistance",
                    strength=0.8,
                    timeframe="weekly",
                    description="Weekly High",
                )
            )
            levels.append(
                SupportResistanceLevel(
                    price=wl,
                    level_type="support",
                    strength=0.8,
                    timeframe="weekly",
                    description="Weekly Low",
                )
            )

        # Monthly high/low (last 22 bars)
        if len(ohlcv) >= 22:
            mh = float(ohlcv["high"].iloc[-22:].max())
            ml = float(ohlcv["low"].iloc[-22:].min())
            levels.append(
                SupportResistanceLevel(
                    price=mh,
                    level_type="resistance",
                    strength=0.85,
                    timeframe="monthly",
                    description="Monthly High",
                )
            )
            levels.append(
                SupportResistanceLevel(
                    price=ml,
                    level_type="support",
                    strength=0.85,
                    timeframe="monthly",
                    description="Monthly Low",
                )
            )

        return levels

    # ------------------------------------------------------------------
    # Fibonacci Retracement Levels  (KEY FIX — 31 Mar post-mortem)
    # ------------------------------------------------------------------

    def _fibonacci_levels(
        self, ohlcv: pd.DataFrame, current_price: float, lookback: int = 50
    ) -> List[SupportResistanceLevel]:
        """
        Detect the most recent significant impulse move (swing low → swing high
        or swing high → swing low) within the last `lookback` bars and place
        Fibonacci retracement levels at 23.6%, 38.2%, 50%, 61.8%, 78.6%.

        These are the levels where corrective pullbacks most frequently reverse.
        The 50% level in particular has the highest empirical hit rate.

        Critical insight: if a large impulsive candle (>3× ATR) has recently
        occurred, the 50% / 61.8% retracement is a HIGH-PROBABILITY bounce zone,
        NOT a continuation zone. These levels must be flagged as strong support
        (for bull impulse) or strong resistance (for bear impulse).
        """
        levels: List[SupportResistanceLevel] = []
        df = ohlcv.tail(lookback).copy()
        df.columns = [c.lower() for c in df.columns]

        highs = df["high"].values
        lows  = df["low"].values

        if len(highs) < 10:
            return levels

        # Find swing high and swing low within lookback
        swing_high_idx = int(np.argmax(highs))
        swing_low_idx  = int(np.argmin(lows))
        swing_high = float(highs[swing_high_idx])
        swing_low  = float(lows[swing_low_idx])

        impulse_size = swing_high - swing_low
        if impulse_size <= 0:
            return levels

        # Determine direction of the most recent impulse
        # (whichever pivot came last determines current retracement direction)
        if swing_high_idx > swing_low_idx:
            # Bull impulse: low came first, then high — retracement is DOWN from high
            origin, terminus = swing_low, swing_high
            retrace_direction = "bullish_impulse"
        else:
            # Bear impulse: high came first, then low — retracement is UP from low
            origin, terminus = swing_high, swing_low
            retrace_direction = "bearish_impulse"

        fib_ratios = {
            "23.6%": 0.236,
            "38.2%": 0.382,
            "50.0%": 0.500,   # Highest hit-rate — the level that stopped us out
            "61.8%": 0.618,   # Golden ratio — second highest hit-rate
            "78.6%": 0.786,
        }

        for label, ratio in fib_ratios.items():
            if retrace_direction == "bullish_impulse":
                # Retracement levels below the swing high
                fib_price = terminus - ratio * impulse_size
                level_type = "support"
                # 50% and 61.8% are highest-confidence bounce zones
                strength = 0.90 if ratio in (0.500, 0.618) else 0.75
            else:
                # Retracement levels above the swing low
                fib_price = terminus + ratio * impulse_size
                level_type = "resistance"
                strength = 0.90 if ratio in (0.500, 0.618) else 0.75

            # Only include levels near current price (within 5%)
            if abs(fib_price - current_price) / current_price <= 0.05:
                levels.append(
                    SupportResistanceLevel(
                        price=round(fib_price, 2),
                        level_type=level_type,
                        strength=strength,
                        timeframe="intraday",
                        description=(
                            f"Fib {label} retrace of {retrace_direction.replace('_', ' ')} "
                            f"({swing_low:.2f}→{swing_high:.2f}, range={impulse_size:.2f})"
                        ),
                    )
                )

        return levels

    # ------------------------------------------------------------------
    # Max Pain & Gamma Exposure (GEX)
    # ------------------------------------------------------------------

    def _max_pain_and_gex(
        self,
        chain: List[OptionContract],
        current_price: float,
    ) -> List[SupportResistanceLevel]:
        levels = []
        if not chain:
            return levels

        # --- Max Pain ---
        strikes = sorted(set(c.strike for c in chain))
        min_pain_strike = None
        min_pain_value = float("inf")

        for test_price in strikes:
            pain = 0.0
            for c in chain:
                if c.option_type == OptionType.CALL:
                    pain += max(0, test_price - c.strike) * c.open_interest
                else:
                    pain += max(0, c.strike - test_price) * c.open_interest
            if pain < min_pain_value:
                min_pain_value = pain
                min_pain_strike = test_price

        if min_pain_strike is not None:
            levels.append(
                SupportResistanceLevel(
                    price=min_pain_strike,
                    level_type="support" if min_pain_strike < current_price else "resistance",
                    strength=0.8,
                    timeframe="weekly",
                    description=f"Options Max Pain (${min_pain_strike:.0f})",
                )
            )

        # --- Gamma Exposure (GEX) wall ---
        gex_by_strike: dict[float, float] = {}
        for c in chain:
            if c.gamma > 0 and c.underlying_price > 0:
                # GEX = gamma * open_interest * 100 * underlying_price
                gex = c.gamma * c.open_interest * 100 * c.underlying_price
                if c.option_type == OptionType.PUT:
                    gex = -gex
                gex_by_strike[c.strike] = gex_by_strike.get(c.strike, 0) + gex

        if gex_by_strike:
            max_gex_strike = max(gex_by_strike, key=lambda k: abs(gex_by_strike[k]))
            gex_val = gex_by_strike[max_gex_strike]
            levels.append(
                SupportResistanceLevel(
                    price=max_gex_strike,
                    level_type="resistance" if gex_val > 0 else "support",
                    strength=0.85,
                    timeframe="weekly",
                    description=f"GEX Wall at ${max_gex_strike:.0f} (GEX={gex_val/1e6:.1f}M)",
                )
            )

        return levels

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _count_touches(self, ohlcv: pd.DataFrame, price: float) -> int:
        zone = price * self.config.sr_zone_pct
        count = 0
        for _, row in ohlcv.iterrows():
            if row["low"] - zone <= price <= row["high"] + zone:
                count += 1
        return count

    def _merge_nearby_levels(
        self, levels: List[SupportResistanceLevel], current_price: float
    ) -> List[SupportResistanceLevel]:
        """Merge levels that are within sr_zone_pct of each other."""
        if not levels:
            return levels
        zone = current_price * self.config.sr_zone_pct
        levels_sorted = sorted(levels, key=lambda x: x.price)
        merged: List[SupportResistanceLevel] = []

        for lv in levels_sorted:
            if merged and abs(lv.price - merged[-1].price) <= zone:
                # Merge: keep stronger level
                if lv.strength > merged[-1].strength:
                    merged[-1] = lv
            else:
                merged.append(lv)
        return merged

    def _build_signal(
        self,
        levels: List[SupportResistanceLevel],
        current_price: float,
        ohlcv: Optional[pd.DataFrame] = None,
    ) -> TechnicalSignal:
        nearest_support = None
        nearest_resistance = None
        min_support_dist = float("inf")
        min_resistance_dist = float("inf")

        for lv in levels:
            dist = abs(current_price - lv.price)
            if lv.level_type == "support" and lv.price <= current_price:
                if dist < min_support_dist:
                    min_support_dist = dist
                    nearest_support = lv
            elif lv.level_type == "resistance" and lv.price >= current_price:
                if dist < min_resistance_dist:
                    min_resistance_dist = dist
                    nearest_resistance = lv

        if nearest_support is None and nearest_resistance is None:
            return TechnicalSignal(
                indicator="SupportResistance",
                value=0.0,
                signal=SignalStrength.NEUTRAL,
                description="No nearby S/R levels found",
            )

        sup_dist_pct = min_support_dist / current_price if current_price else 1.0
        res_dist_pct = min_resistance_dist / current_price if current_price else 1.0

        # ------------------------------------------------------------------
        # Role-reversal check (KEY FIX — 31 Mar post-mortem)
        # If price recently broke ABOVE a prior resistance level (bullish
        # breakout), that level flips to support on the first pullback.
        # Shorting into a role-reversed support is a low-probability trade.
        # ------------------------------------------------------------------
        role_reversal_support = None
        if ohlcv is not None and len(ohlcv) >= 10:
            role_reversal_support = self._detect_role_reversal_support(
                levels, current_price, ohlcv
            )

        if role_reversal_support is not None:
            # Price is pulling back to a former resistance now acting as support
            sig = SignalStrength.STRONG_BUY
            desc = (
                f"ROLE-REVERSAL SUPPORT at {role_reversal_support.price:.2f} — "
                f"prior resistance broken → now support. "
                f"Shorting here is CONTRA-TREND. HIGH-PROB BOUNCE ZONE."
            )
            return TechnicalSignal(
                indicator="SupportResistance",
                value=2.0,
                signal=sig,
                description=desc,
            )

        # Fibonacci 50%/61.8% near current price = strong support/resistance
        fib_near = [
            lv for lv in levels
            if "Fib 50" in lv.description or "Fib 61" in lv.description
            if abs(lv.price - current_price) / current_price < 0.005
        ]
        if fib_near:
            fib_lv = fib_near[0]
            if fib_lv.level_type == "support":
                sig = SignalStrength.STRONG_BUY
                desc = f"Price at {fib_lv.description} — HIGH-PROBABILITY BOUNCE ZONE"
            else:
                sig = SignalStrength.STRONG_SELL
                desc = f"Price at {fib_lv.description} — HIGH-PROBABILITY REJECTION ZONE"
            return TechnicalSignal(
                indicator="SupportResistance",
                value=sig.value * 1.5,
                signal=sig,
                description=desc,
            )

        # Standard S/R signal
        if nearest_support and sup_dist_pct < 0.005 and nearest_support.strength > 0.7:
            sig = SignalStrength.STRONG_BUY
            desc = f"Price at strong support {nearest_support.price:.2f} ({nearest_support.description})"
        elif nearest_support and sup_dist_pct < 0.01:
            sig = SignalStrength.BUY
            desc = f"Price near support {nearest_support.price:.2f}"
        elif nearest_resistance and res_dist_pct < 0.005 and nearest_resistance.strength > 0.7:
            sig = SignalStrength.STRONG_SELL
            desc = f"Price at strong resistance {nearest_resistance.price:.2f} ({nearest_resistance.description})"
        elif nearest_resistance and res_dist_pct < 0.01:
            sig = SignalStrength.SELL
            desc = f"Price near resistance {nearest_resistance.price:.2f}"
        elif nearest_support and sup_dist_pct < res_dist_pct:
            sig = SignalStrength.BUY
            desc = f"Closer to support ({nearest_support.price:.2f}) than resistance"
        else:
            sig = SignalStrength.SELL
            desc = f"Closer to resistance ({nearest_resistance.price:.2f}) than support"

        score = sig.value
        return TechnicalSignal(
            indicator="SupportResistance",
            value=score,
            signal=sig,
            description=desc,
        )

    def _detect_role_reversal_support(
        self,
        levels: List[SupportResistanceLevel],
        current_price: float,
        ohlcv: pd.DataFrame,
        lookback: int = 20,
        proximity_pct: float = 0.008,
    ) -> Optional[SupportResistanceLevel]:
        """
        Returns a level if:
        1. There is a prior resistance level within proximity_pct of current price
        2. Price has recently broken and CLOSED above that level (confirming breakout)
        3. Price is now pulling back to retest it from above

        This is the core "role-reversal" pattern — broken resistance becomes support.
        A pullback to this zone is a buy opportunity, NOT a short.
        """
        df = ohlcv.tail(lookback).copy()
        df.columns = [c.lower() for c in df.columns]
        closes = df["close"].values

        for lv in levels:
            if lv.level_type != "resistance":
                continue
            # Level must be near current price (we're testing it)
            dist_pct = abs(lv.price - current_price) / current_price
            if dist_pct > proximity_pct:
                continue
            # Current price must be at or slightly above the level (pullback retest)
            if current_price < lv.price * 0.995:
                continue
            # There must have been a confirmed close above this level recently
            closes_above = sum(1 for c in closes if c > lv.price * 1.002)
            if closes_above >= 2:
                return lv

        return None

    def _nearest_level(
        self, levels: List[SupportResistanceLevel], current_price: float
    ) -> Optional[str]:
        if not levels:
            return None
        nearest = min(levels, key=lambda lv: abs(lv.price - current_price))
        return f"{nearest.level_type} @ {nearest.price:.2f} ({nearest.description})"
