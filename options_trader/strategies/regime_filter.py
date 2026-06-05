"""
Regime stand-aside filter (prototype).

Shared by the live loop and the backtester so both gate entries identically.
The backtest showed the system bleeds in ranging regimes by chasing mid-range
breakouts that fail. This filter blocks trend-continuation entries while the
regime is RANGING, allowing only mean-reversion fades at an edge.
"""
from __future__ import annotations

from options_trader.core.models import MarketRegime, SignalStrength

_BUY = (SignalStrength.BUY, SignalStrength.STRONG_BUY)
_SELL = (SignalStrength.SELL, SignalStrength.STRONG_SELL)


def range_stand_aside(
    regime: MarketRegime,
    mode: str,
    is_buy: bool,
    band_position: str,
    sr_signal: SignalStrength,
) -> bool:
    """
    Return True if a NEW entry should be SUPPRESSED by the regime filter.

    Only acts while ``regime`` is RANGING. ``mode``:
      * "off"            never suppresses
      * "block_all"      suppress every entry in a range
      * "reversal_only"  allow only mean-reversion fades at an edge
                         (BUY at the lower VWAP band / support, SELL at the
                         upper band / resistance); block everything else
    Non-RANGING regimes are never filtered (return False).
    """
    if mode == "off" or regime != MarketRegime.RANGING:
        return False
    if mode == "block_all":
        return True

    # reversal_only
    at_lower = band_position in ("below_1s", "below_2s", "below_3s")
    at_upper = band_position in ("above_1s", "above_2s", "above_3s")
    sr_buy = sr_signal in _BUY
    sr_sell = sr_signal in _SELL
    if is_buy and (at_lower or sr_buy):
        return False        # valid mean-reversion long at the lower edge
    if (not is_buy) and (at_upper or sr_sell):
        return False        # valid mean-reversion short at the upper edge
    return True             # ranging trend-chase -> stand aside
