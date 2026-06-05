"""
CSV loader for OHLCV data exported from MetaTrader 5.

Accepts both the format written by ``examples/mt5_export_csv.py``
(``time,open,high,low,close,volume`` with ISO-8601 UTC timestamps) and the
tab-separated format MT5's manual "Export Bars" produces
(``<DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>``).

Returns a DataFrame with lowercase columns [open, high, low, close, volume]
and a UTC DatetimeIndex — exactly what every analyzer expects.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional

import pandas as pd

_OHLC = ("open", "high", "low", "close")
_VOL_CANDIDATES = ("volume", "tick_volume", "tickvol", "real_volume", "vol")


def load_ohlcv_csv(path: str) -> pd.DataFrame:
    """Load a single OHLCV CSV/TSV into the canonical analyzer format."""
    # sep=None + python engine auto-detects comma vs tab.
    raw = pd.read_csv(path, sep=None, engine="python")
    cols = {c.lower().strip().strip("<>"): c for c in raw.columns}

    # --- timestamp ---
    if "date" in cols and "time" in cols and "open" in cols:
        ts = raw[cols["date"]].astype(str).str.strip() + " " + \
             raw[cols["time"]].astype(str).str.strip()
    elif "time" in cols:
        ts = raw[cols["time"]]
    elif "datetime" in cols:
        ts = raw[cols["datetime"]]
    else:
        raise ValueError(f"{path}: no time/date column found (got {list(raw.columns)})")

    index = pd.to_datetime(ts, utc=True, errors="coerce")

    out = pd.DataFrame(index=index)
    # Assign positionally (.to_numpy()) — raw has a RangeIndex while out has a
    # DatetimeIndex; label-aligned assignment would produce all-NaN columns.
    for k in _OHLC:
        if k not in cols:
            raise ValueError(f"{path}: missing '{k}' column (got {list(raw.columns)})")
        out[k] = pd.to_numeric(raw[cols[k]], errors="coerce").to_numpy()

    vol_key = next((cols[k] for k in _VOL_CANDIDATES if k in cols), None)
    out["volume"] = pd.to_numeric(raw[vol_key], errors="coerce").to_numpy() if vol_key else 0.0

    out = out.dropna(subset=list(_OHLC)).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    if out.empty:
        raise ValueError(f"{path}: no valid rows after parsing")
    return out


def load_mtf_csvs(paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """Load a {timeframe: path} mapping into {timeframe: DataFrame}."""
    return {tf: load_ohlcv_csv(p) for tf, p in paths.items()}


def discover_mtf(directory: str, symbol: str,
                 timeframes: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """
    Find files named '<SYMBOL>_<TF>.csv' (the export script's convention) in a
    directory and load them. Case-insensitive on the timeframe suffix.
    """
    timeframes = timeframes or ["M15", "H1", "H4", "D1", "W1"]
    frames: Dict[str, pd.DataFrame] = {}
    for tf in timeframes:
        hits = glob.glob(os.path.join(directory, f"{symbol}_{tf}.csv")) or \
               glob.glob(os.path.join(directory, f"{symbol}_{tf.lower()}.csv"))
        if hits:
            frames[tf] = load_ohlcv_csv(hits[0])
    if not frames:
        raise FileNotFoundError(
            f"No '{symbol}_<TF>.csv' files found in {directory} for {timeframes}")
    return frames
