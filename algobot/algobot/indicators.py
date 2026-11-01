"""Technical indicators used by the strategies.

All functions take a pandas Series of closing prices and return a Series
aligned to the same index. Warm-up periods are NaN, never zero, so a
strategy can tell "not enough data yet" apart from a real reading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    """Simple moving average over ``window`` bars."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    return close.rolling(window=window, min_periods=window).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index (0-100).

    The first ``period`` values are NaN (RSI needs ``period`` price
    changes). A run with no losses reads 100, no gains reads 0, and a
    perfectly flat run reads 50.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        out = 100 - 100 / (1 + rs)
    out = out.mask(avg_loss == 0, 100.0)
    out = out.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    return out.mask(avg_gain.isna())
