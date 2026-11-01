"""Trading strategies.

A strategy is a pure function of price history: given closing prices it
returns a *target weight* for every bar — 1.0 means "fully invested in
this symbol", 0.0 means "in cash". The backtester and the live engine
call exactly the same method, so what you test is what you trade.

To write your own strategy, implement ``name``, ``warmup`` and
``target_weights`` and register it in ``STRATEGIES``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .indicators import rsi, sma


class Strategy(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def warmup(self) -> int:
        """Minimum number of bars needed before the signal is meaningful."""
        ...

    def target_weights(self, close: pd.Series) -> pd.Series:
        """Target portfolio weight in [0, 1] for each bar of ``close``."""
        ...


@dataclass(frozen=True)
class BuyAndHold:
    """Always fully invested. The benchmark every strategy must beat."""

    @property
    def name(self) -> str:
        return "buy_and_hold"

    @property
    def warmup(self) -> int:
        return 1

    def target_weights(self, close: pd.Series) -> pd.Series:
        return pd.Series(1.0, index=close.index)


@dataclass(frozen=True)
class SmaCrossover:
    """Trend following: invested while the fast average is above the slow one."""

    fast: int = 20
    slow: int = 50

    def __post_init__(self) -> None:
        if not 1 <= self.fast < self.slow:
            raise ValueError(f"need 1 <= fast < slow, got fast={self.fast} slow={self.slow}")

    @property
    def name(self) -> str:
        return f"sma_{self.fast}_{self.slow}"

    @property
    def warmup(self) -> int:
        return self.slow

    def target_weights(self, close: pd.Series) -> pd.Series:
        fast, slow = sma(close, self.fast), sma(close, self.slow)
        return (fast > slow).astype(float)  # NaN comparisons are False -> 0.0


@dataclass(frozen=True)
class RsiMeanReversion:
    """Mean reversion: buy when RSI is oversold, sell when it is overbought."""

    period: int = 14
    buy_below: float = 30.0
    sell_above: float = 70.0

    def __post_init__(self) -> None:
        if self.period < 1:
            raise ValueError(f"period must be >= 1, got {self.period}")
        if not 0 < self.buy_below < self.sell_above < 100:
            raise ValueError(
                f"need 0 < buy_below < sell_above < 100, got {self.buy_below}/{self.sell_above}"
            )

    @property
    def name(self) -> str:
        return f"rsi_{self.period}_{self.buy_below:g}_{self.sell_above:g}"

    @property
    def warmup(self) -> int:
        return self.period + 1

    def target_weights(self, close: pd.Series) -> pd.Series:
        r = rsi(close, self.period)
        signal = pd.Series(np.nan, index=close.index)
        signal[r < self.buy_below] = 1.0
        signal[r > self.sell_above] = 0.0
        return signal.ffill().fillna(0.0)  # hold the last decision between signals


STRATEGIES: dict[str, type] = {
    "buy_and_hold": BuyAndHold,
    "sma": SmaCrossover,
    "rsi": RsiMeanReversion,
}


def build_strategy(name: str, params: dict[str, Any] | None = None) -> Strategy:
    """Create a strategy by registry name, e.g. ``build_strategy("sma", {"fast": 10})``."""
    try:
        cls = STRATEGIES[name]
    except KeyError:
        raise ValueError(f"unknown strategy {name!r}; choose from {sorted(STRATEGIES)}") from None
    try:
        return cls(**(params or {}))
    except TypeError as exc:
        raise ValueError(f"bad parameters for strategy {name!r}: {exc}") from None
