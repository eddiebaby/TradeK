"""Performance statistics for an equity curve."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class Metrics:
    start_value: float
    end_value: float
    total_return: float  # 0.25 == +25%
    cagr: float  # compound annual growth rate
    volatility: float  # annualized std-dev of daily returns
    sharpe: float  # annualized, risk-free rate assumed 0
    max_drawdown: float  # worst peak-to-trough loss, e.g. -0.30
    exposure: float  # fraction of bars with money in the market
    trades: int  # number of position changes


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline as a negative fraction (0.0 if none)."""
    if equity.empty:
        return 0.0
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def compute_metrics(equity: pd.Series, held: pd.Series, trades: int) -> Metrics:
    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    total = end / start - 1.0
    years = (len(equity) - 1) / TRADING_DAYS
    cagr = (end / start) ** (1 / years) - 1.0 if years > 0 and end > 0 else 0.0

    daily = equity.pct_change().dropna()
    std = float(daily.std(ddof=0)) if len(daily) else 0.0
    vol = std * math.sqrt(TRADING_DAYS)
    sharpe = float(daily.mean()) / std * math.sqrt(TRADING_DAYS) if std > 1e-12 else 0.0

    return Metrics(
        start_value=start,
        end_value=end,
        total_return=total,
        cagr=cagr,
        volatility=vol,
        sharpe=sharpe,
        max_drawdown=max_drawdown(equity),
        exposure=float((held > 0).mean()) if len(held) else 0.0,
        trades=trades,
    )
