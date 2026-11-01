"""Daily-bar backtester.

Execution model (matches how the live bot runs, a few minutes before the
close): the signal is computed from bar *t*'s close and the trade happens
at that close, so the new position earns bar *t+1*'s return. Each change
in position pays ``cost_bps`` of the traded value — Robinhood charges no
commission, but you still pay the bid/ask spread and some slippage.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .metrics import Metrics, compute_metrics
from .strategies import BuyAndHold, Strategy


@dataclass(frozen=True)
class BacktestResult:
    strategy: str
    equity: pd.Series
    held: pd.Series  # weight held during each bar
    metrics: Metrics


def _clean(close: pd.Series) -> pd.Series:
    close = close.dropna().sort_index()
    if (close <= 0).any():
        raise ValueError("prices must be positive")
    if len(close) < 2:
        raise ValueError("need at least 2 bars to backtest")
    return close.astype(float)


def run_backtest(
    close: pd.Series,
    strategy: Strategy,
    starting_cash: float = 200.0,
    cost_bps: float = 5.0,
) -> BacktestResult:
    if starting_cash <= 0:
        raise ValueError("starting_cash must be positive")
    if cost_bps < 0:
        raise ValueError("cost_bps cannot be negative")
    close = _clean(close)

    target = strategy.target_weights(close).reindex(close.index).fillna(0.0).clip(0.0, 1.0)
    held = target.shift(1).fillna(0.0)
    turnover = target.diff().abs()
    turnover.iloc[0] = target.iloc[0]  # the very first entry is a trade too

    returns = close.pct_change().fillna(0.0)
    strat_returns = held * returns - turnover * (cost_bps / 10_000)
    equity = starting_cash * (1.0 + strat_returns).cumprod()
    trades = int((turnover > 1e-12).sum())
    return BacktestResult(strategy.name, equity, held, compute_metrics(equity, held, trades))


def compare(
    close: pd.Series,
    strategies: list[Strategy],
    starting_cash: float = 200.0,
    cost_bps: float = 5.0,
) -> list[BacktestResult]:
    """Backtest several strategies on the same data, always including buy-and-hold."""
    if not any(isinstance(s, BuyAndHold) for s in strategies):
        strategies = [BuyAndHold(), *strategies]
    return [run_backtest(close, s, starting_cash, cost_bps) for s in strategies]


def format_results(results: list[BacktestResult]) -> str:
    header = (
        f"{'strategy':<22}{'final $':>10}{'return':>9}{'CAGR':>8}"
        f"{'max DD':>9}{'sharpe':>8}{'in mkt':>8}{'trades':>8}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        m = r.metrics
        lines.append(
            f"{r.strategy:<22}{m.end_value:>10.2f}{m.total_return:>9.1%}{m.cagr:>8.1%}"
            f"{m.max_drawdown:>9.1%}{m.sharpe:>8.2f}{m.exposure:>8.0%}{m.trades:>8d}"
        )
    return "\n".join(lines)
