from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from algobot.backtest import compare, format_results, run_backtest
from algobot.metrics import compute_metrics, max_drawdown
from algobot.strategies import BuyAndHold, SmaCrossover

from .conftest import series, trend


@dataclass(frozen=True)
class Fixed:
    """Test strategy that returns pre-baked weights."""

    weights: tuple

    name = "fixed"
    warmup = 1

    def target_weights(self, close):
        return pd.Series(self.weights, index=close.index, dtype=float)


def test_buy_and_hold_tracks_price_minus_one_entry_cost():
    close = series([100, 110, 121])
    r = run_backtest(close, BuyAndHold(), starting_cash=100, cost_bps=10)
    assert r.equity.iloc[-1] == pytest.approx(100 * (1 - 0.001) * 1.21)
    assert r.metrics.trades == 1


def test_cash_strategy_never_changes_equity():
    r = run_backtest(trend(50), Fixed((0.0,) * 50), starting_cash=200)
    assert (r.equity == 200).all()
    assert r.metrics.trades == 0 and r.metrics.exposure == 0.0


def test_no_lookahead_signal_on_day_k_earns_day_k_plus_1():
    close = series([100, 100, 100, 150, 150])
    # Decide to buy at the close of bar 2 -> should earn the +50% move into bar 3.
    r = run_backtest(close, Fixed((0, 0, 1, 1, 1)), starting_cash=100, cost_bps=0)
    assert r.equity.iloc[-1] == pytest.approx(150)
    # Deciding at bar 3 (after the jump) must NOT capture it.
    late = run_backtest(close, Fixed((0, 0, 0, 1, 1)), starting_cash=100, cost_bps=0)
    assert late.equity.iloc[-1] == pytest.approx(100)


def test_costs_are_charged_on_every_position_change():
    close = series([100] * 6)
    r = run_backtest(close, Fixed((1, 0, 1, 0, 1, 0)), starting_cash=100, cost_bps=100)
    assert r.metrics.trades == 6
    assert r.equity.iloc[-1] == pytest.approx(100 * 0.99**6)


def test_weights_are_clipped_and_nans_treated_as_cash():
    close = series([100, 200, 400])
    r = run_backtest(close, Fixed((5.0, np.nan, -1)), starting_cash=100, cost_bps=0)
    assert r.equity.iloc[1] == pytest.approx(200)  # clipped to 1.0, not 5x leverage
    assert r.equity.iloc[2] == pytest.approx(200)  # NaN -> 0 -> flat


@pytest.mark.parametrize(
    "close,kwargs",
    [
        (series([100]), {}),
        (series([100, -1]), {}),
        (series([100, 101]), {"starting_cash": 0}),
        (series([100, 101]), {"cost_bps": -1}),
    ],
)
def test_backtest_input_validation(close, kwargs):
    with pytest.raises(ValueError):
        run_backtest(close, BuyAndHold(), **kwargs)


def test_compare_always_includes_benchmark_once():
    results = compare(trend(80), [SmaCrossover(5, 20)])
    assert [r.strategy for r in results] == ["buy_and_hold", "sma_5_20"]
    assert len(compare(trend(80), [BuyAndHold()])) == 1


def test_format_results_has_a_row_per_strategy():
    text = format_results(compare(trend(80), [SmaCrossover(5, 20)]))
    assert "buy_and_hold" in text and "sma_5_20" in text and "max DD" in text


def test_max_drawdown():
    assert max_drawdown(pd.Series([100, 120, 60, 130])) == pytest.approx(-0.5)
    assert max_drawdown(pd.Series([1, 2, 3])) == 0.0
    assert max_drawdown(pd.Series([], dtype=float)) == 0.0


def test_metrics_cagr_over_one_year():
    equity = pd.Series(np.linspace(100, 110, 253))
    m = compute_metrics(equity, pd.Series(1.0, index=equity.index), trades=1)
    assert m.cagr == pytest.approx(0.10)
    assert m.total_return == pytest.approx(0.10)
    assert m.sharpe > 0 and m.exposure == 1.0


def test_metrics_for_flat_equity():
    m = compute_metrics(pd.Series([100.0, 100.0]), pd.Series([0.0, 0.0]), trades=0)
    assert m.sharpe == 0.0 and m.volatility == 0.0 and m.cagr == 0.0
