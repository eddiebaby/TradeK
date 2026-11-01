import numpy as np
import pytest

from algobot.strategies import BuyAndHold, RsiMeanReversion, SmaCrossover, build_strategy

from .conftest import series, trend


def test_buy_and_hold_is_always_fully_invested():
    s = BuyAndHold()
    w = s.target_weights(trend(10))
    assert (w == 1.0).all() and s.warmup == 1 and s.name == "buy_and_hold"


def test_sma_invested_in_uptrend_and_out_in_downtrend():
    s = SmaCrossover(fast=5, slow=20)
    assert s.target_weights(trend(60, step=1)).iloc[-1] == 1.0
    assert s.target_weights(trend(60, start=200, step=-1)).iloc[-1] == 0.0


def test_sma_is_flat_during_warmup():
    w = SmaCrossover(fast=5, slow=20).target_weights(trend(60))
    assert (w.iloc[:19] == 0.0).all()


def test_sma_flips_when_trend_reverses():
    up_then_down = series(list(range(100, 160)) + list(range(160, 100, -1)))
    w = SmaCrossover(fast=5, slow=20).target_weights(up_then_down)
    assert w.iloc[59] == 1.0 and w.iloc[-1] == 0.0
    assert set(w.unique()) <= {0.0, 1.0}


@pytest.mark.parametrize("fast,slow", [(20, 20), (30, 10), (0, 5)])
def test_sma_rejects_invalid_windows(fast, slow):
    with pytest.raises(ValueError):
        SmaCrossover(fast=fast, slow=slow)


def test_sma_name_and_warmup():
    s = SmaCrossover(10, 40)
    assert s.name == "sma_10_40" and s.warmup == 40


def test_rsi_strategy_enters_when_oversold_and_holds_until_overbought():
    # Fall hard (oversold) -> drift sideways (no signal, keep holding) -> rally (overbought).
    prices = list(np.linspace(100, 60, 30)) + [60, 61] * 10 + list(np.linspace(61, 120, 30))
    w = RsiMeanReversion(period=5, buy_below=30, sell_above=70).target_weights(series(prices))
    assert w.iloc[29] == 1.0  # entered on the fall
    assert w.iloc[45] == 1.0  # still holding in the chop
    assert w.iloc[-1] == 0.0  # exited on the rally


def test_rsi_strategy_starts_flat():
    w = RsiMeanReversion(period=14).target_weights(trend(10))
    assert (w == 0.0).all()


@pytest.mark.parametrize("kwargs", [{"period": 0}, {"buy_below": 70, "sell_above": 30}, {"sell_above": 100}])
def test_rsi_strategy_validates(kwargs):
    with pytest.raises(ValueError):
        RsiMeanReversion(**kwargs)


def test_rsi_name_and_warmup():
    s = RsiMeanReversion(2, 10, 90)
    assert s.name == "rsi_2_10_90" and s.warmup == 3


def test_build_strategy_by_name():
    assert build_strategy("sma", {"fast": 3, "slow": 9}) == SmaCrossover(3, 9)
    assert isinstance(build_strategy("buy_and_hold"), BuyAndHold)


def test_build_strategy_errors_are_friendly():
    with pytest.raises(ValueError, match="unknown strategy"):
        build_strategy("magic")
    with pytest.raises(ValueError, match="bad parameters"):
        build_strategy("sma", {"speed": 9000})
