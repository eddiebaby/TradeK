import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from algobot.indicators import rsi, sma

from .conftest import series


def test_sma_matches_hand_computed_values():
    out = sma(series([1, 2, 3, 4, 5]), 3)
    assert out.isna().sum() == 2
    assert out.iloc[2:].tolist() == [2.0, 3.0, 4.0]


@pytest.mark.parametrize("bad", [0, -1])
def test_sma_rejects_bad_window(bad):
    with pytest.raises(ValueError):
        sma(series([1, 2, 3]), bad)


def test_rsi_warmup_is_nan_for_period_bars():
    out = rsi(series(np.arange(1, 31)), period=14)
    assert out.iloc[:14].isna().all()
    assert out.iloc[14:].notna().all()


def test_rsi_only_gains_is_100_only_losses_is_0_flat_is_50():
    assert rsi(series(np.arange(1, 31)), 14).iloc[-1] == 100.0
    assert rsi(series(np.arange(30, 0, -1)), 14).iloc[-1] == 0.0
    assert rsi(series([5.0] * 30), 14).iloc[-1] == 50.0


def test_rsi_known_value_for_alternating_series():
    # Equal up and down moves -> average gain ~= average loss -> RSI near 50.
    values = [100 + (i % 2) for i in range(200)]
    assert rsi(series(values), 14).iloc[-1] == pytest.approx(50, abs=5)


def test_rsi_rejects_bad_period():
    with pytest.raises(ValueError):
        rsi(series([1, 2, 3]), 0)


@settings(max_examples=200, deadline=None)
@given(st.lists(st.floats(min_value=1, max_value=1_000), min_size=20, max_size=120))
def test_rsi_is_always_between_0_and_100(values):
    out = rsi(pd.Series(values), 14).dropna()
    assert ((out >= 0) & (out <= 100)).all()
