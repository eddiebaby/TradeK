from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from algobot.brokers.base import BrokerError, OrderStatus

# Wednesday 2026-10-28, 15:45 New York (EDT, UTC-4) — market open.
MARKET_OPEN_UTC = datetime(2026, 10, 28, 19, 45, tzinfo=UTC)
# Saturday — market closed.
WEEKEND_UTC = datetime(2026, 10, 31, 16, 0, tzinfo=UTC)


def series(values, end="2026-10-28") -> pd.Series:
    idx = pd.bdate_range(end=end, periods=len(values))
    return pd.Series(np.asarray(values, dtype=float), index=idx, name="close")


def trend(n=120, start=100.0, step=1.0, end="2026-10-28") -> pd.Series:
    return series(start + step * np.arange(n), end=end)


class FakeData:
    def __init__(self, histories: dict[str, pd.Series]):
        self.histories = histories
        self.calls: list[str] = []

    def history(self, symbol: str) -> pd.Series:
        self.calls.append(symbol)
        if symbol not in self.histories:
            raise KeyError(f"no data for {symbol}")
        return self.histories[symbol]


class FakeBroker:
    """Scriptable live-like broker: orders stay 'queued' until fill() is called."""

    name = "fake"

    def __init__(self, prices, *, buying_power=1_000.0, market_open=True, is_live=False):
        self.prices = dict(prices)
        self.positions: dict[str, float] = {}
        self.buying_power = buying_power
        self.market_open = market_open
        self.is_live = is_live
        self.orders: dict[str, dict] = {}
        self.fail_next: BrokerError | None = None
        self.auto_fill = True

    def get_price(self, symbol):
        return self.prices[symbol]

    def get_positions(self):
        return dict(self.positions)

    def get_buying_power(self):
        return self.buying_power

    def is_market_open(self, now):
        return self.market_open

    def _new(self, symbol, side, qty):
        if self.fail_next is not None:
            exc, self.fail_next = self.fail_next, None
            raise exc
        oid = f"o{len(self.orders) + 1}"
        self.orders[oid] = {"symbol": symbol, "side": side, "qty": qty, "state": "queued"}
        if self.auto_fill:
            self.fill(oid)
        return oid

    def buy_dollars(self, symbol, dollars):
        return self._new(symbol, "buy", dollars / self.prices[symbol])

    def sell_quantity(self, symbol, quantity):
        return self._new(symbol, "sell", quantity)

    def fill(self, oid, state="filled", qty=None):
        o = self.orders[oid]
        o["state"] = state
        o["filled"] = o["qty"] if qty is None else qty
        sign = 1 if o["side"] == "buy" else -1
        self.positions[o["symbol"]] = self.positions.get(o["symbol"], 0.0) + sign * o["filled"]

    def get_order(self, oid):
        o = self.orders[oid]
        filled = o.get("filled", 0.0)
        return OrderStatus(oid, o["state"], filled, self.prices[o["symbol"]] if filled else None)


@pytest.fixture
def tmp_state(tmp_path) -> Path:
    return tmp_path / "state"
