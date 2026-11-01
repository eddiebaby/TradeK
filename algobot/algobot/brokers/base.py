"""The broker interface every adapter implements.

Orders are deliberately limited to the two things a small fractional
account needs: buy a *dollar amount* and sell a *share quantity*. Both
are market orders, good for the day, regular hours only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


FINAL_STATES = frozenset({"filled", "cancelled", "canceled", "rejected", "failed"})


class BrokerError(Exception):
    """The broker refused or failed an operation.

    ``ambiguous`` is True when we cannot tell whether an order was placed
    (e.g. a network timeout after sending it). The engine halts trading in
    that case rather than risk a duplicate order.
    """

    def __init__(self, message: str, *, ambiguous: bool = False) -> None:
        super().__init__(message)
        self.ambiguous = ambiguous


@dataclass(frozen=True)
class OrderStatus:
    order_id: str
    state: str
    filled_quantity: float
    average_price: float | None

    @property
    def is_final(self) -> bool:
        return self.state in FINAL_STATES


class Broker(Protocol):
    name: str
    is_live: bool

    def get_price(self, symbol: str) -> float: ...

    def get_positions(self) -> dict[str, float]:
        """Shares held per symbol, as the broker reports them."""
        ...

    def get_buying_power(self) -> float: ...

    def is_market_open(self, now: datetime) -> bool: ...

    def buy_dollars(self, symbol: str, dollars: float) -> str:
        """Submit a market buy for ``dollars`` worth; returns the order id."""
        ...

    def sell_quantity(self, symbol: str, quantity: float) -> str:
        """Submit a market sell of ``quantity`` shares; returns the order id."""
        ...

    def get_order(self, order_id: str) -> OrderStatus: ...
