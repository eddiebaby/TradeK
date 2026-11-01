"""A simulated broker that fills orders instantly with fake money.

State is persisted to a JSON file so a paper account survives between
runs exactly like a real one — run the bot on a schedule for a few weeks
and see how it would have done before risking a cent.
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from ..market_hours import is_regular_session
from ..storage import atomic_write_json
from .base import BrokerError, OrderStatus


class PaperBroker:
    name = "paper"
    is_live = False

    def __init__(
        self,
        state_path: Path,
        price_source: Callable[[str], float],
        starting_cash: float = 200.0,
        slippage_bps: float = 5.0,
    ) -> None:
        self._path = Path(state_path)
        self._price_source = price_source
        self._slippage = slippage_bps / 10_000
        if self._path.exists():
            self._state = json.loads(self._path.read_text())
        else:
            self._state = {"cash": float(starting_cash), "positions": {}, "orders": {}}

    # -- account -----------------------------------------------------------
    @property
    def cash(self) -> float:
        return float(self._state["cash"])

    def get_price(self, symbol: str) -> float:
        price = float(self._price_source(symbol))
        if not math.isfinite(price) or price <= 0:
            raise BrokerError(f"no valid price for {symbol}: {price}")
        return price

    def get_positions(self) -> dict[str, float]:
        return {s: float(q) for s, q in self._state["positions"].items()}

    def get_buying_power(self) -> float:
        return self.cash

    def is_market_open(self, now: datetime) -> bool:
        return is_regular_session(now)

    # -- orders --------------------------------------------------------------
    def buy_dollars(self, symbol: str, dollars: float) -> str:
        if dollars <= 0:
            raise BrokerError("buy amount must be positive")
        if dollars > self.cash + 1e-9:
            raise BrokerError(f"insufficient buying power: need {dollars:.2f}, have {self.cash:.2f}")
        price = self.get_price(symbol) * (1 + self._slippage)
        qty = math.floor(dollars / price * 1e6) / 1e6
        self._state["cash"] = self.cash - qty * price
        held = self._state["positions"].get(symbol, 0.0)
        self._state["positions"][symbol] = held + qty
        return self._record(symbol, "buy", qty, price)

    def sell_quantity(self, symbol: str, quantity: float) -> str:
        held = self._state["positions"].get(symbol, 0.0)
        if quantity <= 0:
            raise BrokerError("sell quantity must be positive")
        if quantity > held + 1e-9:
            raise BrokerError(f"cannot sell {quantity} {symbol}: only {held} held")
        price = self.get_price(symbol) * (1 - self._slippage)
        quantity = min(quantity, held)
        self._state["cash"] = self.cash + quantity * price
        remaining = held - quantity
        if remaining > 1e-9:
            self._state["positions"][symbol] = remaining
        else:
            self._state["positions"].pop(symbol, None)
        return self._record(symbol, "sell", quantity, price)

    def get_order(self, order_id: str) -> OrderStatus:
        try:
            o = self._state["orders"][order_id]
        except KeyError:
            raise BrokerError(f"unknown order {order_id}") from None
        return OrderStatus(order_id, o["state"], o["quantity"], o["price"])

    def _record(self, symbol: str, side: str, qty: float, price: float) -> str:
        order_id = uuid.uuid4().hex
        self._state["orders"][order_id] = {
            "symbol": symbol,
            "side": side,
            "quantity": qty,
            "price": price,
            "state": "filled",
        }
        atomic_write_json(self._path, self._state)
        return order_id
