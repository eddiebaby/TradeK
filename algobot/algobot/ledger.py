"""The bot's own books: a virtual sub-account inside your brokerage account.

The bot never treats your whole Robinhood account as its own. It starts
with ``budget`` dollars of *virtual* cash and only counts shares it has
bought itself. So if you also hold SPY by hand, the bot will never sell
your shares — it can only sell what its own fills put on the ledger.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from .storage import atomic_write_json

DUST = 1e-9


@dataclass
class PendingOrder:
    order_id: str
    symbol: str
    side: str  # "buy" | "sell"
    submitted_at: str


@dataclass
class Ledger:
    cash: float
    positions: dict[str, float] = field(default_factory=dict)
    peak_equity: float = 0.0
    halted: bool = False
    halt_reason: str = ""
    pending: list[PendingOrder] = field(default_factory=list)
    last_buy_date: dict[str, str] = field(default_factory=dict)
    orders_today: int = 0
    orders_date: str = ""

    @classmethod
    def new(cls, budget: float) -> Ledger:
        if budget <= 0:
            raise ValueError("budget must be positive")
        return cls(cash=float(budget), peak_equity=float(budget))

    def roll_day(self, today: date) -> None:
        if self.orders_date != today.isoformat():
            self.orders_date = today.isoformat()
            self.orders_today = 0

    def apply_fill(self, symbol: str, side: str, quantity: float, price: float) -> None:
        if quantity <= 0 or price <= 0:
            raise ValueError("fill quantity and price must be positive")
        held = self.positions.get(symbol, 0.0)
        if side == "buy":
            self.cash -= quantity * price
            self.positions[symbol] = held + quantity
        elif side == "sell":
            self.cash += quantity * price
            remaining = held - quantity
            if remaining > DUST:
                self.positions[symbol] = remaining
            else:
                self.positions.pop(symbol, None)
        else:
            raise ValueError(f"unknown side {side!r}")

    def equity(self, prices: dict[str, float], held: dict[str, float] | None = None) -> float:
        """Virtual cash plus market value of the bot's shares."""
        held = self.positions if held is None else held
        total = self.cash
        for symbol, qty in held.items():
            if qty <= DUST:
                continue
            if symbol not in prices:
                raise ValueError(f"no price for held symbol {symbol}")
            total += qty * prices[symbol]
        return total

    def halt(self, reason: str) -> None:
        self.halted = True
        self.halt_reason = reason

    def clear_halt(self) -> None:
        self.halted = False
        self.halt_reason = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Ledger:
        data = dict(data)
        data["pending"] = [PendingOrder(**p) for p in data.get("pending", [])]
        return cls(**data)


class LedgerStore:
    def __init__(self, path: Path, budget: float) -> None:
        self.path = Path(path)
        self.budget = budget

    def load(self) -> Ledger:
        if self.path.exists():
            return Ledger.from_dict(json.loads(self.path.read_text()))
        return Ledger.new(self.budget)

    def save(self, ledger: Ledger) -> None:
        atomic_write_json(self.path, ledger.to_dict())
