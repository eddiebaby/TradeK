"""Turn target weights into concrete orders (pure logic, no I/O)."""

from __future__ import annotations

from dataclasses import dataclass

from .brokers.base import Side
from .ledger import DUST


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: Side
    dollars: float | None = None  # buys are sized in dollars (fractional shares)
    quantity: float | None = None  # sells are sized in shares
    reason: str = ""
    closes_position: bool = False

    def value(self, price: float) -> float:
        if self.dollars is not None:
            return self.dollars
        return (self.quantity or 0.0) * price

    def describe(self, price: float | None = None) -> str:
        if self.side is Side.BUY:
            size = f"${self.dollars:.2f} of"
        else:
            size = f"{self.quantity:.6f} sh" + (f" (~${self.value(price):.2f})" if price else "")
        return f"{self.side.value.upper():4} {size} {self.symbol} — {self.reason}"


def plan_orders(
    weights: dict[str, float],
    prices: dict[str, float],
    held: dict[str, float],
    cash: float,
    equity: float,
    slots: int,
    rebalance_threshold: float = 5.0,
) -> list[OrderRequest]:
    """Orders that move each symbol toward ``weight * equity / slots`` dollars.

    * ``slots`` is the number of configured symbols; each gets an equal
      share of the bot's equity, even if some are skipped this run.
    * Differences smaller than ``rebalance_threshold`` dollars are ignored
      so the bot doesn't churn on tiny price moves.
    * Buys are funded only from cash already on hand (sell proceeds are
      used on the next run, once the sells have filled).
    """
    if slots < 1:
        raise ValueError("slots must be >= 1")
    per_slot = max(equity, 0.0) / slots
    sells: list[OrderRequest] = []
    buys: list[OrderRequest] = []
    cash_left = max(cash, 0.0)

    for symbol in sorted(weights):
        weight = min(max(weights[symbol], 0.0), 1.0)
        price = prices[symbol]
        qty = held.get(symbol, 0.0)
        current = qty * price
        target = weight * per_slot
        diff = target - current

        if weight == 0.0 and qty > DUST:
            sells.append(OrderRequest(symbol, Side.SELL, quantity=qty, reason="signal: exit", closes_position=True))
        elif diff < -rebalance_threshold and qty > DUST:
            sell_qty = min(qty, -diff / price)
            sells.append(OrderRequest(symbol, Side.SELL, quantity=sell_qty, reason="rebalance: trim"))
        elif diff > rebalance_threshold and cash_left > 0:
            dollars = round(min(diff, cash_left), 2)
            cash_left -= dollars
            reason = "signal: enter" if qty <= DUST else "rebalance: add"
            buys.append(OrderRequest(symbol, Side.BUY, dollars=dollars, reason=reason))
    return sells + buys
