"""Risk limits. Every order passes through here before it reaches a broker.

Philosophy: rules that *add* risk (buys) are strict; rules never block an
exit purely because of its size, since selling reduces risk.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

from .brokers.base import Side
from .ledger import Ledger
from .planner import OrderRequest


@dataclass(frozen=True)
class RiskLimits:
    allowed_symbols: frozenset[str]
    max_order_dollars: float = 100.0  # hard cap on any single buy
    min_order_dollars: float = 1.0  # Robinhood's fractional minimum
    max_orders_per_day: int = 4
    max_drawdown_pct: float = 25.0  # halt new buys after this drop from peak
    min_hold_days: int = 1  # never sell something bought today (no day trades)
    kill_switch_path: Path | None = None  # if this file exists, do nothing

    def __post_init__(self) -> None:
        if self.max_order_dollars < self.min_order_dollars:
            raise ValueError("max_order_dollars must be >= min_order_dollars")
        if self.max_orders_per_day < 0:
            raise ValueError("max_orders_per_day cannot be negative")
        if not 0 < self.max_drawdown_pct <= 100:
            raise ValueError("max_drawdown_pct must be in (0, 100]")
        if self.min_hold_days < 0:
            raise ValueError("min_hold_days cannot be negative")


@dataclass(frozen=True)
class Decision:
    order: OrderRequest
    approved: bool
    reason: str = ""


class RiskManager:
    def __init__(self, limits: RiskLimits) -> None:
        self.limits = limits

    def blocked_reason(self, market_open: bool) -> str | None:
        """A reason to skip the whole run, or None if trading may proceed."""
        ks = self.limits.kill_switch_path
        if ks is not None and Path(ks).exists():
            return f"kill switch is on ({ks} exists) — delete it to resume"
        if not market_open:
            return "market is closed"
        return None

    def drawdown_breached(self, ledger: Ledger, equity: float) -> bool:
        if ledger.peak_equity <= 0:
            return False
        return equity < ledger.peak_equity * (1 - self.limits.max_drawdown_pct / 100)

    def review(
        self,
        orders: list[OrderRequest],
        ledger: Ledger,
        prices: dict[str, float],
        today: date,
    ) -> list[Decision]:
        remaining = self.limits.max_orders_per_day - ledger.orders_today
        ordered = [o for o in orders if o.side is Side.SELL] + [o for o in orders if o.side is Side.BUY]
        decisions = []
        for order in ordered:
            order, problem, note = self._check(order, ledger, prices, today)
            if problem:
                decisions.append(Decision(order, False, problem))
            elif remaining <= 0:
                decisions.append(Decision(order, False, "daily order limit reached"))
            else:
                remaining -= 1
                decisions.append(Decision(order, True, note))
        return decisions

    def _check(
        self, order: OrderRequest, ledger: Ledger, prices: dict[str, float], today: date
    ) -> tuple[OrderRequest, str, str]:
        lim = self.limits
        if order.symbol not in lim.allowed_symbols:
            return order, f"{order.symbol} is not in the allowed symbol list", ""
        value = order.value(prices[order.symbol])

        if order.side is Side.BUY:
            if ledger.halted:
                return order, f"buys halted: {ledger.halt_reason}", ""
            note = ""
            if value > lim.max_order_dollars:
                order = replace(order, dollars=lim.max_order_dollars)
                note = f"clipped from ${value:.2f} to max ${lim.max_order_dollars:.2f}"
                value = lim.max_order_dollars
            if value < lim.min_order_dollars:
                return order, f"${value:.2f} is below the ${lim.min_order_dollars:.2f} minimum", ""
            return order, "", note

        bought = ledger.last_buy_date.get(order.symbol)
        if bought and (today - date.fromisoformat(bought)).days < lim.min_hold_days:
            return order, f"bought on {bought}; holding at least {lim.min_hold_days} day(s)", ""
        if not order.closes_position and value < lim.min_order_dollars:
            return order, f"${value:.2f} trim is below the ${lim.min_order_dollars:.2f} minimum", ""
        return order, "", ""
