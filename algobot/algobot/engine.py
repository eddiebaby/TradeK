"""One trading run: reconcile -> check risk -> signal -> plan -> review -> submit.

Designed to be run once per trading day (a few minutes before the close)
from cron or Task Scheduler. Every step is recorded in the run report and
the journal so you can always see *why* the bot did or did not trade.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from .brokers.base import Broker, BrokerError, Side
from .data import DataError, PriceData
from .journal import Journal
from .ledger import Ledger, LedgerStore, PendingOrder
from .market_hours import NEW_YORK, trading_date
from .planner import OrderRequest, plan_orders
from .risk import Decision, RiskManager
from .strategies import Strategy


@dataclass
class RunReport:
    started_at: datetime
    broker: str
    dry_run: bool
    blocked: str | None = None
    weights: dict[str, float] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    prices: dict[str, float] = field(default_factory=dict)
    decisions: list[Decision] = field(default_factory=list)
    submitted: list[tuple[OrderRequest, str]] = field(default_factory=list)
    fills: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    equity: float | None = None
    halted: str = ""

    def summary(self) -> str:
        mode = f"{self.broker}{' (dry run)' if self.dry_run else ''}"
        local = self.started_at.astimezone(NEW_YORK)
        lines = [f"algobot run @ {local:%Y-%m-%d %H:%M %Z} — {mode}"]
        if self.equity is not None:
            lines.append(f"  bot equity: ${self.equity:.2f}")
        if self.halted:
            lines.append(f"  HALTED: {self.halted}")
        if self.blocked:
            lines.append(f"  no trading: {self.blocked}")
        for symbol, weight in self.weights.items():
            lines.append(f"  signal {symbol}: target {weight:.0%} invested")
        for symbol, why in self.skipped.items():
            lines.append(f"  skipped {symbol}: {why}")
        for d in self.decisions:
            mark = "OK " if d.approved else "NO "
            extra = f" [{d.reason}]" if d.reason else ""
            lines.append(f"  {mark}{d.order.describe(self.prices.get(d.order.symbol))}{extra}")
        for order, order_id in self.submitted:
            lines.append(f"  submitted {order.side.value} {order.symbol} -> order {order_id}")
        lines.extend(f"  filled {f}" for f in self.fills)
        lines.extend(f"  ERROR {e}" for e in self.errors)
        if not self.decisions and not self.blocked:
            lines.append("  nothing to do — positions already match the signal")
        return "\n".join(lines)


class Engine:
    def __init__(
        self,
        *,
        broker: Broker,
        data: PriceData,
        strategy: Strategy,
        symbols: Sequence[str],
        store: LedgerStore,
        risk: RiskManager,
        journal: Journal,
        rebalance_threshold: float = 5.0,
        max_staleness_days: int = 5,
        ignore_market_hours: bool = False,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not symbols:
            raise ValueError("need at least one symbol")
        if ignore_market_hours and broker.is_live:
            raise ValueError("ignore_market_hours is only allowed for paper trading")
        self.broker = broker
        self.data = data
        self.strategy = strategy
        self.symbols = list(symbols)
        self.store = store
        self.risk = risk
        self.journal = journal
        self.rebalance_threshold = rebalance_threshold
        self.max_staleness_days = max_staleness_days
        self.ignore_market_hours = ignore_market_hours
        self.clock = clock or (lambda: datetime.now(UTC))

    def run_once(
        self,
        dry_run: bool = False,
        confirm: Callable[[list[OrderRequest], dict[str, float]], bool] | None = None,
    ) -> RunReport:
        now = self.clock()
        report = RunReport(started_at=now, broker=self.broker.name, dry_run=dry_run)
        ledger = self.store.load()
        ledger.roll_day(trading_date(now))
        try:
            self._run(ledger, now, report, dry_run, confirm)
        except BrokerError as exc:
            report.errors.append(f"broker: {exc}")
        finally:
            report.halted = ledger.halt_reason if ledger.halted else ""
            if not dry_run:
                self.store.save(ledger)
        return report

    # ------------------------------------------------------------------------
    def _run(self, ledger: Ledger, now: datetime, report: RunReport, dry_run: bool, confirm) -> None:
        today = trading_date(now)
        self._reconcile(ledger, now, report)

        market_open = self.ignore_market_hours or self.broker.is_market_open(now)
        report.blocked = self.risk.blocked_reason(market_open=market_open)
        if report.blocked:
            return

        prices = {s: self.broker.get_price(s) for s in self.symbols}
        report.prices = prices
        broker_positions = self.broker.get_positions()
        held = {}
        for s in self.symbols:
            mine = ledger.positions.get(s, 0.0)
            theirs = broker_positions.get(s, 0.0)
            if mine > theirs + 1e-6:
                report.errors.append(
                    f"{s}: ledger has {mine:.6f} sh but broker shows {theirs:.6f}; using the smaller"
                )
            held[s] = min(mine, theirs)

        equity = ledger.equity(prices, held)
        report.equity = equity
        ledger.peak_equity = max(ledger.peak_equity, equity)
        if not ledger.halted and self.risk.drawdown_breached(ledger, equity):
            ledger.halt(f"drawdown limit hit: equity ${equity:.2f} vs peak ${ledger.peak_equity:.2f}")

        pending = {p.symbol for p in ledger.pending}
        for s in self.symbols:
            if s in pending:
                report.skipped[s] = "an earlier order is still pending"
                continue
            try:
                report.weights[s] = self._signal(s, today)
            except Exception as exc:  # bad data must never turn into a trade
                report.skipped[s] = f"no signal ({exc})"

        cash = max(0.0, min(ledger.cash, self.broker.get_buying_power()))
        orders = plan_orders(
            report.weights, prices, held, cash, equity, len(self.symbols), self.rebalance_threshold
        )
        report.decisions = self.risk.review(orders, ledger, prices, today)
        approved = [d.order for d in report.decisions if d.approved]
        if not approved or dry_run:
            return
        if confirm is not None and not confirm(approved, prices):
            report.blocked = "declined at confirmation prompt"
            return
        self._submit(approved, ledger, now, prices, report)
        self._reconcile(ledger, now, report)

    def _signal(self, symbol: str, today: date) -> float:
        close = self.data.history(symbol)
        if len(close) < self.strategy.warmup:
            raise DataError(f"only {len(close)} bars, strategy needs {self.strategy.warmup}")
        last_bar = close.index[-1].date()
        if (today - last_bar).days > self.max_staleness_days:
            raise DataError(f"stale data, last bar is {last_bar}")
        weight = float(self.strategy.target_weights(close).iloc[-1])
        if not math.isfinite(weight):
            raise DataError(f"strategy returned {weight}")
        return min(max(weight, 0.0), 1.0)

    def _submit(
        self,
        orders: list[OrderRequest],
        ledger: Ledger,
        now: datetime,
        prices: dict[str, float],
        report: RunReport,
    ) -> None:
        for order in orders:
            side = order.side.value
            try:
                if order.side is Side.BUY:
                    order_id = self.broker.buy_dollars(order.symbol, order.dollars)
                else:
                    order_id = self.broker.sell_quantity(order.symbol, order.quantity)
            except BrokerError as exc:
                report.errors.append(f"{side} {order.symbol}: {exc}")
                self.journal.record(now, "error", order.symbol, side, dollars=order.dollars,
                                    quantity=order.quantity, note=str(exc))
                if exc.ambiguous:
                    ledger.halt(f"unclear whether a {side} {order.symbol} order went through — "
                                "check your Robinhood app, then run `algobot reset-halt`")
                    return
                continue
            ledger.pending.append(PendingOrder(order_id, order.symbol, side, now.isoformat()))
            ledger.orders_today += 1
            if order.side is Side.BUY:
                ledger.last_buy_date[order.symbol] = trading_date(now).isoformat()
            report.submitted.append((order, order_id))
            self.journal.record(now, "submitted", order.symbol, side, dollars=order.dollars,
                                quantity=order.quantity, price=prices.get(order.symbol),
                                order_id=order_id, note=order.reason)

    def _reconcile(self, ledger: Ledger, now: datetime, report: RunReport) -> None:
        """Move finished orders from 'pending' into the ledger's cash/positions."""
        still_pending = []
        for p in ledger.pending:
            try:
                status = self.broker.get_order(p.order_id)
            except BrokerError as exc:
                report.errors.append(f"checking order {p.order_id}: {exc}")
                still_pending.append(p)
                continue
            if not status.is_final:
                still_pending.append(p)
                continue
            if status.filled_quantity > 0:
                if not status.average_price:
                    report.errors.append(f"order {p.order_id} filled without a price; will retry")
                    still_pending.append(p)
                    continue
                ledger.apply_fill(p.symbol, p.side, status.filled_quantity, status.average_price)
                report.fills.append(
                    f"{p.side} {status.filled_quantity:.6f} {p.symbol} @ ${status.average_price:.2f}"
                )
            self.journal.record(now, status.state, p.symbol, p.side, quantity=status.filled_quantity,
                                price=status.average_price, order_id=p.order_id)
        ledger.pending = still_pending
