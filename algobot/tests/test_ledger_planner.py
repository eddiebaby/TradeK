import json
from datetime import date

import pytest

from algobot.brokers.base import Side
from algobot.ledger import Ledger, LedgerStore, PendingOrder
from algobot.planner import OrderRequest, plan_orders


# -- ledger -------------------------------------------------------------------
def test_new_ledger_starts_with_budget_as_cash_and_peak():
    led = Ledger.new(150)
    assert led.cash == 150 and led.peak_equity == 150 and led.positions == {}
    with pytest.raises(ValueError):
        Ledger.new(0)


def test_fills_move_cash_and_shares():
    led = Ledger.new(100)
    led.apply_fill("SPY", "buy", 0.1, 500)
    assert led.cash == pytest.approx(50) and led.positions["SPY"] == pytest.approx(0.1)
    led.apply_fill("SPY", "sell", 0.04, 550)
    assert led.cash == pytest.approx(72) and led.positions["SPY"] == pytest.approx(0.06)
    led.apply_fill("SPY", "sell", 0.06, 550)
    assert "SPY" not in led.positions


@pytest.mark.parametrize("args", [("SPY", "buy", 0, 10), ("SPY", "buy", 1, 0), ("SPY", "short", 1, 10)])
def test_bad_fills_rejected(args):
    with pytest.raises(ValueError):
        Ledger.new(100).apply_fill(*args)


def test_equity_uses_override_holdings_and_requires_prices():
    led = Ledger.new(100)
    led.apply_fill("SPY", "buy", 0.1, 500)
    assert led.equity({"SPY": 600}) == pytest.approx(110)
    assert led.equity({"SPY": 600}, held={"SPY": 0.05}) == pytest.approx(80)
    with pytest.raises(ValueError):
        led.equity({})


def test_roll_day_resets_order_count_only_on_new_day():
    led = Ledger.new(100)
    led.roll_day(date(2026, 10, 28))
    led.orders_today = 3
    led.roll_day(date(2026, 10, 28))
    assert led.orders_today == 3
    led.roll_day(date(2026, 10, 29))
    assert led.orders_today == 0


def test_halt_and_clear():
    led = Ledger.new(100)
    led.halt("because")
    assert led.halted and led.halt_reason == "because"
    led.clear_halt()
    assert not led.halted and led.halt_reason == ""


def test_store_round_trip_and_default(tmp_path):
    store = LedgerStore(tmp_path / "s" / "ledger.json", budget=150)
    assert store.load().cash == 150  # no file yet -> fresh ledger
    led = store.load()
    led.apply_fill("QQQ", "buy", 0.2, 400)
    led.pending.append(PendingOrder("abc", "QQQ", "buy", "2026-10-28T19:45:00+00:00"))
    led.halt("x")
    store.save(led)
    again = store.load()
    assert again == led
    assert json.loads(store.path.read_text())["pending"][0]["order_id"] == "abc"
    assert not list(store.path.parent.glob("*.tmp"))  # atomic write cleaned up


# -- planner ------------------------------------------------------------------
def test_enter_full_position_from_cash():
    orders = plan_orders({"SPY": 1.0}, {"SPY": 500}, {}, cash=150, equity=150, slots=1)
    assert orders == [OrderRequest("SPY", Side.BUY, dollars=150.0, reason="signal: enter")]


def test_exit_sells_entire_position():
    orders = plan_orders({"SPY": 0.0}, {"SPY": 500}, {"SPY": 0.3}, cash=0, equity=150, slots=1)
    assert orders == [OrderRequest("SPY", Side.SELL, quantity=0.3, reason="signal: exit", closes_position=True)]


def test_small_drift_is_ignored():
    orders = plan_orders({"SPY": 1.0}, {"SPY": 500}, {"SPY": 0.29}, cash=3, equity=148, slots=1,
                         rebalance_threshold=5)
    assert orders == []


def test_trim_when_over_target():
    # Target 50% of 200 = 100; holding 0.3 * 500 = 150 -> sell $50 = 0.1 sh.
    orders = plan_orders({"SPY": 0.5}, {"SPY": 500}, {"SPY": 0.3}, cash=50, equity=200, slots=1)
    assert len(orders) == 1 and orders[0].side is Side.SELL
    assert orders[0].quantity == pytest.approx(0.1) and not orders[0].closes_position


def test_add_when_under_target():
    orders = plan_orders({"SPY": 1.0}, {"SPY": 500}, {"SPY": 0.1}, cash=100, equity=150, slots=1)
    assert orders == [OrderRequest("SPY", Side.BUY, dollars=100.0, reason="rebalance: add")]


def test_budget_split_across_slots_and_capped_by_cash():
    orders = plan_orders({"AAA": 1.0, "BBB": 1.0}, {"AAA": 10, "BBB": 20}, {}, cash=120, equity=200, slots=2)
    assert [(o.symbol, o.dollars) for o in orders] == [("AAA", 100.0), ("BBB", 20.0)]


def test_skipped_symbols_still_reserve_their_slot():
    orders = plan_orders({"AAA": 1.0}, {"AAA": 10}, {}, cash=200, equity=200, slots=2)
    assert orders[0].dollars == 100.0


def test_sells_come_before_buys_and_no_buy_without_cash():
    orders = plan_orders({"AAA": 1.0, "ZZZ": 0.0}, {"AAA": 10, "ZZZ": 10}, {"ZZZ": 5}, cash=0,
                         equity=100, slots=2)
    assert [o.side for o in orders] == [Side.SELL]


def test_slots_validated():
    with pytest.raises(ValueError):
        plan_orders({}, {}, {}, 0, 0, slots=0)


def test_order_describe_and_value():
    buy = OrderRequest("SPY", Side.BUY, dollars=12.5, reason="r")
    sell = OrderRequest("SPY", Side.SELL, quantity=0.5, reason="r")
    assert buy.value(500) == 12.5 and sell.value(500) == 250
    assert "BUY  $12.50 of SPY" in buy.describe()
    assert "~$250.00" in sell.describe(500) and "~$" not in sell.describe()
