from datetime import date

import pytest

from algobot.brokers.base import Side
from algobot.ledger import Ledger
from algobot.planner import OrderRequest
from algobot.risk import RiskLimits, RiskManager

TODAY = date(2026, 10, 28)
PRICES = {"SPY": 500.0, "QQQ": 400.0}


def mgr(**kw) -> RiskManager:
    return RiskManager(RiskLimits(allowed_symbols=frozenset({"SPY", "QQQ"}), **kw))


def buy(dollars, symbol="SPY"):
    return OrderRequest(symbol, Side.BUY, dollars=dollars)


def sell(qty, symbol="SPY", closes=False):
    return OrderRequest(symbol, Side.SELL, quantity=qty, closes_position=closes)


def test_kill_switch_and_market_hours_block_everything(tmp_path):
    ks = tmp_path / "STOP"
    m = mgr(kill_switch_path=ks)
    assert m.blocked_reason(market_open=True) is None
    assert m.blocked_reason(market_open=False) == "market is closed"
    ks.touch()
    assert "kill switch" in m.blocked_reason(market_open=True)


def test_symbol_whitelist():
    [d] = mgr().review([buy(10, "TSLA")], Ledger.new(100), {"TSLA": 1.0}, TODAY)
    assert not d.approved and "not in the allowed" in d.reason


def test_buy_clipped_to_max_order_size():
    [d] = mgr(max_order_dollars=50).review([buy(120)], Ledger.new(200), PRICES, TODAY)
    assert d.approved and d.order.dollars == 50 and "clipped" in d.reason


def test_buy_below_minimum_rejected():
    [d] = mgr().review([buy(0.5)], Ledger.new(100), PRICES, TODAY)
    assert not d.approved and "minimum" in d.reason


def test_halted_ledger_blocks_buys_but_not_sells():
    led = Ledger.new(100)
    led.halt("drawdown")
    decisions = mgr().review([buy(10), sell(0.1, closes=True)], led, PRICES, TODAY)
    by_side = {d.order.side: d for d in decisions}
    assert not by_side[Side.BUY].approved and "drawdown" in by_side[Side.BUY].reason
    assert by_side[Side.SELL].approved


def test_no_selling_shares_bought_today():
    led = Ledger.new(100)
    led.last_buy_date["SPY"] = TODAY.isoformat()
    [d] = mgr(min_hold_days=1).review([sell(0.1, closes=True)], led, PRICES, TODAY)
    assert not d.approved and "holding" in d.reason
    [d] = mgr(min_hold_days=1).review([sell(0.1, closes=True)], led, PRICES, date(2026, 10, 29))
    assert d.approved


def test_tiny_trim_rejected_but_tiny_full_exit_allowed():
    tiny = 0.001  # $0.50 at $500
    [d] = mgr().review([sell(tiny)], Ledger.new(100), PRICES, TODAY)
    assert not d.approved
    [d] = mgr().review([sell(tiny, closes=True)], Ledger.new(100), PRICES, TODAY)
    assert d.approved


def test_big_sells_are_never_clipped():
    [d] = mgr(max_order_dollars=10).review([sell(1, closes=True)], Ledger.new(100), PRICES, TODAY)
    assert d.approved and d.order.quantity == 1


def test_daily_order_limit_counts_existing_orders_and_prioritises_sells():
    led = Ledger.new(100)
    led.orders_today = 1
    decisions = mgr(max_orders_per_day=2).review(
        [buy(10), sell(0.1, "QQQ", closes=True)], led, PRICES, TODAY
    )
    assert [(d.order.side, d.approved) for d in decisions] == [(Side.SELL, True), (Side.BUY, False)]
    assert decisions[1].reason == "daily order limit reached"


def test_drawdown_detection():
    led = Ledger.new(100)
    led.peak_equity = 200
    m = mgr(max_drawdown_pct=25)
    assert not m.drawdown_breached(led, 151)
    assert m.drawdown_breached(led, 149)
    led.peak_equity = 0
    assert not m.drawdown_breached(led, 1)


@pytest.mark.parametrize(
    "kw",
    [
        {"max_order_dollars": 0.5},
        {"max_orders_per_day": -1},
        {"max_drawdown_pct": 0},
        {"max_drawdown_pct": 150},
        {"min_hold_days": -1},
    ],
)
def test_limits_validated(kw):
    with pytest.raises(ValueError):
        mgr(**kw)
