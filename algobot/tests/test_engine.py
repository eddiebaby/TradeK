from datetime import timedelta

import pytest

from algobot.brokers.base import BrokerError
from algobot.engine import Engine
from algobot.journal import Journal
from algobot.ledger import LedgerStore
from algobot.risk import RiskLimits, RiskManager
from algobot.strategies import SmaCrossover

from .conftest import MARKET_OPEN_UTC, FakeBroker, FakeData, trend

UP = trend(80, start=100, step=1)  # ends at 179 -> SMA says invested
DOWN = trend(80, start=300, step=-1)  # SMA says cash


def make(tmp_state, *, data=None, broker=None, budget=150.0, clock=None, **risk_kw):
    broker = broker or FakeBroker({"SPY": 179.0})
    limits = dict(allowed_symbols=frozenset({"SPY", "QQQ"}), max_order_dollars=1_000,
                  kill_switch_path=tmp_state / "STOP")
    limits.update(risk_kw)
    engine = Engine(
        broker=broker,
        data=data or FakeData({"SPY": UP}),
        strategy=SmaCrossover(5, 20),
        symbols=["SPY"],
        store=LedgerStore(tmp_state / "ledger.json", budget),
        risk=RiskManager(RiskLimits(**limits)),
        journal=Journal(tmp_state / "journal.csv", broker.name),
        clock=clock or (lambda: MARKET_OPEN_UTC),
    )
    return engine, broker


def test_first_run_buys_the_budget_and_books_the_fill(tmp_state):
    engine, broker = make(tmp_state)
    report = engine.run_once()
    assert report.weights == {"SPY": 1.0}
    assert [o.dollars for o, _ in report.submitted] == [150.0]
    led = engine.store.load()
    assert led.positions["SPY"] == pytest.approx(150 / 179)
    assert led.cash == pytest.approx(0) and led.pending == [] and led.orders_today == 1
    events = [r["event"] for r in engine.journal.read()]
    assert events == ["submitted", "filled"]
    assert "submitted buy SPY" in report.summary()


def test_second_run_does_nothing_when_already_positioned(tmp_state):
    engine, _ = make(tmp_state)
    engine.run_once()
    report = engine.run_once()
    assert report.decisions == [] and report.submitted == []
    assert "nothing to do" in report.summary()


def test_signal_flip_sells_everything_next_day(tmp_state):
    engine, broker = make(tmp_state)
    engine.run_once()
    engine.data = FakeData({"SPY": DOWN})
    engine.clock = lambda: MARKET_OPEN_UTC + timedelta(days=1)
    report = engine.run_once()
    assert [(o.side.value, o.closes_position) for o, _ in report.submitted] == [("sell", True)]
    led = engine.store.load()
    assert led.positions == {} and led.cash == pytest.approx(150)


def test_same_day_exit_is_blocked_to_avoid_a_day_trade(tmp_state):
    engine, _ = make(tmp_state)
    engine.run_once()
    engine.data = FakeData({"SPY": DOWN})
    report = engine.run_once()
    assert report.submitted == [] and "holding" in report.decisions[0].reason


def test_kill_switch_stops_everything(tmp_state):
    engine, broker = make(tmp_state)
    tmp_state.mkdir(parents=True)
    (tmp_state / "STOP").touch()
    report = engine.run_once()
    assert "kill switch" in report.blocked and broker.orders == {}
    assert "no trading" in report.summary()


def test_market_closed_means_no_trading(tmp_state):
    engine, broker = make(tmp_state, broker=FakeBroker({"SPY": 179.0}, market_open=False))
    report = engine.run_once()
    assert report.blocked == "market is closed" and broker.orders == {}


def test_ignore_market_hours_for_paper_only(tmp_state):
    engine, broker = make(tmp_state, broker=FakeBroker({"SPY": 179.0}, market_open=False))
    engine.ignore_market_hours = True
    assert engine.run_once().submitted
    with pytest.raises(ValueError, match="paper"):
        Engine(broker=FakeBroker({}, is_live=True), data=None, strategy=None, symbols=["SPY"],
               store=None, risk=None, journal=None, ignore_market_hours=True)


def test_engine_requires_symbols():
    with pytest.raises(ValueError):
        Engine(broker=FakeBroker({}), data=None, strategy=None, symbols=[], store=None,
               risk=None, journal=None)


def test_dry_run_plans_but_never_sends_or_saves(tmp_state):
    engine, broker = make(tmp_state)
    report = engine.run_once(dry_run=True)
    assert report.decisions[0].approved and report.submitted == [] and broker.orders == {}
    assert not engine.store.path.exists()
    assert "(dry run)" in report.summary()


def test_confirmation_declined(tmp_state):
    engine, broker = make(tmp_state)
    seen = []
    report = engine.run_once(confirm=lambda orders, prices: seen.append(orders) or False)
    assert report.blocked == "declined at confirmation prompt" and broker.orders == {}
    assert len(seen[0]) == 1


def test_pending_order_is_reconciled_on_a_later_run(tmp_state):
    engine, broker = make(tmp_state)
    broker.auto_fill = False
    first = engine.run_once()
    assert first.fills == [] and len(engine.store.load().pending) == 1
    second = engine.run_once()  # still queued -> symbol skipped, no duplicate order
    assert "pending" in second.skipped["SPY"] and len(broker.orders) == 1
    broker.fill("o1")
    third = engine.run_once()
    assert third.fills and engine.store.load().positions["SPY"] > 0


def test_partial_fill_then_cancel_books_only_what_filled(tmp_state):
    engine, broker = make(tmp_state)
    broker.auto_fill = False
    engine.run_once()
    broker.fill("o1", state="cancelled", qty=0.2)
    engine.run_once()
    assert engine.store.load().positions["SPY"] == pytest.approx(0.2)


def test_fill_without_price_stays_pending(tmp_state, monkeypatch):
    engine, broker = make(tmp_state)
    broker.auto_fill = False
    engine.run_once()
    broker.fill("o1")
    from algobot.brokers.base import OrderStatus

    monkeypatch.setattr(broker, "get_order", lambda oid: OrderStatus(oid, "filled", 0.5, None))
    report = engine.run_once()
    assert any("without a price" in e for e in report.errors)
    assert len(engine.store.load().pending) == 1


def test_order_lookup_failure_keeps_order_pending(tmp_state, monkeypatch):
    engine, broker = make(tmp_state)
    broker.auto_fill = False
    engine.run_once()

    def boom(oid):
        raise BrokerError("timeout")

    monkeypatch.setattr(broker, "get_order", boom)
    report = engine.run_once()
    assert any("timeout" in e for e in report.errors) and len(engine.store.load().pending) == 1


def test_rejected_order_is_logged_and_run_continues(tmp_state):
    engine, broker = make(tmp_state)
    broker.fail_next = BrokerError("insufficient funds")
    report = engine.run_once()
    assert report.errors == ["buy SPY: insufficient funds"] and not engine.store.load().halted
    assert engine.journal.read()[0]["event"] == "error"


def test_ambiguous_failure_halts_the_bot(tmp_state):
    engine, broker = make(tmp_state)
    broker.fail_next = BrokerError("no response", ambiguous=True)
    report = engine.run_once()
    led = engine.store.load()
    assert led.halted and "reset-halt" in led.halt_reason
    assert "HALTED" in report.summary()
    assert engine.run_once().decisions[0].approved is False  # buys stay blocked


def test_drawdown_halts_new_buys(tmp_state):
    engine, broker = make(tmp_state, max_drawdown_pct=20)
    led = engine.store.load()
    led.peak_equity = 300  # pretend we were up a lot; equity is now 150
    engine.store.save(led)
    report = engine.run_once()
    assert "drawdown" in report.halted and report.submitted == []


@pytest.mark.parametrize(
    "data,expected",
    [
        (FakeData({}), "no data"),
        (FakeData({"SPY": trend(10)}), "only 10 bars"),
        (FakeData({"SPY": trend(80, end="2026-09-01")}), "stale data"),
    ],
)
def test_bad_data_never_trades(tmp_state, data, expected):
    engine, broker = make(tmp_state, data=data)
    report = engine.run_once()
    assert expected in report.skipped["SPY"] and broker.orders == {}


def test_non_finite_signal_is_skipped(tmp_state, monkeypatch):
    engine, broker = make(tmp_state)
    monkeypatch.setattr(engine.strategy.__class__, "target_weights",
                        lambda self, close: close * float("nan"))
    report = engine.run_once()
    assert "returned nan" in report.skipped["SPY"] and broker.orders == {}


def test_broker_outage_is_reported_not_raised(tmp_state, monkeypatch):
    engine, broker = make(tmp_state)

    def down(symbol):
        raise BrokerError("503")

    monkeypatch.setattr(broker, "get_price", down)
    report = engine.run_once()
    assert report.errors == ["broker: 503"]


def test_shares_you_hold_by_hand_are_never_sold(tmp_state):
    broker = FakeBroker({"SPY": 179.0})
    broker.positions["SPY"] = 10.0  # yours, bought manually
    engine, _ = make(tmp_state, broker=broker, data=FakeData({"SPY": DOWN}))
    report = engine.run_once()
    assert report.decisions == [] and broker.positions["SPY"] == 10.0


def test_ledger_larger_than_broker_uses_smaller_and_warns(tmp_state):
    engine, broker = make(tmp_state)
    led = engine.store.load()
    led.positions["SPY"] = 1.0
    led.cash = 0
    engine.store.save(led)  # broker shows 0 shares (sold in the app)
    report = engine.run_once()
    assert any("using the smaller" in e for e in report.errors)


def test_buying_power_caps_buys(tmp_state):
    engine, broker = make(tmp_state, broker=FakeBroker({"SPY": 179.0}, buying_power=40))
    report = engine.run_once()
    assert [o.dollars for o, _ in report.submitted] == [40.0]
