from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from algobot.brokers.base import BrokerError, OrderStatus
from algobot.brokers.paper import PaperBroker
from algobot.brokers.robinhood import RobinhoodBroker

from .conftest import MARKET_OPEN_UTC, WEEKEND_UTC


# -- paper ----------------------------------------------------------------------
def paper(tmp_path, price=100.0, cash=200.0, slip=0.0):
    return PaperBroker(tmp_path / "paper.json", lambda s: price, starting_cash=cash, slippage_bps=slip)


def test_paper_buy_then_sell_round_trip(tmp_path):
    b = paper(tmp_path, slip=0)
    oid = b.buy_dollars("SPY", 50)
    st = b.get_order(oid)
    assert st == OrderStatus(oid, "filled", 0.5, 100.0) and st.is_final
    assert b.cash == pytest.approx(150) and b.get_positions() == {"SPY": 0.5}
    b.sell_quantity("SPY", 0.5)
    assert b.cash == pytest.approx(200) and b.get_positions() == {}


def test_paper_slippage_costs_money_both_ways(tmp_path):
    b = paper(tmp_path, slip=100)  # 1%
    b.buy_dollars("SPY", 100)
    b.sell_quantity("SPY", b.get_positions()["SPY"])
    assert b.cash < 200 * 0.995


def test_paper_state_persists_between_instances(tmp_path):
    b = paper(tmp_path)
    oid = b.buy_dollars("SPY", 20)
    again = paper(tmp_path)
    assert again.get_positions() == b.get_positions()
    assert again.get_order(oid).state == "filled"
    assert again.get_buying_power() == pytest.approx(180)


@pytest.mark.parametrize(
    "action",
    [
        lambda b: b.buy_dollars("SPY", 500),
        lambda b: b.buy_dollars("SPY", 0),
        lambda b: b.sell_quantity("SPY", 1),
        lambda b: b.sell_quantity("SPY", 0),
        lambda b: b.get_order("nope"),
    ],
)
def test_paper_rejects_impossible_orders(tmp_path, action):
    with pytest.raises(BrokerError):
        action(paper(tmp_path))


@pytest.mark.parametrize("bad", [0.0, -5.0, float("nan")])
def test_paper_rejects_bad_prices(tmp_path, bad):
    with pytest.raises(BrokerError):
        paper(tmp_path, price=bad).get_price("SPY")


def test_paper_market_clock(tmp_path):
    b = paper(tmp_path)
    assert b.is_market_open(MARKET_OPEN_UTC) and not b.is_market_open(WEEKEND_UTC)


# -- robinhood (robin_stocks is replaced by a fake client) -----------------------
class FakeRH:
    def __init__(self, **overrides):
        self.calls = []
        self.responses = {
            "login": {"access_token": "t"},
            "get_latest_price": ["512.34"],
            "get_open_stock_positions": [
                {"symbol": "SPY", "quantity": "0.250000", "instrument": "u1"},
                {"quantity": "2", "instrument": "u2"},
                {"symbol": "ZERO", "quantity": "0", "instrument": "u3"},
            ],
            "get_symbol_by_url": "QQQ",
            "load_account_profile": "123.45",
            "get_market_today_hours": {
                "is_open": True,
                "opens_at": "2026-10-28T13:30:00Z",
                "closes_at": "2026-10-28T20:00:00Z",
            },
            "order_buy_fractional_by_price": {"id": "buy-1", "state": "queued"},
            "order_sell_fractional_by_quantity": {"id": "sell-1", "state": "queued"},
            "get_stock_order_info": {"state": "filled", "cumulative_quantity": "0.1", "average_price": "500.5"},
        }
        self.responses.update(overrides)

    def __getattr__(self, name):
        if name not in self.responses:
            raise AttributeError(name)

        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return self.responses[name]

        return call


def test_rh_login_uses_env_and_totp(monkeypatch):
    monkeypatch.setenv("RH_USERNAME", "me@example.com")
    monkeypatch.setenv("RH_PASSWORD", "pw")
    monkeypatch.setenv("RH_TOTP_SECRET", "JBSWY3DPEHPK3PXP")
    rh = FakeRH()
    RobinhoodBroker(rh).login()
    _, _, kwargs = rh.calls[0]
    assert kwargs["username"] == "me@example.com" and kwargs["store_session"] is True
    assert kwargs["mfa_code"] is not None and len(kwargs["mfa_code"]) == 6


def test_rh_login_without_totp_and_failures(monkeypatch):
    for var in ("RH_USERNAME", "RH_PASSWORD", "RH_TOTP_SECRET"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(BrokerError, match="RH_USERNAME"):
        RobinhoodBroker(FakeRH()).login()
    rh = FakeRH()
    RobinhoodBroker(rh).login("u", "p")
    assert rh.calls[0][2]["mfa_code"] is None
    with pytest.raises(BrokerError, match="login failed"):
        RobinhoodBroker(FakeRH(login=None)).login("u", "p")


def test_rh_price_positions_buying_power():
    b = RobinhoodBroker(FakeRH())
    assert b.get_price("SPY") == 512.34
    assert b.get_positions() == {"SPY": 0.25, "QQQ": 2.0}
    assert b.get_buying_power() == 123.45


@pytest.mark.parametrize("resp", [[None], [], ["0"], ["abc"]])
def test_rh_bad_price_raises(resp):
    with pytest.raises(BrokerError):
        RobinhoodBroker(FakeRH(get_latest_price=resp)).get_price("SPY")


def test_rh_empty_account_is_not_an_error():
    assert RobinhoodBroker(FakeRH(get_open_stock_positions=[])).get_positions() == {}


@pytest.mark.parametrize("resp", [None, [None], [{"symbol": "SPY", "quantity": "1"}, None]])
def test_rh_failed_positions_call_is_not_mistaken_for_empty(resp):
    # robin_stocks returns [None] on HTTP errors — treating that as "no shares" could
    # make the bot think it is flat and buy again.
    with pytest.raises(BrokerError, match="could not load positions"):
        RobinhoodBroker(FakeRH(get_open_stock_positions=resp)).get_positions()


class ExplodingRH(FakeRH):
    def __getattr__(self, name):
        def boom(*args, **kwargs):
            raise ConnectionError("connection reset")

        return boom


def test_rh_crashes_become_broker_errors_reads_are_not_ambiguous():
    with pytest.raises(BrokerError, match="connection reset") as info:
        RobinhoodBroker(ExplodingRH()).get_price("SPY")
    assert not info.value.ambiguous


@pytest.mark.parametrize("action", [lambda b: b.buy_dollars("SPY", 5), lambda b: b.sell_quantity("SPY", 1)])
def test_rh_crash_while_ordering_is_ambiguous(action):
    with pytest.raises(BrokerError) as info:
        action(RobinhoodBroker(ExplodingRH()))
    assert info.value.ambiguous


def test_rh_bad_buying_power_raises():
    with pytest.raises(BrokerError):
        RobinhoodBroker(FakeRH(load_account_profile=None)).get_buying_power()


def test_rh_market_hours():
    b = RobinhoodBroker(FakeRH())
    assert b.is_market_open(datetime(2026, 10, 28, 19, 45, tzinfo=UTC))
    assert not b.is_market_open(datetime(2026, 10, 28, 20, 0, tzinfo=UTC))
    assert not RobinhoodBroker(FakeRH(get_market_today_hours={"is_open": False})).is_market_open(MARKET_OPEN_UTC)
    assert not RobinhoodBroker(FakeRH(get_market_today_hours=None)).is_market_open(MARKET_OPEN_UTC)


def test_rh_orders_use_gfd_regular_hours_and_safe_rounding():
    rh = FakeRH()
    b = RobinhoodBroker(rh)
    assert b.buy_dollars("SPY", 25.129) == "buy-1"
    assert b.sell_quantity("SPY", 0.1234567) == "sell-1"
    (n1, a1, k1), (n2, a2, k2) = rh.calls
    assert a1 == ("SPY", 25.13) and k1 == {"timeInForce": "gfd", "extendedHours": False}
    assert a2 == ("SPY", 0.123456)  # floored, never rounded up
    assert k2 == {"timeInForce": "gfd", "extendedHours": False}


def test_rh_order_rejections():
    with pytest.raises(BrokerError, match="at least"):
        RobinhoodBroker(FakeRH()).buy_dollars("SPY", 0.5)
    with pytest.raises(BrokerError, match="rounds to zero"):
        RobinhoodBroker(FakeRH()).sell_quantity("SPY", 1e-9)
    with pytest.raises(BrokerError, match="Not enough buying power") as info:
        rh = FakeRH(order_buy_fractional_by_price={"detail": "Not enough buying power."})
        RobinhoodBroker(rh).buy_dollars("SPY", 5)
    assert not info.value.ambiguous
    with pytest.raises(BrokerError, match="rejected") as info:
        RobinhoodBroker(FakeRH(order_sell_fractional_by_quantity=["weird"])).sell_quantity("SPY", 1)
    assert not info.value.ambiguous


def test_rh_no_response_is_ambiguous():
    with pytest.raises(BrokerError) as info:
        RobinhoodBroker(FakeRH(order_buy_fractional_by_price=None)).buy_dollars("SPY", 5)
    assert info.value.ambiguous


def test_rh_order_status_mapping():
    st = RobinhoodBroker(FakeRH()).get_order("x")
    assert st == OrderStatus("x", "filled", 0.1, 500.5) and st.is_final
    queued = RobinhoodBroker(
        FakeRH(get_stock_order_info={"state": "queued", "cumulative_quantity": None, "average_price": None})
    ).get_order("y")
    assert queued.filled_quantity == 0 and queued.average_price is None and not queued.is_final
    with pytest.raises(BrokerError):
        RobinhoodBroker(FakeRH(get_stock_order_info={"detail": "Not found."})).get_order("z")


def test_rh_flags():
    b = RobinhoodBroker(SimpleNamespace())
    assert b.is_live and b.name == "robinhood"
