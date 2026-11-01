from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from algobot import cli
from algobot.config import ConfigError, load_config, parse_config
from algobot.data import CsvData, DataError, latest_close, load_close_csv
from algobot.ledger import LedgerStore
from algobot.market_hours import is_regular_session, trading_date

from .conftest import MARKET_OPEN_UTC, trend

EXAMPLE = Path(__file__).resolve().parents[1] / "algobot.example.toml"


def write_prices(directory: Path, symbol: str, close: pd.Series) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{symbol}.csv"
    pd.DataFrame({"Date": close.index.strftime("%Y-%m-%d"), "Close": close.values}).to_csv(path, index=False)
    return path


def write_config(tmp_path: Path, broker="paper", extra="") -> Path:
    cfg = tmp_path / "algobot.toml"
    cfg.write_text(
        f"""
[bot]
broker = "{broker}"
symbols = ["spy"]
budget = 150
data = "csv"
csv_dir = "prices"
{extra}
[strategy]
name = "sma"

[strategy.sma]
fast = 5
slow = 20

[risk]
max_order_dollars = 200
"""
    )
    write_prices(tmp_path / "prices", "SPY", trend(80))
    return cfg


# -- config -----------------------------------------------------------------------
def test_example_config_is_valid():
    cfg = load_config(EXAMPLE)
    assert cfg.broker == "paper" and cfg.symbols == ("SPY",) and cfg.budget == 150
    assert cfg.strategy().name == "sma_20_50"
    limits = cfg.risk_limits()
    assert limits.allowed_symbols == {"SPY"} and limits.kill_switch_path == cfg.state_dir / "STOP"


@pytest.mark.parametrize("name,expected", [("rsi", "rsi_14_30_70"), ("buy_and_hold", "buy_and_hold")])
def test_switching_strategy_is_a_one_line_change(tmp_path, name, expected):
    cfg = tmp_path / "algobot.toml"
    cfg.write_text(EXAMPLE.read_text().replace('name = "sma"', f'name = "{name}"'))
    assert load_config(cfg).strategy().name == expected


def test_paths_resolve_relative_to_config_file(tmp_path):
    cfg = load_config(write_config(tmp_path))
    assert cfg.state_dir == tmp_path.resolve() / "state"
    assert cfg.csv_dir == tmp_path.resolve() / "prices"
    assert cfg.symbols == ("SPY",)  # upper-cased


@pytest.mark.parametrize(
    "raw,match",
    [
        ({"bot": {"broker": "etrade", "symbols": ["SPY"], "budget": 1}}, "broker"),
        ({"bot": {"symbols": [], "budget": 1}}, "symbols"),
        ({"bot": {"symbols": ["SP Y"], "budget": 1}}, "symbols"),
        ({"bot": {"symbols": ["SPY"], "budget": 0}}, "budget"),
        ({"bot": {"symbols": ["SPY"], "budget": 1, "data": "bloomberg"}}, "data"),
        ({"bot": {"symbols": ["SPY"], "budget": 1, "data": "csv"}}, "csv_dir"),
        ({"bot": {"symbols": ["SPY"], "budget": 500}}, "starting_cash"),
        ({"bot": {"symbols": ["SPY"], "budget": 1}, "strategy": {"name": "nope"}}, "unknown strategy"),
        ({"bot": {"symbols": ["SPY"], "budget": 1}, "risk": {"max_leverage": 3}}, r"\[risk\]"),
        ({"bot": {"symbols": ["SPY"], "budget": 1}, "strategy": {"name": "sma", "fast": 5}},
         r"\[strategy.sma\]"),
    ],
)
def test_config_validation(raw, match, tmp_path):
    with pytest.raises(ValueError, match=match):
        parse_config(raw, tmp_path / "x.toml")


def test_config_file_errors(tmp_path):
    with pytest.raises(ConfigError, match="copy algobot.example.toml"):
        load_config(tmp_path / "missing.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text("[bot\n")
    with pytest.raises(ConfigError):
        load_config(bad)


def test_live_budget_may_exceed_paper_cash(tmp_path):
    cfg = parse_config({"bot": {"broker": "robinhood", "symbols": ["SPY"], "budget": 500}}, tmp_path / "x.toml")
    assert cfg.budget == 500


# -- data / clock ------------------------------------------------------------------
def test_csv_loading_handles_yahoo_style_columns(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("Date,Open,Close,Adj Close\n2026-01-03,1,10,9\n2026-01-02,1,11,10\n2026-01-02,1,12,11\n")
    close = load_close_csv(path)
    assert close.tolist() == [11.0, 9.0]  # sorted, de-duplicated, prefers Adj Close
    assert latest_close(CsvData(tmp_path), "x") == 9.0


@pytest.mark.parametrize("content", ["foo,bar\n1,2\n", "Date,Close\n2026-01-02,abc\n"])
def test_csv_errors(tmp_path, content):
    path = tmp_path / "x.csv"
    path.write_text(content)
    with pytest.raises(DataError):
        load_close_csv(path)


def test_csv_data_missing_file(tmp_path):
    with pytest.raises(DataError, match="missing"):
        CsvData(tmp_path).history("SPY")


def test_latest_close_empty():
    with pytest.raises(DataError):
        latest_close(SimpleNamespace(history=lambda s: pd.Series(dtype=float)), "SPY")


def test_market_clock():
    assert is_regular_session(MARKET_OPEN_UTC)
    assert not is_regular_session(datetime(2026, 10, 28, 13, 0, tzinfo=UTC))  # 9:00 NY
    assert not is_regular_session(datetime(2026, 10, 28, 20, 0, tzinfo=UTC))  # 16:00 NY
    # 01:00 UTC on the 29th is still the 28th in New York.
    assert trading_date(datetime(2026, 10, 29, 1, 0, tzinfo=UTC)).day == 28
    with pytest.raises(ValueError):
        is_regular_session(datetime(2026, 10, 28, 12))
    with pytest.raises(ValueError):
        trading_date(datetime(2026, 10, 28, 12))


# -- cli ----------------------------------------------------------------------------
def test_cli_backtest_from_csv(tmp_path, capsys):
    path = write_prices(tmp_path, "SPY", trend(300))
    assert cli.main(["backtest", "SPY", "--csv", str(path), "--param", "fast=5", "--param", "slow=20"]) == 0
    out = capsys.readouterr().out
    assert "buy_and_hold" in out and "sma_5_20" in out and "curve-fitting" in out


def test_cli_backtest_all(tmp_path, capsys):
    path = write_prices(tmp_path, "SPY", trend(300))
    assert cli.main(["backtest", "SPY", "--csv", str(path), "--all"]) == 0
    out = capsys.readouterr().out
    assert out.count("buy_and_hold") == 1 and "rsi_14_30_70" in out


def test_cli_bad_param_is_a_clean_error(tmp_path, capsys):
    path = write_prices(tmp_path, "SPY", trend(300))
    assert cli.main(["backtest", "SPY", "--csv", str(path), "--param", "oops"]) == 2
    assert "key=value" in capsys.readouterr().err


def test_parse_value():
    assert cli._parse_value("3") == 3 and cli._parse_value("2.5") == 2.5 and cli._parse_value("x") == "x"


def test_cli_paper_run_status_and_reset(tmp_path, capsys):
    cfg = write_config(tmp_path)
    assert cli.main(["run", "--config", str(cfg), "--ignore-market-hours"]) == 0
    out = capsys.readouterr().out
    assert "submitted buy SPY" in out
    assert (tmp_path / "state" / "paper_account.json").exists()

    assert cli.main(["status", "--config", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "SPY:" in out and "approx. bot equity" in out

    store = LedgerStore(tmp_path / "state" / "ledger.json", 150)
    led = store.load()
    led.halt("test halt")
    store.save(led)
    (tmp_path / "state" / "STOP").touch()
    cli.main(["status", "--config", str(cfg)])
    out = capsys.readouterr().out
    assert "HALTED: test halt" in out and "KILL SWITCH ON" in out

    assert cli.main(["reset-halt", "--config", str(cfg)]) == 0
    assert not store.load().halted
    assert cli.main(["reset-halt", "--config", str(cfg)]) == 0
    assert "not halted" in capsys.readouterr().out


def test_cli_status_survives_missing_prices(tmp_path, capsys):
    cfg = write_config(tmp_path)
    store = LedgerStore(tmp_path / "state" / "ledger.json", 150)
    led = store.load()
    led.positions["QQQ"] = 1.0
    store.save(led)
    assert cli.main(["status", "--config", str(cfg)]) == 0
    assert "price unavailable" in capsys.readouterr().out


def test_cli_paper_run_reports_market_closed(tmp_path, capsys, monkeypatch):
    cfg = write_config(tmp_path)
    monkeypatch.setattr("algobot.brokers.paper.is_regular_session", lambda now: False)
    assert cli.main(["run", "--config", str(cfg), "--dry-run"]) == 0
    assert "market is closed" in capsys.readouterr().out


def _args(cfg, **kw):
    base = dict(config=str(cfg), live=False, dry_run=False, yes=False, ignore_market_hours=False)
    base.update(kw)
    return SimpleNamespace(**base)


class StubEngine:
    def __init__(self):
        self.calls = []

    def factory(self, cfg, ignore_market_hours=False):
        self.cfg = cfg
        return self

    def run_once(self, dry_run=False, confirm=None):
        self.calls.append((dry_run, confirm))
        return SimpleNamespace(summary=lambda: "ok", errors=[])


def test_live_requires_explicit_flag(tmp_path, capsys):
    cfg = write_config(tmp_path, broker="robinhood")
    stub = StubEngine()
    assert cli.cmd_run(_args(cfg), engine_factory=stub.factory) == 2
    assert "Refusing to trade real money" in capsys.readouterr().out
    assert stub.calls == []


def test_live_flag_rejected_for_paper_and_market_hours_override_rejected_for_live(tmp_path, capsys):
    paper_cfg = write_config(tmp_path)
    assert cli.cmd_run(_args(paper_cfg, live=True), engine_factory=StubEngine().factory) == 2
    live_cfg = write_config(tmp_path, broker="robinhood")
    assert cli.cmd_run(_args(live_cfg, live=True, ignore_market_hours=True),
                       engine_factory=StubEngine().factory) == 2


def test_unattended_live_run_needs_yes(tmp_path, capsys):
    cfg = write_config(tmp_path, broker="robinhood")
    stub = StubEngine()
    assert cli.cmd_run(_args(cfg, live=True), engine_factory=stub.factory, stdin_isatty=False) == 2
    assert "--yes" in capsys.readouterr().out
    assert cli.cmd_run(_args(cfg, live=True, yes=True), engine_factory=stub.factory, stdin_isatty=False) == 0
    assert stub.calls == [(False, None)]


def test_interactive_live_run_asks_for_confirmation(tmp_path):
    cfg = write_config(tmp_path, broker="robinhood")
    stub = StubEngine()
    assert cli.cmd_run(_args(cfg, live=True), engine_factory=stub.factory, stdin_isatty=True) == 0
    _, confirm = stub.calls[0]
    assert confirm is not None


def test_live_dry_run_needs_no_confirmation(tmp_path):
    cfg = write_config(tmp_path, broker="robinhood")
    stub = StubEngine()
    assert cli.cmd_run(_args(cfg, live=True, dry_run=True), engine_factory=stub.factory, stdin_isatty=False) == 0
    assert stub.calls == [(True, None)]


def test_run_exit_code_is_1_when_errors(tmp_path):
    cfg = write_config(tmp_path)

    class Failing(StubEngine):
        def run_once(self, dry_run=False, confirm=None):
            return SimpleNamespace(summary=lambda: "bad", errors=["x"])

    assert cli.cmd_run(_args(cfg), engine_factory=Failing().factory) == 1


def test_interactive_confirm_requires_exact_yes(capsys):
    from algobot.brokers.base import Side
    from algobot.planner import OrderRequest

    orders = [OrderRequest("SPY", Side.BUY, dollars=10, reason="r")]
    assert cli.interactive_confirm(lambda prompt: "YES")(orders, {"SPY": 500})
    assert not cli.interactive_confirm(lambda prompt: "yes")(orders, {"SPY": 500})
    assert "REAL orders" in capsys.readouterr().out


def test_build_engine_wires_paper_broker(tmp_path):
    cfg = load_config(write_config(tmp_path))
    engine = cli.build_engine(cfg, clock=lambda: MARKET_OPEN_UTC)
    assert engine.broker.name == "paper" and engine.symbols == ["SPY"]
    report = engine.run_once()
    assert report.submitted


def test_broker_failure_is_a_clean_error(tmp_path, capsys, monkeypatch):
    from algobot.brokers.base import BrokerError

    def no_login(cfg, data):
        raise BrokerError("Robinhood login failed")

    monkeypatch.setattr(cli, "build_broker", no_login)
    assert cli.main(["run", "--config", str(write_config(tmp_path))]) == 2
    assert "error: Robinhood login failed" in capsys.readouterr().err
