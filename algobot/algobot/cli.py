"""Command line interface: ``python -m algobot <command>``.

    backtest SYMBOL     test a strategy on historical data
    run                 do one trading run (paper by default)
    status              show the bot's ledger
    reset-halt          clear a halt after you've checked things
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .backtest import compare, format_results
from .brokers.base import Broker, BrokerError
from .brokers.paper import PaperBroker
from .config import BotConfig, ConfigError, load_config
from .data import CsvData, DataError, PriceData, YahooData, latest_close, load_close_csv
from .engine import Engine
from .journal import Journal
from .ledger import LedgerStore
from .planner import OrderRequest
from .risk import RiskManager
from .strategies import STRATEGIES, build_strategy

DEFAULT_CONFIG = Path("algobot.toml")


def _parse_value(text: str) -> Any:
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _parse_params(pairs: list[str]) -> dict[str, Any]:
    params = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"--param expects key=value, got {pair!r}")
        params[key.strip()] = _parse_value(value.strip())
    return params


# -- backtest -------------------------------------------------------------------
def cmd_backtest(args: argparse.Namespace) -> int:
    if args.csv:
        close = load_close_csv(Path(args.csv))
    else:
        close = YahooData(period=args.period).history(args.symbol.upper())  # pragma: no cover
    print(f"{args.symbol.upper()}: {len(close)} daily bars, {close.index[0]:%Y-%m-%d} -> {close.index[-1]:%Y-%m-%d}")
    print(f"starting cash ${args.cash:.2f}, trading cost {args.cost_bps} bps per trade\n")
    if args.all:
        strategies = [build_strategy(name) for name in STRATEGIES]
    else:
        strategies = [build_strategy(args.strategy, _parse_params(args.param))]
    results = compare(close, strategies, args.cash, args.cost_bps)  # always includes buy & hold
    print(format_results(results))
    print("\nPast performance says little about the future. Beware of tuning parameters until the")
    print("backtest looks great — that is curve-fitting, and it rarely survives live trading.")
    return 0


# -- engine wiring --------------------------------------------------------------
def build_data(cfg: BotConfig) -> PriceData:
    if cfg.data_source == "csv":
        return CsvData(cfg.csv_dir)
    return YahooData(period=cfg.history_period)  # pragma: no cover


def build_broker(cfg: BotConfig, data: PriceData) -> Broker:
    if cfg.broker == "paper":
        return PaperBroker(
            cfg.state_dir / "paper_account.json",
            price_source=lambda s: latest_close(data, s),
            starting_cash=cfg.paper_starting_cash,
            slippage_bps=cfg.paper_slippage_bps,
        )
    from .brokers.robinhood import RobinhoodBroker  # pragma: no cover

    broker = RobinhoodBroker()  # pragma: no cover
    broker.login()  # pragma: no cover
    return broker  # pragma: no cover


def build_engine(
    cfg: BotConfig,
    *,
    ignore_market_hours: bool = False,
    data: PriceData | None = None,
    broker: Broker | None = None,
    clock=None,
) -> Engine:
    data = data or build_data(cfg)
    broker = broker or build_broker(cfg, data)
    return Engine(
        broker=broker,
        data=data,
        strategy=cfg.strategy(),
        symbols=cfg.symbols,
        store=LedgerStore(cfg.state_dir / "ledger.json", cfg.budget),
        risk=RiskManager(cfg.risk_limits()),
        journal=Journal(cfg.state_dir / "journal.csv", broker.name),
        rebalance_threshold=cfg.rebalance_threshold,
        ignore_market_hours=ignore_market_hours,
        clock=clock,
    )


def interactive_confirm(
    input_fn: Callable[[str], str] = input,
) -> Callable[[list[OrderRequest], dict[str, float]], bool]:
    def confirm(orders: list[OrderRequest], prices: dict[str, float]) -> bool:
        print("\nAbout to place REAL orders on Robinhood:")
        for o in orders:
            print(f"  {o.describe(prices.get(o.symbol))}")
        return input_fn("Type YES to send them: ").strip() == "YES"

    return confirm


def cmd_run(args: argparse.Namespace, engine_factory=build_engine, stdin_isatty=None) -> int:
    cfg = load_config(Path(args.config))
    live = cfg.broker == "robinhood"
    if live and not args.live:
        print("Refusing to trade real money: config says broker = 'robinhood' but --live was not given.")
        return 2
    if args.live and not live:
        print("--live only applies when bot.broker = 'robinhood' in the config.")
        return 2
    if live and args.ignore_market_hours:
        print("--ignore-market-hours is only allowed for paper trading.")
        return 2
    confirm = None
    if live and not args.dry_run and not args.yes:
        isatty = sys.stdin.isatty() if stdin_isatty is None else stdin_isatty
        if not isatty:
            print("Live run without a terminal: pass --yes to allow unattended (scheduled) trading.")
            return 2
        confirm = interactive_confirm()

    engine = engine_factory(cfg, ignore_market_hours=args.ignore_market_hours)
    report = engine.run_once(dry_run=args.dry_run, confirm=confirm)
    print(report.summary())
    return 1 if report.errors else 0


# -- status / reset -------------------------------------------------------------
def cmd_status(args: argparse.Namespace) -> int:
    cfg = load_config(Path(args.config))
    ledger = LedgerStore(cfg.state_dir / "ledger.json", cfg.budget).load()
    print(f"broker: {cfg.broker}   strategy: {cfg.strategy().name}   symbols: {', '.join(cfg.symbols)}")
    print(f"virtual cash: ${ledger.cash:.2f}   peak equity: ${ledger.peak_equity:.2f}")
    data = build_data(cfg)
    value = ledger.cash
    for symbol, qty in sorted(ledger.positions.items()):
        try:
            price = latest_close(data, symbol)
            value += qty * price
            print(f"  {symbol}: {qty:.6f} sh @ ~${price:.2f} = ${qty * price:.2f}")
        except Exception as exc:  # status should still print offline
            print(f"  {symbol}: {qty:.6f} sh (price unavailable: {exc})")
            value = float("nan")
    print(f"approx. bot equity: ${value:.2f}  (budget ${cfg.budget:.2f})")
    for p in ledger.pending:
        print(f"  pending {p.side} {p.symbol} order {p.order_id} since {p.submitted_at}")
    if ledger.halted:
        print(f"HALTED: {ledger.halt_reason}\n  -> fix the cause, then run: python -m algobot reset-halt")
    if cfg.kill_switch.exists():
        print(f"KILL SWITCH ON: delete {cfg.kill_switch} to resume")
    return 0


def cmd_reset_halt(args: argparse.Namespace) -> int:
    cfg = load_config(Path(args.config))
    store = LedgerStore(cfg.state_dir / "ledger.json", cfg.budget)
    ledger = store.load()
    if not ledger.halted:
        print("not halted")
        return 0
    print(f"clearing halt: {ledger.halt_reason}")
    ledger.clear_halt()
    store.save(ledger)
    return 0


# -- entry point ------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="algobot", description="Small, safety-first trading bot.")
    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", help="test a strategy on historical data")
    bt.add_argument("symbol")
    bt.add_argument("--strategy", default="sma", choices=sorted(STRATEGIES))
    bt.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                    help="strategy parameter, e.g. --param fast=10 --param slow=40")
    bt.add_argument("--all", action="store_true", help="compare every built-in strategy")
    bt.add_argument("--period", default="10y", help="Yahoo history period (default 10y)")
    bt.add_argument("--csv", help="read prices from a CSV file instead of Yahoo")
    bt.add_argument("--cash", type=float, default=200.0)
    bt.add_argument("--cost-bps", type=float, default=5.0, help="spread+slippage per trade")
    bt.set_defaults(func=cmd_backtest)

    run = sub.add_parser("run", help="do one trading run")
    run.add_argument("--config", default=str(DEFAULT_CONFIG))
    run.add_argument("--live", action="store_true", help="required to trade real money")
    run.add_argument("--dry-run", action="store_true", help="show orders without sending them")
    run.add_argument("--yes", action="store_true", help="skip the confirmation prompt (for cron)")
    run.add_argument("--ignore-market-hours", action="store_true", help="paper only: trade anytime")
    run.set_defaults(func=cmd_run)

    st = sub.add_parser("status", help="show the bot's ledger")
    st.add_argument("--config", default=str(DEFAULT_CONFIG))
    st.set_defaults(func=cmd_status)

    rh = sub.add_parser("reset-halt", help="clear a trading halt")
    rh.add_argument("--config", default=str(DEFAULT_CONFIG))
    rh.set_defaults(func=cmd_reset_halt)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (BrokerError, ConfigError, DataError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
