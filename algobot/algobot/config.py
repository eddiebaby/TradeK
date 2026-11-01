"""Load and validate ``algobot.toml``."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .risk import RiskLimits
from .strategies import build_strategy

BROKERS = ("paper", "robinhood")
DATA_SOURCES = ("yahoo", "csv")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class BotConfig:
    path: Path
    broker: str
    symbols: tuple[str, ...]
    budget: float
    state_dir: Path
    strategy_name: str
    strategy_params: dict[str, Any] = field(default_factory=dict)
    data_source: str = "yahoo"
    csv_dir: Path | None = None
    history_period: str = "2y"
    rebalance_threshold: float = 5.0
    risk: dict[str, Any] = field(default_factory=dict)
    paper_starting_cash: float = 200.0
    paper_slippage_bps: float = 5.0

    @property
    def kill_switch(self) -> Path:
        return self.state_dir / "STOP"

    def risk_limits(self) -> RiskLimits:
        try:
            return RiskLimits(
                allowed_symbols=frozenset(self.symbols),
                kill_switch_path=self.kill_switch,
                **self.risk,
            )
        except TypeError as exc:
            raise ConfigError(f"[risk]: {exc}") from None

    def strategy(self):
        return build_strategy(self.strategy_name, self.strategy_params)


def load_config(path: Path) -> BotConfig:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"{path} not found — copy algobot.example.toml to {path.name} and edit it")
    try:
        raw = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from None
    return parse_config(raw, path)


def parse_config(raw: dict[str, Any], path: Path) -> BotConfig:
    base = path.parent.resolve()
    bot = dict(raw.get("bot", {}))
    paper = dict(raw.get("paper", {}))
    strategy = dict(raw.get("strategy", {}))
    strategy_name = strategy.get("name", "sma")
    loose = sorted(k for k, v in strategy.items() if k != "name" and not isinstance(v, dict))
    if loose:
        raise ConfigError(
            f"put strategy parameters under [strategy.{strategy_name}], not [strategy]: {loose}"
        )
    strategy_params = strategy.get(strategy_name, {})

    broker = bot.get("broker", "paper")
    if broker not in BROKERS:
        raise ConfigError(f"bot.broker must be one of {BROKERS}, got {broker!r}")
    symbols = tuple(str(s).strip().upper() for s in bot.get("symbols", []))
    if not symbols or any(not s.replace(".", "").isalnum() for s in symbols):
        raise ConfigError("bot.symbols must be a non-empty list of tickers, e.g. ['SPY']")
    budget = float(bot.get("budget", 0))
    if budget <= 0:
        raise ConfigError("bot.budget must be a positive dollar amount")
    data_source = bot.get("data", "yahoo")
    if data_source not in DATA_SOURCES:
        raise ConfigError(f"bot.data must be one of {DATA_SOURCES}")
    csv_dir = base / bot["csv_dir"] if "csv_dir" in bot else None
    if data_source == "csv" and csv_dir is None:
        raise ConfigError("bot.csv_dir is required when bot.data = 'csv'")

    cfg = BotConfig(
        path=path,
        broker=broker,
        symbols=symbols,
        budget=budget,
        state_dir=base / bot.get("state_dir", "state"),
        strategy_name=strategy_name,
        strategy_params=dict(strategy_params),
        data_source=data_source,
        csv_dir=csv_dir,
        history_period=str(bot.get("history_period", "2y")),
        rebalance_threshold=float(bot.get("rebalance_threshold", 5.0)),
        risk=dict(raw.get("risk", {})),
        paper_starting_cash=float(paper.get("starting_cash", 200.0)),
        paper_slippage_bps=float(paper.get("slippage_bps", 5.0)),
    )
    if budget > cfg.paper_starting_cash and broker == "paper":
        raise ConfigError("bot.budget cannot exceed paper.starting_cash")
    cfg.strategy()  # validate strategy name/params early
    cfg.risk_limits()  # validate risk section early
    return cfg
