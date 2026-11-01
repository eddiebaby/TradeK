"""Daily price history sources."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd


class DataError(Exception):
    pass


class PriceData(Protocol):
    def history(self, symbol: str) -> pd.Series:
        """Daily closing prices with a DatetimeIndex, oldest first."""
        ...


def _normalize(close: pd.Series) -> pd.Series:
    close = close.dropna().astype(float)
    index = pd.DatetimeIndex(close.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    close.index = index.normalize()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    close.name = "close"
    return close


def load_close_csv(path: Path) -> pd.Series:
    """Read a CSV with a date column and a close column (Yahoo export format works)."""
    df = pd.read_csv(path)
    lower = {c.lower().strip(): c for c in df.columns}
    date_col = next((lower[c] for c in ("date", "datetime", "timestamp", "time") if c in lower), None)
    close_col = next((lower[c] for c in ("adj close", "adj_close", "close") if c in lower), None)
    if date_col is None or close_col is None:
        raise DataError(f"{path}: need a date column and a close column, got {list(df.columns)}")
    close = pd.Series(
        pd.to_numeric(df[close_col], errors="coerce").to_numpy(),
        index=pd.to_datetime(df[date_col], utc=True).dt.tz_localize(None),
    )
    close = _normalize(close)
    if close.empty:
        raise DataError(f"{path}: no usable prices")
    return close


class CsvData:
    """Reads ``<directory>/<SYMBOL>.csv`` — handy offline and for tests."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def history(self, symbol: str) -> pd.Series:
        path = self.directory / f"{symbol}.csv"
        if not path.exists():
            raise DataError(f"missing price file {path}")
        return load_close_csv(path)


class YahooData:
    """Free daily data from Yahoo Finance via ``yfinance`` (split/dividend adjusted)."""

    def __init__(self, period: str = "2y") -> None:
        self.period = period
        self._cache: dict[str, pd.Series] = {}

    def history(self, symbol: str) -> pd.Series:  # pragma: no cover - needs network
        if symbol not in self._cache:
            import yfinance as yf

            try:
                df = yf.Ticker(symbol).history(period=self.period, interval="1d", auto_adjust=True)
            except Exception as exc:
                raise DataError(f"could not download {symbol} from Yahoo: {exc}") from exc
            if df is None or df.empty or "Close" not in df:
                raise DataError(f"Yahoo returned no data for {symbol}")
            self._cache[symbol] = _normalize(df["Close"])
        return self._cache[symbol]


def latest_close(data: PriceData, symbol: str) -> float:
    close = data.history(symbol)
    if close.empty:
        raise DataError(f"no prices for {symbol}")
    return float(close.iloc[-1])
