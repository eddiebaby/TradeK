"""Append-only CSV record of everything the bot does with orders."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

FIELDS = ["timestamp", "broker", "event", "symbol", "side", "dollars", "quantity", "price", "order_id", "note"]


class Journal:
    def __init__(self, path: Path, broker: str) -> None:
        self.path = Path(path)
        self.broker = broker

    def record(
        self,
        when: datetime,
        event: str,
        symbol: str,
        side: str,
        *,
        dollars: float | None = None,
        quantity: float | None = None,
        price: float | None = None,
        order_id: str = "",
        note: str = "",
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        with self.path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": when.isoformat(timespec="seconds"),
                    "broker": self.broker,
                    "event": event,
                    "symbol": symbol,
                    "side": side,
                    "dollars": "" if dollars is None else f"{dollars:.2f}",
                    "quantity": "" if quantity is None else f"{quantity:.6f}",
                    "price": "" if price is None else f"{price:.4f}",
                    "order_id": order_id,
                    "note": note,
                }
            )

    def read(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        with self.path.open(newline="") as fh:
            return list(csv.DictReader(fh))
