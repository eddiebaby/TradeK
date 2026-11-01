"""US equity market clock helpers."""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")
OPEN, CLOSE = time(9, 30), time(16, 0)


def is_regular_session(now: datetime) -> bool:
    """True during 9:30-16:00 New York time on weekdays.

    Exchange holidays are *not* known here — the Robinhood broker asks
    Robinhood for the real calendar. This clock is only used for paper
    trading, where trading on a holiday is harmless.
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    local = now.astimezone(NEW_YORK)
    return local.weekday() < 5 and OPEN <= local.time() < CLOSE


def trading_date(now: datetime) -> date:
    """The calendar date in New York (what 'today' means for the market)."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return now.astimezone(NEW_YORK).date()
