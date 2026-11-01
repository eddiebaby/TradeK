"""Robinhood adapter built on the community ``robin_stocks`` library.

IMPORTANT: Robinhood has no official public API for stock trading.
``robin_stocks`` talks to the same private endpoints the Robinhood app
uses. It works and is widely used, but it is unofficial: Robinhood can
change those endpoints (breaking this adapter) or restrict accounts that
automate against them. Keep the budget small and watch the app.

Credentials come from environment variables, never from config files:

    RH_USERNAME      your Robinhood login email
    RH_PASSWORD      your Robinhood password
    RH_TOTP_SECRET   (optional) the authenticator-app secret, so MFA codes
                     can be generated without you typing them

robin_stocks caches the session token in ``~/.tokens/robinhood.pickle``
so you normally only approve a login once a day or so.
"""

from __future__ import annotations

import math
import os
from datetime import datetime
from typing import Any

from .base import BrokerError, OrderStatus

NYSE_MIC = "XNYS"


def _load_client() -> Any:  # pragma: no cover - exercised only with the real library
    try:
        import robin_stocks.robinhood as rh
    except ImportError as exc:
        raise BrokerError("robin_stocks is not installed: pip install robin_stocks pyotp") from exc
    return rh


def _floor6(x: float) -> float:
    """Round *down* to 6 decimals so we never try to sell more than we own."""
    return math.floor(x * 1e6) / 1e6


def _summarize(response: Any) -> str:
    if isinstance(response, dict):
        for key in ("detail", "non_field_errors", "error", "message"):
            if response.get(key):
                return str(response[key])
    return repr(response)[:200]


class RobinhoodBroker:
    name = "robinhood"
    is_live = True

    def __init__(self, client: Any | None = None) -> None:
        self._rh = client if client is not None else _load_client()

    def _call(self, method: str, *args: Any, ambiguous: bool = False, **kwargs: Any) -> Any:
        """Call robin_stocks, turning any crash (e.g. a dropped connection) into BrokerError.

        For order submission ``ambiguous=True``: if something blew up we
        cannot know whether Robinhood received the order.
        """
        try:
            return getattr(self._rh, method)(*args, **kwargs)
        except Exception as exc:
            raise BrokerError(f"{method} failed: {exc}", ambiguous=ambiguous) from exc

    # -- session -------------------------------------------------------------
    def login(
        self,
        username: str | None = None,
        password: str | None = None,
        totp_secret: str | None = None,
    ) -> None:
        username = username or os.environ.get("RH_USERNAME")
        password = password or os.environ.get("RH_PASSWORD")
        totp_secret = totp_secret or os.environ.get("RH_TOTP_SECRET")
        if not username or not password:
            raise BrokerError("set RH_USERNAME and RH_PASSWORD environment variables")
        mfa_code = None
        if totp_secret:
            import pyotp

            mfa_code = pyotp.TOTP(totp_secret).now()
        result = self._call(
            "login", username=username, password=password, mfa_code=mfa_code, store_session=True
        )
        if not result or "access_token" not in result:
            raise BrokerError("Robinhood login failed (check credentials / approve in the app)")

    # -- account ---------------------------------------------------------------
    def get_price(self, symbol: str) -> float:
        values = self._call("get_latest_price", symbol, includeExtendedHours=False)
        try:
            price = float(values[0])
        except (TypeError, ValueError, IndexError):
            raise BrokerError(f"no price returned for {symbol}: {values!r}") from None
        if not math.isfinite(price) or price <= 0:
            raise BrokerError(f"invalid price for {symbol}: {price}")
        return price

    def get_positions(self) -> dict[str, float]:
        items = self._call("get_open_stock_positions")
        # robin_stocks signals an HTTP error with None or [None]; a real empty account is [].
        if not isinstance(items, list) or any(not isinstance(i, dict) for i in items):
            raise BrokerError(f"could not load positions: {_summarize(items)}")
        positions: dict[str, float] = {}
        for item in items:
            symbol = item.get("symbol") or self._call("get_symbol_by_url", item["instrument"])
            qty = float(item.get("quantity") or 0)
            if symbol and qty > 0:
                positions[symbol] = positions.get(symbol, 0.0) + qty
        return positions

    def get_buying_power(self) -> float:
        value = self._call("load_account_profile", info="buying_power")
        try:
            return float(value)
        except (TypeError, ValueError):
            raise BrokerError(f"could not read buying power: {value!r}") from None

    def is_market_open(self, now: datetime) -> bool:
        hours = self._call("get_market_today_hours", NYSE_MIC)
        if not hours or not hours.get("is_open"):
            return False
        opens = datetime.fromisoformat(hours["opens_at"].replace("Z", "+00:00"))
        closes = datetime.fromisoformat(hours["closes_at"].replace("Z", "+00:00"))
        return opens <= now < closes

    # -- orders ------------------------------------------------------------------
    def buy_dollars(self, symbol: str, dollars: float) -> str:
        dollars = round(dollars, 2)
        if dollars < 1:
            raise BrokerError("Robinhood fractional orders must be at least $1.00")
        response = self._call(
            "order_buy_fractional_by_price", symbol, dollars,
            timeInForce="gfd", extendedHours=False, ambiguous=True,
        )
        return self._order_id(response, f"buy ${dollars} {symbol}")

    def sell_quantity(self, symbol: str, quantity: float) -> str:
        quantity = _floor6(quantity)
        if quantity <= 0:
            raise BrokerError("sell quantity rounds to zero")
        response = self._call(
            "order_sell_fractional_by_quantity", symbol, quantity,
            timeInForce="gfd", extendedHours=False, ambiguous=True,
        )
        return self._order_id(response, f"sell {quantity} {symbol}")

    def get_order(self, order_id: str) -> OrderStatus:
        info = self._call("get_stock_order_info", order_id)
        if not isinstance(info, dict) or "state" not in info:
            raise BrokerError(f"could not fetch order {order_id}: {_summarize(info)}")
        avg = info.get("average_price")
        return OrderStatus(
            order_id=order_id,
            state=str(info["state"]),
            filled_quantity=float(info.get("cumulative_quantity") or 0),
            average_price=float(avg) if avg else None,
        )

    @staticmethod
    def _order_id(response: Any, what: str) -> str:
        if response is None:
            # robin_stocks returns None on timeouts/network errors — the order
            # may or may not have reached Robinhood.
            raise BrokerError(f"no response placing {what}; check the app", ambiguous=True)
        if isinstance(response, dict) and response.get("id"):
            return str(response["id"])
        raise BrokerError(f"Robinhood rejected {what}: {_summarize(response)}")
