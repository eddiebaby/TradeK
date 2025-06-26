"""
Schwab API Client

Authenticated client for Charles Schwab API with comprehensive features:
- Automatic OAuth token management
- Rate limiting and retry logic
- Comprehensive API coverage (accounts, quotes, orders, market data)
- Error handling and logging
- Performance monitoring
- Production-ready configuration
"""

import asyncio
import logging
import time
from datetime import datetime
from decimal import Decimal
from typing import Any

import aiohttp
from pydantic import BaseModel, Field

from .token_manager import SchwabTokenManager

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter for API requests."""

    def __init__(self, requests_per_minute: int = 120):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute
        """
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0.0
        self.request_count = 0
        self.window_start = time.time()

    async def acquire(self) -> None:
        """Acquire permission to make a request."""
        current_time = time.time()

        # Reset window if needed
        if current_time - self.window_start >= 60.0:
            self.window_start = current_time
            self.request_count = 0

        # Check if we're at the limit
        if self.request_count >= self.requests_per_minute:
            wait_time = 60.0 - (current_time - self.window_start)
            if wait_time > 0:
                logger.warning(f"Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                self.window_start = time.time()
                self.request_count = 0

        # Enforce minimum interval between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            await asyncio.sleep(wait_time)

        self.last_request_time = time.time()
        self.request_count += 1


class APIMetrics:
    """API performance and usage metrics."""

    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.rate_limit_hits = 0
        self.average_response_time = 0.0
        self.last_request_time: float | None = None
        self.response_times: list[float] = []
        self.max_response_times = 100  # Keep last 100 response times

    def record_request(self, success: bool, response_time: float) -> None:
        """Record request metrics."""
        self.total_requests += 1
        self.last_request_time = time.time()

        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        # Update response times
        self.response_times.append(response_time)
        if len(self.response_times) > self.max_response_times:
            self.response_times.pop(0)

        # Calculate average
        if self.response_times:
            self.average_response_time = sum(self.response_times) / len(
                self.response_times
            )

    def record_rate_limit(self) -> None:
        """Record rate limit hit."""
        self.rate_limit_hits += 1

    def get_stats(self) -> dict[str, Any]:
        """Get current metrics stats."""
        success_rate = 0.0
        if self.total_requests > 0:
            success_rate = (self.successful_requests / self.total_requests) * 100

        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate_percent": success_rate,
            "rate_limit_hits": self.rate_limit_hits,
            "average_response_time_ms": self.average_response_time * 1000,
            "last_request_time": self.last_request_time,
        }


class AccountInfo(BaseModel):
    """Account information model."""

    account_id: str
    account_number: str
    type: str
    status: str
    positions: list[dict[str, Any]] | None = None
    balances: dict[str, Any] | None = None


class Quote(BaseModel):
    """Stock quote model."""

    symbol: str
    bid_price: Decimal | None = None
    ask_price: Decimal | None = None
    last_price: Decimal | None = None
    bid_size: int | None = None
    ask_size: int | None = None
    volume: int | None = None
    change: Decimal | None = None
    change_percent: float | None = None
    timestamp: datetime = Field(default_factory=datetime.now)


class PriceHistory(BaseModel):
    """Price history model."""

    symbol: str
    candles: list[dict[str, Any]]
    timeframe: str
    start_date: datetime
    end_date: datetime


class SchwabAPIClient:
    """
    Comprehensive Schwab API client with authentication and rate limiting.

    Features:
    - Automatic OAuth token management
    - Rate limiting to prevent API quota exhaustion
    - Retry logic with exponential backoff
    - Comprehensive error handling
    - Performance metrics and monitoring
    - Full API coverage for trading and market data
    """

    BASE_URL = "https://api.schwabapi.com"
    API_VERSION = "v1"

    def __init__(
        self,
        token_manager: SchwabTokenManager,
        requests_per_minute: int = 120,
        timeout_seconds: int = 30,
        max_retries: int = 3,
    ):
        """
        Initialize Schwab API client.

        Args:
            token_manager: Configured token manager
            requests_per_minute: API rate limit
            timeout_seconds: Request timeout
            max_retries: Maximum retry attempts
        """
        self.token_manager = token_manager
        self.rate_limiter = RateLimiter(requests_per_minute)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

        # HTTP session
        self._session: aiohttp.ClientSession | None = None

        # Metrics
        self.metrics = APIMetrics()

        # Error tracking
        self.consecutive_errors = 0
        self.last_error_time: float | None = None

        logger.info("Schwab API client initialized")

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def _ensure_session(self) -> None:
        """Ensure HTTP session is available."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            connector = aiohttp.TCPConnector(ssl=True, limit=100, ttl_dns_cache=300)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        authenticated: bool = True,
    ) -> dict[str, Any]:
        """
        Make authenticated API request with retries and rate limiting.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Query parameters
            data: Request body data
            authenticated: Whether to include auth headers

        Returns:
            API response data
        """
        await self._ensure_session()

        # Rate limiting
        await self.rate_limiter.acquire()

        # Build URL
        url = f"{self.BASE_URL}/{self.API_VERSION}/{endpoint.lstrip('/')}"

        # Prepare headers
        headers = {}
        if authenticated:
            token = await self.token_manager.get_valid_token()
            headers["Authorization"] = f"{token.token_type} {token.access_token}"

        # Retry logic
        for attempt in range(self.max_retries + 1):
            start_time = time.time()

            try:
                # Make request
                async with self._session.request(
                    method=method, url=url, params=params, json=data, headers=headers
                ) as response:
                    response_time = time.time() - start_time

                    # Handle rate limiting
                    if response.status == 429:
                        self.metrics.record_rate_limit()
                        if attempt < self.max_retries:
                            retry_after = int(response.headers.get("Retry-After", "60"))
                            logger.warning(
                                f"Rate limited, waiting {retry_after}s before retry"
                            )
                            await asyncio.sleep(retry_after)
                            continue
                        else:
                            raise aiohttp.ClientResponseError(
                                request_info=response.request_info,
                                history=response.history,
                                status=response.status,
                                message="Rate limit exceeded",
                            )

                    # Handle other HTTP errors
                    if response.status >= 400:
                        error_text = await response.text()
                        logger.error(f"API error {response.status}: {error_text}")

                        if response.status >= 500 and attempt < self.max_retries:
                            # Server error - retry with backoff
                            wait_time = (2**attempt) + 1
                            logger.warning(f"Server error, retrying in {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            raise aiohttp.ClientResponseError(
                                request_info=response.request_info,
                                history=response.history,
                                status=response.status,
                                message=error_text,
                            )

                    # Parse response
                    if response.content_type == "application/json":
                        result = await response.json()
                    else:
                        result = {"data": await response.text()}

                    # Record success
                    self.metrics.record_request(True, response_time)
                    self.consecutive_errors = 0

                    return result

            except TimeoutError:
                response_time = time.time() - start_time
                self.metrics.record_request(False, response_time)

                if attempt < self.max_retries:
                    wait_time = (2**attempt) + 1
                    logger.warning(f"Request timeout, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise

            except Exception as e:
                response_time = time.time() - start_time
                self.metrics.record_request(False, response_time)
                self.consecutive_errors += 1
                self.last_error_time = time.time()

                if attempt < self.max_retries:
                    wait_time = (2**attempt) + 1
                    logger.warning(f"Request failed: {e}, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(
                        f"Request failed after {self.max_retries} retries: {e}"
                    )
                    raise

        raise RuntimeError("Maximum retries exceeded")

    # Account API methods
    async def get_accounts(self) -> list[AccountInfo]:
        """Get all account information."""
        response = await self._make_request("GET", "accounts")
        accounts = []

        for account_data in response.get("accounts", []):
            account = AccountInfo(
                account_id=account_data["accountId"],
                account_number=account_data["accountNumber"],
                type=account_data["type"],
                status=account_data["status"],
            )
            accounts.append(account)

        return accounts

    async def get_account_details(self, account_id: str) -> AccountInfo:
        """Get detailed account information."""
        response = await self._make_request("GET", f"accounts/{account_id}")
        account_data = response["account"]

        return AccountInfo(
            account_id=account_data["accountId"],
            account_number=account_data["accountNumber"],
            type=account_data["type"],
            status=account_data["status"],
            positions=account_data.get("positions", []),
            balances=account_data.get("currentBalances", {}),
        )

    async def get_positions(self, account_id: str) -> list[dict[str, Any]]:
        """Get account positions."""
        response = await self._make_request("GET", f"accounts/{account_id}/positions")
        return response.get("positions", [])

    # Market data API methods
    async def get_quote(self, symbol: str) -> Quote:
        """Get quote for a single symbol."""
        quotes = await self.get_quotes([symbol])
        return quotes[0] if quotes else None

    async def get_quotes(self, symbols: list[str]) -> list[Quote]:
        """Get quotes for multiple symbols."""
        if not symbols:
            return []

        # Schwab API accepts comma-separated symbols
        symbol_list = ",".join(symbols)
        response = await self._make_request(
            "GET", "marketdata/quotes", {"symbols": symbol_list}
        )

        quotes = []
        for symbol, quote_data in response.items():
            try:
                quote = Quote(
                    symbol=symbol,
                    bid_price=(
                        Decimal(str(quote_data.get("bidPrice", 0)))
                        if quote_data.get("bidPrice")
                        else None
                    ),
                    ask_price=(
                        Decimal(str(quote_data.get("askPrice", 0)))
                        if quote_data.get("askPrice")
                        else None
                    ),
                    last_price=(
                        Decimal(str(quote_data.get("lastPrice", 0)))
                        if quote_data.get("lastPrice")
                        else None
                    ),
                    bid_size=quote_data.get("bidSize"),
                    ask_size=quote_data.get("askSize"),
                    volume=quote_data.get("totalVolume"),
                    change=(
                        Decimal(str(quote_data.get("netChange", 0)))
                        if quote_data.get("netChange")
                        else None
                    ),
                    change_percent=quote_data.get("netPercentChangeInDouble"),
                )
                quotes.append(quote)
            except Exception as e:
                logger.warning(f"Failed to parse quote for {symbol}: {e}")

        return quotes

    async def get_price_history(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        frequency: str = "daily",
        frequency_type: str = "1",
    ) -> PriceHistory:
        """
        Get price history for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            frequency: Frequency (daily, weekly, monthly)
            frequency_type: Frequency type (1, 5, 10, 15, 30)
        """
        # Convert dates to milliseconds
        start_ms = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)

        params = {
            "startDate": start_ms,
            "endDate": end_ms,
            "frequency": frequency,
            "frequencyType": frequency_type,
        }

        response = await self._make_request(
            "GET", f"marketdata/{symbol}/pricehistory", params
        )

        return PriceHistory(
            symbol=symbol,
            candles=response.get("candles", []),
            timeframe=f"{frequency}_{frequency_type}",
            start_date=start_date,
            end_date=end_date,
        )

    async def get_movers(
        self, index: str = "SPX", direction: str = "up", change: str = "percent"
    ) -> list[dict[str, Any]]:
        """
        Get market movers.

        Args:
            index: Market index (SPX, NDX, etc.)
            direction: up or down
            change: percent or value
        """
        params = {"direction": direction, "change": change}

        response = await self._make_request("GET", f"marketdata/movers/{index}", params)
        return response.get("screener", [])

    # Trading API methods (requires additional permissions)
    async def get_orders(
        self, account_id: str, max_results: int = 3000
    ) -> list[dict[str, Any]]:
        """Get orders for account."""
        params = {"maxResults": max_results}

        response = await self._make_request(
            "GET", f"accounts/{account_id}/orders", params
        )
        return response.get("orders", [])

    async def place_order(
        self, account_id: str, order_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Place a trading order.

        Args:
            account_id: Account ID
            order_data: Order specification

        Returns:
            Order confirmation
        """
        response = await self._make_request(
            "POST", f"accounts/{account_id}/orders", data=order_data
        )
        return response

    async def cancel_order(self, account_id: str, order_id: str) -> bool:
        """Cancel an order."""
        try:
            await self._make_request(
                "DELETE", f"accounts/{account_id}/orders/{order_id}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    # Utility methods
    async def get_api_status(self) -> dict[str, Any]:
        """Get API status and health."""
        status = {
            "connected": self._session is not None and not self._session.closed,
            "token_valid": False,
            "metrics": self.metrics.get_stats(),
            "consecutive_errors": self.consecutive_errors,
            "last_error_time": self.last_error_time,
        }

        try:
            token = await self.token_manager.get_current_token()
            status["token_valid"] = token is not None and not token.is_expired
            if token:
                status["token_expires_in"] = token.time_until_expiry
        except Exception:
            pass

        return status

    async def health_check(self) -> dict[str, Any]:
        """Perform comprehensive health check."""
        health = {"healthy": True, "issues": [], "warnings": []}

        # Check token health
        try:
            token = await self.token_manager.get_valid_token()
            if token.time_until_expiry < 300:  # 5 minutes
                health["warnings"].append("Token expiring soon")
        except Exception as e:
            health["healthy"] = False
            health["issues"].append(f"Token issue: {e}")

        # Check API connectivity
        try:
            # Simple connectivity test
            await self.get_quotes(["SPY"])
        except Exception as e:
            health["healthy"] = False
            health["issues"].append(f"API connectivity issue: {e}")

        # Check error rate
        stats = self.metrics.get_stats()
        if stats["success_rate_percent"] < 95 and stats["total_requests"] > 10:
            health["warnings"].append(
                f"Low success rate: {stats['success_rate_percent']:.1f}%"
            )

        return health


# Factory function for easy initialization
async def create_api_client(token_manager: SchwabTokenManager) -> SchwabAPIClient:
    """
    Create configured Schwab API client.

    Args:
        token_manager: Configured token manager

    Returns:
        Configured SchwabAPIClient
    """
    client = SchwabAPIClient(token_manager)
    await client._ensure_session()
    return client


if __name__ == "__main__":
    """API client testing."""

    async def main():
        from .token_manager import create_token_manager

        try:
            # Create token manager and API client
            token_manager = await create_token_manager()

            async with create_api_client(token_manager) as client:
                # Test API status
                status = await client.get_api_status()
                print(f"API Status: {status}")

                # Test quote retrieval
                quotes = await client.get_quotes(["SPY", "AAPL", "MSFT"])
                for quote in quotes:
                    print(f"{quote.symbol}: ${quote.last_price}")

                # Health check
                health = await client.health_check()
                print(f"Health Check: {health}")

        except Exception as e:
            logger.error(f"API client test failed: {e}")

    asyncio.run(main())
