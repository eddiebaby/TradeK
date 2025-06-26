"""
IEX Cloud API Client

Free tier real-time equity data client with rate limiting and error handling.
Provides 15-minute delayed quotes for U.S. equities.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class IEXQuote:
    """IEX Cloud quote data structure"""

    symbol: str
    latest_price: float
    latest_time: datetime
    latest_update: datetime
    latest_volume: int
    previous_close: float
    change: float
    change_percent: float
    avg_total_volume: int
    market_cap: int | None = None
    pe_ratio: float | None = None
    week_52_high: float | None = None
    week_52_low: float | None = None

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "IEXQuote":
        """Create IEXQuote from API response data"""
        return cls(
            symbol=data.get("symbol", ""),
            latest_price=float(data.get("latestPrice", 0)),
            latest_time=datetime.fromtimestamp(
                data.get("latestUpdate", 0) / 1000, tz=UTC
            ),
            latest_update=datetime.fromtimestamp(
                data.get("latestUpdate", 0) / 1000, tz=UTC
            ),
            latest_volume=int(data.get("latestVolume", 0)),
            previous_close=float(data.get("previousClose", 0)),
            change=float(data.get("change", 0)),
            change_percent=float(data.get("changePercent", 0)),
            avg_total_volume=int(data.get("avgTotalVolume", 0)),
            market_cap=data.get("marketCap"),
            pe_ratio=data.get("peRatio"),
            week_52_high=data.get("week52High"),
            week_52_low=data.get("week52Low"),
        )


class RateLimiter:
    """Rate limiter for IEX Cloud API (100 requests/second)"""

    def __init__(self, max_requests: int = 95, time_window: int = 1):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Acquire rate limit slot"""
        async with self._lock:
            now = time.time()

            # Remove old requests outside time window
            self.requests = [
                req_time
                for req_time in self.requests
                if now - req_time < self.time_window
            ]

            # If at limit, wait
            if len(self.requests) >= self.max_requests:
                sleep_time = self.time_window - (now - self.requests[0]) + 0.1
                logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
                return await self.acquire()

            # Add current request
            self.requests.append(now)


class IEXCloudClient:
    """IEX Cloud API client for real-time equity data"""

    BASE_URL = "https://cloud.iexapis.com/stable"

    def __init__(self, api_token: str | None = None, use_sandbox: bool = False):
        """
        Initialize IEX Cloud client

        Args:
            api_token: IEX Cloud API token (optional for some endpoints)
            use_sandbox: Use sandbox environment for testing
        """
        self.api_token = api_token
        self.base_url = (
            "https://sandbox.iexapis.com/stable" if use_sandbox else self.BASE_URL
        )
        self.rate_limiter = RateLimiter()
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"User-Agent": "TradeKnowledge-LDES/1.0"},
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def _make_request(
        self, endpoint: str, params: dict | None = None
    ) -> dict[str, Any]:
        """Make rate-limited API request"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")

        await self.rate_limiter.acquire()

        url = f"{self.base_url}{endpoint}"
        request_params = params or {}

        if self.api_token:
            request_params["token"] = self.api_token

        try:
            async with self.session.get(url, params=request_params) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.debug(f"IEX API request successful: {endpoint}")
                    return data
                elif response.status == 429:
                    logger.warning("IEX API rate limit exceeded")
                    await asyncio.sleep(1)
                    return await self._make_request(endpoint, params)
                else:
                    error_text = await response.text()
                    logger.error(f"IEX API error {response.status}: {error_text}")
                    raise aiohttp.ClientError(f"API request failed: {response.status}")

        except aiohttp.ClientError as e:
            logger.error(f"IEX API request failed: {e}")
            raise

    async def get_quote(self, symbol: str) -> IEXQuote:
        """Get real-time quote for a single symbol"""
        try:
            data = await self._make_request(f"/stock/{symbol.upper()}/quote")
            return IEXQuote.from_api_response(data)
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            raise

    async def get_quotes_batch(
        self, symbols: list[str], batch_size: int = 50
    ) -> dict[str, IEXQuote]:
        """Get quotes for multiple symbols using batch API"""
        quotes = {}

        # Process in batches to respect API limits
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i : i + batch_size]
            symbols_param = ",".join(batch_symbols).upper()

            try:
                data = await self._make_request(
                    "/stock/market/batch",
                    params={"symbols": symbols_param, "types": "quote"},
                )

                for symbol, symbol_data in data.items():
                    if "quote" in symbol_data:
                        quotes[symbol] = IEXQuote.from_api_response(
                            symbol_data["quote"]
                        )

                logger.info(f"Retrieved {len(batch_symbols)} quotes in batch")

            except Exception as e:
                logger.error(
                    f"Batch quote request failed for symbols {batch_symbols}: {e}"
                )
                # Continue with next batch
                continue

        return quotes

    async def get_intraday_prices(self, symbol: str) -> list[dict[str, Any]]:
        """Get intraday pricing data for a symbol"""
        try:
            data = await self._make_request(f"/stock/{symbol.upper()}/chart/1d")
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"Failed to get intraday prices for {symbol}: {e}")
            return []

    async def validate_connection(self) -> bool:
        """Validate API connection and credentials"""
        try:
            # Test with a simple request
            await self._make_request("/stock/AAPL/quote")
            logger.info("IEX Cloud connection validated successfully")
            return True
        except Exception as e:
            logger.error(f"IEX Cloud connection validation failed: {e}")
            return False


# Example usage and testing
async def test_iex_client():
    """Test the IEX Cloud client"""
    print("🧪 Testing IEX Cloud Client")

    async with IEXCloudClient() as client:
        # Test connection
        is_connected = await client.validate_connection()
        print(f"Connection: {'✅' if is_connected else '❌'}")

        if is_connected:
            # Test single quote
            try:
                quote = await client.get_quote("AAPL")
                print(f"AAPL Quote: ${quote.latest_price} ({quote.change_percent:.2%})")
            except Exception as e:
                print(f"Single quote test failed: {e}")

            # Test batch quotes
            try:
                quotes = await client.get_quotes_batch(["AAPL", "MSFT", "GOOGL"])
                print(f"Batch quotes: {len(quotes)} symbols retrieved")
                for symbol, quote in quotes.items():
                    print(f"  {symbol}: ${quote.latest_price}")
            except Exception as e:
                print(f"Batch quotes test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_iex_client())
