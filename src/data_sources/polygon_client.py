"""
Polygon.io API Client

End-of-day equity data client for data verification and historical analysis.
Free tier provides 5 API calls per minute.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class PolygonDailyBar:
    """Polygon daily bar data structure"""

    ticker: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float | None = None
    transactions: int | None = None

    @classmethod
    def from_api_response(cls, ticker: str, data: dict[str, Any]) -> "PolygonDailyBar":
        """Create PolygonDailyBar from API response"""
        return cls(
            ticker=ticker,
            timestamp=datetime.fromtimestamp(data.get("t", 0) / 1000, tz=UTC),
            open=float(data.get("o", 0)),
            high=float(data.get("h", 0)),
            low=float(data.get("l", 0)),
            close=float(data.get("c", 0)),
            volume=int(data.get("v", 0)),
            vwap=data.get("vw"),
            transactions=data.get("n"),
        )


@dataclass
class PolygonPreviousClose:
    """Polygon previous close data structure"""

    ticker: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "PolygonPreviousClose":
        """Create PolygonPreviousClose from API response"""
        return cls(
            ticker=data.get("T", ""),
            timestamp=datetime.fromtimestamp(data.get("t", 0) / 1000, tz=UTC),
            open=float(data.get("o", 0)),
            high=float(data.get("h", 0)),
            low=float(data.get("l", 0)),
            close=float(data.get("c", 0)),
            volume=int(data.get("v", 0)),
        )


class PolygonRateLimiter:
    """Rate limiter for Polygon API (5 requests/minute for free tier)"""

    def __init__(self, max_requests: int = 4, time_window: int = 60):
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
                sleep_time = self.time_window - (now - self.requests[0]) + 1
                logger.info(
                    f"Polygon rate limit reached, sleeping for {sleep_time:.1f}s"
                )
                await asyncio.sleep(sleep_time)
                return await self.acquire()

            # Add current request
            self.requests.append(now)


class PolygonClient:
    """Polygon.io API client for end-of-day equity data"""

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: str):
        """
        Initialize Polygon client

        Args:
            api_key: Polygon.io API key (required)
        """
        if not api_key:
            raise ValueError("Polygon API key is required")

        self.api_key = api_key
        self.rate_limiter = PolygonRateLimiter()
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

        url = f"{self.BASE_URL}{endpoint}"
        request_params = params or {}
        request_params["apikey"] = self.api_key

        try:
            async with self.session.get(url, params=request_params) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.debug(f"Polygon API request successful: {endpoint}")
                    return data
                elif response.status == 429:
                    logger.warning("Polygon API rate limit exceeded")
                    await asyncio.sleep(60)  # Wait a full minute for Polygon
                    return await self._make_request(endpoint, params)
                else:
                    error_text = await response.text()
                    logger.error(f"Polygon API error {response.status}: {error_text}")
                    raise aiohttp.ClientError(f"API request failed: {response.status}")

        except aiohttp.ClientError as e:
            logger.error(f"Polygon API request failed: {e}")
            raise

    async def get_previous_close(self, ticker: str) -> PolygonPreviousClose | None:
        """Get previous day's close for a ticker"""
        try:
            data = await self._make_request(f"/v2/aggs/ticker/{ticker.upper()}/prev")

            if data.get("status") == "OK" and data.get("results"):
                result = data["results"][0]
                result["T"] = ticker.upper()  # Add ticker to result
                return PolygonPreviousClose.from_api_response(result)
            else:
                logger.warning(f"No previous close data for {ticker}")
                return None

        except Exception as e:
            logger.error(f"Failed to get previous close for {ticker}: {e}")
            return None

    async def get_daily_open_close(
        self, ticker: str, date: date
    ) -> PolygonDailyBar | None:
        """Get open/close data for a specific date"""
        try:
            date_str = date.strftime("%Y-%m-%d")
            data = await self._make_request(
                f"/v1/open-close/{ticker.upper()}/{date_str}"
            )

            if data.get("status") == "OK":
                # Convert to daily bar format
                bar_data = {
                    "t": int(
                        datetime.combine(date, datetime.min.time()).timestamp() * 1000
                    ),
                    "o": data.get("open"),
                    "h": data.get("high"),
                    "l": data.get("low"),
                    "c": data.get("close"),
                    "v": data.get("volume"),
                    "vw": None,
                    "n": None,
                }
                return PolygonDailyBar.from_api_response(ticker.upper(), bar_data)
            else:
                logger.warning(f"No daily data for {ticker} on {date_str}")
                return None

        except Exception as e:
            logger.error(f"Failed to get daily data for {ticker} on {date}: {e}")
            return None

    async def get_historical_range(
        self, ticker: str, start_date: date, end_date: date
    ) -> list[PolygonDailyBar]:
        """Get historical daily bars for a date range"""
        try:
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            data = await self._make_request(
                f"/v2/aggs/ticker/{ticker.upper()}/range/1/day/{start_str}/{end_str}",
                params={"adjusted": "true", "sort": "asc"},
            )

            bars = []
            if data.get("status") == "OK" and data.get("results"):
                for result in data["results"]:
                    bar = PolygonDailyBar.from_api_response(ticker.upper(), result)
                    bars.append(bar)

            logger.info(f"Retrieved {len(bars)} historical bars for {ticker}")
            return bars

        except Exception as e:
            logger.error(f"Failed to get historical range for {ticker}: {e}")
            return []

    async def validate_connection(self) -> bool:
        """Validate API connection and credentials"""
        try:
            # Test with a simple request for AAPL previous close
            result = await self.get_previous_close("AAPL")
            if result:
                logger.info("Polygon API connection validated successfully")
                return True
            else:
                logger.error("Polygon API validation failed - no data returned")
                return False
        except Exception as e:
            logger.error(f"Polygon API connection validation failed: {e}")
            return False

    async def get_market_status(self) -> dict[str, Any]:
        """Get current market status"""
        try:
            data = await self._make_request("/v1/marketstatus/now")
            return data
        except Exception as e:
            logger.error(f"Failed to get market status: {e}")
            return {}


# Example usage and testing
async def test_polygon_client():
    """Test the Polygon client"""
    print("🧪 Testing Polygon Client")

    # Note: Replace with actual API key for testing
    api_key = "YOUR_POLYGON_API_KEY"

    if api_key == "YOUR_POLYGON_API_KEY":
        print("❌ Please set a valid Polygon API key for testing")
        return

    async with PolygonClient(api_key) as client:
        # Test connection
        is_connected = await client.validate_connection()
        print(f"Connection: {'✅' if is_connected else '❌'}")

        if is_connected:
            # Test previous close
            try:
                prev_close = await client.get_previous_close("AAPL")
                if prev_close:
                    print(
                        f"AAPL Previous Close: ${prev_close.close} (Volume: {prev_close.volume:,})"
                    )
            except Exception as e:
                print(f"Previous close test failed: {e}")

            # Test daily open/close
            try:
                yesterday = date.today().replace(day=date.today().day - 1)
                daily_data = await client.get_daily_open_close("AAPL", yesterday)
                if daily_data:
                    print(f"AAPL Daily {yesterday}: ${daily_data.close}")
            except Exception as e:
                print(f"Daily data test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_polygon_client())
