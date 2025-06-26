"""
Schwab Market Data Provider

Implements market data access via Charles Schwab API using OAuth 2.0 authentication.
Supports both real-time streaming data via WebSocket and historical data via REST API.
"""

import asyncio
import logging
import os
import tempfile
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from ..core.config import MarketDataConfig
from ..core.interfaces import MarketDataProvider
from ..core.models import MarketData, OrderBook, OrderBookLevel

# Try to import Schwab SDK components
try:
    import schwab
    from schwab import auth, client
    from schwab.streaming import StreamClient

    SCHWAB_AVAILABLE = True
except ImportError:
    # Create stub types for type annotations when SDK not available
    StreamClient = Any
    SCHWAB_AVAILABLE = False


logger = logging.getLogger(__name__)


class SchwabDataProvider(MarketDataProvider):
    """
    Charles Schwab data provider implementation.

    Features:
    - OAuth 2.0 authentication with automatic token refresh
    - Real-time streaming data via WebSocket
    - Historical price data via REST API
    - Level 1 quotes and Level 2 order book data
    - Automatic reconnection and error handling
    - Production-ready token management
    """

    def __init__(self, config: MarketDataConfig):
        """Initialize Schwab data provider."""
        if not SCHWAB_AVAILABLE:
            raise ImportError(
                "Schwab SDK not available. Install with: pip install schwab-py"
            )

        self.config = config
        self.app_key = config.schwab_app_key
        self.app_secret = config.schwab_secret
        self.redirect_uri = config.schwab_redirect_uri

        if not self.app_key or not self.app_secret:
            raise ValueError("Schwab app key and secret are required")

        # Token management
        self.token_path = self._get_token_path()

        # Clients
        self.http_client: client.Client | None = None
        self.stream_client: StreamClient | None = None

        # Connection state
        self._connected = False
        self._stream_connected = False
        self._subscribed_symbols: list[str] = []
        self._data_queue: asyncio.Queue = asyncio.Queue()

        # Performance metrics
        self.quotes_received = 0
        self.level2_updates_received = 0
        self.chart_updates_received = 0
        self.errors_count = 0
        self.reconnection_count = 0

        # Rate limiting
        self._last_request_time = 0
        self._request_count = 0
        self._rate_limit_window = 60  # seconds
        self._max_requests_per_minute = 120  # Conservative estimate

        logger.info("Schwab data provider initialized")

    def _get_token_path(self) -> str:
        """Get secure token storage path."""
        # Use environment variable if set, otherwise use temp directory
        token_dir = os.getenv("SCHWAB_TOKEN_DIR", tempfile.gettempdir())
        token_file = os.getenv("SCHWAB_TOKEN_FILE", "schwab_token.json")
        return os.path.join(token_dir, token_file)

    @property
    def is_connected(self) -> bool:
        """Check if provider is connected."""
        return self._connected and self.http_client is not None

    @property
    def supported_symbols(self) -> list[str]:
        """Get list of supported symbols."""
        # Schwab supports US equities, options, ETFs, and some indices
        # Return a comprehensive list of major symbols
        return [
            # Major indices
            "SPY",
            "QQQ",
            "IWM",
            "DIA",
            "VTI",
            "VXUS",
            "VEA",
            "VWO",
            # Tech giants
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "TSLA",
            "META",
            "NVDA",
            # Banks
            "JPM",
            "BAC",
            "WFC",
            "C",
            "GS",
            "MS",
            # Other major stocks
            "BRK.B",
            "JNJ",
            "V",
            "PG",
            "UNH",
            "HD",
            "MA",
            "PYPL",
            # Bonds and commodities
            "TLT",
            "IEF",
            "GLD",
            "SLV",
            "USO",
            "TIP",
            # Volatility
            "VIX",
            "UVXY",
            "SVXY",
        ]

    async def connect(self) -> None:
        """Establish connection to Schwab API."""
        try:
            # Initialize HTTP client with OAuth
            await self._initialize_http_client()

            # Initialize streaming client
            await self._initialize_stream_client()

            self._connected = True
            logger.info("Connected to Schwab API")

        except Exception as e:
            logger.error(f"Failed to connect to Schwab: {e}")
            raise

    async def _initialize_http_client(self) -> None:
        """Initialize HTTP client with OAuth authentication."""
        try:
            # Check if we're in a server environment (no GUI)
            if os.getenv("SCHWAB_SERVER_MODE", "false").lower() == "true":
                # Server mode - use manual token if available
                if os.path.exists(self.token_path):
                    # Load existing token
                    self.http_client = auth.client_from_token_file(
                        self.token_path, self.app_key, self.app_secret
                    )
                else:
                    raise RuntimeError(
                        "Server mode requires existing token file. "
                        "Run initial authentication in interactive mode first."
                    )
            else:
                # Interactive mode - use easy_client for OAuth flow
                self.http_client = auth.easy_client(
                    api_key=self.app_key,
                    app_secret=self.app_secret,
                    callback_url=self.redirect_uri,
                    token_path=self.token_path,
                )

            # Test the connection
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.http_client.get_user_preferences
            )
            response.raise_for_status()

            logger.info("Schwab HTTP client authenticated successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Schwab HTTP client: {e}")
            raise

    async def _initialize_stream_client(self) -> None:
        """Initialize streaming client."""
        try:
            if not self.http_client:
                raise RuntimeError("HTTP client must be initialized first")

            # Create streaming client
            self.stream_client = StreamClient(self.http_client, account_id=None)

            # Set up data handlers
            self._setup_stream_handlers()

            logger.info("Schwab streaming client initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Schwab streaming client: {e}")
            # Don't raise - streaming is optional
            self.stream_client = None

    def _setup_stream_handlers(self) -> None:
        """Set up streaming data handlers."""
        if not self.stream_client:
            return

        # Level 1 quote handler
        async def handle_level1_quotes(message: dict[str, Any]):
            """Handle Level 1 quote data."""
            try:
                if "content" in message:
                    for quote_data in message["content"]:
                        market_data = self._convert_level1_to_market_data(quote_data)
                        if market_data:
                            await self._data_queue.put(market_data)
                            self.quotes_received += 1
            except Exception as e:
                logger.error(f"Error handling Level 1 quote: {e}")
                self.errors_count += 1

        # Level 2 order book handler
        async def handle_level2_books(message: dict[str, Any]):
            """Handle Level 2 order book data."""
            try:
                if "content" in message:
                    for book_data in message["content"]:
                        market_data = self._convert_level2_to_market_data(book_data)
                        if market_data:
                            await self._data_queue.put(market_data)
                            self.level2_updates_received += 1
            except Exception as e:
                logger.error(f"Error handling Level 2 book: {e}")
                self.errors_count += 1

        # Chart data handler
        async def handle_chart_equity(message: dict[str, Any]):
            """Handle chart/OHLCV data."""
            try:
                if "content" in message:
                    for chart_data in message["content"]:
                        market_data = self._convert_chart_to_market_data(chart_data)
                        if market_data:
                            await self._data_queue.put(market_data)
                            self.chart_updates_received += 1
            except Exception as e:
                logger.error(f"Error handling chart data: {e}")
                self.errors_count += 1

        # Register handlers
        self.stream_client.add_level_one_equity_handler(handle_level1_quotes)
        self.stream_client.add_nasdaq_book_handler(handle_level2_books)
        self.stream_client.add_nyse_book_handler(handle_level2_books)
        self.stream_client.add_chart_equity_handler(handle_chart_equity)

    def _convert_level1_to_market_data(
        self, quote_data: dict[str, Any]
    ) -> MarketData | None:
        """Convert Schwab Level 1 quote to normalized MarketData."""
        try:
            symbol = quote_data.get("key", "")
            if not symbol:
                return None

            # Extract bid/ask data
            bid_price = quote_data.get("BID_PRICE")
            ask_price = quote_data.get("ASK_PRICE")
            last_price = quote_data.get("LAST_PRICE")
            bid_size = quote_data.get("BID_SIZE")
            ask_size = quote_data.get("ASK_SIZE")
            last_size = quote_data.get("LAST_SIZE")
            volume = quote_data.get("TOTAL_VOLUME")

            return MarketData(
                symbol=symbol,
                timestamp=datetime.now(),  # Schwab doesn't provide explicit timestamp
                bid_price=Decimal(str(bid_price)) if bid_price else None,
                bid_size=int(bid_size) if bid_size else None,
                ask_price=Decimal(str(ask_price)) if ask_price else None,
                ask_size=int(ask_size) if ask_size else None,
                last_price=Decimal(str(last_price)) if last_price else None,
                last_size=int(last_size) if last_size else None,
                volume=int(volume) if volume else None,
                source="schwab",
            )

        except Exception as e:
            logger.warning(f"Failed to convert Level 1 data: {e}")
            return None

    def _convert_level2_to_market_data(
        self, book_data: dict[str, Any]
    ) -> MarketData | None:
        """Convert Schwab Level 2 order book to normalized MarketData."""
        try:
            symbol = book_data.get("key", "")
            if not symbol:
                return None

            # Build order book
            bids = []
            asks = []

            # Extract bid levels
            for i in range(10):  # Schwab typically provides 10 levels
                bid_price = book_data.get(f"BID_PRICE_{i}")
                bid_size = book_data.get(f"BID_SIZE_{i}")
                if bid_price and bid_size:
                    bids.append(
                        OrderBookLevel(
                            price=Decimal(str(bid_price)), size=int(bid_size)
                        )
                    )

            # Extract ask levels
            for i in range(10):
                ask_price = book_data.get(f"ASK_PRICE_{i}")
                ask_size = book_data.get(f"ASK_SIZE_{i}")
                if ask_price and ask_size:
                    asks.append(
                        OrderBookLevel(
                            price=Decimal(str(ask_price)), size=int(ask_size)
                        )
                    )

            # Create order book
            order_book = OrderBook(
                symbol=symbol, timestamp=datetime.now(), bids=bids, asks=asks
            )

            # Create market data with order book
            best_bid = bids[0].price if bids else None
            best_ask = asks[0].price if asks else None

            return MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                bid_price=best_bid,
                bid_size=bids[0].size if bids else None,
                ask_price=best_ask,
                ask_size=asks[0].size if asks else None,
                order_book=order_book,
                source="schwab",
            )

        except Exception as e:
            logger.warning(f"Failed to convert Level 2 data: {e}")
            return None

    def _convert_chart_to_market_data(
        self, chart_data: dict[str, Any]
    ) -> MarketData | None:
        """Convert Schwab chart data to normalized MarketData."""
        try:
            symbol = chart_data.get("key", "")
            if not symbol:
                return None

            # Extract OHLCV data
            open_price = chart_data.get("OPEN_PRICE")
            high_price = chart_data.get("HIGH_PRICE")
            low_price = chart_data.get("LOW_PRICE")
            close_price = chart_data.get("CLOSE_PRICE")
            volume = chart_data.get("VOLUME")

            return MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                last_price=Decimal(str(close_price)) if close_price else None,
                volume=int(volume) if volume else None,
                source="schwab",
            )

        except Exception as e:
            logger.warning(f"Failed to convert chart data: {e}")
            return None

    async def disconnect(self) -> None:
        """Close connection to Schwab API."""
        self._connected = False

        # Disconnect streaming client
        if self.stream_client and self._stream_connected:
            try:
                await self.stream_client.logout()
                self._stream_connected = False
                logger.info("Disconnected from Schwab streaming")
            except Exception as e:
                logger.error(f"Error disconnecting from Schwab streaming: {e}")

        # HTTP client doesn't need explicit disconnection
        self.http_client = None
        self.stream_client = None
        self._subscribed_symbols.clear()

        logger.info("Disconnected from Schwab API")

    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to real-time data for symbols."""
        if not self.stream_client:
            logger.warning("Streaming client not available")
            return

        try:
            # Login to streaming if not already connected
            if not self._stream_connected:
                await self.stream_client.login()
                self._stream_connected = True
                logger.info("Logged into Schwab streaming")

            # Subscribe to Level 1 quotes
            await self.stream_client.level_one_equity_subs(symbols)

            # Subscribe to Level 2 data for major exchanges
            nasdaq_symbols = [s for s in symbols if s in self.supported_symbols]
            if nasdaq_symbols:
                await self.stream_client.nasdaq_book_subs(nasdaq_symbols)
                await self.stream_client.nyse_book_subs(nasdaq_symbols)

            # Subscribe to chart data
            await self.stream_client.chart_equity_subs(symbols)

            self._subscribed_symbols.extend(symbols)
            logger.info(f"Subscribed to {len(symbols)} symbols on Schwab")

        except Exception as e:
            logger.error(f"Failed to subscribe to symbols: {e}")
            raise

    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from real-time data for symbols."""
        if not self.stream_client or not self._stream_connected:
            return

        try:
            # Unsubscribe from various data types
            await self.stream_client.level_one_equity_unsubs(symbols)
            await self.stream_client.nasdaq_book_unsubs(symbols)
            await self.stream_client.nyse_book_unsubs(symbols)
            await self.stream_client.chart_equity_unsubs(symbols)

            # Remove from subscribed list
            for symbol in symbols:
                if symbol in self._subscribed_symbols:
                    self._subscribed_symbols.remove(symbol)

            logger.info(f"Unsubscribed from {len(symbols)} symbols on Schwab")

        except Exception as e:
            logger.error(f"Failed to unsubscribe from symbols: {e}")

    async def get_stream(self) -> AsyncGenerator[MarketData, None]:
        """Get real-time market data stream."""
        if not self.stream_client or not self._stream_connected:
            raise RuntimeError("Streaming client not connected")

        # Start message handling task
        message_task = asyncio.create_task(self._handle_stream_messages())

        try:
            while self._stream_connected:
                try:
                    # Wait for data with timeout
                    data = await asyncio.wait_for(self._data_queue.get(), timeout=30.0)
                    yield data
                except TimeoutError:
                    # Continue waiting for data
                    continue
                except Exception as e:
                    logger.error(f"Error in data stream: {e}")
                    break
        finally:
            message_task.cancel()
            try:
                await message_task
            except asyncio.CancelledError:
                pass

    async def _handle_stream_messages(self) -> None:
        """Handle incoming streaming messages."""
        try:
            while self._stream_connected:
                await self.stream_client.handle_message()
                await asyncio.sleep(0.001)  # Small delay to prevent busy loop
        except Exception as e:
            logger.error(f"Error handling stream messages: {e}")
            self.errors_count += 1

    async def _enforce_rate_limit(self) -> None:
        """Enforce API rate limits."""
        current_time = asyncio.get_event_loop().time()

        # Reset counter if window has passed
        if current_time - self._last_request_time > self._rate_limit_window:
            self._request_count = 0
            self._last_request_time = current_time

        # Check if we're approaching rate limit
        if self._request_count >= self._max_requests_per_minute:
            wait_time = self._rate_limit_window - (
                current_time - self._last_request_time
            )
            if wait_time > 0:
                logger.warning(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                await asyncio.sleep(wait_time)
                self._request_count = 0
                self._last_request_time = asyncio.get_event_loop().time()

        self._request_count += 1

    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1min",
    ) -> list[MarketData]:
        """Get historical market data."""
        if not self.http_client:
            raise RuntimeError("HTTP client not connected")

        try:
            await self._enforce_rate_limit()

            # Choose appropriate method based on timeframe
            if timeframe == "1min":
                method = self.http_client.get_price_history_every_minute
            elif timeframe == "5min":
                method = self.http_client.get_price_history_every_five_minutes
            elif timeframe == "15min":
                method = self.http_client.get_price_history_every_fifteen_minutes
            elif timeframe == "30min":
                method = self.http_client.get_price_history_every_thirty_minutes
            elif timeframe == "1day":
                method = self.http_client.get_price_history_every_day
            elif timeframe == "1week":
                method = self.http_client.get_price_history_every_week
            else:
                # Default to daily data
                method = self.http_client.get_price_history_every_day

            # Make API request
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: method(symbol, start_date=start_date, end_date=end_date)
            )
            response.raise_for_status()

            # Convert response to MarketData list
            data = response.json()
            return self._convert_price_history_to_market_data(symbol, data)

        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return []

    def _convert_price_history_to_market_data(
        self, symbol: str, price_data: dict[str, Any]
    ) -> list[MarketData]:
        """Convert Schwab price history to MarketData list."""
        market_data_list = []

        try:
            if "candles" in price_data:
                candles = price_data["candles"]

                for candle in candles:
                    timestamp = datetime.fromtimestamp(candle["datetime"] / 1000)

                    market_data = MarketData(
                        symbol=symbol,
                        timestamp=timestamp,
                        last_price=Decimal(str(candle["close"])),
                        volume=int(candle["volume"]),
                        source="schwab",
                    )
                    market_data_list.append(market_data)

            logger.info(
                f"Converted {len(market_data_list)} historical data points for {symbol}"
            )

        except Exception as e:
            logger.error(f"Error converting price history for {symbol}: {e}")

        return market_data_list

    async def get_latest_quote(self, symbol: str) -> MarketData | None:
        """Get latest quote for a symbol."""
        if not self.http_client:
            raise RuntimeError("HTTP client not connected")

        try:
            await self._enforce_rate_limit()

            # Get quote from API
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.http_client.get_quote, symbol
            )
            response.raise_for_status()

            # Convert to MarketData
            quote_data = response.json()
            return self._convert_quote_to_market_data(symbol, quote_data)

        except Exception as e:
            logger.error(f"Failed to get latest quote for {symbol}: {e}")
            return None

    def _convert_quote_to_market_data(
        self, symbol: str, quote_data: dict[str, Any]
    ) -> MarketData | None:
        """Convert Schwab quote to normalized MarketData."""
        try:
            if symbol in quote_data:
                quote = quote_data[symbol]

                return MarketData(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    bid_price=(
                        Decimal(str(quote.get("bidPrice", 0)))
                        if quote.get("bidPrice")
                        else None
                    ),
                    bid_size=(
                        int(quote.get("bidSize", 0)) if quote.get("bidSize") else None
                    ),
                    ask_price=(
                        Decimal(str(quote.get("askPrice", 0)))
                        if quote.get("askPrice")
                        else None
                    ),
                    ask_size=(
                        int(quote.get("askSize", 0)) if quote.get("askSize") else None
                    ),
                    last_price=(
                        Decimal(str(quote.get("lastPrice", 0)))
                        if quote.get("lastPrice")
                        else None
                    ),
                    last_size=(
                        int(quote.get("lastSize", 0)) if quote.get("lastSize") else None
                    ),
                    volume=(
                        int(quote.get("totalVolume", 0))
                        if quote.get("totalVolume")
                        else None
                    ),
                    source="schwab",
                )

            return None

        except Exception as e:
            logger.warning(f"Failed to convert quote data for {symbol}: {e}")
            return None

    def get_provider_info(self) -> dict[str, Any]:
        """Get provider information and metrics."""
        return {
            "name": "Charles Schwab",
            "is_connected": self.is_connected,
            "stream_connected": self._stream_connected,
            "subscribed_symbols": len(self._subscribed_symbols),
            "quotes_received": self.quotes_received,
            "level2_updates_received": self.level2_updates_received,
            "chart_updates_received": self.chart_updates_received,
            "errors_count": self.errors_count,
            "reconnection_count": self.reconnection_count,
            "supported_symbols_count": len(self.supported_symbols),
            "rate_limit_remaining": max(
                0, self._max_requests_per_minute - self._request_count
            ),
            "token_path": self.token_path,
        }


# Mock implementation for testing
class MockSchwabDataProvider(MarketDataProvider):
    """Mock Schwab provider for testing without API access."""

    def __init__(self, config: MarketDataConfig):
        self.config = config
        self._connected = False
        self._symbols = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def supported_symbols(self) -> list[str]:
        return self._symbols

    async def connect(self) -> None:
        self._connected = True
        logger.info("Mock Schwab provider connected")

    async def disconnect(self) -> None:
        self._connected = False
        logger.info("Mock Schwab provider disconnected")

    async def subscribe(self, symbols: list[str]) -> None:
        logger.info(f"Mock Schwab subscribed to {len(symbols)} symbols")

    async def unsubscribe(self, symbols: list[str]) -> None:
        logger.info(f"Mock Schwab unsubscribed from {len(symbols)} symbols")

    async def get_stream(self) -> AsyncGenerator[MarketData, None]:
        """Generate mock US equity data stream."""
        import random

        while self._connected:
            # Generate mock data for major US stocks
            for symbol in self._symbols[:5]:  # Limit to first 5 symbols
                # US equity prices with realistic ranges
                if symbol == "SPY":
                    base_price = 400.0
                elif symbol == "QQQ":
                    base_price = 350.0
                elif symbol == "AAPL":
                    base_price = 150.0
                elif symbol == "MSFT":
                    base_price = 300.0
                else:
                    base_price = 100.0

                mock_data = MarketData(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    bid_price=Decimal(str(base_price + random.uniform(-2, 2))),
                    ask_price=Decimal(str(base_price + 0.01 + random.uniform(-2, 2))),
                    last_price=Decimal(str(base_price + 0.005 + random.uniform(-2, 2))),
                    bid_size=random.randint(100, 1000) * 100,  # Round lots
                    ask_size=random.randint(100, 1000) * 100,
                    last_size=random.randint(1, 100) * 100,
                    volume=random.randint(100000, 1000000),
                    source="schwab_mock",
                )
                yield mock_data

            await asyncio.sleep(0.1)  # 10 Hz data rate

    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1min",
    ) -> list[MarketData]:
        """Generate mock historical US equity data."""
        import random

        data_points = []
        current_time = start_date

        # Set realistic base price based on symbol
        if symbol == "SPY":
            base_price = 400.0
        elif symbol == "QQQ":
            base_price = 350.0
        elif symbol == "AAPL":
            base_price = 150.0
        elif symbol == "MSFT":
            base_price = 300.0
        else:
            base_price = 100.0

        # Determine time increment
        if timeframe == "1min":
            time_delta = timedelta(minutes=1)
        elif timeframe == "5min":
            time_delta = timedelta(minutes=5)
        elif timeframe == "1day":
            time_delta = timedelta(days=1)
        else:
            time_delta = timedelta(minutes=1)

        while current_time < end_date:
            # Random walk with mean reversion
            base_price += random.uniform(-0.5, 0.5)

            mock_data = MarketData(
                symbol=symbol,
                timestamp=current_time,
                last_price=Decimal(str(base_price)),
                volume=random.randint(10000, 100000),
                source="schwab_mock",
            )
            data_points.append(mock_data)

            current_time += time_delta

            # Limit to 1000 points for performance
            if len(data_points) >= 1000:
                break

        return data_points

    async def get_latest_quote(self, symbol: str) -> MarketData | None:
        """Generate mock latest quote for US equity."""
        import random

        # Set realistic base price based on symbol
        if symbol == "SPY":
            base_price = 400.0
        elif symbol == "QQQ":
            base_price = 350.0
        elif symbol == "AAPL":
            base_price = 150.0
        elif symbol == "MSFT":
            base_price = 300.0
        else:
            base_price = 100.0

        return MarketData(
            symbol=symbol,
            timestamp=datetime.now(),
            bid_price=Decimal(str(base_price + random.uniform(-1, 1))),
            ask_price=Decimal(str(base_price + 0.01 + random.uniform(-1, 1))),
            last_price=Decimal(str(base_price + 0.005 + random.uniform(-1, 1))),
            bid_size=random.randint(100, 1000) * 100,
            ask_size=random.randint(100, 1000) * 100,
            volume=random.randint(100000, 1000000),
            source="schwab_mock",
        )


def create_schwab_provider(
    config: MarketDataConfig, use_mock: bool = False
) -> MarketDataProvider:
    """Factory function to create Schwab provider."""
    if use_mock or not SCHWAB_AVAILABLE:
        logger.warning("Using mock Schwab provider")
        return MockSchwabDataProvider(config)
    else:
        return SchwabDataProvider(config)
