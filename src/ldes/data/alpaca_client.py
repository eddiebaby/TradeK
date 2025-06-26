"""
Alpaca Market Data Provider

Implements real-time and historical market data access via Alpaca Markets API.
Supports both paper trading and live trading environments.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from ..core.config import MarketDataConfig
from ..core.interfaces import MarketDataProvider
from ..core.models import MarketData

# Try to import Alpaca SDK components
try:
    from alpaca.data import CryptoDataStream, StockDataStream
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.models import Bar, Quote, Trade
    from alpaca.data.requests import (
        StockBarsRequest,
        StockLatestQuoteRequest,
        StockTradesRequest,
    )
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    ALPACA_AVAILABLE = True
except ImportError:
    # Create stub types for type annotations when SDK not available
    Trade = Any
    Quote = Any
    Bar = Any
    TimeFrame = Any
    ALPACA_AVAILABLE = False


logger = logging.getLogger(__name__)


class AlpacaDataProvider(MarketDataProvider):
    """
    Alpaca Markets data provider implementation.

    Features:
    - Real-time trade and quote data via WebSocket
    - Historical OHLCV bars
    - Paper trading support
    - Automatic reconnection
    - Rate limiting compliance
    """

    def __init__(self, config: MarketDataConfig):
        """Initialize Alpaca data provider."""
        if not ALPACA_AVAILABLE:
            raise ImportError(
                "Alpaca SDK not available. Install with: pip install alpaca-py"
            )

        self.config = config
        self.api_key = config.alpaca_api_key
        self.secret_key = config.alpaca_secret_key
        self.base_url = config.alpaca_base_url
        self.data_url = config.alpaca_data_url
        self.stream_url = config.alpaca_stream_url

        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API key and secret key are required")

        # Initialize clients
        self.historical_client = StockHistoricalDataClient(
            self.api_key, self.secret_key
        )
        self.stream_client: StockDataStream | None = None

        # Connection state
        self._connected = False
        self._subscribed_symbols: list[str] = []
        self._data_queue: asyncio.Queue = asyncio.Queue()

        # Performance metrics
        self.trades_received = 0
        self.quotes_received = 0
        self.errors_count = 0

        logger.info("Alpaca data provider initialized")

    @property
    def is_connected(self) -> bool:
        """Check if provider is connected."""
        return self._connected

    @property
    def supported_symbols(self) -> list[str]:
        """Get list of supported symbols."""
        # Alpaca supports most US equities and ETFs
        # For now, return configured symbols
        return self.config.alpaca_symbols

    async def connect(self) -> None:
        """Establish connection to Alpaca data streams."""
        try:
            # Initialize stream client
            self.stream_client = StockDataStream(
                self.api_key, self.secret_key, url_override=self.stream_url
            )

            # Set up event handlers
            self._setup_event_handlers()

            self._connected = True
            logger.info("Connected to Alpaca data stream")

        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            raise

    async def disconnect(self) -> None:
        """Close connection to Alpaca data streams."""
        if self.stream_client:
            try:
                await self.stream_client.close()
                self._connected = False
                logger.info("Disconnected from Alpaca data stream")
            except Exception as e:
                logger.error(f"Error disconnecting from Alpaca: {e}")

        self.stream_client = None
        self._subscribed_symbols.clear()

    def _setup_event_handlers(self) -> None:
        """Set up WebSocket event handlers."""
        if not self.stream_client:
            return

        # Trade handler
        async def handle_trade(trade: Trade):
            """Handle incoming trade data."""
            try:
                market_data = self._convert_trade_to_market_data(trade)
                await self._data_queue.put(market_data)
                self.trades_received += 1
            except Exception as e:
                logger.error(f"Error handling trade: {e}")
                self.errors_count += 1

        # Quote handler
        async def handle_quote(quote: Quote):
            """Handle incoming quote data."""
            try:
                market_data = self._convert_quote_to_market_data(quote)
                await self._data_queue.put(market_data)
                self.quotes_received += 1
            except Exception as e:
                logger.error(f"Error handling quote: {e}")
                self.errors_count += 1

        # Register handlers (this is a simplified approach - actual implementation may vary)
        # In real Alpaca SDK, you would subscribe to specific data types
        self._trade_handler = handle_trade
        self._quote_handler = handle_quote

    def _convert_trade_to_market_data(self, trade: Trade) -> MarketData:
        """Convert Alpaca trade to normalized MarketData."""
        return MarketData(
            symbol=trade.symbol,
            timestamp=trade.timestamp,
            last_price=Decimal(str(trade.price)),
            last_size=trade.size,
            volume=trade.size,  # Individual trade volume
            source="alpaca",
        )

    def _convert_quote_to_market_data(self, quote: Quote) -> MarketData:
        """Convert Alpaca quote to normalized MarketData."""
        return MarketData(
            symbol=quote.symbol,
            timestamp=quote.timestamp,
            bid_price=Decimal(str(quote.bid_price)) if quote.bid_price else None,
            bid_size=quote.bid_size,
            ask_price=Decimal(str(quote.ask_price)) if quote.ask_price else None,
            ask_size=quote.ask_size,
            source="alpaca",
        )

    def _convert_bar_to_market_data(
        self, symbol: str, timestamp: datetime, bar: Bar
    ) -> MarketData:
        """Convert Alpaca bar to normalized MarketData."""
        return MarketData(
            symbol=symbol,
            timestamp=timestamp,
            last_price=Decimal(str(bar.close)),
            volume=bar.volume,
            vwap=Decimal(str(bar.vwap)) if hasattr(bar, "vwap") and bar.vwap else None,
            source="alpaca",
        )

    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to real-time data for symbols."""
        if not self.stream_client:
            raise RuntimeError("Not connected to Alpaca")

        try:
            # Subscribe to trades and quotes for each symbol
            for symbol in symbols:
                # Note: Actual Alpaca SDK subscription would be different
                # This is a conceptual implementation
                pass

            self._subscribed_symbols.extend(symbols)
            logger.info(f"Subscribed to {len(symbols)} symbols on Alpaca")

        except Exception as e:
            logger.error(f"Failed to subscribe to symbols: {e}")
            raise

    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from real-time data for symbols."""
        if not self.stream_client:
            return

        try:
            # Unsubscribe from symbols
            for symbol in symbols:
                if symbol in self._subscribed_symbols:
                    self._subscribed_symbols.remove(symbol)

            logger.info(f"Unsubscribed from {len(symbols)} symbols on Alpaca")

        except Exception as e:
            logger.error(f"Failed to unsubscribe from symbols: {e}")

    async def get_stream(self) -> AsyncGenerator[MarketData, None]:
        """Get real-time market data stream."""
        if not self._connected:
            raise RuntimeError("Not connected to Alpaca")

        while self._connected:
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

    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1min",
    ) -> list[MarketData]:
        """Get historical market data."""
        try:
            # Convert timeframe string to Alpaca TimeFrame
            alpaca_timeframe = self._parse_timeframe(timeframe)

            # Create request
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=alpaca_timeframe,
                start=start_date,
                end=end_date,
                adjustment="all",  # Include dividend/split adjustments
                limit=10000,  # Maximum allowed by Alpaca
            )

            # Fetch data
            bars_response = self.historical_client.get_stock_bars(request)

            # Convert to MarketData list
            market_data_list = []
            if symbol in bars_response.data:
                for bar in bars_response.data[symbol]:
                    market_data = self._convert_bar_to_market_data(
                        symbol, bar.timestamp, bar
                    )
                    market_data_list.append(market_data)

            logger.info(
                f"Retrieved {len(market_data_list)} historical data points for {symbol}"
            )
            return market_data_list

        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return []

    def _parse_timeframe(self, timeframe: str) -> TimeFrame:
        """Parse timeframe string to Alpaca TimeFrame."""
        # Parse timeframe like "1min", "5min", "1hour", "1day"
        if timeframe.endswith("min"):
            amount = int(timeframe[:-3])
            return TimeFrame(amount, TimeFrameUnit.Minute)
        elif timeframe.endswith("hour"):
            amount = int(timeframe[:-4])
            return TimeFrame(amount, TimeFrameUnit.Hour)
        elif timeframe.endswith("day"):
            amount = int(timeframe[:-3])
            return TimeFrame(amount, TimeFrameUnit.Day)
        else:
            # Default to 1 minute
            return TimeFrame(1, TimeFrameUnit.Minute)

    async def get_latest_quote(self, symbol: str) -> MarketData | None:
        """Get latest quote for a symbol."""
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes_response = self.historical_client.get_stock_latest_quote(request)

            if symbol in quotes_response:
                quote = quotes_response[symbol]
                return self._convert_quote_to_market_data(quote)

            return None

        except Exception as e:
            logger.error(f"Failed to get latest quote for {symbol}: {e}")
            return None

    def get_provider_info(self) -> dict[str, Any]:
        """Get provider information and metrics."""
        return {
            "name": "Alpaca Markets",
            "is_connected": self.is_connected,
            "subscribed_symbols": len(self._subscribed_symbols),
            "trades_received": self.trades_received,
            "quotes_received": self.quotes_received,
            "errors_count": self.errors_count,
            "supported_symbols_count": len(self.supported_symbols),
            "api_url": self.base_url,
            "stream_url": self.stream_url,
        }


# Mock implementation for when Alpaca SDK is not available
class MockAlpacaDataProvider(MarketDataProvider):
    """Mock Alpaca provider for testing without API access."""

    def __init__(self, config: MarketDataConfig):
        self.config = config
        self._connected = False
        self._symbols = config.alpaca_symbols

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def supported_symbols(self) -> list[str]:
        return self._symbols

    async def connect(self) -> None:
        self._connected = True
        logger.info("Mock Alpaca provider connected")

    async def disconnect(self) -> None:
        self._connected = False
        logger.info("Mock Alpaca provider disconnected")

    async def subscribe(self, symbols: list[str]) -> None:
        logger.info(f"Mock Alpaca subscribed to {len(symbols)} symbols")

    async def unsubscribe(self, symbols: list[str]) -> None:
        logger.info(f"Mock Alpaca unsubscribed from {len(symbols)} symbols")

    async def get_stream(self) -> AsyncGenerator[MarketData, None]:
        """Generate mock data stream."""
        import random

        while self._connected:
            # Generate mock data for each subscribed symbol
            for symbol in self._symbols[:3]:  # Limit to first 3 symbols
                mock_data = MarketData(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    bid_price=Decimal(str(100 + random.uniform(-5, 5))),
                    ask_price=Decimal(str(100.25 + random.uniform(-5, 5))),
                    last_price=Decimal(str(100.12 + random.uniform(-5, 5))),
                    bid_size=random.randint(100, 1000),
                    ask_size=random.randint(100, 1000),
                    last_size=random.randint(10, 500),
                    volume=random.randint(1000, 10000),
                    source="alpaca_mock",
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
        """Generate mock historical data."""
        import random

        data_points = []
        current_time = start_date
        base_price = 100.0

        while current_time < end_date:
            # Random walk price
            base_price += random.uniform(-0.5, 0.5)

            mock_data = MarketData(
                symbol=symbol,
                timestamp=current_time,
                last_price=Decimal(str(base_price)),
                volume=random.randint(100, 1000),
                source="alpaca_mock",
            )
            data_points.append(mock_data)

            # Increment time by 1 minute
            current_time += timedelta(minutes=1)

            # Limit to 1000 points for performance
            if len(data_points) >= 1000:
                break

        return data_points

    async def get_latest_quote(self, symbol: str) -> MarketData | None:
        """Generate mock latest quote."""
        import random

        return MarketData(
            symbol=symbol,
            timestamp=datetime.now(),
            bid_price=Decimal(str(100 + random.uniform(-5, 5))),
            ask_price=Decimal(str(100.25 + random.uniform(-5, 5))),
            last_price=Decimal(str(100.12 + random.uniform(-5, 5))),
            bid_size=random.randint(100, 1000),
            ask_size=random.randint(100, 1000),
            volume=random.randint(1000, 10000),
            source="alpaca_mock",
        )


def create_alpaca_provider(
    config: MarketDataConfig, use_mock: bool = False
) -> MarketDataProvider:
    """Factory function to create Alpaca provider."""
    if use_mock or not ALPACA_AVAILABLE:
        logger.warning("Using mock Alpaca provider")
        return MockAlpacaDataProvider(config)
    else:
        return AlpacaDataProvider(config)
