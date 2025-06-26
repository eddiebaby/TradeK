"""
Binance Market Data Provider

Implements cryptocurrency market data access via Binance API.
Supports real-time WebSocket streams and REST API for historical data.
"""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import aiohttp
import websockets

from ..core.config import MarketDataConfig
from ..core.interfaces import MarketDataProvider
from ..core.models import MarketData

logger = logging.getLogger(__name__)


class BinanceDataProvider(MarketDataProvider):
    """
    Binance cryptocurrency data provider implementation.

    Features:
    - Real-time trade and ticker data via WebSocket
    - Historical kline/candlestick data
    - No authentication required for market data
    - Multiple crypto pairs support
    - Automatic reconnection
    """

    def __init__(self, config: MarketDataConfig):
        """Initialize Binance data provider."""
        self.config = config
        self.base_url = config.binance_base_url
        self.stream_url = config.binance_stream_url

        # Connection state
        self._connected = False
        self._websocket = None
        self._subscribed_symbols: list[str] = []
        self._data_queue: asyncio.Queue = asyncio.Queue()
        self._reconnect_task: asyncio.Task | None = None

        # Performance metrics
        self.trades_received = 0
        self.tickers_received = 0
        self.errors_count = 0
        self.reconnection_count = 0

        logger.info("Binance data provider initialized")

    @property
    def is_connected(self) -> bool:
        """Check if provider is connected."""
        return self._connected and self._websocket is not None

    @property
    def supported_symbols(self) -> list[str]:
        """Get list of supported symbols."""
        return self.config.binance_symbols

    async def connect(self) -> None:
        """Establish connection to Binance WebSocket."""
        try:
            await self._connect_websocket()
            self._connected = True
            logger.info("Connected to Binance WebSocket")

        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            raise

    async def _connect_websocket(self) -> None:
        """Connect to Binance WebSocket with reconnection logic."""
        max_retries = 5
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                # Connect to WebSocket
                self._websocket = await websockets.connect(
                    f"{self.stream_url}/ws/combined",
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10,
                )

                # Start listening task
                self._listen_task = asyncio.create_task(self._listen_to_websocket())

                logger.info("Binance WebSocket connected successfully")
                return

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"WebSocket connection attempt {attempt + 1} failed: {e}"
                    )
                    await asyncio.sleep(retry_delay * (2**attempt))
                else:
                    raise

    async def disconnect(self) -> None:
        """Close connection to Binance WebSocket."""
        self._connected = False

        if hasattr(self, "_listen_task") and self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass

        if self._websocket:
            try:
                await self._websocket.close()
                logger.info("Disconnected from Binance WebSocket")
            except Exception as e:
                logger.error(f"Error disconnecting from Binance: {e}")

        self._websocket = None
        self._subscribed_symbols.clear()

    async def _listen_to_websocket(self) -> None:
        """Listen to WebSocket messages."""
        try:
            async for message in self._websocket:
                if not self._connected:
                    break

                try:
                    data = json.loads(message)
                    await self._process_websocket_message(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse WebSocket message: {e}")
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
                    self.errors_count += 1

        except websockets.exceptions.ConnectionClosed:
            logger.warning("Binance WebSocket connection closed")
            if self._connected:
                await self._handle_reconnection()
        except Exception as e:
            logger.error(f"WebSocket listening error: {e}")
            self.errors_count += 1

    async def _handle_reconnection(self) -> None:
        """Handle WebSocket reconnection."""
        if not self._connected:
            return

        logger.info("Attempting to reconnect to Binance WebSocket...")
        self.reconnection_count += 1

        try:
            await asyncio.sleep(5.0)  # Wait before reconnecting
            await self._connect_websocket()

            # Re-subscribe to symbols
            if self._subscribed_symbols:
                await self._resubscribe_symbols()

        except Exception as e:
            logger.error(f"Reconnection failed: {e}")

    async def _resubscribe_symbols(self) -> None:
        """Re-subscribe to symbols after reconnection."""
        symbols_to_resubscribe = self._subscribed_symbols.copy()
        self._subscribed_symbols.clear()
        await self.subscribe(symbols_to_resubscribe)

    async def _process_websocket_message(self, data: dict[str, Any]) -> None:
        """Process incoming WebSocket message."""
        if "stream" not in data or "data" not in data:
            return

        stream = data["stream"]
        message_data = data["data"]

        # Handle trade stream
        if "@trade" in stream:
            market_data = self._convert_trade_to_market_data(message_data)
            await self._data_queue.put(market_data)
            self.trades_received += 1

        # Handle 24hr ticker stream
        elif "@ticker" in stream:
            market_data = self._convert_ticker_to_market_data(message_data)
            await self._data_queue.put(market_data)
            self.tickers_received += 1

    def _convert_trade_to_market_data(self, trade_data: dict[str, Any]) -> MarketData:
        """Convert Binance trade data to normalized MarketData."""
        return MarketData(
            symbol=trade_data["s"],  # Symbol
            timestamp=datetime.fromtimestamp(trade_data["T"] / 1000),  # Trade time
            last_price=Decimal(trade_data["p"]),  # Price
            last_size=int(float(trade_data["q"])),  # Quantity
            volume=int(float(trade_data["q"])),  # Use trade quantity as volume
            source="binance",
        )

    def _convert_ticker_to_market_data(self, ticker_data: dict[str, Any]) -> MarketData:
        """Convert Binance 24hr ticker data to normalized MarketData."""
        return MarketData(
            symbol=ticker_data["s"],  # Symbol
            timestamp=datetime.fromtimestamp(ticker_data["E"] / 1000),  # Event time
            last_price=Decimal(ticker_data["c"]),  # Close price
            bid_price=(
                Decimal(ticker_data["b"]) if "b" in ticker_data else None
            ),  # Best bid
            ask_price=(
                Decimal(ticker_data["a"]) if "a" in ticker_data else None
            ),  # Best ask
            volume=(
                int(float(ticker_data["v"])) if "v" in ticker_data else None
            ),  # Volume
            source="binance",
        )

    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to real-time data for symbols."""
        if not self._websocket:
            raise RuntimeError("Not connected to Binance")

        try:
            # Create subscription streams
            streams = []
            for symbol in symbols:
                symbol_lower = symbol.lower()
                streams.extend(
                    [
                        f"{symbol_lower}@trade",  # Individual trades
                        f"{symbol_lower}@ticker",  # 24hr ticker statistics
                    ]
                )

            # Subscribe to streams
            subscribe_message = {"method": "SUBSCRIBE", "params": streams, "id": 1}

            await self._websocket.send(json.dumps(subscribe_message))
            self._subscribed_symbols.extend(symbols)

            logger.info(f"Subscribed to {len(symbols)} symbols on Binance")

        except Exception as e:
            logger.error(f"Failed to subscribe to symbols: {e}")
            raise

    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from real-time data for symbols."""
        if not self._websocket:
            return

        try:
            # Create unsubscription streams
            streams = []
            for symbol in symbols:
                symbol_lower = symbol.lower()
                streams.extend([f"{symbol_lower}@trade", f"{symbol_lower}@ticker"])

            # Unsubscribe from streams
            unsubscribe_message = {"method": "UNSUBSCRIBE", "params": streams, "id": 2}

            await self._websocket.send(json.dumps(unsubscribe_message))

            # Remove from subscribed list
            for symbol in symbols:
                if symbol in self._subscribed_symbols:
                    self._subscribed_symbols.remove(symbol)

            logger.info(f"Unsubscribed from {len(symbols)} symbols on Binance")

        except Exception as e:
            logger.error(f"Failed to unsubscribe from symbols: {e}")

    async def get_stream(self) -> AsyncGenerator[MarketData, None]:
        """Get real-time market data stream."""
        if not self.is_connected:
            raise RuntimeError("Not connected to Binance")

        while self.is_connected:
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
        """Get historical kline/candlestick data."""
        try:
            # Convert timeframe to Binance format
            binance_interval = self._convert_timeframe(timeframe)

            # Convert dates to timestamps
            start_time = int(start_date.timestamp() * 1000)
            end_time = int(end_date.timestamp() * 1000)

            # Make API request
            url = f"{self.base_url}/api/v3/klines"
            params = {
                "symbol": symbol,
                "interval": binance_interval,
                "startTime": start_time,
                "endTime": end_time,
                "limit": 1000,  # Maximum allowed by Binance
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._convert_klines_to_market_data(symbol, data)
                    else:
                        logger.error(
                            f"Binance API error {response.status}: {await response.text()}"
                        )
                        return []

        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return []

    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to Binance interval format."""
        # Map common timeframes to Binance intervals
        timeframe_map = {
            "1min": "1m",
            "3min": "3m",
            "5min": "5m",
            "15min": "15m",
            "30min": "30m",
            "1hour": "1h",
            "2hour": "2h",
            "4hour": "4h",
            "6hour": "6h",
            "8hour": "8h",
            "12hour": "12h",
            "1day": "1d",
            "3day": "3d",
            "1week": "1w",
            "1month": "1M",
        }
        return timeframe_map.get(timeframe, "1m")

    def _convert_klines_to_market_data(
        self, symbol: str, klines: list[list]
    ) -> list[MarketData]:
        """Convert Binance kline data to MarketData list."""
        market_data_list = []

        for kline in klines:
            # Binance kline format: [timestamp, open, high, low, close, volume, ...]
            timestamp = datetime.fromtimestamp(kline[0] / 1000)
            close_price = Decimal(kline[4])
            volume = int(float(kline[5]))

            market_data = MarketData(
                symbol=symbol,
                timestamp=timestamp,
                last_price=close_price,
                volume=volume,
                source="binance",
            )
            market_data_list.append(market_data)

        logger.info(
            f"Converted {len(market_data_list)} klines to MarketData for {symbol}"
        )
        return market_data_list

    async def get_latest_quote(self, symbol: str) -> MarketData | None:
        """Get latest ticker/quote for a symbol."""
        try:
            url = f"{self.base_url}/api/v3/ticker/24hr"
            params = {"symbol": symbol}

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        return MarketData(
                            symbol=data["symbol"],
                            timestamp=datetime.now(),
                            last_price=Decimal(data["lastPrice"]),
                            bid_price=(
                                Decimal(data["bidPrice"])
                                if data["bidPrice"] != "0.00000000"
                                else None
                            ),
                            ask_price=(
                                Decimal(data["askPrice"])
                                if data["askPrice"] != "0.00000000"
                                else None
                            ),
                            volume=int(float(data["volume"])),
                            source="binance",
                        )
                    else:
                        logger.error(
                            f"Binance API error {response.status}: {await response.text()}"
                        )
                        return None

        except Exception as e:
            logger.error(f"Failed to get latest quote for {symbol}: {e}")
            return None

    def get_provider_info(self) -> dict[str, Any]:
        """Get provider information and metrics."""
        return {
            "name": "Binance",
            "is_connected": self.is_connected,
            "subscribed_symbols": len(self._subscribed_symbols),
            "trades_received": self.trades_received,
            "tickers_received": self.tickers_received,
            "errors_count": self.errors_count,
            "reconnection_count": self.reconnection_count,
            "supported_symbols_count": len(self.supported_symbols),
            "api_url": self.base_url,
            "stream_url": self.stream_url,
        }


# Mock implementation for testing
class MockBinanceDataProvider(MarketDataProvider):
    """Mock Binance provider for testing without API access."""

    def __init__(self, config: MarketDataConfig):
        self.config = config
        self._connected = False
        self._symbols = config.binance_symbols

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def supported_symbols(self) -> list[str]:
        return self._symbols

    async def connect(self) -> None:
        self._connected = True
        logger.info("Mock Binance provider connected")

    async def disconnect(self) -> None:
        self._connected = False
        logger.info("Mock Binance provider disconnected")

    async def subscribe(self, symbols: list[str]) -> None:
        logger.info(f"Mock Binance subscribed to {len(symbols)} symbols")

    async def unsubscribe(self, symbols: list[str]) -> None:
        logger.info(f"Mock Binance unsubscribed from {len(symbols)} symbols")

    async def get_stream(self) -> AsyncGenerator[MarketData, None]:
        """Generate mock crypto data stream."""
        import random

        while self._connected:
            # Generate mock data for crypto symbols
            for symbol in self._symbols[:2]:  # Limit to first 2 symbols
                # Crypto prices are typically higher and more volatile
                base_price = 30000.0 if "BTC" in symbol else 2000.0

                mock_data = MarketData(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    bid_price=Decimal(str(base_price + random.uniform(-1000, 1000))),
                    ask_price=Decimal(
                        str(base_price + 1 + random.uniform(-1000, 1000))
                    ),
                    last_price=Decimal(
                        str(base_price + 0.5 + random.uniform(-1000, 1000))
                    ),
                    bid_size=random.randint(1, 10),
                    ask_size=random.randint(1, 10),
                    last_size=random.randint(1, 5),
                    volume=random.randint(100, 1000),
                    source="binance_mock",
                )
                yield mock_data

            await asyncio.sleep(0.2)  # 5 Hz data rate

    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1min",
    ) -> list[MarketData]:
        """Generate mock historical crypto data."""
        import random

        data_points = []
        current_time = start_date
        base_price = 30000.0 if "BTC" in symbol else 2000.0

        while current_time < end_date:
            # Random walk price with higher volatility
            base_price += random.uniform(-100, 100)

            mock_data = MarketData(
                symbol=symbol,
                timestamp=current_time,
                last_price=Decimal(str(base_price)),
                volume=random.randint(10, 100),
                source="binance_mock",
            )
            data_points.append(mock_data)

            # Increment time by 1 minute
            current_time += timedelta(minutes=1)

            # Limit to 500 points for performance
            if len(data_points) >= 500:
                break

        return data_points

    async def get_latest_quote(self, symbol: str) -> MarketData | None:
        """Generate mock latest crypto quote."""
        import random

        base_price = 30000.0 if "BTC" in symbol else 2000.0

        return MarketData(
            symbol=symbol,
            timestamp=datetime.now(),
            bid_price=Decimal(str(base_price + random.uniform(-1000, 1000))),
            ask_price=Decimal(str(base_price + 1 + random.uniform(-1000, 1000))),
            last_price=Decimal(str(base_price + 0.5 + random.uniform(-1000, 1000))),
            volume=random.randint(100, 1000),
            source="binance_mock",
        )


def create_binance_provider(
    config: MarketDataConfig, use_mock: bool = False
) -> MarketDataProvider:
    """Factory function to create Binance provider."""
    if use_mock:
        logger.warning("Using mock Binance provider")
        return MockBinanceDataProvider(config)
    else:
        return BinanceDataProvider(config)
