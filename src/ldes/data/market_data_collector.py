"""
Market Data Collector

Orchestrates data collection from multiple sources and normalizes data format.
Implements the MarketDataProvider interface with connection pooling and error handling.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from ..core.config import LDESConfig
from ..core.interfaces import DataStorage, MarketDataProvider
from ..core.models import MarketData

logger = logging.getLogger(__name__)


class MarketDataCollector:
    """
    Orchestrates market data collection from multiple providers.

    Features:
    - Multi-source data aggregation
    - Real-time streaming
    - Historical data backfill
    - Connection management
    - Error handling and retry logic
    """

    def __init__(self, config: LDESConfig, storage: DataStorage | None = None):
        """Initialize market data collector."""
        self.config = config
        self.storage = storage
        self.providers: dict[str, MarketDataProvider] = {}
        self.subscribed_symbols: set[str] = set()
        self.is_running = False
        self._tasks: list[asyncio.Task] = []

        # Performance metrics
        self.data_points_processed = 0
        self.errors_encountered = 0
        self.last_data_timestamp: datetime | None = None

    def add_provider(self, name: str, provider: MarketDataProvider) -> None:
        """Add a data provider."""
        self.providers[name] = provider
        logger.info(f"Added data provider: {name}")

    def remove_provider(self, name: str) -> None:
        """Remove a data provider."""
        if name in self.providers:
            del self.providers[name]
            logger.info(f"Removed data provider: {name}")

    async def connect_all(self) -> None:
        """Connect to all data providers."""
        connection_tasks = []
        for name, provider in self.providers.items():
            logger.info(f"Connecting to {name}...")
            task = asyncio.create_task(self._connect_provider(name, provider))
            connection_tasks.append(task)

        # Wait for all connections
        results = await asyncio.gather(*connection_tasks, return_exceptions=True)

        # Log connection results
        for i, (name, result) in enumerate(zip(self.providers.keys(), results, strict=False)):
            if isinstance(result, Exception):
                logger.error(f"Failed to connect to {name}: {result}")
                self.errors_encountered += 1
            else:
                logger.info(f"Successfully connected to {name}")

    async def _connect_provider(self, name: str, provider: MarketDataProvider) -> None:
        """Connect to a single provider with retry logic."""
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                await provider.connect()
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Connection attempt {attempt + 1} failed for {name}: {e}"
                    )
                    await asyncio.sleep(retry_delay * (2**attempt))
                else:
                    raise

    async def disconnect_all(self) -> None:
        """Disconnect from all data providers."""
        disconnect_tasks = []
        for name, provider in self.providers.items():
            if provider.is_connected:
                logger.info(f"Disconnecting from {name}...")
                task = asyncio.create_task(provider.disconnect())
                disconnect_tasks.append(task)

        # Wait for all disconnections
        if disconnect_tasks:
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)

    async def subscribe_symbols(self, symbols: list[str]) -> None:
        """Subscribe to real-time data for symbols across all providers."""
        self.subscribed_symbols.update(symbols)

        subscription_tasks = []
        for name, provider in self.providers.items():
            if provider.is_connected:
                # Filter symbols supported by this provider
                supported_symbols = [
                    s for s in symbols if s in provider.supported_symbols
                ]
                if supported_symbols:
                    logger.info(
                        f"Subscribing to {len(supported_symbols)} symbols on {name}"
                    )
                    task = asyncio.create_task(provider.subscribe(supported_symbols))
                    subscription_tasks.append(task)

        # Wait for all subscriptions
        if subscription_tasks:
            await asyncio.gather(*subscription_tasks, return_exceptions=True)

    async def unsubscribe_symbols(self, symbols: list[str]) -> None:
        """Unsubscribe from real-time data for symbols."""
        self.subscribed_symbols.difference_update(symbols)

        unsubscription_tasks = []
        for name, provider in self.providers.items():
            if provider.is_connected:
                task = asyncio.create_task(provider.unsubscribe(symbols))
                unsubscription_tasks.append(task)

        # Wait for all unsubscriptions
        if unsubscription_tasks:
            await asyncio.gather(*unsubscription_tasks, return_exceptions=True)

    async def start_collection(self) -> None:
        """Start real-time data collection from all providers."""
        if self.is_running:
            logger.warning("Data collection is already running")
            return

        self.is_running = True
        logger.info("Starting market data collection...")

        # Start collection tasks for each provider
        for name, provider in self.providers.items():
            if provider.is_connected:
                task = asyncio.create_task(self._collect_from_provider(name, provider))
                self._tasks.append(task)

        logger.info(f"Started {len(self._tasks)} data collection tasks")

    async def stop_collection(self) -> None:
        """Stop real-time data collection."""
        if not self.is_running:
            return

        self.is_running = False
        logger.info("Stopping market data collection...")

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        self._tasks.clear()
        logger.info("Market data collection stopped")

    async def _collect_from_provider(
        self, name: str, provider: MarketDataProvider
    ) -> None:
        """Collect data from a single provider."""
        logger.info(f"Starting data collection from {name}")

        try:
            async for data in provider.get_stream():
                if not self.is_running:
                    break

                await self._process_market_data(data, name)

        except asyncio.CancelledError:
            logger.info(f"Data collection from {name} cancelled")
            raise
        except Exception as e:
            logger.error(f"Error collecting from {name}: {e}")
            self.errors_encountered += 1

    async def _process_market_data(self, data: MarketData, source: str) -> None:
        """Process incoming market data."""
        try:
            # Update metrics
            self.data_points_processed += 1
            self.last_data_timestamp = data.timestamp

            # Store data if storage is configured
            if self.storage:
                await self.storage.store_market_data(data)

            # Log periodically for monitoring
            if self.data_points_processed % 1000 == 0:
                logger.debug(
                    f"Processed {self.data_points_processed} data points from {source}"
                )

        except Exception as e:
            logger.error(f"Error processing market data from {source}: {e}")
            self.errors_encountered += 1

    async def get_latest_data(self, symbol: str) -> MarketData | None:
        """Get latest market data for a symbol from any provider."""
        for name, provider in self.providers.items():
            if provider.is_connected and symbol in provider.supported_symbols:
                try:
                    data = await provider.get_latest_quote(symbol)
                    if data:
                        return data
                except Exception as e:
                    logger.warning(
                        f"Failed to get latest data for {symbol} from {name}: {e}"
                    )

        return None

    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1min",
        preferred_provider: str | None = None,
    ) -> list[MarketData]:
        """Get historical data with provider preference."""
        # Try preferred provider first
        if preferred_provider and preferred_provider in self.providers:
            provider = self.providers[preferred_provider]
            if provider.is_connected and symbol in provider.supported_symbols:
                try:
                    return await provider.get_historical_data(
                        symbol, start_date, end_date, timeframe
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to get historical data from {preferred_provider}: {e}"
                    )

        # Try other providers
        for name, provider in self.providers.items():
            if name == preferred_provider:
                continue  # Already tried

            if provider.is_connected and symbol in provider.supported_symbols:
                try:
                    return await provider.get_historical_data(
                        symbol, start_date, end_date, timeframe
                    )
                except Exception as e:
                    logger.warning(f"Failed to get historical data from {name}: {e}")

        logger.error(f"Failed to get historical data for {symbol} from any provider")
        return []

    async def backfill_historical_data(
        self, symbols: list[str], days_back: int = 30, timeframe: str = "1min"
    ) -> None:
        """Backfill historical data for multiple symbols."""
        if not self.storage:
            logger.warning("No storage configured for backfill")
            return

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        logger.info(
            f"Starting backfill for {len(symbols)} symbols, {days_back} days back"
        )

        # Process symbols in batches to avoid overwhelming providers
        batch_size = 5
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            tasks = []

            for symbol in batch:
                task = asyncio.create_task(
                    self._backfill_symbol(symbol, start_date, end_date, timeframe)
                )
                tasks.append(task)

            # Wait for batch to complete
            await asyncio.gather(*tasks, return_exceptions=True)

            # Small delay between batches
            if i + batch_size < len(symbols):
                await asyncio.sleep(1.0)

        logger.info("Historical data backfill completed")

    async def _backfill_symbol(
        self, symbol: str, start_date: datetime, end_date: datetime, timeframe: str
    ) -> None:
        """Backfill historical data for a single symbol."""
        try:
            data_points = await self.get_historical_data(
                symbol, start_date, end_date, timeframe
            )

            if data_points:
                # Store all data points
                for data in data_points:
                    await self.storage.store_market_data(data)

                logger.info(f"Backfilled {len(data_points)} data points for {symbol}")
            else:
                logger.warning(f"No historical data available for {symbol}")

        except Exception as e:
            logger.error(f"Error backfilling {symbol}: {e}")

    def get_status(self) -> dict[str, any]:
        """Get collector status and metrics."""
        connected_providers = [
            name for name, provider in self.providers.items() if provider.is_connected
        ]

        return {
            "is_running": self.is_running,
            "providers_count": len(self.providers),
            "connected_providers": connected_providers,
            "subscribed_symbols_count": len(self.subscribed_symbols),
            "data_points_processed": self.data_points_processed,
            "errors_encountered": self.errors_encountered,
            "last_data_timestamp": (
                self.last_data_timestamp.isoformat()
                if self.last_data_timestamp
                else None
            ),
            "active_tasks": len(self._tasks),
        }

    @asynccontextmanager
    async def managed_collection(self, symbols: list[str]):
        """Context manager for automatic collection lifecycle management."""
        try:
            # Connect and start collection
            await self.connect_all()
            await self.subscribe_symbols(symbols)
            await self.start_collection()

            yield self

        finally:
            # Cleanup
            await self.stop_collection()
            await self.unsubscribe_symbols(symbols)
            await self.disconnect_all()
