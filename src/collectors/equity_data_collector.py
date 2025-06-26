"""
Equity Data Collector

Automated collection service that gathers real-time data from IEX Cloud
and stores it in the LDES InfluxDB system with proper tagging and metadata.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from datetime import time as dt_time

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

from ..core.config import get_settings
from ..data_sources.iex_cloud_client import IEXCloudClient, IEXQuote

logger = logging.getLogger(__name__)


@dataclass
class CollectionMetrics:
    """Metrics for monitoring data collection performance"""

    symbols_requested: int
    symbols_collected: int
    collection_time_seconds: float
    api_errors: int
    storage_errors: int
    timestamp: datetime


class EquityDataCollector:
    """Real-time equity data collector for LDES system"""

    def __init__(
        self,
        symbols: list[str],
        collection_interval: int = 15,
        influxdb_url: str | None = None,
        influxdb_token: str | None = None,
        influxdb_org: str | None = None,
        influxdb_bucket: str | None = None,
    ):
        """
        Initialize equity data collector

        Args:
            symbols: List of stock symbols to collect
            collection_interval: Seconds between collections
            influxdb_*: InfluxDB connection parameters (defaults from env)
        """
        self.symbols = [s.upper() for s in symbols]
        self.collection_interval = collection_interval
        self.is_running = False
        self.collection_task: asyncio.Task | None = None

        # InfluxDB configuration
        settings = get_settings()
        self.influxdb_url = influxdb_url or settings.influxdb_url
        self.influxdb_token = influxdb_token or settings.influxdb_token
        self.influxdb_org = influxdb_org or settings.influxdb_org
        self.influxdb_bucket = influxdb_bucket or settings.influxdb_bucket

        # Initialize InfluxDB client
        self.influx_client = InfluxDBClient(
            url=self.influxdb_url, token=self.influxdb_token, org=self.influxdb_org
        )
        self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)

        # IEX client (will be initialized in async context)
        self.iex_client: IEXCloudClient | None = None

        # Collection metrics
        self.metrics_history: list[CollectionMetrics] = []

    async def __aenter__(self):
        """Async context manager entry"""
        self.iex_client = IEXCloudClient()
        await self.iex_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()
        if self.iex_client:
            await self.iex_client.__aexit__(exc_type, exc_val, exc_tb)
        self.influx_client.close()

    def _create_influx_point(self, quote: IEXQuote) -> Point:
        """Create InfluxDB point from IEX quote"""
        point = (
            Point("equity_prices")
            .tag("symbol", quote.symbol)
            .tag("source", "iex_cloud")
            .tag("market", "us_equity")
            .field("price", quote.latest_price)
            .field("volume", quote.latest_volume)
            .field("previous_close", quote.previous_close)
            .field("change", quote.change)
            .field("change_percent", quote.change_percent)
            .field("avg_total_volume", quote.avg_total_volume)
            .time(quote.latest_time)
        )

        # Add optional fields if available
        if quote.market_cap:
            point.field("market_cap", quote.market_cap)
        if quote.pe_ratio:
            point.field("pe_ratio", quote.pe_ratio)
        if quote.week_52_high:
            point.field("week_52_high", quote.week_52_high)
        if quote.week_52_low:
            point.field("week_52_low", quote.week_52_low)

        return point

    async def collect_batch(self) -> CollectionMetrics:
        """Collect data for all symbols in a single batch"""
        start_time = datetime.now(UTC)
        api_errors = 0
        storage_errors = 0
        collected_count = 0

        try:
            if not self.iex_client:
                raise RuntimeError("IEX client not initialized")

            # Get quotes for all symbols
            quotes = await self.iex_client.get_quotes_batch(self.symbols)

            # Convert to InfluxDB points
            points = []
            for symbol, quote in quotes.items():
                try:
                    point = self._create_influx_point(quote)
                    points.append(point)
                    collected_count += 1
                except Exception as e:
                    logger.error(f"Failed to create point for {symbol}: {e}")
                    storage_errors += 1

            # Write to InfluxDB
            if points:
                try:
                    self.write_api.write(
                        bucket=self.influxdb_bucket,
                        org=self.influxdb_org,
                        record=points,
                    )
                    logger.info(f"Stored {len(points)} equity data points")
                except Exception as e:
                    logger.error(f"Failed to write to InfluxDB: {e}")
                    storage_errors += len(points)

        except Exception as e:
            logger.error(f"Batch collection failed: {e}")
            api_errors += 1

        # Calculate metrics
        end_time = datetime.now(UTC)
        collection_time = (end_time - start_time).total_seconds()

        metrics = CollectionMetrics(
            symbols_requested=len(self.symbols),
            symbols_collected=collected_count,
            collection_time_seconds=collection_time,
            api_errors=api_errors,
            storage_errors=storage_errors,
            timestamp=end_time,
        )

        # Store metrics
        self.metrics_history.append(metrics)

        # Log summary
        logger.info(
            f"Collection complete: {collected_count}/{len(self.symbols)} symbols, "
            f"{collection_time:.2f}s, {api_errors} API errors, {storage_errors} storage errors"
        )

        return metrics

    def _is_market_hours(self) -> bool:
        """Check if it's during market hours (simple US market check)"""
        now = datetime.now()

        # Simple check: weekdays between 9:30 AM and 4:00 PM ET
        # Note: This doesn't account for holidays or exact timezone
        if now.weekday() >= 5:  # Weekend
            return False

        # Approximate market hours (would need proper timezone handling in production)
        market_open = dt_time(9, 30)
        market_close = dt_time(16, 0)
        current_time = now.time()

        return market_open <= current_time <= market_close

    async def _collection_loop(self):
        """Main collection loop"""
        logger.info(f"Starting data collection for {len(self.symbols)} symbols")

        while self.is_running:
            try:
                # Only collect during market hours for real-time data
                if self._is_market_hours():
                    await self.collect_batch()
                else:
                    logger.debug("Outside market hours, skipping collection")

                # Wait for next collection interval
                await asyncio.sleep(self.collection_interval)

            except asyncio.CancelledError:
                logger.info("Collection loop cancelled")
                break
            except Exception as e:
                logger.error(f"Collection loop error: {e}")
                # Continue loop but wait a bit longer on error
                await asyncio.sleep(min(self.collection_interval * 2, 60))

    async def start(self):
        """Start automated data collection"""
        if self.is_running:
            logger.warning("Data collection already running")
            return

        logger.info(
            f"Starting equity data collection every {self.collection_interval}s"
        )
        self.is_running = True
        self.collection_task = asyncio.create_task(self._collection_loop())

    async def stop(self):
        """Stop automated data collection"""
        if not self.is_running:
            return

        logger.info("Stopping equity data collection")
        self.is_running = False

        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
            self.collection_task = None

    def get_recent_metrics(self, count: int = 10) -> list[CollectionMetrics]:
        """Get recent collection metrics"""
        return self.metrics_history[-count:] if self.metrics_history else []

    def get_performance_summary(self) -> dict:
        """Get performance summary statistics"""
        if not self.metrics_history:
            return {"error": "No metrics available"}

        recent_metrics = self.get_recent_metrics(20)  # Last 20 collections

        total_requested = sum(m.symbols_requested for m in recent_metrics)
        total_collected = sum(m.symbols_collected for m in recent_metrics)
        avg_collection_time = sum(
            m.collection_time_seconds for m in recent_metrics
        ) / len(recent_metrics)
        total_api_errors = sum(m.api_errors for m in recent_metrics)
        total_storage_errors = sum(m.storage_errors for m in recent_metrics)

        return {
            "success_rate": (
                (total_collected / total_requested * 100) if total_requested > 0 else 0
            ),
            "avg_collection_time_seconds": round(avg_collection_time, 2),
            "total_api_errors": total_api_errors,
            "total_storage_errors": total_storage_errors,
            "collections_analyzed": len(recent_metrics),
            "last_collection": (
                recent_metrics[-1].timestamp.isoformat() if recent_metrics else None
            ),
        }


# Example usage and testing
async def test_equity_collector():
    """Test the equity data collector"""
    print("🧪 Testing Equity Data Collector")

    # Test symbols
    test_symbols = ["AAPL", "MSFT", "GOOGL"]

    async with EquityDataCollector(
        symbols=test_symbols, collection_interval=30  # 30 seconds for testing
    ) as collector:

        # Test single collection
        try:
            metrics = await collector.collect_batch()
            print("Collection Test:")
            print(f"  Requested: {metrics.symbols_requested}")
            print(f"  Collected: {metrics.symbols_collected}")
            print(f"  Time: {metrics.collection_time_seconds:.2f}s")
            print(
                f"  Errors: API={metrics.api_errors}, Storage={metrics.storage_errors}"
            )
        except Exception as e:
            print(f"Collection test failed: {e}")

        # Test automated collection for a short period
        try:
            print("\nStarting automated collection for 2 minutes...")
            await collector.start()
            await asyncio.sleep(120)  # Run for 2 minutes
            await collector.stop()

            summary = collector.get_performance_summary()
            print("Performance Summary:")
            print(f"  Success Rate: {summary.get('success_rate', 0):.1f}%")
            print(
                f"  Avg Collection Time: {summary.get('avg_collection_time_seconds', 0):.2f}s"
            )
            print(f"  Collections: {summary.get('collections_analyzed', 0)}")

        except Exception as e:
            print(f"Automated collection test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_equity_collector())
