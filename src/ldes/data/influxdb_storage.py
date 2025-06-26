"""
InfluxDB Storage Implementation

Provides time-series storage for market data using InfluxDB.
Implements the DataStorage interface with optimized write performance.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from ..core.config import InfluxDBConfig
from ..core.interfaces import DataStorage
from ..core.models import LiquiditySignal, MarketData, Position, TradeSignal

# Try to import InfluxDB client
try:
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import ASYNCHRONOUS, SYNCHRONOUS
    from influxdb_client.domain.write_precision import WritePrecision

    INFLUXDB_AVAILABLE = True
except ImportError:
    # Create stub types for type annotations when client not available
    Point = Any
    INFLUXDB_AVAILABLE = False


logger = logging.getLogger(__name__)


class InfluxDBStorage(DataStorage):
    """
    InfluxDB storage implementation for LDES time-series data.

    Features:
    - High-performance batch writes
    - Time-series optimized storage
    - Automatic data retention policies
    - Query optimization for market data
    - Connection pooling and retry logic
    """

    def __init__(self, config: InfluxDBConfig):
        """Initialize InfluxDB storage."""
        if not INFLUXDB_AVAILABLE:
            raise ImportError(
                "InfluxDB client not available. Install with: pip install influxdb-client"
            )

        self.config = config
        self.client: InfluxDBClient | None = None
        self.write_api = None
        self.query_api = None
        self._connected = False

        # Write buffer for batch operations
        self._write_buffer: list[Point] = []
        self._buffer_lock = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None

        # Performance metrics
        self.points_written = 0
        self.write_errors = 0
        self.flush_count = 0

        logger.info("InfluxDB storage initialized")

    async def connect(self) -> None:
        """Connect to InfluxDB."""
        try:
            # Create InfluxDB client
            self.client = InfluxDBClient(
                url=self.config.url,
                token=self.config.token,
                org=self.config.org,
                timeout=self.config.timeout_ms,
            )

            # Initialize APIs
            self.write_api = self.client.write_api(
                write_options=ASYNCHRONOUS,
                batch_size=self.config.batch_size,
                flush_interval=self.config.flush_interval_ms,
            )
            self.query_api = self.client.query_api()

            # Test connection
            await self._test_connection()

            # Create bucket if it doesn't exist
            await self._ensure_bucket_exists()

            # Start periodic flush task
            self._flush_task = asyncio.create_task(self._periodic_flush())

            self._connected = True
            logger.info(f"Connected to InfluxDB at {self.config.url}")

        except Exception as e:
            logger.error(f"Failed to connect to InfluxDB: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from InfluxDB."""
        if not self._connected:
            return

        try:
            # Stop flush task
            if self._flush_task:
                self._flush_task.cancel()
                try:
                    await self._flush_task
                except asyncio.CancelledError:
                    pass

            # Flush remaining buffer
            await self._flush_buffer()

            # Close client
            if self.write_api:
                self.write_api.close()
            if self.client:
                self.client.close()

            self._connected = False
            logger.info("Disconnected from InfluxDB")

        except Exception as e:
            logger.error(f"Error disconnecting from InfluxDB: {e}")

    @property
    def is_connected(self) -> bool:
        """Check if connected to InfluxDB."""
        return self._connected and self.client is not None

    async def _test_connection(self) -> None:
        """Test InfluxDB connection."""
        try:
            # Simple health check
            health = self.client.health()
            if health.status != "pass":
                raise RuntimeError(f"InfluxDB health check failed: {health.message}")

        except Exception as e:
            raise RuntimeError(f"InfluxDB connection test failed: {e}")

    async def _ensure_bucket_exists(self) -> None:
        """Ensure the data bucket exists."""
        try:
            buckets_api = self.client.buckets_api()

            # Check if bucket exists
            existing_buckets = buckets_api.find_buckets()
            bucket_names = [bucket.name for bucket in existing_buckets.buckets or []]

            if self.config.bucket not in bucket_names:
                # Create bucket with retention policy
                retention_rules = []
                if self.config.retention_days > 0:
                    retention_rules.append(
                        {
                            "type": "expire",
                            "everySeconds": self.config.retention_days * 24 * 3600,
                        }
                    )

                buckets_api.create_bucket(
                    bucket_name=self.config.bucket,
                    org=self.config.org,
                    retention_rules=retention_rules,
                )
                logger.info(f"Created InfluxDB bucket: {self.config.bucket}")

        except Exception as e:
            logger.warning(f"Could not ensure bucket exists: {e}")

    async def store_market_data(self, data: MarketData) -> None:
        """Store market data point."""
        if not self.is_connected:
            raise RuntimeError("Not connected to InfluxDB")

        try:
            # Create InfluxDB point
            point = self._create_market_data_point(data)

            # Add to buffer
            async with self._buffer_lock:
                self._write_buffer.append(point)

                # Flush if buffer is full
                if len(self._write_buffer) >= self.config.batch_size:
                    await self._flush_buffer()

        except Exception as e:
            logger.error(f"Error storing market data: {e}")
            self.write_errors += 1
            raise

    def _create_market_data_point(self, data: MarketData) -> Point:
        """Create InfluxDB point from market data."""
        point = (
            Point("market_data")
            .tag("symbol", data.symbol)
            .tag("source", data.source)
            .time(data.timestamp, WritePrecision.MS)
        )

        # Add price fields
        if data.bid_price is not None:
            point.field("bid_price", float(data.bid_price))
        if data.ask_price is not None:
            point.field("ask_price", float(data.ask_price))
        if data.last_price is not None:
            point.field("last_price", float(data.last_price))

        # Add size fields
        if data.bid_size is not None:
            point.field("bid_size", data.bid_size)
        if data.ask_size is not None:
            point.field("ask_size", data.ask_size)
        if data.last_size is not None:
            point.field("last_size", data.last_size)

        # Add volume and other fields
        if data.volume is not None:
            point.field("volume", data.volume)
        if data.vwap is not None:
            point.field("vwap", float(data.vwap))

        # Add calculated fields
        if data.spread is not None:
            point.field("spread", float(data.spread))
        if data.mid_price is not None:
            point.field("mid_price", float(data.mid_price))

        return point

    async def store_liquidity_signal(self, signal: LiquiditySignal) -> None:
        """Store liquidity signal."""
        if not self.is_connected:
            raise RuntimeError("Not connected to InfluxDB")

        try:
            point = (
                Point("liquidity_signals")
                .tag("symbol", signal.symbol)
                .tag("signal_type", signal.signal_type.value)
                .tag("expected_direction", signal.expected_direction.value)
                .tag("signal_id", signal.id)
                .field("strength", signal.strength)
                .field("confidence", signal.confidence)
                .field("expected_move_bps", signal.expected_move_bps)
                .field("time_horizon_seconds", signal.time_horizon_seconds)
                .time(signal.timestamp, WritePrecision.MS)
            )

            # Add features if available
            if signal.features:
                for key, value in signal.features.items():
                    point.field(f"feature_{key}", value)

            async with self._buffer_lock:
                self._write_buffer.append(point)

        except Exception as e:
            logger.error(f"Error storing liquidity signal: {e}")
            self.write_errors += 1
            raise

    async def store_position(self, position: Position) -> None:
        """Store position data."""
        if not self.is_connected:
            raise RuntimeError("Not connected to InfluxDB")

        try:
            point = (
                Point("positions")
                .tag("symbol", position.symbol)
                .tag("side", position.side.value)
                .tag("status", position.status.value)
                .tag("position_id", position.id)
                .field("quantity", position.quantity)
                .field("entry_price", float(position.entry_price))
                .time(position.entry_time, WritePrecision.MS)
            )

            # Add optional fields
            if position.current_price is not None:
                point.field("current_price", float(position.current_price))
                point.field("market_value", float(position.market_value))
                point.field("pnl_percentage", position.pnl_percentage)

            if position.target_price is not None:
                point.field("target_price", float(position.target_price))
            if position.stop_price is not None:
                point.field("stop_price", float(position.stop_price))
            if position.signal_id:
                point.tag("signal_id", position.signal_id)

            async with self._buffer_lock:
                self._write_buffer.append(point)

        except Exception as e:
            logger.error(f"Error storing position: {e}")
            self.write_errors += 1
            raise

    async def store_trade_signal(self, signal: TradeSignal) -> None:
        """Store trade signal."""
        if not self.is_connected:
            raise RuntimeError("Not connected to InfluxDB")

        try:
            point = (
                Point("trade_signals")
                .tag("symbol", signal.symbol)
                .tag("side", signal.side.value)
                .tag("status", signal.status.value)
                .tag("signal_id", signal.id)
                .tag("liquidity_signal_id", signal.liquidity_signal_id)
                .field("quantity", signal.quantity)
                .field("entry_price", float(signal.entry_price))
                .field("expected_return", signal.expected_return)
                .field("risk_score", signal.risk_score)
                .field("kelly_fraction", signal.kelly_fraction)
                .field("portfolio_allocation", signal.portfolio_allocation)
                .field("confidence", signal.confidence)
                .field("time_horizon_seconds", signal.time_horizon_seconds)
                .time(signal.timestamp, WritePrecision.MS)
            )

            # Add optional fields
            if signal.target_price is not None:
                point.field("target_price", float(signal.target_price))
            if signal.stop_price is not None:
                point.field("stop_price", float(signal.stop_price))

            async with self._buffer_lock:
                self._write_buffer.append(point)

        except Exception as e:
            logger.error(f"Error storing trade signal: {e}")
            self.write_errors += 1
            raise

    async def _flush_buffer(self) -> None:
        """Flush write buffer to InfluxDB."""
        if not self._write_buffer:
            return

        try:
            # Get current buffer and reset
            points_to_write = self._write_buffer.copy()
            self._write_buffer.clear()

            # Write points
            if points_to_write:
                self.write_api.write(
                    bucket=self.config.bucket,
                    org=self.config.org,
                    record=points_to_write,
                    write_precision=WritePrecision.MS,
                )

                self.points_written += len(points_to_write)
                self.flush_count += 1

                logger.debug(f"Flushed {len(points_to_write)} points to InfluxDB")

        except Exception as e:
            logger.error(f"Error flushing buffer to InfluxDB: {e}")
            self.write_errors += 1
            # Re-add points to buffer for retry
            self._write_buffer.extend(points_to_write)
            raise

    async def _periodic_flush(self) -> None:
        """Periodic buffer flush task."""
        while self._connected:
            try:
                await asyncio.sleep(self.config.flush_interval_ms / 1000.0)

                async with self._buffer_lock:
                    if self._write_buffer:
                        await self._flush_buffer()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic flush: {e}")

    async def get_market_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        source: str | None = None,
    ) -> list[MarketData]:
        """Query market data from InfluxDB."""
        if not self.is_connected:
            raise RuntimeError("Not connected to InfluxDB")

        try:
            # Build query
            query = f"""
                from(bucket: "{self.config.bucket}")
                |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
                |> filter(fn: (r) => r._measurement == "market_data")
                |> filter(fn: (r) => r.symbol == "{symbol}")
            """

            if source:
                query += f'|> filter(fn: (r) => r.source == "{source}")'

            query += '|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'

            # Execute query
            result = self.query_api.query(query)

            # Convert results to MarketData objects
            market_data_list = []
            for table in result:
                for record in table.records:
                    market_data = self._convert_record_to_market_data(record)
                    if market_data:
                        market_data_list.append(market_data)

            logger.info(
                f"Retrieved {len(market_data_list)} market data points for {symbol}"
            )
            return market_data_list

        except Exception as e:
            logger.error(f"Error querying market data: {e}")
            return []

    def _convert_record_to_market_data(self, record) -> MarketData | None:
        """Convert InfluxDB record to MarketData object."""
        try:
            # Extract basic fields
            symbol = record.values.get("symbol")
            timestamp = record.get_time()
            source = record.values.get("source", "unknown")

            if not symbol or not timestamp:
                return None

            # Extract price and size fields
            bid_price = record.values.get("bid_price")
            ask_price = record.values.get("ask_price")
            last_price = record.values.get("last_price")
            bid_size = record.values.get("bid_size")
            ask_size = record.values.get("ask_size")
            last_size = record.values.get("last_size")
            volume = record.values.get("volume")
            vwap = record.values.get("vwap")

            return MarketData(
                symbol=symbol,
                timestamp=timestamp,
                bid_price=Decimal(str(bid_price)) if bid_price is not None else None,
                ask_price=Decimal(str(ask_price)) if ask_price is not None else None,
                last_price=Decimal(str(last_price)) if last_price is not None else None,
                bid_size=int(bid_size) if bid_size is not None else None,
                ask_size=int(ask_size) if ask_size is not None else None,
                last_size=int(last_size) if last_size is not None else None,
                volume=int(volume) if volume is not None else None,
                vwap=Decimal(str(vwap)) if vwap is not None else None,
                source=source,
            )

        except Exception as e:
            logger.warning(f"Error converting record to MarketData: {e}")
            return None

    async def get_latest_market_data(
        self, symbol: str, source: str | None = None
    ) -> MarketData | None:
        """Get the latest market data for a symbol."""
        if not self.is_connected:
            raise RuntimeError("Not connected to InfluxDB")

        try:
            # Query for the latest data point
            query = f"""
                from(bucket: "{self.config.bucket}")
                |> range(start: -1h)
                |> filter(fn: (r) => r._measurement == "market_data")
                |> filter(fn: (r) => r.symbol == "{symbol}")
            """

            if source:
                query += f'|> filter(fn: (r) => r.source == "{source}")'

            query += """
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                |> sort(columns: ["_time"], desc: true)
                |> limit(n: 1)
            """

            result = self.query_api.query(query)

            # Convert first result to MarketData
            for table in result:
                for record in table.records:
                    return self._convert_record_to_market_data(record)

            return None

        except Exception as e:
            logger.error(f"Error getting latest market data: {e}")
            return None

    def get_storage_info(self) -> dict[str, Any]:
        """Get storage information and metrics."""
        return {
            "type": "InfluxDB",
            "is_connected": self.is_connected,
            "url": self.config.url,
            "bucket": self.config.bucket,
            "org": self.config.org,
            "points_written": self.points_written,
            "write_errors": self.write_errors,
            "flush_count": self.flush_count,
            "buffer_size": len(self._write_buffer),
            "batch_size": self.config.batch_size,
            "flush_interval_ms": self.config.flush_interval_ms,
        }

    async def store_signal(self, signal: LiquiditySignal) -> None:
        """Store liquidity signal (interface method)."""
        await self.store_liquidity_signal(signal)

    async def get_historical_signals(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> list[LiquiditySignal]:
        """Retrieve historical signals from InfluxDB."""
        if not self.is_connected:
            raise RuntimeError("Not connected to InfluxDB")

        try:
            query = f"""
                from(bucket: "{self.config.bucket}")
                |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
                |> filter(fn: (r) => r._measurement == "liquidity_signals")
                |> filter(fn: (r) => r.symbol == "{symbol}")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            """

            result = self.query_api.query(query)

            # Convert results to LiquiditySignal objects
            signals = []
            for table in result:
                for record in table.records:
                    signal = self._convert_record_to_signal(record)
                    if signal:
                        signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"Error querying historical signals: {e}")
            return []

    def _convert_record_to_signal(self, record) -> LiquiditySignal | None:
        """Convert InfluxDB record to LiquiditySignal object."""
        try:
            from ..core.models import Side, SignalType

            symbol = record.values.get("symbol")
            timestamp = record.get_time()
            signal_type_str = record.values.get("signal_type")

            if not symbol or not timestamp or not signal_type_str:
                return None

            return LiquiditySignal(
                id=record.values.get("signal_id", ""),
                symbol=symbol,
                timestamp=timestamp,
                signal_type=SignalType(signal_type_str),
                strength=record.values.get("strength", 0.0),
                confidence=record.values.get("confidence", 0.0),
                expected_direction=Side(
                    record.values.get("expected_direction", "LONG")
                ),
                expected_move_bps=record.values.get("expected_move_bps", 0.0),
                time_horizon_seconds=record.values.get("time_horizon_seconds", 0),
            )

        except Exception as e:
            logger.warning(f"Error converting record to LiquiditySignal: {e}")
            return None

    async def get_position_history(
        self, start_date: datetime, end_date: datetime
    ) -> list[Position]:
        """Retrieve position history from InfluxDB."""
        if not self.is_connected:
            raise RuntimeError("Not connected to InfluxDB")

        try:
            query = f"""
                from(bucket: "{self.config.bucket}")
                |> range(start: {start_date.isoformat()}Z, stop: {end_date.isoformat()}Z)
                |> filter(fn: (r) => r._measurement == "positions")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            """

            result = self.query_api.query(query)

            # Convert results to Position objects
            positions = []
            for table in result:
                for record in table.records:
                    position = self._convert_record_to_position(record)
                    if position:
                        positions.append(position)

            return positions

        except Exception as e:
            logger.error(f"Error querying position history: {e}")
            return []

    def _convert_record_to_position(self, record) -> Position | None:
        """Convert InfluxDB record to Position object."""
        try:
            from ..core.models import PositionStatus, Side

            symbol = record.values.get("symbol")
            entry_time = record.get_time()
            side_str = record.values.get("side")

            if not symbol or not entry_time or not side_str:
                return None

            return Position(
                id=record.values.get("position_id", ""),
                symbol=symbol,
                side=Side(side_str),
                quantity=record.values.get("quantity", 0.0),
                entry_price=Decimal(str(record.values.get("entry_price", "0"))),
                entry_time=entry_time,
                current_price=(
                    Decimal(str(record.values.get("current_price", "0")))
                    if record.values.get("current_price")
                    else None
                ),
                target_price=(
                    Decimal(str(record.values.get("target_price", "0")))
                    if record.values.get("target_price")
                    else None
                ),
                stop_price=(
                    Decimal(str(record.values.get("stop_price", "0")))
                    if record.values.get("stop_price")
                    else None
                ),
                status=PositionStatus(record.values.get("status", "OPEN")),
                signal_id=record.values.get("signal_id"),
            )

        except Exception as e:
            logger.warning(f"Error converting record to Position: {e}")
            return None

    async def cleanup_old_data(self, retention_days: int) -> int:
        """Clean up old data and return number of records deleted."""
        if not self.is_connected:
            raise RuntimeError("Not connected to InfluxDB")

        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)

            # Delete old market data
            delete_query = f"""
                from(bucket: "{self.config.bucket}")
                |> range(start: 1970-01-01T00:00:00Z, stop: {cutoff_date.isoformat()}Z)
                |> filter(fn: (r) => r._measurement == "market_data" or r._measurement == "liquidity_signals" or r._measurement == "positions" or r._measurement == "trade_signals")
                |> drop()
            """

            # Note: InfluxDB delete is different - this is a simplified example
            # In practice, you'd use the delete API or retention policies
            logger.info(f"Would delete data older than {cutoff_date}")

            return 0  # Return count of deleted records

        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return 0


# Mock implementation for testing
class MockInfluxDBStorage(DataStorage):
    """Mock InfluxDB storage for testing without database."""

    def __init__(self, config: InfluxDBConfig):
        self.config = config
        self._connected = False
        self._data_store: dict[str, list[dict]] = {
            "market_data": [],
            "liquidity_signals": [],
            "positions": [],
            "trade_signals": [],
        }

    async def connect(self) -> None:
        self._connected = True
        logger.info("Mock InfluxDB storage connected")

    async def disconnect(self) -> None:
        self._connected = False
        logger.info("Mock InfluxDB storage disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def store_market_data(self, data: MarketData) -> None:
        if not self._connected:
            raise RuntimeError("Not connected")

        self._data_store["market_data"].append(
            {
                "symbol": data.symbol,
                "timestamp": data.timestamp,
                "last_price": float(data.last_price) if data.last_price else None,
                "volume": data.volume,
                "source": data.source,
            }
        )
        logger.debug(f"Mock stored market data for {data.symbol}")

    async def store_signal(self, signal: LiquiditySignal) -> None:
        """Store liquidity signal (interface method)."""
        await self.store_liquidity_signal(signal)

    async def store_liquidity_signal(self, signal: LiquiditySignal) -> None:
        if not self._connected:
            raise RuntimeError("Not connected")

        self._data_store["liquidity_signals"].append(
            {
                "id": signal.id,
                "symbol": signal.symbol,
                "timestamp": signal.timestamp,
                "signal_type": signal.signal_type.value,
                "strength": signal.strength,
            }
        )
        logger.debug(f"Mock stored liquidity signal for {signal.symbol}")

    async def store_position(self, position: Position) -> None:
        if not self._connected:
            raise RuntimeError("Not connected")

        self._data_store["positions"].append(
            {
                "id": position.id,
                "symbol": position.symbol,
                "side": position.side.value,
                "quantity": position.quantity,
                "entry_price": float(position.entry_price),
            }
        )
        logger.debug(f"Mock stored position for {position.symbol}")

    async def store_trade_signal(self, signal: TradeSignal) -> None:
        if not self._connected:
            raise RuntimeError("Not connected")

        self._data_store["trade_signals"].append(
            {
                "id": signal.id,
                "symbol": signal.symbol,
                "side": signal.side.value,
                "quantity": signal.quantity,
                "expected_return": signal.expected_return,
            }
        )
        logger.debug(f"Mock stored trade signal for {signal.symbol}")

    async def get_market_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        source: str | None = None,
    ) -> list[MarketData]:
        # Return empty list for mock
        return []

    async def get_latest_market_data(
        self, symbol: str, source: str | None = None
    ) -> MarketData | None:
        # Return None for mock
        return None

    async def get_historical_signals(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> list[LiquiditySignal]:
        """Retrieve historical signals (mock returns empty list)."""
        return []

    async def get_position_history(
        self, start_date: datetime, end_date: datetime
    ) -> list[Position]:
        """Retrieve position history (mock returns empty list)."""
        return []

    async def cleanup_old_data(self, retention_days: int) -> int:
        """Clean up old data (mock returns 0)."""
        return 0

    def get_data_count(self, measurement: str) -> int:
        """Get count of stored data for testing."""
        return len(self._data_store.get(measurement, []))

    def get_storage_info(self) -> dict[str, Any]:
        """Get storage information and metrics (mock implementation)."""
        return {
            "type": "InfluxDB (Mock)",
            "is_connected": self.is_connected,
            "url": self.config.url,
            "bucket": self.config.bucket,
            "org": self.config.org,
            "points_written": sum(len(data) for data in self._data_store.values()),
            "write_errors": 0,
            "flush_count": 0,
            "buffer_size": 0,
            "batch_size": self.config.batch_size,
            "flush_interval_ms": self.config.flush_interval_ms,
        }


def create_influxdb_storage(
    config: InfluxDBConfig, use_mock: bool = False
) -> DataStorage:
    """Factory function to create InfluxDB storage."""
    if use_mock or not INFLUXDB_AVAILABLE:
        logger.warning("Using mock InfluxDB storage")
        return MockInfluxDBStorage(config)
    else:
        return InfluxDBStorage(config)
