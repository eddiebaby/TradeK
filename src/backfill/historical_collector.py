"""
Historical Data Collector

Aggressive collection of historical 1-minute OHLC data from Polygon.io
with optimized chunking, rate limiting, and resumable execution.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from ..data_sources.polygon_client import PolygonClient
from .progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)


@dataclass
class DataChunk:
    """Represents a chunk of historical data to collect"""

    symbol: str
    start_date: date
    end_date: date
    chunk_id: str

    @property
    def date_range_str(self) -> str:
        """Get human-readable date range"""
        return f"{self.start_date} to {self.end_date}"


@dataclass
class CollectionResult:
    """Result of data collection for a chunk"""

    chunk: DataChunk
    success: bool
    data_points: int
    error: str | None = None
    execution_time: float = 0.0


class HistoricalCollector:
    """Aggressive historical data collector using Polygon.io"""

    def __init__(
        self,
        polygon_client: PolygonClient,
        progress_tracker: ProgressTracker,
        chunk_size_days: int = 30,
        max_retries: int = 3,
        rate_limit_buffer: float = 0.8,
    ):
        """
        Initialize historical collector

        Args:
            polygon_client: Polygon.io API client
            progress_tracker: Progress tracking instance
            chunk_size_days: Days per API call chunk
            max_retries: Maximum retry attempts per chunk
            rate_limit_buffer: Buffer factor for rate limiting (0.8 = 80% of limit)
        """
        self.polygon_client = polygon_client
        self.progress_tracker = progress_tracker
        self.chunk_size_days = chunk_size_days
        self.max_retries = max_retries

        # Rate limiting: Polygon free tier = 5 calls/minute
        # Use 4 calls/minute (80% buffer) = 15 seconds between calls
        self.call_interval = 60.0 / (5 * rate_limit_buffer)  # 15 seconds
        self.last_call_time = 0.0

        # Statistics
        self.total_chunks_processed = 0
        self.successful_chunks = 0
        self.failed_chunks = 0
        self.total_data_points = 0

    async def _wait_for_rate_limit(self):
        """Ensure we don't exceed rate limits"""
        current_time = asyncio.get_event_loop().time()
        time_since_last_call = current_time - self.last_call_time

        if time_since_last_call < self.call_interval:
            wait_time = self.call_interval - time_since_last_call
            logger.debug(f"Rate limiting: waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)

        self.last_call_time = asyncio.get_event_loop().time()

    def _generate_chunks(
        self, symbol: str, start_date: date, end_date: date
    ) -> list[DataChunk]:
        """Generate optimized chunks for data collection"""
        chunks = []
        current_date = start_date
        chunk_number = 0

        while current_date < end_date:
            chunk_end = min(
                current_date + timedelta(days=self.chunk_size_days), end_date
            )

            chunk_id = f"{symbol}_{current_date.strftime('%Y%m%d')}_{chunk_end.strftime('%Y%m%d')}"

            chunk = DataChunk(
                symbol=symbol,
                start_date=current_date,
                end_date=chunk_end,
                chunk_id=chunk_id,
            )

            chunks.append(chunk)
            current_date = chunk_end + timedelta(days=1)
            chunk_number += 1

        logger.info(
            f"Generated {len(chunks)} chunks for {symbol} ({start_date} to {end_date})"
        )
        return chunks

    async def _collect_chunk_data(self, chunk: DataChunk) -> CollectionResult:
        """Collect data for a single chunk with retries"""
        start_time = asyncio.get_event_loop().time()

        for attempt in range(self.max_retries + 1):
            try:
                await self._wait_for_rate_limit()

                logger.debug(
                    f"Collecting {chunk.symbol} {chunk.date_range_str} (attempt {attempt + 1})"
                )

                # Get historical range from Polygon
                bars = await self.polygon_client.get_historical_range(
                    ticker=chunk.symbol,
                    start_date=chunk.start_date,
                    end_date=chunk.end_date,
                )

                execution_time = asyncio.get_event_loop().time() - start_time

                if bars:
                    # Record successful collection
                    self.progress_tracker.record_api_call(True, len(bars))

                    result = CollectionResult(
                        chunk=chunk,
                        success=True,
                        data_points=len(bars),
                        execution_time=execution_time,
                    )

                    logger.info(
                        f"✅ {chunk.symbol} {chunk.date_range_str}: {len(bars)} data points"
                    )
                    return result
                else:
                    logger.warning(
                        f"No data returned for {chunk.symbol} {chunk.date_range_str}"
                    )

            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1} failed for {chunk.chunk_id}: {e}"
                )

                if attempt < self.max_retries:
                    # Exponential backoff
                    wait_time = (2**attempt) * self.call_interval
                    await asyncio.sleep(wait_time)
                else:
                    # Final failure
                    self.progress_tracker.record_api_call(False)
                    execution_time = asyncio.get_event_loop().time() - start_time

                    return CollectionResult(
                        chunk=chunk,
                        success=False,
                        data_points=0,
                        error=str(e),
                        execution_time=execution_time,
                    )

        # Should not reach here, but just in case
        return CollectionResult(
            chunk=chunk,
            success=False,
            data_points=0,
            error="Max retries exceeded",
            execution_time=asyncio.get_event_loop().time() - start_time,
        )

    async def collect_symbol_history(
        self, symbol: str, start_date: date, end_date: date, resume: bool = True
    ) -> dict[str, Any]:
        """
        Collect complete historical data for a symbol

        Args:
            symbol: Stock symbol to collect
            start_date: Start date for collection
            end_date: End date for collection
            resume: Whether to resume from previous progress

        Returns:
            Collection summary with statistics
        """
        logger.info(
            f"🚀 Starting historical collection: {symbol} ({start_date} to {end_date})"
        )

        # Generate chunks
        chunks = self._generate_chunks(symbol, start_date, end_date)

        # Check for resumable progress
        if resume:
            failed_chunks = self.progress_tracker.get_failed_chunks(symbol)
            if failed_chunks:
                logger.info(
                    f"Found {len(failed_chunks)} failed chunks to retry for {symbol}"
                )

        # Collection statistics
        total_chunks = len(chunks)
        processed_chunks = 0
        successful_chunks = 0
        total_data_points = 0
        failed_chunk_ids = []

        # Process chunks sequentially (required for rate limiting)
        for i, chunk in enumerate(chunks, 1):
            logger.info(
                f"📊 Processing chunk {i}/{total_chunks}: {chunk.symbol} {chunk.date_range_str}"
            )

            result = await self._collect_chunk_data(chunk)
            processed_chunks += 1

            if result.success:
                successful_chunks += 1
                total_data_points += result.data_points

                # Update progress tracker
                days_in_chunk = (chunk.end_date - chunk.start_date).days
                self.progress_tracker.update_symbol_progress(
                    symbol=symbol,
                    current_date=chunk.end_date,
                    data_points_collected=result.data_points,
                    days_completed=days_in_chunk,
                )

            else:
                failed_chunk_ids.append(chunk.chunk_id)
                self.progress_tracker.record_failed_chunk(symbol, chunk.chunk_id)
                logger.error(f"❌ Failed chunk: {chunk.chunk_id} - {result.error}")

            # Progress update every 10 chunks
            if i % 10 == 0:
                progress_pct = (successful_chunks / total_chunks) * 100
                logger.info(
                    f"🏃 Progress: {successful_chunks}/{total_chunks} chunks ({progress_pct:.1f}%)"
                )

        # Final summary
        success_rate = (
            (successful_chunks / total_chunks) * 100 if total_chunks > 0 else 0
        )

        summary = {
            "symbol": symbol,
            "date_range": f"{start_date} to {end_date}",
            "total_chunks": total_chunks,
            "successful_chunks": successful_chunks,
            "failed_chunks": len(failed_chunk_ids),
            "success_rate": f"{success_rate:.1f}%",
            "total_data_points": total_data_points,
            "failed_chunk_ids": failed_chunk_ids,
        }

        logger.info(
            f"✅ Completed {symbol}: {successful_chunks}/{total_chunks} chunks, "
            f"{total_data_points} data points, {success_rate:.1f}% success rate"
        )

        return summary

    async def collect_multiple_symbols(
        self, symbols: list[str], start_date: date, end_date: date, resume: bool = True
    ) -> dict[str, Any]:
        """
        Collect historical data for multiple symbols

        Args:
            symbols: List of symbols to collect
            start_date: Start date for collection
            end_date: End date for collection
            resume: Whether to resume from previous progress

        Returns:
            Overall collection summary
        """
        logger.info(f"🎯 Starting multi-symbol collection: {symbols}")

        overall_start_time = datetime.now(UTC)
        symbol_summaries = {}
        overall_data_points = 0
        overall_success_rate = 0

        # Process symbols sequentially to respect rate limits
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"📈 Processing symbol {i}/{len(symbols)}: {symbol}")

            try:
                summary = await self.collect_symbol_history(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    resume=resume,
                )

                symbol_summaries[symbol] = summary
                overall_data_points += summary["total_data_points"]
                overall_success_rate += float(summary["success_rate"].rstrip("%"))

            except Exception as e:
                logger.error(f"Failed to collect {symbol}: {e}")
                symbol_summaries[symbol] = {
                    "error": str(e),
                    "total_data_points": 0,
                    "success_rate": "0%",
                }

        # Calculate overall statistics
        overall_end_time = datetime.now(UTC)
        execution_time = (overall_end_time - overall_start_time).total_seconds()
        avg_success_rate = overall_success_rate / len(symbols) if symbols else 0

        overall_summary = {
            "execution_summary": {
                "symbols_processed": len(symbols),
                "start_time": overall_start_time.isoformat(),
                "end_time": overall_end_time.isoformat(),
                "execution_time_minutes": execution_time / 60,
                "total_data_points": overall_data_points,
                "average_success_rate": f"{avg_success_rate:.1f}%",
            },
            "symbol_details": symbol_summaries,
        }

        logger.info(
            f"🎉 Multi-symbol collection complete: {len(symbols)} symbols, "
            f"{overall_data_points} total data points, "
            f"{execution_time/60:.1f} minutes"
        )

        return overall_summary

    def get_collection_stats(self) -> dict[str, Any]:
        """Get current collection statistics"""
        return {
            "total_chunks_processed": self.total_chunks_processed,
            "successful_chunks": self.successful_chunks,
            "failed_chunks": self.failed_chunks,
            "success_rate": f"{(self.successful_chunks/max(self.total_chunks_processed,1))*100:.1f}%",
            "total_data_points": self.total_data_points,
            "rate_limit_interval": self.call_interval,
        }


# Example usage and testing
async def test_historical_collector():
    """Test the historical collector"""
    print("🧪 Testing Historical Collector")

    # Note: Requires valid Polygon API key
    polygon_key = "YOUR_POLYGON_API_KEY"

    if polygon_key == "YOUR_POLYGON_API_KEY":
        print("❌ Please set a valid Polygon API key for testing")
        return

    from .progress_tracker import ProgressTracker

    async with PolygonClient(polygon_key) as polygon_client:
        progress_tracker = ProgressTracker("test_backfill_progress")

        collector = HistoricalCollector(
            polygon_client=polygon_client,
            progress_tracker=progress_tracker,
            chunk_size_days=7,  # Small chunks for testing
        )

        # Test single symbol collection
        test_start = date(2024, 1, 1)
        test_end = date(2024, 1, 31)

        summary = await collector.collect_symbol_history(
            symbol="SPY", start_date=test_start, end_date=test_end
        )

        print("Collection Summary:")
        print(f"  Symbol: {summary['symbol']}")
        print(f"  Success Rate: {summary['success_rate']}")
        print(f"  Data Points: {summary['total_data_points']}")


if __name__ == "__main__":
    asyncio.run(test_historical_collector())
