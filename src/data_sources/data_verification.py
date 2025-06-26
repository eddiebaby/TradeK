"""
Data Verification Service

Compares IEX Cloud real-time data with Polygon.io end-of-day data
to ensure data quality and identify discrepancies.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from .iex_cloud_client import IEXCloudClient
from .polygon_client import PolygonClient

logger = logging.getLogger(__name__)


class DiscrepancyLevel(Enum):
    """Severity levels for data discrepancies"""

    INFO = "info"  # < 0.5% difference
    WARNING = "warning"  # 0.5% - 1% difference
    CRITICAL = "critical"  # > 1% difference


@dataclass
class DataDiscrepancy:
    """Data discrepancy between IEX and Polygon sources"""

    symbol: str
    iex_price: float
    polygon_price: float
    difference: float
    difference_percent: float
    level: DiscrepancyLevel
    timestamp: datetime
    iex_timestamp: datetime
    polygon_timestamp: datetime

    def __post_init__(self):
        """Calculate derived fields"""
        if self.polygon_price > 0:
            self.difference = self.iex_price - self.polygon_price
            self.difference_percent = (self.difference / self.polygon_price) * 100

            # Determine severity level
            abs_diff_percent = abs(self.difference_percent)
            if abs_diff_percent > 1.0:
                self.level = DiscrepancyLevel.CRITICAL
            elif abs_diff_percent > 0.5:
                self.level = DiscrepancyLevel.WARNING
            else:
                self.level = DiscrepancyLevel.INFO


@dataclass
class VerificationResult:
    """Results of data verification process"""

    symbol: str
    verified: bool
    discrepancy: DataDiscrepancy | None
    error: str | None = None


@dataclass
class VerificationSummary:
    """Summary of verification results for multiple symbols"""

    total_symbols: int
    verified_count: int
    discrepancy_count: int
    error_count: int
    critical_discrepancies: list[DataDiscrepancy]
    warning_discrepancies: list[DataDiscrepancy]
    verification_time: datetime


class DataVerificationService:
    """Service for verifying equity data between IEX and Polygon sources"""

    def __init__(
        self,
        iex_client: IEXCloudClient,
        polygon_client: PolygonClient,
        discrepancy_threshold: float = 1.0,
    ):
        """
        Initialize verification service

        Args:
            iex_client: IEX Cloud API client
            polygon_client: Polygon.io API client
            discrepancy_threshold: Percentage threshold for flagging discrepancies
        """
        self.iex_client = iex_client
        self.polygon_client = polygon_client
        self.discrepancy_threshold = discrepancy_threshold

    async def verify_symbol(self, symbol: str) -> VerificationResult:
        """Verify data for a single symbol"""
        try:
            # Get current IEX quote
            iex_quote = await self.iex_client.get_quote(symbol)

            # Get Polygon previous close (most recent EOD data)
            polygon_close = await self.polygon_client.get_previous_close(symbol)

            if not polygon_close:
                return VerificationResult(
                    symbol=symbol,
                    verified=False,
                    discrepancy=None,
                    error="No Polygon data available",
                )

            # Compare IEX latest price with Polygon previous close
            # Note: This comparison assumes we're comparing close prices
            discrepancy = DataDiscrepancy(
                symbol=symbol,
                iex_price=iex_quote.latest_price,
                polygon_price=polygon_close.close,
                difference=0,  # Will be calculated in __post_init__
                difference_percent=0,  # Will be calculated in __post_init__
                level=DiscrepancyLevel.INFO,  # Will be calculated in __post_init__
                timestamp=datetime.now(UTC),
                iex_timestamp=iex_quote.latest_time,
                polygon_timestamp=polygon_close.timestamp,
            )

            # Determine if verification passed
            verified = abs(discrepancy.difference_percent) <= self.discrepancy_threshold

            return VerificationResult(
                symbol=symbol, verified=verified, discrepancy=discrepancy
            )

        except Exception as e:
            logger.error(f"Verification failed for {symbol}: {e}")
            return VerificationResult(
                symbol=symbol, verified=False, discrepancy=None, error=str(e)
            )

    async def verify_symbols_batch(self, symbols: list[str]) -> VerificationSummary:
        """Verify data for multiple symbols"""
        logger.info(f"Starting verification for {len(symbols)} symbols")
        start_time = datetime.now(UTC)

        # Verify all symbols concurrently (with some concurrency limit)
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests

        async def verify_with_semaphore(symbol: str) -> VerificationResult:
            async with semaphore:
                return await self.verify_symbol(symbol)

        results = await asyncio.gather(
            *[verify_with_semaphore(symbol) for symbol in symbols],
            return_exceptions=True,
        )

        # Process results
        verified_count = 0
        discrepancy_count = 0
        error_count = 0
        critical_discrepancies = []
        warning_discrepancies = []

        for result in results:
            if isinstance(result, Exception):
                error_count += 1
                logger.error(f"Verification exception: {result}")
                continue

            if result.error:
                error_count += 1
            elif result.verified:
                verified_count += 1
            else:
                discrepancy_count += 1

                if result.discrepancy:
                    if result.discrepancy.level == DiscrepancyLevel.CRITICAL:
                        critical_discrepancies.append(result.discrepancy)
                    elif result.discrepancy.level == DiscrepancyLevel.WARNING:
                        warning_discrepancies.append(result.discrepancy)

        summary = VerificationSummary(
            total_symbols=len(symbols),
            verified_count=verified_count,
            discrepancy_count=discrepancy_count,
            error_count=error_count,
            critical_discrepancies=critical_discrepancies,
            warning_discrepancies=warning_discrepancies,
            verification_time=datetime.now(UTC),
        )

        logger.info(
            f"Verification complete: {verified_count}/{len(symbols)} verified, "
            f"{len(critical_discrepancies)} critical discrepancies"
        )

        return summary

    async def generate_verification_report(
        self, summary: VerificationSummary
    ) -> dict[str, Any]:
        """Generate detailed verification report"""
        report = {
            "summary": {
                "total_symbols": summary.total_symbols,
                "verified_count": summary.verified_count,
                "discrepancy_count": summary.discrepancy_count,
                "error_count": summary.error_count,
                "verification_accuracy": (
                    (summary.verified_count / summary.total_symbols * 100)
                    if summary.total_symbols > 0
                    else 0
                ),
                "verification_time": summary.verification_time.isoformat(),
            },
            "critical_discrepancies": [
                {
                    "symbol": d.symbol,
                    "iex_price": d.iex_price,
                    "polygon_price": d.polygon_price,
                    "difference_percent": round(d.difference_percent, 3),
                    "iex_timestamp": d.iex_timestamp.isoformat(),
                    "polygon_timestamp": d.polygon_timestamp.isoformat(),
                }
                for d in summary.critical_discrepancies
            ],
            "warning_discrepancies": [
                {
                    "symbol": d.symbol,
                    "iex_price": d.iex_price,
                    "polygon_price": d.polygon_price,
                    "difference_percent": round(d.difference_percent, 3),
                    "iex_timestamp": d.iex_timestamp.isoformat(),
                    "polygon_timestamp": d.polygon_timestamp.isoformat(),
                }
                for d in summary.warning_discrepancies
            ],
        }

        return report

    async def get_data_quality_metrics(self, symbols: list[str]) -> dict[str, Any]:
        """Get data quality metrics for monitoring"""
        summary = await self.verify_symbols_batch(symbols)

        metrics = {
            "data_accuracy_percent": (
                (summary.verified_count / summary.total_symbols * 100)
                if summary.total_symbols > 0
                else 0
            ),
            "critical_discrepancy_count": len(summary.critical_discrepancies),
            "warning_discrepancy_count": len(summary.warning_discrepancies),
            "error_rate_percent": (
                (summary.error_count / summary.total_symbols * 100)
                if summary.total_symbols > 0
                else 0
            ),
            "verification_timestamp": summary.verification_time.isoformat(),
        }

        return metrics


# Example usage and testing
async def test_verification_service():
    """Test the data verification service"""
    print("🧪 Testing Data Verification Service")

    # Note: Requires valid API keys
    iex_token = None  # IEX is free for basic endpoints
    polygon_key = "YOUR_POLYGON_API_KEY"

    if polygon_key == "YOUR_POLYGON_API_KEY":
        print("❌ Please set a valid Polygon API key for testing")
        return

    async with (
        IEXCloudClient(iex_token) as iex_client,
        PolygonClient(polygon_key) as polygon_client,
    ):

        verification_service = DataVerificationService(
            iex_client=iex_client,
            polygon_client=polygon_client,
            discrepancy_threshold=1.0,
        )

        # Test single symbol verification
        try:
            result = await verification_service.verify_symbol("AAPL")
            print(f"AAPL Verification: {'✅' if result.verified else '❌'}")
            if result.discrepancy:
                print(f"  IEX: ${result.discrepancy.iex_price}")
                print(f"  Polygon: ${result.discrepancy.polygon_price}")
                print(f"  Difference: {result.discrepancy.difference_percent:.3f}%")
        except Exception as e:
            print(f"Single verification test failed: {e}")

        # Test batch verification
        try:
            test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            summary = await verification_service.verify_symbols_batch(test_symbols)

            print("\nBatch Verification Results:")
            print(f"  Total: {summary.total_symbols}")
            print(f"  Verified: {summary.verified_count}")
            print(f"  Discrepancies: {summary.discrepancy_count}")
            print(f"  Errors: {summary.error_count}")
            print(f"  Critical Issues: {len(summary.critical_discrepancies)}")

        except Exception as e:
            print(f"Batch verification test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_verification_service())
