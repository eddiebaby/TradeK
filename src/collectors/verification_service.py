"""
Verification Service

Automated service that runs daily verification of equity data
by comparing IEX real-time closes with Polygon end-of-day data.
"""

import asyncio
import logging
from datetime import datetime
from datetime import time as dt_time

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

from ..core.config import get_settings
from ..data_sources.data_verification import (
    DataVerificationService,
    VerificationSummary,
)
from ..data_sources.iex_cloud_client import IEXCloudClient
from ..data_sources.polygon_client import PolygonClient

logger = logging.getLogger(__name__)


class VerificationService:
    """Automated verification service for equity data quality"""

    def __init__(
        self,
        symbols: list[str],
        verification_time: dt_time = dt_time(16, 30),  # 4:30 PM ET
        influxdb_url: str | None = None,
        influxdb_token: str | None = None,
        influxdb_org: str | None = None,
        influxdb_bucket: str | None = None,
        iex_token: str | None = None,
        polygon_key: str | None = None,
    ):
        """
        Initialize verification service

        Args:
            symbols: List of symbols to verify
            verification_time: Daily time to run verification
            influxdb_*: InfluxDB connection parameters
            iex_token: IEX Cloud API token
            polygon_key: Polygon.io API key
        """
        self.symbols = [s.upper() for s in symbols]
        self.verification_time = verification_time
        self.is_running = False
        self.verification_task: asyncio.Task | None = None

        # Configuration
        settings = get_settings()
        self.influxdb_url = influxdb_url or settings.influxdb_url
        self.influxdb_token = influxdb_token or settings.influxdb_token
        self.influxdb_org = influxdb_org or settings.influxdb_org
        self.influxdb_bucket = influxdb_bucket or settings.influxdb_bucket
        self.iex_token = iex_token or getattr(settings, "iex_cloud_api_token", None)
        self.polygon_key = polygon_key or getattr(settings, "polygon_api_key", None)

        # InfluxDB client
        self.influx_client = InfluxDBClient(
            url=self.influxdb_url, token=self.influxdb_token, org=self.influxdb_org
        )
        self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)

        # API clients (will be initialized in async context)
        self.iex_client: IEXCloudClient | None = None
        self.polygon_client: PolygonClient | None = None
        self.verification_service: DataVerificationService | None = None

    async def __aenter__(self):
        """Async context manager entry"""
        self.iex_client = IEXCloudClient(self.iex_token)
        await self.iex_client.__aenter__()

        if self.polygon_key:
            self.polygon_client = PolygonClient(self.polygon_key)
            await self.polygon_client.__aenter__()

            self.verification_service = DataVerificationService(
                iex_client=self.iex_client, polygon_client=self.polygon_client
            )
        else:
            logger.warning("Polygon API key not provided, verification will be limited")

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()

        if self.iex_client:
            await self.iex_client.__aexit__(exc_type, exc_val, exc_tb)
        if self.polygon_client:
            await self.polygon_client.__aexit__(exc_type, exc_val, exc_tb)

        self.influx_client.close()

    def _create_verification_points(self, summary: VerificationSummary) -> list[Point]:
        """Create InfluxDB points from verification summary"""
        points = []

        # Overall verification metrics
        metrics_point = (
            Point("data_verification")
            .tag("type", "summary")
            .field("total_symbols", summary.total_symbols)
            .field("verified_count", summary.verified_count)
            .field("discrepancy_count", summary.discrepancy_count)
            .field("error_count", summary.error_count)
            .field(
                "accuracy_percent",
                (
                    (summary.verified_count / summary.total_symbols * 100)
                    if summary.total_symbols > 0
                    else 0
                ),
            )
            .time(summary.verification_time)
        )

        points.append(metrics_point)

        # Individual discrepancy points
        for discrepancy in (
            summary.critical_discrepancies + summary.warning_discrepancies
        ):
            discrepancy_point = (
                Point("data_verification")
                .tag("type", "discrepancy")
                .tag("symbol", discrepancy.symbol)
                .tag("level", discrepancy.level.value)
                .field("iex_price", discrepancy.iex_price)
                .field("polygon_price", discrepancy.polygon_price)
                .field("difference", discrepancy.difference)
                .field("difference_percent", discrepancy.difference_percent)
                .time(discrepancy.timestamp)
            )

            points.append(discrepancy_point)

        return points

    async def run_verification(self) -> VerificationSummary | None:
        """Run verification for all symbols"""
        if not self.verification_service:
            logger.error(
                "Verification service not initialized (missing Polygon API key?)"
            )
            return None

        logger.info(f"Starting verification for {len(self.symbols)} symbols")

        try:
            # Run verification
            summary = await self.verification_service.verify_symbols_batch(self.symbols)

            # Store results in InfluxDB
            points = self._create_verification_points(summary)
            if points:
                self.write_api.write(
                    bucket=self.influxdb_bucket, org=self.influxdb_org, record=points
                )
                logger.info(f"Stored {len(points)} verification data points")

            # Log summary
            logger.info(
                f"Verification complete: {summary.verified_count}/{summary.total_symbols} verified, "
                f"{len(summary.critical_discrepancies)} critical discrepancies, "
                f"{len(summary.warning_discrepancies)} warnings"
            )

            # Log critical discrepancies
            for discrepancy in summary.critical_discrepancies:
                logger.warning(
                    f"CRITICAL: {discrepancy.symbol} - IEX: ${discrepancy.iex_price}, "
                    f"Polygon: ${discrepancy.polygon_price}, "
                    f"Diff: {discrepancy.difference_percent:.3f}%"
                )

            return summary

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return None

    def _time_until_next_verification(self) -> float:
        """Calculate seconds until next verification time"""
        now = datetime.now()
        today_verification = datetime.combine(now.date(), self.verification_time)

        # If today's verification time has passed, schedule for tomorrow
        if now > today_verification:
            from datetime import timedelta

            next_verification = today_verification + timedelta(days=1)
        else:
            next_verification = today_verification

        return (next_verification - now).total_seconds()

    async def _verification_loop(self):
        """Main verification loop"""
        logger.info(f"Starting daily verification service at {self.verification_time}")

        while self.is_running:
            try:
                # Calculate time until next verification
                wait_seconds = self._time_until_next_verification()

                # If it's less than 1 minute, run verification now
                if wait_seconds < 60:
                    await self.run_verification()
                    # After running, wait until tomorrow
                    wait_seconds = self._time_until_next_verification()

                logger.info(f"Next verification in {wait_seconds/3600:.1f} hours")

                # Wait with periodic checks (every hour)
                while wait_seconds > 0 and self.is_running:
                    sleep_time = min(wait_seconds, 3600)  # Max 1 hour
                    await asyncio.sleep(sleep_time)
                    wait_seconds -= sleep_time

            except asyncio.CancelledError:
                logger.info("Verification loop cancelled")
                break
            except Exception as e:
                logger.error(f"Verification loop error: {e}")
                # Wait 10 minutes before retrying
                await asyncio.sleep(600)

    async def start(self):
        """Start automated verification service"""
        if self.is_running:
            logger.warning("Verification service already running")
            return

        if not self.verification_service:
            logger.error("Cannot start verification service without Polygon API key")
            return

        logger.info("Starting equity data verification service")
        self.is_running = True
        self.verification_task = asyncio.create_task(self._verification_loop())

    async def stop(self):
        """Stop verification service"""
        if not self.is_running:
            return

        logger.info("Stopping verification service")
        self.is_running = False

        if self.verification_task:
            self.verification_task.cancel()
            try:
                await self.verification_task
            except asyncio.CancelledError:
                pass
            self.verification_task = None


# Example usage and testing
async def test_verification_service():
    """Test the verification service"""
    print("🧪 Testing Verification Service")

    test_symbols = ["AAPL", "MSFT", "GOOGL"]

    async with VerificationService(
        symbols=test_symbols, verification_time=dt_time(16, 30)  # 4:30 PM
    ) as service:

        # Test single verification run
        try:
            summary = await service.run_verification()
            if summary:
                print("Verification Results:")
                print(f"  Total: {summary.total_symbols}")
                print(f"  Verified: {summary.verified_count}")
                print(f"  Discrepancies: {summary.discrepancy_count}")
                print(f"  Critical: {len(summary.critical_discrepancies)}")
                print(f"  Warnings: {len(summary.warning_discrepancies)}")
            else:
                print("Verification could not be completed (check API keys)")
        except Exception as e:
            print(f"Verification test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_verification_service())
