"""
Progress Tracker for Historical Data Backfill

Provides resumable execution, progress monitoring, and state persistence
for long-running backfill operations.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BackfillProgress:
    """Progress tracking for backfill operations"""

    symbol: str
    start_date: date
    end_date: date
    current_date: date
    total_days: int
    completed_days: int
    total_data_points: int
    failed_chunks: list[str]
    last_update: datetime

    @property
    def progress_percent(self) -> float:
        """Calculate completion percentage"""
        if self.total_days == 0:
            return 0.0
        return (self.completed_days / self.total_days) * 100

    @property
    def remaining_days(self) -> int:
        """Calculate remaining days"""
        return max(0, self.total_days - self.completed_days)


@dataclass
class BackfillStats:
    """Overall backfill statistics"""

    start_time: datetime
    symbols: list[str]
    total_api_calls: int
    successful_calls: int
    failed_calls: int
    total_data_points: int
    average_points_per_call: float
    estimated_completion: datetime | None = None

    @property
    def success_rate(self) -> float:
        """Calculate API call success rate"""
        if self.total_api_calls == 0:
            return 0.0
        return (self.successful_calls / self.total_api_calls) * 100

    @property
    def elapsed_time(self) -> float:
        """Calculate elapsed time in seconds"""
        return (datetime.now(UTC) - self.start_time).total_seconds()


class ProgressTracker:
    """Track and persist backfill progress"""

    def __init__(self, progress_dir: str = "data/backfill_progress"):
        """
        Initialize progress tracker

        Args:
            progress_dir: Directory to store progress files
        """
        self.progress_dir = Path(progress_dir)
        self.progress_dir.mkdir(parents=True, exist_ok=True)

        self.symbol_progress: dict[str, BackfillProgress] = {}
        self.stats: BackfillStats | None = None

    def start_backfill(self, symbols: list[str], start_date: date, end_date: date):
        """Initialize backfill tracking"""
        self.stats = BackfillStats(
            start_time=datetime.now(UTC),
            symbols=symbols,
            total_api_calls=0,
            successful_calls=0,
            failed_calls=0,
            total_data_points=0,
            average_points_per_call=0.0,
        )

        # Calculate total days (trading days approximation)
        total_days = (end_date - start_date).days
        trading_days = int(total_days * 0.72)  # Approximate 5/7 ratio

        for symbol in symbols:
            self.symbol_progress[symbol] = BackfillProgress(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                current_date=start_date,
                total_days=trading_days,
                completed_days=0,
                total_data_points=0,
                failed_chunks=[],
                last_update=datetime.now(UTC),
            )

        self._save_progress()
        logger.info(f"Started backfill tracking for {len(symbols)} symbols")

    def update_symbol_progress(
        self,
        symbol: str,
        current_date: date,
        data_points_collected: int,
        days_completed: int = 1,
    ):
        """Update progress for a specific symbol"""
        if symbol not in self.symbol_progress:
            logger.warning(f"Symbol {symbol} not in progress tracking")
            return

        progress = self.symbol_progress[symbol]
        progress.current_date = current_date
        progress.completed_days += days_completed
        progress.total_data_points += data_points_collected
        progress.last_update = datetime.now(UTC)

        if self.stats:
            self.stats.total_data_points += data_points_collected

        # Save progress periodically (every 10 updates)
        if progress.completed_days % 10 == 0:
            self._save_progress()

        logger.debug(f"Updated {symbol}: {progress.progress_percent:.1f}% complete")

    def record_api_call(self, success: bool, data_points: int = 0):
        """Record API call statistics"""
        if not self.stats:
            return

        self.stats.total_api_calls += 1
        if success:
            self.stats.successful_calls += 1
            if data_points > 0:
                # Update average
                total_points = self.stats.total_data_points + data_points
                self.stats.average_points_per_call = (
                    total_points / self.stats.successful_calls
                )
        else:
            self.stats.failed_calls += 1

    def record_failed_chunk(self, symbol: str, chunk_identifier: str):
        """Record a failed chunk for retry"""
        if symbol in self.symbol_progress:
            self.symbol_progress[symbol].failed_chunks.append(chunk_identifier)
            self._save_progress()
            logger.warning(f"Recorded failed chunk for {symbol}: {chunk_identifier}")

    def get_failed_chunks(self, symbol: str) -> list[str]:
        """Get list of failed chunks for retry"""
        if symbol in self.symbol_progress:
            return self.symbol_progress[symbol].failed_chunks.copy()
        return []

    def clear_failed_chunks(self, symbol: str):
        """Clear failed chunks after successful retry"""
        if symbol in self.symbol_progress:
            self.symbol_progress[symbol].failed_chunks.clear()
            self._save_progress()

    def estimate_completion(self) -> datetime | None:
        """Estimate completion time based on current progress"""
        if not self.stats or self.stats.total_api_calls == 0:
            return None

        # Calculate average time per API call
        elapsed = self.stats.elapsed_time
        avg_time_per_call = elapsed / self.stats.total_api_calls

        # Estimate remaining calls (rough approximation)
        total_progress = sum(p.progress_percent for p in self.symbol_progress.values())
        avg_progress = (
            total_progress / len(self.symbol_progress) if self.symbol_progress else 0
        )

        if avg_progress > 0:
            remaining_progress = 100 - avg_progress
            estimated_remaining_calls = (
                self.stats.total_api_calls * remaining_progress
            ) / avg_progress
            estimated_remaining_time = estimated_remaining_calls * avg_time_per_call

            completion_time = datetime.now(UTC)
            completion_time = completion_time.replace(
                second=int(completion_time.second + estimated_remaining_time)
            )

            return completion_time

        return None

    def get_progress_summary(self) -> dict[str, Any]:
        """Get comprehensive progress summary"""
        summary = {
            "overall": {
                "symbols": list(self.symbol_progress.keys()),
                "elapsed_time_minutes": (
                    self.stats.elapsed_time / 60 if self.stats else 0
                ),
                "total_api_calls": self.stats.total_api_calls if self.stats else 0,
                "success_rate": (
                    f"{self.stats.success_rate:.1f}%" if self.stats else "0%"
                ),
                "total_data_points": self.stats.total_data_points if self.stats else 0,
                "estimated_completion": self.estimate_completion(),
            },
            "symbols": {},
        }

        for symbol, progress in self.symbol_progress.items():
            summary["symbols"][symbol] = {
                "progress_percent": f"{progress.progress_percent:.1f}%",
                "completed_days": progress.completed_days,
                "total_days": progress.total_days,
                "remaining_days": progress.remaining_days,
                "current_date": progress.current_date.isoformat(),
                "data_points": progress.total_data_points,
                "failed_chunks": len(progress.failed_chunks),
                "last_update": progress.last_update.isoformat(),
            }

        return summary

    def _save_progress(self):
        """Save progress to disk"""
        try:
            # Save symbol progress
            for symbol, progress in self.symbol_progress.items():
                progress_file = self.progress_dir / f"{symbol}_progress.json"
                with open(progress_file, "w") as f:
                    # Convert dataclass to dict and handle date serialization
                    progress_dict = asdict(progress)
                    progress_dict["start_date"] = progress.start_date.isoformat()
                    progress_dict["end_date"] = progress.end_date.isoformat()
                    progress_dict["current_date"] = progress.current_date.isoformat()
                    progress_dict["last_update"] = progress.last_update.isoformat()
                    json.dump(progress_dict, f, indent=2)

            # Save overall stats
            if self.stats:
                stats_file = self.progress_dir / "backfill_stats.json"
                with open(stats_file, "w") as f:
                    stats_dict = asdict(self.stats)
                    stats_dict["start_time"] = self.stats.start_time.isoformat()
                    if self.stats.estimated_completion:
                        stats_dict["estimated_completion"] = (
                            self.stats.estimated_completion.isoformat()
                        )
                    json.dump(stats_dict, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save progress: {e}")

    def load_progress(self, symbols: list[str]) -> bool:
        """Load existing progress from disk"""
        try:
            loaded_symbols = 0

            for symbol in symbols:
                progress_file = self.progress_dir / f"{symbol}_progress.json"
                if progress_file.exists():
                    with open(progress_file) as f:
                        progress_dict = json.load(f)

                    # Convert back to proper types
                    progress = BackfillProgress(
                        symbol=progress_dict["symbol"],
                        start_date=date.fromisoformat(progress_dict["start_date"]),
                        end_date=date.fromisoformat(progress_dict["end_date"]),
                        current_date=date.fromisoformat(progress_dict["current_date"]),
                        total_days=progress_dict["total_days"],
                        completed_days=progress_dict["completed_days"],
                        total_data_points=progress_dict["total_data_points"],
                        failed_chunks=progress_dict["failed_chunks"],
                        last_update=datetime.fromisoformat(
                            progress_dict["last_update"]
                        ),
                    )

                    self.symbol_progress[symbol] = progress
                    loaded_symbols += 1

            # Load stats
            stats_file = self.progress_dir / "backfill_stats.json"
            if stats_file.exists():
                with open(stats_file) as f:
                    stats_dict = json.load(f)

                self.stats = BackfillStats(
                    start_time=datetime.fromisoformat(stats_dict["start_time"]),
                    symbols=stats_dict["symbols"],
                    total_api_calls=stats_dict["total_api_calls"],
                    successful_calls=stats_dict["successful_calls"],
                    failed_calls=stats_dict["failed_calls"],
                    total_data_points=stats_dict["total_data_points"],
                    average_points_per_call=stats_dict["average_points_per_call"],
                    estimated_completion=(
                        datetime.fromisoformat(stats_dict["estimated_completion"])
                        if stats_dict.get("estimated_completion")
                        else None
                    ),
                )

            if loaded_symbols > 0:
                logger.info(f"Loaded progress for {loaded_symbols} symbols")
                return True

        except Exception as e:
            logger.error(f"Failed to load progress: {e}")

        return False

    def is_complete(self) -> bool:
        """Check if backfill is complete"""
        if not self.symbol_progress:
            return False

        return all(
            progress.completed_days >= progress.total_days
            for progress in self.symbol_progress.values()
        )

    def cleanup_progress_files(self):
        """Clean up progress files after successful completion"""
        try:
            for file in self.progress_dir.glob("*.json"):
                file.unlink()
            logger.info("Cleaned up progress files")
        except Exception as e:
            logger.error(f"Failed to cleanup progress files: {e}")


# Example usage and testing
def test_progress_tracker():
    """Test the progress tracker"""
    print("🧪 Testing Progress Tracker")

    tracker = ProgressTracker("test_progress")

    # Test initialization
    symbols = ["SPY", "QQQ"]
    start_date = date(2022, 1, 1)
    end_date = date(2024, 12, 19)

    tracker.start_backfill(symbols, start_date, end_date)
    print("✅ Backfill tracking initialized")

    # Test progress updates
    tracker.update_symbol_progress("SPY", date(2022, 1, 15), 1000, 10)
    tracker.record_api_call(True, 1000)

    tracker.update_symbol_progress("QQQ", date(2022, 1, 10), 800, 8)
    tracker.record_api_call(True, 800)

    # Test summary
    summary = tracker.get_progress_summary()
    print("Progress Summary:")
    print(f"  SPY: {summary['symbols']['SPY']['progress_percent']}")
    print(f"  QQQ: {summary['symbols']['QQQ']['progress_percent']}")
    print(f"  Total API calls: {summary['overall']['total_api_calls']}")
    print(f"  Success rate: {summary['overall']['success_rate']}")


if __name__ == "__main__":
    test_progress_tracker()
