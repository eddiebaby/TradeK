"""
Data Validator for Historical Backfill

Validates data quality, detects gaps, and ensures data integrity
for historical equity data collection.
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime
from datetime import time as dt_time
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Data validation issue"""

    level: ValidationLevel
    issue_type: str
    description: str
    symbol: str
    timestamp: datetime | None = None
    value: float | None = None
    expected_value: float | None = None


@dataclass
class GapInfo:
    """Information about a data gap"""

    symbol: str
    gap_start: datetime
    gap_end: datetime
    expected_points: int
    gap_duration_minutes: int

    @property
    def is_during_market_hours(self) -> bool:
        """Check if gap occurs during market hours"""
        # Simple market hours check (9:30 AM - 4:00 PM ET)
        market_open = dt_time(9, 30)
        market_close = dt_time(16, 0)

        start_time = self.gap_start.time()
        end_time = self.gap_end.time()

        # Check if any part of the gap is during market hours
        return (
            market_open <= start_time <= market_close
            or market_open <= end_time <= market_close
        )


@dataclass
class ValidationReport:
    """Comprehensive validation report"""

    symbol: str
    total_data_points: int
    validation_issues: list[ValidationIssue]
    detected_gaps: list[GapInfo]
    quality_score: float  # 0-100
    completeness_score: float  # 0-100

    @property
    def critical_issues_count(self) -> int:
        """Count of critical validation issues"""
        return len(
            [
                issue
                for issue in self.validation_issues
                if issue.level == ValidationLevel.CRITICAL
            ]
        )

    @property
    def error_issues_count(self) -> int:
        """Count of error validation issues"""
        return len(
            [
                issue
                for issue in self.validation_issues
                if issue.level == ValidationLevel.ERROR
            ]
        )

    @property
    def market_hours_gaps_count(self) -> int:
        """Count of gaps during market hours"""
        return len([gap for gap in self.detected_gaps if gap.is_during_market_hours])


class DataValidator:
    """Comprehensive data validator for historical equity data"""

    def __init__(self):
        """Initialize data validator with default thresholds"""
        self.price_change_threshold = 0.20  # 20% max price change per minute
        self.volume_threshold_multiplier = 10.0  # 10x average volume
        self.expected_minutes_per_day = 390  # 6.5 hours × 60 minutes

    def validate_ohlc_consistency(
        self, data_point: dict[str, Any]
    ) -> list[ValidationIssue]:
        """Validate OHLC price consistency"""
        issues = []
        symbol = data_point.get("symbol", "UNKNOWN")
        timestamp = data_point.get("timestamp")

        try:
            open_price = float(data_point.get("open", 0))
            high_price = float(data_point.get("high", 0))
            low_price = float(data_point.get("low", 0))
            close_price = float(data_point.get("close", 0))

            # Check OHLC relationships
            if high_price < max(open_price, close_price):
                issues.append(
                    ValidationIssue(
                        level=ValidationLevel.ERROR,
                        issue_type="ohlc_inconsistency",
                        description=f"High price ({high_price}) less than max(open, close)",
                        symbol=symbol,
                        timestamp=timestamp,
                        value=high_price,
                        expected_value=max(open_price, close_price),
                    )
                )

            if low_price > min(open_price, close_price):
                issues.append(
                    ValidationIssue(
                        level=ValidationLevel.ERROR,
                        issue_type="ohlc_inconsistency",
                        description=f"Low price ({low_price}) greater than min(open, close)",
                        symbol=symbol,
                        timestamp=timestamp,
                        value=low_price,
                        expected_value=min(open_price, close_price),
                    )
                )

            # Check for zero or negative prices
            for price_type, price_value in [
                ("open", open_price),
                ("high", high_price),
                ("low", low_price),
                ("close", close_price),
            ]:
                if price_value <= 0:
                    issues.append(
                        ValidationIssue(
                            level=ValidationLevel.CRITICAL,
                            issue_type="invalid_price",
                            description=f"{price_type.title()} price is zero or negative: {price_value}",
                            symbol=symbol,
                            timestamp=timestamp,
                            value=price_value,
                        )
                    )

        except (ValueError, TypeError) as e:
            issues.append(
                ValidationIssue(
                    level=ValidationLevel.CRITICAL,
                    issue_type="data_format_error",
                    description=f"Failed to parse OHLC data: {e}",
                    symbol=symbol,
                    timestamp=timestamp,
                )
            )

        return issues

    def validate_volume(
        self, data_point: dict[str, Any], average_volume: float | None = None
    ) -> list[ValidationIssue]:
        """Validate volume data"""
        issues = []
        symbol = data_point.get("symbol", "UNKNOWN")
        timestamp = data_point.get("timestamp")

        try:
            volume = float(data_point.get("volume", 0))

            # Check for negative volume
            if volume < 0:
                issues.append(
                    ValidationIssue(
                        level=ValidationLevel.ERROR,
                        issue_type="invalid_volume",
                        description=f"Negative volume: {volume}",
                        symbol=symbol,
                        timestamp=timestamp,
                        value=volume,
                    )
                )

            # Check for extremely high volume (if average is available)
            if average_volume and volume > (
                average_volume * self.volume_threshold_multiplier
            ):
                issues.append(
                    ValidationIssue(
                        level=ValidationLevel.WARNING,
                        issue_type="unusual_volume",
                        description=f"Volume {volume} is {volume/average_volume:.1f}x average",
                        symbol=symbol,
                        timestamp=timestamp,
                        value=volume,
                        expected_value=average_volume,
                    )
                )

        except (ValueError, TypeError) as e:
            issues.append(
                ValidationIssue(
                    level=ValidationLevel.ERROR,
                    issue_type="data_format_error",
                    description=f"Failed to parse volume data: {e}",
                    symbol=symbol,
                    timestamp=timestamp,
                )
            )

        return issues

    def validate_price_continuity(
        self, data_points: list[dict[str, Any]]
    ) -> list[ValidationIssue]:
        """Validate price continuity between consecutive data points"""
        issues = []

        if len(data_points) < 2:
            return issues

        # Sort by timestamp
        sorted_points = sorted(
            data_points, key=lambda x: x.get("timestamp", datetime.min)
        )

        for i in range(1, len(sorted_points)):
            prev_point = sorted_points[i - 1]
            curr_point = sorted_points[i]

            try:
                prev_close = float(prev_point.get("close", 0))
                curr_open = float(curr_point.get("open", 0))

                if prev_close > 0 and curr_open > 0:
                    price_change = abs(curr_open - prev_close) / prev_close

                    if price_change > self.price_change_threshold:
                        issues.append(
                            ValidationIssue(
                                level=ValidationLevel.WARNING,
                                issue_type="large_price_gap",
                                description=f"Large price gap: {price_change:.2%} change from {prev_close} to {curr_open}",
                                symbol=curr_point.get("symbol", "UNKNOWN"),
                                timestamp=curr_point.get("timestamp"),
                                value=curr_open,
                                expected_value=prev_close,
                            )
                        )

            except (ValueError, TypeError):
                continue  # Skip invalid data points

        return issues

    def detect_gaps(
        self, data_points: list[dict[str, Any]], symbol: str
    ) -> list[GapInfo]:
        """Detect time gaps in the data"""
        gaps = []

        if len(data_points) < 2:
            return gaps

        # Sort by timestamp
        sorted_points = sorted(
            data_points, key=lambda x: x.get("timestamp", datetime.min)
        )

        for i in range(1, len(sorted_points)):
            prev_timestamp = sorted_points[i - 1].get("timestamp")
            curr_timestamp = sorted_points[i].get("timestamp")

            if prev_timestamp and curr_timestamp:
                time_diff = curr_timestamp - prev_timestamp
                minutes_diff = time_diff.total_seconds() / 60

                # Consider gaps > 5 minutes during market hours
                if minutes_diff > 5:
                    # Check if this is during market hours
                    expected_points = max(
                        1, int(minutes_diff) - 1
                    )  # -1 for normal 1-minute interval

                    gap = GapInfo(
                        symbol=symbol,
                        gap_start=prev_timestamp,
                        gap_end=curr_timestamp,
                        expected_points=expected_points,
                        gap_duration_minutes=int(minutes_diff),
                    )

                    gaps.append(gap)

        return gaps

    def calculate_completeness_score(
        self, data_points: list[dict[str, Any]], start_date: date, end_date: date
    ) -> float:
        """Calculate data completeness score (0-100)"""
        try:
            # Calculate expected trading days
            total_days = (end_date - start_date).days
            expected_trading_days = int(total_days * 0.72)  # ~252 trading days per year

            # Calculate expected data points (390 minutes per trading day)
            expected_points = expected_trading_days * self.expected_minutes_per_day

            if expected_points == 0:
                return 100.0

            actual_points = len(data_points)
            completeness = min(100.0, (actual_points / expected_points) * 100)

            return completeness

        except Exception as e:
            logger.error(f"Failed to calculate completeness score: {e}")
            return 0.0

    def calculate_quality_score(self, issues: list[ValidationIssue]) -> float:
        """Calculate data quality score (0-100) based on validation issues"""
        if not issues:
            return 100.0

        # Weight different issue levels
        weights = {
            ValidationLevel.INFO: 0.1,
            ValidationLevel.WARNING: 1.0,
            ValidationLevel.ERROR: 5.0,
            ValidationLevel.CRITICAL: 10.0,
        }

        total_penalty = sum(weights.get(issue.level, 1.0) for issue in issues)

        # Calculate score (max penalty of 100 gives score of 0)
        quality_score = max(0.0, 100.0 - total_penalty)

        return quality_score

    def validate_dataset(
        self,
        data_points: list[dict[str, Any]],
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> ValidationReport:
        """Perform comprehensive validation of a dataset"""
        logger.info(f"🔍 Validating dataset: {symbol} ({len(data_points)} data points)")

        all_issues = []

        # Calculate average volume for volume validation
        volumes = [
            float(dp.get("volume", 0))
            for dp in data_points
            if dp.get("volume") is not None
        ]
        avg_volume = sum(volumes) / len(volumes) if volumes else None

        # Validate individual data points
        for data_point in data_points:
            # OHLC validation
            ohlc_issues = self.validate_ohlc_consistency(data_point)
            all_issues.extend(ohlc_issues)

            # Volume validation
            volume_issues = self.validate_volume(data_point, avg_volume)
            all_issues.extend(volume_issues)

        # Price continuity validation
        continuity_issues = self.validate_price_continuity(data_points)
        all_issues.extend(continuity_issues)

        # Gap detection
        detected_gaps = self.detect_gaps(data_points, symbol)

        # Calculate scores
        quality_score = self.calculate_quality_score(all_issues)
        completeness_score = self.calculate_completeness_score(
            data_points, start_date, end_date
        )

        report = ValidationReport(
            symbol=symbol,
            total_data_points=len(data_points),
            validation_issues=all_issues,
            detected_gaps=detected_gaps,
            quality_score=quality_score,
            completeness_score=completeness_score,
        )

        logger.info(
            f"✅ Validation complete: {symbol} - "
            f"Quality: {quality_score:.1f}, "
            f"Completeness: {completeness_score:.1f}, "
            f"Issues: {len(all_issues)}, "
            f"Gaps: {len(detected_gaps)}"
        )

        return report

    def generate_validation_summary(
        self, reports: list[ValidationReport]
    ) -> dict[str, Any]:
        """Generate summary across multiple validation reports"""
        if not reports:
            return {"error": "No validation reports provided"}

        total_data_points = sum(r.total_data_points for r in reports)
        total_issues = sum(len(r.validation_issues) for r in reports)
        total_gaps = sum(len(r.detected_gaps) for r in reports)

        avg_quality = sum(r.quality_score for r in reports) / len(reports)
        avg_completeness = sum(r.completeness_score for r in reports) / len(reports)

        critical_symbols = [r.symbol for r in reports if r.critical_issues_count > 0]
        incomplete_symbols = [r.symbol for r in reports if r.completeness_score < 95.0]

        summary = {
            "overall_statistics": {
                "total_symbols": len(reports),
                "total_data_points": total_data_points,
                "total_validation_issues": total_issues,
                "total_gaps_detected": total_gaps,
                "average_quality_score": f"{avg_quality:.1f}",
                "average_completeness_score": f"{avg_completeness:.1f}",
            },
            "quality_breakdown": {
                "symbols_with_critical_issues": len(critical_symbols),
                "symbols_with_completeness_issues": len(incomplete_symbols),
                "critical_symbols": critical_symbols,
                "incomplete_symbols": incomplete_symbols,
            },
            "symbol_details": {
                report.symbol: {
                    "data_points": report.total_data_points,
                    "quality_score": report.quality_score,
                    "completeness_score": report.completeness_score,
                    "total_issues": len(report.validation_issues),
                    "critical_issues": report.critical_issues_count,
                    "gaps_detected": len(report.detected_gaps),
                    "market_hours_gaps": report.market_hours_gaps_count,
                }
                for report in reports
            },
        }

        return summary


# Example usage and testing
def test_data_validator():
    """Test the data validator"""
    print("🧪 Testing Data Validator")

    validator = DataValidator()

    # Test data points (sample 1-minute OHLC data)
    test_data = [
        {
            "symbol": "SPY",
            "timestamp": datetime(2024, 1, 1, 9, 30),
            "open": 100.0,
            "high": 101.0,
            "low": 99.5,
            "close": 100.5,
            "volume": 1000,
        },
        {
            "symbol": "SPY",
            "timestamp": datetime(2024, 1, 1, 9, 31),
            "open": 100.5,
            "high": 102.0,
            "low": 100.0,
            "close": 101.5,
            "volume": 1200,
        },
        # Intentional issue: gap and invalid data
        {
            "symbol": "SPY",
            "timestamp": datetime(2024, 1, 1, 9, 40),  # 9-minute gap
            "open": 101.5,
            "high": 101.0,  # High < Open (invalid)
            "low": 102.0,  # Low > Close (invalid)
            "close": 101.8,
            "volume": -100,  # Negative volume (invalid)
        },
    ]

    # Validate dataset
    report = validator.validate_dataset(
        data_points=test_data,
        symbol="SPY",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 1),
    )

    print("Validation Report:")
    print(f"  Data Points: {report.total_data_points}")
    print(f"  Quality Score: {report.quality_score:.1f}")
    print(f"  Completeness Score: {report.completeness_score:.1f}")
    print(f"  Issues Found: {len(report.validation_issues)}")
    print(f"  Gaps Detected: {len(report.detected_gaps)}")

    for issue in report.validation_issues:
        print(f"    {issue.level.value.upper()}: {issue.description}")


if __name__ == "__main__":
    test_data_validator()
