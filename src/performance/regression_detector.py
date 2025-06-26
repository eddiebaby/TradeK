"""
Performance Regression Detection for TradeKnowledge.

This module provides comprehensive performance regression detection
capabilities including baseline establishment, trend analysis, and
automated alerting for performance degradations.
"""

import asyncio
import functools
import json
import logging
import sqlite3
import statistics
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import scipy.stats as stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    np = None
    stats = None

logger = logging.getLogger(__name__)


class RegressionSeverity(Enum):
    """Severity levels for performance regressions"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ChangeDirection(Enum):
    """Direction of performance change"""

    IMPROVEMENT = "improvement"
    DEGRADATION = "degradation"
    STABLE = "stable"


@dataclass
class PerformanceBaseline:
    """Baseline performance metrics for comparison"""

    metric_name: str
    operation_name: str
    baseline_value: float
    sample_size: int
    confidence_interval: tuple[float, float]
    created_at: datetime
    last_updated: datetime
    standard_deviation: float
    percentiles: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMeasurement:
    """Single performance measurement"""

    metric_name: str
    operation_name: str
    value: float
    timestamp: datetime
    duration_ms: float | None = None
    context: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class RegressionAlert:
    """Performance regression alert"""

    alert_id: str
    metric_name: str
    operation_name: str
    severity: RegressionSeverity
    change_direction: ChangeDirection
    baseline_value: float
    current_value: float
    change_percentage: float
    detection_time: datetime
    confidence_level: float
    sample_size: int
    statistical_significance: float
    description: str
    remediation_suggestions: list[str] = field(default_factory=list)


@dataclass
class TrendAnalysis:
    """Trend analysis results"""

    metric_name: str
    operation_name: str
    time_period: timedelta
    trend_direction: ChangeDirection
    trend_strength: float  # R-squared value
    slope: float
    p_value: float
    confidence_level: float
    data_points: int
    prediction: float | None = None
    seasonality_detected: bool = False


class PerformanceRegressionDetector:
    """
    Comprehensive performance regression detection system that monitors
    performance metrics, establishes baselines, and detects regressions
    using statistical analysis and machine learning techniques.
    """

    def __init__(self, db_path: str = "data/performance_regression.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self.baselines: dict[str, PerformanceBaseline] = {}
        self.recent_measurements: dict[str, list[PerformanceMeasurement]] = {}
        self.alerts: list[RegressionAlert] = []

        # Configuration
        self.baseline_sample_size = 100
        self.regression_threshold = 0.15  # 15% degradation threshold
        self.confidence_level = 0.95
        self.trend_analysis_window = timedelta(hours=24)
        self.measurement_retention = timedelta(days=30)

        self._init_database()
        self._load_baselines()

    def _init_database(self):
        """Initialize SQLite database for persistence"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS baselines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    operation_name TEXT NOT NULL,
                    baseline_value REAL NOT NULL,
                    sample_size INTEGER NOT NULL,
                    confidence_interval_lower REAL NOT NULL,
                    confidence_interval_upper REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    standard_deviation REAL NOT NULL,
                    percentiles TEXT,
                    metadata TEXT,
                    UNIQUE(metric_name, operation_name)
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS measurements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    operation_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    duration_ms REAL,
                    context TEXT,
                    tags TEXT
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE NOT NULL,
                    metric_name TEXT NOT NULL,
                    operation_name TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    change_direction TEXT NOT NULL,
                    baseline_value REAL NOT NULL,
                    current_value REAL NOT NULL,
                    change_percentage REAL NOT NULL,
                    detection_time TEXT NOT NULL,
                    confidence_level REAL NOT NULL,
                    sample_size INTEGER NOT NULL,
                    statistical_significance REAL NOT NULL,
                    description TEXT NOT NULL,
                    remediation_suggestions TEXT
                )
            """
            )

            # Create indices for better performance
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_measurements_timestamp ON measurements(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_measurements_operation ON measurements(operation_name)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_detection_time ON alerts(detection_time)"
            )

    def _load_baselines(self):
        """Load existing baselines from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT metric_name, operation_name, baseline_value, sample_size,
                       confidence_interval_lower, confidence_interval_upper,
                       created_at, last_updated, standard_deviation,
                       percentiles, metadata
                FROM baselines
            """
            )

            for row in cursor.fetchall():
                metric_name, operation_name = row[0], row[1]
                key = f"{metric_name}:{operation_name}"

                percentiles = json.loads(row[9]) if row[9] else {}
                metadata = json.loads(row[10]) if row[10] else {}

                baseline = PerformanceBaseline(
                    metric_name=metric_name,
                    operation_name=operation_name,
                    baseline_value=row[2],
                    sample_size=row[3],
                    confidence_interval=(row[4], row[5]),
                    created_at=datetime.fromisoformat(row[6]),
                    last_updated=datetime.fromisoformat(row[7]),
                    standard_deviation=row[8],
                    percentiles=percentiles,
                    metadata=metadata,
                )

                self.baselines[key] = baseline

    def record_measurement(
        self,
        metric_name: str,
        operation_name: str,
        value: float,
        duration_ms: float | None = None,
        context: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ):
        """
        Record a performance measurement.

        Args:
            metric_name: Name of the performance metric
            operation_name: Name of the operation being measured
            value: Measured value
            duration_ms: Duration in milliseconds (optional)
            context: Additional context information
            tags: Metric tags for filtering
        """
        measurement = PerformanceMeasurement(
            metric_name=metric_name,
            operation_name=operation_name,
            value=value,
            timestamp=datetime.now(),
            duration_ms=duration_ms,
            context=context or {},
            tags=tags or {},
        )

        key = f"{metric_name}:{operation_name}"

        with self._lock:
            if key not in self.recent_measurements:
                self.recent_measurements[key] = []

            self.recent_measurements[key].append(measurement)

            # Keep only recent measurements
            cutoff_time = datetime.now() - self.measurement_retention
            self.recent_measurements[key] = [
                m for m in self.recent_measurements[key] if m.timestamp >= cutoff_time
            ]

        # Persist to database
        self._save_measurement(measurement)

        # Check for regressions
        if key in self.baselines:
            alert = self._check_for_regression(measurement, self.baselines[key])
            if alert:
                self.alerts.append(alert)
                self._save_alert(alert)
                logger.warning(f"Performance regression detected: {alert.description}")

        # Update baseline if needed
        self._update_baseline_if_needed(key, measurement)

    def _save_measurement(self, measurement: PerformanceMeasurement):
        """Save measurement to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO measurements 
                (metric_name, operation_name, value, timestamp, duration_ms, context, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    measurement.metric_name,
                    measurement.operation_name,
                    measurement.value,
                    measurement.timestamp.isoformat(),
                    measurement.duration_ms,
                    json.dumps(measurement.context),
                    json.dumps(measurement.tags),
                ),
            )

    def _save_alert(self, alert: RegressionAlert):
        """Save alert to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO alerts 
                (alert_id, metric_name, operation_name, severity, change_direction,
                 baseline_value, current_value, change_percentage, detection_time,
                 confidence_level, sample_size, statistical_significance,
                 description, remediation_suggestions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    alert.alert_id,
                    alert.metric_name,
                    alert.operation_name,
                    alert.severity.value,
                    alert.change_direction.value,
                    alert.baseline_value,
                    alert.current_value,
                    alert.change_percentage,
                    alert.detection_time.isoformat(),
                    alert.confidence_level,
                    alert.sample_size,
                    alert.statistical_significance,
                    alert.description,
                    json.dumps(alert.remediation_suggestions),
                ),
            )

    def establish_baseline(
        self,
        metric_name: str,
        operation_name: str,
        measurements: list[float] | None = None,
    ) -> PerformanceBaseline:
        """
        Establish performance baseline for a metric.

        Args:
            metric_name: Name of the performance metric
            operation_name: Name of the operation
            measurements: Optional list of measurements, uses recent data if not provided

        Returns:
            Established baseline
        """
        key = f"{metric_name}:{operation_name}"

        if measurements is None:
            # Use recent measurements
            if key not in self.recent_measurements:
                raise ValueError(f"No measurements available for {key}")

            measurements = [m.value for m in self.recent_measurements[key]]

        if len(measurements) < 10:
            raise ValueError(
                f"Insufficient measurements for baseline (need at least 10, got {len(measurements)})"
            )

        # Calculate baseline statistics
        baseline_value = statistics.mean(measurements)
        std_dev = statistics.stdev(measurements)

        # Calculate confidence interval
        n = len(measurements)
        if SCIPY_AVAILABLE:
            confidence_interval = stats.t.interval(
                self.confidence_level,
                df=n - 1,
                loc=baseline_value,
                scale=std_dev / np.sqrt(n),
            )

            # Calculate percentiles
            percentiles = {
                "p50": np.percentile(measurements, 50),
                "p75": np.percentile(measurements, 75),
                "p90": np.percentile(measurements, 90),
                "p95": np.percentile(measurements, 95),
                "p99": np.percentile(measurements, 99),
            }
        else:
            # Fallback without scipy
            margin = 1.96 * std_dev / (n**0.5)  # Approximate 95% CI
            confidence_interval = (baseline_value - margin, baseline_value + margin)

            # Calculate percentiles using built-in statistics
            sorted_measurements = sorted(measurements)
            percentiles = {
                "p50": sorted_measurements[int(n * 0.5)],
                "p75": sorted_measurements[int(n * 0.75)],
                "p90": sorted_measurements[int(n * 0.9)],
                "p95": sorted_measurements[int(n * 0.95)],
                "p99": sorted_measurements[min(int(n * 0.99), n - 1)],
            }

        baseline = PerformanceBaseline(
            metric_name=metric_name,
            operation_name=operation_name,
            baseline_value=baseline_value,
            sample_size=n,
            confidence_interval=confidence_interval,
            created_at=datetime.now(),
            last_updated=datetime.now(),
            standard_deviation=std_dev,
            percentiles=percentiles,
            metadata={"auto_established": True},
        )

        with self._lock:
            self.baselines[key] = baseline

        # Save to database
        self._save_baseline(baseline)

        logger.info(
            f"Established baseline for {key}: {baseline_value:.2f} ± {std_dev:.2f}"
        )

        return baseline

    def _save_baseline(self, baseline: PerformanceBaseline):
        """Save baseline to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO baselines 
                (metric_name, operation_name, baseline_value, sample_size,
                 confidence_interval_lower, confidence_interval_upper,
                 created_at, last_updated, standard_deviation,
                 percentiles, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    baseline.metric_name,
                    baseline.operation_name,
                    baseline.baseline_value,
                    baseline.sample_size,
                    baseline.confidence_interval[0],
                    baseline.confidence_interval[1],
                    baseline.created_at.isoformat(),
                    baseline.last_updated.isoformat(),
                    baseline.standard_deviation,
                    json.dumps(baseline.percentiles),
                    json.dumps(baseline.metadata),
                ),
            )

    def _check_for_regression(
        self, measurement: PerformanceMeasurement, baseline: PerformanceBaseline
    ) -> RegressionAlert | None:
        """Check if measurement indicates a performance regression"""

        # Calculate change percentage
        if baseline.baseline_value == 0:
            if measurement.value == 0:
                return None
            change_percentage = float("inf")
        else:
            change_percentage = (
                measurement.value - baseline.baseline_value
            ) / baseline.baseline_value

        # Determine if this is a significant change
        abs_change = abs(change_percentage)
        if abs_change < self.regression_threshold:
            return None

        # Determine change direction (for performance metrics, higher is usually worse)
        if change_percentage > 0:
            change_direction = ChangeDirection.DEGRADATION
            severity = self._calculate_severity(abs_change)
        else:
            change_direction = ChangeDirection.IMPROVEMENT
            severity = RegressionSeverity.LOW  # Improvements are low priority alerts

        # Calculate statistical significance using recent measurements
        key = f"{measurement.metric_name}:{measurement.operation_name}"
        recent_values = [m.value for m in self.recent_measurements.get(key, [])]

        if len(recent_values) < 5:
            statistical_significance = 0.0
        else:
            # Perform t-test against baseline
            if SCIPY_AVAILABLE:
                t_stat, p_value = stats.ttest_1samp(
                    recent_values, baseline.baseline_value
                )
                statistical_significance = 1 - p_value
            else:
                # Simple statistical significance approximation without scipy
                recent_mean = statistics.mean(recent_values)
                if len(recent_values) > 1:
                    recent_std = statistics.stdev(recent_values)
                    z_score = abs(recent_mean - baseline.baseline_value) / (
                        recent_std / (len(recent_values) ** 0.5)
                    )
                    # Rough approximation: z > 2 indicates significance
                    statistical_significance = min(1.0, z_score / 2.0)
                else:
                    statistical_significance = 0.0

        # Generate alert
        alert_id = (
            f"{measurement.metric_name}_{measurement.operation_name}_{int(time.time())}"
        )

        description = (
            f"Performance {change_direction.value} detected in {measurement.operation_name}: "
            f"{measurement.metric_name} changed from {baseline.baseline_value:.2f} "
            f"to {measurement.value:.2f} ({change_percentage*100:+.1f}%)"
        )

        remediation_suggestions = self._generate_remediation_suggestions(
            measurement, baseline, change_direction, severity
        )

        alert = RegressionAlert(
            alert_id=alert_id,
            metric_name=measurement.metric_name,
            operation_name=measurement.operation_name,
            severity=severity,
            change_direction=change_direction,
            baseline_value=baseline.baseline_value,
            current_value=measurement.value,
            change_percentage=change_percentage,
            detection_time=measurement.timestamp,
            confidence_level=self.confidence_level,
            sample_size=len(recent_values),
            statistical_significance=statistical_significance,
            description=description,
            remediation_suggestions=remediation_suggestions,
        )

        return alert

    def _calculate_severity(self, change_percentage: float) -> RegressionSeverity:
        """Calculate severity based on change percentage"""
        if change_percentage >= 0.5:  # 50% or more
            return RegressionSeverity.CRITICAL
        elif change_percentage >= 0.3:  # 30-49%
            return RegressionSeverity.HIGH
        elif change_percentage >= 0.2:  # 20-29%
            return RegressionSeverity.MEDIUM
        else:  # 15-19%
            return RegressionSeverity.LOW

    def _generate_remediation_suggestions(
        self,
        measurement: PerformanceMeasurement,
        baseline: PerformanceBaseline,
        direction: ChangeDirection,
        severity: RegressionSeverity,
    ) -> list[str]:
        """Generate remediation suggestions based on the regression"""
        suggestions = []

        if direction == ChangeDirection.DEGRADATION:
            if "response_time" in measurement.metric_name.lower():
                suggestions.extend(
                    [
                        "Check for increased load or concurrent users",
                        "Review recent code changes for performance impact",
                        "Analyze database query performance",
                        "Check system resource utilization (CPU, memory, I/O)",
                        "Consider implementing caching strategies",
                    ]
                )

            elif "throughput" in measurement.metric_name.lower():
                suggestions.extend(
                    [
                        "Check for bottlenecks in request processing",
                        "Review connection pool configurations",
                        "Analyze thread pool utilization",
                        "Consider horizontal scaling",
                        "Check for memory leaks or resource exhaustion",
                    ]
                )

            elif "memory" in measurement.metric_name.lower():
                suggestions.extend(
                    [
                        "Check for memory leaks in recent code changes",
                        "Review object creation and cleanup patterns",
                        "Analyze garbage collection performance",
                        "Consider implementing object pooling",
                        "Review data structure usage efficiency",
                    ]
                )

            if severity in [RegressionSeverity.HIGH, RegressionSeverity.CRITICAL]:
                suggestions.extend(
                    [
                        "Consider rolling back recent changes",
                        "Implement circuit breakers to prevent cascading failures",
                        "Set up additional monitoring and alerting",
                        "Plan for emergency scaling procedures",
                    ]
                )

        else:  # Improvement
            suggestions.extend(
                [
                    "Document the improvement for future reference",
                    "Consider if the improvement indicates a change in usage patterns",
                    "Update performance benchmarks and SLAs",
                    "Share successful optimization techniques with the team",
                ]
            )

        return suggestions

    def _update_baseline_if_needed(self, key: str, measurement: PerformanceMeasurement):
        """Update baseline if sufficient new data is available"""
        if key not in self.recent_measurements:
            return

        measurements = self.recent_measurements[key]
        if len(measurements) < self.baseline_sample_size:
            return

        # Check if baseline is old enough to warrant updating
        if key in self.baselines:
            baseline = self.baselines[key]
            age = datetime.now() - baseline.last_updated
            if age < timedelta(hours=24):  # Don't update more than once per day
                return

        # Update baseline with recent measurements
        try:
            recent_values = [
                m.value for m in measurements[-self.baseline_sample_size :]
            ]
            self.establish_baseline(
                measurement.metric_name, measurement.operation_name, recent_values
            )
            logger.info(f"Updated baseline for {key}")
        except Exception as e:
            logger.warning(f"Failed to update baseline for {key}: {e}")

    def analyze_trends(
        self,
        metric_name: str,
        operation_name: str,
        time_window: timedelta | None = None,
    ) -> TrendAnalysis:
        """
        Analyze performance trends over time.

        Args:
            metric_name: Name of the performance metric
            operation_name: Name of the operation
            time_window: Time window for analysis (default: 24 hours)

        Returns:
            Trend analysis results
        """
        if time_window is None:
            time_window = self.trend_analysis_window

        # Get measurements from database
        cutoff_time = datetime.now() - time_window

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT value, timestamp FROM measurements
                WHERE metric_name = ? AND operation_name = ? 
                AND timestamp >= ?
                ORDER BY timestamp
            """,
                (metric_name, operation_name, cutoff_time.isoformat()),
            )

            rows = cursor.fetchall()

        if len(rows) < 10:
            raise ValueError(
                f"Insufficient data for trend analysis (need at least 10 points, got {len(rows)})"
            )

        # Prepare data for analysis
        values = [row[0] for row in rows]
        timestamps = [datetime.fromisoformat(row[1]) for row in rows]

        # Convert timestamps to numeric values (seconds since first measurement)
        start_time = timestamps[0]
        x_values = [(t - start_time).total_seconds() for t in timestamps]
        y_values = values

        # Perform linear regression
        if SCIPY_AVAILABLE:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                x_values, y_values
            )
        else:
            # Simple linear regression without scipy
            n = len(x_values)
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x * y for x, y in zip(x_values, y_values, strict=False))
            sum_x2 = sum(x * x for x in x_values)

            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            intercept = (sum_y - slope * sum_x) / n

            # Calculate correlation coefficient
            mean_x = sum_x / n
            mean_y = sum_y / n
            numerator = sum(
                (x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values, strict=False)
            )
            sum_sq_x = sum((x - mean_x) ** 2 for x in x_values)
            sum_sq_y = sum((y - mean_y) ** 2 for y in y_values)
            denominator = (sum_sq_x * sum_sq_y) ** 0.5

            r_value = numerator / denominator if denominator != 0 else 0
            p_value = 0.05  # Default p-value when can't calculate
            std_err = 0

        # Determine trend direction
        if abs(slope) < 0.001:  # Very small slope considered stable
            trend_direction = ChangeDirection.STABLE
        elif slope > 0:
            trend_direction = (
                ChangeDirection.DEGRADATION
            )  # Assuming higher values are worse
        else:
            trend_direction = ChangeDirection.IMPROVEMENT

        # Calculate prediction for next time period
        next_x = x_values[-1] + time_window.total_seconds()
        prediction = slope * next_x + intercept

        # Detect seasonality (basic check for cyclical patterns)
        seasonality_detected = self._detect_seasonality(y_values)

        return TrendAnalysis(
            metric_name=metric_name,
            operation_name=operation_name,
            time_period=time_window,
            trend_direction=trend_direction,
            trend_strength=r_value**2,  # R-squared
            slope=slope,
            p_value=p_value,
            confidence_level=1 - p_value,
            data_points=len(values),
            prediction=prediction,
            seasonality_detected=seasonality_detected,
        )

    def _detect_seasonality(self, values: list[float]) -> bool:
        """Simple seasonality detection using autocorrelation"""
        if len(values) < 24:  # Need at least 24 points for hourly seasonality
            return False

        try:
            if SCIPY_AVAILABLE:
                # Check for daily seasonality (24-hour cycle)
                autocorr_24 = np.corrcoef(values[:-24], values[24:])[0, 1]

                # Check for weekly seasonality (7-day cycle, if we have enough data)
                autocorr_168 = 0
                if len(values) >= 168:  # 7 * 24 hours
                    autocorr_168 = np.corrcoef(values[:-168], values[168:])[0, 1]

                # Consider seasonality detected if autocorrelation is significant
                return abs(autocorr_24) > 0.3 or abs(autocorr_168) > 0.3
            else:
                # Simple pattern detection without numpy
                # Check if there's a repeating pattern every 24 values
                if len(values) >= 48:
                    first_day = values[:24]
                    second_day = values[24:48]

                    # Calculate simple correlation
                    mean1 = statistics.mean(first_day)
                    mean2 = statistics.mean(second_day)

                    numerator = sum(
                        (a - mean1) * (b - mean2) for a, b in zip(first_day, second_day, strict=False)
                    )
                    sum_sq1 = sum((a - mean1) ** 2 for a in first_day)
                    sum_sq2 = sum((b - mean2) ** 2 for b in second_day)

                    if sum_sq1 > 0 and sum_sq2 > 0:
                        correlation = numerator / ((sum_sq1 * sum_sq2) ** 0.5)
                        return abs(correlation) > 0.3

                return False

        except:
            return False

    def get_alerts(
        self,
        severity: RegressionSeverity | None = None,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[RegressionAlert]:
        """
        Get performance regression alerts.

        Args:
            severity: Filter by severity level
            since: Get alerts since this time
            limit: Maximum number of alerts to return

        Returns:
            List of regression alerts
        """
        query = "SELECT * FROM alerts WHERE 1=1"
        params = []

        if severity:
            query += " AND severity = ?"
            params.append(severity.value)

        if since:
            query += " AND detection_time >= ?"
            params.append(since.isoformat())

        query += " ORDER BY detection_time DESC"

        if limit:
            query += f" LIMIT {limit}"

        alerts = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            for row in cursor.fetchall():
                alert = RegressionAlert(
                    alert_id=row[1],
                    metric_name=row[2],
                    operation_name=row[3],
                    severity=RegressionSeverity(row[4]),
                    change_direction=ChangeDirection(row[5]),
                    baseline_value=row[6],
                    current_value=row[7],
                    change_percentage=row[8],
                    detection_time=datetime.fromisoformat(row[9]),
                    confidence_level=row[10],
                    sample_size=row[11],
                    statistical_significance=row[12],
                    description=row[13],
                    remediation_suggestions=json.loads(row[14]) if row[14] else [],
                )
                alerts.append(alert)

        return alerts

    def get_baseline(
        self, metric_name: str, operation_name: str
    ) -> PerformanceBaseline | None:
        """Get baseline for a specific metric and operation"""
        key = f"{metric_name}:{operation_name}"
        return self.baselines.get(key)

    def get_recent_measurements(
        self, metric_name: str, operation_name: str, limit: int = 100
    ) -> list[PerformanceMeasurement]:
        """Get recent measurements for a metric"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT metric_name, operation_name, value, timestamp, duration_ms, context, tags
                FROM measurements
                WHERE metric_name = ? AND operation_name = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (metric_name, operation_name, limit),
            )

            measurements = []
            for row in cursor.fetchall():
                measurement = PerformanceMeasurement(
                    metric_name=row[0],
                    operation_name=row[1],
                    value=row[2],
                    timestamp=datetime.fromisoformat(row[3]),
                    duration_ms=row[4],
                    context=json.loads(row[5]) if row[5] else {},
                    tags=json.loads(row[6]) if row[6] else {},
                )
                measurements.append(measurement)

            return measurements

    def generate_performance_report(
        self,
        operations: list[str] | None = None,
        time_window: timedelta | None = None,
    ) -> dict[str, Any]:
        """
        Generate comprehensive performance report.

        Args:
            operations: List of operations to include (all if None)
            time_window: Time window for analysis

        Returns:
            Performance report dictionary
        """
        if time_window is None:
            time_window = timedelta(hours=24)

        report = {
            "generated_at": datetime.now().isoformat(),
            "time_window": str(time_window),
            "summary": {
                "total_baselines": len(self.baselines),
                "active_alerts": len(
                    self.get_alerts(since=datetime.now() - timedelta(hours=24))
                ),
                "critical_alerts": len(
                    self.get_alerts(
                        severity=RegressionSeverity.CRITICAL,
                        since=datetime.now() - timedelta(hours=24),
                    )
                ),
            },
            "baselines": {},
            "trends": {},
            "alerts": [],
            "recommendations": [],
        }

        # Include baseline information
        for key, baseline in self.baselines.items():
            if operations and baseline.operation_name not in operations:
                continue

            report["baselines"][key] = {
                "metric_name": baseline.metric_name,
                "operation_name": baseline.operation_name,
                "baseline_value": baseline.baseline_value,
                "standard_deviation": baseline.standard_deviation,
                "sample_size": baseline.sample_size,
                "last_updated": baseline.last_updated.isoformat(),
                "percentiles": baseline.percentiles,
            }

        # Include trend analysis
        for key, baseline in self.baselines.items():
            if operations and baseline.operation_name not in operations:
                continue

            try:
                trend = self.analyze_trends(
                    baseline.metric_name, baseline.operation_name, time_window
                )
                report["trends"][key] = {
                    "trend_direction": trend.trend_direction.value,
                    "trend_strength": trend.trend_strength,
                    "slope": trend.slope,
                    "confidence_level": trend.confidence_level,
                    "data_points": trend.data_points,
                    "prediction": trend.prediction,
                    "seasonality_detected": trend.seasonality_detected,
                }
            except Exception as e:
                logger.debug(f"Could not analyze trends for {key}: {e}")

        # Include recent alerts
        recent_alerts = self.get_alerts(since=datetime.now() - time_window, limit=50)
        for alert in recent_alerts:
            if operations and alert.operation_name not in operations:
                continue

            report["alerts"].append(
                {
                    "alert_id": alert.alert_id,
                    "metric_name": alert.metric_name,
                    "operation_name": alert.operation_name,
                    "severity": alert.severity.value,
                    "change_direction": alert.change_direction.value,
                    "change_percentage": alert.change_percentage * 100,
                    "detection_time": alert.detection_time.isoformat(),
                    "description": alert.description,
                    "remediation_suggestions": alert.remediation_suggestions,
                }
            )

        # Generate recommendations
        report["recommendations"] = self._generate_report_recommendations(report)

        return report

    def _generate_report_recommendations(self, report: dict[str, Any]) -> list[str]:
        """Generate recommendations based on the performance report"""
        recommendations = []

        critical_alerts = [a for a in report["alerts"] if a["severity"] == "critical"]
        high_alerts = [a for a in report["alerts"] if a["severity"] == "high"]

        if critical_alerts:
            recommendations.append(
                f"🚨 URGENT: {len(critical_alerts)} critical performance regressions detected. "
                "Immediate investigation required."
            )

        if high_alerts:
            recommendations.append(
                f"⚠️  {len(high_alerts)} high-severity performance issues require attention within 24 hours."
            )

        # Check for operations with poor trends
        degrading_operations = [
            k
            for k, trend in report["trends"].items()
            if trend["trend_direction"] == "degradation"
            and trend["confidence_level"] > 0.8
        ]

        if degrading_operations:
            recommendations.append(
                f"📉 {len(degrading_operations)} operations showing consistent performance degradation: "
                f"{', '.join([k.split(':')[1] for k in degrading_operations[:3]])}"
            )

        # Check for seasonality patterns
        seasonal_operations = [
            k for k, trend in report["trends"].items() if trend["seasonality_detected"]
        ]

        if seasonal_operations:
            recommendations.append(
                f"🔄 {len(seasonal_operations)} operations show seasonal patterns. "
                "Consider predictive scaling based on these patterns."
            )

        # Check baseline age
        old_baselines = [
            k
            for k, baseline in report["baselines"].items()
            if (datetime.now() - datetime.fromisoformat(baseline["last_updated"])).days
            > 7
        ]

        if old_baselines:
            recommendations.append(
                f"📊 {len(old_baselines)} baselines are over 7 days old. "
                "Consider updating with recent performance data."
            )

        if not recommendations:
            recommendations.append(
                "✅ No immediate performance concerns detected. System operating within normal parameters."
            )

        return recommendations


# Global detector instance
_global_detector: PerformanceRegressionDetector | None = None


def get_regression_detector() -> PerformanceRegressionDetector:
    """Get or create global regression detector"""
    global _global_detector
    if _global_detector is None:
        _global_detector = PerformanceRegressionDetector()
    return _global_detector


def performance_monitor(metric_name: str, operation_name: str | None = None):
    """
    Decorator for automatic performance monitoring and regression detection.

    Args:
        metric_name: Name of the metric to monitor
        operation_name: Name of the operation (defaults to function name)
    """

    def decorator(func: Callable) -> Callable:
        nonlocal operation_name
        if operation_name is None:
            operation_name = func.__name__

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            detector = get_regression_detector()
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                detector.record_measurement(metric_name, operation_name, duration_ms)
                return result
            except Exception:
                duration_ms = (time.time() - start_time) * 1000
                detector.record_measurement(
                    f"{metric_name}_error", operation_name, duration_ms
                )
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            detector = get_regression_detector()
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                detector.record_measurement(metric_name, operation_name, duration_ms)
                return result
            except Exception:
                duration_ms = (time.time() - start_time) * 1000
                detector.record_measurement(
                    f"{metric_name}_error", operation_name, duration_ms
                )
                raise

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Convenience functions
def establish_baseline_for_operation(
    operation_name: str,
    metric_name: str = "response_time_ms",
    sample_measurements: list[float] | None = None,
):
    """Establish baseline for an operation"""
    detector = get_regression_detector()
    return detector.establish_baseline(metric_name, operation_name, sample_measurements)


def check_performance_regression(
    operation_name: str, current_value: float, metric_name: str = "response_time_ms"
) -> RegressionAlert | None:
    """Check if a measurement indicates regression"""
    detector = get_regression_detector()
    detector.record_measurement(metric_name, operation_name, current_value)

    # Return the most recent alert for this operation if any
    recent_alerts = detector.get_alerts(since=datetime.now() - timedelta(minutes=1))
    for alert in recent_alerts:
        if alert.operation_name == operation_name and alert.metric_name == metric_name:
            return alert

    return None
