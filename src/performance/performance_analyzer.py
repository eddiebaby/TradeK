"""
Performance Analysis and Monitoring for TradeKnowledge.

This module provides tools for analyzing performance data, detecting regressions,
and generating performance reports.
"""

import json
import logging
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration"""

    operation: str
    max_duration_ms: float
    max_memory_mb: float
    max_cpu_percent: float
    min_success_rate: float = 0.95


@dataclass
class PerformanceAlert:
    """Performance alert information"""

    operation: str
    metric_type: str
    current_value: float
    threshold_value: float
    severity: str  # 'warning', 'critical'
    timestamp: datetime
    details: dict[str, Any] = None


class PerformanceAnalyzer:
    """Analyzes performance data and detects regressions"""

    def __init__(self, config_path: str | None = None):
        self.thresholds = self._load_thresholds(config_path)
        self.historical_data = []
        self.alerts = []

    def _load_thresholds(
        self, config_path: str | None
    ) -> dict[str, PerformanceThreshold]:
        """Load performance thresholds from configuration"""
        default_thresholds = {
            "single_search": PerformanceThreshold(
                operation="single_search",
                max_duration_ms=100.0,
                max_memory_mb=50.0,
                max_cpu_percent=30.0,
            ),
            "concurrent_5_searches": PerformanceThreshold(
                operation="concurrent_5_searches",
                max_duration_ms=200.0,
                max_memory_mb=100.0,
                max_cpu_percent=60.0,
            ),
            "search_load_20_queries": PerformanceThreshold(
                operation="search_load_20_queries",
                max_duration_ms=1000.0,
                max_memory_mb=200.0,
                max_cpu_percent=70.0,
            ),
            "single_embedding": PerformanceThreshold(
                operation="single_embedding",
                max_duration_ms=50.0,
                max_memory_mb=100.0,
                max_cpu_percent=40.0,
            ),
            "batch_50_embeddings": PerformanceThreshold(
                operation="batch_50_embeddings",
                max_duration_ms=1000.0,
                max_memory_mb=500.0,
                max_cpu_percent=80.0,
            ),
            "query_embedding": PerformanceThreshold(
                operation="query_embedding",
                max_duration_ms=20.0,
                max_memory_mb=50.0,
                max_cpu_percent=20.0,
            ),
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path) as f:
                    config_data = json.load(f)

                # Override defaults with config data
                for op_name, threshold_data in config_data.get(
                    "thresholds", {}
                ).items():
                    default_thresholds[op_name] = PerformanceThreshold(**threshold_data)

            except Exception as e:
                logger.warning(f"Failed to load threshold config: {e}")

        return default_thresholds

    def analyze_performance_data(
        self, performance_summary: dict[str, Any]
    ) -> list[PerformanceAlert]:
        """Analyze performance data and generate alerts"""
        alerts = []

        for operation, stats in performance_summary.items():
            if operation not in self.thresholds:
                logger.warning(f"No thresholds defined for operation: {operation}")
                continue

            threshold = self.thresholds[operation]
            duration_stats = stats.get("duration_stats", {})
            memory_stats = stats.get("memory_stats", {})
            cpu_stats = stats.get("cpu_stats", {})
            success_rate = stats.get("success_rate", 1.0)

            # Check duration threshold
            p95_duration = duration_stats.get("p95_ms", 0)
            if p95_duration > threshold.max_duration_ms:
                alerts.append(
                    PerformanceAlert(
                        operation=operation,
                        metric_type="duration_p95",
                        current_value=p95_duration,
                        threshold_value=threshold.max_duration_ms,
                        severity=(
                            "critical"
                            if p95_duration > threshold.max_duration_ms * 1.5
                            else "warning"
                        ),
                        timestamp=datetime.now(),
                        details={"mean_ms": duration_stats.get("mean_ms", 0)},
                    )
                )

            # Check memory threshold
            max_memory = memory_stats.get("max_mb", 0)
            if max_memory > threshold.max_memory_mb:
                alerts.append(
                    PerformanceAlert(
                        operation=operation,
                        metric_type="memory_max",
                        current_value=max_memory,
                        threshold_value=threshold.max_memory_mb,
                        severity=(
                            "critical"
                            if max_memory > threshold.max_memory_mb * 1.5
                            else "warning"
                        ),
                        timestamp=datetime.now(),
                        details={"mean_mb": memory_stats.get("mean_mb", 0)},
                    )
                )

            # Check CPU threshold
            max_cpu = cpu_stats.get("max_percent", 0)
            if max_cpu > threshold.max_cpu_percent:
                alerts.append(
                    PerformanceAlert(
                        operation=operation,
                        metric_type="cpu_max",
                        current_value=max_cpu,
                        threshold_value=threshold.max_cpu_percent,
                        severity=(
                            "critical"
                            if max_cpu > threshold.max_cpu_percent * 1.5
                            else "warning"
                        ),
                        timestamp=datetime.now(),
                        details={"mean_percent": cpu_stats.get("mean_percent", 0)},
                    )
                )

            # Check success rate threshold
            if success_rate < threshold.min_success_rate:
                alerts.append(
                    PerformanceAlert(
                        operation=operation,
                        metric_type="success_rate",
                        current_value=success_rate,
                        threshold_value=threshold.min_success_rate,
                        severity="critical",
                        timestamp=datetime.now(),
                        details={"total_runs": stats.get("total_runs", 0)},
                    )
                )

        self.alerts.extend(alerts)
        return alerts

    def detect_performance_regression(
        self,
        current_data: dict[str, Any],
        historical_data: list[dict[str, Any]],
        regression_threshold: float = 1.5,
    ) -> list[PerformanceAlert]:
        """Detect performance regressions compared to historical data"""
        alerts = []

        if not historical_data:
            logger.info("No historical data available for regression detection")
            return alerts

        # Calculate historical baselines
        historical_baselines = self._calculate_historical_baselines(historical_data)

        for operation, current_stats in current_data.items():
            if operation not in historical_baselines:
                continue

            baseline = historical_baselines[operation]
            current_p95 = current_stats.get("duration_stats", {}).get("p95_ms", 0)
            baseline_p95 = baseline.get("p95_ms", 0)

            if baseline_p95 > 0 and current_p95 > baseline_p95 * regression_threshold:
                alerts.append(
                    PerformanceAlert(
                        operation=operation,
                        metric_type="performance_regression",
                        current_value=current_p95,
                        threshold_value=baseline_p95 * regression_threshold,
                        severity=(
                            "critical"
                            if current_p95 > baseline_p95 * 2.0
                            else "warning"
                        ),
                        timestamp=datetime.now(),
                        details={
                            "baseline_p95_ms": baseline_p95,
                            "regression_factor": current_p95 / baseline_p95,
                        },
                    )
                )

        return alerts

    def _calculate_historical_baselines(
        self, historical_data: list[dict[str, Any]]
    ) -> dict[str, dict[str, float]]:
        """Calculate baseline performance metrics from historical data"""
        baselines = {}

        # Group historical data by operation
        operation_history = {}
        for data_point in historical_data:
            for operation, stats in data_point.items():
                if operation not in operation_history:
                    operation_history[operation] = []
                operation_history[operation].append(stats)

        # Calculate baselines for each operation
        for operation, history in operation_history.items():
            p95_values = []
            mean_values = []

            for stats in history:
                duration_stats = stats.get("duration_stats", {})
                if "p95_ms" in duration_stats:
                    p95_values.append(duration_stats["p95_ms"])
                if "mean_ms" in duration_stats:
                    mean_values.append(duration_stats["mean_ms"])

            if p95_values:
                baselines[operation] = {
                    "p95_ms": statistics.median(p95_values),
                    "mean_ms": statistics.median(mean_values) if mean_values else 0,
                }

        return baselines

    def generate_performance_report(self, performance_summary: dict[str, Any]) -> str:
        """Generate a comprehensive performance report"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("TRADEKNOWLEDGE PERFORMANCE REPORT")
        report_lines.append("=" * 60)
        report_lines.append(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report_lines.append("")

        # Overall summary
        total_operations = len(performance_summary)
        operations_with_issues = len(
            [
                op
                for op in performance_summary.values()
                if op.get("success_rate", 1.0) < 1.0
            ]
        )

        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-" * 20)
        report_lines.append(f"Total Operations Tested: {total_operations}")
        report_lines.append(f"Operations with Issues: {operations_with_issues}")
        report_lines.append("")

        # Detailed operation analysis
        report_lines.append("DETAILED ANALYSIS")
        report_lines.append("-" * 20)

        for operation, stats in performance_summary.items():
            report_lines.append(f"\n{operation.upper()}")
            report_lines.append("-" * len(operation))

            # Duration analysis
            duration_stats = stats.get("duration_stats", {})
            if duration_stats:
                report_lines.append("Duration (ms):")
                report_lines.append(f"  Mean: {duration_stats.get('mean_ms', 0):.2f}")
                report_lines.append(
                    f"  Median: {duration_stats.get('median_ms', 0):.2f}"
                )
                report_lines.append(f"  P95: {duration_stats.get('p95_ms', 0):.2f}")
                report_lines.append(f"  P99: {duration_stats.get('p99_ms', 0):.2f}")
                report_lines.append(
                    f"  Range: {duration_stats.get('min_ms', 0):.2f} - {duration_stats.get('max_ms', 0):.2f}"
                )

            # Memory analysis
            memory_stats = stats.get("memory_stats", {})
            if memory_stats:
                report_lines.append("Memory (MB):")
                report_lines.append(f"  Mean: {memory_stats.get('mean_mb', 0):.2f}")
                report_lines.append(f"  Max: {memory_stats.get('max_mb', 0):.2f}")

            # CPU analysis
            cpu_stats = stats.get("cpu_stats", {})
            if cpu_stats:
                report_lines.append("CPU (%):")
                report_lines.append(f"  Mean: {cpu_stats.get('mean_percent', 0):.2f}")
                report_lines.append(f"  Max: {cpu_stats.get('max_percent', 0):.2f}")

            # Success rate
            success_rate = stats.get("success_rate", 1.0)
            report_lines.append(f"Success Rate: {success_rate:.2%}")
            report_lines.append(f"Total Runs: {stats.get('total_runs', 0)}")

            # Threshold compliance
            if operation in self.thresholds:
                threshold = self.thresholds[operation]
                p95_duration = duration_stats.get("p95_ms", 0)
                max_memory = memory_stats.get("max_mb", 0)
                max_cpu = cpu_stats.get("max_percent", 0)

                report_lines.append("Threshold Compliance:")
                report_lines.append(
                    f"  Duration: {'✓' if p95_duration <= threshold.max_duration_ms else '✗'} "
                    f"({p95_duration:.2f} <= {threshold.max_duration_ms})"
                )
                report_lines.append(
                    f"  Memory: {'✓' if max_memory <= threshold.max_memory_mb else '✗'} "
                    f"({max_memory:.2f} <= {threshold.max_memory_mb})"
                )
                report_lines.append(
                    f"  CPU: {'✓' if max_cpu <= threshold.max_cpu_percent else '✗'} "
                    f"({max_cpu:.2f} <= {threshold.max_cpu_percent})"
                )
                report_lines.append(
                    f"  Success: {'✓' if success_rate >= threshold.min_success_rate else '✗'} "
                    f"({success_rate:.2%} >= {threshold.min_success_rate:.2%})"
                )

        # Alerts section
        if self.alerts:
            report_lines.append("\n\nPERFORMANCE ALERTS")
            report_lines.append("-" * 20)

            critical_alerts = [a for a in self.alerts if a.severity == "critical"]
            warning_alerts = [a for a in self.alerts if a.severity == "warning"]

            if critical_alerts:
                report_lines.append(f"\nCRITICAL ({len(critical_alerts)}):")
                for alert in critical_alerts:
                    report_lines.append(
                        f"  • {alert.operation}: {alert.metric_type} = {alert.current_value:.2f} "
                        f"(threshold: {alert.threshold_value:.2f})"
                    )

            if warning_alerts:
                report_lines.append(f"\nWARNINGS ({len(warning_alerts)}):")
                for alert in warning_alerts:
                    report_lines.append(
                        f"  • {alert.operation}: {alert.metric_type} = {alert.current_value:.2f} "
                        f"(threshold: {alert.threshold_value:.2f})"
                    )

        # Recommendations
        report_lines.append("\n\nRECOMMENDATIONS")
        report_lines.append("-" * 15)

        recommendations = self._generate_recommendations(performance_summary)
        for recommendation in recommendations:
            report_lines.append(f"• {recommendation}")

        report_lines.append("\n" + "=" * 60)

        return "\n".join(report_lines)

    def _generate_recommendations(
        self, performance_summary: dict[str, Any]
    ) -> list[str]:
        """Generate performance optimization recommendations"""
        recommendations = []

        for operation, stats in performance_summary.items():
            duration_stats = stats.get("duration_stats", {})
            memory_stats = stats.get("memory_stats", {})
            cpu_stats = stats.get("cpu_stats", {})

            # Duration recommendations
            p95_duration = duration_stats.get("p95_ms", 0)
            if operation in self.thresholds:
                threshold = self.thresholds[operation]

                if p95_duration > threshold.max_duration_ms:
                    if "search" in operation:
                        recommendations.append(
                            f"Optimize {operation}: Consider implementing result caching or index optimization"
                        )
                    elif "embedding" in operation:
                        recommendations.append(
                            f"Optimize {operation}: Consider batch processing or GPU acceleration"
                        )

                # Memory recommendations
                max_memory = memory_stats.get("max_mb", 0)
                if max_memory > threshold.max_memory_mb:
                    recommendations.append(
                        f"Reduce memory usage in {operation}: Implement memory pooling or streaming processing"
                    )

                # CPU recommendations
                max_cpu = cpu_stats.get("max_percent", 0)
                if max_cpu > threshold.max_cpu_percent:
                    recommendations.append(
                        f"Optimize CPU usage in {operation}: Consider async processing or algorithm optimization"
                    )

        # General recommendations
        if len(recommendations) == 0:
            recommendations.append(
                "Performance is within acceptable thresholds. Monitor trends for proactive optimization."
            )

        return recommendations

    def save_performance_data(self, performance_summary: dict[str, Any], filepath: str):
        """Save performance data for historical analysis"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "performance_summary": performance_summary,
            "alerts": [asdict(alert) for alert in self.alerts],
            "thresholds": {
                name: asdict(threshold) for name, threshold in self.thresholds.items()
            },
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Performance data saved to {filepath}")

    def load_historical_data(self, data_dir: str) -> list[dict[str, Any]]:
        """Load historical performance data for trend analysis"""
        historical_data = []
        data_path = Path(data_dir)

        if not data_path.exists():
            logger.warning(f"Historical data directory not found: {data_dir}")
            return historical_data

        # Load all JSON files in the directory
        for json_file in data_path.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    if "performance_summary" in data:
                        historical_data.append(data["performance_summary"])
            except Exception as e:
                logger.error(f"Failed to load historical data from {json_file}: {e}")

        logger.info(f"Loaded {len(historical_data)} historical data points")
        return historical_data


class PerformanceMonitor:
    """Real-time performance monitoring"""

    def __init__(self, analyzer: PerformanceAnalyzer):
        self.analyzer = analyzer
        self.monitoring_active = False
        self.performance_log = []

    async def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous performance monitoring"""
        self.monitoring_active = True
        logger.info(
            f"Starting performance monitoring with {interval_seconds}s interval"
        )

        while self.monitoring_active:
            try:
                # Collect current performance metrics
                metrics = await self._collect_current_metrics()
                self.performance_log.append(
                    {"timestamp": datetime.now(), "metrics": metrics}
                )

                # Analyze for alerts
                alerts = self.analyzer.analyze_performance_data(metrics)
                if alerts:
                    await self._handle_alerts(alerts)

                await asyncio.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(interval_seconds)

    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        logger.info("Performance monitoring stopped")

    async def _collect_current_metrics(self) -> dict[str, Any]:
        """Collect current system performance metrics"""
        import psutil

        # This would integrate with actual system metrics
        # For now, return basic system stats
        return {
            "system_memory": {
                "duration_stats": {"mean_ms": 0, "p95_ms": 0},
                "memory_stats": {
                    "mean_mb": psutil.virtual_memory().percent,
                    "max_mb": 0,
                },
                "cpu_stats": {"mean_percent": psutil.cpu_percent(), "max_percent": 0},
                "success_rate": 1.0,
                "total_runs": 1,
            }
        }

    async def _handle_alerts(self, alerts: list[PerformanceAlert]):
        """Handle performance alerts"""
        for alert in alerts:
            logger.warning(
                f"Performance Alert: {alert.operation} - {alert.metric_type} = {alert.current_value} "
                f"(threshold: {alert.threshold_value}) - {alert.severity}"
            )

            # Here you could integrate with alerting systems
            # - Send email notifications
            # - Post to Slack/Teams
            # - Create monitoring dashboard alerts
            # - Trigger auto-scaling if available
