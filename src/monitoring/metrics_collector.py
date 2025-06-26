"""
Metrics Collection and Monitoring for TradeKnowledge.

This module provides comprehensive metrics collection, aggregation,
and real-time monitoring capabilities for system health and performance.
"""

import asyncio
import json
import logging
import threading
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import psutil

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected"""

    COUNTER = "counter"  # Monotonically increasing values
    GAUGE = "gauge"  # Current value that can go up or down
    HISTOGRAM = "histogram"  # Distribution of values
    TIMER = "timer"  # Duration measurements
    RATE = "rate"  # Rate per time unit


@dataclass
class MetricValue:
    """Individual metric measurement"""

    timestamp: datetime
    value: int | float
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricDefinition:
    """Definition of a metric"""

    name: str
    metric_type: MetricType
    description: str
    unit: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    retention_hours: int = 24


class MetricsCollector:
    """
    Core metrics collection and storage system.

    Collects, aggregates, and stores metrics from various system components
    with configurable retention and sampling policies.
    """

    def __init__(self, max_retention_hours: int = 24, max_series_length: int = 10000):
        self.metrics: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_series_length)
        )
        self.metric_definitions: dict[str, MetricDefinition] = {}
        self.max_retention_hours = max_retention_hours
        self.max_series_length = max_series_length
        self._lock = threading.RLock()
        self._cleanup_task = None
        self._collection_handlers: dict[str, Callable] = {}

        # Built-in system metrics
        self._setup_system_metrics()

        logger.info(
            f"MetricsCollector initialized with {max_retention_hours}h retention"
        )

    def _setup_system_metrics(self):
        """Set up built-in system metrics"""
        system_metrics = [
            MetricDefinition(
                "system.cpu.percent", MetricType.GAUGE, "CPU usage percentage", "%"
            ),
            MetricDefinition(
                "system.memory.used_mb", MetricType.GAUGE, "Memory usage in MB", "MB"
            ),
            MetricDefinition(
                "system.memory.percent",
                MetricType.GAUGE,
                "Memory usage percentage",
                "%",
            ),
            MetricDefinition(
                "system.disk.used_percent",
                MetricType.GAUGE,
                "Disk usage percentage",
                "%",
            ),
            MetricDefinition(
                "system.load.avg_1m", MetricType.GAUGE, "1-minute load average", ""
            ),
            MetricDefinition(
                "api.requests.total",
                MetricType.COUNTER,
                "Total API requests",
                "requests",
            ),
            MetricDefinition(
                "api.requests.errors",
                MetricType.COUNTER,
                "API request errors",
                "errors",
            ),
            MetricDefinition(
                "api.response_time", MetricType.HISTOGRAM, "API response time", "ms"
            ),
            MetricDefinition(
                "search.queries.total",
                MetricType.COUNTER,
                "Total search queries",
                "queries",
            ),
            MetricDefinition(
                "search.query_time", MetricType.HISTOGRAM, "Search query duration", "ms"
            ),
            MetricDefinition(
                "embedding.generation.total",
                MetricType.COUNTER,
                "Total embeddings generated",
                "embeddings",
            ),
            MetricDefinition(
                "embedding.generation_time",
                MetricType.HISTOGRAM,
                "Embedding generation time",
                "ms",
            ),
            MetricDefinition(
                "circuit_breaker.state",
                MetricType.GAUGE,
                "Circuit breaker state (0=closed, 1=half-open, 2=open)",
                "",
            ),
            MetricDefinition(
                "circuit_breaker.failures",
                MetricType.COUNTER,
                "Circuit breaker failures",
                "failures",
            ),
        ]

        for metric_def in system_metrics:
            self.register_metric(metric_def)

    def register_metric(self, metric_def: MetricDefinition):
        """Register a new metric definition"""
        with self._lock:
            self.metric_definitions[metric_def.name] = metric_def
            logger.debug(
                f"Registered metric: {metric_def.name} ({metric_def.metric_type.value})"
            )

    def record_metric(
        self,
        name: str,
        value: int | float,
        tags: dict[str, str] | None = None,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Record a metric value.

        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for the metric
            timestamp: Optional timestamp (defaults to now)
            metadata: Optional metadata
        """
        if timestamp is None:
            timestamp = datetime.now()

        if tags is None:
            tags = {}

        if metadata is None:
            metadata = {}

        metric_value = MetricValue(
            timestamp=timestamp, value=value, tags=tags, metadata=metadata
        )

        with self._lock:
            self.metrics[name].append(metric_value)

            # Handle counter-specific logic
            if name in self.metric_definitions:
                metric_def = self.metric_definitions[name]
                if metric_def.metric_type == MetricType.COUNTER:
                    # For counters, also update rate metrics
                    self._update_rate_metric(name, value, timestamp)

        logger.debug(f"Recorded metric {name}: {value} at {timestamp}")

    def _update_rate_metric(
        self, counter_name: str, value: int | float, timestamp: datetime
    ):
        """Update rate metric for a counter"""
        rate_name = f"{counter_name}.rate"

        # Calculate rate based on recent values
        counter_series = self.metrics[counter_name]
        if len(counter_series) >= 2:
            recent_values = list(counter_series)[-2:]
            time_diff = (
                recent_values[1].timestamp - recent_values[0].timestamp
            ).total_seconds()
            if time_diff > 0:
                value_diff = recent_values[1].value - recent_values[0].value
                rate = value_diff / time_diff

                rate_metric = MetricValue(
                    timestamp=timestamp,
                    value=rate,
                    tags={"source": counter_name},
                    metadata={"time_window_seconds": time_diff},
                )

                self.metrics[rate_name].append(rate_metric)

    def get_metric_values(
        self,
        name: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        tags_filter: dict[str, str] | None = None,
    ) -> list[MetricValue]:
        """
        Get metric values within time range and tag filters.

        Args:
            name: Metric name
            start_time: Start time filter (defaults to all time)
            end_time: End time filter (defaults to all time)
            tags_filter: Tag filters

        Returns:
            List of metric values matching criteria
        """
        with self._lock:
            if name not in self.metrics:
                return []

            filtered_values = []
            for metric_value in self.metrics[name]:
                # Time filter
                if start_time and metric_value.timestamp < start_time:
                    continue
                if end_time and metric_value.timestamp > end_time:
                    continue

                # Tags filter
                if tags_filter:
                    if not all(
                        metric_value.tags.get(key) == value
                        for key, value in tags_filter.items()
                    ):
                        continue

                filtered_values.append(metric_value)

            return filtered_values

    def get_metric_summary(
        self,
        name: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Get statistical summary of a metric.

        Args:
            name: Metric name
            start_time: Start time for analysis
            end_time: End time for analysis

        Returns:
            Dictionary with statistical summary
        """
        values = self.get_metric_values(name, start_time, end_time)

        if not values:
            return {"metric_name": name, "count": 0, "error": "No data points found"}

        numeric_values = [v.value for v in values]

        summary = {
            "metric_name": name,
            "count": len(numeric_values),
            "min": min(numeric_values),
            "max": max(numeric_values),
            "mean": sum(numeric_values) / len(numeric_values),
            "latest": numeric_values[-1],
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": end_time.isoformat() if end_time else None,
        }

        # Add percentiles for larger datasets
        if len(numeric_values) >= 10:
            sorted_values = sorted(numeric_values)
            summary.update(
                {
                    "p50": self._percentile(sorted_values, 50),
                    "p95": self._percentile(sorted_values, 95),
                    "p99": self._percentile(sorted_values, 99),
                }
            )

        return summary

    def _percentile(self, sorted_values: list[float], percentile: int) -> float:
        """Calculate percentile of sorted values"""
        if not sorted_values:
            return 0

        index = (percentile / 100) * (len(sorted_values) - 1)
        if index == int(index):
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))

    def get_all_metrics_summary(self) -> dict[str, dict[str, Any]]:
        """Get summary for all metrics"""
        with self._lock:
            summaries = {}
            for metric_name in self.metrics.keys():
                summaries[metric_name] = self.get_metric_summary(metric_name)
            return summaries

    def collect_system_metrics(self):
        """Collect current system metrics"""
        try:
            now = datetime.now()

            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            self.record_metric("system.cpu.percent", cpu_percent, timestamp=now)

            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_metric(
                "system.memory.used_mb", memory.used / 1024 / 1024, timestamp=now
            )
            self.record_metric("system.memory.percent", memory.percent, timestamp=now)

            # Disk metrics
            disk = psutil.disk_usage("/")
            disk_percent = (disk.used / disk.total) * 100
            self.record_metric("system.disk.used_percent", disk_percent, timestamp=now)

            # Load average (if available)
            if hasattr(psutil, "getloadavg"):
                load_avg = psutil.getloadavg()
                self.record_metric("system.load.avg_1m", load_avg[0], timestamp=now)

            logger.debug("Collected system metrics")

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    def start_automatic_collection(self, interval_seconds: int = 30):
        """Start automatic metrics collection"""
        if self._cleanup_task and not self._cleanup_task.done():
            logger.warning("Automatic collection already running")
            return

        async def collection_loop():
            while True:
                try:
                    self.collect_system_metrics()
                    await asyncio.sleep(interval_seconds)
                except asyncio.CancelledError:
                    logger.info("Metrics collection stopped")
                    break
                except Exception as e:
                    logger.error(f"Error in metrics collection loop: {e}")
                    await asyncio.sleep(interval_seconds)

        self._cleanup_task = asyncio.create_task(collection_loop())
        logger.info(
            f"Started automatic metrics collection (interval: {interval_seconds}s)"
        )

    def stop_automatic_collection(self):
        """Stop automatic metrics collection"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            logger.info("Stopped automatic metrics collection")

    def cleanup_old_metrics(self, older_than_hours: int | None = None):
        """Remove metrics older than specified hours"""
        if older_than_hours is None:
            older_than_hours = self.max_retention_hours

        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)

        with self._lock:
            total_removed = 0
            for metric_name, metric_series in self.metrics.items():
                original_length = len(metric_series)

                # Remove old values
                while metric_series and metric_series[0].timestamp < cutoff_time:
                    metric_series.popleft()

                removed = original_length - len(metric_series)
                total_removed += removed

            logger.info(
                f"Cleaned up {total_removed} old metric values (older than {older_than_hours}h)"
            )

    def export_metrics(self, format_type: str = "json") -> str:
        """
        Export metrics in specified format.

        Args:
            format_type: Export format ("json", "prometheus")

        Returns:
            Exported metrics as string
        """
        if format_type == "json":
            return self._export_json()
        elif format_type == "prometheus":
            return self._export_prometheus()
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

    def _export_json(self) -> str:
        """Export metrics in JSON format"""
        export_data = {"timestamp": datetime.now().isoformat(), "metrics": {}}

        with self._lock:
            for metric_name, metric_series in self.metrics.items():
                export_data["metrics"][metric_name] = [
                    {
                        "timestamp": mv.timestamp.isoformat(),
                        "value": mv.value,
                        "tags": mv.tags,
                        "metadata": mv.metadata,
                    }
                    for mv in metric_series
                ]

        return json.dumps(export_data, indent=2)

    def _export_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []

        with self._lock:
            for metric_name, metric_series in self.metrics.items():
                if not metric_series:
                    continue

                # Get metric definition for help text
                help_text = ""
                metric_type = "gauge"
                if metric_name in self.metric_definitions:
                    help_text = self.metric_definitions[metric_name].description
                    metric_type = self.metric_definitions[metric_name].metric_type.value

                # Add help and type comments
                safe_name = metric_name.replace(".", "_").replace("-", "_")
                lines.append(f"# HELP {safe_name} {help_text}")
                lines.append(f"# TYPE {safe_name} {metric_type}")

                # Get latest value
                latest_value = metric_series[-1]

                # Format tags
                tag_str = ""
                if latest_value.tags:
                    tag_pairs = [f'{k}="{v}"' for k, v in latest_value.tags.items()]
                    tag_str = "{" + ",".join(tag_pairs) + "}"

                lines.append(f"{safe_name}{tag_str} {latest_value.value}")

        return "\n".join(lines)


class AlertRule:
    """Definition of an alerting rule"""

    def __init__(
        self,
        name: str,
        metric_name: str,
        condition: str,
        threshold: int | float,
        duration_minutes: int = 1,
        severity: str = "warning",
        description: str = "",
        tags: dict[str, str] | None = None,
    ):
        self.name = name
        self.metric_name = metric_name
        self.condition = (
            condition  # "greater_than", "less_than", "equals", "not_equals"
        )
        self.threshold = threshold
        self.duration_minutes = duration_minutes
        self.severity = severity
        self.description = description
        self.tags = tags or {}
        self.last_triggered = None
        self.is_firing = False


@dataclass
class Alert:
    """Active alert instance"""

    rule_name: str
    metric_name: str
    current_value: int | float
    threshold: int | float
    condition: str
    severity: str
    description: str
    started_at: datetime
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class AlertManager:
    """
    Alert management system for monitoring metrics and triggering alerts.

    Evaluates alert rules against metrics and manages alert lifecycle.
    """

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules: dict[str, AlertRule] = {}
        self.active_alerts: dict[str, Alert] = {}
        self.alert_history: list[Alert] = []
        self._evaluation_task = None
        self._alert_handlers: list[Callable[[Alert], None]] = []

        # Set up default alert rules
        self._setup_default_alert_rules()

        logger.info("AlertManager initialized")

    def _setup_default_alert_rules(self):
        """Set up default alert rules for system monitoring"""
        default_rules = [
            AlertRule(
                name="high_cpu_usage",
                metric_name="system.cpu.percent",
                condition="greater_than",
                threshold=80.0,
                duration_minutes=2,
                severity="warning",
                description="CPU usage is above 80%",
            ),
            AlertRule(
                name="high_memory_usage",
                metric_name="system.memory.percent",
                condition="greater_than",
                threshold=85.0,
                duration_minutes=1,
                severity="warning",
                description="Memory usage is above 85%",
            ),
            AlertRule(
                name="high_disk_usage",
                metric_name="system.disk.used_percent",
                condition="greater_than",
                threshold=90.0,
                duration_minutes=5,
                severity="critical",
                description="Disk usage is above 90%",
            ),
            AlertRule(
                name="api_error_rate_high",
                metric_name="api.requests.errors.rate",
                condition="greater_than",
                threshold=10.0,
                duration_minutes=1,
                severity="critical",
                description="API error rate is above 10 errors/second",
            ),
            AlertRule(
                name="search_response_time_slow",
                metric_name="search.query_time",
                condition="greater_than",
                threshold=1000.0,  # 1 second
                duration_minutes=2,
                severity="warning",
                description="Search response time is above 1 second",
            ),
        ]

        for rule in default_rules:
            self.add_alert_rule(rule)

    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule"""
        self.alert_rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name} for metric {rule.metric_name}")

    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule"""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")

    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add a handler function for alerts"""
        self._alert_handlers.append(handler)
        handler_name = getattr(handler, "__name__", "unknown_handler")
        logger.info(f"Added alert handler: {handler_name}")

    def evaluate_alert_rules(self):
        """Evaluate all alert rules against current metrics"""
        now = datetime.now()

        for rule in self.alert_rules.values():
            try:
                self._evaluate_single_rule(rule, now)
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule.name}: {e}")

    def _evaluate_single_rule(self, rule: AlertRule, evaluation_time: datetime):
        """Evaluate a single alert rule"""
        # Get recent metric values for the duration window
        start_time = evaluation_time - timedelta(minutes=rule.duration_minutes)
        metric_values = self.metrics_collector.get_metric_values(
            rule.metric_name, start_time=start_time, end_time=evaluation_time
        )

        if not metric_values:
            # No data available for evaluation
            # If alert was firing and we have no data, keep it firing
            return

        # Check if condition is met consistently
        latest_value = metric_values[-1].value if metric_values else None

        # For triggering: all values in window must meet condition
        # For resolving: any value not meeting condition resolves the alert
        condition_met_consistently = True
        if metric_values:
            for metric_value in metric_values:
                if not self._check_condition(
                    metric_value.value, rule.condition, rule.threshold
                ):
                    condition_met_consistently = False
                    break

        # Handle alert state changes
        if (
            condition_met_consistently
            and rule.name not in self.active_alerts
            and latest_value is not None
        ):
            # Trigger new alert
            alert = Alert(
                rule_name=rule.name,
                metric_name=rule.metric_name,
                current_value=latest_value,
                threshold=rule.threshold,
                condition=rule.condition,
                severity=rule.severity,
                description=rule.description,
                started_at=evaluation_time,
                tags=rule.tags.copy(),
            )

            self.active_alerts[rule.name] = alert
            self.alert_history.append(alert)
            rule.last_triggered = evaluation_time
            rule.is_firing = True

            # Notify handlers
            for handler in self._alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Error in alert handler: {e}")

            logger.warning(
                f"ALERT TRIGGERED: {rule.name} - {rule.description} "
                f"(current: {latest_value}, threshold: {rule.threshold})"
            )

        elif not condition_met_consistently and rule.name in self.active_alerts:
            # Resolve alert
            del self.active_alerts[rule.name]
            rule.is_firing = False

            logger.info(f"ALERT RESOLVED: {rule.name}")

    def _check_condition(
        self, value: int | float, condition: str, threshold: int | float
    ) -> bool:
        """Check if a condition is met"""
        if condition == "greater_than":
            return value > threshold
        elif condition == "less_than":
            return value < threshold
        elif condition == "equals":
            return value == threshold
        elif condition == "not_equals":
            return value != threshold
        elif condition == "greater_than_or_equal":
            return value >= threshold
        elif condition == "less_than_or_equal":
            return value <= threshold
        else:
            logger.error(f"Unknown condition: {condition}")
            return False

    def start_alert_evaluation(self, interval_seconds: int = 30):
        """Start automatic alert rule evaluation"""
        if self._evaluation_task and not self._evaluation_task.done():
            logger.warning("Alert evaluation already running")
            return

        async def evaluation_loop():
            while True:
                try:
                    self.evaluate_alert_rules()
                    await asyncio.sleep(interval_seconds)
                except asyncio.CancelledError:
                    logger.info("Alert evaluation stopped")
                    break
                except Exception as e:
                    logger.error(f"Error in alert evaluation loop: {e}")
                    await asyncio.sleep(interval_seconds)

        self._evaluation_task = asyncio.create_task(evaluation_loop())
        logger.info(f"Started alert evaluation (interval: {interval_seconds}s)")

    def stop_alert_evaluation(self):
        """Stop automatic alert evaluation"""
        if self._evaluation_task:
            self._evaluation_task.cancel()
            logger.info("Stopped alert evaluation")

    def get_active_alerts(self) -> list[Alert]:
        """Get list of currently active alerts"""
        return list(self.active_alerts.values())

    def get_alert_summary(self) -> dict[str, Any]:
        """Get summary of alert status"""
        active_alerts = self.get_active_alerts()

        severity_counts = defaultdict(int)
        for alert in active_alerts:
            severity_counts[alert.severity] += 1

        return {
            "total_active_alerts": len(active_alerts),
            "severity_breakdown": dict(severity_counts),
            "total_rules": len(self.alert_rules),
            "firing_rules": len([r for r in self.alert_rules.values() if r.is_firing]),
            "total_historical_alerts": len(self.alert_history),
        }


# Convenience functions and default alert handlers
def console_alert_handler(alert: Alert):
    """Simple console alert handler"""
    print(f"[{alert.severity.upper()}] {alert.rule_name}: {alert.description}")
    print(
        f"  Metric: {alert.metric_name} = {alert.current_value} (threshold: {alert.threshold})"
    )
    print(f"  Started: {alert.started_at}")


def log_alert_handler(alert: Alert):
    """Log-based alert handler"""
    log_level = logging.ERROR if alert.severity == "critical" else logging.WARNING
    logger.log(
        log_level,
        f"Alert {alert.rule_name}: {alert.description} "
        f"({alert.metric_name}={alert.current_value}, threshold={alert.threshold})",
    )


# Global instances
_metrics_collector = None
_alert_manager = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def get_alert_manager() -> AlertManager:
    """Get global alert manager instance"""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager(get_metrics_collector())
    return _alert_manager


def setup_monitoring(auto_collect_interval: int = 30, alert_eval_interval: int = 30):
    """Initialize monitoring system with default settings"""
    collector = get_metrics_collector()
    alert_manager = get_alert_manager()

    # Add default alert handlers
    alert_manager.add_alert_handler(log_alert_handler)

    # Start automatic collection and evaluation
    collector.start_automatic_collection(auto_collect_interval)
    alert_manager.start_alert_evaluation(alert_eval_interval)

    logger.info("Monitoring system initialized and started")
