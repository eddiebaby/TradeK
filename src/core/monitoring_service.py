"""
Comprehensive Monitoring and Observability Service
Provides real-time monitoring, metrics collection, alerting, and observability
"""

import asyncio
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import psutil
import structlog

from ..core.config import get_config
from .database_optimization_service import get_optimization_service
from .database_optimizer import get_database_optimizer

logger = structlog.get_logger(__name__)


@dataclass
class MetricPoint:
    """A single metric data point"""

    timestamp: datetime
    value: float
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """System alert definition"""

    id: str
    name: str
    description: str
    severity: str  # critical, warning, info
    condition: str
    threshold: float
    triggered_at: datetime | None = None
    resolved_at: datetime | None = None
    status: str = "active"  # active, triggered, resolved


@dataclass
class HealthCheck:
    """Component health check"""

    component: str
    status: str  # healthy, degraded, unhealthy
    last_check: datetime
    response_time_ms: float
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and stores system metrics"""

    def __init__(self, retention_minutes: int = 60):
        self.metrics: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=retention_minutes * 60)
        )  # 1 per second
        self.retention_minutes = retention_minutes

    def record_metric(
        self, metric_name: str, value: float, labels: dict[str, str] = None
    ):
        """Record a metric value"""
        point = MetricPoint(
            timestamp=datetime.utcnow(), value=value, labels=labels or {}
        )
        self.metrics[metric_name].append(point)

    def get_metric_history(
        self, metric_name: str, minutes: int = 10
    ) -> list[MetricPoint]:
        """Get metric history for specified time period"""
        if metric_name not in self.metrics:
            return []

        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        return [
            point
            for point in self.metrics[metric_name]
            if point.timestamp >= cutoff_time
        ]

    def get_metric_summary(
        self, metric_name: str, minutes: int = 5
    ) -> dict[str, float]:
        """Get metric summary statistics"""
        history = self.get_metric_history(metric_name, minutes)
        if not history:
            return {}

        values = [point.value for point in history]
        return {
            "current": values[-1] if values else 0,
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "count": len(values),
        }


class AlertManager:
    """Manages system alerts and notifications"""

    def __init__(self):
        self.alerts: dict[str, Alert] = {}
        self.alert_handlers: list[Callable] = []
        self.metrics_collector: MetricsCollector | None = None

    def register_alert(self, alert: Alert):
        """Register a new alert"""
        self.alerts[alert.id] = alert
        logger.info("Registered alert", alert_id=alert.id, name=alert.name)

    def register_handler(self, handler: Callable[[Alert], None]):
        """Register an alert handler"""
        self.alert_handlers.append(handler)

    def set_metrics_collector(self, collector: MetricsCollector):
        """Set the metrics collector for alert evaluation"""
        self.metrics_collector = collector

    async def evaluate_alerts(self):
        """Evaluate all alerts against current metrics"""
        if not self.metrics_collector:
            return

        for alert in self.alerts.values():
            await self._evaluate_alert(alert)

    async def _evaluate_alert(self, alert: Alert):
        """Evaluate a single alert"""
        try:
            # Get recent metric data
            metric_name = alert.condition.split()[0]  # Simple parsing
            summary = self.metrics_collector.get_metric_summary(metric_name, minutes=1)

            if not summary:
                return

            current_value = summary["current"]

            # Simple threshold evaluation
            should_trigger = False
            if "greater_than" in alert.condition:
                should_trigger = current_value > alert.threshold
            elif "less_than" in alert.condition:
                should_trigger = current_value < alert.threshold

            # Handle alert state changes
            if should_trigger and alert.status != "triggered":
                await self._trigger_alert(alert, current_value)
            elif not should_trigger and alert.status == "triggered":
                await self._resolve_alert(alert, current_value)

        except Exception as e:
            logger.error("Failed to evaluate alert", alert_id=alert.id, error=str(e))

    async def _trigger_alert(self, alert: Alert, value: float):
        """Trigger an alert"""
        alert.status = "triggered"
        alert.triggered_at = datetime.utcnow()

        logger.warning(
            "Alert triggered",
            alert_id=alert.id,
            name=alert.name,
            severity=alert.severity,
            value=value,
            threshold=alert.threshold,
        )

        # Notify handlers
        for handler in self.alert_handlers:
            try:
                (
                    await handler(alert)
                    if asyncio.iscoroutinefunction(handler)
                    else handler(alert)
                )
            except Exception as e:
                logger.error("Alert handler failed", error=str(e))

    async def _resolve_alert(self, alert: Alert, value: float):
        """Resolve an alert"""
        alert.status = "resolved"
        alert.resolved_at = datetime.utcnow()

        logger.info("Alert resolved", alert_id=alert.id, name=alert.name, value=value)


class HealthMonitor:
    """Monitors component health"""

    def __init__(self):
        self.health_checks: dict[str, HealthCheck] = {}
        self.check_functions: dict[str, Callable] = {}

    def register_health_check(self, component: str, check_function: Callable):
        """Register a health check for a component"""
        self.check_functions[component] = check_function
        logger.info("Registered health check", component=component)

    async def run_health_checks(self) -> dict[str, HealthCheck]:
        """Run all health checks"""
        results = {}

        for component, check_function in self.check_functions.items():
            try:
                start_time = time.time()

                if asyncio.iscoroutinefunction(check_function):
                    health_data = await check_function()
                else:
                    health_data = check_function()

                response_time = (time.time() - start_time) * 1000  # Convert to ms

                health_check = HealthCheck(
                    component=component,
                    status=health_data.get("status", "unknown"),
                    last_check=datetime.utcnow(),
                    response_time_ms=response_time,
                    error_message=health_data.get("error"),
                    metadata=health_data.get("metadata", {}),
                )

                self.health_checks[component] = health_check
                results[component] = health_check

            except Exception as e:
                health_check = HealthCheck(
                    component=component,
                    status="unhealthy",
                    last_check=datetime.utcnow(),
                    response_time_ms=0,
                    error_message=str(e),
                )

                self.health_checks[component] = health_check
                results[component] = health_check

                logger.error("Health check failed", component=component, error=str(e))

        return results

    def get_overall_health(self) -> str:
        """Get overall system health status"""
        if not self.health_checks:
            return "unknown"

        statuses = [check.status for check in self.health_checks.values()]

        if any(status == "unhealthy" for status in statuses):
            return "unhealthy"
        elif any(status == "degraded" for status in statuses):
            return "degraded"
        elif all(status == "healthy" for status in statuses):
            return "healthy"
        else:
            return "unknown"


class MonitoringService:
    """Main monitoring and observability service"""

    def __init__(self):
        self.config = get_config()
        self.is_running = False
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.health_monitor = HealthMonitor()

        # Background tasks
        self.metrics_task = None
        self.alerts_task = None
        self.health_task = None

        # Performance tracking
        self.request_times = deque(maxlen=1000)
        self.error_counts = defaultdict(int)

        # System metrics
        self.process = psutil.Process()

    async def initialize(self):
        """Initialize monitoring service"""
        try:
            logger.info("Initializing monitoring service")

            # Setup alert manager
            self.alert_manager.set_metrics_collector(self.metrics_collector)

            # Register default alerts
            await self._register_default_alerts()

            # Register default health checks
            await self._register_default_health_checks()

            # Start background tasks
            await self._start_background_tasks()

            self.is_running = True
            logger.info("✅ Monitoring service initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize monitoring service", error=str(e))
            raise

    async def _register_default_alerts(self):
        """Register default system alerts"""
        alerts = [
            Alert(
                id="high_cpu_usage",
                name="High CPU Usage",
                description="CPU usage is above 80%",
                severity="warning",
                condition="cpu_usage_percent greater_than",
                threshold=80.0,
            ),
            Alert(
                id="high_memory_usage",
                name="High Memory Usage",
                description="Memory usage is above 85%",
                severity="warning",
                condition="memory_usage_percent greater_than",
                threshold=85.0,
            ),
            Alert(
                id="slow_response_time",
                name="Slow Response Time",
                description="Average response time is above 2 seconds",
                severity="warning",
                condition="avg_response_time_ms greater_than",
                threshold=2000.0,
            ),
            Alert(
                id="high_error_rate",
                name="High Error Rate",
                description="Error rate is above 5%",
                severity="critical",
                condition="error_rate_percent greater_than",
                threshold=5.0,
            ),
            Alert(
                id="low_cache_hit_rate",
                name="Low Cache Hit Rate",
                description="Cache hit rate is below 30%",
                severity="warning",
                condition="cache_hit_rate_percent less_than",
                threshold=30.0,
            ),
        ]

        for alert in alerts:
            self.alert_manager.register_alert(alert)

    async def _register_default_health_checks(self):
        """Register default health checks"""

        async def database_health():
            """Check database optimization service health"""
            try:
                optimization_service = await get_optimization_service()
                if optimization_service.is_initialized:
                    return {"status": "healthy", "metadata": {"service_running": True}}
                else:
                    return {
                        "status": "degraded",
                        "metadata": {"service_running": False},
                    }
            except Exception as e:
                return {"status": "unhealthy", "error": str(e)}

        async def database_optimizer_health():
            """Check database optimizer health"""
            try:
                optimizer = await get_database_optimizer()
                cache_stats = optimizer.query_cache.get_stats()
                return {
                    "status": "healthy",
                    "metadata": {
                        "cache_hit_rate": cache_stats.get("hit_rate", 0),
                        "total_queries": len(optimizer.query_metrics),
                    },
                }
            except Exception as e:
                return {"status": "unhealthy", "error": str(e)}

        def disk_space_health():
            """Check disk space"""
            try:
                disk_usage = psutil.disk_usage("/")
                usage_percent = (disk_usage.used / disk_usage.total) * 100

                if usage_percent > 90:
                    status = "unhealthy"
                elif usage_percent > 80:
                    status = "degraded"
                else:
                    status = "healthy"

                return {
                    "status": status,
                    "metadata": {
                        "usage_percent": usage_percent,
                        "free_gb": disk_usage.free / (1024**3),
                    },
                }
            except Exception as e:
                return {"status": "unhealthy", "error": str(e)}

        self.health_monitor.register_health_check(
            "database_optimization", database_health
        )
        self.health_monitor.register_health_check(
            "database_optimizer", database_optimizer_health
        )
        self.health_monitor.register_health_check("disk_space", disk_space_health)

    async def _start_background_tasks(self):
        """Start background monitoring tasks"""
        self.metrics_task = asyncio.create_task(self._metrics_collection_loop())
        self.alerts_task = asyncio.create_task(self._alerts_evaluation_loop())
        self.health_task = asyncio.create_task(self._health_monitoring_loop())

        logger.info("Started monitoring background tasks")

    async def _metrics_collection_loop(self):
        """Background metrics collection"""
        while self.is_running:
            try:
                await self._collect_system_metrics()
                await self._collect_application_metrics()
                await asyncio.sleep(10)  # Collect every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Metrics collection error", error=str(e))
                await asyncio.sleep(5)

    async def _alerts_evaluation_loop(self):
        """Background alert evaluation"""
        while self.is_running:
            try:
                await self.alert_manager.evaluate_alerts()
                await asyncio.sleep(30)  # Evaluate every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Alert evaluation error", error=str(e))
                await asyncio.sleep(5)

    async def _health_monitoring_loop(self):
        """Background health monitoring"""
        while self.is_running:
            try:
                await self.health_monitor.run_health_checks()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health monitoring error", error=str(e))
                await asyncio.sleep(10)

    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU metrics
            cpu_percent = self.process.cpu_percent()
            self.metrics_collector.record_metric("cpu_usage_percent", cpu_percent)

            # Memory metrics
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            self.metrics_collector.record_metric("memory_usage_percent", memory_percent)
            self.metrics_collector.record_metric(
                "memory_usage_mb", memory_info.rss / (1024 * 1024)
            )

            # Disk metrics
            disk_usage = psutil.disk_usage("/")
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            self.metrics_collector.record_metric("disk_usage_percent", disk_percent)

            # Network metrics (if available)
            try:
                network_io = psutil.net_io_counters()
                self.metrics_collector.record_metric(
                    "network_bytes_sent", network_io.bytes_sent
                )
                self.metrics_collector.record_metric(
                    "network_bytes_recv", network_io.bytes_recv
                )
            except:
                pass  # Network metrics not available on all systems

        except Exception as e:
            logger.warning("Failed to collect system metrics", error=str(e))

    async def _collect_application_metrics(self):
        """Collect application-specific metrics"""
        try:
            # Database optimization metrics
            optimization_service = await get_optimization_service()
            if optimization_service.is_initialized:
                optimizer = await get_database_optimizer()

                # Cache metrics
                cache_stats = optimizer.query_cache.get_stats()
                self.metrics_collector.record_metric(
                    "cache_hit_rate_percent", cache_stats.get("hit_rate", 0) * 100
                )
                self.metrics_collector.record_metric(
                    "cache_total_requests", cache_stats.get("total_requests", 0)
                )

                # Query performance metrics
                performance_report = optimizer.get_performance_report()
                if "query_performance" in performance_report:
                    total_queries = len(performance_report["query_performance"])
                    self.metrics_collector.record_metric(
                        "total_query_types", total_queries
                    )

                    # Average query time across all query types
                    avg_times = [
                        metrics.get("avg_time", 0)
                        for metrics in performance_report["query_performance"].values()
                    ]
                    if avg_times:
                        overall_avg = sum(avg_times) / len(avg_times)
                        self.metrics_collector.record_metric(
                            "avg_query_time_ms", overall_avg * 1000
                        )

                # Slow queries count
                slow_queries = performance_report.get("slow_queries", [])
                self.metrics_collector.record_metric(
                    "slow_queries_count", len(slow_queries)
                )

        except Exception as e:
            logger.warning("Failed to collect application metrics", error=str(e))

    def record_request(self, duration_ms: float, status_code: int):
        """Record API request metrics"""
        self.request_times.append(duration_ms)
        self.metrics_collector.record_metric("response_time_ms", duration_ms)
        self.metrics_collector.record_metric("requests_total", 1)

        if status_code >= 400:
            self.error_counts[status_code] += 1
            self.metrics_collector.record_metric("errors_total", 1)

        # Calculate error rate
        if len(self.request_times) > 0:
            total_requests = len(self.request_times)
            total_errors = sum(self.error_counts.values())
            error_rate = (total_errors / total_requests) * 100
            self.metrics_collector.record_metric("error_rate_percent", error_rate)

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get comprehensive metrics summary"""
        summary = {}

        # Key metrics with recent values
        key_metrics = [
            "cpu_usage_percent",
            "memory_usage_percent",
            "disk_usage_percent",
            "cache_hit_rate_percent",
            "avg_query_time_ms",
            "response_time_ms",
            "error_rate_percent",
        ]

        for metric in key_metrics:
            summary[metric] = self.metrics_collector.get_metric_summary(
                metric, minutes=5
            )

        return summary

    def get_dashboard_data(self) -> dict[str, Any]:
        """Get data for monitoring dashboard"""
        return {
            "system_health": self.health_monitor.get_overall_health(),
            "health_checks": {
                name: {
                    "status": check.status,
                    "response_time_ms": check.response_time_ms,
                    "last_check": check.last_check.isoformat(),
                    "error_message": check.error_message,
                }
                for name, check in self.health_monitor.health_checks.items()
            },
            "active_alerts": {
                alert_id: {
                    "name": alert.name,
                    "severity": alert.severity,
                    "status": alert.status,
                    "triggered_at": (
                        alert.triggered_at.isoformat() if alert.triggered_at else None
                    ),
                }
                for alert_id, alert in self.alert_manager.alerts.items()
                if alert.status == "triggered"
            },
            "metrics_summary": self.get_metrics_summary(),
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "disk_total_gb": psutil.disk_usage("/").total / (1024**3),
                "uptime_seconds": time.time() - psutil.boot_time(),
            },
            "generated_at": datetime.utcnow().isoformat(),
        }

    async def cleanup(self):
        """Cleanup monitoring service"""
        logger.info("Shutting down monitoring service")
        self.is_running = False

        # Cancel background tasks
        tasks = [self.metrics_task, self.alerts_task, self.health_task]
        for task in tasks:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info("✅ Monitoring service shutdown completed")


# Global monitoring service instance
monitoring_service = MonitoringService()


async def get_monitoring_service() -> MonitoringService:
    """Get the global monitoring service instance"""
    return monitoring_service


async def initialize_monitoring():
    """Initialize monitoring globally"""
    await monitoring_service.initialize()


async def cleanup_monitoring():
    """Cleanup monitoring globally"""
    await monitoring_service.cleanup()
