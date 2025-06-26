"""
Comprehensive Health Monitoring System

This module provides real-time health monitoring for all system components
including databases, services, and external dependencies.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import aiofiles
import psutil

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels"""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check"""

    component: str
    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "component": self.component,
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


@dataclass
class SystemHealth:
    """Overall system health status"""

    overall_status: HealthStatus
    components: dict[str, HealthCheckResult]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    uptime_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "overall_status": self.overall_status.value,
            "timestamp": self.timestamp.isoformat(),
            "uptime_seconds": self.uptime_seconds,
            "components": {
                name: result.to_dict() for name, result in self.components.items()
            },
        }


class HealthChecker:
    """Base class for health checkers"""

    def __init__(self, component_name: str, timeout: float = 5.0):
        self.component_name = component_name
        self.timeout = timeout

    async def check(self) -> HealthCheckResult:
        """Perform health check"""
        start_time = time.time()

        try:
            # Run the actual check with timeout
            result = await asyncio.wait_for(self._perform_check(), timeout=self.timeout)
            duration_ms = (time.time() - start_time) * 1000
            result.duration_ms = duration_ms
            return result
        except TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component=self.component_name,
                status=HealthStatus.CRITICAL,
                message=f"Health check timed out after {self.timeout}s",
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component=self.component_name,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                duration_ms=duration_ms,
                metadata={"error_type": type(e).__name__},
            )

    async def _perform_check(self) -> HealthCheckResult:
        """Override this method to implement specific health check"""
        raise NotImplementedError


class SQLiteHealthChecker(HealthChecker):
    """Health checker for SQLite database"""

    def __init__(self, db_path: str = "data/knowledge.db"):
        super().__init__("sqlite_database")
        self.db_path = db_path

    async def _perform_check(self) -> HealthCheckResult:
        """Check SQLite database health"""
        import os
        import sqlite3

        # Check if database file exists
        if not os.path.exists(self.db_path):
            return HealthCheckResult(
                component=self.component_name,
                status=HealthStatus.CRITICAL,
                message="Database file does not exist",
            )

        # Check file size
        file_size = os.path.getsize(self.db_path)

        # Try to connect and run a simple query
        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            cursor = conn.cursor()

            # Test basic functionality
            cursor.execute("SELECT COUNT(*) FROM books")
            book_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM chunks")
            chunk_count = cursor.fetchone()[0]

            conn.close()

            return HealthCheckResult(
                component=self.component_name,
                status=HealthStatus.HEALTHY,
                message="Database operational",
                metadata={
                    "book_count": book_count,
                    "chunk_count": chunk_count,
                    "file_size_mb": round(file_size / (1024 * 1024), 2),
                },
            )

        except sqlite3.OperationalError as e:
            return HealthCheckResult(
                component=self.component_name,
                status=HealthStatus.CRITICAL,
                message=f"Database operational error: {str(e)}",
            )


class QdrantHealthChecker(HealthChecker):
    """Health checker for Qdrant vector database"""

    def __init__(self, host: str = "localhost", port: int = 6333):
        super().__init__("qdrant_database")
        self.host = host
        self.port = port

    async def _perform_check(self) -> HealthCheckResult:
        """Check Qdrant health"""
        import httpx

        try:
            async with httpx.AsyncClient() as client:
                # Check health endpoint
                response = await client.get(f"http://{self.host}:{self.port}/health")

                if response.status_code == 200:
                    # Get collection info
                    collections_response = await client.get(
                        f"http://{self.host}:{self.port}/collections"
                    )
                    collections_data = collections_response.json()

                    collection_count = len(
                        collections_data.get("result", {}).get("collections", [])
                    )

                    return HealthCheckResult(
                        component=self.component_name,
                        status=HealthStatus.HEALTHY,
                        message="Qdrant operational",
                        metadata={
                            "collection_count": collection_count,
                            "version": response.headers.get("server", "unknown"),
                        },
                    )
                else:
                    return HealthCheckResult(
                        component=self.component_name,
                        status=HealthStatus.CRITICAL,
                        message=f"Qdrant health check failed: HTTP {response.status_code}",
                    )

        except httpx.ConnectError:
            return HealthCheckResult(
                component=self.component_name,
                status=HealthStatus.CRITICAL,
                message="Cannot connect to Qdrant service",
            )


class OllamaHealthChecker(HealthChecker):
    """Health checker for Ollama service"""

    def __init__(self, host: str = "http://localhost:11434"):
        super().__init__("ollama_service")
        self.host = host

    async def _perform_check(self) -> HealthCheckResult:
        """Check Ollama health"""
        import httpx

        try:
            async with httpx.AsyncClient() as client:
                # Check version endpoint
                response = await client.get(f"{self.host}/api/version")

                if response.status_code == 200:
                    version_data = response.json()

                    # Check available models
                    models_response = await client.get(f"{self.host}/api/tags")
                    models_data = models_response.json()

                    model_count = len(models_data.get("models", []))

                    return HealthCheckResult(
                        component=self.component_name,
                        status=HealthStatus.HEALTHY,
                        message="Ollama operational",
                        metadata={
                            "version": version_data.get("version", "unknown"),
                            "model_count": model_count,
                        },
                    )
                else:
                    return HealthCheckResult(
                        component=self.component_name,
                        status=HealthStatus.CRITICAL,
                        message=f"Ollama health check failed: HTTP {response.status_code}",
                    )

        except httpx.ConnectError:
            return HealthCheckResult(
                component=self.component_name,
                status=HealthStatus.CRITICAL,
                message="Cannot connect to Ollama service",
            )


class SystemResourceHealthChecker(HealthChecker):
    """Health checker for system resources"""

    def __init__(self, memory_threshold: float = 0.85, disk_threshold: float = 0.90):
        super().__init__("system_resources")
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold

    async def _perform_check(self) -> HealthCheckResult:
        """Check system resource health"""
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent / 100

        # Disk usage
        disk = psutil.disk_usage("/")
        disk_usage = (disk.total - disk.free) / disk.total

        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=1) / 100

        # Determine status
        if memory_usage > self.memory_threshold or disk_usage > self.disk_threshold:
            status = HealthStatus.CRITICAL
            message = "Resource usage critical"
        elif memory_usage > 0.7 or disk_usage > 0.8:
            status = HealthStatus.WARNING
            message = "Resource usage elevated"
        else:
            status = HealthStatus.HEALTHY
            message = "Resource usage normal"

        return HealthCheckResult(
            component=self.component_name,
            status=status,
            message=message,
            metadata={
                "memory_usage": round(memory_usage, 3),
                "disk_usage": round(disk_usage, 3),
                "cpu_usage": round(cpu_usage, 3),
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "disk_total_gb": round(disk.total / (1024**3), 2),
            },
        )


class FileSystemHealthChecker(HealthChecker):
    """Health checker for file system access"""

    def __init__(self, directories: list[str] = None):
        super().__init__("file_system")
        self.directories = directories or ["data", "logs", "cache"]

    async def _perform_check(self) -> HealthCheckResult:
        """Check file system health"""
        issues = []

        for directory in self.directories:
            # Check if directory exists
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory, exist_ok=True)
                except Exception as e:
                    issues.append(f"Cannot create {directory}: {e}")
                    continue

            # Check read/write access
            test_file = os.path.join(directory, ".health_check")
            try:
                async with aiofiles.open(test_file, "w") as f:
                    await f.write("health_check")

                async with aiofiles.open(test_file) as f:
                    content = await f.read()
                    if content != "health_check":
                        issues.append(f"Read/write verification failed for {directory}")

                os.remove(test_file)

            except Exception as e:
                issues.append(f"Cannot access {directory}: {e}")

        if issues:
            return HealthCheckResult(
                component=self.component_name,
                status=HealthStatus.CRITICAL,
                message=f"File system issues: {'; '.join(issues)}",
                metadata={"issues": issues},
            )
        else:
            return HealthCheckResult(
                component=self.component_name,
                status=HealthStatus.HEALTHY,
                message="File system access normal",
                metadata={"directories_checked": self.directories},
            )


class HealthMonitor:
    """
    Comprehensive health monitoring system.

    Features:
    - Configurable health checkers
    - Periodic health checks
    - Alert thresholds
    - Health history tracking
    """

    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.checkers: dict[str, HealthChecker] = {}
        self.health_history: list[SystemHealth] = []
        self.max_history = 100
        self.start_time = time.time()
        self.is_running = False
        self.monitor_task: asyncio.Task | None = None

        # Alert callbacks
        self.alert_callbacks: list[Callable[[HealthCheckResult], None]] = []

        # Initialize default checkers
        self._initialize_default_checkers()

    def _initialize_default_checkers(self):
        """Initialize default health checkers"""
        self.add_checker(SQLiteHealthChecker())
        self.add_checker(QdrantHealthChecker())
        self.add_checker(OllamaHealthChecker())
        self.add_checker(SystemResourceHealthChecker())
        self.add_checker(FileSystemHealthChecker())

    def add_checker(self, checker: HealthChecker):
        """Add a health checker"""
        self.checkers[checker.component_name] = checker
        logger.info(f"Added health checker: {checker.component_name}")

    def remove_checker(self, component_name: str):
        """Remove a health checker"""
        if component_name in self.checkers:
            del self.checkers[component_name]
            logger.info(f"Removed health checker: {component_name}")

    def add_alert_callback(self, callback: Callable[[HealthCheckResult], None]):
        """Add callback for health alerts"""
        self.alert_callbacks.append(callback)

    async def check_all_components(self) -> SystemHealth:
        """Run health checks on all components"""
        start_time = time.time()

        # Run all health checks concurrently
        tasks = {name: checker.check() for name, checker in self.checkers.items()}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        # Process results
        components = {}
        overall_status = HealthStatus.HEALTHY

        for (name, _), result in zip(tasks.items(), results, strict=False):
            if isinstance(result, Exception):
                # Handle checker exceptions
                components[name] = HealthCheckResult(
                    component=name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health checker failed: {str(result)}",
                )
            else:
                components[name] = result

            # Update overall status
            component_status = components[name].status
            if component_status == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
            elif (
                component_status == HealthStatus.WARNING
                and overall_status == HealthStatus.HEALTHY
            ):
                overall_status = HealthStatus.WARNING

        # Create system health summary
        uptime = time.time() - self.start_time
        system_health = SystemHealth(
            overall_status=overall_status, components=components, uptime_seconds=uptime
        )

        # Store in history
        self.health_history.append(system_health)
        if len(self.health_history) > self.max_history:
            self.health_history.pop(0)

        # Trigger alerts for critical/warning components
        for component_result in components.values():
            if component_result.status in [HealthStatus.CRITICAL, HealthStatus.WARNING]:
                for callback in self.alert_callbacks:
                    try:
                        await callback(component_result)
                    except Exception as e:
                        logger.error(f"Alert callback failed: {e}")

        logger.info(
            f"Health check completed: {overall_status.value} ({len(components)} components, {(time.time() - start_time)*1000:.1f}ms)"
        )

        return system_health

    async def start_monitoring(self):
        """Start continuous health monitoring"""
        if self.is_running:
            logger.warning("Health monitoring already running")
            return

        self.is_running = True
        logger.info(f"Starting health monitoring (interval: {self.check_interval}s)")

        async def monitor_loop():
            while self.is_running:
                try:
                    await self.check_all_components()
                    await asyncio.sleep(self.check_interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")
                    await asyncio.sleep(self.check_interval)

        self.monitor_task = asyncio.create_task(monitor_loop())

    async def stop_monitoring(self):
        """Stop health monitoring"""
        if not self.is_running:
            return

        self.is_running = False

        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("Health monitoring stopped")

    def get_current_health(self) -> SystemHealth | None:
        """Get most recent health status"""
        return self.health_history[-1] if self.health_history else None

    def get_health_history(self, hours: int = 24) -> list[SystemHealth]:
        """Get health history for specified hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            health for health in self.health_history if health.timestamp > cutoff_time
        ]

    def get_component_health(self, component_name: str) -> HealthCheckResult | None:
        """Get health status for specific component"""
        current_health = self.get_current_health()
        if current_health and component_name in current_health.components:
            return current_health.components[component_name]
        return None

    def get_stats(self) -> dict[str, Any]:
        """Get monitoring statistics"""
        current_health = self.get_current_health()

        stats = {
            "monitoring_active": self.is_running,
            "check_interval_seconds": self.check_interval,
            "checkers_count": len(self.checkers),
            "uptime_seconds": time.time() - self.start_time,
            "history_entries": len(self.health_history),
        }

        if current_health:
            stats["current_status"] = current_health.overall_status.value
            stats["last_check"] = current_health.timestamp.isoformat()

            # Component status summary
            status_counts = {}
            for component in current_health.components.values():
                status = component.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            stats["component_status_counts"] = status_counts

        return stats


# Global health monitor instance
_health_monitor = None


def get_health_monitor() -> HealthMonitor:
    """Get the global health monitor instance"""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor


# Convenience functions


async def check_system_health() -> SystemHealth:
    """Check system health"""
    return await get_health_monitor().check_all_components()


async def start_health_monitoring():
    """Start health monitoring"""
    await get_health_monitor().start_monitoring()


async def stop_health_monitoring():
    """Stop health monitoring"""
    await get_health_monitor().stop_monitoring()


def get_current_system_health() -> SystemHealth | None:
    """Get current system health"""
    return get_health_monitor().get_current_health()
