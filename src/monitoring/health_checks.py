"""
Health Check Endpoints for TradeKnowledge.

This module provides comprehensive health monitoring capabilities
including system health checks, dependency checks, and status reporting.
"""

import asyncio
import json
import logging
import os
import sqlite3
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
import psutil

try:
    import qdrant_client

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    qdrant_client = None

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CheckPriority(Enum):
    """Priority levels for health checks"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class HealthCheckResult:
    """Result of a health check"""

    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    duration_ms: float
    priority: CheckPriority
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    remediation_steps: list[str] = field(default_factory=list)


@dataclass
class DependencyStatus:
    """Status of an external dependency"""

    name: str
    url: str | None
    status: HealthStatus
    response_time_ms: float
    version: str | None = None
    last_checked: datetime = field(default_factory=datetime.now)
    error_count: int = 0
    consecutive_failures: int = 0


@dataclass
class SystemMetrics:
    """System resource metrics"""

    cpu_percent: float
    memory_percent: float
    disk_percent: float
    disk_usage_gb: float
    available_memory_gb: float
    load_average: float | None
    uptime_seconds: float
    process_count: int
    open_files: int
    network_connections: int


class HealthCheckRegistry:
    """Registry for health check functions"""

    def __init__(self):
        self.checks: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        check_function: Callable,
        priority: CheckPriority = CheckPriority.MEDIUM,
        timeout_seconds: float = 30.0,
        interval_seconds: float = 60.0,
        enabled: bool = True,
    ):
        """Register a health check function"""
        with self._lock:
            self.checks[name] = {
                "function": check_function,
                "priority": priority,
                "timeout": timeout_seconds,
                "interval": interval_seconds,
                "enabled": enabled,
                "last_run": None,
                "last_result": None,
            }
        logger.info(f"Registered health check: {name}")

    def unregister(self, name: str):
        """Unregister a health check"""
        with self._lock:
            if name in self.checks:
                del self.checks[name]
                logger.info(f"Unregistered health check: {name}")

    def get_checks(
        self, priority: CheckPriority | None = None
    ) -> dict[str, dict[str, Any]]:
        """Get registered checks, optionally filtered by priority"""
        with self._lock:
            if priority:
                return {
                    name: check
                    for name, check in self.checks.items()
                    if check["priority"] == priority and check["enabled"]
                }
            return {
                name: check for name, check in self.checks.items() if check["enabled"]
            }

    def enable_check(self, name: str):
        """Enable a health check"""
        with self._lock:
            if name in self.checks:
                self.checks[name]["enabled"] = True

    def disable_check(self, name: str):
        """Disable a health check"""
        with self._lock:
            if name in self.checks:
                self.checks[name]["enabled"] = False


class HealthCheckManager:
    """
    Comprehensive health check manager that monitors system health,
    dependencies, and application components.
    """

    def __init__(
        self, db_path: str = "data/health_checks.db", check_interval: float = 60.0
    ):
        self.db_path = db_path
        self.check_interval = check_interval
        self.registry = HealthCheckRegistry()
        self.dependencies: dict[str, DependencyStatus] = {}
        self.recent_results: list[HealthCheckResult] = []
        self.max_recent_results = 1000
        self._executor = ThreadPoolExecutor(max_workers=8)
        self._running = False
        self._background_task: asyncio.Task | None = None

        # Initialize database
        self._init_database()

        # Register default health checks
        self._register_default_checks()

    def _init_database(self):
        """Initialize health check database"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS health_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT,
                    timestamp TEXT NOT NULL,
                    duration_ms REAL,
                    priority TEXT,
                    details TEXT,
                    error TEXT
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dependency_status (
                    name TEXT PRIMARY KEY,
                    url TEXT,
                    status TEXT NOT NULL,
                    response_time_ms REAL,
                    version TEXT,
                    last_checked TEXT,
                    error_count INTEGER DEFAULT 0,
                    consecutive_failures INTEGER DEFAULT 0
                )
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_health_timestamp 
                ON health_results(timestamp)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_health_status 
                ON health_results(status)
            """
            )

    def _register_default_checks(self):
        """Register default health checks"""
        # System resource checks
        self.registry.register(
            "system_resources",
            self._check_system_resources,
            CheckPriority.CRITICAL,
            timeout_seconds=10.0,
        )

        # Database connectivity
        self.registry.register(
            "database_connection",
            self._check_database_connection,
            CheckPriority.CRITICAL,
            timeout_seconds=15.0,
        )

        # Vector database (Qdrant)
        if QDRANT_AVAILABLE:
            self.registry.register(
                "vector_database",
                self._check_vector_database,
                CheckPriority.HIGH,
                timeout_seconds=10.0,
            )

        # API endpoints
        self.registry.register(
            "api_endpoints",
            self._check_api_endpoints,
            CheckPriority.HIGH,
            timeout_seconds=20.0,
        )

        # File system
        self.registry.register(
            "file_system",
            self._check_file_system,
            CheckPriority.MEDIUM,
            timeout_seconds=5.0,
        )

        # Application state
        self.registry.register(
            "application_state",
            self._check_application_state,
            CheckPriority.MEDIUM,
            timeout_seconds=5.0,
        )

    async def start_monitoring(self):
        """Start continuous health monitoring"""
        if self._running:
            logger.warning("Health monitoring is already running")
            return

        self._running = True
        self._background_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started health monitoring")

    async def stop_monitoring(self):
        """Stop continuous health monitoring"""
        self._running = False
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped health monitoring")

    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self._running:
            try:
                await self.run_all_checks()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5.0)  # Short delay before retrying

    async def run_all_checks(self) -> dict[str, HealthCheckResult]:
        """Run all registered health checks"""
        checks = self.registry.get_checks()
        results = {}

        # Run checks concurrently
        tasks = []
        for name, check_config in checks.items():
            task = asyncio.create_task(self._run_single_check(name, check_config))
            tasks.append((name, task))

        # Collect results
        for name, task in tasks:
            try:
                result = await task
                results[name] = result
                self._store_result(result)
            except Exception as e:
                error_result = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check execution failed: {e}",
                    timestamp=datetime.now(),
                    duration_ms=0.0,
                    priority=CheckPriority.CRITICAL,
                    error=str(e),
                )
                results[name] = error_result
                self._store_result(error_result)

        return results

    async def _run_single_check(
        self, name: str, check_config: dict[str, Any]
    ) -> HealthCheckResult:
        """Run a single health check"""
        start_time = time.time()

        try:
            # Run check with timeout
            check_function = check_config["function"]
            timeout = check_config["timeout"]

            if asyncio.iscoroutinefunction(check_function):
                result = await asyncio.wait_for(check_function(), timeout=timeout)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(self._executor, check_function)

            duration_ms = (time.time() - start_time) * 1000

            # Update registry
            check_config["last_run"] = datetime.now()
            check_config["last_result"] = result

            return result

        except TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {check_config['timeout']}s",
                timestamp=datetime.now(),
                duration_ms=duration_ms,
                priority=check_config["priority"],
                error="TimeoutError",
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
                timestamp=datetime.now(),
                duration_ms=duration_ms,
                priority=check_config["priority"],
                error=str(e),
            )

    def _store_result(self, result: HealthCheckResult):
        """Store health check result"""
        # Store in memory
        self.recent_results.append(result)
        if len(self.recent_results) > self.max_recent_results:
            self.recent_results = self.recent_results[-self.max_recent_results :]

        # Store in database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO health_results 
                    (name, status, message, timestamp, duration_ms, priority, details, error)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        result.name,
                        result.status.value,
                        result.message,
                        result.timestamp.isoformat(),
                        result.duration_ms,
                        result.priority.value,
                        json.dumps(result.details),
                        result.error,
                    ),
                )
        except Exception as e:
            logger.error(f"Failed to store health check result: {e}")

    async def _check_system_resources(self) -> HealthCheckResult:
        """Check system resource usage"""
        try:
            metrics = self._get_system_metrics()

            # Determine status based on resource usage
            status = HealthStatus.HEALTHY
            issues = []

            if metrics.cpu_percent > 90:
                status = HealthStatus.UNHEALTHY
                issues.append(f"CPU usage critical: {metrics.cpu_percent:.1f}%")
            elif metrics.cpu_percent > 80:
                status = HealthStatus.DEGRADED
                issues.append(f"CPU usage high: {metrics.cpu_percent:.1f}%")

            if metrics.memory_percent > 95:
                status = HealthStatus.UNHEALTHY
                issues.append(f"Memory usage critical: {metrics.memory_percent:.1f}%")
            elif metrics.memory_percent > 85:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
                issues.append(f"Memory usage high: {metrics.memory_percent:.1f}%")

            if metrics.disk_percent > 95:
                status = HealthStatus.UNHEALTHY
                issues.append(f"Disk usage critical: {metrics.disk_percent:.1f}%")
            elif metrics.disk_percent > 90:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
                issues.append(f"Disk usage high: {metrics.disk_percent:.1f}%")

            message = "System resources healthy" if not issues else "; ".join(issues)

            return HealthCheckResult(
                name="system_resources",
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=0.0,  # Will be set by caller
                priority=CheckPriority.CRITICAL,
                details={
                    "cpu_percent": metrics.cpu_percent,
                    "memory_percent": metrics.memory_percent,
                    "disk_percent": metrics.disk_percent,
                    "available_memory_gb": metrics.available_memory_gb,
                    "disk_usage_gb": metrics.disk_usage_gb,
                    "process_count": metrics.process_count,
                    "uptime_seconds": metrics.uptime_seconds,
                },
            )

        except Exception as e:
            return HealthCheckResult(
                name="system_resources",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check system resources: {e}",
                timestamp=datetime.now(),
                duration_ms=0.0,
                priority=CheckPriority.CRITICAL,
                error=str(e),
            )

    def _get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        available_memory_gb = memory.available / (1024**3)

        # Disk usage
        disk = psutil.disk_usage("/")
        disk_percent = (disk.used / disk.total) * 100
        disk_usage_gb = disk.used / (1024**3)

        # Load average (Unix-like systems)
        try:
            load_average = (
                psutil.getloadavg()[0] if hasattr(psutil, "getloadavg") else None
            )
        except (AttributeError, OSError):
            load_average = None

        # System uptime
        boot_time = psutil.boot_time()
        uptime_seconds = time.time() - boot_time

        # Process info
        process_count = len(psutil.pids())

        # Current process info
        current_process = psutil.Process()
        try:
            open_files = len(current_process.open_files())
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            open_files = 0

        try:
            network_connections = len(current_process.net_connections())
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            network_connections = 0

        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_percent=disk_percent,
            disk_usage_gb=disk_usage_gb,
            available_memory_gb=available_memory_gb,
            load_average=load_average,
            uptime_seconds=uptime_seconds,
            process_count=process_count,
            open_files=open_files,
            network_connections=network_connections,
        )

    async def _check_database_connection(self) -> HealthCheckResult:
        """Check database connectivity"""
        try:
            # Check SQLite database
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                cursor = conn.execute("SELECT 1")
                result = cursor.fetchone()
                if result and result[0] == 1:
                    db_status = "Connected successfully"
                else:
                    raise Exception("Database query returned unexpected result")

            return HealthCheckResult(
                name="database_connection",
                status=HealthStatus.HEALTHY,
                message=db_status,
                timestamp=datetime.now(),
                duration_ms=0.0,
                priority=CheckPriority.CRITICAL,
                details={"database_type": "SQLite", "path": self.db_path},
            )

        except Exception as e:
            return HealthCheckResult(
                name="database_connection",
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {e}",
                timestamp=datetime.now(),
                duration_ms=0.0,
                priority=CheckPriority.CRITICAL,
                error=str(e),
                remediation_steps=[
                    "Check database file permissions",
                    "Verify database file integrity",
                    "Check available disk space",
                    "Restart database service if applicable",
                ],
            )

    async def _check_vector_database(self) -> HealthCheckResult:
        """Check vector database (Qdrant) connectivity"""
        if not QDRANT_AVAILABLE:
            return HealthCheckResult(
                name="vector_database",
                status=HealthStatus.UNKNOWN,
                message="Qdrant client not available",
                timestamp=datetime.now(),
                duration_ms=0.0,
                priority=CheckPriority.HIGH,
                error="qdrant_client not installed",
            )

        try:
            # Try to connect to Qdrant
            client = qdrant_client.QdrantClient(
                host="localhost", port=6333, timeout=5.0
            )

            # Check if client is responsive
            collections = client.get_collections()
            collection_count = len(collections.collections) if collections else 0

            return HealthCheckResult(
                name="vector_database",
                status=HealthStatus.HEALTHY,
                message=f"Vector database connected, {collection_count} collections",
                timestamp=datetime.now(),
                duration_ms=0.0,
                priority=CheckPriority.HIGH,
                details={
                    "host": "localhost",
                    "port": 6333,
                    "collection_count": collection_count,
                },
            )

        except Exception as e:
            return HealthCheckResult(
                name="vector_database",
                status=HealthStatus.UNHEALTHY,
                message=f"Vector database connection failed: {e}",
                timestamp=datetime.now(),
                duration_ms=0.0,
                priority=CheckPriority.HIGH,
                error=str(e),
                remediation_steps=[
                    "Check if Qdrant service is running",
                    "Verify Qdrant configuration",
                    "Check network connectivity to Qdrant",
                    "Review Qdrant logs for errors",
                ],
            )

    async def _check_api_endpoints(self) -> HealthCheckResult:
        """Check API endpoint availability"""
        try:
            # Check if the API server is running locally
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Try health endpoint first
                try:
                    response = await client.get("http://localhost:8000/health")
                    if response.status_code == 200:
                        api_status = "API endpoints healthy"
                        status = HealthStatus.HEALTHY
                    else:
                        api_status = (
                            f"API health endpoint returned {response.status_code}"
                        )
                        status = HealthStatus.DEGRADED
                except httpx.ConnectError:
                    # API might not be running, which is okay for development
                    api_status = "API server not running (development mode)"
                    status = HealthStatus.DEGRADED

            return HealthCheckResult(
                name="api_endpoints",
                status=status,
                message=api_status,
                timestamp=datetime.now(),
                duration_ms=0.0,
                priority=CheckPriority.HIGH,
                details={"base_url": "http://localhost:8000"},
            )

        except Exception as e:
            return HealthCheckResult(
                name="api_endpoints",
                status=HealthStatus.UNKNOWN,
                message=f"API endpoint check failed: {e}",
                timestamp=datetime.now(),
                duration_ms=0.0,
                priority=CheckPriority.HIGH,
                error=str(e),
            )

    async def _check_file_system(self) -> HealthCheckResult:
        """Check file system access and permissions"""
        try:
            issues = []

            # Check data directory
            data_dir = Path("data")
            if not data_dir.exists():
                data_dir.mkdir(parents=True, exist_ok=True)

            # Test write access
            test_file = data_dir / "health_check_test.tmp"
            try:
                test_file.write_text("health check test")
                test_file.unlink()
            except Exception as e:
                issues.append(f"Data directory not writable: {e}")

            # Check important directories
            important_dirs = ["data/qdrant", "data", "logs"]
            for dir_path in important_dirs:
                path = Path(dir_path)
                if path.exists() and not path.is_dir():
                    issues.append(f"{dir_path} exists but is not a directory")
                elif path.exists() and not os.access(path, os.R_OK | os.W_OK):
                    issues.append(f"{dir_path} has insufficient permissions")

            status = HealthStatus.HEALTHY if not issues else HealthStatus.DEGRADED
            message = "File system access healthy" if not issues else "; ".join(issues)

            return HealthCheckResult(
                name="file_system",
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=0.0,
                priority=CheckPriority.MEDIUM,
                details={"checked_directories": important_dirs},
            )

        except Exception as e:
            return HealthCheckResult(
                name="file_system",
                status=HealthStatus.UNHEALTHY,
                message=f"File system check failed: {e}",
                timestamp=datetime.now(),
                duration_ms=0.0,
                priority=CheckPriority.MEDIUM,
                error=str(e),
            )

    async def _check_application_state(self) -> HealthCheckResult:
        """Check application-specific state and configuration"""
        try:
            issues = []

            # Check environment variables
            required_env_vars = ["ANTHROPIC_API_KEY"]  # Add other required vars
            for var in required_env_vars:
                if not os.getenv(var):
                    issues.append(f"Missing environment variable: {var}")

            # Check configuration files
            config_files = [".env", "src", "tests"]
            for config_file in config_files:
                if not Path(config_file).exists():
                    issues.append(f"Missing configuration: {config_file}")

            status = HealthStatus.HEALTHY if not issues else HealthStatus.DEGRADED
            message = "Application state healthy" if not issues else "; ".join(issues)

            return HealthCheckResult(
                name="application_state",
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=0.0,
                priority=CheckPriority.MEDIUM,
                details={"checked_configs": config_files},
            )

        except Exception as e:
            return HealthCheckResult(
                name="application_state",
                status=HealthStatus.UNKNOWN,
                message=f"Application state check failed: {e}",
                timestamp=datetime.now(),
                duration_ms=0.0,
                priority=CheckPriority.MEDIUM,
                error=str(e),
            )

    def get_overall_health(self) -> dict[str, Any]:
        """Get overall system health status"""
        if not self.recent_results:
            return {
                "status": HealthStatus.UNKNOWN.value,
                "message": "No health check results available",
                "timestamp": datetime.now().isoformat(),
                "checks": {},
            }

        # Get latest results for each check
        latest_results = {}
        for result in reversed(self.recent_results):
            if result.name not in latest_results:
                latest_results[result.name] = result

        # Calculate overall status
        critical_unhealthy = sum(
            1
            for r in latest_results.values()
            if r.priority == CheckPriority.CRITICAL
            and r.status == HealthStatus.UNHEALTHY
        )

        high_unhealthy = sum(
            1
            for r in latest_results.values()
            if r.priority == CheckPriority.HIGH and r.status == HealthStatus.UNHEALTHY
        )

        any_degraded = any(
            r.status == HealthStatus.DEGRADED for r in latest_results.values()
        )

        # Determine overall status
        if critical_unhealthy > 0:
            overall_status = HealthStatus.UNHEALTHY
            message = f"{critical_unhealthy} critical system(s) unhealthy"
        elif high_unhealthy > 1:
            overall_status = HealthStatus.UNHEALTHY
            message = f"{high_unhealthy} important system(s) unhealthy"
        elif high_unhealthy == 1:
            overall_status = HealthStatus.DEGRADED
            message = "1 important system unhealthy"
        elif any_degraded:
            overall_status = HealthStatus.DEGRADED
            message = "Some systems degraded"
        else:
            overall_status = HealthStatus.HEALTHY
            message = "All systems healthy"

        return {
            "status": overall_status.value,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "checks": {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "priority": result.priority.value,
                    "timestamp": result.timestamp.isoformat(),
                    "duration_ms": result.duration_ms,
                }
                for name, result in latest_results.items()
            },
            "summary": {
                "total_checks": len(latest_results),
                "healthy": sum(
                    1
                    for r in latest_results.values()
                    if r.status == HealthStatus.HEALTHY
                ),
                "degraded": sum(
                    1
                    for r in latest_results.values()
                    if r.status == HealthStatus.DEGRADED
                ),
                "unhealthy": sum(
                    1
                    for r in latest_results.values()
                    if r.status == HealthStatus.UNHEALTHY
                ),
                "unknown": sum(
                    1
                    for r in latest_results.values()
                    if r.status == HealthStatus.UNKNOWN
                ),
            },
        }

    def get_health_history(
        self,
        check_name: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get health check history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                query = "SELECT * FROM health_results"
                params = []

                conditions = []
                if check_name:
                    conditions.append("name = ?")
                    params.append(check_name)

                if since:
                    conditions.append("timestamp >= ?")
                    params.append(since.isoformat())

                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)

                cursor = conn.execute(query, params)
                rows = cursor.fetchall()

                return [
                    {
                        "name": row["name"],
                        "status": row["status"],
                        "message": row["message"],
                        "timestamp": row["timestamp"],
                        "duration_ms": row["duration_ms"],
                        "priority": row["priority"],
                        "details": json.loads(row["details"]) if row["details"] else {},
                        "error": row["error"],
                    }
                    for row in rows
                ]

        except Exception as e:
            logger.error(f"Failed to get health history: {e}")
            return []


# Global health check manager instance
_global_health_manager: HealthCheckManager | None = None


def get_health_manager() -> HealthCheckManager:
    """Get or create global health check manager"""
    global _global_health_manager
    if _global_health_manager is None:
        _global_health_manager = HealthCheckManager()
    return _global_health_manager


def health_check(
    name: str,
    priority: CheckPriority = CheckPriority.MEDIUM,
    timeout_seconds: float = 30.0,
):
    """Decorator to register a function as a health check"""

    def decorator(func: Callable) -> Callable:
        manager = get_health_manager()
        manager.registry.register(name, func, priority, timeout_seconds)
        return func

    return decorator


# Convenience functions
async def get_system_health() -> dict[str, Any]:
    """Get current system health status"""
    manager = get_health_manager()
    return manager.get_overall_health()


async def run_health_checks() -> dict[str, HealthCheckResult]:
    """Run all health checks once"""
    manager = get_health_manager()
    return await manager.run_all_checks()


def add_dependency_check(name: str, url: str, timeout: float = 5.0):
    """Add a dependency health check"""

    async def dependency_check():
        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url)
                duration_ms = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    return HealthCheckResult(
                        name=name,
                        status=HealthStatus.HEALTHY,
                        message=f"Dependency {name} is healthy",
                        timestamp=datetime.now(),
                        duration_ms=duration_ms,
                        priority=CheckPriority.HIGH,
                        details={"url": url, "status_code": response.status_code},
                    )
                else:
                    return HealthCheckResult(
                        name=name,
                        status=HealthStatus.DEGRADED,
                        message=f"Dependency {name} returned {response.status_code}",
                        timestamp=datetime.now(),
                        duration_ms=duration_ms,
                        priority=CheckPriority.HIGH,
                        details={"url": url, "status_code": response.status_code},
                    )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Dependency {name} check failed: {e}",
                timestamp=datetime.now(),
                duration_ms=duration_ms,
                priority=CheckPriority.HIGH,
                error=str(e),
                details={"url": url},
            )

    manager = get_health_manager()
    manager.registry.register(
        f"dependency_{name}", dependency_check, CheckPriority.HIGH, timeout
    )
