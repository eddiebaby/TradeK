"""
Bulkhead Pattern Implementation for TradeKnowledge.

This module provides bulkhead isolation patterns to prevent failures
in one component from affecting others by isolating resources.
"""

import asyncio
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources that can be isolated"""

    THREAD_POOL = "thread_pool"
    ASYNC_SEMAPHORE = "async_semaphore"
    CONNECTION_POOL = "connection_pool"
    MEMORY_POOL = "memory_pool"
    CPU_POOL = "cpu_pool"


@dataclass
class BulkheadConfig:
    """Configuration for bulkhead isolation"""

    name: str
    resource_type: ResourceType
    max_concurrent: int = 10  # Maximum concurrent operations
    max_queue_size: int = 100  # Maximum queued operations
    timeout_seconds: float = 30.0  # Operation timeout
    isolation_level: str = "strict"  # "strict" or "elastic"
    priority_levels: int = 3  # Number of priority levels (1-3)
    enable_metrics: bool = True  # Enable detailed metrics collection


@dataclass
class BulkheadMetrics:
    """Metrics for bulkhead operations"""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0  # Rejected due to capacity
    timed_out_requests: int = 0
    current_active: int = 0
    current_queued: int = 0
    max_concurrent_reached: int = 0
    max_queue_reached: int = 0
    average_wait_time: float = 0.0
    last_request_time: datetime | None = None
    resource_utilization: float = 0.0


class BulkheadCapacityExceededException(Exception):
    """Raised when bulkhead capacity is exceeded"""

    pass


class BulkheadTimeoutException(Exception):
    """Raised when operation times out in bulkhead"""

    pass


class PriorityQueue:
    """Priority queue for bulkhead operations"""

    def __init__(self, max_size: int, priority_levels: int = 3):
        self.max_size = max_size
        self.priority_levels = priority_levels
        self.queues = [deque() for _ in range(priority_levels)]
        self.total_size = 0
        self._lock = threading.Lock()

    def put(self, item: Any, priority: int = 1) -> bool:
        """
        Put item in queue with given priority.

        Args:
            item: Item to queue
            priority: Priority level (0 = highest, priority_levels-1 = lowest)

        Returns:
            True if item was queued, False if queue is full
        """
        with self._lock:
            if self.total_size >= self.max_size:
                return False

            priority = max(0, min(priority, self.priority_levels - 1))
            self.queues[priority].append(item)
            self.total_size += 1
            return True

    def get(self) -> Any | None:
        """Get highest priority item from queue"""
        with self._lock:
            # Check queues from highest to lowest priority
            for queue in self.queues:
                if queue:
                    item = queue.popleft()
                    self.total_size -= 1
                    return item
            return None

    def size(self) -> int:
        """Get total queue size"""
        return self.total_size

    def is_full(self) -> bool:
        """Check if queue is full"""
        return self.total_size >= self.max_size

    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return self.total_size == 0


class Bulkhead:
    """
    Bulkhead isolation implementation that prevents failures in one
    component from cascading to others by limiting resource usage.
    """

    def __init__(self, config: BulkheadConfig):
        self.config = config
        self.metrics = BulkheadMetrics()

        # Initialize resource isolation mechanisms
        if config.resource_type == ResourceType.ASYNC_SEMAPHORE:
            self._semaphore = asyncio.Semaphore(config.max_concurrent)
        elif config.resource_type == ResourceType.THREAD_POOL:
            from concurrent.futures import ThreadPoolExecutor

            self._thread_pool = ThreadPoolExecutor(
                max_workers=config.max_concurrent,
                thread_name_prefix=f"bulkhead_{config.name}",
            )

        # Priority queue for managing requests
        self._queue = PriorityQueue(config.max_queue_size, config.priority_levels)
        self._active_operations = set()
        self._lock = asyncio.Lock()

        # Metrics tracking
        self._start_times = {}
        self._wait_times = deque(maxlen=1000)  # Keep last 1000 wait times

        logger.info(
            f"Initialized bulkhead '{config.name}' with {config.max_concurrent} concurrent slots"
        )

    async def execute(self, func: Callable, *args, priority: int = 1, **kwargs) -> Any:
        """
        Execute a function within the bulkhead isolation.

        Args:
            func: Function to execute (sync or async)
            *args: Positional arguments
            priority: Priority level (0 = highest)
            **kwargs: Keyword arguments

        Returns:
            Result of function execution

        Raises:
            BulkheadCapacityExceededException: When capacity is exceeded
            BulkheadTimeoutException: When operation times out
        """
        request_id = id((func, args, kwargs, time.time()))
        start_time = time.time()

        self.metrics.total_requests += 1
        self.metrics.last_request_time = datetime.now()

        try:
            # Check if we can execute immediately or need to queue
            async with self._lock:
                if len(self._active_operations) < self.config.max_concurrent:
                    # Can execute immediately
                    self._active_operations.add(request_id)
                    self.metrics.current_active = len(self._active_operations)
                    self.metrics.max_concurrent_reached = max(
                        self.metrics.max_concurrent_reached, self.metrics.current_active
                    )

                    # Execute directly
                    result = await self._execute_function(
                        func, args, kwargs, request_id
                    )
                    return result
                else:
                    # Need to queue the request
                    queued = self._queue.put((request_id, func, args, kwargs), priority)
                    if not queued:
                        self.metrics.rejected_requests += 1
                        raise BulkheadCapacityExceededException(
                            f"Bulkhead '{self.config.name}' queue is full"
                        )

                    self.metrics.current_queued = self._queue.size()
                    self.metrics.max_queue_reached = max(
                        self.metrics.max_queue_reached, self.metrics.current_queued
                    )

            # Wait for our turn from the queue
            result = await self._wait_and_execute_from_queue(request_id, start_time)
            return result

        except Exception as e:
            self.metrics.failed_requests += 1
            raise e

        finally:
            # Clean up active operations tracking
            async with self._lock:
                if request_id in self._active_operations:
                    self._active_operations.remove(request_id)
                    self.metrics.current_active = len(self._active_operations)

                if request_id in self._start_times:
                    del self._start_times[request_id]

                # Update resource utilization
                self.metrics.resource_utilization = (
                    self.metrics.current_active / self.config.max_concurrent
                )

    async def _wait_and_execute_from_queue(
        self, request_id: str, start_time: float
    ) -> Any:
        """Wait for slot availability and execute from queue"""
        queued_time = time.time()

        while True:
            # Check for timeout
            if time.time() - start_time > self.config.timeout_seconds:
                raise BulkheadTimeoutException(
                    f"Operation timed out after {self.config.timeout_seconds} seconds"
                )

            # Try to get a slot
            async with self._lock:
                if len(self._active_operations) < self.config.max_concurrent:
                    # Find our item in the queue
                    queue_item = None
                    for i in range(len(self._queue.queues)):
                        for j, item in enumerate(self._queue.queues[i]):
                            if item[0] == request_id:
                                queue_item = self._queue.queues[i][j]
                                del self._queue.queues[i][j]
                                self._queue.total_size -= 1
                                break
                        if queue_item:
                            break

                    if queue_item:
                        _, func, args, kwargs = queue_item
                        self._active_operations.add(request_id)
                        self.metrics.current_active = len(self._active_operations)
                        self.metrics.current_queued = self._queue.size()

                        # Record wait time
                        wait_time = time.time() - queued_time
                        self._wait_times.append(wait_time)
                        if self._wait_times:
                            self.metrics.average_wait_time = sum(
                                self._wait_times
                            ) / len(self._wait_times)

                        # Execute the function
                        result = await self._execute_function(
                            func, args, kwargs, request_id
                        )
                        return result

            # Wait a bit before checking again
            await asyncio.sleep(0.01)

    async def _execute_function(
        self, func: Callable, args: tuple, kwargs: dict, request_id: str
    ) -> Any:
        """Execute the function with proper resource management"""
        self._start_times[request_id] = time.time()

        try:
            if self.config.resource_type == ResourceType.ASYNC_SEMAPHORE:
                async with self._semaphore:
                    result = await self._execute_with_timeout(func, args, kwargs)
            elif self.config.resource_type == ResourceType.THREAD_POOL:
                if asyncio.iscoroutinefunction(func):
                    result = await self._execute_with_timeout(func, args, kwargs)
                else:
                    # Run sync function in thread pool
                    import functools

                    partial_func = functools.partial(func, *args, **kwargs)
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(self._thread_pool, partial_func),
                        timeout=self.config.timeout_seconds,
                    )
            else:
                # Default async execution
                result = await self._execute_with_timeout(func, args, kwargs)

            self.metrics.successful_requests += 1
            return result

        except TimeoutError:
            self.metrics.timed_out_requests += 1
            raise BulkheadTimeoutException(
                f"Operation timed out after {self.config.timeout_seconds} seconds"
            )

    async def _execute_with_timeout(
        self, func: Callable, args: tuple, kwargs: dict
    ) -> Any:
        """Execute function with timeout"""
        if asyncio.iscoroutinefunction(func):
            return await asyncio.wait_for(
                func(*args, **kwargs), timeout=self.config.timeout_seconds
            )
        else:
            # For sync functions, run in executor
            import functools

            partial_func = functools.partial(func, *args, **kwargs)
            loop = asyncio.get_event_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(None, partial_func),
                timeout=self.config.timeout_seconds,
            )

    def get_metrics(self) -> dict[str, Any]:
        """Get current bulkhead metrics"""
        success_rate = (
            self.metrics.successful_requests / self.metrics.total_requests
            if self.metrics.total_requests > 0
            else 0
        )

        rejection_rate = (
            self.metrics.rejected_requests / self.metrics.total_requests
            if self.metrics.total_requests > 0
            else 0
        )

        return {
            "name": self.config.name,
            "resource_type": self.config.resource_type.value,
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "rejected_requests": self.metrics.rejected_requests,
            "timed_out_requests": self.metrics.timed_out_requests,
            "success_rate": success_rate,
            "rejection_rate": rejection_rate,
            "current_active": self.metrics.current_active,
            "current_queued": self.metrics.current_queued,
            "max_concurrent_reached": self.metrics.max_concurrent_reached,
            "max_queue_reached": self.metrics.max_queue_reached,
            "average_wait_time": self.metrics.average_wait_time,
            "resource_utilization": self.metrics.resource_utilization,
            "capacity": self.config.max_concurrent,
            "queue_capacity": self.config.max_queue_size,
            "last_request_time": (
                self.metrics.last_request_time.isoformat()
                if self.metrics.last_request_time
                else None
            ),
        }

    def reset_metrics(self):
        """Reset bulkhead metrics"""
        self.metrics = BulkheadMetrics()
        self._wait_times.clear()
        logger.info(f"Reset metrics for bulkhead '{self.config.name}'")

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on bulkhead"""
        metrics = self.get_metrics()

        # Determine health status
        utilization = metrics["resource_utilization"]
        rejection_rate = metrics["rejection_rate"]

        if rejection_rate > 0.1:  # More than 10% rejections
            status = "critical"
        elif utilization > 0.9:  # More than 90% utilization
            status = "warning"
        elif utilization > 0.7:  # More than 70% utilization
            status = "degraded"
        else:
            status = "healthy"

        return {
            "status": status,
            "utilization": utilization,
            "rejection_rate": rejection_rate,
            "active_operations": metrics["current_active"],
            "queued_operations": metrics["current_queued"],
            "capacity": metrics["capacity"],
        }


class BulkheadManager:
    """
    Manager for multiple bulkheads across the application.

    Provides centralized management and monitoring of all
    bulkhead isolations in the system.
    """

    def __init__(self):
        self.bulkheads: dict[str, Bulkhead] = {}
        self._default_configs: dict[str, BulkheadConfig] = {}

    def register_bulkhead(self, config: BulkheadConfig) -> Bulkhead:
        """Register a new bulkhead"""
        if config.name in self.bulkheads:
            logger.warning(f"Bulkhead '{config.name}' already exists, replacing it")

        bulkhead = Bulkhead(config)
        self.bulkheads[config.name] = bulkhead
        self._default_configs[config.name] = config

        logger.info(f"Registered bulkhead '{config.name}'")
        return bulkhead

    def get_bulkhead(self, name: str) -> Bulkhead | None:
        """Get bulkhead by name"""
        return self.bulkheads.get(name)

    async def execute_in_bulkhead(
        self, bulkhead_name: str, func: Callable, *args, priority: int = 1, **kwargs
    ) -> Any:
        """
        Execute function in specified bulkhead.

        Args:
            bulkhead_name: Name of bulkhead to use
            func: Function to execute
            *args: Positional arguments
            priority: Priority level
            **kwargs: Keyword arguments

        Returns:
            Result of function execution

        Raises:
            ValueError: If bulkhead not found
        """
        bulkhead = self.get_bulkhead(bulkhead_name)
        if not bulkhead:
            raise ValueError(f"Bulkhead '{bulkhead_name}' not found")

        return await bulkhead.execute(func, *args, priority=priority, **kwargs)

    def get_all_metrics(self) -> dict[str, dict[str, Any]]:
        """Get metrics for all bulkheads"""
        return {
            name: bulkhead.get_metrics() for name, bulkhead in self.bulkheads.items()
        }

    async def get_system_health(self) -> dict[str, Any]:
        """Get overall system health from bulkhead perspective"""
        health_checks = {}

        for name, bulkhead in self.bulkheads.items():
            health_checks[name] = await bulkhead.health_check()

        # Determine overall health
        statuses = [health["status"] for health in health_checks.values()]

        if "critical" in statuses:
            overall_status = "critical"
        elif "warning" in statuses:
            overall_status = "warning"
        elif "degraded" in statuses:
            overall_status = "degraded"
        else:
            overall_status = "healthy"

        total_bulkheads = len(self.bulkheads)
        healthy_bulkheads = len([s for s in statuses if s == "healthy"])

        return {
            "overall_status": overall_status,
            "total_bulkheads": total_bulkheads,
            "healthy_bulkheads": healthy_bulkheads,
            "bulkhead_health": health_checks,
            "system_utilization": sum(
                health["utilization"] for health in health_checks.values()
            )
            / max(1, len(health_checks)),
        }

    def create_default_configs(self) -> dict[str, BulkheadConfig]:
        """Create default bulkhead configurations for TradeKnowledge"""
        return {
            "search_operations": BulkheadConfig(
                name="search_operations",
                resource_type=ResourceType.ASYNC_SEMAPHORE,
                max_concurrent=20,
                max_queue_size=100,
                timeout_seconds=30.0,
                priority_levels=3,
            ),
            "embedding_generation": BulkheadConfig(
                name="embedding_generation",
                resource_type=ResourceType.THREAD_POOL,
                max_concurrent=5,
                max_queue_size=50,
                timeout_seconds=60.0,
                priority_levels=2,
            ),
            "database_operations": BulkheadConfig(
                name="database_operations",
                resource_type=ResourceType.ASYNC_SEMAPHORE,
                max_concurrent=10,
                max_queue_size=200,
                timeout_seconds=15.0,
                priority_levels=3,
            ),
            "file_operations": BulkheadConfig(
                name="file_operations",
                resource_type=ResourceType.THREAD_POOL,
                max_concurrent=8,
                max_queue_size=50,
                timeout_seconds=20.0,
                priority_levels=2,
            ),
            "api_requests": BulkheadConfig(
                name="api_requests",
                resource_type=ResourceType.ASYNC_SEMAPHORE,
                max_concurrent=15,
                max_queue_size=75,
                timeout_seconds=25.0,
                priority_levels=3,
            ),
        }

    def setup_default_bulkheads(self):
        """Set up default bulkheads for TradeKnowledge"""
        default_configs = self.create_default_configs()

        for name, config in default_configs.items():
            self.register_bulkhead(config)

        logger.info(f"Set up {len(default_configs)} default bulkheads")


# Global bulkhead manager instance
_bulkhead_manager = BulkheadManager()


def get_bulkhead_manager() -> BulkheadManager:
    """Get the global bulkhead manager instance"""
    return _bulkhead_manager


def setup_bulkheads():
    """Initialize bulkheads for the application"""
    manager = get_bulkhead_manager()
    manager.setup_default_bulkheads()
    logger.info("Bulkheads initialized for TradeKnowledge")


# Convenience decorator for bulkhead isolation
def bulkhead_isolation(bulkhead_name: str, priority: int = 1):
    """
    Decorator to execute functions within bulkhead isolation.

    Args:
        bulkhead_name: Name of bulkhead to use
        priority: Priority level for the operation

    Returns:
        Decorated function with bulkhead protection
    """

    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            manager = get_bulkhead_manager()
            return await manager.execute_in_bulkhead(
                bulkhead_name, func, *args, priority=priority, **kwargs
            )

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.bulkhead_name = bulkhead_name
        wrapper.priority = priority

        return wrapper

    return decorator
