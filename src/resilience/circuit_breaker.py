"""
Circuit Breaker Pattern Implementation for TradeKnowledge.

This module provides circuit breaker functionality to handle failures
in external dependencies gracefully and prevent cascading failures.
"""

import asyncio
import functools
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""

    failure_threshold: int = 5  # Number of failures before opening
    recovery_timeout: float = 60.0  # Seconds before trying half-open
    success_threshold: int = 3  # Successful calls to close from half-open
    timeout: float = 30.0  # Operation timeout in seconds
    expected_exception: tuple = (Exception,)  # Exceptions that count as failures
    name: str = "default"  # Circuit breaker name for logging


@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker metrics and statistics"""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    circuit_opened_count: int = 0
    last_failure_time: datetime | None = None
    last_success_time: datetime | None = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    current_state: CircuitState = CircuitState.CLOSED
    state_changed_time: datetime = field(default_factory=datetime.now)


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""

    pass


class CircuitBreakerTimeoutException(Exception):
    """Exception raised when operation times out"""

    pass


class CircuitBreaker:
    """
    Circuit breaker implementation that protects against cascading failures.

    The circuit breaker monitors calls to external services and automatically
    opens when failure thresholds are exceeded, preventing further calls
    until the service potentially recovers.
    """

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.metrics = CircuitBreakerMetrics()
        self._lock = asyncio.Lock()
        self._next_attempt_time = 0

        logger.info(
            f"Initialized circuit breaker '{config.name}' with config: {config}"
        )

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function through the circuit breaker.

        Args:
            func: Function to execute (can be sync or async)
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the function call

        Raises:
            CircuitBreakerOpenException: When circuit is open
            CircuitBreakerTimeoutException: When operation times out
            Other exceptions: As raised by the wrapped function
        """
        async with self._lock:
            self._check_state()

            if self.metrics.current_state == CircuitState.OPEN:
                self.metrics.total_requests += 1
                raise CircuitBreakerOpenException(
                    f"Circuit breaker '{self.config.name}' is OPEN. "
                    f"Next attempt allowed at {datetime.fromtimestamp(self._next_attempt_time)}"
                )

        # Execute the function with timeout
        try:
            self.metrics.total_requests += 1

            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs), timeout=self.config.timeout
                )
            else:
                # Run sync function in executor with timeout
                # Need to use partial for kwargs support
                import functools

                partial_func = functools.partial(func, *args, **kwargs)
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, partial_func),
                    timeout=self.config.timeout,
                )

            await self._on_success()
            return result

        except TimeoutError:
            await self._on_failure()
            raise CircuitBreakerTimeoutException(
                f"Operation timed out after {self.config.timeout} seconds"
            )
        except self.config.expected_exception as e:
            await self._on_failure()
            raise e
        except Exception as e:
            # Unexpected exceptions don't count as failures
            logger.warning(
                f"Unexpected exception in circuit breaker '{self.config.name}': {e}"
            )
            raise e

    def _check_state(self):
        """Check and update circuit breaker state"""
        current_time = time.time()

        if self.metrics.current_state == CircuitState.OPEN:
            if current_time >= self._next_attempt_time:
                self._transition_to_half_open()
        elif self.metrics.current_state == CircuitState.HALF_OPEN:
            # Stay in half-open, will transition based on call results
            pass
        # CLOSED state doesn't need time-based transitions

    async def _on_success(self):
        """Handle successful operation"""
        async with self._lock:
            self.metrics.successful_requests += 1
            self.metrics.consecutive_successes += 1
            self.metrics.consecutive_failures = 0
            self.metrics.last_success_time = datetime.now()

            if self.metrics.current_state == CircuitState.HALF_OPEN:
                if self.metrics.consecutive_successes >= self.config.success_threshold:
                    self._transition_to_closed()

            logger.debug(
                f"Circuit breaker '{self.config.name}' recorded success. "
                f"Consecutive successes: {self.metrics.consecutive_successes}"
            )

    async def _on_failure(self):
        """Handle failed operation"""
        async with self._lock:
            self.metrics.failed_requests += 1
            self.metrics.consecutive_failures += 1
            self.metrics.consecutive_successes = 0
            self.metrics.last_failure_time = datetime.now()

            if self.metrics.current_state == CircuitState.CLOSED:
                if self.metrics.consecutive_failures >= self.config.failure_threshold:
                    self._transition_to_open()
            elif self.metrics.current_state == CircuitState.HALF_OPEN:
                # Any failure in half-open state goes back to open
                self._transition_to_open()

            logger.warning(
                f"Circuit breaker '{self.config.name}' recorded failure. "
                f"Consecutive failures: {self.metrics.consecutive_failures}"
            )

    def _transition_to_open(self):
        """Transition circuit breaker to OPEN state"""
        self.metrics.current_state = CircuitState.OPEN
        self.metrics.state_changed_time = datetime.now()
        self.metrics.circuit_opened_count += 1
        self._next_attempt_time = time.time() + self.config.recovery_timeout

        logger.error(
            f"Circuit breaker '{self.config.name}' opened after "
            f"{self.metrics.consecutive_failures} consecutive failures. "
            f"Will retry at {datetime.fromtimestamp(self._next_attempt_time)}"
        )

    def _transition_to_half_open(self):
        """Transition circuit breaker to HALF_OPEN state"""
        self.metrics.current_state = CircuitState.HALF_OPEN
        self.metrics.state_changed_time = datetime.now()
        self.metrics.consecutive_successes = 0
        self.metrics.consecutive_failures = 0

        logger.info(
            f"Circuit breaker '{self.config.name}' transitioned to HALF_OPEN. "
            f"Testing service recovery..."
        )

    def _transition_to_closed(self):
        """Transition circuit breaker to CLOSED state"""
        self.metrics.current_state = CircuitState.CLOSED
        self.metrics.state_changed_time = datetime.now()
        self.metrics.consecutive_failures = 0

        logger.info(
            f"Circuit breaker '{self.config.name}' closed after "
            f"{self.metrics.consecutive_successes} consecutive successes. "
            f"Normal operation resumed."
        )

    def get_metrics(self) -> dict[str, Any]:
        """Get current circuit breaker metrics"""
        return {
            "name": self.config.name,
            "state": self.metrics.current_state.value,
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "success_rate": (
                self.metrics.successful_requests / self.metrics.total_requests
                if self.metrics.total_requests > 0
                else 0
            ),
            "consecutive_failures": self.metrics.consecutive_failures,
            "consecutive_successes": self.metrics.consecutive_successes,
            "circuit_opened_count": self.metrics.circuit_opened_count,
            "last_failure_time": (
                self.metrics.last_failure_time.isoformat()
                if self.metrics.last_failure_time
                else None
            ),
            "last_success_time": (
                self.metrics.last_success_time.isoformat()
                if self.metrics.last_success_time
                else None
            ),
            "state_changed_time": self.metrics.state_changed_time.isoformat(),
            "next_attempt_time": (
                datetime.fromtimestamp(self._next_attempt_time).isoformat()
                if self.metrics.current_state == CircuitState.OPEN
                else None
            ),
        }

    def reset(self):
        """Reset circuit breaker to initial state"""
        self.metrics = CircuitBreakerMetrics()
        self._next_attempt_time = 0
        logger.info(f"Circuit breaker '{self.config.name}' reset to initial state")

    def force_open(self):
        """Force circuit breaker to OPEN state (for testing/emergency)"""
        self.metrics.current_state = CircuitState.OPEN
        self.metrics.state_changed_time = datetime.now()
        self._next_attempt_time = time.time() + self.config.recovery_timeout
        logger.warning(f"Circuit breaker '{self.config.name}' forced to OPEN state")

    def force_close(self):
        """Force circuit breaker to CLOSED state (for testing/recovery)"""
        self.metrics.current_state = CircuitState.CLOSED
        self.metrics.state_changed_time = datetime.now()
        self.metrics.consecutive_failures = 0
        self._next_attempt_time = 0
        logger.info(f"Circuit breaker '{self.config.name}' forced to CLOSED state")


def circuit_breaker(config: CircuitBreakerConfig):
    """
    Decorator to wrap functions with circuit breaker protection.

    Args:
        config: Circuit breaker configuration

    Returns:
        Decorated function with circuit breaker protection
    """

    def decorator(func: Callable) -> Callable:
        cb = CircuitBreaker(config)

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await cb.call(func, *args, **kwargs)

            # Add circuit breaker methods to wrapper
            async_wrapper.circuit_breaker = cb
            async_wrapper.get_metrics = cb.get_metrics
            async_wrapper.reset = cb.reset
            async_wrapper.force_open = cb.force_open
            async_wrapper.force_close = cb.force_close

            return async_wrapper
        else:

            @functools.wraps(func)
            async def sync_wrapper(*args, **kwargs):
                return await cb.call(func, *args, **kwargs)

            # Add circuit breaker methods to wrapper
            sync_wrapper.circuit_breaker = cb
            sync_wrapper.get_metrics = cb.get_metrics
            sync_wrapper.reset = cb.reset
            sync_wrapper.force_open = cb.force_open
            sync_wrapper.force_close = cb.force_close

            return sync_wrapper

    return decorator


class CircuitBreakerManager:
    """
    Manager for multiple circuit breakers across the application.

    Provides centralized management, monitoring, and configuration
    of all circuit breakers in the system.
    """

    def __init__(self):
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self._default_configs: dict[str, CircuitBreakerConfig] = {}

    def register_circuit_breaker(
        self, name: str, config: CircuitBreakerConfig
    ) -> CircuitBreaker:
        """Register a new circuit breaker"""
        if name in self.circuit_breakers:
            logger.warning(f"Circuit breaker '{name}' already exists, replacing it")

        config.name = name
        circuit_breaker = CircuitBreaker(config)
        self.circuit_breakers[name] = circuit_breaker
        self._default_configs[name] = config

        logger.info(f"Registered circuit breaker '{name}'")
        return circuit_breaker

    def get_circuit_breaker(self, name: str) -> CircuitBreaker | None:
        """Get circuit breaker by name"""
        return self.circuit_breakers.get(name)

    def get_all_metrics(self) -> dict[str, dict[str, Any]]:
        """Get metrics for all circuit breakers"""
        return {name: cb.get_metrics() for name, cb in self.circuit_breakers.items()}

    def get_health_summary(self) -> dict[str, Any]:
        """Get overall health summary"""
        total_breakers = len(self.circuit_breakers)
        open_breakers = sum(
            1
            for cb in self.circuit_breakers.values()
            if cb.metrics.current_state == CircuitState.OPEN
        )
        half_open_breakers = sum(
            1
            for cb in self.circuit_breakers.values()
            if cb.metrics.current_state == CircuitState.HALF_OPEN
        )

        return {
            "total_circuit_breakers": total_breakers,
            "open_circuit_breakers": open_breakers,
            "half_open_circuit_breakers": half_open_breakers,
            "healthy_circuit_breakers": total_breakers
            - open_breakers
            - half_open_breakers,
            "overall_health": (
                "healthy"
                if open_breakers == 0
                else "degraded" if open_breakers < total_breakers else "critical"
            ),
        }

    def reset_all(self):
        """Reset all circuit breakers"""
        for cb in self.circuit_breakers.values():
            cb.reset()
        logger.info("Reset all circuit breakers")

    def create_default_configs(self) -> dict[str, CircuitBreakerConfig]:
        """Create default circuit breaker configurations for TradeKnowledge services"""
        return {
            "qdrant_vector_db": CircuitBreakerConfig(
                name="qdrant_vector_db",
                failure_threshold=3,
                recovery_timeout=30.0,
                success_threshold=2,
                timeout=10.0,
                expected_exception=(ConnectionError, TimeoutError, Exception),
            ),
            "ollama_embedding_service": CircuitBreakerConfig(
                name="ollama_embedding_service",
                failure_threshold=5,
                recovery_timeout=60.0,
                success_threshold=3,
                timeout=30.0,
                expected_exception=(ConnectionError, TimeoutError, Exception),
            ),
            "sqlite_database": CircuitBreakerConfig(
                name="sqlite_database",
                failure_threshold=3,
                recovery_timeout=15.0,
                success_threshold=2,
                timeout=5.0,
                expected_exception=(Exception,),
            ),
            "file_system": CircuitBreakerConfig(
                name="file_system",
                failure_threshold=5,
                recovery_timeout=10.0,
                success_threshold=2,
                timeout=10.0,
                expected_exception=(IOError, OSError, PermissionError),
            ),
            "external_api": CircuitBreakerConfig(
                name="external_api",
                failure_threshold=3,
                recovery_timeout=120.0,
                success_threshold=3,
                timeout=20.0,
                expected_exception=(ConnectionError, TimeoutError, Exception),
            ),
        }

    def setup_default_circuit_breakers(self):
        """Set up default circuit breakers for common TradeKnowledge dependencies"""
        default_configs = self.create_default_configs()

        for name, config in default_configs.items():
            self.register_circuit_breaker(name, config)

        logger.info(f"Set up {len(default_configs)} default circuit breakers")


# Global circuit breaker manager instance
_circuit_breaker_manager = CircuitBreakerManager()


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get the global circuit breaker manager instance"""
    return _circuit_breaker_manager


def setup_circuit_breakers():
    """Initialize circuit breakers for the application"""
    manager = get_circuit_breaker_manager()
    manager.setup_default_circuit_breakers()
    logger.info("Circuit breakers initialized for TradeKnowledge")


# Convenience functions for common circuit breaker operations
async def call_with_circuit_breaker(
    circuit_breaker_name: str, func: Callable, *args, **kwargs
) -> Any:
    """
    Call a function through a named circuit breaker.

    Args:
        circuit_breaker_name: Name of the circuit breaker to use
        func: Function to call
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Result of the function call

    Raises:
        ValueError: If circuit breaker not found
        CircuitBreakerOpenException: If circuit breaker is open
        Other exceptions: As raised by the function
    """
    manager = get_circuit_breaker_manager()
    cb = manager.get_circuit_breaker(circuit_breaker_name)

    if not cb:
        raise ValueError(f"Circuit breaker '{circuit_breaker_name}' not found")

    return await cb.call(func, *args, **kwargs)
