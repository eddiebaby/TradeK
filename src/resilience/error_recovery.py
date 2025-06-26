"""
Error Recovery Mechanisms for TradeKnowledge.

This module provides comprehensive error recovery patterns including
automatic recovery, fallback mechanisms, and graceful failure handling.
"""

import asyncio
import functools
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Types of error recovery strategies"""

    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    BULKHEAD = "bulkhead"
    TIMEOUT = "timeout"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    COMPENSATION = "compensation"


class ErrorSeverity(Enum):
    """Error severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorRecoveryConfig:
    """Configuration for error recovery mechanisms"""

    name: str
    strategies: list[RecoveryStrategy] = field(default_factory=list)
    max_retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    retry_backoff_multiplier: float = 2.0
    timeout_seconds: float = 30.0
    fallback_function: Callable | None = None
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    compensation_function: Callable | None = None
    error_classes: list[type[Exception]] = field(default_factory=lambda: [Exception])
    severity_threshold: ErrorSeverity = ErrorSeverity.MEDIUM
    auto_recovery_enabled: bool = True


@dataclass
class ErrorContext:
    """Context information for error recovery"""

    error: Exception
    function_name: str
    args: tuple
    kwargs: dict
    attempt_number: int
    start_time: datetime
    error_time: datetime
    severity: ErrorSeverity
    recovery_strategies_used: list[RecoveryStrategy] = field(default_factory=list)
    recovery_successful: bool = False
    final_result: Any = None


@dataclass
class RecoveryMetrics:
    """Metrics for error recovery operations"""

    total_errors: int = 0
    recovered_errors: int = 0
    failed_recoveries: int = 0
    retry_attempts: int = 0
    fallback_invocations: int = 0
    circuit_breaker_trips: int = 0
    compensation_executions: int = 0
    recovery_times: list[float] = field(default_factory=list)
    error_distribution: dict[str, int] = field(default_factory=dict)
    strategy_effectiveness: dict[RecoveryStrategy, dict[str, int]] = field(
        default_factory=dict
    )


class ErrorRecoveryException(Exception):
    """Exception raised when error recovery fails"""

    pass


class FallbackNotAvailableException(Exception):
    """Exception raised when fallback is not available"""

    pass


class CompensationFailedException(Exception):
    """Exception raised when compensation action fails"""

    pass


class ErrorRecoveryManager:
    """
    Comprehensive error recovery manager that provides multiple
    recovery strategies and automatic error handling.
    """

    def __init__(self):
        self.configs: dict[str, ErrorRecoveryConfig] = {}
        self.metrics: dict[str, RecoveryMetrics] = {}
        self.circuit_breakers: dict[str, dict[str, Any]] = {}
        self.active_recoveries: dict[str, list[ErrorContext]] = {}
        self._executor = ThreadPoolExecutor(max_workers=4)

    def register_recovery_config(self, config: ErrorRecoveryConfig):
        """Register an error recovery configuration"""
        self.configs[config.name] = config
        self.metrics[config.name] = RecoveryMetrics()

        # Initialize strategy effectiveness tracking
        for strategy in config.strategies:
            if strategy not in self.metrics[config.name].strategy_effectiveness:
                self.metrics[config.name].strategy_effectiveness[strategy] = {
                    "attempts": 0,
                    "successes": 0,
                    "failures": 0,
                }

        logger.info(f"Registered error recovery config: {config.name}")

    async def execute_with_recovery(
        self, config_name: str, function: Callable, *args, **kwargs
    ) -> Any:
        """
        Execute a function with comprehensive error recovery.

        Args:
            config_name: Name of the recovery configuration
            function: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result or fallback result

        Raises:
            ErrorRecoveryException: If all recovery strategies fail
        """
        if config_name not in self.configs:
            raise ValueError(f"Recovery config '{config_name}' not found")

        config = self.configs[config_name]
        metrics = self.metrics[config_name]

        start_time = datetime.now()
        function_name = getattr(function, "__name__", str(function))

        # Initialize error context
        error_context = ErrorContext(
            error=None,
            function_name=function_name,
            args=args,
            kwargs=kwargs,
            attempt_number=0,
            start_time=start_time,
            error_time=start_time,
            severity=ErrorSeverity.LOW,
        )

        try:
            return await self._execute_with_strategies(config, function, error_context)

        except Exception as e:
            # Don't double-count errors here - they're already counted in _execute_with_strategies
            logger.error(f"All recovery strategies failed for {function_name}: {e}")
            raise ErrorRecoveryException(
                f"Recovery failed for {function_name}: {e}"
            ) from e

        finally:
            # Record recovery time
            recovery_time = (datetime.now() - start_time).total_seconds()
            metrics.recovery_times.append(recovery_time)

    async def _execute_with_strategies(
        self,
        config: ErrorRecoveryConfig,
        function: Callable,
        error_context: ErrorContext,
    ) -> Any:
        """Execute function with recovery strategies"""

        # Check circuit breaker first
        if RecoveryStrategy.CIRCUIT_BREAKER in config.strategies:
            if self._is_circuit_breaker_open(config.name):
                raise ErrorRecoveryException(f"Circuit breaker open for {config.name}")

        # Try main execution with timeout
        try:
            if RecoveryStrategy.TIMEOUT in config.strategies:
                result = await self._execute_with_timeout(
                    function, config, error_context
                )
            else:
                result = await self._execute_function(function, error_context)

            # Success - reset circuit breaker if applicable
            if RecoveryStrategy.CIRCUIT_BREAKER in config.strategies:
                self._reset_circuit_breaker(config.name)

            error_context.recovery_successful = True
            error_context.final_result = result
            self.metrics[config.name].recovered_errors += 1

            return result

        except Exception as e:
            error_context.error = e
            error_context.error_time = datetime.now()
            error_context.severity = self._determine_error_severity(e, config)

            # Count this as an error (the original function failed)
            self.metrics[config.name].total_errors += 1

            # Record error distribution
            error_type = type(e).__name__
            self.metrics[config.name].error_distribution[error_type] = (
                self.metrics[config.name].error_distribution.get(error_type, 0) + 1
            )

            # Record circuit breaker failure
            if RecoveryStrategy.CIRCUIT_BREAKER in config.strategies:
                self._record_circuit_breaker_failure(config.name)

            # Try recovery strategies in order
            for strategy in config.strategies:
                try:
                    result = await self._apply_recovery_strategy(
                        strategy, config, function, error_context
                    )
                    # Check if recovery was successful (either returned result or compensation succeeded)
                    if result is not None or error_context.recovery_successful:
                        if not error_context.recovery_successful:
                            error_context.recovery_successful = True
                        error_context.final_result = result
                        self.metrics[config.name].recovered_errors += 1
                        return result

                except Exception as strategy_error:
                    logger.debug(
                        f"Recovery strategy {strategy} failed: {strategy_error}"
                    )
                    continue

            # All strategies failed
            self.metrics[config.name].failed_recoveries += 1
            raise e

    async def _execute_with_timeout(
        self,
        function: Callable,
        config: ErrorRecoveryConfig,
        error_context: ErrorContext,
    ) -> Any:
        """Execute function with timeout protection"""
        try:
            return await asyncio.wait_for(
                self._execute_function(function, error_context),
                timeout=config.timeout_seconds,
            )
        except TimeoutError:
            raise TimeoutError(
                f"Function {error_context.function_name} timed out after {config.timeout_seconds}s"
            )

    async def _execute_function(
        self, function: Callable, error_context: ErrorContext
    ) -> Any:
        """Execute the target function"""
        if asyncio.iscoroutinefunction(function):
            return await function(*error_context.args, **error_context.kwargs)
        else:
            # Run sync function in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor,
                functools.partial(
                    function, *error_context.args, **error_context.kwargs
                ),
            )

    async def _apply_recovery_strategy(
        self,
        strategy: RecoveryStrategy,
        config: ErrorRecoveryConfig,
        function: Callable,
        error_context: ErrorContext,
    ) -> Any:
        """Apply a specific recovery strategy"""
        metrics = self.metrics[config.name]
        strategy_metrics = metrics.strategy_effectiveness[strategy]
        strategy_metrics["attempts"] += 1

        error_context.recovery_strategies_used.append(strategy)

        try:
            if strategy == RecoveryStrategy.RETRY:
                result = await self._retry_strategy(config, function, error_context)
            elif strategy == RecoveryStrategy.FALLBACK:
                result = await self._fallback_strategy(config, error_context)
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                result = await self._graceful_degradation_strategy(
                    config, error_context
                )
            elif strategy == RecoveryStrategy.COMPENSATION:
                result = await self._compensation_strategy(config, error_context)
            else:
                logger.warning(f"Unknown recovery strategy: {strategy}")
                return None

            strategy_metrics["successes"] += 1
            return result

        except Exception as e:
            strategy_metrics["failures"] += 1
            raise e

    async def _retry_strategy(
        self,
        config: ErrorRecoveryConfig,
        function: Callable,
        error_context: ErrorContext,
    ) -> Any:
        """Implement retry strategy with exponential backoff"""
        delay = config.retry_delay_seconds

        for attempt in range(config.max_retry_attempts):
            error_context.attempt_number = attempt + 1
            self.metrics[config.name].retry_attempts += 1

            try:
                await asyncio.sleep(delay)
                return await self._execute_function(function, error_context)

            except Exception as e:
                if attempt == config.max_retry_attempts - 1:
                    raise e

                # Exponential backoff
                delay *= config.retry_backoff_multiplier
                logger.debug(
                    f"Retry {attempt + 1}/{config.max_retry_attempts} failed, waiting {delay}s"
                )

        raise error_context.error

    async def _fallback_strategy(
        self, config: ErrorRecoveryConfig, error_context: ErrorContext
    ) -> Any:
        """Implement fallback strategy"""
        if not config.fallback_function:
            raise FallbackNotAvailableException("No fallback function configured")

        self.metrics[config.name].fallback_invocations += 1

        logger.info(f"Executing fallback for {error_context.function_name}")

        try:
            if asyncio.iscoroutinefunction(config.fallback_function):
                return await config.fallback_function(
                    *error_context.args, **error_context.kwargs
                )
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    self._executor,
                    functools.partial(
                        config.fallback_function,
                        *error_context.args,
                        **error_context.kwargs,
                    ),
                )
        except Exception as e:
            raise FallbackNotAvailableException(f"Fallback function failed: {e}") from e

    async def _graceful_degradation_strategy(
        self, config: ErrorRecoveryConfig, error_context: ErrorContext
    ) -> Any:
        """Implement graceful degradation strategy"""
        logger.info(f"Applying graceful degradation for {error_context.function_name}")

        # Return a default/degraded response based on function name
        if "search" in error_context.function_name.lower():
            return {
                "results": [],
                "degraded": True,
                "message": "Search service temporarily unavailable",
            }
        elif "embedding" in error_context.function_name.lower():
            return {
                "embedding": [0.0] * 384,
                "degraded": True,
                "message": "Embedding service degraded",
            }
        elif "database" in error_context.function_name.lower():
            return {
                "data": [],
                "degraded": True,
                "message": "Database temporarily unavailable",
            }
        else:
            return {
                "degraded": True,
                "message": "Service temporarily degraded",
                "error": str(error_context.error),
            }

    async def _compensation_strategy(
        self, config: ErrorRecoveryConfig, error_context: ErrorContext
    ) -> Any:
        """Implement compensation strategy"""
        if not config.compensation_function:
            logger.warning("No compensation function configured")
            return None

        self.metrics[config.name].compensation_executions += 1

        logger.info(f"Executing compensation for {error_context.function_name}")

        try:
            if asyncio.iscoroutinefunction(config.compensation_function):
                await config.compensation_function(error_context)
            else:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self._executor,
                    functools.partial(config.compensation_function, error_context),
                )

            # Mark recovery as successful and return None to indicate compensation handled the error
            error_context.recovery_successful = True
            return None

        except Exception as e:
            raise CompensationFailedException(f"Compensation failed: {e}") from e

    def _is_circuit_breaker_open(self, config_name: str) -> bool:
        """Check if circuit breaker is open"""
        if config_name not in self.circuit_breakers:
            return False

        cb_data = self.circuit_breakers[config_name]
        if cb_data.get("state") == "open":
            # Check if timeout has elapsed
            if time.time() > cb_data.get("open_until", 0):
                # Transition to half-open
                cb_data["state"] = "half-open"
                return False
            return True

        return False

    def _record_circuit_breaker_failure(self, config_name: str):
        """Record a circuit breaker failure"""
        config = self.configs[config_name]

        if config_name not in self.circuit_breakers:
            self.circuit_breakers[config_name] = {
                "failure_count": 0,
                "state": "closed",
                "open_until": 0,
            }

        cb_data = self.circuit_breakers[config_name]
        cb_data["failure_count"] += 1

        # Check if we should open the circuit breaker
        if cb_data["failure_count"] >= config.circuit_breaker_threshold:
            cb_data["state"] = "open"
            cb_data["open_until"] = time.time() + config.circuit_breaker_timeout
            self.metrics[config_name].circuit_breaker_trips += 1
            logger.warning(f"Circuit breaker opened for {config_name}")

    def _reset_circuit_breaker(self, config_name: str):
        """Reset circuit breaker on successful execution"""
        if config_name in self.circuit_breakers:
            self.circuit_breakers[config_name] = {
                "failure_count": 0,
                "state": "closed",
                "open_until": 0,
            }

    def _determine_error_severity(
        self, error: Exception, config: ErrorRecoveryConfig
    ) -> ErrorSeverity:
        """Determine error severity based on error type and configuration"""
        error_type = type(error).__name__

        # Critical errors
        if any(
            keyword in error_type.lower()
            for keyword in ["system", "memory", "security"]
        ):
            return ErrorSeverity.CRITICAL

        # High severity errors
        if any(
            keyword in error_type.lower()
            for keyword in ["database", "connection", "timeout"]
        ):
            return ErrorSeverity.HIGH

        # Medium severity errors
        if any(
            keyword in error_type.lower() for keyword in ["http", "request", "network"]
        ):
            return ErrorSeverity.MEDIUM

        return ErrorSeverity.LOW

    def get_recovery_metrics(self, config_name: str) -> dict[str, Any]:
        """Get recovery metrics for a configuration"""
        if config_name not in self.metrics:
            return {}

        metrics = self.metrics[config_name]
        recovery_rate = (metrics.recovered_errors / max(1, metrics.total_errors)) * 100

        avg_recovery_time = (
            sum(metrics.recovery_times) / len(metrics.recovery_times)
            if metrics.recovery_times
            else 0
        )

        return {
            "config_name": config_name,
            "total_errors": metrics.total_errors,
            "recovered_errors": metrics.recovered_errors,
            "failed_recoveries": metrics.failed_recoveries,
            "recovery_rate_percent": recovery_rate,
            "retry_attempts": metrics.retry_attempts,
            "fallback_invocations": metrics.fallback_invocations,
            "circuit_breaker_trips": metrics.circuit_breaker_trips,
            "compensation_executions": metrics.compensation_executions,
            "average_recovery_time_seconds": avg_recovery_time,
            "error_distribution": dict(metrics.error_distribution),
            "strategy_effectiveness": {
                str(strategy): effectiveness
                for strategy, effectiveness in metrics.strategy_effectiveness.items()
            },
        }

    def get_all_metrics(self) -> dict[str, dict[str, Any]]:
        """Get metrics for all recovery configurations"""
        return {
            config_name: self.get_recovery_metrics(config_name)
            for config_name in self.configs.keys()
        }

    def reset_metrics(self, config_name: str | None = None):
        """Reset metrics for specific config or all configs"""
        if config_name:
            if config_name in self.metrics:
                self.metrics[config_name] = RecoveryMetrics()
        else:
            for name in self.metrics:
                self.metrics[name] = RecoveryMetrics()

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on error recovery system"""
        total_configs = len(self.configs)
        healthy_configs = 0

        circuit_breaker_status = {}
        for config_name in self.configs:
            cb_data = self.circuit_breakers.get(config_name, {})
            is_healthy = cb_data.get("state", "closed") != "open"

            if is_healthy:
                healthy_configs += 1

            circuit_breaker_status[config_name] = {
                "state": cb_data.get("state", "closed"),
                "failure_count": cb_data.get("failure_count", 0),
                "healthy": is_healthy,
            }

        overall_health = "healthy" if healthy_configs == total_configs else "degraded"
        if healthy_configs < total_configs * 0.5:
            overall_health = "unhealthy"

        return {
            "status": overall_health,
            "total_configs": total_configs,
            "healthy_configs": healthy_configs,
            "circuit_breaker_status": circuit_breaker_status,
            "timestamp": datetime.now().isoformat(),
        }


def error_recovery(config_name: str, manager: ErrorRecoveryManager | None = None):
    """
    Decorator for automatic error recovery.

    Args:
        config_name: Name of the recovery configuration to use
        manager: Optional custom recovery manager instance
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            recovery_manager = manager or get_error_recovery_manager()
            return await recovery_manager.execute_with_recovery(
                config_name, func, *args, **kwargs
            )

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            recovery_manager = manager or get_error_recovery_manager()
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                recovery_manager.execute_with_recovery(
                    config_name, func, *args, **kwargs
                )
            )

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Global recovery manager instance
_global_recovery_manager: ErrorRecoveryManager | None = None


def get_error_recovery_manager() -> ErrorRecoveryManager:
    """Get or create global error recovery manager"""
    global _global_recovery_manager
    if _global_recovery_manager is None:
        _global_recovery_manager = ErrorRecoveryManager()
    return _global_recovery_manager


def setup_default_recovery_configs():
    """Set up default recovery configurations for common scenarios"""
    manager = get_error_recovery_manager()

    # Search operations recovery
    search_config = ErrorRecoveryConfig(
        name="search_operations",
        strategies=[
            RecoveryStrategy.TIMEOUT,
            RecoveryStrategy.RETRY,
            RecoveryStrategy.FALLBACK,
            RecoveryStrategy.GRACEFUL_DEGRADATION,
        ],
        max_retry_attempts=2,
        retry_delay_seconds=0.5,
        timeout_seconds=10.0,
        fallback_function=lambda *args, **kwargs: {"results": [], "degraded": True},
    )

    # Database operations recovery
    database_config = ErrorRecoveryConfig(
        name="database_operations",
        strategies=[
            RecoveryStrategy.CIRCUIT_BREAKER,
            RecoveryStrategy.RETRY,
            RecoveryStrategy.FALLBACK,
        ],
        max_retry_attempts=3,
        retry_delay_seconds=1.0,
        circuit_breaker_threshold=5,
        circuit_breaker_timeout=30.0,
        fallback_function=lambda *args, **kwargs: {"data": [], "cached": True},
    )

    # API requests recovery
    api_config = ErrorRecoveryConfig(
        name="api_requests",
        strategies=[
            RecoveryStrategy.TIMEOUT,
            RecoveryStrategy.CIRCUIT_BREAKER,
            RecoveryStrategy.RETRY,
            RecoveryStrategy.GRACEFUL_DEGRADATION,
        ],
        max_retry_attempts=3,
        retry_delay_seconds=1.0,
        timeout_seconds=30.0,
        circuit_breaker_threshold=3,
        circuit_breaker_timeout=60.0,
    )

    # File operations recovery
    file_config = ErrorRecoveryConfig(
        name="file_operations",
        strategies=[RecoveryStrategy.RETRY, RecoveryStrategy.COMPENSATION],
        max_retry_attempts=2,
        retry_delay_seconds=0.5,
    )

    # Embedding generation recovery
    embedding_config = ErrorRecoveryConfig(
        name="embedding_operations",
        strategies=[
            RecoveryStrategy.TIMEOUT,
            RecoveryStrategy.RETRY,
            RecoveryStrategy.FALLBACK,
        ],
        max_retry_attempts=2,
        retry_delay_seconds=1.0,
        timeout_seconds=15.0,
        fallback_function=lambda *args, **kwargs: {
            "embedding": [0.0] * 384,
            "degraded": True,
        },
    )

    # Register all configs
    for config in [
        search_config,
        database_config,
        api_config,
        file_config,
        embedding_config,
    ]:
        manager.register_recovery_config(config)

    logger.info("Default error recovery configurations set up successfully")


# Convenience functions for common recovery patterns
async def retry_on_failure(
    func: Callable,
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    args: tuple = (),
    kwargs: dict = None,
) -> Any:
    """Simple retry mechanism for functions"""
    if kwargs is None:
        kwargs = {}

    for attempt in range(max_attempts):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_attempts - 1:
                raise e
            await asyncio.sleep(delay)
            delay *= backoff


async def with_fallback(
    func: Callable, fallback_func: Callable, args: tuple = (), kwargs: dict = None
) -> Any:
    """Execute function with fallback on failure"""
    if kwargs is None:
        kwargs = {}

    try:
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    except Exception:
        if asyncio.iscoroutinefunction(fallback_func):
            return await fallback_func(*args, **kwargs)
        else:
            return fallback_func(*args, **kwargs)


async def with_timeout(
    func: Callable, timeout_seconds: float, args: tuple = (), kwargs: dict = None
) -> Any:
    """Execute function with timeout protection"""
    if kwargs is None:
        kwargs = {}

    if asyncio.iscoroutinefunction(func):
        return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
    else:
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, functools.partial(func, *args, **kwargs)),
            timeout=timeout_seconds,
        )
