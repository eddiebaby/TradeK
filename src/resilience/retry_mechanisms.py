"""
Advanced Retry Mechanisms for TradeKnowledge.

This module provides sophisticated retry patterns including exponential backoff,
jitter, and adaptive retry strategies for improved resilience.
"""

import asyncio
import functools
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Retry strategy types"""

    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIBONACCI_BACKOFF = "fibonacci_backoff"


class JitterType(Enum):
    """Types of jitter to apply to retry delays"""

    NONE = "none"
    FULL = "full"  # Random between 0 and calculated delay
    EQUAL = "equal"  # Half fixed, half random
    DECORRELATED = "decorrelated"  # Based on previous delay


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms"""

    max_attempts: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay between retries
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    jitter: JitterType = JitterType.EQUAL
    exponential_base: float = 2.0  # Multiplier for exponential backoff
    retryable_exceptions: tuple = (Exception,)
    non_retryable_exceptions: tuple = ()
    retry_on_result: Callable[[Any], bool] | None = None
    on_retry: Callable[[int, Exception, float], None] | None = None


@dataclass
class RetryMetrics:
    """Metrics for retry operations"""

    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    total_retry_delay: float = 0.0
    last_attempt_time: datetime | None = None
    attempt_history: list[dict[str, Any]] = field(default_factory=list)


class RetryExhaustedException(Exception):
    """Raised when all retry attempts have been exhausted"""

    def __init__(self, message: str, last_exception: Exception, attempts: int):
        super().__init__(message)
        self.last_exception = last_exception
        self.attempts = attempts


class RetryableOperation:
    """
    Advanced retry mechanism with configurable strategies and jitter.

    Provides exponential backoff, linear backoff, fibonacci backoff,
    and various jitter patterns for resilient operation execution.
    """

    def __init__(self, config: RetryConfig):
        self.config = config
        self.metrics = RetryMetrics()

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with retry logic.

        Args:
            func: Function to execute (sync or async)
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of successful function execution

        Raises:
            RetryExhaustedException: When all retries are exhausted
        """
        last_exception = None

        for attempt in range(1, self.config.max_attempts + 1):
            self.metrics.total_attempts += 1
            self.metrics.last_attempt_time = datetime.now()

            try:
                # Execute the function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                # Check if result indicates retry is needed
                if self.config.retry_on_result and self.config.retry_on_result(result):
                    raise ValueError(f"Retry needed based on result: {result}")

                # Success
                self.metrics.successful_attempts += 1
                self._record_attempt(attempt, success=True)

                logger.debug(f"Operation succeeded on attempt {attempt}")
                return result

            except self.config.non_retryable_exceptions as e:
                # Non-retryable exception, fail immediately
                self.metrics.failed_attempts += 1
                self._record_attempt(attempt, success=False, exception=e)
                logger.error(f"Non-retryable exception: {e}")
                raise e

            except self.config.retryable_exceptions as e:
                last_exception = e
                self.metrics.failed_attempts += 1
                self._record_attempt(attempt, success=False, exception=e)

                if attempt == self.config.max_attempts:
                    # Last attempt failed
                    logger.error(
                        f"All {self.config.max_attempts} retry attempts exhausted"
                    )
                    raise RetryExhaustedException(
                        f"Operation failed after {self.config.max_attempts} attempts",
                        last_exception,
                        attempt,
                    )

                # Calculate delay and wait
                delay = self._calculate_delay(attempt)
                self.metrics.total_retry_delay += delay

                # Call retry callback if provided
                if self.config.on_retry:
                    try:
                        self.config.on_retry(attempt, e, delay)
                    except Exception as callback_error:
                        logger.warning(f"Error in retry callback: {callback_error}")

                logger.warning(
                    f"Attempt {attempt} failed: {e}. Retrying in {delay:.2f}s"
                )
                await asyncio.sleep(delay)

        # Should not reach here, but just in case
        raise RetryExhaustedException(
            f"Operation failed after {self.config.max_attempts} attempts",
            last_exception or Exception("Unknown error"),
            self.config.max_attempts,
        )

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt number"""
        if self.config.strategy == RetryStrategy.FIXED_DELAY:
            base_delay = self.config.base_delay
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            base_delay = self.config.base_delay * (
                self.config.exponential_base ** (attempt - 1)
            )
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            base_delay = self.config.base_delay * attempt
        elif self.config.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            base_delay = self.config.base_delay * self._fibonacci(attempt)
        else:
            base_delay = self.config.base_delay

        # Apply maximum delay limit
        base_delay = min(base_delay, self.config.max_delay)

        # Apply jitter
        final_delay = self._apply_jitter(base_delay, attempt)

        return max(0, final_delay)

    def _apply_jitter(self, delay: float, attempt: int) -> float:
        """Apply jitter to the delay"""
        if self.config.jitter == JitterType.NONE:
            return delay
        elif self.config.jitter == JitterType.FULL:
            return random.uniform(0, delay)
        elif self.config.jitter == JitterType.EQUAL:
            return delay * 0.5 + random.uniform(0, delay * 0.5)
        elif self.config.jitter == JitterType.DECORRELATED:
            # Use previous delay for decorrelated jitter
            if hasattr(self, "_last_delay"):
                return random.uniform(self.config.base_delay, self._last_delay * 3)
            else:
                return delay
        else:
            return delay

    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number"""
        if n <= 1:
            return 1
        elif n == 2:
            return 1
        else:
            a, b = 1, 1
            for _ in range(3, n + 1):
                a, b = b, a + b
            return b

    def _record_attempt(
        self, attempt: int, success: bool, exception: Exception | None = None
    ):
        """Record attempt details for metrics"""
        attempt_record = {
            "attempt": attempt,
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "exception": str(exception) if exception else None,
            "exception_type": type(exception).__name__ if exception else None,
        }
        self.metrics.attempt_history.append(attempt_record)

    def get_metrics(self) -> dict[str, Any]:
        """Get retry operation metrics"""
        return {
            "total_attempts": self.metrics.total_attempts,
            "successful_attempts": self.metrics.successful_attempts,
            "failed_attempts": self.metrics.failed_attempts,
            "success_rate": (
                self.metrics.successful_attempts / self.metrics.total_attempts
                if self.metrics.total_attempts > 0
                else 0
            ),
            "total_retry_delay": self.metrics.total_retry_delay,
            "average_delay_per_retry": (
                self.metrics.total_retry_delay / max(1, self.metrics.failed_attempts)
            ),
            "last_attempt_time": (
                self.metrics.last_attempt_time.isoformat()
                if self.metrics.last_attempt_time
                else None
            ),
            "attempt_history": self.metrics.attempt_history,
        }


def retry_with_backoff(config: RetryConfig):
    """
    Decorator for automatic retry with configurable backoff strategies.

    Args:
        config: Retry configuration

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: Callable) -> Callable:
        retry_op = RetryableOperation(config)

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await retry_op.execute(func, *args, **kwargs)

            # Add retry methods to wrapper
            async_wrapper.retry_operation = retry_op
            async_wrapper.get_metrics = retry_op.get_metrics
            return async_wrapper
        else:

            @functools.wraps(func)
            async def sync_wrapper(*args, **kwargs):
                return await retry_op.execute(func, *args, **kwargs)

            # Add retry methods to wrapper
            sync_wrapper.retry_operation = retry_op
            sync_wrapper.get_metrics = retry_op.get_metrics
            return sync_wrapper

    return decorator


class AdaptiveRetryManager:
    """
    Adaptive retry manager that learns from historical performance
    and adjusts retry strategies dynamically.
    """

    def __init__(self):
        self.operation_history: dict[str, list[dict[str, Any]]] = {}
        self.adaptive_configs: dict[str, RetryConfig] = {}
        self.learning_window_size = 100  # Number of operations to consider for learning

    def get_adaptive_config(
        self, operation_name: str, base_config: RetryConfig
    ) -> RetryConfig:
        """
        Get an adaptive retry configuration based on historical performance.

        Args:
            operation_name: Name/identifier for the operation
            base_config: Base configuration to adapt from

        Returns:
            Adapted retry configuration
        """
        if operation_name not in self.operation_history:
            # No history yet, use base config
            return base_config

        history = self.operation_history[operation_name]
        if len(history) < 10:
            # Not enough data for adaptation
            return base_config

        # Analyze recent performance
        recent_history = history[-self.learning_window_size :]

        # Calculate success rates by attempt number
        success_by_attempt = {}
        for record in recent_history:
            for attempt_info in record.get("attempts", []):
                attempt_num = attempt_info["attempt"]
                success = attempt_info["success"]

                if attempt_num not in success_by_attempt:
                    success_by_attempt[attempt_num] = {"success": 0, "total": 0}

                success_by_attempt[attempt_num]["total"] += 1
                if success:
                    success_by_attempt[attempt_num]["success"] += 1

        # Adapt configuration based on learned patterns
        adapted_config = RetryConfig(
            max_attempts=base_config.max_attempts,
            base_delay=base_config.base_delay,
            max_delay=base_config.max_delay,
            strategy=base_config.strategy,
            jitter=base_config.jitter,
            exponential_base=base_config.exponential_base,
            retryable_exceptions=base_config.retryable_exceptions,
            non_retryable_exceptions=base_config.non_retryable_exceptions,
        )

        # Adjust max attempts based on success patterns
        if len(success_by_attempt) > 0:
            # Find the attempt number where success rate drops significantly
            for attempt_num in sorted(success_by_attempt.keys()):
                success_rate = (
                    success_by_attempt[attempt_num]["success"]
                    / success_by_attempt[attempt_num]["total"]
                )

                # If success rate is very low after certain attempts, reduce max attempts
                if attempt_num > 2 and success_rate < 0.1:
                    adapted_config.max_attempts = min(
                        adapted_config.max_attempts, attempt_num - 1
                    )
                    break

        # Adjust delay based on average delays that led to success
        successful_delays = []
        for record in recent_history:
            if record.get("final_success", False):
                successful_delays.append(record.get("total_delay", 0))

        if successful_delays:
            avg_successful_delay = sum(successful_delays) / len(successful_delays)
            # Adjust base delay to be closer to what historically works
            adapted_config.base_delay = min(
                adapted_config.base_delay * 1.5,
                max(adapted_config.base_delay * 0.5, avg_successful_delay / 3),
            )

        self.adaptive_configs[operation_name] = adapted_config
        logger.debug(f"Adapted retry config for {operation_name}: {adapted_config}")

        return adapted_config

    def record_operation_result(self, operation_name: str, result: dict[str, Any]):
        """
        Record the result of a retry operation for learning.

        Args:
            operation_name: Name/identifier for the operation
            result: Result metrics from RetryableOperation
        """
        if operation_name not in self.operation_history:
            self.operation_history[operation_name] = []

        # Add timestamp and clean up old records
        result["recorded_at"] = datetime.now().isoformat()
        self.operation_history[operation_name].append(result)

        # Keep only recent history
        if len(self.operation_history[operation_name]) > self.learning_window_size * 2:
            self.operation_history[operation_name] = self.operation_history[
                operation_name
            ][-self.learning_window_size :]

    def get_operation_analytics(self, operation_name: str) -> dict[str, Any]:
        """Get analytics for a specific operation"""
        if operation_name not in self.operation_history:
            return {"error": "No history found for operation"}

        history = self.operation_history[operation_name]
        if not history:
            return {"error": "No history records"}

        total_operations = len(history)
        successful_operations = len(
            [r for r in history if r.get("successful_attempts", 0) > 0]
        )

        total_attempts = sum(r.get("total_attempts", 0) for r in history)
        total_delays = sum(r.get("total_retry_delay", 0) for r in history)

        return {
            "operation_name": operation_name,
            "total_operations": total_operations,
            "successful_operations": successful_operations,
            "operation_success_rate": (
                successful_operations / total_operations if total_operations > 0 else 0
            ),
            "average_attempts_per_operation": (
                total_attempts / total_operations if total_operations > 0 else 0
            ),
            "average_delay_per_operation": (
                total_delays / total_operations if total_operations > 0 else 0
            ),
            "has_adaptive_config": operation_name in self.adaptive_configs,
            "history_size": len(history),
        }


# Global adaptive retry manager
_adaptive_retry_manager = AdaptiveRetryManager()


def get_adaptive_retry_manager() -> AdaptiveRetryManager:
    """Get the global adaptive retry manager"""
    return _adaptive_retry_manager


# Convenience functions for common retry patterns
def create_database_retry_config() -> RetryConfig:
    """Create retry configuration optimized for database operations"""
    return RetryConfig(
        max_attempts=3,
        base_delay=0.5,
        max_delay=5.0,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        jitter=JitterType.EQUAL,
        exponential_base=2.0,
        retryable_exceptions=(ConnectionError, TimeoutError, OSError),
        non_retryable_exceptions=(PermissionError, FileNotFoundError),
    )


def create_api_retry_config() -> RetryConfig:
    """Create retry configuration optimized for API calls"""
    return RetryConfig(
        max_attempts=5,
        base_delay=1.0,
        max_delay=30.0,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        jitter=JitterType.DECORRELATED,
        exponential_base=2.0,
        retryable_exceptions=(ConnectionError, TimeoutError, Exception),
        non_retryable_exceptions=(PermissionError, ValueError),
    )


def create_file_retry_config() -> RetryConfig:
    """Create retry configuration optimized for file operations"""
    return RetryConfig(
        max_attempts=3,
        base_delay=0.1,
        max_delay=2.0,
        strategy=RetryStrategy.LINEAR_BACKOFF,
        jitter=JitterType.EQUAL,
        retryable_exceptions=(IOError, OSError),
        non_retryable_exceptions=(PermissionError, FileNotFoundError),
    )


async def retry_operation(
    func: Callable,
    config: RetryConfig | None = None,
    operation_name: str | None = None,
    *args,
    **kwargs,
) -> Any:
    """
    Convenience function to retry an operation with optional adaptive learning.

    Args:
        func: Function to retry
        config: Retry configuration (uses default if None)
        operation_name: Name for adaptive learning (optional)
        *args: Arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Result of successful function execution
    """
    if config is None:
        config = RetryConfig()

    # Use adaptive configuration if operation name is provided
    if operation_name:
        manager = get_adaptive_retry_manager()
        config = manager.get_adaptive_config(operation_name, config)

    retry_op = RetryableOperation(config)

    try:
        result = await retry_op.execute(func, *args, **kwargs)

        # Record successful operation for learning
        if operation_name:
            metrics = retry_op.get_metrics()
            metrics["final_success"] = True
            manager.record_operation_result(operation_name, metrics)

        return result

    except RetryExhaustedException as e:
        # Record failed operation for learning
        if operation_name:
            metrics = retry_op.get_metrics()
            metrics["final_success"] = False
            manager.record_operation_result(operation_name, metrics)

        raise e
