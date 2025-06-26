"""
Tests for Advanced Retry Mechanisms.

This module tests the retry patterns including exponential backoff,
jitter, and adaptive retry strategies.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.resilience.retry_mechanisms import (
    RetryableOperation,
    RetryConfig,
    RetryStrategy,
    JitterType,
    RetryExhaustedException,
    AdaptiveRetryManager,
    retry_with_backoff,
    create_database_retry_config,
    create_api_retry_config,
    create_file_retry_config,
    retry_operation,
    get_adaptive_retry_manager
)


class TestRetryableOperation:
    """Test core retry operation functionality"""
    
    @pytest.fixture
    def basic_config(self):
        """Create basic retry configuration for testing"""
        return RetryConfig(
            max_attempts=3,
            base_delay=0.1,  # Short delay for testing
            max_delay=1.0,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            jitter=JitterType.NONE  # No jitter for predictable testing
        )
    
    @pytest.fixture
    def retry_operation(self, basic_config):
        """Create retry operation instance"""
        return RetryableOperation(basic_config)
    
    @pytest.mark.asyncio
    async def test_successful_operation_first_attempt(self, retry_operation):
        """Test operation that succeeds on first attempt"""
        async def successful_func():
            return "success"
        
        result = await retry_operation.execute(successful_func)
        
        assert result == "success"
        metrics = retry_operation.get_metrics()
        assert metrics["total_attempts"] == 1
        assert metrics["successful_attempts"] == 1
        assert metrics["failed_attempts"] == 0
        assert metrics["success_rate"] == 1.0
    
    @pytest.mark.asyncio
    async def test_successful_operation_after_retries(self, retry_operation):
        """Test operation that succeeds after some failures"""
        attempt_count = 0
        
        async def eventually_successful_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError(f"Attempt {attempt_count} failed")
            return "success"
        
        result = await retry_operation.execute(eventually_successful_func)
        
        assert result == "success"
        metrics = retry_operation.get_metrics()
        assert metrics["total_attempts"] == 3
        assert metrics["successful_attempts"] == 1
        assert metrics["failed_attempts"] == 2
        assert metrics["success_rate"] == 1/3
    
    @pytest.mark.asyncio
    async def test_retry_exhaustion(self, retry_operation):
        """Test behavior when all retry attempts are exhausted"""
        async def always_failing_func():
            raise ConnectionError("Always fails")
        
        with pytest.raises(RetryExhaustedException) as exc_info:
            await retry_operation.execute(always_failing_func)
        
        assert exc_info.value.attempts == 3
        assert isinstance(exc_info.value.last_exception, ConnectionError)
        
        metrics = retry_operation.get_metrics()
        assert metrics["total_attempts"] == 3
        assert metrics["successful_attempts"] == 0
        assert metrics["failed_attempts"] == 3
        assert metrics["success_rate"] == 0.0
    
    @pytest.mark.asyncio
    async def test_non_retryable_exception(self):
        """Test that non-retryable exceptions are not retried"""
        config = RetryConfig(
            max_attempts=3,
            base_delay=0.1,
            retryable_exceptions=(ConnectionError,),
            non_retryable_exceptions=(ValueError,)
        )
        retry_op = RetryableOperation(config)
        
        async def non_retryable_func():
            raise ValueError("This should not be retried")
        
        with pytest.raises(ValueError):
            await retry_op.execute(non_retryable_func)
        
        metrics = retry_op.get_metrics()
        assert metrics["total_attempts"] == 1  # Should not retry
        assert metrics["failed_attempts"] == 1
    
    @pytest.mark.asyncio
    async def test_sync_function_execution(self, retry_operation):
        """Test execution of synchronous functions"""
        def sync_function(value):
            return f"sync_{value}"
        
        result = await retry_operation.execute(sync_function, "test")
        
        assert result == "sync_test"
        metrics = retry_operation.get_metrics()
        assert metrics["total_attempts"] == 1
        assert metrics["successful_attempts"] == 1
    
    def test_delay_calculation_exponential_backoff(self):
        """Test exponential backoff delay calculation"""
        config = RetryConfig(
            base_delay=1.0,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            exponential_base=2.0,
            jitter=JitterType.NONE
        )
        retry_op = RetryableOperation(config)
        
        delay1 = retry_op._calculate_delay(1)
        delay2 = retry_op._calculate_delay(2)
        delay3 = retry_op._calculate_delay(3)
        
        assert delay1 == 1.0  # base_delay * 2^0
        assert delay2 == 2.0  # base_delay * 2^1
        assert delay3 == 4.0  # base_delay * 2^2
    
    def test_delay_calculation_linear_backoff(self):
        """Test linear backoff delay calculation"""
        config = RetryConfig(
            base_delay=1.0,
            strategy=RetryStrategy.LINEAR_BACKOFF,
            jitter=JitterType.NONE
        )
        retry_op = RetryableOperation(config)
        
        delay1 = retry_op._calculate_delay(1)
        delay2 = retry_op._calculate_delay(2)
        delay3 = retry_op._calculate_delay(3)
        
        assert delay1 == 1.0  # base_delay * 1
        assert delay2 == 2.0  # base_delay * 2
        assert delay3 == 3.0  # base_delay * 3
    
    def test_delay_calculation_fibonacci_backoff(self):
        """Test Fibonacci backoff delay calculation"""
        config = RetryConfig(
            base_delay=1.0,
            strategy=RetryStrategy.FIBONACCI_BACKOFF,
            jitter=JitterType.NONE
        )
        retry_op = RetryableOperation(config)
        
        delay1 = retry_op._calculate_delay(1)
        delay2 = retry_op._calculate_delay(2)
        delay3 = retry_op._calculate_delay(3)
        delay4 = retry_op._calculate_delay(4)
        
        assert delay1 == 1.0  # base_delay * fib(1) = 1
        assert delay2 == 1.0  # base_delay * fib(2) = 1
        assert delay3 == 2.0  # base_delay * fib(3) = 2
        assert delay4 == 3.0  # base_delay * fib(4) = 3
    
    def test_max_delay_limit(self):
        """Test that delays don't exceed max_delay"""
        config = RetryConfig(
            base_delay=1.0,
            max_delay=5.0,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            exponential_base=2.0,
            jitter=JitterType.NONE
        )
        retry_op = RetryableOperation(config)
        
        # This would normally be 16.0 (1 * 2^4), but should be capped at 5.0
        delay = retry_op._calculate_delay(5)
        assert delay == 5.0
    
    def test_jitter_application(self):
        """Test that jitter is applied correctly"""
        config = RetryConfig(
            base_delay=1.0,
            strategy=RetryStrategy.FIXED_DELAY,
            jitter=JitterType.FULL
        )
        retry_op = RetryableOperation(config)
        
        # With full jitter, delay should be between 0 and base_delay
        delays = [retry_op._calculate_delay(1) for _ in range(10)]
        
        assert all(0 <= delay <= 1.0 for delay in delays)
        # Should have some variation (very unlikely all delays are the same)
        assert len(set(delays)) > 1
    
    def test_fibonacci_calculation(self):
        """Test Fibonacci number calculation"""
        config = RetryConfig()
        retry_op = RetryableOperation(config)
        
        assert retry_op._fibonacci(1) == 1
        assert retry_op._fibonacci(2) == 1
        assert retry_op._fibonacci(3) == 2
        assert retry_op._fibonacci(4) == 3
        assert retry_op._fibonacci(5) == 5
        assert retry_op._fibonacci(6) == 8
    
    @pytest.mark.asyncio
    async def test_retry_callback(self):
        """Test retry callback functionality"""
        callback_calls = []
        
        def retry_callback(attempt, exception, delay):
            callback_calls.append({
                "attempt": attempt,
                "exception": str(exception),
                "delay": delay
            })
        
        config = RetryConfig(
            max_attempts=3,
            base_delay=0.1,
            on_retry=retry_callback
        )
        retry_op = RetryableOperation(config)
        
        attempt_count = 0
        async def failing_then_success():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError(f"Attempt {attempt_count}")
            return "success"
        
        result = await retry_op.execute(failing_then_success)
        
        assert result == "success"
        assert len(callback_calls) == 2  # Called for first 2 failures
        assert callback_calls[0]["attempt"] == 1
        assert callback_calls[1]["attempt"] == 2
    
    @pytest.mark.asyncio
    async def test_retry_on_result_condition(self):
        """Test retrying based on result condition"""
        def should_retry_result(result):
            return result == "retry_me"
        
        config = RetryConfig(
            max_attempts=3,
            base_delay=0.1,
            retry_on_result=should_retry_result
        )
        retry_op = RetryableOperation(config)
        
        attempt_count = 0
        async def conditional_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                return "retry_me"
            return "success"
        
        result = await retry_op.execute(conditional_func)
        
        assert result == "success"
        metrics = retry_op.get_metrics()
        assert metrics["total_attempts"] == 3
        assert metrics["failed_attempts"] == 2  # First 2 attempts "failed" due to result


class TestRetryDecorator:
    """Test retry decorator functionality"""
    
    @pytest.mark.asyncio
    async def test_async_function_decorator(self):
        """Test decorator on async functions"""
        config = RetryConfig(max_attempts=3, base_delay=0.1)
        
        attempt_count = 0
        
        @retry_with_backoff(config)
        async def decorated_async_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Retry needed")
            return "success"
        
        result = await decorated_async_func()
        
        assert result == "success"
        assert attempt_count == 3
        
        # Check that retry methods are available
        assert hasattr(decorated_async_func, 'retry_operation')
        assert hasattr(decorated_async_func, 'get_metrics')
        
        metrics = decorated_async_func.get_metrics()
        assert metrics["total_attempts"] == 3
    
    @pytest.mark.asyncio
    async def test_sync_function_decorator(self):
        """Test decorator on synchronous functions"""
        config = RetryConfig(max_attempts=3, base_delay=0.1)
        
        attempt_count = 0
        
        @retry_with_backoff(config)
        def decorated_sync_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise ConnectionError("Retry needed")
            return "sync_success"
        
        result = await decorated_sync_func()
        
        assert result == "sync_success"
        assert attempt_count == 2


class TestAdaptiveRetryManager:
    """Test adaptive retry manager functionality"""
    
    @pytest.fixture
    def manager(self):
        """Create adaptive retry manager for testing"""
        return AdaptiveRetryManager()
    
    def test_initial_config_no_history(self, manager):
        """Test that base config is returned when no history exists"""
        base_config = RetryConfig(max_attempts=5, base_delay=1.0)
        
        adapted_config = manager.get_adaptive_config("new_operation", base_config)
        
        assert adapted_config.max_attempts == base_config.max_attempts
        assert adapted_config.base_delay == base_config.base_delay
    
    def test_insufficient_history(self, manager):
        """Test behavior with insufficient history for adaptation"""
        base_config = RetryConfig(max_attempts=5, base_delay=1.0)
        
        # Add a few records (less than threshold)
        for i in range(5):
            manager.record_operation_result("test_op", {
                "total_attempts": 2,
                "successful_attempts": 1,
                "total_retry_delay": 1.0,
                "attempts": [
                    {"attempt": 1, "success": False},
                    {"attempt": 2, "success": True}
                ]
            })
        
        adapted_config = manager.get_adaptive_config("test_op", base_config)
        
        # Should return base config due to insufficient history
        assert adapted_config.max_attempts == base_config.max_attempts
    
    def test_adaptive_max_attempts_reduction(self, manager):
        """Test that max attempts is reduced when later attempts rarely succeed"""
        base_config = RetryConfig(max_attempts=5, base_delay=1.0)
        
        # Simulate history where attempts beyond 3 never succeed
        for i in range(50):
            manager.record_operation_result("test_op", {
                "total_attempts": 5,
                "successful_attempts": 0 if i % 10 == 0 else 1,  # Mostly fail completely
                "total_retry_delay": 5.0,
                "attempts": [
                    {"attempt": 1, "success": False},
                    {"attempt": 2, "success": False},
                    {"attempt": 3, "success": i % 10 != 0},  # Sometimes succeed on attempt 3
                    {"attempt": 4, "success": False},        # Never succeed on 4+
                    {"attempt": 5, "success": False}
                ]
            })
        
        adapted_config = manager.get_adaptive_config("test_op", base_config)
        
        # Should reduce max attempts since later attempts don't help
        assert adapted_config.max_attempts <= 3
    
    def test_operation_analytics(self, manager):
        """Test operation analytics generation"""
        # Add some operation history
        for i in range(10):
            manager.record_operation_result("analytics_test", {
                "total_attempts": 2,
                "successful_attempts": 1,
                "failed_attempts": 1,
                "total_retry_delay": 1.5
            })
        
        analytics = manager.get_operation_analytics("analytics_test")
        
        assert analytics["operation_name"] == "analytics_test"
        assert analytics["total_operations"] == 10
        assert analytics["successful_operations"] == 10
        assert analytics["operation_success_rate"] == 1.0
        assert analytics["average_attempts_per_operation"] == 2.0
        assert analytics["average_delay_per_operation"] == 1.5
    
    def test_analytics_for_unknown_operation(self, manager):
        """Test analytics for operation with no history"""
        analytics = manager.get_operation_analytics("unknown_operation")
        
        assert "error" in analytics
        assert "No history found" in analytics["error"]


class TestRetryConfigFactories:
    """Test retry configuration factory functions"""
    
    def test_database_retry_config(self):
        """Test database-specific retry configuration"""
        config = create_database_retry_config()
        
        assert config.max_attempts == 3
        assert config.base_delay == 0.5
        assert config.max_delay == 5.0
        assert config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF
        assert ConnectionError in config.retryable_exceptions
        assert PermissionError in config.non_retryable_exceptions
    
    def test_api_retry_config(self):
        """Test API-specific retry configuration"""
        config = create_api_retry_config()
        
        assert config.max_attempts == 5
        assert config.base_delay == 1.0
        assert config.max_delay == 30.0
        assert config.jitter == JitterType.DECORRELATED
        assert ConnectionError in config.retryable_exceptions
    
    def test_file_retry_config(self):
        """Test file operation-specific retry configuration"""
        config = create_file_retry_config()
        
        assert config.max_attempts == 3
        assert config.strategy == RetryStrategy.LINEAR_BACKOFF
        assert IOError in config.retryable_exceptions
        assert FileNotFoundError in config.non_retryable_exceptions


class TestConvenienceFunctions:
    """Test convenience functions for retry operations"""
    
    @pytest.mark.asyncio
    async def test_retry_operation_function(self):
        """Test retry_operation convenience function"""
        attempt_count = 0
        
        async def test_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Retry needed")
            return "success"
        
        config = RetryConfig(max_attempts=3, base_delay=0.1)
        result = await retry_operation(test_func, config)
        
        assert result == "success"
        assert attempt_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_operation_with_adaptive_learning(self):
        """Test retry_operation with adaptive learning enabled"""
        attempt_count = 0
        
        async def test_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise ConnectionError("Retry needed")
            return "adaptive_success"
        
        config = RetryConfig(max_attempts=3, base_delay=0.1)
        
        # First call should record results for learning
        result = await retry_operation(test_func, config, operation_name="adaptive_test")
        
        assert result == "adaptive_success"
        
        # Check that the operation was recorded
        manager = get_adaptive_retry_manager()
        analytics = manager.get_operation_analytics("adaptive_test")
        assert analytics["total_operations"] == 1
    
    @pytest.mark.asyncio
    async def test_retry_operation_default_config(self):
        """Test retry_operation with default configuration"""
        async def simple_func():
            return "default_success"
        
        result = await retry_operation(simple_func)
        
        assert result == "default_success"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])