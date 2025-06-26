"""
Tests for Error Recovery Mechanisms.

This module tests the error recovery capabilities including
retry mechanisms, fallback strategies, and automatic recovery.
"""

import pytest
import asyncio
import time
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.resilience.error_recovery import (
    ErrorRecoveryManager,
    ErrorRecoveryConfig,
    RecoveryStrategy,
    ErrorSeverity,
    ErrorContext,
    ErrorRecoveryException,
    FallbackNotAvailableException,
    CompensationFailedException,
    error_recovery,
    get_error_recovery_manager,
    setup_default_recovery_configs,
    retry_on_failure,
    with_fallback,
    with_timeout
)


class TestErrorRecoveryConfig:
    """Test error recovery configuration"""
    
    def test_basic_config_creation(self):
        """Test basic configuration creation"""
        config = ErrorRecoveryConfig(
            name="test_recovery",
            strategies=[RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK],
            max_retry_attempts=5,
            timeout_seconds=30.0
        )
        
        assert config.name == "test_recovery"
        assert RecoveryStrategy.RETRY in config.strategies
        assert RecoveryStrategy.FALLBACK in config.strategies
        assert config.max_retry_attempts == 5
        assert config.timeout_seconds == 30.0
        assert config.auto_recovery_enabled is True
    
    def test_config_with_fallback_function(self):
        """Test configuration with fallback function"""
        def fallback_func():
            return "fallback_result"
        
        config = ErrorRecoveryConfig(
            name="fallback_test",
            fallback_function=fallback_func,
            strategies=[RecoveryStrategy.FALLBACK]
        )
        
        assert config.fallback_function == fallback_func
        assert RecoveryStrategy.FALLBACK in config.strategies
    
    def test_config_with_compensation_function(self):
        """Test configuration with compensation function"""
        def compensation_func(error_context):
            return f"compensated_{error_context.function_name}"
        
        config = ErrorRecoveryConfig(
            name="compensation_test",
            compensation_function=compensation_func,
            strategies=[RecoveryStrategy.COMPENSATION]
        )
        
        assert config.compensation_function == compensation_func
        assert RecoveryStrategy.COMPENSATION in config.strategies


class TestErrorContext:
    """Test error context functionality"""
    
    def test_error_context_creation(self):
        """Test error context creation with all fields"""
        start_time = datetime.now()
        error = ValueError("Test error")
        
        context = ErrorContext(
            error=error,
            function_name="test_function",
            args=("arg1", "arg2"),
            kwargs={"key": "value"},
            attempt_number=1,
            start_time=start_time,
            error_time=start_time,
            severity=ErrorSeverity.HIGH
        )
        
        assert context.error == error
        assert context.function_name == "test_function"
        assert context.args == ("arg1", "arg2")
        assert context.kwargs == {"key": "value"}
        assert context.attempt_number == 1
        assert context.severity == ErrorSeverity.HIGH
        assert context.recovery_successful is False
        assert len(context.recovery_strategies_used) == 0


class TestErrorRecoveryManager:
    """Test error recovery manager functionality"""
    
    @pytest.fixture
    def manager(self):
        """Create error recovery manager for testing"""
        return ErrorRecoveryManager()
    
    @pytest.fixture
    def basic_config(self):
        """Create basic recovery configuration"""
        return ErrorRecoveryConfig(
            name="test_config",
            strategies=[RecoveryStrategy.RETRY],
            max_retry_attempts=2,
            retry_delay_seconds=0.1,  # Short delay for testing
            timeout_seconds=5.0
        )
    
    def test_register_recovery_config(self, manager, basic_config):
        """Test registering recovery configuration"""
        manager.register_recovery_config(basic_config)
        
        assert "test_config" in manager.configs
        assert manager.configs["test_config"] == basic_config
        assert "test_config" in manager.metrics
        assert len(manager.metrics["test_config"].strategy_effectiveness) > 0
    
    @pytest.mark.asyncio
    async def test_successful_execution_no_error(self, manager, basic_config):
        """Test successful execution when no error occurs"""
        manager.register_recovery_config(basic_config)
        
        async def success_func(value):
            return f"success_{value}"
        
        result = await manager.execute_with_recovery("test_config", success_func, "test")
        
        assert result == "success_test"
        metrics = manager.get_recovery_metrics("test_config")
        assert metrics["recovered_errors"] == 1
        assert metrics["total_errors"] == 0
    
    @pytest.mark.asyncio
    async def test_retry_strategy_success(self, manager):
        """Test successful retry strategy"""
        config = ErrorRecoveryConfig(
            name="retry_test",
            strategies=[RecoveryStrategy.RETRY],
            max_retry_attempts=3,
            retry_delay_seconds=0.01
        )
        manager.register_recovery_config(config)
        
        call_count = 0
        
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success_after_retries"
        
        result = await manager.execute_with_recovery("retry_test", flaky_func)
        
        assert result == "success_after_retries"
        assert call_count == 3
        metrics = manager.get_recovery_metrics("retry_test")
        assert metrics["retry_attempts"] > 0
        assert metrics["recovered_errors"] == 1
    
    @pytest.mark.asyncio
    async def test_retry_strategy_failure(self, manager):
        """Test retry strategy when all attempts fail"""
        config = ErrorRecoveryConfig(
            name="retry_fail_test",
            strategies=[RecoveryStrategy.RETRY],
            max_retry_attempts=2,
            retry_delay_seconds=0.01
        )
        manager.register_recovery_config(config)
        
        async def always_fail_func():
            raise ValueError("Always fails")
        
        with pytest.raises(ErrorRecoveryException):
            await manager.execute_with_recovery("retry_fail_test", always_fail_func)
        
        metrics = manager.get_recovery_metrics("retry_fail_test")
        assert metrics["retry_attempts"] == 2
        assert metrics["failed_recoveries"] == 1
    
    @pytest.mark.asyncio
    async def test_fallback_strategy_success(self, manager):
        """Test successful fallback strategy"""
        def fallback_func(*args, **kwargs):
            return f"fallback_result_{args[0] if args else 'default'}"
        
        config = ErrorRecoveryConfig(
            name="fallback_test",
            strategies=[RecoveryStrategy.FALLBACK],
            fallback_function=fallback_func
        )
        manager.register_recovery_config(config)
        
        async def failing_func(value):
            raise ConnectionError("Connection failed")
        
        result = await manager.execute_with_recovery("fallback_test", failing_func, "test")
        
        assert result == "fallback_result_test"
        metrics = manager.get_recovery_metrics("fallback_test")
        assert metrics["fallback_invocations"] == 1
        assert metrics["recovered_errors"] == 1
    
    @pytest.mark.asyncio
    async def test_fallback_strategy_async_fallback(self, manager):
        """Test fallback strategy with async fallback function"""
        async def async_fallback_func(*args, **kwargs):
            await asyncio.sleep(0.01)
            return f"async_fallback_{args[0] if args else 'default'}"
        
        config = ErrorRecoveryConfig(
            name="async_fallback_test",
            strategies=[RecoveryStrategy.FALLBACK],
            fallback_function=async_fallback_func
        )
        manager.register_recovery_config(config)
        
        async def failing_func(value):
            raise TimeoutError("Request timeout")
        
        result = await manager.execute_with_recovery("async_fallback_test", failing_func, "test")
        
        assert result == "async_fallback_test"
        metrics = manager.get_recovery_metrics("async_fallback_test")
        assert metrics["fallback_invocations"] == 1
    
    @pytest.mark.asyncio
    async def test_fallback_strategy_no_function(self, manager):
        """Test fallback strategy when no fallback function is configured"""
        config = ErrorRecoveryConfig(
            name="no_fallback_test",
            strategies=[RecoveryStrategy.FALLBACK]
            # No fallback_function provided
        )
        manager.register_recovery_config(config)
        
        async def failing_func():
            raise ValueError("Always fails")
        
        with pytest.raises(ErrorRecoveryException):
            await manager.execute_with_recovery("no_fallback_test", failing_func)
    
    @pytest.mark.asyncio
    async def test_timeout_strategy(self, manager):
        """Test timeout strategy"""
        config = ErrorRecoveryConfig(
            name="timeout_test",
            strategies=[RecoveryStrategy.TIMEOUT],
            timeout_seconds=0.1  # Very short timeout
        )
        manager.register_recovery_config(config)
        
        async def slow_func():
            await asyncio.sleep(0.5)  # Longer than timeout
            return "should_not_reach"
        
        with pytest.raises(ErrorRecoveryException):
            await manager.execute_with_recovery("timeout_test", slow_func)
        
        metrics = manager.get_recovery_metrics("timeout_test")
        assert metrics["failed_recoveries"] == 1
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_strategy(self, manager):
        """Test graceful degradation strategy"""
        config = ErrorRecoveryConfig(
            name="degradation_test",
            strategies=[RecoveryStrategy.GRACEFUL_DEGRADATION]
        )
        manager.register_recovery_config(config)
        
        async def search_func():
            raise ConnectionError("Search service down")
        search_func.__name__ = "search_function"
        
        result = await manager.execute_with_recovery("degradation_test", search_func)
        
        assert isinstance(result, dict)
        assert result["degraded"] is True
        assert "results" in result
        assert result["results"] == []
        metrics = manager.get_recovery_metrics("degradation_test")
        assert metrics["recovered_errors"] == 1
    
    @pytest.mark.asyncio
    async def test_compensation_strategy(self, manager):
        """Test compensation strategy"""
        compensation_called = False
        
        def compensation_func(error_context):
            nonlocal compensation_called
            compensation_called = True
            assert error_context.function_name == "compensated_function"
        
        config = ErrorRecoveryConfig(
            name="compensation_test",
            strategies=[RecoveryStrategy.COMPENSATION],
            compensation_function=compensation_func
        )
        manager.register_recovery_config(config)
        
        async def failing_func():
            raise ValueError("Operation failed")
        failing_func.__name__ = "compensated_function"
        
        result = await manager.execute_with_recovery("compensation_test", failing_func)
        
        assert result is None  # Compensation doesn't return a result
        assert compensation_called is True
        metrics = manager.get_recovery_metrics("compensation_test")
        assert metrics["compensation_executions"] == 1
        assert metrics["recovered_errors"] == 1
    
    @pytest.mark.asyncio
    async def test_multiple_strategies_cascade(self, manager):
        """Test multiple recovery strategies in cascade"""
        def fallback_func(*args, **kwargs):
            return "fallback_success"
        
        config = ErrorRecoveryConfig(
            name="cascade_test",
            strategies=[RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK],
            max_retry_attempts=2,
            retry_delay_seconds=0.01,
            fallback_function=fallback_func
        )
        manager.register_recovery_config(config)
        
        async def always_fail_func():
            raise ConnectionError("Always fails")
        
        result = await manager.execute_with_recovery("cascade_test", always_fail_func)
        
        assert result == "fallback_success"
        metrics = manager.get_recovery_metrics("cascade_test")
        assert metrics["retry_attempts"] == 2
        assert metrics["fallback_invocations"] == 1
        assert metrics["recovered_errors"] == 1
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, manager):
        """Test circuit breaker functionality"""
        config = ErrorRecoveryConfig(
            name="circuit_breaker_test",
            strategies=[RecoveryStrategy.CIRCUIT_BREAKER],
            circuit_breaker_threshold=3,
            circuit_breaker_timeout=0.1  # Short timeout for testing
        )
        manager.register_recovery_config(config)
        
        async def failing_func():
            raise ConnectionError("Service unavailable")
        
        # Trigger circuit breaker by failing threshold times
        for _ in range(3):
            with pytest.raises(ErrorRecoveryException):
                await manager.execute_with_recovery("circuit_breaker_test", failing_func)
        
        # Circuit breaker should now be open
        with pytest.raises(ErrorRecoveryException):
            await manager.execute_with_recovery("circuit_breaker_test", failing_func)
        
        metrics = manager.get_recovery_metrics("circuit_breaker_test")
        assert metrics["circuit_breaker_trips"] >= 1
    
    @pytest.mark.asyncio
    async def test_sync_function_execution(self, manager, basic_config):
        """Test execution of synchronous functions"""
        manager.register_recovery_config(basic_config)
        
        def sync_func(value):
            return f"sync_{value}"
        
        result = await manager.execute_with_recovery("test_config", sync_func, "test")
        
        assert result == "sync_test"
        metrics = manager.get_recovery_metrics("test_config")
        assert metrics["recovered_errors"] == 1
    
    @pytest.mark.asyncio
    async def test_error_severity_determination(self, manager):
        """Test error severity determination"""
        config = ErrorRecoveryConfig(name="severity_test")
        manager.register_recovery_config(config)
        
        # Test different error types
        database_error = ConnectionError("Database connection failed")
        severity = manager._determine_error_severity(database_error, config)
        assert severity == ErrorSeverity.HIGH
        
        memory_error = MemoryError("Out of memory")
        severity = manager._determine_error_severity(memory_error, config)
        assert severity == ErrorSeverity.CRITICAL
        
        http_error = ValueError("HTTP request failed")
        severity = manager._determine_error_severity(http_error, config)
        assert severity == ErrorSeverity.LOW
    
    def test_metrics_collection(self, manager, basic_config):
        """Test metrics collection and calculation"""
        manager.register_recovery_config(basic_config)
        
        # Simulate some metrics
        metrics = manager.metrics["test_config"]
        metrics.total_errors = 10
        metrics.recovered_errors = 8
        metrics.failed_recoveries = 2
        metrics.retry_attempts = 15
        metrics.recovery_times = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        result_metrics = manager.get_recovery_metrics("test_config")
        
        assert result_metrics["total_errors"] == 10
        assert result_metrics["recovered_errors"] == 8
        assert result_metrics["failed_recoveries"] == 2
        assert result_metrics["recovery_rate_percent"] == 80.0
        assert result_metrics["retry_attempts"] == 15
        assert result_metrics["average_recovery_time_seconds"] == 0.3
    
    def test_metrics_reset(self, manager, basic_config):
        """Test metrics reset functionality"""
        manager.register_recovery_config(basic_config)
        
        # Set some metrics
        metrics = manager.metrics["test_config"]
        metrics.total_errors = 5
        metrics.recovered_errors = 3
        
        manager.reset_metrics("test_config")
        
        reset_metrics = manager.get_recovery_metrics("test_config")
        assert reset_metrics["total_errors"] == 0
        assert reset_metrics["recovered_errors"] == 0
    
    @pytest.mark.asyncio
    async def test_health_check(self, manager, basic_config):
        """Test error recovery system health check"""
        manager.register_recovery_config(basic_config)
        
        health = await manager.health_check()
        
        assert "status" in health
        assert "total_configs" in health
        assert "healthy_configs" in health
        assert "circuit_breaker_status" in health
        assert health["total_configs"] == 1
        assert health["healthy_configs"] == 1
        assert health["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_nonexistent_config_error(self, manager):
        """Test error when using nonexistent configuration"""
        async def test_func():
            return "test"
        
        with pytest.raises(ValueError, match="Recovery config 'nonexistent' not found"):
            await manager.execute_with_recovery("nonexistent", test_func)


class TestErrorRecoveryDecorator:
    """Test error recovery decorator"""
    
    @pytest.mark.asyncio
    async def test_decorator_with_async_function(self):
        """Test decorator with async function"""
        manager = ErrorRecoveryManager()
        config = ErrorRecoveryConfig(
            name="decorator_test",
            strategies=[RecoveryStrategy.RETRY],
            max_retry_attempts=2,
            retry_delay_seconds=0.01
        )
        manager.register_recovery_config(config)
        
        call_count = 0
        
        @error_recovery("decorator_test", manager=manager)
        async def decorated_func(value):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("First call fails")
            return f"decorated_{value}"
        
        result = await decorated_func("test")
        
        assert result == "decorated_test"
        assert call_count == 2
    
    def test_decorator_with_sync_function(self):
        """Test decorator with sync function"""
        manager = ErrorRecoveryManager()
        config = ErrorRecoveryConfig(
            name="sync_decorator_test",
            strategies=[RecoveryStrategy.RETRY],
            max_retry_attempts=2,
            retry_delay_seconds=0.01
        )
        manager.register_recovery_config(config)
        
        call_count = 0
        
        @error_recovery("sync_decorator_test", manager=manager)
        def sync_decorated_func(value):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("First call fails")
            return f"sync_decorated_{value}"
        
        # Note: This runs the async wrapper in the event loop
        result = sync_decorated_func("test")
        
        assert result == "sync_decorated_test"
        assert call_count == 2


class TestConvenienceFunctions:
    """Test convenience functions for common recovery patterns"""
    
    @pytest.mark.asyncio
    async def test_retry_on_failure_success(self):
        """Test retry_on_failure function success"""
        call_count = 0
        
        async def flaky_func(value):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return f"success_{value}"
        
        result = await retry_on_failure(flaky_func, max_attempts=3, delay=0.01, args=("test",))
        
        assert result == "success_test"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_on_failure_exhausted(self):
        """Test retry_on_failure when retries are exhausted"""
        async def always_fail_func():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            await retry_on_failure(always_fail_func, max_attempts=2, delay=0.01)
    
    @pytest.mark.asyncio
    async def test_with_fallback_success(self):
        """Test with_fallback function success"""
        async def primary_func(value):
            return f"primary_{value}"
        
        async def fallback_func(value):
            return f"fallback_{value}"
        
        result = await with_fallback(primary_func, fallback_func, args=("test",))
        
        assert result == "primary_test"
    
    @pytest.mark.asyncio
    async def test_with_fallback_failure(self):
        """Test with_fallback when primary fails"""
        async def failing_func(value):
            raise ValueError("Primary failed")
        
        async def fallback_func(value):
            return f"fallback_{value}"
        
        result = await with_fallback(failing_func, fallback_func, args=("test",))
        
        assert result == "fallback_test"
    
    @pytest.mark.asyncio
    async def test_with_timeout_success(self):
        """Test with_timeout function success"""
        async def quick_func(value):
            await asyncio.sleep(0.01)
            return f"quick_{value}"
        
        result = await with_timeout(quick_func, timeout_seconds=1.0, args=("test",))
        
        assert result == "quick_test"
    
    @pytest.mark.asyncio
    async def test_with_timeout_failure(self):
        """Test with_timeout when timeout is exceeded"""
        async def slow_func():
            await asyncio.sleep(0.5)
            return "should_not_reach"
        
        with pytest.raises(asyncio.TimeoutError):
            await with_timeout(slow_func, timeout_seconds=0.1)


class TestDefaultRecoveryConfigs:
    """Test default recovery configurations"""
    
    def test_setup_default_recovery_configs(self):
        """Test setting up default recovery configurations"""
        # Get the global manager that setup_default_recovery_configs uses
        manager = get_error_recovery_manager()
        
        # Clear any existing configs
        manager.configs.clear()
        manager.metrics.clear()
        
        setup_default_recovery_configs()
        
        # Check that default configs were created
        expected_configs = [
            "search_operations",
            "database_operations", 
            "api_requests",
            "file_operations",
            "embedding_operations"
        ]
        
        for config_name in expected_configs:
            assert config_name in manager.configs
            config = manager.configs[config_name]
            assert isinstance(config, ErrorRecoveryConfig)
            assert config.name == config_name
            assert len(config.strategies) > 0


class TestErrorRecoveryIntegration:
    """Test integration scenarios and edge cases"""
    
    @pytest.mark.asyncio
    async def test_complex_recovery_scenario(self):
        """Test complex recovery scenario with multiple strategies"""
        manager = ErrorRecoveryManager()
        
        fallback_called = False
        compensation_called = False
        
        def fallback_func(*args, **kwargs):
            nonlocal fallback_called
            fallback_called = True
            return "fallback_result"
        
        def compensation_func(error_context):
            nonlocal compensation_called
            compensation_called = True
        
        config = ErrorRecoveryConfig(
            name="complex_recovery",
            strategies=[
                RecoveryStrategy.TIMEOUT,
                RecoveryStrategy.RETRY,
                RecoveryStrategy.CIRCUIT_BREAKER,
                RecoveryStrategy.FALLBACK,
                RecoveryStrategy.COMPENSATION
            ],
            max_retry_attempts=2,
            retry_delay_seconds=0.01,
            timeout_seconds=1.0,
            circuit_breaker_threshold=5,
            fallback_function=fallback_func,
            compensation_function=compensation_func
        )
        manager.register_recovery_config(config)
        
        async def complex_failing_func():
            raise ConnectionError("Complex failure")
        
        result = await manager.execute_with_recovery("complex_recovery", complex_failing_func)
        
        # Should succeed with fallback
        assert result == "fallback_result"
        assert fallback_called is True
        
        metrics = manager.get_recovery_metrics("complex_recovery")
        assert metrics["retry_attempts"] == 2
        assert metrics["fallback_invocations"] == 1
        assert metrics["recovered_errors"] == 1
    
    @pytest.mark.asyncio
    async def test_high_load_recovery_performance(self):
        """Test error recovery under high load"""
        manager = ErrorRecoveryManager()
        
        config = ErrorRecoveryConfig(
            name="high_load_test",
            strategies=[RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK],
            max_retry_attempts=1,
            retry_delay_seconds=0.001,
            fallback_function=lambda *args, **kwargs: "fallback"
        )
        manager.register_recovery_config(config)
        
        async def failing_func(request_id):
            raise ValueError(f"Request {request_id} failed")
        
        # Execute many concurrent operations
        tasks = [
            manager.execute_with_recovery("high_load_test", failing_func, i)
            for i in range(50)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed with fallback
        assert len(results) == 50
        assert all(result == "fallback" for result in results)
        
        metrics = manager.get_recovery_metrics("high_load_test")
        assert metrics["total_errors"] == 50
        assert metrics["recovered_errors"] == 50
        assert metrics["fallback_invocations"] == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])