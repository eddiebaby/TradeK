"""
Circuit Breaker Tests for TradeKnowledge.

This module tests the circuit breaker implementation including:
- Basic circuit breaker functionality
- State transitions (closed -> open -> half-open -> closed)
- Timeout handling and failure detection
- Metrics collection and reporting
- Circuit breaker manager functionality
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CircuitBreakerOpenException,
    CircuitBreakerTimeoutException,
    CircuitBreakerManager,
    circuit_breaker,
    get_circuit_breaker_manager,
    setup_circuit_breakers,
    call_with_circuit_breaker
)


class TestCircuitBreakerBasics:
    """Test basic circuit breaker functionality"""
    
    @pytest.fixture
    def config(self):
        """Create test circuit breaker configuration"""
        return CircuitBreakerConfig(
            name="test_service",
            failure_threshold=3,
            recovery_timeout=1.0,  # Short timeout for testing
            success_threshold=2,
            timeout=0.5,
            expected_exception=(ValueError, ConnectionError)
        )
    
    @pytest.fixture
    def circuit_breaker(self, config):
        """Create circuit breaker instance"""
        return CircuitBreaker(config)
    
    @pytest.mark.asyncio
    async def test_successful_calls(self, circuit_breaker):
        """Test circuit breaker with successful calls"""
        
        async def successful_function(value):
            return value * 2
        
        # Multiple successful calls should work normally
        for i in range(5):
            result = await circuit_breaker.call(successful_function, i)
            assert result == i * 2
        
        # Check metrics
        metrics = circuit_breaker.get_metrics()
        assert metrics["state"] == "closed"
        assert metrics["total_requests"] == 5
        assert metrics["successful_requests"] == 5
        assert metrics["failed_requests"] == 0
        assert metrics["success_rate"] == 1.0
    
    @pytest.mark.asyncio
    async def test_failing_calls_open_circuit(self, circuit_breaker):
        """Test that failing calls open the circuit"""
        
        async def failing_function():
            raise ValueError("Service unavailable")
        
        # Make failing calls up to threshold
        for i in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failing_function)
            
            # Should still be closed until threshold reached
            if i < circuit_breaker.config.failure_threshold - 1:
                assert circuit_breaker.metrics.current_state == CircuitState.CLOSED
        
        # Circuit should now be open
        assert circuit_breaker.metrics.current_state == CircuitState.OPEN
        
        # Further calls should fail fast
        with pytest.raises(CircuitBreakerOpenException):
            await circuit_breaker.call(failing_function)
        
        # Check metrics
        metrics = circuit_breaker.get_metrics()
        assert metrics["state"] == "open"
        assert metrics["failed_requests"] == circuit_breaker.config.failure_threshold
        assert metrics["circuit_opened_count"] == 1
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, circuit_breaker):
        """Test circuit breaker timeout handling"""
        
        async def slow_function():
            await asyncio.sleep(1.0)  # Longer than timeout
            return "completed"
        
        # Should timeout and count as failure
        with pytest.raises(CircuitBreakerTimeoutException):
            await circuit_breaker.call(slow_function)
        
        metrics = circuit_breaker.get_metrics()
        assert metrics["failed_requests"] == 1
        assert metrics["consecutive_failures"] == 1
    
    @pytest.mark.asyncio
    async def test_recovery_cycle(self, circuit_breaker):
        """Test complete recovery cycle: closed -> open -> half-open -> closed"""
        
        async def controlled_function(should_fail=True):
            if should_fail:
                raise ValueError("Service error")
            return "success"
        
        # Step 1: Fail enough times to open circuit
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(ValueError):
                await circuit_breaker.call(controlled_function, should_fail=True)
        
        assert circuit_breaker.metrics.current_state == CircuitState.OPEN
        
        # Step 2: Wait for recovery timeout
        await asyncio.sleep(circuit_breaker.config.recovery_timeout + 0.1)
        
        # Step 3: First call after timeout should transition to half-open
        result = await circuit_breaker.call(controlled_function, should_fail=False)
        assert result == "success"
        assert circuit_breaker.metrics.current_state == CircuitState.HALF_OPEN
        
        # Step 4: Enough successful calls should close the circuit
        for _ in range(circuit_breaker.config.success_threshold - 1):
            await circuit_breaker.call(controlled_function, should_fail=False)
        
        assert circuit_breaker.metrics.current_state == CircuitState.CLOSED
        
        # Verify metrics
        metrics = circuit_breaker.get_metrics()
        assert metrics["state"] == "closed"
        assert metrics["successful_requests"] == circuit_breaker.config.success_threshold
    
    @pytest.mark.asyncio
    async def test_half_open_failure_reopens_circuit(self, circuit_breaker):
        """Test that failure in half-open state reopens circuit"""
        
        async def controlled_function(should_fail=True):
            if should_fail:
                raise ValueError("Service error")
            return "success"
        
        # Open the circuit
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(ValueError):
                await circuit_breaker.call(controlled_function, should_fail=True)
        
        assert circuit_breaker.metrics.current_state == CircuitState.OPEN
        
        # Wait for recovery timeout
        await asyncio.sleep(circuit_breaker.config.recovery_timeout + 0.1)
        
        # One successful call to enter half-open
        await circuit_breaker.call(controlled_function, should_fail=False)
        assert circuit_breaker.metrics.current_state == CircuitState.HALF_OPEN
        
        # Failure in half-open should reopen circuit
        with pytest.raises(ValueError):
            await circuit_breaker.call(controlled_function, should_fail=True)
        
        assert circuit_breaker.metrics.current_state == CircuitState.OPEN
    
    def test_sync_function_support(self, circuit_breaker):
        """Test circuit breaker with synchronous functions"""
        
        def sync_function(value):
            return value * 3
        
        async def test_sync():
            result = await circuit_breaker.call(sync_function, 5)
            assert result == 15
        
        # Run the async test
        asyncio.run(test_sync())
        
        metrics = circuit_breaker.get_metrics()
        assert metrics["successful_requests"] == 1
    
    def test_unexpected_exception_handling(self, circuit_breaker):
        """Test handling of unexpected exceptions"""
        
        async def function_with_unexpected_error():
            raise RuntimeError("Unexpected error")  # Not in expected_exception
        
        async def test_unexpected():
            # Unexpected exceptions should not count as failures
            with pytest.raises(RuntimeError):
                await circuit_breaker.call(function_with_unexpected_error)
            
            # Circuit should remain closed
            assert circuit_breaker.metrics.current_state == CircuitState.CLOSED
            assert circuit_breaker.metrics.failed_requests == 0
        
        asyncio.run(test_unexpected())
    
    def test_circuit_breaker_reset(self, circuit_breaker):
        """Test circuit breaker reset functionality"""
        
        async def failing_function():
            raise ValueError("Service error")
        
        async def test_reset():
            # Fail some calls
            for _ in range(2):
                with pytest.raises(ValueError):
                    await circuit_breaker.call(failing_function)
            
            assert circuit_breaker.metrics.failed_requests == 2
            
            # Reset circuit breaker
            circuit_breaker.reset()
            
            # Check that metrics are reset
            metrics = circuit_breaker.get_metrics()
            assert metrics["state"] == "closed"
            assert metrics["total_requests"] == 0
            assert metrics["failed_requests"] == 0
            assert metrics["consecutive_failures"] == 0
        
        asyncio.run(test_reset())
    
    def test_force_state_changes(self, circuit_breaker):
        """Test forcing circuit breaker state changes"""
        
        # Force open
        circuit_breaker.force_open()
        assert circuit_breaker.metrics.current_state == CircuitState.OPEN
        
        # Force close
        circuit_breaker.force_close()
        assert circuit_breaker.metrics.current_state == CircuitState.CLOSED


class TestCircuitBreakerDecorator:
    """Test circuit breaker decorator functionality"""
    
    def test_async_function_decorator(self):
        """Test decorator with async functions"""
        
        config = CircuitBreakerConfig(
            name="decorated_async",
            failure_threshold=2,
            recovery_timeout=0.5,
            timeout=0.2
        )
        
        @circuit_breaker(config)
        async def decorated_async_function(value, should_fail=False):
            if should_fail:
                raise ValueError("Decorator test error")
            return value * 2
        
        async def test_decorator():
            # Test successful call
            result = await decorated_async_function(5)
            assert result == 10
            
            # Test failure
            with pytest.raises(ValueError):
                await decorated_async_function(5, should_fail=True)
            
            # Check that circuit breaker methods are available
            assert hasattr(decorated_async_function, 'circuit_breaker')
            assert hasattr(decorated_async_function, 'get_metrics')
            
            metrics = decorated_async_function.get_metrics()
            assert metrics['successful_requests'] == 1
            assert metrics['failed_requests'] == 1
        
        asyncio.run(test_decorator())
    
    def test_sync_function_decorator(self):
        """Test decorator with sync functions"""
        
        config = CircuitBreakerConfig(
            name="decorated_sync",
            failure_threshold=2,
            recovery_timeout=0.5,
            timeout=0.2
        )
        
        @circuit_breaker(config)
        def decorated_sync_function(value, should_fail=False):
            if should_fail:
                raise ValueError("Sync decorator test error")
            return value * 3
        
        async def test_sync_decorator():
            # Test successful call
            result = await decorated_sync_function(4)
            assert result == 12
            
            # Test failure
            with pytest.raises(ValueError):
                await decorated_sync_function(4, should_fail=True)
            
            metrics = decorated_sync_function.get_metrics()
            assert metrics['successful_requests'] == 1
            assert metrics['failed_requests'] == 1
        
        asyncio.run(test_sync_decorator())


class TestCircuitBreakerManager:
    """Test circuit breaker manager functionality"""
    
    @pytest.fixture
    def manager(self):
        """Create fresh circuit breaker manager for testing"""
        return CircuitBreakerManager()
    
    def test_register_and_get_circuit_breaker(self, manager):
        """Test registering and retrieving circuit breakers"""
        
        config = CircuitBreakerConfig(name="test_service")
        cb = manager.register_circuit_breaker("test_service", config)
        
        assert cb is not None
        assert manager.get_circuit_breaker("test_service") is cb
        assert manager.get_circuit_breaker("nonexistent") is None
    
    def test_default_configurations(self, manager):
        """Test creation of default configurations"""
        
        default_configs = manager.create_default_configs()
        
        expected_services = [
            "qdrant_vector_db",
            "ollama_embedding_service", 
            "sqlite_database",
            "file_system",
            "external_api"
        ]
        
        for service in expected_services:
            assert service in default_configs
            config = default_configs[service]
            assert config.name == service
            assert config.failure_threshold > 0
            assert config.recovery_timeout > 0
            assert config.timeout > 0
    
    def test_setup_default_circuit_breakers(self, manager):
        """Test setting up default circuit breakers"""
        
        manager.setup_default_circuit_breakers()
        
        # Check that all default circuit breakers were created
        expected_services = [
            "qdrant_vector_db",
            "ollama_embedding_service",
            "sqlite_database", 
            "file_system",
            "external_api"
        ]
        
        for service in expected_services:
            cb = manager.get_circuit_breaker(service)
            assert cb is not None
            assert cb.config.name == service
    
    def test_get_all_metrics(self, manager):
        """Test getting metrics for all circuit breakers"""
        
        # Set up some circuit breakers
        manager.setup_default_circuit_breakers()
        
        all_metrics = manager.get_all_metrics()
        
        assert len(all_metrics) == 5  # Number of default circuit breakers
        for service_name, metrics in all_metrics.items():
            assert "state" in metrics
            assert "total_requests" in metrics
            assert "success_rate" in metrics
    
    def test_health_summary(self, manager):
        """Test health summary functionality"""
        
        # Start with empty manager
        health = manager.get_health_summary()
        assert health["total_circuit_breakers"] == 0
        assert health["overall_health"] == "healthy"
        
        # Add circuit breakers
        manager.setup_default_circuit_breakers()
        
        health = manager.get_health_summary()
        assert health["total_circuit_breakers"] == 5
        assert health["open_circuit_breakers"] == 0
        assert health["overall_health"] == "healthy"
        
        # Force one circuit breaker open
        cb = manager.get_circuit_breaker("qdrant_vector_db")
        cb.force_open()
        
        health = manager.get_health_summary()
        assert health["open_circuit_breakers"] == 1
        assert health["overall_health"] == "degraded"
    
    def test_reset_all_circuit_breakers(self, manager):
        """Test resetting all circuit breakers"""
        
        manager.setup_default_circuit_breakers()
        
        # Force some circuit breakers to different states
        cb1 = manager.get_circuit_breaker("qdrant_vector_db")
        cb2 = manager.get_circuit_breaker("ollama_embedding_service")
        
        cb1.force_open()
        cb2.metrics.total_requests = 100
        
        # Reset all
        manager.reset_all()
        
        # Check that all are reset
        for cb in manager.circuit_breakers.values():
            assert cb.metrics.current_state == CircuitState.CLOSED
            assert cb.metrics.total_requests == 0


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_call_with_circuit_breaker_function(self):
        """Test convenience function for calling with circuit breakers"""
        
        # Set up circuit breakers
        setup_circuit_breakers()
        
        async def test_function(value):
            return value + 10
        
        # Test successful call
        result = await call_with_circuit_breaker(
            "qdrant_vector_db",
            test_function,
            5
        )
        assert result == 15
        
        # Test with nonexistent circuit breaker
        with pytest.raises(ValueError, match="Circuit breaker 'nonexistent' not found"):
            await call_with_circuit_breaker("nonexistent", test_function, 5)
    
    @pytest.mark.asyncio
    async def test_real_world_service_simulation(self):
        """Test circuit breaker with simulated real-world service behavior"""
        
        config = CircuitBreakerConfig(
            name="unstable_service",
            failure_threshold=3,
            recovery_timeout=0.5,
            success_threshold=2,
            timeout=0.3
        )
        
        cb = CircuitBreaker(config)
        
        # Simulate unstable service
        call_count = 0
        
        async def unstable_service():
            nonlocal call_count
            call_count += 1
            
            # Fail for first 5 calls, then succeed
            if call_count <= 5:
                raise ConnectionError("Service temporarily unavailable")
            return f"Success on call {call_count}"
        
        # Make calls that will initially fail
        failures = 0
        for i in range(10):
            try:
                result = await cb.call(unstable_service)
                print(f"Call {i+1}: {result}")
                break
            except (ConnectionError, CircuitBreakerOpenException):
                failures += 1
                print(f"Call {i+1}: Failed")
                
                # If circuit is open, wait for recovery
                if cb.metrics.current_state == CircuitState.OPEN:
                    print("Circuit open, waiting for recovery...")
                    await asyncio.sleep(0.6)  # Wait longer than recovery timeout
        
        # Circuit should eventually recover and succeed
        final_metrics = cb.get_metrics()
        print(f"Final metrics: {final_metrics}")
        
        # We should have some failures and eventual success
        assert final_metrics["failed_requests"] >= 3
        assert final_metrics["successful_requests"] >= 1
    
    @pytest.mark.asyncio
    async def test_concurrent_circuit_breaker_access(self):
        """Test circuit breaker behavior under concurrent access"""
        
        config = CircuitBreakerConfig(
            name="concurrent_service",
            failure_threshold=5,
            recovery_timeout=0.5,
            timeout=0.2
        )
        
        cb = CircuitBreaker(config)
        
        async def service_call(call_id, should_fail=False):
            if should_fail:
                raise ValueError(f"Call {call_id} failed")
            await asyncio.sleep(0.01)  # Small delay
            return f"Success {call_id}"
        
        # Launch concurrent successful calls
        success_tasks = [
            cb.call(service_call, i, should_fail=False)
            for i in range(10)
        ]
        
        results = await asyncio.gather(*success_tasks, return_exceptions=True)
        
        # All should succeed
        successful_results = [r for r in results if isinstance(r, str)]
        assert len(successful_results) == 10
        
        # Launch concurrent failing calls
        failure_tasks = [
            cb.call(service_call, i, should_fail=True)
            for i in range(10)
        ]
        
        failure_results = await asyncio.gather(*failure_tasks, return_exceptions=True)
        
        # Some should fail with ValueError, others might fail fast if circuit opens
        value_errors = [r for r in failure_results if isinstance(r, ValueError)]
        circuit_open_errors = [r for r in failure_results if isinstance(r, CircuitBreakerOpenException)]
        
        assert len(value_errors) + len(circuit_open_errors) == 10
        
        final_metrics = cb.get_metrics()
        print(f"Concurrent test metrics: {final_metrics}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])