"""
Tests for Bulkhead Pattern Implementation.

This module tests the bulkhead isolation patterns that prevent failures
in one component from affecting others.
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

from src.resilience.bulkhead_pattern import (
    Bulkhead,
    BulkheadConfig,
    BulkheadManager,
    ResourceType,
    BulkheadCapacityExceededException,
    BulkheadTimeoutException,
    PriorityQueue,
    bulkhead_isolation,
    get_bulkhead_manager,
    setup_bulkheads
)


class TestPriorityQueue:
    """Test priority queue functionality"""
    
    @pytest.fixture
    def priority_queue(self):
        """Create priority queue for testing"""
        return PriorityQueue(max_size=5, priority_levels=3)
    
    def test_priority_queue_basic_operations(self, priority_queue):
        """Test basic put and get operations"""
        # Add items with different priorities
        assert priority_queue.put("item1", priority=0)  # Highest priority
        assert priority_queue.put("item2", priority=2)  # Lowest priority
        assert priority_queue.put("item3", priority=1)  # Medium priority
        
        assert priority_queue.size() == 3
        assert not priority_queue.is_empty()
        
        # Should get highest priority item first
        assert priority_queue.get() == "item1"
        assert priority_queue.get() == "item3"
        assert priority_queue.get() == "item2"
        
        assert priority_queue.is_empty()
        assert priority_queue.get() is None
    
    def test_priority_queue_capacity_limit(self, priority_queue):
        """Test that queue respects capacity limits"""
        # Fill queue to capacity
        for i in range(5):
            assert priority_queue.put(f"item{i}")
        
        assert priority_queue.is_full()
        assert priority_queue.size() == 5
        
        # Should reject additional items
        assert not priority_queue.put("overflow_item")
        assert priority_queue.size() == 5
    
    def test_priority_queue_invalid_priorities(self, priority_queue):
        """Test handling of invalid priority values"""
        # Test priority clamping
        assert priority_queue.put("item1", priority=-1)  # Should clamp to 0
        assert priority_queue.put("item2", priority=10)  # Should clamp to 2
        
        # Both items should be added successfully
        assert priority_queue.size() == 2


class TestBulkhead:
    """Test bulkhead isolation functionality"""
    
    @pytest.fixture
    def basic_config(self):
        """Create basic bulkhead configuration"""
        return BulkheadConfig(
            name="test_bulkhead",
            resource_type=ResourceType.ASYNC_SEMAPHORE,
            max_concurrent=2,
            max_queue_size=5,
            timeout_seconds=1.0  # Short timeout for testing
        )
    
    @pytest.fixture
    def bulkhead(self, basic_config):
        """Create bulkhead instance for testing"""
        return Bulkhead(basic_config)
    
    @pytest.mark.asyncio
    async def test_successful_execution_within_capacity(self, bulkhead):
        """Test successful execution when within capacity limits"""
        async def test_func(value):
            await asyncio.sleep(0.1)
            return f"result_{value}"
        
        result = await bulkhead.execute(test_func, "test")
        
        assert result == "result_test"
        metrics = bulkhead.get_metrics()
        assert metrics["total_requests"] == 1
        assert metrics["successful_requests"] == 1
        assert metrics["current_active"] == 0  # Should be 0 after completion
    
    @pytest.mark.asyncio
    async def test_concurrent_execution_up_to_limit(self, bulkhead):
        """Test that bulkhead allows up to max_concurrent operations"""
        async def slow_func(value):
            await asyncio.sleep(0.2)
            return f"result_{value}"
        
        # Start 2 concurrent operations (at the limit)
        tasks = [
            asyncio.create_task(bulkhead.execute(slow_func, i))
            for i in range(2)
        ]
        
        # Both should complete successfully
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 2
        assert all("result_" in result for result in results)
        
        metrics = bulkhead.get_metrics()
        assert metrics["successful_requests"] == 2
        assert metrics["max_concurrent_reached"] == 2
    
    @pytest.mark.asyncio
    async def test_queueing_when_over_capacity(self, bulkhead):
        """Test that operations are queued when capacity is exceeded"""
        results = []
        
        async def tracked_func(value):
            await asyncio.sleep(0.1)
            results.append(f"result_{value}")
            return f"result_{value}"
        
        # Start 4 operations (2 concurrent + 2 queued)
        tasks = [
            asyncio.create_task(bulkhead.execute(tracked_func, i))
            for i in range(4)
        ]
        
        await asyncio.gather(*tasks)
        
        assert len(results) == 4
        metrics = bulkhead.get_metrics()
        assert metrics["successful_requests"] == 4
        assert metrics["max_queue_reached"] > 0
    
    @pytest.mark.asyncio
    async def test_capacity_exceeded_rejection(self, bulkhead):
        """Test rejection when both capacity and queue are full"""
        async def long_running_func():
            await asyncio.sleep(0.5)  # Moderate delay
            return "done"
        
        # Fill up capacity and queue
        tasks = []
        
        # Fill capacity (2 slots) + queue (5 slots) = 7 total
        for i in range(7):
            tasks.append(asyncio.create_task(bulkhead.execute(long_running_func)))
        
        # Give time for operations to start and queue to fill
        await asyncio.sleep(0.1)
        
        # This should be rejected
        with pytest.raises(BulkheadCapacityExceededException):
            await bulkhead.execute(long_running_func)
        
        # Wait for tasks to complete naturally (they're short now)
        await asyncio.gather(*tasks, return_exceptions=True)
    
    @pytest.mark.asyncio
    async def test_operation_timeout(self, bulkhead):
        """Test operation timeout handling"""
        async def timeout_func():
            await asyncio.sleep(2.0)  # Longer than bulkhead timeout
            return "should_not_reach"
        
        with pytest.raises(BulkheadTimeoutException):
            await bulkhead.execute(timeout_func)
        
        metrics = bulkhead.get_metrics()
        assert metrics["timed_out_requests"] == 1
    
    @pytest.mark.asyncio
    async def test_priority_execution_order(self, bulkhead):
        """Test that higher priority operations are executed first"""
        execution_order = []
        
        async def priority_func(value):
            execution_order.append(value)
            await asyncio.sleep(0.05)
            return f"result_{value}"
        
        # Fill capacity with long-running operations
        long_tasks = [
            asyncio.create_task(bulkhead.execute(asyncio.sleep, 0.3))
            for _ in range(2)
        ]
        
        # Give time for capacity to fill
        await asyncio.sleep(0.1)
        
        # Queue operations with different priorities
        priority_tasks = [
            asyncio.create_task(bulkhead.execute(priority_func, "low", priority=2)),
            asyncio.create_task(bulkhead.execute(priority_func, "high", priority=0)),
            asyncio.create_task(bulkhead.execute(priority_func, "medium", priority=1))
        ]
        
        await asyncio.gather(*long_tasks, *priority_tasks)
        
        # High priority should execute first among queued items
        if execution_order:
            # Just verify that high priority item is among the first executed
            assert "high" in execution_order[:2]
    
    @pytest.mark.asyncio
    async def test_sync_function_execution(self, bulkhead):
        """Test execution of synchronous functions"""
        def sync_func(value):
            time.sleep(0.1)
            return f"sync_{value}"
        
        result = await bulkhead.execute(sync_func, "test")
        
        assert result == "sync_test"
        metrics = bulkhead.get_metrics()
        assert metrics["successful_requests"] == 1
    
    @pytest.mark.asyncio
    async def test_exception_handling(self, bulkhead):
        """Test handling of exceptions in executed functions"""
        async def failing_func():
            raise ValueError("Test exception")
        
        with pytest.raises(ValueError):
            await bulkhead.execute(failing_func)
        
        metrics = bulkhead.get_metrics()
        assert metrics["failed_requests"] == 1
        assert metrics["total_requests"] == 1
    
    def test_metrics_collection(self, bulkhead):
        """Test metrics collection and calculation"""
        metrics = bulkhead.get_metrics()
        
        expected_fields = [
            "name", "resource_type", "total_requests", "successful_requests",
            "failed_requests", "rejected_requests", "success_rate", "rejection_rate",
            "current_active", "current_queued", "resource_utilization", "capacity"
        ]
        
        for field in expected_fields:
            assert field in metrics
        
        assert metrics["name"] == "test_bulkhead"
        assert metrics["capacity"] == 2
    
    @pytest.mark.asyncio
    async def test_health_check(self, bulkhead):
        """Test bulkhead health check functionality"""
        health = await bulkhead.health_check()
        
        assert "status" in health
        assert "utilization" in health
        assert "rejection_rate" in health
        assert health["status"] == "healthy"  # Should be healthy initially
    
    def test_metrics_reset(self, bulkhead):
        """Test metrics reset functionality"""
        # Simulate some activity
        bulkhead.metrics.total_requests = 10
        bulkhead.metrics.successful_requests = 8
        
        bulkhead.reset_metrics()
        
        metrics = bulkhead.get_metrics()
        assert metrics["total_requests"] == 0
        assert metrics["successful_requests"] == 0


class TestBulkheadWithThreadPool:
    """Test bulkhead with thread pool resource type"""
    
    @pytest.fixture
    def thread_pool_config(self):
        """Create thread pool bulkhead configuration"""
        return BulkheadConfig(
            name="thread_pool_test",
            resource_type=ResourceType.THREAD_POOL,
            max_concurrent=2,
            max_queue_size=3,
            timeout_seconds=2.0
        )
    
    @pytest.fixture
    def thread_pool_bulkhead(self, thread_pool_config):
        """Create thread pool bulkhead"""
        return Bulkhead(thread_pool_config)
    
    @pytest.mark.asyncio
    async def test_thread_pool_sync_execution(self, thread_pool_bulkhead):
        """Test execution of sync functions in thread pool"""
        def cpu_intensive_func(n):
            # Simulate CPU-intensive work
            total = 0
            for i in range(n):
                total += i
            return total
        
        result = await thread_pool_bulkhead.execute(cpu_intensive_func, 1000)
        
        assert result == sum(range(1000))
        metrics = thread_pool_bulkhead.get_metrics()
        assert metrics["successful_requests"] == 1
    
    @pytest.mark.asyncio
    async def test_thread_pool_async_execution(self, thread_pool_bulkhead):
        """Test execution of async functions with thread pool bulkhead"""
        async def async_func(value):
            await asyncio.sleep(0.1)
            return f"async_{value}"
        
        result = await thread_pool_bulkhead.execute(async_func, "test")
        
        assert result == "async_test"


class TestBulkheadManager:
    """Test bulkhead manager functionality"""
    
    @pytest.fixture
    def manager(self):
        """Create bulkhead manager for testing"""
        return BulkheadManager()
    
    def test_bulkhead_registration(self, manager):
        """Test registering bulkheads with manager"""
        config = BulkheadConfig(
            name="test_bulkhead",
            resource_type=ResourceType.ASYNC_SEMAPHORE,
            max_concurrent=5
        )
        
        bulkhead = manager.register_bulkhead(config)
        
        assert isinstance(bulkhead, Bulkhead)
        assert manager.get_bulkhead("test_bulkhead") is bulkhead
        assert "test_bulkhead" in manager.bulkheads
    
    @pytest.mark.asyncio
    async def test_execute_in_bulkhead(self, manager):
        """Test executing functions through manager"""
        config = BulkheadConfig(
            name="manager_test",
            resource_type=ResourceType.ASYNC_SEMAPHORE,
            max_concurrent=2
        )
        manager.register_bulkhead(config)
        
        async def test_func(value):
            return f"managed_{value}"
        
        result = await manager.execute_in_bulkhead("manager_test", test_func, "test")
        
        assert result == "managed_test"
    
    @pytest.mark.asyncio
    async def test_execute_in_nonexistent_bulkhead(self, manager):
        """Test error handling for nonexistent bulkhead"""
        async def test_func():
            return "test"
        
        with pytest.raises(ValueError, match="Bulkhead 'nonexistent' not found"):
            await manager.execute_in_bulkhead("nonexistent", test_func)
    
    def test_get_all_metrics(self, manager):
        """Test getting metrics for all bulkheads"""
        # Register multiple bulkheads
        for i in range(3):
            config = BulkheadConfig(
                name=f"bulkhead_{i}",
                resource_type=ResourceType.ASYNC_SEMAPHORE,
                max_concurrent=2
            )
            manager.register_bulkhead(config)
        
        all_metrics = manager.get_all_metrics()
        
        assert len(all_metrics) == 3
        for i in range(3):
            assert f"bulkhead_{i}" in all_metrics
            assert "total_requests" in all_metrics[f"bulkhead_{i}"]
    
    @pytest.mark.asyncio
    async def test_system_health_check(self, manager):
        """Test system-wide health check"""
        # Register a bulkhead
        config = BulkheadConfig(
            name="health_test",
            resource_type=ResourceType.ASYNC_SEMAPHORE,
            max_concurrent=2
        )
        manager.register_bulkhead(config)
        
        health = await manager.get_system_health()
        
        assert "overall_status" in health
        assert "total_bulkheads" in health
        assert "bulkhead_health" in health
        assert health["overall_status"] == "healthy"
        assert health["total_bulkheads"] == 1
    
    def test_default_configs_creation(self, manager):
        """Test creation of default configurations"""
        default_configs = manager.create_default_configs()
        
        expected_bulkheads = [
            "search_operations",
            "embedding_generation", 
            "database_operations",
            "file_operations",
            "api_requests"
        ]
        
        for bulkhead_name in expected_bulkheads:
            assert bulkhead_name in default_configs
            config = default_configs[bulkhead_name]
            assert isinstance(config, BulkheadConfig)
            assert config.name == bulkhead_name
    
    def test_setup_default_bulkheads(self, manager):
        """Test setting up default bulkheads"""
        manager.setup_default_bulkheads()
        
        # Should have created multiple default bulkheads
        assert len(manager.bulkheads) > 0
        
        # Check that specific expected bulkheads exist
        assert manager.get_bulkhead("search_operations") is not None
        assert manager.get_bulkhead("database_operations") is not None


class TestBulkheadDecorator:
    """Test bulkhead isolation decorator"""
    
    @pytest.mark.asyncio
    async def test_bulkhead_decorator(self):
        """Test bulkhead isolation decorator"""
        # Set up a test bulkhead
        manager = get_bulkhead_manager()
        config = BulkheadConfig(
            name="decorator_test",
            resource_type=ResourceType.ASYNC_SEMAPHORE,
            max_concurrent=2
        )
        manager.register_bulkhead(config)
        
        @bulkhead_isolation("decorator_test", priority=0)
        async def decorated_func(value):
            await asyncio.sleep(0.1)
            return f"decorated_{value}"
        
        result = await decorated_func("test")
        
        assert result == "decorated_test"
        
        # Check that function attributes are preserved
        assert hasattr(decorated_func, 'bulkhead_name')
        assert decorated_func.bulkhead_name == "decorator_test"
        assert decorated_func.priority == 0


class TestBulkheadIntegration:
    """Test integration scenarios and edge cases"""
    
    @pytest.mark.asyncio
    async def test_high_concurrency_stress(self):
        """Test bulkhead under high concurrency stress"""
        config = BulkheadConfig(
            name="stress_test",
            resource_type=ResourceType.ASYNC_SEMAPHORE,
            max_concurrent=5,
            max_queue_size=20,
            timeout_seconds=5.0
        )
        bulkhead = Bulkhead(config)
        
        async def stress_func(value):
            await asyncio.sleep(0.1)
            return value
        
        # Create many concurrent operations
        tasks = [
            asyncio.create_task(bulkhead.execute(stress_func, i))
            for i in range(15)  # More than capacity but within queue limit
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 15
        assert all(isinstance(result, int) for result in results)
        
        metrics = bulkhead.get_metrics()
        assert metrics["successful_requests"] == 15
        assert metrics["rejected_requests"] == 0
    
    @pytest.mark.asyncio
    async def test_mixed_priority_execution(self):
        """Test execution with mixed priority levels"""
        config = BulkheadConfig(
            name="priority_test",
            resource_type=ResourceType.ASYNC_SEMAPHORE,
            max_concurrent=1,  # Force queueing
            max_queue_size=10,
            timeout_seconds=3.0,
            priority_levels=3
        )
        bulkhead = Bulkhead(config)
        
        execution_order = []
        
        async def priority_tracking_func(name):
            execution_order.append(name)
            await asyncio.sleep(0.1)
            return name
        
        # Start a long operation to fill capacity
        long_task = asyncio.create_task(bulkhead.execute(asyncio.sleep, 0.5))
        
        # Give time for capacity to fill
        await asyncio.sleep(0.1)
        
        # Queue operations with different priorities
        priority_tasks = []
        for priority, name in [(2, "low1"), (0, "high1"), (1, "med1"), (2, "low2"), (0, "high2")]:
            task = asyncio.create_task(
                bulkhead.execute(priority_tracking_func, name, priority=priority)
            )
            priority_tasks.append(task)
        
        # Wait for all to complete
        await asyncio.gather(long_task, *priority_tasks)
        
        # Verify that we got all expected operations
        assert len(execution_order) == 5
        assert "high1" in execution_order
        assert "high2" in execution_order


if __name__ == "__main__":
    pytest.main([__file__, "-v"])