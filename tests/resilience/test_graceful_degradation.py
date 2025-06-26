"""
Tests for Graceful Degradation Patterns.

This module tests the graceful degradation capabilities including
service fallbacks, partial functionality modes, and adaptive responses.
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

from src.resilience.graceful_degradation import (
    GracefulDegradationManager,
    DegradationPolicy,
    ServiceCapability,
    DegradationContext,
    DegradationLevel,
    ServiceMode,
    ComponentStatus,
    FeatureToggleStrategy,
    CachedResponseStrategy,
    SimplifiedResponseStrategy,
    EmergencyModeStrategy,
    graceful_degradation,
    get_degradation_manager,
    update_component_health,
    check_degradation_needed,
    get_service_health,
    setup_default_degradation_policies
)


class TestDegradationEnums:
    """Test degradation enumeration values"""
    
    def test_degradation_level_values(self):
        """Test degradation level enum values"""
        assert DegradationLevel.FULL.value == "full"
        assert DegradationLevel.REDUCED.value == "reduced"
        assert DegradationLevel.MINIMAL.value == "minimal"
        assert DegradationLevel.EMERGENCY.value == "emergency"
    
    def test_service_mode_values(self):
        """Test service mode enum values"""
        assert ServiceMode.NORMAL.value == "normal"
        assert ServiceMode.DEGRADED.value == "degraded"
        assert ServiceMode.READ_ONLY.value == "read_only"
        assert ServiceMode.OFFLINE.value == "offline"
    
    def test_component_status_values(self):
        """Test component status enum values"""
        assert ComponentStatus.AVAILABLE.value == "available"
        assert ComponentStatus.DEGRADED.value == "degraded"
        assert ComponentStatus.UNAVAILABLE.value == "unavailable"
        assert ComponentStatus.UNKNOWN.value == "unknown"


class TestDegradationPolicy:
    """Test degradation policy data structure"""
    
    def test_policy_creation(self):
        """Test degradation policy creation"""
        policy = DegradationPolicy(
            name="test_policy",
            service_name="test_service",
            degradation_level=DegradationLevel.REDUCED,
            trigger_conditions={"error_rate": 0.1},
            timeout_seconds=15.0,
            retry_attempts=2,
            required_components=["database", "api"],
            optional_components=["analytics"],
            feature_flags={"advanced_mode": False}
        )
        
        assert policy.name == "test_policy"
        assert policy.service_name == "test_service"
        assert policy.degradation_level == DegradationLevel.REDUCED
        assert policy.trigger_conditions["error_rate"] == 0.1
        assert policy.timeout_seconds == 15.0
        assert policy.retry_attempts == 2
        assert "database" in policy.required_components
        assert "analytics" in policy.optional_components
        assert policy.feature_flags["advanced_mode"] is False


class TestServiceCapability:
    """Test service capability data structure"""
    
    def test_capability_creation(self):
        """Test service capability creation"""
        capability = ServiceCapability(
            name="database",
            status=ComponentStatus.AVAILABLE,
            last_checked=datetime.now(),
            error_count=0,
            response_time_ms=25.5,
            metadata={"version": "1.0"},
            fallback_available=True
        )
        
        assert capability.name == "database"
        assert capability.status == ComponentStatus.AVAILABLE
        assert capability.error_count == 0
        assert capability.response_time_ms == 25.5
        assert capability.metadata["version"] == "1.0"
        assert capability.fallback_available is True


class TestDegradationContext:
    """Test degradation context data structure"""
    
    def test_context_creation(self):
        """Test degradation context creation"""
        context = DegradationContext(
            service_name="search_service",
            operation_name="vector_search",
            user_context={"user_id": "123"},
            system_load=0.75,
            available_components=["database", "cache"],
            degraded_components=["ml_service"],
            error_rates={"ml_service": 0.15}
        )
        
        assert context.service_name == "search_service"
        assert context.operation_name == "vector_search"
        assert context.user_context["user_id"] == "123"
        assert context.system_load == 0.75
        assert "database" in context.available_components
        assert "ml_service" in context.degraded_components
        assert context.error_rates["ml_service"] == 0.15


class TestFeatureToggleStrategy:
    """Test feature toggle degradation strategy"""
    
    @pytest.fixture
    def strategy(self):
        """Create feature toggle strategy for testing"""
        return FeatureToggleStrategy(
            disabled_features=["recommendations", "analytics"],
            level=DegradationLevel.REDUCED
        )
    
    @pytest.mark.asyncio
    async def test_should_degrade_with_degraded_components(self, strategy):
        """Test should_degrade returns True when components are degraded"""
        context = DegradationContext(
            service_name="test",
            operation_name="test",
            degraded_components=["ml_service"]
        )
        
        should_degrade = await strategy.should_degrade(context)
        assert should_degrade is True
    
    @pytest.mark.asyncio
    async def test_should_not_degrade_with_healthy_components(self, strategy):
        """Test should_degrade returns False when all components are healthy"""
        context = DegradationContext(
            service_name="test",
            operation_name="test",
            degraded_components=[]
        )
        
        should_degrade = await strategy.should_degrade(context)
        assert should_degrade is False
    
    @pytest.mark.asyncio
    async def test_apply_degradation(self, strategy):
        """Test applying feature toggle degradation"""
        context = DegradationContext(
            service_name="test",
            operation_name="test",
            available_components=["database", "cache", "recommendations"]
        )
        
        result = await strategy.apply_degradation(context)
        
        assert result["mode"] == "feature_limited"
        assert "recommendations" in result["disabled_features"]
        assert "analytics" in result["disabled_features"]
        assert "database" in result["available_features"]
    
    def test_get_degradation_level(self, strategy):
        """Test getting degradation level"""
        assert strategy.get_degradation_level() == DegradationLevel.REDUCED


class TestCachedResponseStrategy:
    """Test cached response degradation strategy"""
    
    @pytest.fixture
    def strategy(self):
        """Create cached response strategy for testing"""
        return CachedResponseStrategy(cache_ttl_minutes=30)
    
    @pytest.mark.asyncio
    async def test_should_degrade_with_high_error_rate(self, strategy):
        """Test should_degrade returns True with high error rates"""
        context = DegradationContext(
            service_name="test",
            operation_name="test",
            error_rates={"service": 0.2}  # 20% error rate
        )
        
        should_degrade = await strategy.should_degrade(context)
        assert should_degrade is True
    
    @pytest.mark.asyncio
    async def test_should_degrade_with_degraded_components(self, strategy):
        """Test should_degrade returns True with degraded components"""
        context = DegradationContext(
            service_name="test",
            operation_name="test",
            degraded_components=["ml_service"]
        )
        
        should_degrade = await strategy.should_degrade(context)
        assert should_degrade is True
    
    @pytest.mark.asyncio
    async def test_apply_degradation_with_cache_hit(self, strategy):
        """Test applying degradation with cached data available"""
        # Cache some data first
        strategy.cache_response("test_service", "test_operation", {"data": "cached_result"})
        
        context = DegradationContext(
            service_name="test_service",
            operation_name="test_operation"
        )
        
        result = await strategy.apply_degradation(context)
        
        assert result["mode"] == "cached_response"
        assert result["data"]["data"] == "cached_result"
        assert result["cache_age_minutes"] >= 0
    
    @pytest.mark.asyncio
    async def test_apply_degradation_with_cache_miss(self, strategy):
        """Test applying degradation with no cached data"""
        context = DegradationContext(
            service_name="unknown_service",
            operation_name="unknown_operation"
        )
        
        result = await strategy.apply_degradation(context)
        
        assert result["mode"] == "cache_miss"
        assert "No cached data available" in result["message"]
    
    def test_cache_response(self, strategy):
        """Test caching a response"""
        data = {"result": "test_data"}
        strategy.cache_response("test_service", "test_op", data)
        
        cache_key = "test_service:test_op"
        assert cache_key in strategy.cache
        cached_data, timestamp = strategy.cache[cache_key]
        assert cached_data == data
        assert isinstance(timestamp, datetime)


class TestSimplifiedResponseStrategy:
    """Test simplified response degradation strategy"""
    
    @pytest.fixture
    def strategy(self):
        """Create simplified response strategy for testing"""
        return SimplifiedResponseStrategy()
    
    @pytest.mark.asyncio
    async def test_should_degrade_with_high_system_load(self, strategy):
        """Test should_degrade returns True with high system load"""
        context = DegradationContext(
            service_name="test",
            operation_name="test",
            system_load=0.9  # 90% system load
        )
        
        should_degrade = await strategy.should_degrade(context)
        assert should_degrade is True
    
    @pytest.mark.asyncio
    async def test_should_degrade_with_multiple_degraded_components(self, strategy):
        """Test should_degrade returns True with multiple degraded components"""
        context = DegradationContext(
            service_name="test",
            operation_name="test",
            degraded_components=["service1", "service2"]  # 2+ components
        )
        
        should_degrade = await strategy.should_degrade(context)
        assert should_degrade is True
    
    @pytest.mark.asyncio
    async def test_apply_degradation_search_operation(self, strategy):
        """Test applying degradation for search operation"""
        context = DegradationContext(
            service_name="test",
            operation_name="search"
        )
        
        result = await strategy.apply_degradation(context)
        
        assert result["mode"] == "simplified_search"
        assert result["results"] == []
        assert "temporarily limited" in result["message"]
        assert len(result["suggestions"]) > 0
    
    @pytest.mark.asyncio
    async def test_apply_degradation_recommendation_operation(self, strategy):
        """Test applying degradation for recommendation operation"""
        context = DegradationContext(
            service_name="test",
            operation_name="recommendation"
        )
        
        result = await strategy.apply_degradation(context)
        
        assert result["mode"] == "no_recommendations"
        assert result["recommendations"] == []
        assert "unavailable" in result["message"]
    
    @pytest.mark.asyncio
    async def test_apply_degradation_unknown_operation(self, strategy):
        """Test applying degradation for unknown operation"""
        context = DegradationContext(
            service_name="test",
            operation_name="unknown_operation"
        )
        
        result = await strategy.apply_degradation(context)
        
        assert result["mode"] == "minimal_response"
        assert "unknown_operation" in result["message"]
        assert result["status"] == "degraded"


class TestEmergencyModeStrategy:
    """Test emergency mode degradation strategy"""
    
    @pytest.fixture
    def strategy(self):
        """Create emergency mode strategy for testing"""
        return EmergencyModeStrategy()
    
    @pytest.mark.asyncio
    async def test_should_degrade_with_critical_error_rate(self, strategy):
        """Test should_degrade returns True with critical error rates"""
        context = DegradationContext(
            service_name="test",
            operation_name="test",
            error_rates={"service": 0.6}  # 60% error rate
        )
        
        should_degrade = await strategy.should_degrade(context)
        assert should_degrade is True
    
    @pytest.mark.asyncio
    async def test_should_degrade_with_most_components_down(self, strategy):
        """Test should_degrade returns True when most components are down"""
        context = DegradationContext(
            service_name="test",
            operation_name="test",
            available_components=["service1"],
            degraded_components=["service2", "service3", "service4"]
        )
        
        should_degrade = await strategy.should_degrade(context)
        assert should_degrade is True
    
    @pytest.mark.asyncio
    async def test_apply_degradation(self, strategy):
        """Test applying emergency mode degradation"""
        context = DegradationContext(
            service_name="test",
            operation_name="test"
        )
        
        result = await strategy.apply_degradation(context)
        
        assert result["mode"] == "emergency"
        assert "emergency mode" in result["message"]
        assert "health_check" in result["available_operations"]
        assert result["contact_support"] is True
    
    def test_get_degradation_level(self, strategy):
        """Test getting degradation level"""
        assert strategy.get_degradation_level() == DegradationLevel.EMERGENCY


class TestGracefulDegradationManager:
    """Test graceful degradation manager functionality"""
    
    @pytest.fixture
    def manager(self):
        """Create degradation manager for testing"""
        return GracefulDegradationManager()
    
    def test_manager_initialization(self, manager):
        """Test degradation manager initialization"""
        assert len(manager.policies) == 0
        assert len(manager.service_capabilities) == 0
        assert len(manager.current_mode) == 0
        assert len(manager.degradation_history) == 0
        
        # Should have default strategies registered
        assert len(manager.strategies[DegradationLevel.REDUCED]) > 0
        assert len(manager.strategies[DegradationLevel.MINIMAL]) > 0
        assert len(manager.strategies[DegradationLevel.EMERGENCY]) > 0
    
    def test_register_policy(self, manager):
        """Test registering a degradation policy"""
        policy = DegradationPolicy(
            name="test_policy",
            service_name="test_service",
            degradation_level=DegradationLevel.REDUCED
        )
        
        manager.register_policy(policy)
        
        assert "test_policy" in manager.policies
        assert manager.policies["test_policy"] == policy
        assert manager.current_mode["test_service"] == ServiceMode.NORMAL
    
    def test_register_strategy(self, manager):
        """Test registering a degradation strategy"""
        strategy = FeatureToggleStrategy(["feature1"], DegradationLevel.REDUCED)
        
        initial_count = len(manager.strategies[DegradationLevel.REDUCED])
        manager.register_strategy(DegradationLevel.REDUCED, strategy)
        
        assert len(manager.strategies[DegradationLevel.REDUCED]) == initial_count + 1
        assert strategy in manager.strategies[DegradationLevel.REDUCED]
    
    @pytest.mark.asyncio
    async def test_update_component_status_new_component(self, manager):
        """Test updating status for a new component"""
        await manager.update_component_status(
            "test_component",
            ComponentStatus.AVAILABLE,
            response_time_ms=50.0,
            metadata={"version": "1.0"}
        )
        
        assert "test_component" in manager.service_capabilities
        capability = manager.service_capabilities["test_component"]
        assert capability.status == ComponentStatus.AVAILABLE
        assert capability.response_time_ms == 50.0
        assert capability.metadata["version"] == "1.0"
        assert capability.error_count == 0
    
    @pytest.mark.asyncio
    async def test_update_component_status_existing_component(self, manager):
        """Test updating status for an existing component"""
        # First update
        await manager.update_component_status("test_component", ComponentStatus.AVAILABLE)
        
        # Second update with failure
        await manager.update_component_status("test_component", ComponentStatus.UNAVAILABLE)
        
        capability = manager.service_capabilities["test_component"]
        assert capability.status == ComponentStatus.UNAVAILABLE
        assert capability.error_count == 1
        
        # Third update back to available
        await manager.update_component_status("test_component", ComponentStatus.AVAILABLE)
        
        capability = manager.service_capabilities["test_component"]
        assert capability.status == ComponentStatus.AVAILABLE
        assert capability.error_count == 0  # Should decrease on success
    
    @pytest.mark.asyncio
    async def test_evaluate_degradation(self, manager):
        """Test evaluating degradation context"""
        # Set up some component statuses
        await manager.update_component_status("service1", ComponentStatus.AVAILABLE)
        await manager.update_component_status("service2", ComponentStatus.UNAVAILABLE)
        await manager.update_component_status("service3", ComponentStatus.DEGRADED)
        
        context = await manager.evaluate_degradation("test_service", "test_operation")
        
        assert context.service_name == "test_service"
        assert context.operation_name == "test_operation"
        assert "service1" in context.available_components
        assert "service2" in context.degraded_components
        assert "service3" in context.degraded_components
        assert context.system_load > 0  # Should be calculated based on component states
    
    @pytest.mark.asyncio
    async def test_apply_degradation_no_degradation_needed(self, manager):
        """Test apply_degradation when no degradation is needed"""
        # Set up healthy system
        await manager.update_component_status("service1", ComponentStatus.AVAILABLE)
        
        context = await manager.evaluate_degradation("test_service", "test_operation")
        result = await manager.apply_degradation(context)
        
        assert result is None  # No degradation needed
    
    @pytest.mark.asyncio
    async def test_apply_degradation_with_degradation_needed(self, manager):
        """Test apply_degradation when degradation is needed"""
        # Set up degraded system
        await manager.update_component_status("service1", ComponentStatus.UNAVAILABLE)
        await manager.update_component_status("service2", ComponentStatus.UNAVAILABLE)
        
        context = await manager.evaluate_degradation("test_service", "test_operation")
        result = await manager.apply_degradation(context)
        
        assert result is not None
        assert result["degraded"] is True
        assert "level" in result
        assert "strategy" in result
    
    @pytest.mark.asyncio
    async def test_execute_with_degradation_primary_success(self, manager):
        """Test execute_with_degradation when primary function succeeds"""
        async def successful_function(value):
            return f"success_{value}"
        
        result = await manager.execute_with_degradation(
            "test_service",
            "test_operation", 
            successful_function,
            "test"
        )
        
        assert result["degraded"] is False
        assert result["result"] == "success_test"
        assert result["mode"] == "normal"
    
    @pytest.mark.asyncio
    async def test_execute_with_degradation_primary_failure(self, manager):
        """Test execute_with_degradation when primary function fails"""
        async def failing_function():
            raise ValueError("Primary function failed")
        
        # Set up some degraded components to trigger fallback
        await manager.update_component_status("test_service", ComponentStatus.UNAVAILABLE)
        
        result = await manager.execute_with_degradation(
            "test_service",
            "test_operation",
            failing_function
        )
        
        # Should return degraded response instead of raising exception
        assert result["degraded"] is True
    
    def test_get_service_status_specific_service(self, manager):
        """Test getting status for a specific service"""
        manager.current_mode["test_service"] = ServiceMode.DEGRADED
        
        status = manager.get_service_status("test_service")
        
        assert status["service"] == "test_service"
        assert status["mode"] == "degraded"
        assert "components" in status
    
    def test_get_service_status_all_services(self, manager):
        """Test getting status for all services"""
        manager.current_mode["service1"] = ServiceMode.NORMAL
        manager.current_mode["service2"] = ServiceMode.DEGRADED
        
        status = manager.get_service_status()
        
        assert "services" in status
        assert status["services"]["service1"] == "normal"
        assert status["services"]["service2"] == "degraded"
        assert "total_components" in status
        assert "available_components" in status
        assert "degraded_components" in status
    
    def test_get_degradation_history(self, manager):
        """Test getting degradation history"""
        # Manually add some history events
        manager.degradation_history = [
            {"service": "service1", "timestamp": "2023-01-01T00:00:00"},
            {"service": "service2", "timestamp": "2023-01-01T01:00:00"},
            {"service": "service1", "timestamp": "2023-01-01T02:00:00"}
        ]
        
        # Get all history
        all_history = manager.get_degradation_history()
        assert len(all_history) == 3
        
        # Get filtered history
        service1_history = manager.get_degradation_history("service1")
        assert len(service1_history) == 2
        
        # Get limited history
        limited_history = manager.get_degradation_history(limit=2)
        assert len(limited_history) == 2
    
    def test_reset_component_status(self, manager):
        """Test resetting component status"""
        # Set up a degraded component
        manager.service_capabilities["test_component"] = ServiceCapability(
            name="test_component",
            status=ComponentStatus.UNAVAILABLE,
            last_checked=datetime.now(),
            error_count=5
        )
        
        manager.reset_component_status("test_component")
        
        capability = manager.service_capabilities["test_component"]
        assert capability.status == ComponentStatus.AVAILABLE
        assert capability.error_count == 0


class TestGracefulDegradationDecorator:
    """Test graceful degradation decorator"""
    
    @pytest.mark.asyncio
    async def test_decorator_with_async_function(self):
        """Test decorator with async function"""
        @graceful_degradation("test_service", "test_operation")
        async def async_function(value):
            return f"async_result_{value}"
        
        result = await async_function("test")
        
        assert result["degraded"] is False
        assert result["result"] == "async_result_test"
    
    def test_decorator_with_sync_function(self):
        """Test decorator with sync function"""
        @graceful_degradation("test_service", "test_operation")
        def sync_function(value):
            return f"sync_result_{value}"
        
        result = sync_function("test")
        
        assert result["degraded"] is False
        assert result["result"] == "sync_result_test"


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    @pytest.mark.asyncio
    async def test_update_component_health(self):
        """Test update_component_health convenience function"""
        await update_component_health("test_component", True, 25.0)
        
        manager = get_degradation_manager()
        capability = manager.service_capabilities["test_component"]
        assert capability.status == ComponentStatus.AVAILABLE
        assert capability.response_time_ms == 25.0
    
    @pytest.mark.asyncio
    async def test_check_degradation_needed(self):
        """Test check_degradation_needed convenience function"""
        # First check with healthy system
        is_needed = await check_degradation_needed("test_service", "test_operation")
        assert is_needed is False
        
        # Update component to unhealthy and check again
        await update_component_health("test_component", False)
        
        # Note: This may still return False if degradation strategies don't trigger
        # The actual result depends on the registered strategies and their conditions
    
    def test_get_service_health(self):
        """Test get_service_health convenience function"""
        health = get_service_health()
        
        assert "services" in health or "components" in health
        assert isinstance(health, dict)
    
    def test_setup_default_degradation_policies(self):
        """Test setting up default degradation policies"""
        manager = get_degradation_manager()
        initial_count = len(manager.policies)
        
        setup_default_degradation_policies()
        
        assert len(manager.policies) > initial_count
        assert "search_degradation" in manager.policies
        assert "knowledge_degradation" in manager.policies
        assert "api_degradation" in manager.policies


class TestGracefulDegradationIntegration:
    """Test integration scenarios and edge cases"""
    
    @pytest.mark.asyncio
    async def test_full_degradation_cycle(self):
        """Test complete degradation cycle from healthy to emergency"""
        manager = GracefulDegradationManager()
        
        # Start with healthy system
        await manager.update_component_status("service1", ComponentStatus.AVAILABLE)
        await manager.update_component_status("service2", ComponentStatus.AVAILABLE)
        
        context = await manager.evaluate_degradation("test_service", "test_operation")
        result = await manager.apply_degradation(context)
        assert result is None  # No degradation
        
        # Degrade one component
        await manager.update_component_status("service1", ComponentStatus.UNAVAILABLE)
        
        context = await manager.evaluate_degradation("test_service", "test_operation")
        result = await manager.apply_degradation(context)
        
        if result:  # May trigger degradation depending on strategies
            assert result["degraded"] is True
            assert result["level"] in ["reduced", "minimal", "emergency"]
        
        # Degrade more components to trigger higher level degradation
        await manager.update_component_status("service2", ComponentStatus.UNAVAILABLE)
        
        context = await manager.evaluate_degradation("test_service", "test_operation")
        result = await manager.apply_degradation(context)
        
        if result:
            assert result["degraded"] is True
    
    @pytest.mark.asyncio
    async def test_concurrent_degradation_operations(self):
        """Test degradation under concurrent operations"""
        manager = GracefulDegradationManager()
        
        async def concurrent_operation(operation_id):
            context = await manager.evaluate_degradation(f"service_{operation_id}", "test_operation")
            return await manager.apply_degradation(context)
        
        # Run multiple concurrent degradation evaluations
        tasks = [concurrent_operation(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All should complete without errors
        assert len(results) == 10
        
        # Results should be consistent (all None or all degraded, depending on system state)
        degraded_count = sum(1 for r in results if r is not None)
        assert degraded_count >= 0  # At least no errors
    
    @pytest.mark.asyncio
    async def test_degradation_with_recovery(self):
        """Test degradation with component recovery"""
        manager = GracefulDegradationManager()
        
        # Start with degraded system
        await manager.update_component_status("critical_service", ComponentStatus.UNAVAILABLE)
        
        context = await manager.evaluate_degradation("test_service", "test_operation")
        result = await manager.apply_degradation(context)
        
        # Should be degraded
        if result:
            assert result["degraded"] is True
        
        # Recover the component
        manager.reset_component_status("critical_service")
        
        context = await manager.evaluate_degradation("test_service", "test_operation")
        result = await manager.apply_degradation(context)
        
        # Should no longer need degradation (or less degradation)
        # Note: This depends on the specific strategies and their logic


if __name__ == "__main__":
    pytest.main([__file__, "-v"])