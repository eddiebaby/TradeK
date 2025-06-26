"""
Graceful Degradation Patterns for TradeKnowledge.

This module provides comprehensive graceful degradation capabilities
including service fallbacks, partial functionality modes, and adaptive responses.
"""

import asyncio
import functools
import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class DegradationLevel(Enum):
    """Levels of service degradation"""

    FULL = "full"  # All features available
    REDUCED = "reduced"  # Some features disabled
    MINIMAL = "minimal"  # Basic functionality only
    EMERGENCY = "emergency"  # Critical functions only


class ServiceMode(Enum):
    """Service operational modes"""

    NORMAL = "normal"
    DEGRADED = "degraded"
    READ_ONLY = "read_only"
    OFFLINE = "offline"


class ComponentStatus(Enum):
    """Component availability status"""

    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass
class DegradationPolicy:
    """Policy defining degradation behavior"""

    name: str
    service_name: str
    degradation_level: DegradationLevel
    trigger_conditions: dict[str, Any] = field(default_factory=dict)
    fallback_function: Callable | None = None
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    required_components: list[str] = field(default_factory=list)
    optional_components: list[str] = field(default_factory=list)
    emergency_response: str | None = None
    feature_flags: dict[str, bool] = field(default_factory=dict)


@dataclass
class ServiceCapability:
    """Represents a service capability and its status"""

    name: str
    status: ComponentStatus
    last_checked: datetime
    error_count: int = 0
    response_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    fallback_available: bool = False


@dataclass
class DegradationContext:
    """Context for degradation decisions"""

    service_name: str
    operation_name: str
    user_context: dict[str, Any] = field(default_factory=dict)
    system_load: float = 0.0
    available_components: list[str] = field(default_factory=list)
    degraded_components: list[str] = field(default_factory=list)
    error_rates: dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class DegradationStrategy(ABC):
    """Abstract base class for degradation strategies"""

    @abstractmethod
    async def should_degrade(self, context: DegradationContext) -> bool:
        """Determine if degradation should be applied"""
        pass

    @abstractmethod
    async def apply_degradation(self, context: DegradationContext) -> Any:
        """Apply degradation strategy"""
        pass

    @abstractmethod
    def get_degradation_level(self) -> DegradationLevel:
        """Get the degradation level this strategy applies"""
        pass


class FeatureToggleStrategy(DegradationStrategy):
    """Strategy that disables optional features"""

    def __init__(
        self,
        disabled_features: list[str],
        level: DegradationLevel = DegradationLevel.REDUCED,
    ):
        self.disabled_features = disabled_features
        self.level = level

    async def should_degrade(self, context: DegradationContext) -> bool:
        # Degrade if any required components are unavailable
        unavailable = set(context.degraded_components)
        return bool(unavailable)

    async def apply_degradation(self, context: DegradationContext) -> dict[str, Any]:
        return {
            "mode": "feature_limited",
            "disabled_features": self.disabled_features,
            "message": "Some features disabled due to system issues",
            "available_features": [
                f
                for f in context.available_components
                if f not in self.disabled_features
            ],
        }

    def get_degradation_level(self) -> DegradationLevel:
        return self.level


class CachedResponseStrategy(DegradationStrategy):
    """Strategy that serves cached/stale data"""

    def __init__(
        self,
        cache_ttl_minutes: int = 60,
        level: DegradationLevel = DegradationLevel.REDUCED,
    ):
        self.cache_ttl_minutes = cache_ttl_minutes
        self.level = level
        self.cache: dict[str, tuple[Any, datetime]] = {}

    async def should_degrade(self, context: DegradationContext) -> bool:
        # Use cache if error rates are high or components unavailable
        high_error_rate = any(rate > 0.1 for rate in context.error_rates.values())
        return high_error_rate or bool(context.degraded_components)

    async def apply_degradation(self, context: DegradationContext) -> dict[str, Any]:
        cache_key = f"{context.service_name}:{context.operation_name}"

        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            age_minutes = (datetime.now() - timestamp).total_seconds() / 60

            if age_minutes <= self.cache_ttl_minutes:
                return {
                    "mode": "cached_response",
                    "data": data,
                    "cache_age_minutes": age_minutes,
                    "message": "Serving cached data due to service issues",
                }

        return {
            "mode": "cache_miss",
            "message": "No cached data available",
            "suggested_action": "retry_later",
        }

    def cache_response(self, service_name: str, operation_name: str, data: Any):
        """Cache a response for future use"""
        cache_key = f"{service_name}:{operation_name}"
        self.cache[cache_key] = (data, datetime.now())

    def get_degradation_level(self) -> DegradationLevel:
        return self.level


class SimplifiedResponseStrategy(DegradationStrategy):
    """Strategy that returns simplified/reduced functionality responses"""

    def __init__(self, level: DegradationLevel = DegradationLevel.MINIMAL):
        self.level = level

    async def should_degrade(self, context: DegradationContext) -> bool:
        # Degrade if system load is high or multiple components are down
        return context.system_load > 0.8 or len(context.degraded_components) >= 2

    async def apply_degradation(self, context: DegradationContext) -> dict[str, Any]:
        if context.operation_name == "search":
            return {
                "mode": "simplified_search",
                "results": [],
                "message": "Search temporarily limited. Try basic keywords.",
                "suggestions": [
                    "Use simpler search terms",
                    "Try again in a few minutes",
                ],
            }
        elif context.operation_name == "recommendation":
            return {
                "mode": "no_recommendations",
                "recommendations": [],
                "message": "Recommendations temporarily unavailable",
                "fallback": "Browse recent content",
            }
        elif context.operation_name == "analysis":
            return {
                "mode": "basic_analysis",
                "analysis": {"summary": "Analysis temporarily limited"},
                "message": "Detailed analysis unavailable. Basic summary provided.",
            }
        else:
            return {
                "mode": "minimal_response",
                "message": f"{context.operation_name} temporarily limited",
                "status": "degraded",
            }

    def get_degradation_level(self) -> DegradationLevel:
        return self.level


class EmergencyModeStrategy(DegradationStrategy):
    """Strategy for emergency situations with minimal functionality"""

    def __init__(self):
        self.level = DegradationLevel.EMERGENCY

    async def should_degrade(self, context: DegradationContext) -> bool:
        # Emergency mode if most components are down or error rates are critical
        critical_error_rate = any(rate > 0.5 for rate in context.error_rates.values())
        most_components_down = len(context.degraded_components) > len(
            context.available_components
        )
        return critical_error_rate or most_components_down

    async def apply_degradation(self, context: DegradationContext) -> dict[str, Any]:
        return {
            "mode": "emergency",
            "message": "System in emergency mode. Only critical functions available.",
            "available_operations": ["health_check", "status"],
            "estimated_recovery": "15-30 minutes",
            "contact_support": True,
        }

    def get_degradation_level(self) -> DegradationLevel:
        return self.level


class GracefulDegradationManager:
    """
    Main manager for graceful degradation patterns.
    Coordinates strategies and policies to maintain service availability.
    """

    def __init__(self):
        self.policies: dict[str, DegradationPolicy] = {}
        self.strategies: dict[DegradationLevel, list[DegradationStrategy]] = {
            level: [] for level in DegradationLevel
        }
        self.service_capabilities: dict[str, ServiceCapability] = {}
        self.current_mode: dict[str, ServiceMode] = {}
        self.degradation_history: list[dict[str, Any]] = []
        self.max_history = 1000
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._lock = threading.Lock()

        # Register default strategies
        self._register_default_strategies()

    def _register_default_strategies(self):
        """Register default degradation strategies"""
        # Feature toggle strategies
        search_features = FeatureToggleStrategy(
            disabled_features=["advanced_filters", "recommendations", "analytics"],
            level=DegradationLevel.REDUCED,
        )
        self.register_strategy(DegradationLevel.REDUCED, search_features)

        # Cached response strategy
        cache_strategy = CachedResponseStrategy(cache_ttl_minutes=30)
        self.register_strategy(DegradationLevel.REDUCED, cache_strategy)

        # Simplified response strategy
        simple_strategy = SimplifiedResponseStrategy()
        self.register_strategy(DegradationLevel.MINIMAL, simple_strategy)

        # Emergency mode strategy
        emergency_strategy = EmergencyModeStrategy()
        self.register_strategy(DegradationLevel.EMERGENCY, emergency_strategy)

    def register_policy(self, policy: DegradationPolicy):
        """Register a degradation policy"""
        with self._lock:
            self.policies[policy.name] = policy
            self.current_mode[policy.service_name] = ServiceMode.NORMAL
        logger.info(f"Registered degradation policy: {policy.name}")

    def register_strategy(self, level: DegradationLevel, strategy: DegradationStrategy):
        """Register a degradation strategy for a specific level"""
        with self._lock:
            self.strategies[level].append(strategy)
        logger.info(f"Registered degradation strategy for level: {level.value}")

    async def update_component_status(
        self,
        component_name: str,
        status: ComponentStatus,
        response_time_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ):
        """Update the status of a service component"""
        with self._lock:
            if component_name in self.service_capabilities:
                capability = self.service_capabilities[component_name]
                if status == ComponentStatus.UNAVAILABLE:
                    capability.error_count += 1
                else:
                    capability.error_count = max(0, capability.error_count - 1)
            else:
                capability = ServiceCapability(
                    name=component_name,
                    status=status,
                    last_checked=datetime.now(),
                    error_count=1 if status == ComponentStatus.UNAVAILABLE else 0,
                )

            capability.status = status
            capability.last_checked = datetime.now()
            capability.response_time_ms = response_time_ms
            capability.metadata = metadata or {}

            self.service_capabilities[component_name] = capability

        logger.debug(f"Updated component {component_name} status to {status.value}")

    async def evaluate_degradation(
        self,
        service_name: str,
        operation_name: str,
        user_context: dict[str, Any] | None = None,
    ) -> DegradationContext:
        """Evaluate current system state and determine degradation needs"""

        # Gather system information
        available_components = []
        degraded_components = []
        error_rates = {}

        for name, capability in self.service_capabilities.items():
            if capability.status == ComponentStatus.AVAILABLE:
                available_components.append(name)
            elif capability.status in [
                ComponentStatus.DEGRADED,
                ComponentStatus.UNAVAILABLE,
            ]:
                degraded_components.append(name)

            # Calculate error rate (simplified)
            total_checks = max(
                1, capability.error_count + 10
            )  # Assume some successful checks
            error_rates[name] = capability.error_count / total_checks

        # Estimate system load (simplified)
        system_load = len(degraded_components) / max(1, len(self.service_capabilities))

        context = DegradationContext(
            service_name=service_name,
            operation_name=operation_name,
            user_context=user_context or {},
            system_load=system_load,
            available_components=available_components,
            degraded_components=degraded_components,
            error_rates=error_rates,
        )

        return context

    async def apply_degradation(
        self, context: DegradationContext
    ) -> dict[str, Any] | None:
        """Apply appropriate degradation strategy based on context"""

        # Try strategies in order of severity (least to most degraded)
        for level in [
            DegradationLevel.REDUCED,
            DegradationLevel.MINIMAL,
            DegradationLevel.EMERGENCY,
        ]:
            strategies = self.strategies.get(level, [])

            for strategy in strategies:
                try:
                    if await strategy.should_degrade(context):
                        result = await strategy.apply_degradation(context)

                        # Record degradation event
                        self._record_degradation_event(context, level, result)

                        # Update service mode
                        self._update_service_mode(context.service_name, level)

                        return {
                            "degraded": True,
                            "level": level.value,
                            "strategy": strategy.__class__.__name__,
                            "context": context.service_name,
                            "result": result,
                        }

                except Exception as e:
                    logger.error(
                        f"Error applying degradation strategy {strategy.__class__.__name__}: {e}"
                    )
                    continue

        # No degradation needed
        self._update_service_mode(context.service_name, None)
        return None

    def _record_degradation_event(
        self,
        context: DegradationContext,
        level: DegradationLevel,
        result: dict[str, Any],
    ):
        """Record a degradation event for analysis"""
        event = {
            "timestamp": context.timestamp.isoformat(),
            "service": context.service_name,
            "operation": context.operation_name,
            "level": level.value,
            "system_load": context.system_load,
            "available_components": len(context.available_components),
            "degraded_components": len(context.degraded_components),
            "result_mode": result.get("mode", "unknown"),
        }

        with self._lock:
            self.degradation_history.append(event)
            if len(self.degradation_history) > self.max_history:
                self.degradation_history = self.degradation_history[-self.max_history :]

    def _update_service_mode(
        self, service_name: str, degradation_level: DegradationLevel | None
    ):
        """Update service operational mode"""
        with self._lock:
            if degradation_level is None:
                self.current_mode[service_name] = ServiceMode.NORMAL
            elif degradation_level == DegradationLevel.EMERGENCY:
                self.current_mode[service_name] = ServiceMode.OFFLINE
            elif degradation_level in [
                DegradationLevel.MINIMAL,
                DegradationLevel.REDUCED,
            ]:
                self.current_mode[service_name] = ServiceMode.DEGRADED

    async def execute_with_degradation(
        self,
        service_name: str,
        operation_name: str,
        primary_function: Callable,
        *args,
        user_context: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Execute an operation with graceful degradation"""

        # Evaluate current state
        context = await self.evaluate_degradation(
            service_name, operation_name, user_context
        )

        # Check if degradation is needed
        degradation_result = await self.apply_degradation(context)

        if degradation_result:
            # Return degraded response
            return degradation_result

        # Try primary function
        try:
            if asyncio.iscoroutinefunction(primary_function):
                result = await primary_function(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self._executor, functools.partial(primary_function, *args, **kwargs)
                )

            return {"degraded": False, "result": result, "mode": "normal"}

        except Exception as e:
            logger.warning(
                f"Primary function failed for {service_name}.{operation_name}: {e}"
            )

            # Update component status to reflect failure
            await self.update_component_status(
                service_name, ComponentStatus.UNAVAILABLE
            )

            # Re-evaluate with updated status
            updated_context = await self.evaluate_degradation(
                service_name, operation_name, user_context
            )
            degradation_result = await self.apply_degradation(updated_context)

            if degradation_result:
                return degradation_result

            # If no degradation strategy applies, re-raise the error
            raise e

    def get_service_status(self, service_name: str | None = None) -> dict[str, Any]:
        """Get current service status and degradation information"""
        with self._lock:
            if service_name:
                return {
                    "service": service_name,
                    "mode": self.current_mode.get(
                        service_name, ServiceMode.NORMAL
                    ).value,
                    "components": {
                        name: {
                            "status": cap.status.value,
                            "error_count": cap.error_count,
                            "response_time_ms": cap.response_time_ms,
                            "last_checked": cap.last_checked.isoformat(),
                        }
                        for name, cap in self.service_capabilities.items()
                        if name.startswith(service_name)
                    },
                }
            else:
                return {
                    "services": {
                        name: mode.value for name, mode in self.current_mode.items()
                    },
                    "components": {
                        name: {
                            "status": cap.status.value,
                            "error_count": cap.error_count,
                            "response_time_ms": cap.response_time_ms,
                        }
                        for name, cap in self.service_capabilities.items()
                    },
                    "total_components": len(self.service_capabilities),
                    "available_components": len(
                        [
                            cap
                            for cap in self.service_capabilities.values()
                            if cap.status == ComponentStatus.AVAILABLE
                        ]
                    ),
                    "degraded_components": len(
                        [
                            cap
                            for cap in self.service_capabilities.values()
                            if cap.status != ComponentStatus.AVAILABLE
                        ]
                    ),
                }

    def get_degradation_history(
        self, service_name: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get degradation history"""
        with self._lock:
            history = self.degradation_history

            if service_name:
                history = [
                    event for event in history if event["service"] == service_name
                ]

            return history[-limit:] if limit else history

    def reset_component_status(self, component_name: str):
        """Reset component status to available"""
        with self._lock:
            if component_name in self.service_capabilities:
                self.service_capabilities[component_name].status = (
                    ComponentStatus.AVAILABLE
                )
                self.service_capabilities[component_name].error_count = 0
                self.service_capabilities[component_name].last_checked = datetime.now()


# Global degradation manager instance
_global_degradation_manager: GracefulDegradationManager | None = None


def get_degradation_manager() -> GracefulDegradationManager:
    """Get or create global degradation manager"""
    global _global_degradation_manager
    if _global_degradation_manager is None:
        _global_degradation_manager = GracefulDegradationManager()
    return _global_degradation_manager


def graceful_degradation(service_name: str, operation_name: str):
    """Decorator for automatic graceful degradation"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            manager = get_degradation_manager()
            return await manager.execute_with_degradation(
                service_name, operation_name, func, *args, **kwargs
            )

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            async def run():
                manager = get_degradation_manager()
                return await manager.execute_with_degradation(
                    service_name, operation_name, func, *args, **kwargs
                )

            loop = asyncio.get_event_loop()
            return loop.run_until_complete(run())

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Convenience functions
async def update_component_health(
    component_name: str, is_healthy: bool, response_time_ms: float = 0.0
):
    """Update component health status"""
    manager = get_degradation_manager()
    status = ComponentStatus.AVAILABLE if is_healthy else ComponentStatus.UNAVAILABLE
    await manager.update_component_status(component_name, status, response_time_ms)


async def check_degradation_needed(service_name: str, operation_name: str) -> bool:
    """Check if degradation is currently needed for a service operation"""
    manager = get_degradation_manager()
    context = await manager.evaluate_degradation(service_name, operation_name)
    result = await manager.apply_degradation(context)
    return result is not None


def get_service_health() -> dict[str, Any]:
    """Get overall service health and degradation status"""
    manager = get_degradation_manager()
    return manager.get_service_status()


def setup_default_degradation_policies():
    """Set up default degradation policies for common services"""
    manager = get_degradation_manager()

    # Search service policy
    search_policy = DegradationPolicy(
        name="search_degradation",
        service_name="search",
        degradation_level=DegradationLevel.REDUCED,
        trigger_conditions={"error_rate": 0.1, "response_time_ms": 5000},
        required_components=["vector_database", "embedding_service"],
        optional_components=["analytics", "recommendations"],
        feature_flags={"advanced_search": False, "recommendations": False},
    )

    # Knowledge service policy
    knowledge_policy = DegradationPolicy(
        name="knowledge_degradation",
        service_name="knowledge",
        degradation_level=DegradationLevel.MINIMAL,
        trigger_conditions={"error_rate": 0.2, "component_failures": 2},
        required_components=["database", "file_system"],
        optional_components=["ai_analysis", "content_extraction"],
    )

    # API service policy
    api_policy = DegradationPolicy(
        name="api_degradation",
        service_name="api",
        degradation_level=DegradationLevel.EMERGENCY,
        trigger_conditions={"error_rate": 0.5, "system_load": 0.9},
        required_components=["database", "auth_service"],
        emergency_response="read_only_mode",
    )

    for policy in [search_policy, knowledge_policy, api_policy]:
        manager.register_policy(policy)

    logger.info("Default degradation policies configured")
