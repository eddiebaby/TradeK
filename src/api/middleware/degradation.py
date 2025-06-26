"""
Graceful Degradation Middleware for FastAPI.

This module provides middleware to automatically apply graceful degradation
patterns to API endpoints based on system health and component availability.
"""

import asyncio
import functools
import logging
import time
from typing import Any

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ...monitoring.health_checks import get_health_manager
from ...resilience.graceful_degradation import (
    ComponentStatus,
    DegradationLevel,
    get_degradation_manager,
    update_component_health,
)

logger = logging.getLogger(__name__)


class GracefulDegradationMiddleware(BaseHTTPMiddleware):
    """
    Middleware that applies graceful degradation patterns to API requests
    based on current system health and component availability.
    """

    def __init__(self, app, enabled: bool = True, check_interval_seconds: float = 30.0):
        super().__init__(app)
        self.enabled = enabled
        self.check_interval_seconds = check_interval_seconds
        self.last_health_check = 0.0
        self.degradation_manager = get_degradation_manager()
        self.health_manager = get_health_manager()

        # Define critical endpoints that should always be available
        self.critical_endpoints = {
            "/health/",
            "/health/live",
            "/health/ready",
            "/health/status",
        }

        # Define service mappings for different endpoints
        self.endpoint_service_mapping = {
            "/search": "search_service",
            "/knowledge": "knowledge_service",
            "/analytics": "analytics_service",
            "/admin": "admin_service",
            "/ingestion": "ingestion_service",
        }

    async def dispatch(self, request: Request, call_next) -> Response:
        """Main middleware dispatch method"""
        if not self.enabled:
            return await call_next(request)

        start_time = time.time()

        try:
            # Update component health status periodically
            await self._update_component_health_if_needed()

            # Check if this is a critical endpoint that should never be degraded
            if self._is_critical_endpoint(request.url.path):
                return await call_next(request)

            # Determine service name for this endpoint
            service_name = self._get_service_name(request.url.path)
            operation_name = self._get_operation_name(request.method, request.url.path)

            # Evaluate if degradation is needed
            context = await self.degradation_manager.evaluate_degradation(
                service_name,
                operation_name,
                user_context={
                    "method": request.method,
                    "path": request.url.path,
                    "client_ip": request.client.host if request.client else "unknown",
                },
            )

            # Check if degradation should be applied
            degradation_result = await self.degradation_manager.apply_degradation(
                context
            )

            if degradation_result:
                # Return degraded response
                return await self._create_degraded_response(request, degradation_result)

            # Proceed with normal request processing
            response = await call_next(request)

            # Record successful operation
            duration_ms = (time.time() - start_time) * 1000
            await update_component_health(service_name, True, duration_ms)

            return response

        except Exception as e:
            # Record failed operation
            duration_ms = (time.time() - start_time) * 1000
            service_name = self._get_service_name(request.url.path)
            await update_component_health(service_name, False, duration_ms)

            logger.error(f"Error in degradation middleware: {e}")

            # Try to provide a degraded response instead of failing completely
            try:
                emergency_context = await self.degradation_manager.evaluate_degradation(
                    service_name, "emergency_fallback"
                )
                degradation_result = await self.degradation_manager.apply_degradation(
                    emergency_context
                )

                if degradation_result:
                    return await self._create_degraded_response(
                        request, degradation_result
                    )
            except Exception as fallback_error:
                logger.error(f"Emergency fallback failed: {fallback_error}")

            # If all else fails, re-raise the original exception
            raise e

    async def _update_component_health_if_needed(self):
        """Update component health based on health check results"""
        current_time = time.time()

        if current_time - self.last_health_check >= self.check_interval_seconds:
            try:
                # Get current health status
                health_data = self.health_manager.get_overall_health()

                # Update component statuses based on health checks
                for check_name, check_result in health_data.get("checks", {}).items():
                    status = check_result.get("status", "unknown")
                    duration_ms = check_result.get("duration_ms", 0.0)

                    # Map health status to component status
                    if status == "healthy":
                        component_status = ComponentStatus.AVAILABLE
                    elif status == "degraded":
                        component_status = ComponentStatus.DEGRADED
                    else:
                        component_status = ComponentStatus.UNAVAILABLE

                    await self.degradation_manager.update_component_status(
                        check_name, component_status, duration_ms
                    )

                self.last_health_check = current_time

            except Exception as e:
                logger.warning(f"Failed to update component health: {e}")

    def _is_critical_endpoint(self, path: str) -> bool:
        """Check if an endpoint is critical and should never be degraded"""
        return any(path.startswith(critical) for critical in self.critical_endpoints)

    def _get_service_name(self, path: str) -> str:
        """Get service name based on endpoint path"""
        for endpoint_prefix, service_name in self.endpoint_service_mapping.items():
            if path.startswith(endpoint_prefix):
                return service_name
        return "unknown_service"

    def _get_operation_name(self, method: str, path: str) -> str:
        """Get operation name based on HTTP method and path"""
        # Extract operation from path
        path_parts = path.strip("/").split("/")

        if len(path_parts) >= 2:
            operation = path_parts[1]
        else:
            operation = "index"

        return f"{method.lower()}_{operation}"

    async def _create_degraded_response(
        self, request: Request, degradation_result: dict[str, Any]
    ) -> JSONResponse:
        """Create a degraded response based on degradation result"""

        level = degradation_result.get("level", "unknown")
        result = degradation_result.get("result", {})

        # Determine appropriate HTTP status code
        if level == DegradationLevel.EMERGENCY.value:
            status_code = 503  # Service Unavailable
        elif level == DegradationLevel.MINIMAL.value:
            status_code = 206  # Partial Content
        else:
            status_code = 200  # OK but degraded

        # Create response content
        response_content = {
            "degraded": True,
            "degradation_level": level,
            "message": result.get("message", "Service is temporarily degraded"),
            "data": result.get("data"),
            "available_features": result.get("available_features", []),
            "disabled_features": result.get("disabled_features", []),
            "mode": result.get("mode", "degraded"),
            "timestamp": time.time(),
            "request_id": getattr(request.state, "request_id", None),
        }

        # Add specific response data based on degradation type
        if result.get("mode") == "cached_response":
            response_content["cache_info"] = {
                "age_minutes": result.get("cache_age_minutes", 0),
                "message": "Serving cached data due to service issues",
            }

        elif result.get("mode") == "simplified_search":
            response_content.update(
                {
                    "results": result.get("results", []),
                    "suggestions": result.get("suggestions", []),
                    "search_limited": True,
                }
            )

        elif result.get("mode") == "emergency":
            response_content.update(
                {
                    "available_operations": result.get("available_operations", []),
                    "estimated_recovery": result.get("estimated_recovery", "unknown"),
                    "contact_support": result.get("contact_support", False),
                }
            )

        # Add degradation headers
        headers = {
            "X-Degradation-Level": level,
            "X-Degradation-Strategy": degradation_result.get("strategy", "unknown"),
            "X-Service-Mode": result.get("mode", "degraded"),
            "Cache-Control": "no-cache, no-store, must-revalidate",  # Don't cache degraded responses
        }

        return JSONResponse(
            content=response_content, status_code=status_code, headers=headers
        )


class ServiceHealthTracker:
    """
    Utility class to track service health and automatically update
    degradation manager with component statuses.
    """

    def __init__(self):
        self.degradation_manager = get_degradation_manager()
        self.service_response_times: dict[str, float] = {}
        self.service_error_counts: dict[str, int] = {}

    async def record_service_call(
        self,
        service_name: str,
        success: bool,
        response_time_ms: float,
        error: Exception | None = None,
    ):
        """Record a service call result"""

        # Update response time tracking
        self.service_response_times[service_name] = response_time_ms

        # Update error counting
        if not success:
            self.service_error_counts[service_name] = (
                self.service_error_counts.get(service_name, 0) + 1
            )
        else:
            # Decrease error count on success (with minimum of 0)
            self.service_error_counts[service_name] = max(
                0, self.service_error_counts.get(service_name, 0) - 1
            )

        # Determine component status
        error_count = self.service_error_counts.get(service_name, 0)

        if not success and error_count >= 5:
            status = ComponentStatus.UNAVAILABLE
        elif not success or error_count >= 2:
            status = ComponentStatus.DEGRADED
        else:
            status = ComponentStatus.AVAILABLE

        # Update degradation manager
        await self.degradation_manager.update_component_status(
            service_name,
            status,
            response_time_ms,
            metadata={
                "error_count": error_count,
                "last_error": str(error) if error else None,
            },
        )

    def get_service_health_summary(self) -> dict[str, Any]:
        """Get a summary of service health"""
        return {
            "services": list(self.service_response_times.keys()),
            "average_response_times": self.service_response_times.copy(),
            "error_counts": self.service_error_counts.copy(),
            "total_services": len(self.service_response_times),
            "healthy_services": len(
                [
                    service
                    for service, errors in self.service_error_counts.items()
                    if errors == 0
                ]
            ),
        }


# Global service health tracker
_global_health_tracker: ServiceHealthTracker | None = None


def get_service_health_tracker() -> ServiceHealthTracker:
    """Get or create global service health tracker"""
    global _global_health_tracker
    if _global_health_tracker is None:
        _global_health_tracker = ServiceHealthTracker()
    return _global_health_tracker


# Decorator for automatic service health tracking
def track_service_health(service_name: str):
    """Decorator to automatically track service health"""

    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracker = get_service_health_tracker()
            start_time = time.time()

            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                response_time_ms = (time.time() - start_time) * 1000
                await tracker.record_service_call(service_name, True, response_time_ms)

                return result

            except Exception as e:
                response_time_ms = (time.time() - start_time) * 1000
                await tracker.record_service_call(
                    service_name, False, response_time_ms, e
                )
                raise e

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            async def run():
                return await async_wrapper(*args, **kwargs)

            loop = asyncio.get_event_loop()
            return loop.run_until_complete(run())

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
