"""
Health Check API Endpoints for TradeKnowledge.

This module provides REST API endpoints for system health monitoring
and status reporting using the comprehensive health check system.
"""

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ...core.models import User
from ...monitoring.health_checks import (
    HealthStatus,
    get_system_health,
)
from ..auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


class HealthResponse(BaseModel):
    """Response model for health check endpoints"""

    status: str = Field(..., description="Overall health status")
    message: str = Field(..., description="Human-readable status message")
    timestamp: str = Field(..., description="ISO timestamp of health check")
    checks: dict[str, Any] = Field(
        default_factory=dict, description="Individual check results"
    )
    summary: dict[str, int] = Field(
        default_factory=dict, description="Health check summary"
    )


class CheckResult(BaseModel):
    """Model for individual health check result"""

    name: str = Field(..., description="Name of the health check")
    status: str = Field(..., description="Health status")
    message: str = Field(..., description="Status message")
    priority: str = Field(..., description="Check priority level")
    timestamp: str = Field(..., description="Check timestamp")
    duration_ms: float = Field(..., description="Check duration in milliseconds")
    details: dict[str, Any] = Field(
        default_factory=dict, description="Additional details"
    )
    error: str | None = Field(None, description="Error message if check failed")


@router.get("/", response_model=HealthResponse)
async def get_health():
    """
    Get overall system health status.

    Returns comprehensive health information including:
    - Overall status (healthy/degraded/unhealthy)
    - Individual check results
    - Summary statistics
    """
    try:
        health_data = await get_system_health()
        return JSONResponse(
            status_code=200 if health_data["status"] == "healthy" else 503,
            content=health_data,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": f"Health check system error: {e}",
                "timestamp": datetime.now().isoformat(),
                "checks": {},
                "summary": {
                    "total_checks": 0,
                    "healthy": 0,
                    "degraded": 0,
                    "unhealthy": 1,
                    "unknown": 0,
                },
            },
        )


@router.get("/status")
async def get_health_status():
    """
    Get simplified health status for load balancers.

    Returns:
    - 200 OK if system is healthy
    - 503 Service Unavailable if system is degraded or unhealthy
    """
    try:
        health_data = await get_system_health()
        status_code = 200 if health_data["status"] == "healthy" else 503

        return JSONResponse(
            status_code=status_code,
            content={
                "status": health_data["status"],
                "timestamp": health_data["timestamp"],
            },
        )
    except Exception as e:
        logger.error(f"Health status check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "timestamp": datetime.now().isoformat()},
        )


@router.get("/component/{component_name}", summary="Component Health Check")
async def component_health(component_name: str):
    """
    Check health of a specific component.

    Args:
        component_name: Name of component to check

    Returns:
        Health status for the specified component
    """
    monitor = get_health_monitor()
    component_result = monitor.get_component_health(component_name)

    if not component_result:
        # Try to run a fresh check for this component
        if component_name in monitor.checkers:
            component_result = await monitor.checkers[component_name].check()
        else:
            raise HTTPException(
                status_code=404, detail=f"Component '{component_name}' not found"
            )

    response_data = component_result.to_dict()

    # Set HTTP status based on component health
    if component_result.status == HealthStatus.HEALTHY:
        status_code = 200
    elif component_result.status == HealthStatus.WARNING:
        status_code = 200
    else:
        status_code = 503

    return JSONResponse(status_code=status_code, content=response_data)


@router.get("/history", summary="Health History")
async def health_history(
    hours: int = Query(24, description="Hours of history to retrieve", ge=1, le=168),
    current_user: User = Depends(get_current_user),
):
    """
    Get health check history for the specified time period.

    Requires authentication.

    Args:
        hours: Number of hours of history to retrieve (1-168)

    Returns:
        List of health check results over time
    """
    monitor = get_health_monitor()
    history = monitor.get_health_history(hours=hours)

    return {
        "period_hours": hours,
        "entries_count": len(history),
        "history": [health.to_dict() for health in history],
    }


@router.get("/stats", summary="Monitoring Statistics")
async def monitoring_stats(current_user: User = Depends(get_current_user)):
    """
    Get health monitoring statistics and configuration.

    Requires authentication.

    Returns:
        Monitoring system statistics and status
    """
    monitor = get_health_monitor()
    stats = monitor.get_stats()

    # Add component list
    stats["components"] = list(monitor.checkers.keys())

    return stats


@router.post("/check", summary="Trigger Health Check")
async def trigger_health_check(current_user: User = Depends(get_current_user)):
    """
    Manually trigger a comprehensive health check.

    Requires authentication.

    Returns:
        Fresh health check results for all components
    """
    try:
        system_health = await check_system_health()

        return {
            "message": "Health check completed",
            "triggered_by": current_user.username,
            "result": system_health.to_dict(),
        }

    except Exception as e:
        logger.error(
            "Manual health check failed", error=str(e), user=current_user.username
        )
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/readiness", summary="Readiness Check")
async def readiness_check():
    """
    Kubernetes-style readiness check.

    Returns 200 if the service is ready to receive traffic,
    503 if not ready (e.g., still initializing).
    """
    try:
        # Quick check of critical components
        monitor = get_health_monitor()

        # Check if we have any recent health data
        current_health = monitor.get_current_health()
        if not current_health:
            # No health data yet, trigger a quick check
            current_health = await check_system_health()

        # Service is ready if critical components are not failing
        critical_components = ["sqlite_database", "file_system"]

        for component_name in critical_components:
            if component_name in current_health.components:
                component = current_health.components[component_name]
                if component.status == HealthStatus.CRITICAL:
                    return JSONResponse(
                        status_code=503,
                        content={
                            "status": "not_ready",
                            "reason": f"Critical component failure: {component_name}",
                            "message": component.message,
                        },
                    )

        return {
            "status": "ready",
            "timestamp": current_health.timestamp.isoformat(),
            "uptime_seconds": current_health.uptime_seconds,
        }

    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "reason": "readiness_check_failed",
                "message": str(e),
            },
        )


@router.get("/liveness", summary="Liveness Check")
async def liveness_check():
    """
    Kubernetes-style liveness check.

    Returns 200 if the service is alive and should not be restarted,
    500 if the service is dead and should be restarted.
    """
    try:
        # Very basic check - can we respond and access basic functionality?
        monitor = get_health_monitor()

        # Check if monitoring is working
        if not monitor.is_running:
            # Try to get at least basic status
            import os

            if not os.path.exists("data"):
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": "dead",
                        "reason": "critical_file_system_failure",
                    },
                )

        return {
            "status": "alive",
            "monitoring_active": monitor.is_running,
            "uptime_seconds": monitor.get_stats().get("uptime_seconds", 0),
        }

    except Exception as e:
        logger.error("Liveness check failed", error=str(e))
        return JSONResponse(
            status_code=500,
            content={
                "status": "dead",
                "reason": "liveness_check_failed",
                "message": str(e),
            },
        )


@router.get("/metrics", summary="Health Metrics")
async def health_metrics(current_user: User = Depends(get_current_user)):
    """
    Get health metrics in a format suitable for monitoring systems.

    Requires authentication.

    Returns:
        Health metrics for external monitoring systems
    """
    monitor = get_health_monitor()
    current_health = monitor.get_current_health()

    if not current_health:
        raise HTTPException(status_code=503, detail="No health data available")

    # Convert to metrics format
    metrics = {
        "system_health_status": {
            "healthy": (
                1 if current_health.overall_status == HealthStatus.HEALTHY else 0
            ),
            "warning": (
                1 if current_health.overall_status == HealthStatus.WARNING else 0
            ),
            "critical": (
                1 if current_health.overall_status == HealthStatus.CRITICAL else 0
            ),
        },
        "system_uptime_seconds": current_health.uptime_seconds,
        "components": {},
    }

    # Component metrics
    for name, component in current_health.components.items():
        metrics["components"][name] = {
            "status_healthy": 1 if component.status == HealthStatus.HEALTHY else 0,
            "status_warning": 1 if component.status == HealthStatus.WARNING else 0,
            "status_critical": 1 if component.status == HealthStatus.CRITICAL else 0,
            "check_duration_ms": component.duration_ms,
            "metadata": component.metadata,
        }

    return metrics
