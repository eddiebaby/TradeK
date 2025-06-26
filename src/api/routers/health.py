"""
Health Check API Endpoints for TradeKnowledge.

This module provides REST API endpoints for system health monitoring
and status reporting using the comprehensive health check system.
"""

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ...core.models import User
from ...monitoring.health_checks import (
    CheckPriority,
    HealthStatus,
    get_health_manager,
    get_system_health,
    run_health_checks,
)
from ..main import get_current_user

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


@router.get("/live")
async def liveness_probe():
    """
    Kubernetes/container liveness probe endpoint.

    Returns 200 if the application is running and can handle requests.
    This is a minimal check that doesn't perform expensive operations.
    """
    try:
        return JSONResponse(
            status_code=200,
            content={"status": "alive", "timestamp": datetime.now().isoformat()},
        )
    except Exception as e:
        logger.error(f"Liveness probe failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "dead",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            },
        )


@router.get("/ready")
async def readiness_probe():
    """
    Kubernetes/container readiness probe endpoint.

    Returns 200 if the application is ready to serve traffic.
    Performs essential checks to ensure the service can function properly.
    """
    try:
        manager = get_health_manager()

        # Run only critical checks for readiness
        critical_checks = manager.registry.get_checks(CheckPriority.CRITICAL)

        if not critical_checks:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "ready",
                    "timestamp": datetime.now().isoformat(),
                    "message": "No critical checks registered",
                },
            )

        # Run critical checks
        results = {}
        for name, check_config in critical_checks.items():
            try:
                result = await manager._run_single_check(name, check_config)
                results[name] = result
            except Exception as e:
                logger.error(f"Critical check {name} failed: {e}")
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "not_ready",
                        "timestamp": datetime.now().isoformat(),
                        "message": f"Critical check {name} failed: {e}",
                    },
                )

        # Check if any critical checks failed
        failed_critical = [
            name
            for name, result in results.items()
            if result.status == HealthStatus.UNHEALTHY
        ]

        if failed_critical:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "timestamp": datetime.now().isoformat(),
                    "message": f"Critical systems unhealthy: {', '.join(failed_critical)}",
                },
            )

        return JSONResponse(
            status_code=200,
            content={
                "status": "ready",
                "timestamp": datetime.now().isoformat(),
                "critical_checks": len(results),
            },
        )

    except Exception as e:
        logger.error(f"Readiness probe failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            },
        )


@router.get("/checks")
async def get_all_checks():
    """
    Run and return all health checks.

    Returns detailed results for all registered health checks.
    """
    try:
        results = await run_health_checks()

        formatted_results = {}
        for name, result in results.items():
            formatted_results[name] = {
                "name": result.name,
                "status": result.status.value,
                "message": result.message,
                "priority": result.priority.value,
                "timestamp": result.timestamp.isoformat(),
                "duration_ms": result.duration_ms,
                "details": result.details,
                "error": result.error,
            }

        return JSONResponse(status_code=200, content=formatted_results)

    except Exception as e:
        logger.error(f"Failed to run health checks: {e}")
        raise HTTPException(
            status_code=500, detail=f"Health check execution failed: {e}"
        )


@router.get("/checks/{check_name}")
async def get_specific_check(
    check_name: str = Path(..., description="Name of the specific health check")
):
    """
    Run and return a specific health check.

    Args:
        check_name: Name of the health check to run

    Returns:
        Detailed result for the specified health check
    """
    try:
        manager = get_health_manager()
        checks = manager.registry.get_checks()

        if check_name not in checks:
            raise HTTPException(
                status_code=404, detail=f"Health check '{check_name}' not found"
            )

        check_config = checks[check_name]
        result = await manager._run_single_check(check_name, check_config)

        return JSONResponse(
            status_code=200,
            content={
                "name": result.name,
                "status": result.status.value,
                "message": result.message,
                "priority": result.priority.value,
                "timestamp": result.timestamp.isoformat(),
                "duration_ms": result.duration_ms,
                "details": result.details,
                "error": result.error,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to run health check {check_name}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Health check execution failed: {e}"
        )


@router.get("/history")
async def get_health_history(
    check_name: str | None = Query(
        None, description="Filter by specific check name"
    ),
    since: str | None = Query(
        None, description="ISO timestamp to filter results since"
    ),
    limit: int = Query(
        100, ge=1, le=1000, description="Maximum number of results to return"
    ),
    current_user: User = Depends(get_current_user),
):
    """
    Get health check history.

    Args:
        check_name: Optional filter for specific health check
        since: Optional ISO timestamp to filter results since this time
        limit: Maximum number of results (1-1000)

    Returns:
        List of historical health check results
    """
    try:
        manager = get_health_manager()

        since_datetime = None
        if since:
            try:
                since_datetime = datetime.fromisoformat(since.replace("Z", "+00:00"))
            except ValueError:
                raise HTTPException(
                    status_code=400, detail="Invalid 'since' timestamp format"
                )

        history = manager.get_health_history(
            check_name=check_name, since=since_datetime, limit=limit
        )

        return JSONResponse(
            status_code=200,
            content={
                "count": len(history),
                "check_name": check_name,
                "since": since,
                "limit": limit,
                "results": history,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get health history: {e}")
        raise HTTPException(
            status_code=500, detail=f"Health history retrieval failed: {e}"
        )


@router.get("/metrics")
async def get_system_metrics():
    """
    Get current system resource metrics.

    Returns:
        Current system metrics including CPU, memory, disk usage, etc.
    """
    try:
        manager = get_health_manager()
        metrics = manager._get_system_metrics()

        return JSONResponse(
            status_code=200,
            content={
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "disk_percent": metrics.disk_percent,
                "available_memory_gb": metrics.available_memory_gb,
                "disk_usage_gb": metrics.disk_usage_gb,
                "process_count": metrics.process_count,
                "uptime_seconds": metrics.uptime_seconds,
                "load_average": metrics.load_average,
                "open_files": metrics.open_files,
                "network_connections": metrics.network_connections,
                "timestamp": datetime.now().isoformat(),
            },
        )

    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(
            status_code=500, detail=f"System metrics retrieval failed: {e}"
        )


@router.post("/monitoring/start")
async def start_health_monitoring(current_user: User = Depends(get_current_user)):
    """Start continuous health monitoring."""
    try:
        manager = get_health_manager()

        if manager._running:
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Health monitoring is already running",
                    "monitoring_active": True,
                    "timestamp": datetime.now().isoformat(),
                },
            )

        await manager.start_monitoring()

        return JSONResponse(
            status_code=200,
            content={
                "message": "Health monitoring started",
                "monitoring_active": True,
                "check_interval_seconds": manager.check_interval,
                "timestamp": datetime.now().isoformat(),
            },
        )

    except Exception as e:
        logger.error(f"Failed to start health monitoring: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start health monitoring: {e}"
        )


@router.post("/monitoring/stop")
async def stop_health_monitoring(current_user: User = Depends(get_current_user)):
    """Stop continuous health monitoring."""
    try:
        manager = get_health_manager()

        if not manager._running:
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Health monitoring is not running",
                    "monitoring_active": False,
                    "timestamp": datetime.now().isoformat(),
                },
            )

        await manager.stop_monitoring()

        return JSONResponse(
            status_code=200,
            content={
                "message": "Health monitoring stopped",
                "monitoring_active": False,
                "timestamp": datetime.now().isoformat(),
            },
        )

    except Exception as e:
        logger.error(f"Failed to stop health monitoring: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to stop health monitoring: {e}"
        )
