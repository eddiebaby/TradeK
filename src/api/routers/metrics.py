"""
Metrics API Endpoints

Provides access to system metrics, analytics, and Prometheus monitoring data.
"""

import structlog
from fastapi import APIRouter, Depends, Response
from fastapi.responses import PlainTextResponse

from ...core.models import User
from ..auth import get_current_user
from ..metrics import CONTENT_TYPE_LATEST, PROMETHEUS_AVAILABLE, get_metrics_collector

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/metrics", tags=["Metrics"])


@router.get("/prometheus", summary="Prometheus Metrics Endpoint")
async def prometheus_metrics():
    """
    Prometheus metrics endpoint for external monitoring systems.

    Returns metrics in Prometheus text format for scraping by
    monitoring systems like Prometheus, Grafana, etc.
    """
    metrics_collector = get_metrics_collector()

    if not PROMETHEUS_AVAILABLE:
        return PlainTextResponse(
            content="# Prometheus metrics not available - prometheus_client package not installed\n",
            status_code=503,
        )

    try:
        metrics_data = metrics_collector.get_prometheus_metrics()
        return Response(
            content=metrics_data,
            media_type=CONTENT_TYPE_LATEST if PROMETHEUS_AVAILABLE else "text/plain",
        )
    except Exception as e:
        logger.error("Failed to generate Prometheus metrics", error=str(e))
        return PlainTextResponse(
            content=f"# Error generating metrics: {str(e)}\n", status_code=500
        )


@router.get("/system", summary="System Metrics")
async def system_metrics(
    hours: int = 24, current_user: User = Depends(get_current_user)
):
    """
    Get system performance metrics for the specified time period.

    Requires authentication.

    Args:
        hours: Number of hours of metrics to retrieve (default: 24)

    Returns:
        System performance metrics including CPU, memory, disk usage
    """
    metrics_collector = get_metrics_collector()
    return await metrics_collector.get_system_metrics(period_hours=hours)


@router.get("/search", summary="Search Analytics")
async def search_analytics(
    hours: int = 24, current_user: User = Depends(get_current_user)
):
    """
    Get search analytics for the specified time period.

    Requires authentication.

    Args:
        hours: Number of hours of analytics to retrieve (default: 24)

    Returns:
        Search analytics including query patterns, performance, intent distribution
    """
    metrics_collector = get_metrics_collector()
    return await metrics_collector.get_search_analytics(period_hours=hours)


@router.get("/users", summary="User Analytics")
async def user_analytics(
    hours: int = 24, current_user: User = Depends(get_current_user)
):
    """
    Get user behavior analytics for the specified time period.

    Requires authentication.

    Args:
        hours: Number of hours of analytics to retrieve (default: 24)

    Returns:
        User analytics including activity patterns, most active users
    """
    metrics_collector = get_metrics_collector()
    return await metrics_collector.get_user_analytics(period_hours=hours)


@router.get("/errors", summary="Error Metrics")
async def error_metrics(
    hours: int = 24, current_user: User = Depends(get_current_user)
):
    """
    Get error metrics for the specified time period.

    Requires authentication.

    Args:
        hours: Number of hours of metrics to retrieve (default: 24)

    Returns:
        Error metrics including error rates, types, and trends
    """
    metrics_collector = get_metrics_collector()
    return await metrics_collector.get_error_metrics(period_hours=hours)


@router.get("/uptime", summary="Uptime Information")
async def uptime_info(current_user: User = Depends(get_current_user)):
    """
    Get system uptime and performance summary.

    Requires authentication.

    Returns:
        Uptime information and request/error statistics
    """
    metrics_collector = get_metrics_collector()
    return await metrics_collector.get_uptime_info()


@router.get("/summary", summary="Metrics Summary")
async def metrics_summary(current_user: User = Depends(get_current_user)):
    """
    Get comprehensive metrics summary dashboard.

    Requires authentication.

    Returns:
        Combined view of system, search, user, and error metrics
    """
    metrics_collector = get_metrics_collector()

    # Gather all metrics concurrently
    import asyncio

    system_task = metrics_collector.get_system_metrics(period_hours=24)
    search_task = metrics_collector.get_search_analytics(period_hours=24)
    user_task = metrics_collector.get_user_analytics(period_hours=24)
    error_task = metrics_collector.get_error_metrics(period_hours=24)
    uptime_task = metrics_collector.get_uptime_info()

    system_metrics, search_analytics, user_analytics, error_metrics, uptime_info = (
        await asyncio.gather(
            system_task, search_task, user_task, error_task, uptime_task
        )
    )

    return {
        "summary": {
            "service_status": "operational",
            "uptime_hours": uptime_info["uptime_hours"],
            "total_requests": uptime_info["total_requests"],
            "error_rate": error_metrics["error_rate"],
            "active_users_24h": user_analytics["active_users"],
            "searches_24h": search_analytics["total_searches"],
            "prometheus_enabled": PROMETHEUS_AVAILABLE,
        },
        "system": system_metrics,
        "search": search_analytics,
        "users": user_analytics,
        "errors": error_metrics,
        "uptime": uptime_info,
    }
