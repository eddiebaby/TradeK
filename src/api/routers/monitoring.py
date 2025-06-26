"""
Monitoring and Observability API Endpoints
Provides real-time monitoring data, metrics, health checks, and alerts
"""

import json
from datetime import datetime
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse

from ...core.monitoring_service import get_monitoring_service
from ..auth.authentication import User, require_admin
from ..middleware.security import validate_request_security

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.get("/dashboard")
@validate_request_security
async def get_monitoring_dashboard(admin_user: User = Depends(require_admin)):
    """
    Get comprehensive monitoring dashboard data

    Returns:
    - System health overview
    - Key performance metrics
    - Active alerts
    - Component health checks
    - System information
    """
    try:
        monitoring_service = await get_monitoring_service()
        dashboard_data = monitoring_service.get_dashboard_data()

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "dashboard": dashboard_data,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    except Exception as e:
        logger.error("Failed to get monitoring dashboard", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve dashboard data: {str(e)}"
        )


@router.get("/metrics")
@validate_request_security
async def get_metrics(
    metric_names: str | None = Query(
        None, description="Comma-separated metric names"
    ),
    minutes: int = Query(5, ge=1, le=60, description="Time window in minutes"),
    admin_user: User = Depends(require_admin),
):
    """
    Get specific metrics data

    Returns time-series data for requested metrics
    """
    try:
        monitoring_service = await get_monitoring_service()

        if metric_names:
            requested_metrics = [name.strip() for name in metric_names.split(",")]
        else:
            # Default key metrics
            requested_metrics = [
                "cpu_usage_percent",
                "memory_usage_percent",
                "response_time_ms",
                "cache_hit_rate_percent",
            ]

        metrics_data = {}
        for metric_name in requested_metrics:
            history = monitoring_service.metrics_collector.get_metric_history(
                metric_name, minutes
            )
            summary = monitoring_service.metrics_collector.get_metric_summary(
                metric_name, minutes
            )

            metrics_data[metric_name] = {
                "summary": summary,
                "history": [
                    {
                        "timestamp": point.timestamp.isoformat(),
                        "value": point.value,
                        "labels": point.labels,
                    }
                    for point in history
                ],
            }

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "metrics": metrics_data,
                "time_window_minutes": minutes,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    except Exception as e:
        logger.error("Failed to get metrics", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve metrics: {str(e)}"
        )


@router.get("/health")
async def get_system_health():
    """
    Get overall system health status

    Public endpoint for health monitoring and load balancers
    """
    try:
        monitoring_service = await get_monitoring_service()

        if not monitoring_service.is_running:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "message": "Monitoring service not running",
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

        overall_health = monitoring_service.health_monitor.get_overall_health()
        health_checks = await monitoring_service.health_monitor.run_health_checks()

        status_code = 200
        if overall_health == "unhealthy":
            status_code = 503
        elif overall_health == "degraded":
            status_code = 200  # Still serving traffic but with warnings

        return JSONResponse(
            status_code=status_code,
            content={
                "status": overall_health,
                "components": {
                    name: {
                        "status": check.status,
                        "response_time_ms": check.response_time_ms,
                    }
                    for name, check in health_checks.items()
                },
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": "Health check service unavailable",
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


@router.get("/health/detailed")
@validate_request_security
async def get_detailed_health(admin_user: User = Depends(require_admin)):
    """
    Get detailed health information for all components

    Returns comprehensive health data including error messages and metadata
    """
    try:
        monitoring_service = await get_monitoring_service()
        health_checks = await monitoring_service.health_monitor.run_health_checks()

        detailed_health = {
            "overall_status": monitoring_service.health_monitor.get_overall_health(),
            "components": {
                name: {
                    "status": check.status,
                    "last_check": check.last_check.isoformat(),
                    "response_time_ms": check.response_time_ms,
                    "error_message": check.error_message,
                    "metadata": check.metadata,
                }
                for name, check in health_checks.items()
            },
            "monitoring_service": {
                "is_running": monitoring_service.is_running,
                "metrics_task_running": monitoring_service.metrics_task
                and not monitoring_service.metrics_task.done(),
                "alerts_task_running": monitoring_service.alerts_task
                and not monitoring_service.alerts_task.done(),
                "health_task_running": monitoring_service.health_task
                and not monitoring_service.health_task.done(),
            },
        }

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "health": detailed_health,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    except Exception as e:
        logger.error("Failed to get detailed health", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve detailed health: {str(e)}"
        )


@router.get("/alerts")
@validate_request_security
async def get_alerts(
    status: str | None = Query(None, regex="^(active|triggered|resolved)$"),
    severity: str | None = Query(None, regex="^(critical|warning|info)$"),
    admin_user: User = Depends(require_admin),
):
    """
    Get system alerts

    Returns alerts filtered by status and severity
    """
    try:
        monitoring_service = await get_monitoring_service()
        alerts = monitoring_service.alert_manager.alerts

        # Filter alerts
        filtered_alerts = {}
        for alert_id, alert in alerts.items():
            if status and alert.status != status:
                continue
            if severity and alert.severity != severity:
                continue

            filtered_alerts[alert_id] = {
                "id": alert.id,
                "name": alert.name,
                "description": alert.description,
                "severity": alert.severity,
                "condition": alert.condition,
                "threshold": alert.threshold,
                "status": alert.status,
                "triggered_at": (
                    alert.triggered_at.isoformat() if alert.triggered_at else None
                ),
                "resolved_at": (
                    alert.resolved_at.isoformat() if alert.resolved_at else None
                ),
            }

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "alerts": filtered_alerts,
                "total_count": len(filtered_alerts),
                "filters": {"status": status, "severity": severity},
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    except Exception as e:
        logger.error("Failed to get alerts", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve alerts: {str(e)}"
        )


@router.post("/alerts/acknowledge/{alert_id}")
@validate_request_security
async def acknowledge_alert(alert_id: str, admin_user: User = Depends(require_admin)):
    """
    Acknowledge an alert

    Marks an alert as acknowledged by an administrator
    """
    try:
        monitoring_service = await get_monitoring_service()

        if alert_id not in monitoring_service.alert_manager.alerts:
            raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found")

        alert = monitoring_service.alert_manager.alerts[alert_id]

        # Add acknowledgment metadata
        if not hasattr(alert, "metadata"):
            alert.metadata = {}

        alert.metadata["acknowledged_by"] = admin_user.id
        alert.metadata["acknowledged_at"] = datetime.utcnow().isoformat()

        logger.info("Alert acknowledged", alert_id=alert_id, admin_user=admin_user.id)

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": f"Alert '{alert_id}' acknowledged",
                "acknowledged_by": admin_user.id,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to acknowledge alert", alert_id=alert_id, error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to acknowledge alert: {str(e)}"
        )


@router.get("/metrics/export")
@validate_request_security
async def export_metrics(
    format: str = Query("json", regex="^(json|csv|prometheus)$"),
    minutes: int = Query(60, ge=1, le=1440, description="Time window in minutes"),
    admin_user: User = Depends(require_admin),
):
    """
    Export metrics data in various formats

    Supports JSON, CSV, and Prometheus formats
    """
    try:
        monitoring_service = await get_monitoring_service()

        # Get all available metrics
        all_metrics = {}
        for metric_name in monitoring_service.metrics_collector.metrics.keys():
            history = monitoring_service.metrics_collector.get_metric_history(
                metric_name, minutes
            )
            all_metrics[metric_name] = history

        if format == "json":
            export_data = {
                "metrics": {
                    metric_name: [
                        {
                            "timestamp": point.timestamp.isoformat(),
                            "value": point.value,
                            "labels": point.labels,
                        }
                        for point in history
                    ]
                    for metric_name, history in all_metrics.items()
                },
                "exported_at": datetime.utcnow().isoformat(),
                "time_window_minutes": minutes,
            }

            return JSONResponse(status_code=200, content=export_data)

        elif format == "csv":
            import csv
            import io

            output = io.StringIO()
            writer = csv.writer(output)

            # Write header
            writer.writerow(["metric_name", "timestamp", "value", "labels"])

            # Write data
            for metric_name, history in all_metrics.items():
                for point in history:
                    writer.writerow(
                        [
                            metric_name,
                            point.timestamp.isoformat(),
                            point.value,
                            json.dumps(point.labels),
                        ]
                    )

            output.seek(0)

            def generate():
                yield output.getvalue()

            return StreamingResponse(
                generate(),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=metrics_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
                },
            )

        elif format == "prometheus":
            # Simple Prometheus text format
            output_lines = []

            for metric_name, history in all_metrics.items():
                if history:
                    # Use the latest value
                    latest_point = history[-1]
                    labels_str = ",".join(
                        [f'{k}="{v}"' for k, v in latest_point.labels.items()]
                    )
                    if labels_str:
                        output_lines.append(
                            f"{metric_name}{{{labels_str}}} {latest_point.value}"
                        )
                    else:
                        output_lines.append(f"{metric_name} {latest_point.value}")

            def generate():
                yield "\n".join(output_lines)

            return StreamingResponse(
                generate(),
                media_type="text/plain",
                headers={
                    "Content-Disposition": f"attachment; filename=metrics_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt"
                },
            )

    except Exception as e:
        logger.error("Failed to export metrics", format=format, error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to export metrics: {str(e)}"
        )


@router.get("/performance/report")
@validate_request_security
async def get_performance_report(
    hours: int = Query(1, ge=1, le=24, description="Time window in hours"),
    admin_user: User = Depends(require_admin),
):
    """
    Get comprehensive performance report

    Returns detailed performance analysis over specified time window
    """
    try:
        monitoring_service = await get_monitoring_service()
        minutes = hours * 60

        # Get key performance metrics
        key_metrics = [
            "cpu_usage_percent",
            "memory_usage_percent",
            "response_time_ms",
            "cache_hit_rate_percent",
            "error_rate_percent",
            "avg_query_time_ms",
        ]

        performance_data = {}
        for metric in key_metrics:
            summary = monitoring_service.metrics_collector.get_metric_summary(
                metric, minutes
            )
            history = monitoring_service.metrics_collector.get_metric_history(
                metric, minutes
            )

            performance_data[metric] = {
                "summary": summary,
                "trend": "stable",  # Could implement trend analysis
                "data_points": len(history),
            }

        # Calculate performance score (simple heuristic)
        score_factors = {
            "cpu_usage_percent": lambda x: max(0, 100 - x.get("avg", 100)),
            "memory_usage_percent": lambda x: max(0, 100 - x.get("avg", 100)),
            "response_time_ms": lambda x: max(0, 100 - (x.get("avg", 1000) / 10)),
            "cache_hit_rate_percent": lambda x: x.get("avg", 0),
            "error_rate_percent": lambda x: max(0, 100 - (x.get("avg", 100) * 10)),
        }

        scores = []
        for metric, calc_func in score_factors.items():
            if metric in performance_data and performance_data[metric]["summary"]:
                score = calc_func(performance_data[metric]["summary"])
                scores.append(score)

        overall_score = sum(scores) / len(scores) if scores else 0

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "performance_report": {
                    "overall_score": round(overall_score, 2),
                    "time_window_hours": hours,
                    "metrics": performance_data,
                    "recommendations": [
                        "Monitor response times during peak hours",
                        "Consider increasing cache size if hit rate is low",
                        "Review error patterns for optimization opportunities",
                    ],
                },
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    except Exception as e:
        logger.error("Failed to generate performance report", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to generate performance report: {str(e)}"
        )


@router.post("/metrics/manual")
@validate_request_security
async def record_manual_metric(
    metric_data: dict[str, Any], admin_user: User = Depends(require_admin)
):
    """
    Manually record a metric value

    Allows administrators to record custom metrics
    """
    try:
        monitoring_service = await get_monitoring_service()

        # Validate required fields
        if "name" not in metric_data or "value" not in metric_data:
            raise HTTPException(
                status_code=400, detail="Metric name and value are required"
            )

        metric_name = metric_data["name"]
        metric_value = float(metric_data["value"])
        labels = metric_data.get("labels", {})

        # Record the metric
        monitoring_service.metrics_collector.record_metric(
            metric_name=metric_name, value=metric_value, labels=labels
        )

        logger.info(
            "Manual metric recorded",
            metric_name=metric_name,
            value=metric_value,
            admin_user=admin_user.id,
        )

        return JSONResponse(
            status_code=201,
            content={
                "status": "success",
                "message": f"Metric '{metric_name}' recorded",
                "value": metric_value,
                "labels": labels,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid metric value: {str(e)}")
    except Exception as e:
        logger.error("Failed to record manual metric", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to record metric: {str(e)}"
        )
