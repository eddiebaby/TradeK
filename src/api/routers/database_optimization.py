"""
Database Optimization API Endpoints
Provides monitoring and management of database optimization features
"""

from datetime import datetime
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from ...core.database_optimization_service import get_optimization_service
from ...core.database_optimizer import get_database_optimizer
from ...core.index_manager import get_index_manager
from ...core.query_optimizer import get_query_analyzer
from ..auth.authentication import User, require_admin
from ..middleware.security import validate_request_security

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.get("/status")
@validate_request_security
async def get_optimization_status(admin_user: User = Depends(require_admin)):
    """
    Get comprehensive database optimization status

    Returns detailed information about:
    - Service status and configuration
    - Performance metrics and statistics
    - Connection pool status
    - Index statistics
    - Recent optimization recommendations
    """
    try:
        optimization_service = await get_optimization_service()
        status = await optimization_service.get_comprehensive_status()

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "data": status,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    except Exception as e:
        logger.error("Failed to get optimization status", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve optimization status: {str(e)}"
        )


@router.get("/performance")
@validate_request_security
async def get_performance_metrics(admin_user: User = Depends(require_admin)):
    """
    Get detailed performance metrics

    Returns:
    - Query execution statistics
    - Slow query analysis
    - Cache performance metrics
    - Connection pool utilization
    """
    try:
        database_optimizer = await get_database_optimizer()
        performance_report = database_optimizer.get_performance_report()

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "performance_report": performance_report,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    except Exception as e:
        logger.error("Failed to get performance metrics", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve performance metrics: {str(e)}"
        )


@router.get("/recommendations")
@validate_request_security
async def get_optimization_recommendations(
    priority: str | None = Query(None, regex="^(high|medium|low)$"),
    limit: int = Query(20, ge=1, le=100),
    admin_user: User = Depends(require_admin),
):
    """
    Get optimization recommendations

    Returns actionable recommendations for:
    - Index creation/removal
    - Query optimization
    - Configuration adjustments
    - Performance improvements
    """
    try:
        optimization_service = await get_optimization_service()
        recommendations = await optimization_service.get_optimization_recommendations()

        # Filter by priority if specified
        if priority:
            recommendations = [
                rec
                for rec in recommendations
                if rec.get("priority", "").lower() == priority.lower()
            ]

        # Limit results
        recommendations = recommendations[:limit]

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "recommendations": recommendations,
                "total_count": len(recommendations),
                "filtered_by_priority": priority,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    except Exception as e:
        logger.error("Failed to get optimization recommendations", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve recommendations: {str(e)}"
        )


@router.post("/maintenance/run")
@validate_request_security
async def run_optimization_maintenance(
    background_tasks: BackgroundTasks,
    force: bool = Query(False, description="Force immediate execution"),
    admin_user: User = Depends(require_admin),
):
    """
    Trigger database optimization maintenance

    Performs:
    - Index analysis and optimization
    - Cache cleanup
    - Connection pool optimization
    - Performance metrics collection
    """
    try:
        optimization_service = await get_optimization_service()

        if force:
            # Run immediately and return results
            result = await optimization_service.force_optimization_run()

            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "message": "Optimization maintenance completed",
                    "execution_details": result,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )
        else:
            # Run in background
            background_tasks.add_task(optimization_service.run_maintenance)

            return JSONResponse(
                status_code=202,
                content={
                    "status": "accepted",
                    "message": "Optimization maintenance scheduled in background",
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

    except Exception as e:
        logger.error("Failed to run optimization maintenance", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to run maintenance: {str(e)}"
        )


@router.get("/indexes/{database}")
@validate_request_security
async def get_database_indexes(
    database: str, admin_user: User = Depends(require_admin)
):
    """
    Get detailed information about database indexes

    Returns:
    - Existing indexes and their properties
    - Index usage statistics
    - Index recommendations
    - Storage impact analysis
    """
    try:
        # Map database names to paths
        database_paths = {"main": "data/knowledge.db", "knowledge": "data/knowledge.db"}

        if database not in database_paths:
            raise HTTPException(
                status_code=404, detail=f"Database '{database}' not found"
            )

        database_path = database_paths[database]
        index_manager = await get_index_manager(database_path)

        # Get comprehensive index information
        index_stats = await index_manager.get_index_statistics()
        usage_analysis = await index_manager.analyze_index_usage()
        recommendations = await index_manager.get_optimization_recommendations()

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "database": database,
                "index_statistics": index_stats,
                "usage_analysis": usage_analysis,
                "recommendations": recommendations,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get database indexes", database=database, error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve index information: {str(e)}"
        )


@router.post("/indexes/{database}/optimize")
@validate_request_security
async def optimize_database_indexes(
    database: str,
    max_indexes: int = Query(5, ge=1, le=20, description="Maximum indexes to create"),
    admin_user: User = Depends(require_admin),
):
    """
    Optimize database indexes

    Automatically:
    - Creates recommended indexes
    - Removes unused indexes (with confirmation)
    - Updates index statistics
    """
    try:
        # Map database names to paths
        database_paths = {"main": "data/knowledge.db", "knowledge": "data/knowledge.db"}

        if database not in database_paths:
            raise HTTPException(
                status_code=404, detail=f"Database '{database}' not found"
            )

        database_path = database_paths[database]
        index_manager = await get_index_manager(database_path)

        # Apply optimization recommendations
        optimization_results = await index_manager.apply_recommendations(
            max_indexes=max_indexes
        )

        # Get updated statistics
        updated_stats = await index_manager.get_index_statistics()

        logger.info(
            "Database indexes optimized",
            database=database,
            created_count=len(optimization_results.get("created", [])),
            error_count=len(optimization_results.get("errors", [])),
        )

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "database": database,
                "optimization_results": optimization_results,
                "updated_statistics": updated_stats,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to optimize database indexes", database=database, error=str(e)
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to optimize indexes: {str(e)}"
        )


@router.get("/cache/stats")
@validate_request_security
async def get_cache_statistics(admin_user: User = Depends(require_admin)):
    """
    Get cache performance statistics

    Returns:
    - Cache hit/miss ratios
    - Cache size and utilization
    - Most cached queries
    - Performance impact metrics
    """
    try:
        database_optimizer = await get_database_optimizer()
        cache_stats = database_optimizer.query_cache.get_stats()

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "cache_statistics": cache_stats,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    except Exception as e:
        logger.error("Failed to get cache statistics", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve cache statistics: {str(e)}"
        )


@router.post("/cache/clear")
@validate_request_security
async def clear_query_cache(
    pattern: str | None = Query(
        None, description="Cache pattern to clear (default: all)"
    ),
    admin_user: User = Depends(require_admin),
):
    """
    Clear query cache

    Options:
    - Clear all cache entries
    - Clear entries matching specific pattern
    - Clear expired entries only
    """
    try:
        database_optimizer = await get_database_optimizer()

        # Clear cache based on pattern
        await database_optimizer.invalidate_caches(pattern)

        # Get updated stats
        updated_stats = database_optimizer.query_cache.get_stats()

        logger.info("Query cache cleared", pattern=pattern or "all")

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": f"Cache cleared (pattern: {pattern or 'all'})",
                "updated_statistics": updated_stats,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    except Exception as e:
        logger.error("Failed to clear cache", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@router.get("/queries/analysis")
@validate_request_security
async def get_query_analysis(admin_user: User = Depends(require_admin)):
    """
    Get query analysis and optimization report

    Returns:
    - Query pattern analysis
    - Performance bottlenecks
    - Optimization suggestions
    - Usage statistics
    """
    try:
        query_analyzer = await get_query_analyzer()
        analysis_report = query_analyzer.get_optimization_report()

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "query_analysis": analysis_report,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    except Exception as e:
        logger.error("Failed to get query analysis", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve query analysis: {str(e)}"
        )


@router.post("/configuration")
@validate_request_security
async def update_optimization_configuration(
    configuration: dict[str, Any], admin_user: User = Depends(require_admin)
):
    """
    Update optimization service configuration

    Configurable options:
    - Enable/disable optimization features
    - Adjust performance thresholds
    - Configure maintenance schedules
    - Set cache parameters
    """
    try:
        optimization_service = await get_optimization_service()

        # Validate configuration
        allowed_settings = {
            "optimization_enabled",
            "auto_maintenance_enabled",
            "performance_monitoring_enabled",
            "slow_query_threshold",
            "cache_ttl",
        }

        # Filter to allowed settings
        filtered_config = {
            k: v for k, v in configuration.items() if k in allowed_settings
        }

        if not filtered_config:
            raise HTTPException(
                status_code=400, detail="No valid configuration settings provided"
            )

        # Apply configuration
        success = await optimization_service.configure_optimization(filtered_config)

        if not success:
            raise HTTPException(
                status_code=500, detail="Failed to apply configuration changes"
            )

        logger.info(
            "Optimization configuration updated",
            settings=filtered_config,
            admin_user=admin_user.id,
        )

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Configuration updated successfully",
                "applied_settings": filtered_config,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update configuration", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to update configuration: {str(e)}"
        )


@router.get("/health")
async def get_optimization_health():
    """
    Get optimization service health status

    Public endpoint for health monitoring
    Returns basic status without sensitive information
    """
    try:
        optimization_service = await get_optimization_service()

        health_status = {
            "service_running": optimization_service.is_initialized,
            "optimization_enabled": optimization_service.optimization_enabled,
            "maintenance_active": (
                optimization_service.maintenance_task is not None
                and not optimization_service.maintenance_task.done()
            ),
            "monitoring_active": (
                optimization_service.performance_monitoring_task is not None
                and not optimization_service.performance_monitoring_task.done()
            ),
            "databases_managed": len(optimization_service.databases),
            "timestamp": datetime.utcnow().isoformat(),
        }

        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy" if health_status["service_running"] else "degraded",
                "details": health_status,
            },
        )

    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": "Service unavailable",
                "timestamp": datetime.utcnow().isoformat(),
            },
        )
