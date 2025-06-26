"""
Health check router for FastAPI application.
Provides HTTP endpoints for health monitoring.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from src.api.health.service import HealthService
from src.api.health.models import HealthResponse, HealthStatus


# Create router instance
router = APIRouter(
    prefix="/health",
    tags=["health"],
    responses={
        200: {"description": "Application is healthy"},
        503: {"description": "Application is unhealthy"}
    }
)


def get_health_service() -> HealthService:
    """Dependency injection for health service."""
    return HealthService()


@router.get(
    "",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check the health status of the application and its dependencies",
    responses={
        200: {
            "description": "Application is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "timestamp": "2024-01-01T12:00:00Z",
                        "version": "1.0.0",
                        "checks": {
                            "database": True,
                            "cache": True,
                            "filesystem": True
                        }
                    }
                }
            }
        },
        503: {
            "description": "Application is unhealthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "unhealthy",
                        "timestamp": "2024-01-01T12:00:00Z",
                        "version": "1.0.0",
                        "checks": {
                            "database": False,
                            "cache": True,
                            "filesystem": True
                        }
                    }
                }
            }
        }
    }
)
async def health_check(
    health_service: HealthService = Depends(get_health_service)
) -> JSONResponse:
    """
    Perform health check on application and dependencies.
    
    This endpoint checks the health of:
    - Database connectivity
    - Cache system
    - Filesystem access
    
    Returns:
        JSONResponse: Health status with individual check results
        
    Raises:
        HTTPException: 503 status if any critical dependency is unhealthy
    """
    # Get health status from service
    health_status = await health_service.get_health_status()
    
    # Return appropriate HTTP status code
    if health_status.status == HealthStatus.HEALTHY:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=health_status.dict()
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=health_status.dict()
        )


@router.get(
    "/live",
    summary="Liveness Probe",
    description="Simple liveness check for Kubernetes",
    responses={200: {"description": "Application is alive"}}
)
async def liveness_check():
    """
    Simple liveness check for Kubernetes liveness probe.
    
    This endpoint always returns 200 OK if the application is running.
    It doesn't check dependencies - only that the application can respond.
    
    Returns:
        dict: Simple alive status
    """
    return {"status": "alive"}


@router.get(
    "/ready",
    summary="Readiness Probe", 
    description="Readiness check for Kubernetes",
    responses={
        200: {"description": "Application is ready"},
        503: {"description": "Application is not ready"}
    }
)
async def readiness_check(
    health_service: HealthService = Depends(get_health_service)
):
    """
    Readiness check for Kubernetes readiness probe.
    
    This endpoint checks if the application is ready to serve traffic.
    It performs lightweight dependency checks.
    
    Returns:
        dict: Readiness status
        
    Raises:
        HTTPException: 503 if application is not ready
    """
    # Perform lightweight readiness checks
    health_status = await health_service.get_health_status()
    
    if health_status.status == HealthStatus.HEALTHY:
        return {"status": "ready"}
    else:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Application is not ready"
        )