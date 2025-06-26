"""
Health check data models using Pydantic.
Defines the structure for health check requests and responses.
"""
from enum import Enum
from typing import Dict, Any
from pydantic import BaseModel, Field


class HealthStatus(str, Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


class HealthResponse(BaseModel):
    """Health check response model."""
    status: HealthStatus = Field(
        description="Overall health status of the application"
    )
    timestamp: str = Field(
        description="ISO 8601 timestamp when health check was performed"
    )
    version: str = Field(
        description="Application version"
    )
    checks: Dict[str, Any] = Field(
        description="Individual dependency check results",
        default_factory=dict
    )
    
    class Config:
        """Pydantic model configuration."""
        json_encoders = {
            HealthStatus: lambda v: v.value
        }
        schema_extra = {
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