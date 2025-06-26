"""
Unit tests for health service - isolated from FastAPI app.
These tests focus purely on the business logic without app dependencies.
"""
import pytest
from unittest.mock import Mock, patch
from src.api.health.service import HealthService
from src.api.health.models import HealthStatus, HealthResponse


@pytest.mark.asyncio
class TestHealthServiceUnit:
    """Unit tests for HealthService business logic."""
    
    async def test_health_service_check_database_returns_boolean(self):
        """Database check should return boolean result."""
        # Arrange
        service = HealthService()
        
        # Act
        result = await service.check_database()
        
        # Assert
        assert isinstance(result, bool)
        assert result is True  # Should be True in test environment
        
    async def test_health_service_check_cache_returns_boolean(self):
        """Cache check should return boolean result."""
        # Arrange
        service = HealthService()
        
        # Act
        result = await service.check_cache()
        
        # Assert
        assert isinstance(result, bool)
        assert result is True  # Should be True in test environment
        
    async def test_health_service_check_filesystem_returns_boolean(self):
        """Filesystem check should return boolean result."""
        # Arrange
        service = HealthService()
        
        # Act
        result = await service.check_filesystem()
        
        # Assert
        assert isinstance(result, bool)
        
    def test_get_overall_status_healthy_when_all_pass(self):
        """Overall status should be healthy when all checks pass."""
        # Arrange
        service = HealthService()
        checks = {"database": True, "cache": True, "filesystem": True}
        
        # Act
        status = service.get_overall_status(checks)
        
        # Assert
        assert status == HealthStatus.HEALTHY
        
    def test_get_overall_status_unhealthy_when_any_fail(self):
        """Overall status should be unhealthy when any check fails."""
        # Arrange
        service = HealthService()
        checks = {"database": True, "cache": False, "filesystem": True}
        
        # Act
        status = service.get_overall_status(checks)
        
        # Assert
        assert status == HealthStatus.UNHEALTHY
        
    async def test_get_health_status_returns_health_response(self):
        """get_health_status should return HealthResponse object."""
        # Arrange
        service = HealthService()
        
        # Act
        result = await service.get_health_status()
        
        # Assert
        assert isinstance(result, HealthResponse)
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.UNHEALTHY]
        assert result.timestamp is not None
        assert result.version is not None
        assert isinstance(result.checks, dict)
        
    async def test_perform_all_checks_returns_dict(self):
        """_perform_all_checks should return dictionary of check results."""
        # Arrange
        service = HealthService()
        
        # Act
        result = await service._perform_all_checks()
        
        # Assert
        assert isinstance(result, dict)
        assert "database" in result
        assert "cache" in result
        assert "filesystem" in result
        assert all(isinstance(value, bool) for value in result.values())
        
    async def test_health_check_with_timeout_protection(self):
        """Health checks should complete within reasonable time."""
        # Arrange
        service = HealthService()
        import time
        start_time = time.time()
        
        # Act
        result = await service.get_health_status()
        end_time = time.time()
        
        # Assert
        duration = end_time - start_time
        assert duration < 5.0  # Should complete within 5 seconds
        assert isinstance(result, HealthResponse)
        
    @patch('src.api.health.service.HealthService.check_database')
    async def test_health_status_unhealthy_when_database_fails(self, mock_db_check):
        """Health status should be unhealthy when database check fails."""
        # Arrange
        mock_db_check.return_value = False
        service = HealthService()
        
        # Act
        result = await service.get_health_status()
        
        # Assert
        assert result.status == HealthStatus.UNHEALTHY
        assert result.checks["database"] is False


class TestHealthModels:
    """Test health check data models."""
    
    def test_health_response_model_validation(self):
        """HealthResponse model should validate correctly."""
        # Arrange & Act
        response = HealthResponse(
            status=HealthStatus.HEALTHY,
            timestamp="2024-01-01T00:00:00Z",
            version="1.0.0",
            checks={"database": True}
        )
        
        # Assert
        assert response.status == HealthStatus.HEALTHY
        assert response.timestamp == "2024-01-01T00:00:00Z"
        assert response.version == "1.0.0"
        assert response.checks == {"database": True}
        
    def test_health_status_enum_values(self):
        """HealthStatus enum should have correct values."""
        # Arrange & Act & Assert
        assert HealthStatus.HEALTHY == "healthy"
        assert HealthStatus.UNHEALTHY == "unhealthy"
        
    def test_health_response_json_serialization(self):
        """HealthResponse should serialize to JSON correctly."""
        # Arrange
        response = HealthResponse(
            status=HealthStatus.HEALTHY,
            timestamp="2024-01-01T00:00:00Z",
            version="1.0.0",
            checks={"database": True, "cache": True}
        )
        
        # Act
        json_data = response.dict()
        
        # Assert
        assert json_data["status"] == "healthy"
        assert json_data["timestamp"] == "2024-01-01T00:00:00Z"
        assert json_data["version"] == "1.0.0"
        assert json_data["checks"] == {"database": True, "cache": True}