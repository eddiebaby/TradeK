"""
Test suite for health check endpoint following London School TDD principles.
Tests are written before implementation to drive design.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import time
from src.api.health.models import HealthResponse, HealthStatus
from src.api.health.service import HealthService


class TestHealthEndpoint:
    """Test health check endpoint functionality."""
    
    def test_health_endpoint_returns_200_when_healthy(self, client: TestClient):
        """Health endpoint should return 200 when all dependencies are healthy."""
        # Arrange & Act
        response = client.get("/health")
        
        # Assert
        assert response.status_code == 200
        
    def test_health_endpoint_returns_json_response(self, client: TestClient):
        """Health endpoint should return JSON response."""
        # Arrange & Act
        response = client.get("/health")
        
        # Assert
        assert response.headers["content-type"] == "application/json"
        
    def test_health_response_contains_required_fields(self, client: TestClient):
        """Health response should contain status, timestamp, and version."""
        # Arrange & Act
        response = client.get("/health")
        data = response.json()
        
        # Assert
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "checks" in data
        
    def test_health_endpoint_returns_healthy_status_when_all_dependencies_ok(self, client: TestClient):
        """Should return 'healthy' status when all dependency checks pass."""
        # Arrange & Act
        response = client.get("/health")
        data = response.json()
        
        # Assert
        assert data["status"] == "healthy"
        
    def test_health_endpoint_returns_503_when_dependencies_fail(self, client: TestClient):
        """Should return 503 when critical dependencies fail."""
        # Arrange - Mock a failing dependency
        with patch('src.api.health.service.HealthService.check_database', return_value=False):
            # Act
            response = client.get("/health")
            
            # Assert
            assert response.status_code == 503
            
    def test_health_endpoint_includes_dependency_check_results(self, client: TestClient):
        """Health response should include individual dependency check results."""
        # Arrange & Act
        response = client.get("/health")
        data = response.json()
        
        # Assert
        assert "checks" in data
        assert isinstance(data["checks"], dict)
        
    def test_health_endpoint_response_time_under_10ms(self, client: TestClient):
        """Basic health check should respond in under 10ms."""
        # Arrange
        start_time = time.perf_counter()
        
        # Act
        response = client.get("/health")
        end_time = time.perf_counter()
        
        # Assert
        response_time_ms = (end_time - start_time) * 1000
        assert response_time_ms < 10
        assert response.status_code == 200


class TestHealthService:
    """Test health service business logic."""
    
    @pytest.mark.asyncio
    async def test_health_service_check_database_returns_boolean(self):
        """Database check should return boolean result."""
        # Arrange
        service = HealthService()
        
        # Act
        result = await service.check_database()
        
        # Assert
        assert isinstance(result, bool)
        
    @pytest.mark.asyncio
    async def test_health_service_check_cache_returns_boolean(self):
        """Cache check should return boolean result."""
        # Arrange
        service = HealthService()
        
        # Act
        result = await service.check_cache()
        
        # Assert
        assert isinstance(result, bool)
        
    def test_health_service_get_overall_status_healthy_when_all_pass(self):
        """Overall status should be healthy when all checks pass."""
        # Arrange
        service = HealthService()
        checks = {"database": True, "cache": True, "filesystem": True}
        
        # Act
        status = service.get_overall_status(checks)
        
        # Assert
        assert status == HealthStatus.HEALTHY
        
    def test_health_service_get_overall_status_unhealthy_when_any_fail(self):
        """Overall status should be unhealthy when any check fails."""
        # Arrange
        service = HealthService()
        checks = {"database": True, "cache": False, "filesystem": True}
        
        # Act
        status = service.get_overall_status(checks)
        
        # Assert
        assert status == HealthStatus.UNHEALTHY


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


# Fixtures
@pytest.fixture
def client():
    """FastAPI test client fixture."""
    from src.api.main import app
    return TestClient(app)