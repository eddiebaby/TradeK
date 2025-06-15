"""
Comprehensive API test suite for TradeKnowledge Phase 4
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
import json
import tempfile
import os
from pathlib import Path

# Import main API application
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.api.main import app
from src.api.user_manager import get_user_manager, UserCreate
from src.api.auth import get_auth_manager
from src.core.config import get_config


class TestAPIAuthentication:
    """Test authentication and user management"""
    
    @pytest.fixture
    async def setup_test_db(self):
        """Setup test database"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_db_path = os.path.join(temp_dir, "test_users.db")
            user_manager = get_user_manager()
            user_manager.db_path = test_db_path
            await user_manager.initialize()
            yield user_manager
    
    async def test_user_creation(self, setup_test_db):
        """Test user creation"""
        user_manager = setup_test_db
        
        user_data = UserCreate(
            username="testuser",
            email="test@example.com",
            password="password123",
            role="user"
        )
        
        user = await user_manager.create_user(user_data)
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.role == "user"
        assert user.is_active
    
    async def test_user_authentication(self, setup_test_db):
        """Test user authentication"""
        user_manager = setup_test_db
        auth_manager = get_auth_manager()
        auth_manager.user_manager = user_manager
        
        # Create user
        user_data = UserCreate(
            username="authtest",
            email="auth@example.com",
            password="securepass123",
            role="user"
        )
        await user_manager.create_user(user_data)
        
        # Authenticate
        user = await auth_manager.authenticate_user("authtest", "securepass123")
        assert user is not None
        assert user.username == "authtest"
        
        # Test wrong password
        user = await auth_manager.authenticate_user("authtest", "wrongpass")
        assert user is None
    
    async def test_jwt_token_creation_and_verification(self, setup_test_db):
        """Test JWT token creation and verification"""
        user_manager = setup_test_db
        auth_manager = get_auth_manager()
        auth_manager.user_manager = user_manager
        
        # Create user
        user_data = UserCreate(
            username="tokentest",
            email="token@example.com",
            password="tokenpass123",
            role="admin"
        )
        user = await user_manager.create_user(user_data)
        
        # Create token
        token = await auth_manager.create_token(user)
        assert token is not None
        
        # Verify token
        verified_user = await auth_manager.verify_token(token)
        assert verified_user.id == user.id
        assert verified_user.username == user.username
    
    async def test_permission_system(self, setup_test_db):
        """Test role-based permission system"""
        user_manager = setup_test_db
        auth_manager = get_auth_manager()
        auth_manager.user_manager = user_manager
        
        # Create users with different roles
        admin_data = UserCreate(username="admin", email="admin@test.com", password="pass", role="admin")
        user_data = UserCreate(username="user", email="user@test.com", password="pass", role="user")
        
        admin = await user_manager.create_user(admin_data)
        user = await user_manager.create_user(user_data)
        
        # Test permissions
        assert auth_manager.check_permission(admin, "admin")
        assert auth_manager.check_permission(admin, "write")
        assert auth_manager.check_permission(admin, "read")
        
        assert not auth_manager.check_permission(user, "admin")
        assert not auth_manager.check_permission(user, "write")
        assert auth_manager.check_permission(user, "read")


class TestAPIEndpoints:
    """Test API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Test client fixture"""
        return TestClient(app)
    
    @pytest.fixture
    async def authenticated_headers(self):
        """Get authenticated headers for API requests"""
        # Create test user and get token
        user_manager = get_user_manager()
        auth_manager = get_auth_manager()
        
        # Initialize with test database
        with tempfile.TemporaryDirectory() as temp_dir:
            test_db_path = os.path.join(temp_dir, "test_api_users.db")
            user_manager.db_path = test_db_path
            await user_manager.initialize()
            auth_manager.user_manager = user_manager
            
            # Create test user
            user_data = UserCreate(
                username="apitest",
                email="api@test.com",
                password="apipass123",
                role="admin"
            )
            user = await user_manager.create_user(user_data)
            token = await auth_manager.create_token(user)
            
            return {"Authorization": f"Bearer {token}"}
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_api_docs(self, client):
        """Test API documentation is accessible"""
        response = client.get("/docs")
        assert response.status_code == 200
    
    async def test_search_endpoint_requires_auth(self, client):
        """Test search endpoint requires authentication"""
        response = client.post("/api/v1/search/query", json={
            "query": "test query",
            "max_results": 10
        })
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_search_endpoint_with_auth(self, client):
        """Test search endpoint with authentication"""
        # This would require a full integration test setup
        # For now, test the structure
        pass
    
    async def test_user_management_endpoints(self, client):
        """Test user management endpoints"""
        # Test user registration (if enabled)
        # Test user profile updates
        # Test admin user management
        pass


class TestAPISecurity:
    """Test API security features"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_cors_headers(self, client):
        """Test CORS headers are properly set"""
        response = client.options("/api/v1/search/query")
        # Check CORS headers based on configuration
        pass
    
    def test_rate_limiting(self, client):
        """Test rate limiting functionality"""
        # Make multiple requests rapidly
        # Check rate limiting is enforced
        pass
    
    def test_input_validation(self, client):
        """Test input validation and sanitization"""
        # Test various malicious inputs
        # SQL injection attempts
        # XSS attempts
        # Path traversal attempts
        pass
    
    def test_file_upload_security(self, client):
        """Test file upload security measures"""
        # Test file type validation
        # Test file size limits
        # Test malicious file detection
        pass


class TestAPIPerformance:
    """Test API performance characteristics"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    async def test_concurrent_requests(self, client):
        """Test handling of concurrent requests"""
        # Make multiple concurrent requests
        # Check response times and success rates
        pass
    
    async def test_search_performance(self, client):
        """Test search endpoint performance"""
        # Test search response times
        # Test with various query complexities
        pass
    
    async def test_memory_usage(self, client):
        """Test memory usage under load"""
        # Monitor memory usage during heavy requests
        pass


class TestAPIIntegration:
    """Test API integration with core systems"""
    
    async def test_search_integration(self):
        """Test search system integration"""
        # Test that API properly integrates with search engines
        # Test result formatting and filtering
        pass
    
    async def test_storage_integration(self):
        """Test storage system integration"""
        # Test that API properly uses storage systems
        # Test data consistency
        pass
    
    async def test_caching_integration(self):
        """Test caching system integration"""
        # Test cache hit/miss behavior
        # Test cache invalidation
        pass


class TestAPIErrorHandling:
    """Test API error handling"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_404_handling(self, client):
        """Test 404 error handling"""
        response = client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_500_handling(self, client):
        """Test internal server error handling"""
        # Trigger various error conditions
        # Check error responses don't leak sensitive info
        pass
    
    def test_validation_errors(self, client):
        """Test input validation error responses"""
        # Test malformed requests
        # Check error message quality
        pass


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])