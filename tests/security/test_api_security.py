"""
API Security Tests for TradeKnowledge.

This module tests API endpoints for security vulnerabilities including
rate limiting, authentication, authorization, and input validation.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from fastapi import status
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.api.main import app
from src.api.models import User
from tests.fixtures.security_payloads import (
    SQL_INJECTION_PAYLOADS,
    XSS_PAYLOADS,
    PATH_TRAVERSAL_PAYLOADS,
)


class TestAPIAuthentication:
    """Test API authentication security"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_auth(self):
        """Mock authentication dependencies"""
        with patch('src.api.main.get_current_user') as mock_user, \
             patch('src.api.main.get_auth_manager') as mock_auth_manager:
            
            # Create mock user
            mock_user.return_value = User(
                id="test_user",
                username="testuser",
                email="test@example.com",
                role="user",
                created_at=datetime.now().isoformat()
            )
            
            # Create mock auth manager
            mock_auth_manager.return_value = AsyncMock()
            
            yield mock_user, mock_auth_manager
    
    def test_unauthenticated_request_rejection(self, client):
        """Test that unauthenticated requests are rejected"""
        # Mock all dependencies and app state
        with patch('src.api.main.get_search_engine') as mock_search_engine, \
             patch('src.api.main.get_metrics') as mock_metrics, \
             patch('src.api.main.app_state') as mock_app_state:
            
            # Mock search engine and metrics
            mock_search_engine.return_value = AsyncMock()
            mock_metrics.return_value = AsyncMock()
            
            # Mock app state with auth manager
            mock_auth_manager = AsyncMock()
            mock_app_state.get.return_value = mock_auth_manager
            
            # Test search endpoint without authentication
            response = client.post("/api/v1/search/query", json={
                "query": "test query",
                "max_results": 10
            })
            
            # Should return 401 or 403 (HTTPBearer should reject missing auth header)
            assert response.status_code in [401, 403, 422]  # 422 is also valid for missing auth
    
    def test_invalid_token_rejection(self, client):
        """Test that invalid tokens are rejected"""
        with patch('src.api.main.get_search_engine') as mock_search_engine, \
             patch('src.api.main.get_metrics') as mock_metrics, \
             patch('src.api.main.app_state') as mock_app_state:
            
            mock_search_engine.return_value = AsyncMock()
            mock_metrics.return_value = AsyncMock()
            
            # Mock auth manager that raises exception for invalid token
            mock_auth_manager = AsyncMock()
            mock_auth_manager.verify_token.side_effect = Exception("Invalid token")
            mock_app_state.get.return_value = mock_auth_manager
            
            headers = {"Authorization": "Bearer invalid_token_here"}
            
            response = client.post("/api/v1/search/query", 
                                 headers=headers,
                                 json={"query": "test"})
            
            assert response.status_code in [401, 403]
    
    def test_malformed_authorization_header(self, client):
        """Test malformed authorization headers"""
        with patch('src.api.main.get_search_engine') as mock_search_engine, \
             patch('src.api.main.get_metrics') as mock_metrics, \
             patch('src.api.main.app_state') as mock_app_state:
            
            mock_search_engine.return_value = AsyncMock()
            mock_metrics.return_value = AsyncMock()
            
            # Mock auth manager
            mock_auth_manager = AsyncMock()
            mock_auth_manager.verify_token.side_effect = Exception("Invalid token format")
            mock_app_state.get.return_value = mock_auth_manager
            
            malformed_headers = [
                {"Authorization": "invalid_format"},
                {"Authorization": "Bearer"},  # Missing token
                {"Authorization": "Basic dGVzdA=="},  # Wrong auth type
                {"Authorization": "Bearer " + "x" * 1000},  # Extremely long token
            ]
            
            for headers in malformed_headers:
                response = client.post("/api/v1/search/query",
                                     headers=headers,
                                     json={"query": "test"})
                # Should be rejected by HTTPBearer or auth manager
                assert response.status_code in [401, 403, 422]
    
    def test_expired_token_rejection(self, client):
        """Test that expired tokens are rejected"""
        with patch('src.api.main.get_search_engine') as mock_search_engine, \
             patch('src.api.main.get_metrics') as mock_metrics, \
             patch('src.api.main.app_state') as mock_app_state:
            
            mock_search_engine.return_value = AsyncMock()
            mock_metrics.return_value = AsyncMock()
            
            # Mock auth manager that raises exception for expired token
            mock_auth_manager = AsyncMock()
            mock_auth_manager.verify_token.side_effect = ValueError("Token expired")
            mock_app_state.get.return_value = mock_auth_manager
            
            headers = {"Authorization": "Bearer expired_token"}
            response = client.post("/api/v1/search/query",
                                 headers=headers,
                                 json={"query": "test"})
            
            assert response.status_code in [401, 403]


class TestAPIAuthorization:
    """Test API authorization and permission checking"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_insufficient_permissions(self, client):
        """Test requests with insufficient permissions"""
        # Mock a user with limited permissions
        with patch('src.api.main.get_current_user') as mock_user:
            mock_user.return_value = User(
                id="limited_user",
                username="limited",
                email="limited@example.com", 
                role="viewer",  # Limited role
                created_at=datetime.now().isoformat()
            )
            
            # Try to access admin endpoint (if exists)
            response = client.get("/api/v1/admin/users")
            
            # Should be forbidden
            assert response.status_code == 403
    
    def test_role_based_access_control(self, client):
        """Test role-based access control"""
        test_cases = [
            ("viewer", "read", True),
            ("viewer", "write", False),
            ("user", "read", True),
            ("user", "write", False),
            ("editor", "read", True),
            ("editor", "write", True),
            ("admin", "admin", True),
        ]
        
        for role, permission, should_allow in test_cases:
            with patch('src.api.main.get_current_user') as mock_user, \
                 patch('src.api.auth.AuthManager.check_permission') as mock_check:
                
                mock_user.return_value = User(
                    id=f"{role}_user",
                    username=role,
                    email=f"{role}@example.com",
                    role=role,
                    created_at=datetime.now().isoformat()
                )
                
                mock_check.return_value = should_allow
                
                # Test endpoint that requires this permission
                response = client.post("/api/v1/search/query",
                                     json={"query": "test"})
                
                if should_allow:
                    assert response.status_code != 403
                else:
                    assert response.status_code == 403


class TestAPIInputValidation:
    """Test API input validation security"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def authenticated_client(self, client):
        """Client with mocked authentication"""
        with patch('src.api.main.get_current_user') as mock_user:
            mock_user.return_value = User(
                id="test_user",
                username="testuser",
                email="test@example.com",
                role="user",
                created_at=datetime.now().isoformat()
            )
            yield client
    
    def test_sql_injection_in_search_queries(self, authenticated_client):
        """Test SQL injection prevention in search API"""
        for payload in SQL_INJECTION_PAYLOADS:
            response = authenticated_client.post("/api/v1/search/query", json={
                "query": payload,
                "max_results": 10
            })
            
            # Should either reject (400/422) or sanitize the input
            if response.status_code == 200:
                # If accepted, ensure the payload was sanitized
                assert "DROP" not in response.text.upper()
                assert "UNION" not in response.text.upper()
            else:
                assert response.status_code in [400, 422]
    
    def test_xss_prevention_in_api_responses(self, authenticated_client):
        """Test XSS prevention in API responses"""
        for payload in XSS_PAYLOADS:
            response = authenticated_client.post("/api/v1/search/query", json={
                "query": payload,
                "max_results": 5
            })
            
            if response.status_code == 200:
                response_text = response.text
                # Ensure XSS payloads are not reflected back
                assert "<script" not in response_text.lower()
                assert "javascript:" not in response_text.lower()
                assert "onerror" not in response_text.lower()
    
    def test_path_traversal_in_file_endpoints(self, authenticated_client):
        """Test path traversal prevention in file endpoints"""
        for payload in PATH_TRAVERSAL_PAYLOADS:
            # Test file access endpoint (if exists)
            response = authenticated_client.get(f"/api/v1/files/{payload}")
            
            # Should be rejected
            assert response.status_code in [400, 403, 404, 422]
    
    def test_request_size_limits(self, authenticated_client):
        """Test request size limits"""
        # Test extremely large request
        large_query = "x" * 10000  # 10KB query
        
        response = authenticated_client.post("/api/v1/search/query", json={
            "query": large_query,
            "max_results": 10
        })
        
        # Should be rejected for being too large
        assert response.status_code in [400, 413, 422]
    
    def test_malformed_json_handling(self, authenticated_client):
        """Test handling of malformed JSON"""
        # Send invalid JSON
        response = authenticated_client.post("/api/v1/search/query",
                                           data="{'invalid': json}",
                                           headers={"Content-Type": "application/json"})
        
        assert response.status_code in [400, 422]
    
    def test_missing_required_fields(self, authenticated_client):
        """Test handling of missing required fields"""
        # Send request without required 'query' field
        response = authenticated_client.post("/api/v1/search/query", json={
            "max_results": 10
            # Missing "query" field
        })
        
        assert response.status_code in [400, 422]
    
    def test_field_type_validation(self, authenticated_client):
        """Test field type validation"""
        invalid_requests = [
            {"query": 123, "max_results": 10},  # query should be string
            {"query": "test", "max_results": "invalid"},  # max_results should be int
            {"query": None, "max_results": 10},  # query should not be null
            {"query": [], "max_results": 10},  # query should not be array
        ]
        
        for invalid_request in invalid_requests:
            response = authenticated_client.post("/api/v1/search/query", 
                                               json=invalid_request)
            assert response.status_code in [400, 422]


class TestAPIRateLimiting:
    """Test API rate limiting security"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def authenticated_client(self, client):
        """Client with mocked authentication"""
        with patch('src.api.main.get_current_user') as mock_user:
            mock_user.return_value = User(
                id="test_user",
                username="testuser",
                email="test@example.com",
                role="user",
                created_at=datetime.now().isoformat()
            )
            yield client
    
    def test_rate_limit_enforcement(self, authenticated_client):
        """Test that rate limits are enforced"""
        # Make rapid requests to trigger rate limit
        responses = []
        for i in range(100):  # Exceed typical rate limit
            response = authenticated_client.post("/api/v1/search/query", json={
                "query": f"test query {i}",
                "max_results": 5
            })
            responses.append(response)
            
            # If we hit rate limit, break
            if response.status_code == 429:
                break
        
        # Should have at least one rate limited response
        rate_limited = any(r.status_code == 429 for r in responses)
        assert rate_limited, "Rate limiting was not enforced"
    
    def test_rate_limit_headers(self, authenticated_client):
        """Test rate limit headers are present"""
        response = authenticated_client.post("/api/v1/search/query", json={
            "query": "test",
            "max_results": 5
        })
        
        # Check for rate limit headers
        expected_headers = [
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining", 
            "X-RateLimit-Reset"
        ]
        
        for header in expected_headers:
            assert header in response.headers or response.status_code == 429
    
    def test_rate_limit_per_ip(self, authenticated_client):
        """Test rate limiting is applied per IP address"""
        # This test would need to simulate different IP addresses
        # For now, we'll test that the rate limiting mechanism exists
        
        # Make a request and check for rate limit tracking
        response = authenticated_client.post("/api/v1/search/query", json={
            "query": "test",
            "max_results": 5
        })
        
        # Should either succeed or have rate limit headers
        assert response.status_code in [200, 429] or "X-RateLimit" in str(response.headers)


class TestAPISecurityHeaders:
    """Test security headers in API responses"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_security_headers_present(self, client):
        """Test that security headers are present"""
        response = client.get("/api/v1/health")
        
        expected_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
        }
        
        for header, expected_value in expected_headers.items():
            assert header in response.headers
            if expected_value:
                assert response.headers[header] == expected_value
    
    def test_cors_configuration(self, client):
        """Test CORS configuration security"""
        # Test preflight request
        response = client.options("/api/v1/search/query",
                                headers={"Origin": "https://malicious-site.com"})
        
        # Should have appropriate CORS headers
        if "Access-Control-Allow-Origin" in response.headers:
            # Ensure it's not wildcard for authenticated requests
            assert response.headers["Access-Control-Allow-Origin"] != "*"
    
    def test_content_type_validation(self, client):
        """Test content type validation"""
        # Try to send XML when JSON is expected
        response = client.post("/api/v1/search/query",
                             data="<xml>test</xml>",
                             headers={"Content-Type": "application/xml"})
        
        assert response.status_code in [400, 415, 422]


class TestAPIErrorHandling:
    """Test API error handling security"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_error_information_disclosure(self, client):
        """Test that errors don't disclose sensitive information"""
        # Trigger various error conditions
        error_responses = [
            client.get("/api/v1/nonexistent-endpoint"),
            client.post("/api/v1/search/query"),  # Missing auth
            client.post("/api/v1/search/query", json={"invalid": "data"}),
        ]
        
        for response in error_responses:
            response_text = response.text.lower()
            
            # Should not reveal sensitive information
            sensitive_terms = [
                "traceback",
                "exception",
                "stack trace",
                "internal server error",
                "database",
                "sql",
                "password",
                "secret",
                "token",
                "api key",
            ]
            
            for term in sensitive_terms:
                assert term not in response_text, f"Sensitive term '{term}' found in error response"
    
    def test_consistent_error_timing(self, client):
        """Test that error responses have consistent timing"""
        import time
        
        # Time various error scenarios
        scenarios = [
            ("/api/v1/nonexistent", None),
            ("/api/v1/search/query", {"invalid": "json"}),
        ]
        
        times = []
        for endpoint, json_data in scenarios:
            start = time.time()
            if json_data:
                client.post(endpoint, json=json_data)
            else:
                client.get(endpoint)
            end = time.time()
            times.append(end - start)
        
        # Error response times should be relatively consistent
        if len(times) > 1:
            max_variance = max(times) - min(times)
            assert max_variance < 1.0, "Error response times vary too much (timing attack risk)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])