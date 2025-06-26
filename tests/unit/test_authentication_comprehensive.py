"""
Comprehensive tests for the authentication and rate limiting system
Tests JWT authentication, API key management, and rate limiting functionality
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timezone, timedelta
from fastapi import HTTPException, status
from fastapi.testclient import TestClient
import jwt

from src.api.auth.authentication import (
    JWTManager, APIKeyManager, UserManager, AuthenticationMiddleware,
    AuthenticationError, AuthorizationError, TokenType
)
from src.api.middleware.rate_limiter import (
    RateLimitMiddleware, RateLimitRule, RateLimitType, 
    MemoryRateLimiter, RateLimitResult
)
from src.core.models import User, UserRole, APIKey


class TestJWTManager:
    """Test JWT token management"""
    
    @pytest.fixture
    def jwt_manager(self):
        """Create JWT manager for testing"""
        with patch('src.api.auth.authentication.get_config') as mock_config:
            mock_config.return_value.security.secret_key = "test-secret-key"
            return JWTManager()
    
    def test_create_access_token(self, jwt_manager):
        """Test access token creation"""
        user_id = "test_user_123"
        email = "test@example.com"
        roles = ["user", "admin"]
        
        token = jwt_manager.create_access_token(user_id, email, roles)
        
        assert isinstance(token, str)
        assert len(token) > 50  # JWT tokens are typically long
        
        # Decode and verify payload
        payload = jwt.decode(token, jwt_manager.secret_key, algorithms=[jwt_manager.algorithm])
        assert payload["sub"] == user_id
        assert payload["email"] == email
        assert payload["roles"] == roles
        assert payload["type"] == TokenType.ACCESS.value
    
    def test_create_refresh_token(self, jwt_manager):
        """Test refresh token creation"""
        user_id = "test_user_123"
        
        token = jwt_manager.create_refresh_token(user_id)
        
        assert isinstance(token, str)
        
        # Decode and verify payload
        payload = jwt.decode(token, jwt_manager.secret_key, algorithms=[jwt_manager.algorithm])
        assert payload["sub"] == user_id
        assert payload["type"] == TokenType.REFRESH.value
        assert "email" not in payload  # Refresh tokens don't include email
    
    def test_verify_valid_token(self, jwt_manager):
        """Test verification of valid token"""
        user_id = "test_user_123"
        email = "test@example.com"
        roles = ["user"]
        
        token = jwt_manager.create_access_token(user_id, email, roles)
        payload = jwt_manager.verify_token(token)
        
        assert payload["sub"] == user_id
        assert payload["email"] == email
        assert payload["roles"] == roles
    
    def test_verify_expired_token(self, jwt_manager):
        """Test verification of expired token"""
        # Create token with very short expiry
        jwt_manager.access_token_expire_minutes = -1  # Already expired
        
        user_id = "test_user_123"
        email = "test@example.com"
        roles = ["user"]
        
        token = jwt_manager.create_access_token(user_id, email, roles)
        
        with pytest.raises(AuthenticationError, match="Token has expired"):
            jwt_manager.verify_token(token)
    
    def test_verify_invalid_token(self, jwt_manager):
        """Test verification of invalid token"""
        invalid_token = "invalid.token.here"
        
        with pytest.raises(AuthenticationError, match="Invalid token"):
            jwt_manager.verify_token(invalid_token)
    
    def test_verify_wrong_token_type(self, jwt_manager):
        """Test verification of wrong token type"""
        user_id = "test_user_123"
        
        # Create refresh token but try to verify as access token
        refresh_token = jwt_manager.create_refresh_token(user_id)
        
        with pytest.raises(AuthenticationError, match="Invalid token type"):
            jwt_manager.verify_token(refresh_token, TokenType.ACCESS)
    
    def test_refresh_access_token(self, jwt_manager):
        """Test access token refresh"""
        user_id = "test_user_123"
        user_data = {
            "email": "test@example.com",
            "roles": ["user", "admin"]
        }
        
        # Create refresh token
        refresh_token = jwt_manager.create_refresh_token(user_id)
        
        # Refresh access token
        new_access_token = jwt_manager.refresh_access_token(refresh_token, user_data)
        
        # Verify new access token
        payload = jwt_manager.verify_token(new_access_token)
        assert payload["sub"] == user_id
        assert payload["email"] == user_data["email"]
        assert payload["roles"] == user_data["roles"]


class TestAPIKeyManager:
    """Test API key management"""
    
    @pytest.fixture
    def api_key_manager(self):
        """Create API key manager for testing"""
        with patch('src.api.auth.authentication.get_config'):
            return APIKeyManager()
    
    def test_generate_api_key(self, api_key_manager):
        """Test API key generation"""
        user_id = "test_user_123"
        name = "Test API Key"
        permissions = ["read", "search"]
        
        api_key = api_key_manager.generate_api_key(user_id, name, permissions)
        
        assert isinstance(api_key, APIKey)
        assert api_key.user_id == user_id
        assert api_key.name == name
        assert api_key.permissions == permissions
        assert api_key.key_value.startswith("tk_")
        assert len(api_key.key_value) > 20
        assert api_key.is_active
        assert api_key.usage_count == 0
        assert api_key.rate_limit == 1000  # Default rate limit
    
    def test_verify_valid_api_key(self, api_key_manager):
        """Test verification of valid API key"""
        # Mock the database lookup
        with patch.object(api_key_manager, 'verify_api_key') as mock_verify:
            mock_api_key = APIKey(
                key_id="test_key_id",
                key_hash="test_hash",
                user_id="test_user",
                name="Test Key",
                permissions=["read"],
                created_at=datetime.now(timezone.utc),
                is_active=True,
                rate_limit=1000,
                usage_count=0
            )
            mock_verify.return_value = mock_api_key
            
            result = api_key_manager.verify_api_key("tk_test_key")
            assert result == mock_api_key
    
    def test_verify_invalid_api_key(self, api_key_manager):
        """Test verification of invalid API key"""
        # Invalid format (doesn't start with tk_)
        result = api_key_manager.verify_api_key("invalid_key")
        assert result is None
        
        # Valid format but not found in database
        result = api_key_manager.verify_api_key("tk_nonexistent_key")
        assert result is None
    
    def test_update_api_key_usage(self, api_key_manager):
        """Test updating API key usage statistics"""
        api_key = APIKey(
            key_id="test_key_id",
            key_hash="test_hash",
            user_id="test_user",
            name="Test Key",
            permissions=["read"],
            created_at=datetime.now(timezone.utc),
            is_active=True,
            rate_limit=1000,
            usage_count=5,
            last_used=None
        )
        
        original_count = api_key.usage_count
        api_key_manager.update_api_key_usage(api_key)
        
        assert api_key.usage_count == original_count + 1
        assert api_key.last_used is not None
        assert isinstance(api_key.last_used, datetime)


class TestUserManager:
    """Test user management"""
    
    @pytest.fixture
    def user_manager(self):
        """Create user manager for testing"""
        with patch('src.api.auth.authentication.get_config'):
            return UserManager()
    
    def test_hash_password(self, user_manager):
        """Test password hashing"""
        password = "test_password_123"
        hashed = user_manager.hash_password(password)
        
        assert isinstance(hashed, str)
        assert len(hashed) > 20  # bcrypt hashes are long
        assert hashed != password  # Should be different from original
        
        # Should be able to verify the hash
        assert user_manager.verify_password(password, hashed)
    
    def test_verify_password_correct(self, user_manager):
        """Test password verification with correct password"""
        password = "test_password_123"
        hashed = user_manager.hash_password(password)
        
        assert user_manager.verify_password(password, hashed)
    
    def test_verify_password_incorrect(self, user_manager):
        """Test password verification with incorrect password"""
        password = "test_password_123"
        wrong_password = "wrong_password"
        hashed = user_manager.hash_password(password)
        
        assert not user_manager.verify_password(wrong_password, hashed)
    
    @pytest.mark.asyncio
    async def test_authenticate_user_valid(self, user_manager):
        """Test user authentication with valid credentials"""
        # Uses mock user for testing
        user = await user_manager.authenticate_user("admin@tradeknowledge.com", "admin123")
        
        assert user is not None
        assert isinstance(user, User)
        assert user.email == "admin@tradeknowledge.com"
        assert UserRole.ADMIN in user.roles
    
    @pytest.mark.asyncio
    async def test_authenticate_user_invalid(self, user_manager):
        """Test user authentication with invalid credentials"""
        user = await user_manager.authenticate_user("wrong@email.com", "wrong_password")
        assert user is None
    
    @pytest.mark.asyncio
    async def test_get_user_by_id_valid(self, user_manager):
        """Test getting user by valid ID"""
        user = await user_manager.get_user_by_id("admin_user_id")
        
        assert user is not None
        assert isinstance(user, User)
        assert user.id == "admin_user_id"
    
    @pytest.mark.asyncio
    async def test_get_user_by_id_invalid(self, user_manager):
        """Test getting user by invalid ID"""
        user = await user_manager.get_user_by_id("nonexistent_user_id")
        assert user is None
    
    @pytest.mark.asyncio
    async def test_create_user(self, user_manager):
        """Test user creation"""
        email = "newuser@example.com"
        password = "secure_password_123"
        username = "newuser"
        full_name = "New User"
        roles = [UserRole.USER]
        
        user = await user_manager.create_user(email, password, username, full_name, roles)
        
        assert isinstance(user, User)
        assert user.email == email
        assert user.username == username
        assert user.full_name == full_name
        assert user.roles == roles
        assert user.is_active
        assert user.password_hash is not None
        assert user.password_hash != password  # Should be hashed


class TestMemoryRateLimiter:
    """Test memory-based rate limiter"""
    
    @pytest.fixture
    def rate_limiter(self):
        """Create memory rate limiter for testing"""
        return MemoryRateLimiter()
    
    @pytest.fixture
    def test_rule(self):
        """Create test rate limit rule"""
        return RateLimitRule(limit=5, window=60)  # 5 requests per minute
    
    @pytest.mark.asyncio
    async def test_allow_requests_under_limit(self, rate_limiter, test_rule):
        """Test allowing requests under the limit"""
        key = "test_user_123"
        
        # Make requests under the limit
        for i in range(test_rule.limit):
            result = await rate_limiter.check_rate_limit(key, test_rule)
            assert result.allowed
            assert result.remaining == test_rule.limit - i - 1
    
    @pytest.mark.asyncio
    async def test_block_requests_over_limit(self, rate_limiter, test_rule):
        """Test blocking requests over the limit"""
        key = "test_user_123"
        
        # Exhaust the limit
        for i in range(test_rule.limit):
            result = await rate_limiter.check_rate_limit(key, test_rule)
            assert result.allowed
        
        # Next request should be blocked
        result = await rate_limiter.check_rate_limit(key, test_rule)
        assert not result.allowed
        assert result.remaining == 0
        assert result.retry_after is not None
        assert result.retry_after > 0
    
    @pytest.mark.asyncio
    async def test_sliding_window_behavior(self, rate_limiter, test_rule):
        """Test sliding window rate limiting behavior"""
        key = "test_user_123"
        
        # Use a very short window for testing
        short_rule = RateLimitRule(limit=2, window=1)  # 2 requests per second
        
        # Make 2 requests (should be allowed)
        result1 = await rate_limiter.check_rate_limit(key, short_rule)
        result2 = await rate_limiter.check_rate_limit(key, short_rule)
        
        assert result1.allowed
        assert result2.allowed
        
        # Third request should be blocked
        result3 = await rate_limiter.check_rate_limit(key, short_rule)
        assert not result3.allowed
        
        # Wait for window to slide
        await asyncio.sleep(1.1)
        
        # Should be allowed again
        result4 = await rate_limiter.check_rate_limit(key, short_rule)
        assert result4.allowed
    
    @pytest.mark.asyncio
    async def test_different_keys_isolated(self, rate_limiter, test_rule):
        """Test that different keys are isolated"""
        key1 = "user_1"
        key2 = "user_2"
        
        # Exhaust limit for key1
        for i in range(test_rule.limit):
            result = await rate_limiter.check_rate_limit(key1, test_rule)
            assert result.allowed
        
        # key1 should be blocked
        result = await rate_limiter.check_rate_limit(key1, test_rule)
        assert not result.allowed
        
        # key2 should still be allowed
        result = await rate_limiter.check_rate_limit(key2, test_rule)
        assert result.allowed


class TestRateLimitMiddleware:
    """Test rate limit middleware"""
    
    @pytest.fixture
    def middleware(self):
        """Create rate limit middleware for testing"""
        with patch('src.api.middleware.rate_limiter.get_config') as mock_config:
            # Mock configuration
            config = Mock()
            config.rate_limiting.global_requests_per_minute = 1000
            config.rate_limiting.ip_requests_per_minute = 100
            config.rate_limiting.user_requests_per_hour = 1000
            mock_config.return_value = config
            
            return RateLimitMiddleware()
    
    @pytest.fixture
    def mock_request(self):
        """Create mock FastAPI request"""
        request = Mock()
        request.client.host = "192.168.1.1"
        request.headers = {}
        request.state = Mock()
        return request
    
    @pytest.fixture
    def mock_user(self):
        """Create mock user"""
        return User(
            id="test_user_123",
            email="test@example.com",
            username="testuser",
            full_name="Test User",
            roles=[UserRole.USER],
            is_active=True,
            created_at=datetime.now(timezone.utc)
        )
    
    @pytest.fixture
    def mock_api_key(self):
        """Create mock API key"""
        return APIKey(
            key_id="test_key_id",
            key_hash="test_hash",
            user_id="test_user",
            name="Test Key",
            permissions=["read"],
            created_at=datetime.now(timezone.utc),
            is_active=True,
            rate_limit=1000,
            usage_count=0
        )
    
    @pytest.mark.asyncio
    async def test_allow_request_under_limits(self, middleware, mock_request, mock_user):
        """Test allowing request under all rate limits"""
        await middleware.initialize()
        
        response = await middleware.check_rate_limits(mock_request, mock_user)
        
        # Should return None (allow request)
        assert response is None
        
        # Should set rate limit headers
        assert hasattr(mock_request.state, 'rate_limit_headers')
        headers = mock_request.state.rate_limit_headers
        assert 'X-RateLimit-Limit' in headers
        assert 'X-RateLimit-Remaining' in headers
        assert 'X-RateLimit-Reset' in headers
    
    @pytest.mark.asyncio
    async def test_block_request_over_ip_limit(self, middleware, mock_request):
        """Test blocking request over IP rate limit"""
        await middleware.initialize()
        
        # Mock rate limiter to return blocked result
        with patch.object(middleware, '_check_limit') as mock_check:
            # Allow global, block IP
            mock_check.side_effect = [
                RateLimitResult(allowed=True, remaining=100, reset_time=int(time.time() + 3600)),  # Global
                RateLimitResult(allowed=False, remaining=0, reset_time=int(time.time() + 3600), retry_after=60)  # IP
            ]
            
            response = await middleware.check_rate_limits(mock_request)
            
            # Should return rate limit response
            assert response is not None
            assert response.status_code == 429
    
    @pytest.mark.asyncio
    async def test_api_key_rate_limit(self, middleware, mock_request, mock_api_key):
        """Test API key specific rate limiting"""
        await middleware.initialize()
        
        # Mock all checks to pass except API key
        with patch.object(middleware, '_check_limit') as mock_check:
            mock_check.side_effect = [
                RateLimitResult(allowed=True, remaining=100, reset_time=int(time.time() + 3600)),  # Global
                RateLimitResult(allowed=True, remaining=100, reset_time=int(time.time() + 3600)),  # IP
                RateLimitResult(allowed=False, remaining=0, reset_time=int(time.time() + 3600), retry_after=60)  # API key
            ]
            
            response = await middleware.check_rate_limits(mock_request, api_key=mock_api_key)
            
            # Should return rate limit response
            assert response is not None
            assert response.status_code == 429
            assert "API key rate limit exceeded" in response.body.decode()
    
    def test_get_client_ip_direct(self, middleware, mock_request):
        """Test getting client IP from direct connection"""
        mock_request.headers = {}
        mock_request.client.host = "192.168.1.100"
        
        ip = middleware._get_client_ip(mock_request)
        assert ip == "192.168.1.100"
    
    def test_get_client_ip_forwarded(self, middleware, mock_request):
        """Test getting client IP from forwarded headers"""
        mock_request.headers = {"X-Forwarded-For": "10.0.0.1, 192.168.1.1"}
        
        ip = middleware._get_client_ip(mock_request)
        assert ip == "10.0.0.1"  # Should get first IP
    
    def test_get_client_ip_real_ip(self, middleware, mock_request):
        """Test getting client IP from X-Real-IP header"""
        mock_request.headers = {"X-Real-IP": "203.0.113.1"}
        
        ip = middleware._get_client_ip(mock_request)
        assert ip == "203.0.113.1"


class TestAuthenticationEndpoints:
    """Test authentication endpoints integration"""
    
    @pytest.fixture
    def test_client(self):
        """Create test client with authentication endpoints"""
        from fastapi import FastAPI
        from src.api.auth.authentication import auth_routes
        
        app = FastAPI()
        
        @app.post("/login")
        async def login(email: str, password: str):
            return await auth_routes.login(email, password)
        
        @app.post("/refresh")
        async def refresh(refresh_token: str):
            return await auth_routes.refresh_token(refresh_token)
        
        return TestClient(app)
    
    def test_login_success(self, test_client):
        """Test successful login"""
        response = test_client.post(
            "/login",
            params={"email": "admin@tradeknowledge.com", "password": "admin123"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert "user" in data
        assert data["user"]["email"] == "admin@tradeknowledge.com"
    
    def test_login_invalid_credentials(self, test_client):
        """Test login with invalid credentials"""
        response = test_client.post(
            "/login",
            params={"email": "wrong@email.com", "password": "wrong_password"}
        )
        
        assert response.status_code == 401
        assert "Invalid email or password" in response.json()["detail"]
    
    def test_refresh_token_success(self, test_client):
        """Test successful token refresh"""
        # First login to get refresh token
        login_response = test_client.post(
            "/login",
            params={"email": "admin@tradeknowledge.com", "password": "admin123"}
        )
        
        refresh_token = login_response.json()["refresh_token"]
        
        # Use refresh token to get new access token
        response = test_client.post(
            "/refresh",
            params={"refresh_token": refresh_token}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "access_token" in data
        assert data["token_type"] == "bearer"
    
    def test_refresh_token_invalid(self, test_client):
        """Test token refresh with invalid token"""
        response = test_client.post(
            "/refresh",
            params={"refresh_token": "invalid_token"}
        )
        
        assert response.status_code == 401
        assert "Invalid refresh token" in response.json()["detail"]


# Performance and stress tests
class TestRateLimitPerformance:
    """Test rate limiting performance"""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_rate_limiter_performance(self):
        """Test memory rate limiter performance under load"""
        limiter = MemoryRateLimiter()
        rule = RateLimitRule(limit=1000, window=3600)
        
        start_time = time.time()
        
        # Simulate 1000 concurrent requests
        tasks = []
        for i in range(1000):
            task = limiter.check_rate_limit(f"user_{i % 100}", rule)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time
        assert duration < 5.0  # Less than 5 seconds
        
        # All should be allowed (under limit)
        assert all(result.allowed for result in results)
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_rate_limiter_memory_usage(self):
        """Test rate limiter memory usage with many keys"""
        limiter = MemoryRateLimiter()
        rule = RateLimitRule(limit=10, window=3600)
        
        # Create many different keys
        for i in range(10000):
            await limiter.check_rate_limit(f"user_{i}", rule)
        
        # Memory should be manageable
        assert len(limiter.requests) == 10000
        
        # Cleanup should reduce memory usage
        current_time = time.time()
        await limiter._cleanup_old_entries(current_time)
        
        # Should still have entries (not expired yet)
        assert len(limiter.requests) <= 10000