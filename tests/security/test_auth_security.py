"""
Security tests for authentication and authorization systems.

This module tests for authentication bypass vulnerabilities, 
token manipulation attacks, and authorization flaws.
"""

import pytest
import jwt
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.api.auth import AuthManager
from src.api.models import User


class TestAuthenticationSecurity:
    """Test authentication security measures"""
    
    @pytest.fixture
    def auth_manager(self):
        """Create AuthManager instance for testing"""
        with patch('src.api.auth.get_config') as mock_config, \
             patch('src.api.auth.get_user_manager') as mock_user_manager:
            
            # Mock config
            mock_auth_config = MagicMock()
            mock_auth_config.secret_key = "test-secret-key"
            mock_auth_config.algorithm = "HS256"
            mock_auth_config.token_expiry_hours = 24
            
            mock_config.return_value.api.auth = mock_auth_config
            
            # Mock user manager
            mock_user_manager.return_value = AsyncMock()
            
            return AuthManager()
    
    @pytest.mark.asyncio
    async def test_jwt_token_manipulation_attacks(self, auth_manager):
        """Test JWT token manipulation resistance"""
        # Create valid token
        user = User(
            id="test_user", 
            username="testuser", 
            email="test@example.com",
            role="user",
            created_at=datetime.now().isoformat()
        )
        token = await auth_manager.create_token(user)
        
        # Test 1: Modified payload attack
        decoded = jwt.decode(token, options={"verify_signature": False})
        decoded["user_id"] = "admin_user"  # Try to escalate privileges
        
        # Re-encode without proper signature
        malicious_token = jwt.encode(decoded, "wrong_secret", algorithm="HS256")
        
        with pytest.raises((jwt.InvalidSignatureError, jwt.DecodeError, ValueError)):
            await auth_manager.verify_token(malicious_token)
    
    @pytest.mark.asyncio
    async def test_token_algorithm_confusion_attack(self, auth_manager):
        """Test algorithm confusion attacks (HS256 vs RS256)"""
        user = User(
            id="test_user", 
            username="testuser", 
            email="test@example.com",
            role="user",
            created_at=datetime.now().isoformat()
        )
        token = await auth_manager.create_token(user)
        
        # Try to decode with 'none' algorithm
        with pytest.raises((jwt.InvalidSignatureError, jwt.DecodeError)):
            jwt.decode(token, algorithms=["none"], options={"verify_signature": False})
    
    @pytest.mark.asyncio
    async def test_expired_token_rejection(self, auth_manager):
        """Test that expired tokens are properly rejected"""
        user = User(
            id="test_user", 
            username="testuser", 
            email="test@example.com",
            role="user",
            created_at=datetime.now().isoformat()
        )
        
        # Mock time to create expired token
        with patch('src.api.auth.datetime') as mock_datetime:
            # Set time to past
            mock_datetime.utcnow.return_value = datetime.utcnow() - timedelta(hours=25)
            token = await auth_manager.create_token(user)
        
        # Token should be rejected as expired
        with pytest.raises((jwt.ExpiredSignatureError, ValueError)):
            await auth_manager.verify_token(token)
    
    @pytest.mark.asyncio
    async def test_token_with_no_expiration(self, auth_manager):
        """Test that tokens without expiration are rejected"""
        # Create token without exp claim
        payload = {"user_id": "test_user"}
        token = jwt.encode(payload, auth_manager.secret_key, algorithm="HS256")
        
        with pytest.raises((jwt.MissingRequiredClaimError, ValueError)):
            await auth_manager.verify_token(token)
    
    @pytest.mark.asyncio
    async def test_brute_force_protection(self, auth_manager):
        """Test brute force protection mechanisms"""
        username = "test_user"
        
        # Simulate multiple failed attempts
        for _ in range(6):  # Exceed typical limit of 5
            try:
                await auth_manager.authenticate_user(username, "wrong_password")
            except:
                pass
        
        # Account should be locked after failed attempts
        # This test assumes lockout functionality exists
        with pytest.raises(Exception):  # Should raise lockout exception
            await auth_manager.authenticate_user(username, "correct_password")


class TestAuthorizationSecurity:
    """Test authorization and privilege escalation security"""
    
    @pytest.fixture
    def auth_manager(self):
        """Create AuthManager instance for testing"""
        with patch('src.api.auth.get_config') as mock_config, \
             patch('src.api.auth.get_user_manager') as mock_user_manager:
            
            # Mock config
            mock_auth_config = MagicMock()
            mock_auth_config.secret_key = "test-secret-key"
            mock_auth_config.algorithm = "HS256"
            mock_auth_config.token_expiry_hours = 24
            
            mock_config.return_value.api.auth = mock_auth_config
            
            # Mock user manager
            mock_user_manager.return_value = AsyncMock()
            
            return AuthManager()
    
    def test_privilege_escalation_prevention(self, auth_manager):
        """Test prevention of privilege escalation attacks"""
        # Create regular user
        regular_user = User(
            id="regular_user", 
            username="regular",
            email="regular@example.com",
            role="user",
            created_at=datetime.now().isoformat()
        )
        
        # Test accessing admin-only resources should fail
        assert not auth_manager.check_permission(regular_user, "admin")
        assert not auth_manager.check_permission(regular_user, "write")
        assert not auth_manager.check_permission(regular_user, "delete")
        
        # Test that user can access read permission
        assert auth_manager.check_permission(regular_user, "read")
    
    def test_horizontal_privilege_escalation(self, auth_manager):
        """Test prevention of accessing other users' data"""
        user1 = User(id="user1", username="user1")
        user2 = User(id="user2", username="user2")
        
        token1 = auth_manager.create_token(user1.id)
        
        # User1 should not be able to access User2's data
        with pytest.raises(Exception):
            auth_manager.check_permission(token1, "read", resource_owner=user2.id)
    
    def test_session_fixation_protection(self, auth_manager):
        """Test session fixation attack protection"""
        # Create initial session
        user = User(id="test_user", username="testuser")
        old_token = auth_manager.create_token(user.id)
        
        # After login, new token should be issued
        new_token = auth_manager.create_token(user.id)
        
        # Tokens should be different (new session)
        assert old_token != new_token
    
    def test_token_reuse_prevention(self, auth_manager):
        """Test that tokens cannot be reused after logout"""
        user = User(id="test_user", username="testuser")
        token = auth_manager.create_token(user.id)
        
        # Verify token works initially
        result = auth_manager.verify_token(token)
        assert result is not None
        
        # Simulate logout (token invalidation)
        # This assumes logout functionality exists
        auth_manager.logout(token)
        
        # Token should no longer be valid
        with pytest.raises(Exception):
            auth_manager.verify_token(token)


class TestPasswordSecurity:
    """Test password security measures"""
    
    @pytest.fixture
    def auth_manager(self):
        return AuthManager()
    
    def test_password_complexity_enforcement(self, auth_manager):
        """Test password complexity requirements"""
        weak_passwords = [
            "123456",
            "password",
            "abc123",
            "qwerty",
            "",
            "a" * 7,  # Too short
        ]
        
        for password in weak_passwords:
            with pytest.raises(ValueError):
                auth_manager.validate_password(password)
    
    def test_password_hashing_security(self, auth_manager):
        """Test that passwords are properly hashed"""
        password = "secure_password_123!"
        hashed = auth_manager.hash_password(password)
        
        # Hash should not equal plaintext
        assert hashed != password
        
        # Hash should be verifiable
        assert auth_manager.verify_password(password, hashed)
        
        # Wrong password should not verify
        assert not auth_manager.verify_password("wrong_password", hashed)
    
    def test_timing_attack_resistance(self, auth_manager):
        """Test resistance to timing attacks"""
        import time
        
        password = "correct_password"
        hashed = auth_manager.hash_password(password)
        
        # Time verification of correct password
        start = time.time()
        auth_manager.verify_password(password, hashed)
        correct_time = time.time() - start
        
        # Time verification of incorrect password
        start = time.time()
        auth_manager.verify_password("wrong_password", hashed)
        wrong_time = time.time() - start
        
        # Times should be similar (within reasonable variance)
        # This is a basic check - real timing attack tests require more sophisticated measurement
        time_difference = abs(correct_time - wrong_time)
        assert time_difference < 0.1  # 100ms variance allowed


class TestSessionSecurity:
    """Test session management security"""
    
    @pytest.fixture
    def auth_manager(self):
        return AuthManager()
    
    def test_session_timeout_enforcement(self, auth_manager):
        """Test session timeout is properly enforced"""
        user = User(id="test_user", username="testuser")
        
        # Create token that should expire soon
        with patch('src.api.auth.datetime') as mock_datetime:
            # Set current time
            current_time = datetime.utcnow()
            mock_datetime.utcnow.return_value = current_time
            token = auth_manager.create_token(user.id, expires_minutes=1)
            
            # Fast forward time past expiration
            mock_datetime.utcnow.return_value = current_time + timedelta(minutes=2)
            
            with pytest.raises(jwt.ExpiredSignatureError):
                auth_manager.verify_token(token)
    
    def test_concurrent_session_limits(self, auth_manager):
        """Test concurrent session limits"""
        user = User(id="test_user", username="testuser")
        
        # Create multiple tokens for same user
        tokens = []
        for _ in range(5):  # Assume limit is 3
            token = auth_manager.create_token(user.id)
            tokens.append(token)
        
        # Older tokens should be invalidated
        # This test assumes session limit functionality exists
        for token in tokens[:2]:  # First two should be invalid
            with pytest.raises(Exception):
                auth_manager.verify_token(token)
        
        # Recent tokens should still work
        for token in tokens[2:]:  # Last three should be valid
            result = auth_manager.verify_token(token)
            assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])