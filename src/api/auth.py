"""
Authentication manager for TradeKnowledge API

Handles user authentication, JWT tokens, and authorization.
"""

import jwt
from datetime import datetime, timedelta
from typing import Optional
import structlog

from ..core.config import get_config
from .models import User
from .user_manager import get_user_manager

logger = structlog.get_logger(__name__)

class AuthManager:
    """Authentication and authorization manager"""
    
    def __init__(self):
        self.config = get_config()
        self.user_manager = get_user_manager()
        self.secret_key = self.config.api.auth.secret_key
        self.algorithm = self.config.api.auth.algorithm
        self.token_expiry_hours = self.config.api.auth.token_expiry_hours
    
    async def verify_token(self, token: str) -> User:
        """Verify JWT token and return user"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            user_id = payload.get('user_id')
            
            if not user_id:
                raise ValueError("Invalid token")
            
            # Get user from database
            user = await self.user_manager.get_user_by_id(user_id)
            if not user or not user.is_active:
                raise ValueError("User not found or inactive")
            
            return user
            
        except jwt.ExpiredSignatureError:
            raise ValueError("Token expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")
    
    async def create_token(self, user: User) -> str:
        """Create JWT token for user"""
        payload = {
            'user_id': user.id,
            'username': user.username,
            'role': user.role,
            'exp': datetime.utcnow() + timedelta(hours=self.token_expiry_hours)
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    async def authenticate_user(self, username: str, password: str, ip_address: str = "unknown") -> Optional[User]:
        """Authenticate user with username/password"""
        return await self.user_manager.authenticate_user(username, password, ip_address)
    
    async def create_user(self, username: str, email: str, password: str, role: str = "user") -> User:
        """Create a new user"""
        from .user_manager import UserCreate
        user_data = UserCreate(
            username=username,
            email=email,
            password=password,
            role=role
        )
        return await self.user_manager.create_user(user_data)
    
    def check_permission(self, user: User, permission: str) -> bool:
        """Check if user has specific permission"""
        # Define role-based permissions
        role_permissions = {
            "admin": ["read", "write", "delete", "admin", "upload", "manage_users"],
            "editor": ["read", "write", "upload"],
            "user": ["read"],
            "viewer": ["read"]
        }
        
        user_permissions = role_permissions.get(user.role, [])
        return permission in user_permissions
    
    def require_permission(self, permission: str):
        """Decorator to require specific permission"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # Get user from request context
                user = kwargs.get('user')
                if not user or not self.check_permission(user, permission):
                    from fastapi import HTTPException
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                return await func(*args, **kwargs)
            return wrapper
        return decorator


# Singleton instance
_auth_manager: Optional[AuthManager] = None

def get_auth_manager() -> AuthManager:
    """Get authentication manager singleton"""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager