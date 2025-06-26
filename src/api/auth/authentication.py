"""
Advanced Authentication System for TradeKnowledge API
Provides JWT-based authentication, API key management, and role-based access control
"""

import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import bcrypt
import jwt
import structlog
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ...core.config import get_config
from ...core.models import APIKey, User, UserRole

logger = structlog.get_logger(__name__)


class AuthenticationError(Exception):
    """Custom authentication error"""

    pass


class AuthorizationError(Exception):
    """Custom authorization error"""

    pass


class TokenType(Enum):
    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"


class JWTManager:
    """JWT token management"""

    def __init__(self):
        self.config = get_config()
        self.secret_key = self.config.security.secret_key
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7

    def create_access_token(self, user_id: str, email: str, roles: list[str]) -> str:
        """Create JWT access token"""
        now = datetime.now(UTC)
        expire = now + timedelta(minutes=self.access_token_expire_minutes)

        payload = {
            "sub": user_id,
            "email": email,
            "roles": roles,
            "type": TokenType.ACCESS.value,
            "iat": now.timestamp(),
            "exp": expire.timestamp(),
            "jti": secrets.token_urlsafe(16),  # JWT ID for revocation
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def create_refresh_token(self, user_id: str) -> str:
        """Create JWT refresh token"""
        now = datetime.now(UTC)
        expire = now + timedelta(days=self.refresh_token_expire_days)

        payload = {
            "sub": user_id,
            "type": TokenType.REFRESH.value,
            "iat": now.timestamp(),
            "exp": expire.timestamp(),
            "jti": secrets.token_urlsafe(16),
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(
        self, token: str, expected_type: TokenType = TokenType.ACCESS
    ) -> dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Verify token type
            if payload.get("type") != expected_type.value:
                raise AuthenticationError(f"Invalid token type: {payload.get('type')}")

            # Check expiration
            if datetime.now(UTC).timestamp() > payload.get("exp", 0):
                raise AuthenticationError("Token has expired")

            return payload

        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")

    def refresh_access_token(
        self, refresh_token: str, user_data: dict[str, Any]
    ) -> str:
        """Generate new access token from refresh token"""
        payload = self.verify_token(refresh_token, TokenType.REFRESH)

        # Create new access token
        return self.create_access_token(
            user_id=payload["sub"], email=user_data["email"], roles=user_data["roles"]
        )


class APIKeyManager:
    """API Key management system"""

    def __init__(self):
        self.config = get_config()

    def generate_api_key(
        self, user_id: str, name: str, permissions: list[str]
    ) -> APIKey:
        """Generate new API key"""
        # Generate cryptographically secure key
        key_value = f"tk_{secrets.token_urlsafe(32)}"

        # Hash for storage
        key_hash = hashlib.sha256(key_value.encode()).hexdigest()

        api_key = APIKey(
            key_id=secrets.token_urlsafe(16),
            key_hash=key_hash,
            user_id=user_id,
            name=name,
            permissions=permissions,
            created_at=datetime.now(UTC),
            last_used=None,
            is_active=True,
            rate_limit=1000,  # Default rate limit per hour
            usage_count=0,
        )

        # Return key with actual value (only time it's visible)
        api_key.key_value = key_value
        return api_key

    def verify_api_key(self, key_value: str) -> APIKey | None:
        """Verify API key and return key info"""
        if not key_value.startswith("tk_"):
            return None

        key_hash = hashlib.sha256(key_value.encode()).hexdigest()

        # TODO: Implement database lookup
        # For now, return mock API key
        if key_hash == "valid_hash":  # Replace with actual DB lookup
            return APIKey(
                key_id="mock_key_id",
                key_hash=key_hash,
                user_id="mock_user",
                name="Test API Key",
                permissions=["read", "search"],
                created_at=datetime.now(UTC),
                is_active=True,
                rate_limit=1000,
                usage_count=0,
            )

        return None

    def update_api_key_usage(self, api_key: APIKey):
        """Update API key usage statistics"""
        api_key.usage_count += 1
        api_key.last_used = datetime.now(UTC)

        # TODO: Update in database
        logger.info(
            "API key used",
            key_id=api_key.key_id,
            user_id=api_key.user_id,
            usage_count=api_key.usage_count,
        )


class UserManager:
    """User management system"""

    def __init__(self):
        self.config = get_config()

    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))

    async def authenticate_user(self, email: str, password: str) -> User | None:
        """Authenticate user with email/password"""
        # TODO: Implement database lookup
        # For now, return mock user for demo
        if email == "admin@tradeknowledge.com" and password == "admin123":
            return User(
                id="admin_user_id",
                email=email,
                username="admin",
                full_name="Admin User",
                roles=[UserRole.ADMIN],
                is_active=True,
                created_at=datetime.now(UTC),
                last_login=datetime.now(UTC),
            )

        return None

    async def get_user_by_id(self, user_id: str) -> User | None:
        """Get user by ID"""
        # TODO: Implement database lookup
        if user_id == "admin_user_id":
            return User(
                id=user_id,
                email="admin@tradeknowledge.com",
                username="admin",
                full_name="Admin User",
                roles=[UserRole.ADMIN],
                is_active=True,
                created_at=datetime.now(UTC),
                last_login=datetime.now(UTC),
            )

        return None

    async def create_user(
        self,
        email: str,
        password: str,
        username: str,
        full_name: str,
        roles: list[UserRole] = None,
    ) -> User:
        """Create new user"""
        if roles is None:
            roles = [UserRole.USER]

        hashed_password = self.hash_password(password)

        user = User(
            id=secrets.token_urlsafe(16),
            email=email,
            username=username,
            full_name=full_name,
            password_hash=hashed_password,
            roles=roles,
            is_active=True,
            created_at=datetime.now(UTC),
        )

        # TODO: Save to database
        logger.info("User created", user_id=user.id, email=user.email)

        return user


class AuthenticationMiddleware:
    """Authentication middleware for FastAPI"""

    def __init__(self):
        self.jwt_manager = JWTManager()
        self.api_key_manager = APIKeyManager()
        self.user_manager = UserManager()
        self.security = HTTPBearer(auto_error=False)

    async def get_current_user(
        self,
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = Depends(
            HTTPBearer(auto_error=False)
        ),
    ) -> User | None:
        """Get current authenticated user"""

        # Try API key authentication first
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return await self._authenticate_api_key(api_key)

        # Try JWT authentication
        if credentials:
            return await self._authenticate_jwt(credentials.credentials)

        return None

    async def require_authentication(
        self, current_user: User = Depends(get_current_user)
    ) -> User:
        """Require authentication - raises exception if not authenticated"""
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not current_user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is disabled",
            )

        return current_user

    async def require_roles(
        self,
        required_roles: list[UserRole],
        current_user: User = Depends(require_authentication),
    ) -> User:
        """Require specific roles"""
        if not any(role in current_user.roles for role in required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions"
            )

        return current_user

    async def require_admin(
        self, current_user: User = Depends(require_authentication)
    ) -> User:
        """Require admin role"""
        if UserRole.ADMIN not in current_user.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required"
            )

        return current_user

    async def _authenticate_jwt(self, token: str) -> User | None:
        """Authenticate using JWT token"""
        try:
            payload = self.jwt_manager.verify_token(token)
            user = await self.user_manager.get_user_by_id(payload["sub"])

            if user and user.is_active:
                # Update last login
                user.last_login = datetime.now(UTC)
                return user

        except AuthenticationError as e:
            logger.warning("JWT authentication failed", error=str(e))

        return None

    async def _authenticate_api_key(self, api_key_value: str) -> User | None:
        """Authenticate using API key"""
        try:
            api_key = self.api_key_manager.verify_api_key(api_key_value)

            if api_key and api_key.is_active:
                # Update usage statistics
                self.api_key_manager.update_api_key_usage(api_key)

                # Get associated user
                user = await self.user_manager.get_user_by_id(api_key.user_id)
                if user and user.is_active:
                    return user

        except Exception as e:
            logger.warning("API key authentication failed", error=str(e))

        return None


# Global instances
jwt_manager = JWTManager()
api_key_manager = APIKeyManager()
user_manager = UserManager()
auth_middleware = AuthenticationMiddleware()


# Dependency functions for FastAPI
async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(
        HTTPBearer(auto_error=False)
    ),
) -> User | None:
    """FastAPI dependency to get current user"""
    return await auth_middleware.get_current_user(request, credentials)


async def require_authentication(
    current_user: User = Depends(get_current_user),
) -> User:
    """FastAPI dependency to require authentication"""
    return await auth_middleware.require_authentication(current_user)


async def require_admin(current_user: User = Depends(require_authentication)) -> User:
    """FastAPI dependency to require admin role"""
    return await auth_middleware.require_admin(current_user)


def require_roles(*roles: UserRole):
    """FastAPI dependency factory to require specific roles"""

    async def _require_roles(
        current_user: User = Depends(require_authentication),
    ) -> User:
        return await auth_middleware.require_roles(list(roles), current_user)

    return _require_roles


# Authentication endpoints
class AuthenticationRoutes:
    """Authentication endpoint handlers"""

    def __init__(self):
        self.jwt_manager = jwt_manager
        self.user_manager = user_manager
        self.api_key_manager = api_key_manager

    async def login(self, email: str, password: str) -> dict[str, Any]:
        """User login endpoint"""
        user = await self.user_manager.authenticate_user(email, password)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )

        # Create tokens
        access_token = self.jwt_manager.create_access_token(
            user_id=user.id, email=user.email, roles=[role.value for role in user.roles]
        )

        refresh_token = self.jwt_manager.create_refresh_token(user.id)

        logger.info("User logged in", user_id=user.id, email=user.email)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": 1800,  # 30 minutes
            "user": {
                "id": user.id,
                "email": user.email,
                "username": user.username,
                "full_name": user.full_name,
                "roles": [role.value for role in user.roles],
            },
        }

    async def refresh_token(self, refresh_token: str) -> dict[str, str]:
        """Refresh access token"""
        try:
            payload = self.jwt_manager.verify_token(refresh_token, TokenType.REFRESH)
            user = await self.user_manager.get_user_by_id(payload["sub"])

            if not user or not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found or inactive",
                )

            # Create new access token
            access_token = self.jwt_manager.create_access_token(
                user_id=user.id,
                email=user.email,
                roles=[role.value for role in user.roles],
            )

            return {
                "access_token": access_token,
                "token_type": "bearer",
                "expires_in": 1800,
            }

        except AuthenticationError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
            )

    async def create_api_key(
        self, user: User, name: str, permissions: list[str]
    ) -> dict[str, Any]:
        """Create new API key"""
        api_key = self.api_key_manager.generate_api_key(
            user_id=user.id, name=name, permissions=permissions
        )

        logger.info(
            "API key created", user_id=user.id, key_id=api_key.key_id, name=name
        )

        return {
            "key_id": api_key.key_id,
            "key_value": api_key.key_value,  # Only returned once
            "name": api_key.name,
            "permissions": api_key.permissions,
            "rate_limit": api_key.rate_limit,
            "created_at": api_key.created_at.isoformat(),
        }

    async def list_api_keys(self, user: User) -> list[dict[str, Any]]:
        """List user's API keys (without values)"""
        # TODO: Implement database query
        return [
            {
                "key_id": "mock_key_id",
                "name": "Mock API Key",
                "permissions": ["read", "search"],
                "rate_limit": 1000,
                "created_at": datetime.now(UTC).isoformat(),
                "last_used": None,
                "usage_count": 0,
                "is_active": True,
            }
        ]

    async def revoke_api_key(self, user: User, key_id: str) -> dict[str, str]:
        """Revoke API key"""
        # TODO: Implement database update
        logger.info("API key revoked", user_id=user.id, key_id=key_id)

        return {"message": f"API key {key_id} has been revoked"}


# Global authentication routes instance
auth_routes = AuthenticationRoutes()
