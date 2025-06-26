"""
Schwab API Integration Module

Provides secure OAuth 2.0 authentication and API client for Charles Schwab services.
Implements production-ready security features including PKCE, token encryption,
and comprehensive error handling.

Features:
- OAuth 2.0 with PKCE for enhanced security
- Automatic token refresh and management
- Comprehensive API client for trading and market data
- FastAPI middleware integration
- Production-ready security and monitoring
"""

from .api_client import (
    AccountInfo,
    PriceHistory,
    Quote,
    SchwabAPIClient,
    create_api_client,
)
from .middleware import (
    SchwabAuthMiddleware,
    SchwabAuthState,
    add_schwab_api_routes,
    add_schwab_oauth_routes,
    get_schwab_client,
    is_schwab_authenticated,
    setup_schwab_integration,
)
from .oauth_client import (
    OAuthToken,
    PKCEChallenge,
    SchwabOAuthClient,
    authenticate_interactive,
    create_oauth_client,
)
from .token_manager import SchwabTokenManager, TokenMetrics, create_token_manager

__all__ = [
    # OAuth components
    "SchwabOAuthClient",
    "OAuthToken",
    "PKCEChallenge",
    "create_oauth_client",
    "authenticate_interactive",
    # Token management
    "SchwabTokenManager",
    "TokenMetrics",
    "create_token_manager",
    # API client
    "SchwabAPIClient",
    "AccountInfo",
    "Quote",
    "PriceHistory",
    "create_api_client",
    # FastAPI integration
    "SchwabAuthMiddleware",
    "SchwabAuthState",
    "setup_schwab_integration",
    "add_schwab_oauth_routes",
    "add_schwab_api_routes",
    "get_schwab_client",
    "is_schwab_authenticated",
]
