"""
Schwab OAuth FastAPI Middleware

Provides FastAPI middleware integration for Schwab OAuth authentication:
- OAuth callback handling
- Authentication state management
- Request authentication
- Session management
- Security headers
- Error handling
"""

import logging
import time
from collections.abc import Callable
from urllib.parse import parse_qs, urlparse

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

from .api_client import SchwabAPIClient
from .oauth_client import PKCEChallenge, SchwabOAuthClient
from .token_manager import SchwabTokenManager

logger = logging.getLogger(__name__)


class SchwabAuthState:
    """Manages authentication state for the application."""

    def __init__(self):
        self.oauth_client: SchwabOAuthClient | None = None
        self.token_manager: SchwabTokenManager | None = None
        self.api_client: SchwabAPIClient | None = None
        self.pkce_challenges: dict[str, PKCEChallenge] = {}
        self.is_initialized = False

    async def initialize(self) -> None:
        """Initialize Schwab authentication components."""
        if self.is_initialized:
            return

        try:
            from .api_client import create_api_client
            from .oauth_client import create_oauth_client
            from .token_manager import create_token_manager

            # Initialize OAuth client
            self.oauth_client = await create_oauth_client()

            # Initialize token manager
            self.token_manager = await create_token_manager()

            # Initialize API client
            self.api_client = await create_api_client(self.token_manager)

            self.is_initialized = True
            logger.info("Schwab authentication state initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Schwab auth state: {e}")
            raise

    async def cleanup(self) -> None:
        """Cleanup authentication components."""
        if self.api_client:
            await self.api_client.close()
        if self.token_manager:
            await self.token_manager.cleanup()
        if self.oauth_client:
            await self.oauth_client.close()

        self.is_initialized = False
        logger.info("Schwab authentication state cleaned up")


# Global authentication state
auth_state = SchwabAuthState()


class SchwabAuthMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for Schwab OAuth authentication.

    Features:
    - Automatic authentication initialization
    - Security headers
    - Request logging
    - Error handling
    - Performance monitoring
    """

    def __init__(self, app: FastAPI, enable_security_headers: bool = True):
        """
        Initialize authentication middleware.

        Args:
            app: FastAPI application
            enable_security_headers: Add security headers to responses
        """
        super().__init__(app)
        self.enable_security_headers = enable_security_headers
        self.request_count = 0
        self.start_time = time.time()

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> StarletteResponse:
        """Process request through middleware."""
        start_time = time.time()
        self.request_count += 1

        # Initialize auth state if needed
        if not auth_state.is_initialized:
            try:
                await auth_state.initialize()
            except Exception as e:
                logger.error(f"Failed to initialize Schwab auth: {e}")
                return JSONResponse(
                    status_code=503,
                    content={"error": "Authentication service unavailable"},
                )

        # Add auth state to request
        request.state.schwab_auth = auth_state

        try:
            # Process request
            response = await call_next(request)

            # Add security headers
            if self.enable_security_headers:
                self._add_security_headers(response)

            # Add performance headers
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-Count"] = str(self.request_count)

            return response

        except Exception as e:
            logger.error(f"Middleware error: {e}")
            return JSONResponse(
                status_code=500, content={"error": "Internal server error"}
            )

    def _add_security_headers(self, response: StarletteResponse) -> None:
        """Add security headers to response."""
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }

        for header, value in security_headers.items():
            response.headers[header] = value


# FastAPI OAuth routes
def add_schwab_oauth_routes(app: FastAPI, mount_path: str = "/auth/schwab") -> None:
    """
    Add Schwab OAuth routes to FastAPI application.

    Args:
        app: FastAPI application
        mount_path: Base path for OAuth routes
    """

    @app.get(f"{mount_path}/login")
    async def schwab_login(request: Request, scope: str = "readonly"):
        """
        Initiate Schwab OAuth login flow.

        Args:
            scope: OAuth scope (readonly, trade)
        """
        try:
            # Generate PKCE challenge
            pkce_challenge = auth_state.oauth_client.generate_pkce_challenge()

            # Store challenge for later verification
            auth_state.pkce_challenges[pkce_challenge.state] = pkce_challenge

            # Generate authorization URL
            auth_url = auth_state.oauth_client.get_authorization_url(
                pkce_challenge, scope
            )

            logger.info(f"Redirecting to Schwab OAuth with scope: {scope}")
            return RedirectResponse(url=auth_url)

        except Exception as e:
            logger.error(f"OAuth login failed: {e}")
            raise HTTPException(status_code=500, detail="OAuth login failed")

    @app.get(f"{mount_path}/callback")
    async def schwab_callback(request: Request):
        """
        Handle Schwab OAuth callback.

        Processes the authorization code and exchanges it for tokens.
        """
        try:
            # Get callback URL
            callback_url = str(request.url)

            # Parse query parameters
            parsed_url = urlparse(callback_url)
            query_params = parse_qs(parsed_url.query)

            # Extract parameters
            code = query_params.get("code", [None])[0]
            state = query_params.get("state", [None])[0]
            error = query_params.get("error", [None])[0]

            if error:
                logger.error(f"OAuth error: {error}")
                raise HTTPException(status_code=400, detail=f"OAuth error: {error}")

            if not code or not state:
                raise HTTPException(
                    status_code=400, detail="Missing authorization code or state"
                )

            # Verify state and get PKCE challenge
            if state not in auth_state.pkce_challenges:
                logger.warning("Invalid state parameter - possible CSRF attack")
                raise HTTPException(status_code=400, detail="Invalid state parameter")

            pkce_challenge = auth_state.pkce_challenges.pop(state)

            # Exchange code for token
            token = await auth_state.token_manager.authenticate(pkce_challenge, code)

            logger.info("OAuth callback processed successfully")
            return JSONResponse(
                content={
                    "message": "Authentication successful",
                    "token_expires_in": token.time_until_expiry,
                    "scope": token.scope,
                }
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"OAuth callback failed: {e}")
            raise HTTPException(status_code=500, detail="OAuth callback failed")

    @app.get(f"{mount_path}/status")
    async def schwab_auth_status(request: Request):
        """Get current authentication status."""
        try:
            status = await auth_state.token_manager.get_token_status()
            api_status = await auth_state.api_client.get_api_status()

            return JSONResponse(
                content={
                    "token_status": status,
                    "api_status": api_status,
                    "timestamp": time.time(),
                }
            )

        except Exception as e:
            logger.error(f"Status check failed: {e}")
            raise HTTPException(status_code=500, detail="Status check failed")

    @app.post(f"{mount_path}/logout")
    async def schwab_logout(request: Request):
        """Logout and revoke tokens."""
        try:
            success = await auth_state.token_manager.revoke_token()

            return JSONResponse(
                content={
                    "message": (
                        "Logout successful"
                        if success
                        else "Logout completed with errors"
                    ),
                    "success": success,
                }
            )

        except Exception as e:
            logger.error(f"Logout failed: {e}")
            return JSONResponse(
                content={
                    "message": "Logout completed with errors",
                    "success": False,
                    "error": str(e),
                }
            )


# FastAPI API routes
def add_schwab_api_routes(app: FastAPI, mount_path: str = "/api/schwab") -> None:
    """
    Add Schwab API routes to FastAPI application.

    Args:
        app: FastAPI application
        mount_path: Base path for API routes
    """

    # Security scheme
    security = HTTPBearer()

    async def get_authenticated_client(
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> SchwabAPIClient:
        """Dependency to get authenticated API client."""
        try:
            # Verify token is valid
            await auth_state.token_manager.get_valid_token()
            return auth_state.api_client
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise HTTPException(status_code=401, detail="Authentication required")

    @app.get(f"{mount_path}/accounts")
    async def get_accounts(client: SchwabAPIClient = Depends(get_authenticated_client)):
        """Get user accounts."""
        try:
            accounts = await client.get_accounts()
            return [account.model_dump() for account in accounts]
        except Exception as e:
            logger.error(f"Get accounts failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve accounts")

    @app.get(f"{mount_path}/accounts/{{account_id}}")
    async def get_account_details(
        account_id: str, client: SchwabAPIClient = Depends(get_authenticated_client)
    ):
        """Get account details."""
        try:
            account = await client.get_account_details(account_id)
            return account.model_dump()
        except Exception as e:
            logger.error(f"Get account details failed: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to retrieve account details"
            )

    @app.get(f"{mount_path}/quotes")
    async def get_quotes(
        symbols: str, client: SchwabAPIClient = Depends(get_authenticated_client)
    ):
        """Get quotes for symbols."""
        try:
            symbol_list = [s.strip().upper() for s in symbols.split(",")]
            quotes = await client.get_quotes(symbol_list)
            return [quote.model_dump() for quote in quotes]
        except Exception as e:
            logger.error(f"Get quotes failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve quotes")

    @app.get(f"{mount_path}/price-history/{{symbol}}")
    async def get_price_history(
        symbol: str,
        start_date: str,
        end_date: str,
        frequency: str = "daily",
        client: SchwabAPIClient = Depends(get_authenticated_client),
    ):
        """Get price history for symbol."""
        try:
            from datetime import datetime

            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)

            history = await client.get_price_history(symbol, start, end, frequency)
            return history.model_dump()
        except Exception as e:
            logger.error(f"Get price history failed: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to retrieve price history"
            )

    @app.get(f"{mount_path}/health")
    async def api_health_check(
        client: SchwabAPIClient = Depends(get_authenticated_client),
    ):
        """Perform API health check."""
        try:
            health = await client.health_check()
            return health
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=500, detail="Health check failed")


# Setup function for complete integration
def setup_schwab_integration(
    app: FastAPI,
    enable_middleware: bool = True,
    enable_oauth_routes: bool = True,
    enable_api_routes: bool = True,
    oauth_mount_path: str = "/auth/schwab",
    api_mount_path: str = "/api/schwab",
) -> None:
    """
    Complete Schwab integration setup for FastAPI application.

    Args:
        app: FastAPI application
        enable_middleware: Enable authentication middleware
        enable_oauth_routes: Enable OAuth routes
        enable_api_routes: Enable API routes
        oauth_mount_path: Mount path for OAuth routes
        api_mount_path: Mount path for API routes
    """

    # Add middleware
    if enable_middleware:
        app.add_middleware(SchwabAuthMiddleware)
        logger.info("Schwab authentication middleware added")

    # Add OAuth routes
    if enable_oauth_routes:
        add_schwab_oauth_routes(app, oauth_mount_path)
        logger.info(f"Schwab OAuth routes added at {oauth_mount_path}")

    # Add API routes
    if enable_api_routes:
        add_schwab_api_routes(app, api_mount_path)
        logger.info(f"Schwab API routes added at {api_mount_path}")

    # Add startup and shutdown events
    @app.on_event("startup")
    async def startup_schwab():
        """Initialize Schwab authentication on startup."""
        try:
            await auth_state.initialize()
            logger.info("Schwab integration started successfully")
        except Exception as e:
            logger.error(f"Failed to start Schwab integration: {e}")

    @app.on_event("shutdown")
    async def shutdown_schwab():
        """Cleanup Schwab authentication on shutdown."""
        try:
            await auth_state.cleanup()
            logger.info("Schwab integration shutdown completed")
        except Exception as e:
            logger.error(f"Error during Schwab shutdown: {e}")


# Utility functions
async def get_schwab_client() -> SchwabAPIClient:
    """Get the global Schwab API client."""
    if not auth_state.is_initialized:
        await auth_state.initialize()
    return auth_state.api_client


async def is_schwab_authenticated() -> bool:
    """Check if Schwab is authenticated."""
    try:
        if not auth_state.is_initialized:
            return False

        token = await auth_state.token_manager.get_current_token()
        return token is not None and not token.is_expired
    except Exception:
        return False


if __name__ == "__main__":
    """Middleware testing."""
    import uvicorn
    from fastapi import FastAPI

    # Create test app
    app = FastAPI(title="Schwab Integration Test")

    # Setup Schwab integration
    setup_schwab_integration(app)

    @app.get("/")
    async def root():
        """Test endpoint."""
        return {"message": "Schwab integration test server"}

    @app.get("/test/auth")
    async def test_auth():
        """Test authentication status."""
        authenticated = await is_schwab_authenticated()
        return {"authenticated": authenticated}

    # Run test server
    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=8000)
