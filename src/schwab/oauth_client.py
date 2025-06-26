"""
Schwab OAuth 2.0 Client with PKCE Implementation

Implements secure OAuth 2.0 authentication with PKCE (Proof Key for Code Exchange)
following Charles Schwab's authentication protocol.

Features:
- OAuth 2.0 with PKCE for enhanced security
- Automatic token refresh
- Secure token storage with encryption
- State parameter for CSRF protection
- Comprehensive error handling
- Production-ready configuration
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import secrets
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import aiohttp
from cryptography.fernet import Fernet
from pydantic import BaseModel, Field

# Import config conditionally to avoid import errors
try:
    from src.core.config import get_config
except ImportError:

    def get_config():
        return None


logger = logging.getLogger(__name__)


class OAuthToken(BaseModel):
    """OAuth token model with security features."""

    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int
    scope: str
    created_at: float = Field(default_factory=time.time)

    @property
    def expires_at(self) -> float:
        """Calculate absolute expiration time."""
        return self.created_at + self.expires_in

    @property
    def is_expired(self) -> bool:
        """Check if token is expired (with 5-minute buffer)."""
        return time.time() >= (self.expires_at - 300)

    @property
    def time_until_expiry(self) -> float:
        """Get seconds until token expires."""
        return max(0, self.expires_at - time.time())


class PKCEChallenge(BaseModel):
    """PKCE challenge data for secure OAuth flow."""

    code_verifier: str
    code_challenge: str
    code_challenge_method: str = "S256"
    state: str


class SchwabOAuthClient:
    """
    Schwab OAuth 2.0 client with PKCE implementation.

    Provides secure authentication flow with automatic token management,
    encryption at rest, and comprehensive security features.
    """

    # Schwab OAuth endpoints
    AUTHORIZATION_URL = "https://api.schwabapi.com/v1/oauth/authorize"
    TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"

    def __init__(
        self,
        app_key: str,
        app_secret: str,
        redirect_uri: str,
        token_storage_path: str | None = None,
        encryption_key: bytes | None = None,
    ):
        """
        Initialize Schwab OAuth client.

        Args:
            app_key: Schwab app key (client ID)
            app_secret: Schwab app secret
            redirect_uri: Authorized redirect URI
            token_storage_path: Path to store encrypted tokens
            encryption_key: Key for token encryption (auto-generated if None)
        """
        self.app_key = app_key
        self.app_secret = app_secret
        self.redirect_uri = redirect_uri

        # Token storage configuration
        self.token_storage_path = Path(token_storage_path or "data/schwab_tokens")
        self.token_storage_path.mkdir(parents=True, exist_ok=True)
        self.token_file = self.token_storage_path / "oauth_token.encrypted"

        # Initialize encryption
        self._setup_encryption(encryption_key)

        # Session management
        self._session: aiohttp.ClientSession | None = None
        self._current_token: OAuthToken | None = None
        self._refresh_lock = asyncio.Lock()

        # Security settings
        self.timeout = aiohttp.ClientTimeout(total=30)
        self.max_retries = 3
        self.retry_delay = 1.0

        logger.info("Schwab OAuth client initialized")

    def _setup_encryption(self, encryption_key: bytes | None = None) -> None:
        """Initialize encryption for token storage."""
        key_file = self.token_storage_path / "encryption.key"

        if encryption_key:
            self.encryption_key = encryption_key
        elif key_file.exists():
            # Load existing key
            with open(key_file, "rb") as f:
                self.encryption_key = f.read()
        else:
            # Generate new key
            self.encryption_key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(self.encryption_key)
            # Secure the key file
            os.chmod(key_file, 0o600)

        self.cipher = Fernet(self.encryption_key)

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def _ensure_session(self) -> None:
        """Ensure HTTP session is available."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                ssl=True,  # Enforce SSL
                limit=100,
                ttl_dns_cache=300,
                use_dns_cache=True,
            )
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=self.timeout,
                headers={
                    "User-Agent": "TradeKnowledge/1.0",
                    "Accept": "application/json",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def generate_pkce_challenge(self) -> PKCEChallenge:
        """
        Generate PKCE challenge for secure OAuth flow.

        Returns:
            PKCEChallenge containing verifier, challenge, and state
        """
        # Generate cryptographically secure random values
        code_verifier = (
            base64.urlsafe_b64encode(secrets.token_bytes(96))
            .decode("utf-8")
            .rstrip("=")
        )
        state = secrets.token_urlsafe(32)

        # Create SHA256 challenge
        challenge_bytes = hashlib.sha256(code_verifier.encode("utf-8")).digest()
        code_challenge = (
            base64.urlsafe_b64encode(challenge_bytes).decode("utf-8").rstrip("=")
        )

        return PKCEChallenge(
            code_verifier=code_verifier, code_challenge=code_challenge, state=state
        )

    def get_authorization_url(
        self, pkce_challenge: PKCEChallenge, scope: str = "readonly"
    ) -> str:
        """
        Generate authorization URL for user consent.

        Args:
            pkce_challenge: PKCE challenge data
            scope: OAuth scope (readonly, trade)

        Returns:
            Authorization URL for user to visit
        """
        params = {
            "response_type": "code",
            "client_id": self.app_key,
            "redirect_uri": self.redirect_uri,
            "scope": scope,
            "code_challenge": pkce_challenge.code_challenge,
            "code_challenge_method": pkce_challenge.code_challenge_method,
            "state": pkce_challenge.state,
        }

        auth_url = f"{self.AUTHORIZATION_URL}?{urlencode(params)}"
        logger.info(f"Generated authorization URL with scope: {scope}")
        return auth_url

    def extract_authorization_code(
        self, callback_url: str, expected_state: str
    ) -> tuple[str, bool]:
        """
        Extract authorization code from callback URL.

        Args:
            callback_url: Full callback URL from redirect
            expected_state: Expected state parameter for validation

        Returns:
            Tuple of (authorization_code, state_valid)

        Raises:
            ValueError: If URL parsing fails or required parameters missing
        """
        try:
            parsed_url = urlparse(callback_url)
            query_params = parse_qs(parsed_url.query)

            # Extract parameters
            code = query_params.get("code", [None])[0]
            state = query_params.get("state", [None])[0]
            error = query_params.get("error", [None])[0]

            if error:
                raise ValueError(f"OAuth error: {error}")

            if not code:
                raise ValueError("Authorization code not found in callback URL")

            if not state:
                raise ValueError("State parameter not found in callback URL")

            # Validate state to prevent CSRF attacks
            state_valid = state == expected_state
            if not state_valid:
                logger.warning(
                    "State parameter validation failed - possible CSRF attack"
                )

            return code, state_valid

        except Exception as e:
            logger.error(f"Failed to extract authorization code: {e}")
            raise ValueError(f"Invalid callback URL: {e}")

    async def exchange_code_for_token(
        self, authorization_code: str, pkce_challenge: PKCEChallenge
    ) -> OAuthToken:
        """
        Exchange authorization code for access token using PKCE.

        Args:
            authorization_code: Authorization code from callback
            pkce_challenge: Original PKCE challenge data

        Returns:
            OAuthToken with access and refresh tokens
        """
        await self._ensure_session()

        # Prepare token request
        data = {
            "grant_type": "authorization_code",
            "code": authorization_code,
            "client_id": self.app_key,
            "client_secret": self.app_secret,
            "redirect_uri": self.redirect_uri,
            "code_verifier": pkce_challenge.code_verifier,
        }

        try:
            async with self._session.post(self.TOKEN_URL, data=data) as response:
                response_data = await response.json()

                if response.status != 200:
                    error_msg = response_data.get(
                        "error_description", "Token exchange failed"
                    )
                    logger.error(f"Token exchange failed: {error_msg}")
                    raise ValueError(f"Token exchange failed: {error_msg}")

                # Create token object
                token = OAuthToken(
                    access_token=response_data["access_token"],
                    refresh_token=response_data["refresh_token"],
                    token_type=response_data.get("token_type", "Bearer"),
                    expires_in=response_data["expires_in"],
                    scope=response_data.get("scope", ""),
                    created_at=time.time(),
                )

                # Store token securely
                await self._save_token(token)
                self._current_token = token

                logger.info("Successfully exchanged authorization code for token")
                return token

        except aiohttp.ClientError as e:
            logger.error(f"Network error during token exchange: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during token exchange: {e}")
            raise

    async def refresh_access_token(
        self, refresh_token: str | None = None
    ) -> OAuthToken:
        """
        Refresh access token using refresh token.

        Args:
            refresh_token: Refresh token (uses stored token if None)

        Returns:
            New OAuthToken with refreshed access token
        """
        async with self._refresh_lock:
            # Use provided refresh token or current stored token
            if refresh_token is None:
                if self._current_token is None:
                    await self.load_token()
                if self._current_token is None:
                    raise ValueError("No refresh token available")
                refresh_token = self._current_token.refresh_token

            await self._ensure_session()

            # Prepare refresh request
            data = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": self.app_key,
                "client_secret": self.app_secret,
            }

            try:
                async with self._session.post(self.TOKEN_URL, data=data) as response:
                    response_data = await response.json()

                    if response.status != 200:
                        error_msg = response_data.get(
                            "error_description", "Token refresh failed"
                        )
                        logger.error(f"Token refresh failed: {error_msg}")
                        raise ValueError(f"Token refresh failed: {error_msg}")

                    # Create new token object
                    token = OAuthToken(
                        access_token=response_data["access_token"],
                        refresh_token=response_data.get("refresh_token", refresh_token),
                        token_type=response_data.get("token_type", "Bearer"),
                        expires_in=response_data["expires_in"],
                        scope=response_data.get("scope", ""),
                        created_at=time.time(),
                    )

                    # Store refreshed token
                    await self._save_token(token)
                    self._current_token = token

                    logger.info("Successfully refreshed access token")
                    return token

            except aiohttp.ClientError as e:
                logger.error(f"Network error during token refresh: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error during token refresh: {e}")
                raise

    async def get_valid_token(self) -> OAuthToken:
        """
        Get a valid access token, refreshing if necessary.

        Returns:
            Valid OAuthToken

        Raises:
            ValueError: If no token available or refresh fails
        """
        # Load token if not in memory
        if self._current_token is None:
            await self.load_token()

        # Check if token exists
        if self._current_token is None:
            raise ValueError("No token available - please authenticate first")

        # Refresh if expired or expiring soon
        if self._current_token.is_expired:
            logger.info("Token expired, refreshing...")
            try:
                await self.refresh_access_token()
            except Exception as e:
                logger.error(f"Failed to refresh token: {e}")
                raise ValueError("Token refresh failed - please re-authenticate")

        return self._current_token

    async def _save_token(self, token: OAuthToken) -> None:
        """Save token to encrypted storage."""
        try:
            # Serialize token
            token_data = token.model_dump()
            token_json = json.dumps(token_data)

            # Encrypt and save
            encrypted_data = self.cipher.encrypt(token_json.encode())

            # Write atomically
            temp_file = self.token_file.with_suffix(".tmp")
            with open(temp_file, "wb") as f:
                f.write(encrypted_data)
            temp_file.replace(self.token_file)

            # Secure the token file
            os.chmod(self.token_file, 0o600)

            logger.debug("Token saved to encrypted storage")

        except Exception as e:
            logger.error(f"Failed to save token: {e}")
            raise

    async def load_token(self) -> OAuthToken | None:
        """Load token from encrypted storage."""
        try:
            if not self.token_file.exists():
                logger.debug("No stored token found")
                return None

            # Read and decrypt
            with open(self.token_file, "rb") as f:
                encrypted_data = f.read()

            decrypted_data = self.cipher.decrypt(encrypted_data)
            token_data = json.loads(decrypted_data.decode())

            # Create token object
            token = OAuthToken(**token_data)
            self._current_token = token

            logger.debug("Token loaded from encrypted storage")
            return token

        except Exception as e:
            logger.warning(f"Failed to load token: {e}")
            return None

    async def revoke_token(self) -> bool:
        """
        Revoke current token.

        Returns:
            True if revocation successful
        """
        if self._current_token is None:
            return True

        try:
            await self._ensure_session()

            # Schwab doesn't have a standard revoke endpoint, so we'll just delete locally
            # In a production environment, you might want to implement proper revocation

            # Clear stored token
            if self.token_file.exists():
                self.token_file.unlink()

            self._current_token = None

            logger.info("Token revoked successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to revoke token: {e}")
            return False

    def get_authentication_status(self) -> dict[str, Any]:
        """
        Get current authentication status.

        Returns:
            Dictionary with authentication status information
        """
        if self._current_token is None:
            return {
                "authenticated": False,
                "token_exists": False,
                "expires_at": None,
                "time_until_expiry": None,
                "scope": None,
            }

        return {
            "authenticated": not self._current_token.is_expired,
            "token_exists": True,
            "expires_at": datetime.fromtimestamp(
                self._current_token.expires_at
            ).isoformat(),
            "time_until_expiry": self._current_token.time_until_expiry,
            "scope": self._current_token.scope,
            "token_type": self._current_token.token_type,
        }


# Factory function for easy initialization
async def create_oauth_client() -> SchwabOAuthClient:
    """
    Create Schwab OAuth client with configuration from environment.

    Returns:
        Configured SchwabOAuthClient instance
    """
    app_key = os.getenv("SCHWAB_APP_KEY")
    app_secret = os.getenv("SCHWAB_SECRET")
    redirect_uri = os.getenv(
        "SCHWAB_REDIRECT_URI", "http://localhost:8000/auth/schwab/callback"
    )
    token_dir = os.getenv("SCHWAB_TOKEN_DIR", "./data/schwab_tokens")

    if not app_key or not app_secret:
        raise ValueError(
            "SCHWAB_APP_KEY and SCHWAB_SECRET environment variables are required"
        )

    client = SchwabOAuthClient(
        app_key=app_key,
        app_secret=app_secret,
        redirect_uri=redirect_uri,
        token_storage_path=token_dir,
    )

    return client


# Convenience function for authentication flow
async def authenticate_interactive() -> SchwabOAuthClient:
    """
    Run interactive authentication flow.

    Returns:
        Authenticated SchwabOAuthClient
    """
    client = await create_oauth_client()

    try:
        # Try to load existing token
        await client.load_token()
        if client._current_token and not client._current_token.is_expired:
            logger.info("Using existing valid token")
            return client

        # Generate PKCE challenge
        pkce_challenge = client.generate_pkce_challenge()

        # Get authorization URL
        auth_url = client.get_authorization_url(pkce_challenge, scope="readonly")

        print("\n" + "=" * 80)
        print("SCHWAB OAUTH AUTHENTICATION REQUIRED")
        print("=" * 80)
        print("\n1. Open this URL in your browser:")
        print(f"   {auth_url}")
        print("\n2. Log in to your Schwab account and authorize the application")
        print("3. Copy the full callback URL after authorization")
        print("4. Paste it below when prompted")
        print("\n" + "=" * 80)

        # Get callback URL from user
        callback_url = input("\nPaste the callback URL here: ").strip()

        # Extract authorization code
        auth_code, state_valid = client.extract_authorization_code(
            callback_url, pkce_challenge.state
        )

        if not state_valid:
            raise ValueError("State validation failed - possible security issue")

        # Exchange code for token
        await client.exchange_code_for_token(auth_code, pkce_challenge)

        print("\n✅ Authentication successful!")
        print("🔐 Token saved securely and ready for use")

        return client

    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        await client.close()
        raise


if __name__ == "__main__":
    """Interactive authentication for testing."""

    async def main():
        client = await authenticate_interactive()
        status = client.get_authentication_status()
        print(f"\nAuthentication Status: {status}")
        await client.close()

    asyncio.run(main())
