"""
Schwab Token Manager

Provides secure token management with automatic refresh, encryption,
backup/recovery, and comprehensive monitoring for Schwab OAuth tokens.

Features:
- Automatic token refresh with configurable timing
- Encrypted token storage with key rotation
- Background refresh monitoring
- Token validation and health checks
- Backup and recovery mechanisms
- Performance metrics and monitoring
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .oauth_client import OAuthToken, SchwabOAuthClient

logger = logging.getLogger(__name__)


@dataclass
class TokenMetrics:
    """Token performance and health metrics."""

    refresh_count: int = 0
    refresh_failures: int = 0
    last_refresh_time: float | None = None
    last_refresh_duration: float | None = None
    token_age_seconds: float = 0
    refresh_success_rate: float = 100.0
    next_refresh_time: float | None = None

    def calculate_success_rate(self) -> float:
        """Calculate token refresh success rate."""
        if self.refresh_count == 0:
            return 100.0
        return (
            (self.refresh_count - self.refresh_failures) / self.refresh_count
        ) * 100.0


class TokenRefreshScheduler:
    """Handles automatic token refresh scheduling."""

    def __init__(self, refresh_buffer_minutes: int = 5):
        """
        Initialize refresh scheduler.

        Args:
            refresh_buffer_minutes: Minutes before expiry to refresh token
        """
        self.refresh_buffer_minutes = refresh_buffer_minutes
        self.refresh_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._callbacks: list[Callable] = []

    def add_refresh_callback(self, callback: Callable[[OAuthToken], None]) -> None:
        """Add callback to be called after successful refresh."""
        self._callbacks.append(callback)

    async def start_monitoring(self, token_manager: "SchwabTokenManager") -> None:
        """Start automatic refresh monitoring."""
        if self.refresh_task and not self.refresh_task.done():
            logger.warning("Refresh monitoring already running")
            return

        self.refresh_task = asyncio.create_task(self._refresh_loop(token_manager))
        logger.info("Token refresh monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop automatic refresh monitoring."""
        self._stop_event.set()
        if self.refresh_task:
            try:
                await self.refresh_task
            except asyncio.CancelledError:
                pass
        logger.info("Token refresh monitoring stopped")

    async def _refresh_loop(self, token_manager: "SchwabTokenManager") -> None:
        """Main refresh monitoring loop."""
        while not self._stop_event.is_set():
            try:
                # Check if refresh is needed
                token = await token_manager.get_current_token()
                if token and self._should_refresh(token):
                    logger.info("Token refresh required, initiating refresh...")
                    refreshed_token = await token_manager.refresh_token()

                    # Notify callbacks
                    for callback in self._callbacks:
                        try:
                            await callback(refreshed_token)
                        except Exception as e:
                            logger.error(f"Token refresh callback failed: {e}")

                # Wait before next check (check every minute)
                await asyncio.wait_for(self._stop_event.wait(), timeout=60)

            except TimeoutError:
                # Timeout is expected for periodic checks
                continue
            except Exception as e:
                logger.error(f"Error in token refresh loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    def _should_refresh(self, token: OAuthToken) -> bool:
        """Check if token should be refreshed."""
        buffer_seconds = self.refresh_buffer_minutes * 60
        return token.time_until_expiry <= buffer_seconds


class SchwabTokenManager:
    """
    Advanced token manager for Schwab OAuth tokens.

    Provides comprehensive token lifecycle management including:
    - Secure storage with encryption
    - Automatic refresh with configurable timing
    - Health monitoring and metrics
    - Backup and recovery
    - Performance optimization
    """

    def __init__(
        self,
        oauth_client: SchwabOAuthClient,
        refresh_buffer_minutes: int = 5,
        enable_auto_refresh: bool = True,
        backup_enabled: bool = True,
    ):
        """
        Initialize token manager.

        Args:
            oauth_client: Configured Schwab OAuth client
            refresh_buffer_minutes: Minutes before expiry to refresh
            enable_auto_refresh: Enable automatic background refresh
            backup_enabled: Enable token backup functionality
        """
        self.oauth_client = oauth_client
        self.refresh_buffer_minutes = refresh_buffer_minutes
        self.enable_auto_refresh = enable_auto_refresh
        self.backup_enabled = backup_enabled

        # Token state
        self._current_token: OAuthToken | None = None
        self._refresh_lock = asyncio.Lock()

        # Metrics and monitoring
        self.metrics = TokenMetrics()

        # Auto-refresh scheduler
        self.scheduler = TokenRefreshScheduler(refresh_buffer_minutes)

        # Backup configuration
        if backup_enabled:
            self.backup_dir = Path(oauth_client.token_storage_path) / "backups"
            self.backup_dir.mkdir(exist_ok=True)
            self.max_backups = 10

        logger.info("Schwab token manager initialized")

    async def initialize(self) -> bool:
        """
        Initialize token manager and load existing token.

        Returns:
            True if token is available and valid
        """
        try:
            # Load existing token
            token = await self.oauth_client.load_token()
            if token:
                self._current_token = token
                self.metrics.token_age_seconds = time.time() - token.created_at
                logger.info("Existing token loaded successfully")

                # Start auto-refresh if enabled
                if self.enable_auto_refresh:
                    await self.scheduler.start_monitoring(self)

                return not token.is_expired
            else:
                logger.info("No existing token found")
                return False

        except Exception as e:
            logger.error(f"Failed to initialize token manager: {e}")
            return False

    async def authenticate(self, pkce_challenge, authorization_code: str) -> OAuthToken:
        """
        Authenticate and store new token.

        Args:
            pkce_challenge: PKCE challenge from OAuth flow
            authorization_code: Authorization code from callback

        Returns:
            New OAuth token
        """
        try:
            # Exchange code for token
            token = await self.oauth_client.exchange_code_for_token(
                authorization_code, pkce_challenge
            )

            self._current_token = token
            self.metrics.token_age_seconds = 0

            # Create backup
            if self.backup_enabled:
                await self._backup_token(token)

            # Start auto-refresh
            if self.enable_auto_refresh:
                await self.scheduler.start_monitoring(self)

            logger.info("Authentication completed and token stored")
            return token

        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise

    async def get_current_token(self) -> OAuthToken | None:
        """Get current token without refresh."""
        return self._current_token

    async def get_valid_token(self) -> OAuthToken:
        """
        Get a valid token, refreshing if necessary.

        Returns:
            Valid OAuth token

        Raises:
            ValueError: If no token available or refresh fails
        """
        if self._current_token is None:
            raise ValueError("No token available - please authenticate first")

        # Check if refresh is needed
        if self._should_refresh_now():
            await self.refresh_token()

        if self._current_token.is_expired:
            raise ValueError("Token expired and refresh failed")

        return self._current_token

    async def refresh_token(self) -> OAuthToken:
        """
        Refresh current token with comprehensive error handling.

        Returns:
            Refreshed OAuth token
        """
        async with self._refresh_lock:
            if self._current_token is None:
                raise ValueError("No token to refresh")

            start_time = time.time()

            try:
                logger.info("Refreshing OAuth token...")

                # Attempt token refresh
                refreshed_token = await self.oauth_client.refresh_access_token(
                    self._current_token.refresh_token
                )

                # Update state
                self._current_token = refreshed_token

                # Update metrics
                refresh_duration = time.time() - start_time
                self.metrics.refresh_count += 1
                self.metrics.last_refresh_time = time.time()
                self.metrics.last_refresh_duration = refresh_duration
                self.metrics.token_age_seconds = 0
                self.metrics.refresh_success_rate = (
                    self.metrics.calculate_success_rate()
                )

                # Create backup
                if self.backup_enabled:
                    await self._backup_token(refreshed_token)

                logger.info(f"Token refreshed successfully in {refresh_duration:.2f}s")
                return refreshed_token

            except Exception as e:
                self.metrics.refresh_failures += 1
                self.metrics.refresh_success_rate = (
                    self.metrics.calculate_success_rate()
                )

                logger.error(f"Token refresh failed: {e}")

                # Try to recover from backup if available
                if self.backup_enabled:
                    recovery_token = await self._recover_from_backup()
                    if recovery_token:
                        self._current_token = recovery_token
                        logger.warning("Recovered token from backup")
                        return recovery_token

                raise ValueError(f"Token refresh failed: {e}")

    def _should_refresh_now(self) -> bool:
        """Check if token should be refreshed immediately."""
        if self._current_token is None:
            return False

        buffer_seconds = self.refresh_buffer_minutes * 60
        return self._current_token.time_until_expiry <= buffer_seconds

    async def _backup_token(self, token: OAuthToken) -> None:
        """Create encrypted backup of token."""
        if not self.backup_enabled:
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"token_backup_{timestamp}.encrypted"

            # Save backup using OAuth client's encryption
            temp_token_file = self.oauth_client.token_file
            self.oauth_client.token_file = backup_file
            await self.oauth_client._save_token(token)
            self.oauth_client.token_file = temp_token_file

            # Cleanup old backups
            await self._cleanup_old_backups()

            logger.debug(f"Token backup created: {backup_file}")

        except Exception as e:
            logger.warning(f"Failed to create token backup: {e}")

    async def _recover_from_backup(self) -> OAuthToken | None:
        """Attempt to recover token from latest backup."""
        if not self.backup_enabled:
            return None

        try:
            # Find latest backup
            backup_files = sorted(
                self.backup_dir.glob("token_backup_*.encrypted"),
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )

            if not backup_files:
                logger.warning("No token backups found")
                return None

            # Try to load latest backup
            latest_backup = backup_files[0]
            temp_token_file = self.oauth_client.token_file
            self.oauth_client.token_file = latest_backup

            token = await self.oauth_client.load_token()
            self.oauth_client.token_file = temp_token_file

            if token and not token.is_expired:
                logger.info(f"Token recovered from backup: {latest_backup}")
                return token
            else:
                logger.warning("Backup token is expired or invalid")
                return None

        except Exception as e:
            logger.error(f"Token recovery from backup failed: {e}")
            return None

    async def _cleanup_old_backups(self) -> None:
        """Remove old backup files beyond max_backups limit."""
        try:
            backup_files = sorted(
                self.backup_dir.glob("token_backup_*.encrypted"),
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )

            # Remove excess backups
            for old_backup in backup_files[self.max_backups :]:
                old_backup.unlink()
                logger.debug(f"Removed old backup: {old_backup}")

        except Exception as e:
            logger.warning(f"Failed to cleanup old backups: {e}")

    async def revoke_token(self) -> bool:
        """
        Revoke current token and cleanup.

        Returns:
            True if revocation successful
        """
        try:
            # Stop auto-refresh
            await self.scheduler.stop_monitoring()

            # Revoke with OAuth client
            success = await self.oauth_client.revoke_token()

            # Clear local state
            self._current_token = None
            self.metrics = TokenMetrics()

            # Cleanup backups if desired
            if self.backup_enabled:
                try:
                    for backup_file in self.backup_dir.glob("token_backup_*.encrypted"):
                        backup_file.unlink()
                    logger.info("Token backups cleaned up")
                except Exception as e:
                    logger.warning(f"Failed to cleanup backups: {e}")

            logger.info("Token revoked and cleaned up")
            return success

        except Exception as e:
            logger.error(f"Token revocation failed: {e}")
            return False

    async def get_token_status(self) -> dict[str, Any]:
        """
        Get comprehensive token status and metrics.

        Returns:
            Dictionary with token status information
        """
        oauth_status = self.oauth_client.get_authentication_status()

        status = {
            **oauth_status,
            "auto_refresh_enabled": self.enable_auto_refresh,
            "refresh_buffer_minutes": self.refresh_buffer_minutes,
            "backup_enabled": self.backup_enabled,
            "metrics": {
                "refresh_count": self.metrics.refresh_count,
                "refresh_failures": self.metrics.refresh_failures,
                "refresh_success_rate": self.metrics.refresh_success_rate,
                "token_age_seconds": self.metrics.token_age_seconds,
                "last_refresh_time": self.metrics.last_refresh_time,
                "last_refresh_duration": self.metrics.last_refresh_duration,
            },
        }

        # Add next refresh prediction
        if self._current_token:
            buffer_seconds = self.refresh_buffer_minutes * 60
            next_refresh = self._current_token.expires_at - buffer_seconds
            status["next_refresh_time"] = next_refresh
            status["minutes_until_refresh"] = max(0, (next_refresh - time.time()) / 60)

        return status

    async def health_check(self) -> dict[str, Any]:
        """
        Perform comprehensive health check.

        Returns:
            Health status with recommendations
        """
        health = {"healthy": True, "warnings": [], "errors": [], "recommendations": []}

        # Check token availability
        if self._current_token is None:
            health["healthy"] = False
            health["errors"].append("No token available")
            health["recommendations"].append("Perform authentication")
        else:
            # Check token expiry
            if self._current_token.is_expired:
                health["healthy"] = False
                health["errors"].append("Token expired")
                health["recommendations"].append("Refresh token immediately")
            elif self._should_refresh_now():
                health["warnings"].append("Token expiring soon")
                health["recommendations"].append("Schedule token refresh")

        # Check refresh success rate
        if self.metrics.refresh_success_rate < 95.0:
            health["warnings"].append(
                f"Low refresh success rate: {self.metrics.refresh_success_rate:.1f}%"
            )
            health["recommendations"].append(
                "Monitor network connectivity and API status"
            )

        # Check token age
        if self.metrics.token_age_seconds > 24 * 3600:  # 24 hours
            health["warnings"].append("Token is very old")
            health["recommendations"].append("Consider re-authentication")

        return health

    async def cleanup(self) -> None:
        """Cleanup resources and stop background tasks."""
        await self.scheduler.stop_monitoring()
        await self.oauth_client.close()
        logger.info("Token manager cleanup completed")


# Factory function for easy initialization
async def create_token_manager(
    enable_auto_refresh: bool = True, refresh_buffer_minutes: int = 5
) -> SchwabTokenManager:
    """
    Create configured token manager.

    Args:
        enable_auto_refresh: Enable automatic token refresh
        refresh_buffer_minutes: Minutes before expiry to refresh

    Returns:
        Configured SchwabTokenManager
    """
    from .oauth_client import create_oauth_client

    oauth_client = await create_oauth_client()

    token_manager = SchwabTokenManager(
        oauth_client=oauth_client,
        refresh_buffer_minutes=refresh_buffer_minutes,
        enable_auto_refresh=enable_auto_refresh,
    )

    await token_manager.initialize()
    return token_manager


if __name__ == "__main__":
    """Token manager testing and monitoring."""

    async def main():
        try:
            manager = await create_token_manager()

            # Display status
            status = await manager.get_token_status()
            print(f"Token Status: {status}")

            # Health check
            health = await manager.health_check()
            print(f"Health Check: {health}")

            # Wait for a bit to test auto-refresh
            print("Monitoring for 60 seconds...")
            await asyncio.sleep(60)

        finally:
            await manager.cleanup()

    asyncio.run(main())
