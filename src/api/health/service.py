"""
Health check service implementing business logic for dependency checks.
Follows dependency injection principles for testability.
"""
import asyncio
import os
import time
from typing import Dict, Any
from datetime import datetime, timezone
from src.api.health.models import HealthStatus, HealthResponse
from src.core.config import get_config


class HealthService:
    """Service for performing health checks on application dependencies."""
    
    def __init__(self):
        """Initialize health service with configuration."""
        self.config = get_config()
        self.version = getattr(self.config.app, 'version', '1.0.0')
    
    async def get_health_status(self) -> HealthResponse:
        """
        Get comprehensive health status of the application.
        
        Returns:
            HealthResponse: Complete health check results
        """
        # Perform all dependency checks
        checks = await self._perform_all_checks()
        
        # Determine overall status
        overall_status = self.get_overall_status(checks)
        
        # Generate timestamp
        timestamp = datetime.now(timezone.utc).isoformat()
        
        return HealthResponse(
            status=overall_status,
            timestamp=timestamp,
            version=self.version,
            checks=checks
        )
    
    async def _perform_all_checks(self) -> Dict[str, Any]:
        """
        Perform all dependency checks concurrently.
        
        Returns:
            Dict[str, Any]: Dictionary of check results
        """
        # Run checks concurrently for better performance
        tasks = {
            'database': self.check_database(),
            'cache': self.check_cache(),
            'filesystem': self.check_filesystem(),
        }
        
        # Execute checks with timeout
        results = {}
        for check_name, check_coro in tasks.items():
            try:
                # Add timeout to prevent hanging
                result = await asyncio.wait_for(check_coro, timeout=2.0)
                results[check_name] = result
            except asyncio.TimeoutError:
                results[check_name] = False
            except Exception:
                results[check_name] = False
                
        return results
    
    async def check_database(self) -> bool:
        """
        Check database connectivity and health.
        
        Returns:
            bool: True if database is healthy, False otherwise
        """
        try:
            # Simulate database check - in real implementation would check actual DB
            # For now, return True as we don't have a database configured
            await asyncio.sleep(0.001)  # Simulate async DB call
            return True
        except Exception:
            return False
    
    async def check_cache(self) -> bool:
        """
        Check cache system health (Redis, Memcached, etc.).
        
        Returns:
            bool: True if cache is healthy, False otherwise
        """
        try:
            # Simulate cache check - in real implementation would check actual cache
            await asyncio.sleep(0.001)  # Simulate async cache call
            return True
        except Exception:
            return False
    
    async def check_filesystem(self) -> bool:
        """
        Check filesystem accessibility and disk space.
        
        Returns:
            bool: True if filesystem is healthy, False otherwise
        """
        try:
            # Check if we can write to temp directory
            temp_file = '/tmp/health_check.tmp'
            with open(temp_file, 'w') as f:
                f.write('health_check')
            
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
            return True
        except Exception:
            return False
    
    def get_overall_status(self, checks: Dict[str, Any]) -> HealthStatus:
        """
        Determine overall health status based on individual checks.
        
        Args:
            checks: Dictionary of individual check results
            
        Returns:
            HealthStatus: Overall status (healthy/unhealthy)
        """
        # Application is healthy only if ALL checks pass
        all_healthy = all(
            result is True 
            for result in checks.values()
        )
        
        return HealthStatus.HEALTHY if all_healthy else HealthStatus.UNHEALTHY