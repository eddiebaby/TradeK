"""
Health check middleware for rate limiting and security.
Provides protection against health check endpoint abuse.
"""
import time
from typing import Dict, Any
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict, deque


class HealthCheckRateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware specifically for health check endpoints.
    Prevents abuse while allowing legitimate monitoring.
    """
    
    def __init__(self, app, requests_per_minute: int = 120):
        """
        Initialize rate limiting middleware.
        
        Args:
            app: FastAPI application
            requests_per_minute: Maximum requests per minute per IP
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_times: Dict[str, deque] = defaultdict(deque)
        
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process request with rate limiting for health endpoints.
        
        Args:
            request: HTTP request
            call_next: Next middleware in chain
            
        Returns:
            Response: HTTP response
        """
        # Only apply rate limiting to health endpoints
        if not request.url.path.startswith('/health'):
            return await call_next(request)
        
        # Get client IP
        client_ip = self._get_client_ip(request)
        
        # Check rate limit
        current_time = time.time()
        if self._is_rate_limited(client_ip, current_time):
            return Response(
                content='{"error": "Rate limit exceeded for health checks"}',
                status_code=429,
                headers={"Content-Type": "application/json"}
            )
        
        # Record request time
        self._record_request(client_ip, current_time)
        
        return await call_next(request)
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check forwarded headers first (load balancer/proxy)
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('X-Real-IP')
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        return request.client.host if request.client else 'unknown'
    
    def _is_rate_limited(self, client_ip: str, current_time: float) -> bool:
        """Check if client has exceeded rate limit."""
        request_times = self.request_times[client_ip]
        
        # Remove old requests (older than 1 minute)
        cutoff_time = current_time - 60
        while request_times and request_times[0] < cutoff_time:
            request_times.popleft()
        
        # Check if limit exceeded
        return len(request_times) >= self.requests_per_minute
    
    def _record_request(self, client_ip: str, current_time: float):
        """Record request time for rate limiting."""
        self.request_times[client_ip].append(current_time)