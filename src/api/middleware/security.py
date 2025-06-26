"""
Security Middleware for TradeKnowledge API
Provides comprehensive security features including CORS, security headers, and request validation
"""

import re
import time

import structlog
from fastapi import HTTPException, Request, Response, status
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ...core.config import get_config
from ...core.input_validator import ValidationError, sanitize_api_params

logger = structlog.get_logger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Comprehensive security middleware"""

    def __init__(self, app):
        super().__init__(app)
        self.config = get_config()

        # Security patterns
        self.suspicious_patterns = [
            r"<script[^>]*>.*?</script>",  # XSS
            r"javascript:",  # JavaScript protocol
            r"on\w+\s*=",  # Event handlers
            r"\.\./",  # Directory traversal
            r"\.\.\\",  # Windows path traversal
            r"\/etc\/passwd",  # Unix system files
            r"\/proc\/.*",  # Linux proc filesystem
            r"SELECT.*FROM",  # SQL injection
            r"UNION.*SELECT",  # SQL injection
            r"INSERT.*INTO",  # SQL injection
            r"DELETE.*FROM",  # SQL injection
            r"DROP.*TABLE",  # SQL injection
        ]

        # Compile patterns for performance
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.suspicious_patterns
        ]

        # Request size limits
        self.max_request_size = 10 * 1024 * 1024  # 10MB
        self.max_json_size = 1 * 1024 * 1024  # 1MB for JSON

        # Rate tracking for anomaly detection
        self.request_tracker: dict[str, list[float]] = {}

    async def dispatch(self, request: Request, call_next):
        """Process request through security middleware"""
        start_time = time.time()

        try:
            # Check request size
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self.max_request_size:
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={"error": "Request too large"},
                )

            # Validate request path and query parameters
            security_check = await self._check_request_security(request)
            if security_check:
                return security_check

            # Track request for anomaly detection
            await self._track_request(request)

            # Process request
            response = await call_next(request)

            # Add security headers
            response = self._add_security_headers(response)

            # Add rate limit headers if available
            if hasattr(request.state, "rate_limit_headers"):
                for header, value in request.state.rate_limit_headers.items():
                    response.headers[header] = value

            # Log request
            processing_time = (time.time() - start_time) * 1000
            await self._log_request(request, response, processing_time)

            return response

        except Exception as e:
            logger.error("Security middleware error", error=str(e), exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "Internal server error"},
            )

    async def _check_request_security(self, request: Request) -> Response | None:
        """Check request for security violations"""

        # Check URL path
        if self._contains_suspicious_content(str(request.url.path)):
            logger.warning(
                "Suspicious URL path detected",
                path=request.url.path,
                client_ip=self._get_client_ip(request),
            )
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "Invalid request"},
            )

        # Check query parameters
        for key, value in request.query_params.items():
            if self._contains_suspicious_content(
                key
            ) or self._contains_suspicious_content(value):
                logger.warning(
                    "Suspicious query parameter detected",
                    param=key,
                    client_ip=self._get_client_ip(request),
                )
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"error": "Invalid request parameters"},
                )

        # Check headers for suspicious content
        for header_name, header_value in request.headers.items():
            if header_name.lower() in ["user-agent", "referer", "x-forwarded-for"]:
                if self._contains_suspicious_content(header_value):
                    logger.warning(
                        "Suspicious header detected",
                        header=header_name,
                        client_ip=self._get_client_ip(request),
                    )
                    return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content={"error": "Invalid request headers"},
                    )

        return None

    def _contains_suspicious_content(self, content: str) -> bool:
        """Check if content contains suspicious patterns"""
        if not content:
            return False

        # Check against compiled patterns
        for pattern in self.compiled_patterns:
            if pattern.search(content):
                return True

        return False

    async def _track_request(self, request: Request):
        """Track requests for anomaly detection"""
        client_ip = self._get_client_ip(request)
        current_time = time.time()

        # Initialize tracking for new IPs
        if client_ip not in self.request_tracker:
            self.request_tracker[client_ip] = []

        # Add current request time
        self.request_tracker[client_ip].append(current_time)

        # Clean old requests (older than 1 hour)
        cutoff_time = current_time - 3600
        self.request_tracker[client_ip] = [
            req_time
            for req_time in self.request_tracker[client_ip]
            if req_time > cutoff_time
        ]

        # Check for anomalies (more than 1000 requests per hour from single IP)
        if len(self.request_tracker[client_ip]) > 1000:
            logger.warning(
                "High request volume detected",
                client_ip=client_ip,
                request_count=len(self.request_tracker[client_ip]),
            )

    def _add_security_headers(self, response: Response) -> Response:
        """Add security headers to response"""
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        }

        for header, value in security_headers.items():
            response.headers[header] = value

        return response

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        # Check for forwarded headers (behind proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fallback to direct connection
        if hasattr(request.client, "host"):
            return request.client.host

        return "unknown"

    async def _log_request(
        self, request: Request, response: Response, processing_time: float
    ):
        """Log request details"""
        logger.info(
            "Request processed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            processing_time_ms=processing_time,
            client_ip=self._get_client_ip(request),
            user_agent=request.headers.get("User-Agent", "Unknown"),
        )


# CORS Configuration
def get_cors_middleware():
    """Get CORS middleware with security-conscious settings"""
    config = get_config()

    return CORSMiddleware(
        allow_origins=config.cors.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-API-Key",
            "X-Requested-With",
            "Accept",
            "Origin",
            "User-Agent",
        ],
        expose_headers=[
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
            "X-Total-Count",
        ],
    )


# Global security middleware instance
security_middleware = SecurityMiddleware


# Request validation decorator
def validate_request_security(func):
    """Decorator to add security validation to endpoints"""

    async def wrapper(*args, **kwargs):
        try:
            # Extract request from args/kwargs
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if not request:
                # Look in kwargs
                request = kwargs.get("request")

            if request:
                # Validate request parameters
                try:
                    # Sanitize query parameters
                    sanitized_params = {}
                    for key, value in request.query_params.items():
                        sanitized_params[key] = sanitize_api_params({key: value})[key]

                    # Replace query params with sanitized versions
                    request._query_params = sanitized_params

                except ValidationError as e:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid request parameters: {str(e)}",
                    )

            return await func(*args, **kwargs)

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Request validation error", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Request validation failed",
            )

    return wrapper
