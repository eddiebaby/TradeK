"""
Monitoring Middleware
Tracks API requests, response times, and errors for observability
"""

import time
from collections.abc import Callable

import structlog
from fastapi import Request, Response
from fastapi.responses import JSONResponse

from ...core.monitoring_service import get_monitoring_service

logger = structlog.get_logger(__name__)


class MonitoringMiddleware:
    """Middleware for monitoring API requests and responses"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        start_time = time.time()

        # Track request start
        monitoring_service = await get_monitoring_service()

        # Wrap the send function to capture response
        response_captured = {"status_code": 500, "body": b""}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                response_captured["status_code"] = message["status"]
            elif message["type"] == "http.response.body":
                response_captured["body"] = message.get("body", b"")
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            # Record error
            duration_ms = (time.time() - start_time) * 1000
            monitoring_service.record_request(duration_ms, 500)

            logger.error(
                "Request failed with exception",
                method=request.method,
                path=request.url.path,
                duration_ms=duration_ms,
                error=str(e),
            )

            # Re-raise the exception
            raise
        else:
            # Record successful request
            duration_ms = (time.time() - start_time) * 1000
            status_code = response_captured["status_code"]

            monitoring_service.record_request(duration_ms, status_code)

            # Log request details
            log_level = "info"
            if status_code >= 500:
                log_level = "error"
            elif status_code >= 400:
                log_level = "warning"

            logger.log(
                log_level,
                "Request completed",
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                duration_ms=duration_ms,
                user_agent=request.headers.get("user-agent", ""),
                client_ip=request.client.host if request.client else "unknown",
            )


async def add_monitoring_headers(request: Request, call_next: Callable) -> Response:
    """
    Add monitoring and observability headers to responses
    """
    start_time = time.time()

    # Generate request ID for tracing
    import uuid

    request_id = str(uuid.uuid4())[:8]

    # Add request ID to request state
    request.state.request_id = request_id

    try:
        response = await call_next(request)

        # Calculate response time
        process_time = time.time() - start_time

        # Add monitoring headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))
        response.headers["X-Server-Name"] = "TradeKnowledge"

        return response

    except Exception as e:
        logger.error(
            "Request processing failed",
            request_id=request_id,
            error=str(e),
            path=request.url.path,
        )

        # Return error response with monitoring headers
        error_response = JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "request_id": request_id,
                "timestamp": time.time(),
            },
        )

        error_response.headers["X-Request-ID"] = request_id
        error_response.headers["X-Process-Time"] = str(
            round((time.time() - start_time) * 1000, 2)
        )

        return error_response


async def request_size_monitor(request: Request, call_next: Callable) -> Response:
    """
    Monitor request and response sizes
    """
    # Get request size
    request_size = 0
    if request.headers.get("content-length"):
        try:
            request_size = int(request.headers["content-length"])
        except ValueError:
            pass

    # Process request
    response = await call_next(request)

    # Estimate response size
    response_size = 0
    if hasattr(response, "body") and response.body:
        response_size = len(response.body)
    elif hasattr(response, "headers") and response.headers.get("content-length"):
        try:
            response_size = int(response.headers["content-length"])
        except ValueError:
            pass

    # Record size metrics
    try:
        monitoring_service = await get_monitoring_service()
        monitoring_service.metrics_collector.record_metric(
            "request_size_bytes", request_size
        )
        monitoring_service.metrics_collector.record_metric(
            "response_size_bytes", response_size
        )
    except Exception as e:
        logger.warning("Failed to record size metrics", error=str(e))

    return response


class RateLimitMonitoringMiddleware:
    """
    Middleware to monitor rate limiting events
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)

        # Check for rate limit headers in response
        rate_limit_info = {"limited": False, "remaining": None, "reset_time": None}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))

                # Check for rate limit headers
                if b"x-ratelimit-remaining" in headers:
                    try:
                        rate_limit_info["remaining"] = int(
                            headers[b"x-ratelimit-remaining"].decode()
                        )
                    except:
                        pass

                if b"x-ratelimit-reset" in headers:
                    try:
                        rate_limit_info["reset_time"] = int(
                            headers[b"x-ratelimit-reset"].decode()
                        )
                    except:
                        pass

                # Check if request was rate limited
                if message["status"] == 429:
                    rate_limit_info["limited"] = True

            await send(message)

        await self.app(scope, receive, send_wrapper)

        # Record rate limit metrics
        try:
            monitoring_service = await get_monitoring_service()

            if rate_limit_info["limited"]:
                monitoring_service.metrics_collector.record_metric(
                    "rate_limit_exceeded", 1
                )
                logger.warning(
                    "Rate limit exceeded",
                    path=request.url.path,
                    client_ip=request.client.host if request.client else "unknown",
                )

            if rate_limit_info["remaining"] is not None:
                monitoring_service.metrics_collector.record_metric(
                    "rate_limit_remaining", rate_limit_info["remaining"]
                )

        except Exception as e:
            logger.warning("Failed to record rate limit metrics", error=str(e))


async def error_tracking_middleware(request: Request, call_next: Callable) -> Response:
    """
    Track and categorize errors for monitoring
    """
    try:
        response = await call_next(request)

        # Track error responses
        if response.status_code >= 400:
            monitoring_service = await get_monitoring_service()

            # Categorize errors
            error_category = (
                "client_error" if response.status_code < 500 else "server_error"
            )

            monitoring_service.metrics_collector.record_metric(
                f"errors_{error_category}",
                1,
                labels={
                    "status_code": str(response.status_code),
                    "endpoint": request.url.path,
                    "method": request.method,
                },
            )

            # Log error details
            logger.warning(
                "HTTP error response",
                status_code=response.status_code,
                path=request.url.path,
                method=request.method,
                error_category=error_category,
                request_id=getattr(request.state, "request_id", "unknown"),
            )

        return response

    except Exception as e:
        # Track unhandled exceptions
        try:
            monitoring_service = await get_monitoring_service()
            monitoring_service.metrics_collector.record_metric(
                "unhandled_exceptions",
                1,
                labels={
                    "exception_type": type(e).__name__,
                    "endpoint": request.url.path,
                    "method": request.method,
                },
            )
        except:
            pass  # Don't let monitoring failures break the request

        raise  # Re-raise the original exception


def setup_monitoring_middleware(app):
    """
    Set up all monitoring middleware on the FastAPI app
    """
    # Add middleware in reverse order (last added is executed first)

    # Error tracking (innermost)
    app.middleware("http")(error_tracking_middleware)

    # Request/response size monitoring
    app.middleware("http")(request_size_monitor)

    # Add monitoring headers and request ID
    app.middleware("http")(add_monitoring_headers)

    # Rate limit monitoring
    app.add_middleware(RateLimitMonitoringMiddleware)

    # Main monitoring middleware (outermost)
    app.add_middleware(MonitoringMiddleware)

    logger.info("✅ Monitoring middleware configured")
