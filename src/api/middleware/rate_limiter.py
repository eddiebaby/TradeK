"""
Rate Limiting Middleware for TradeKnowledge API
Provides configurable rate limiting with Redis backend support
"""

import time
from dataclasses import dataclass
from enum import Enum

import structlog
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse

from ...core.config import get_config
from ...core.models import APIKey, User

logger = structlog.get_logger(__name__)


class RateLimitType(Enum):
    USER = "user"
    API_KEY = "api_key"
    IP = "ip"
    GLOBAL = "global"


@dataclass
class RateLimitRule:
    """Rate limit rule configuration"""

    limit: int  # Number of requests
    window: int  # Time window in seconds
    burst_limit: int | None = None  # Burst allowance


class RateLimitResult:
    """Result of rate limit check"""

    def __init__(
        self,
        allowed: bool,
        remaining: int,
        reset_time: int,
        retry_after: int | None = None,
    ):
        self.allowed = allowed
        self.remaining = remaining
        self.reset_time = reset_time
        self.retry_after = retry_after


class MemoryRateLimiter:
    """In-memory rate limiter using sliding window"""

    def __init__(self):
        self.requests: dict[str, list] = {}
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()

    async def check_rate_limit(self, key: str, rule: RateLimitRule) -> RateLimitResult:
        """Check if request is allowed under rate limit"""
        current_time = time.time()
        window_start = current_time - rule.window

        # Cleanup old entries periodically
        if current_time - self.last_cleanup > self.cleanup_interval:
            await self._cleanup_old_entries(current_time)
            self.last_cleanup = current_time

        # Get or create request list for this key
        if key not in self.requests:
            self.requests[key] = []

        request_list = self.requests[key]

        # Remove expired requests from sliding window
        self.requests[key] = [
            req_time for req_time in request_list if req_time > window_start
        ]
        request_list = self.requests[key]

        # Check if limit exceeded
        if len(request_list) >= rule.limit:
            # Calculate when the oldest request will expire
            oldest_request = min(request_list)
            retry_after = int(oldest_request + rule.window - current_time)

            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=int(oldest_request + rule.window),
                retry_after=max(1, retry_after),
            )

        # Add current request
        self.requests[key].append(current_time)

        remaining = rule.limit - len(self.requests[key])
        reset_time = int(current_time + rule.window)

        return RateLimitResult(allowed=True, remaining=remaining, reset_time=reset_time)

    async def _cleanup_old_entries(self, current_time: float):
        """Remove old entries to prevent memory leaks"""
        keys_to_remove = []

        for key, request_list in self.requests.items():
            # Remove requests older than 1 hour
            cutoff_time = current_time - 3600
            filtered_requests = [
                req_time for req_time in request_list if req_time > cutoff_time
            ]

            if filtered_requests:
                self.requests[key] = filtered_requests
            else:
                keys_to_remove.append(key)

        # Remove empty entries
        for key in keys_to_remove:
            del self.requests[key]

        logger.debug(f"Cleaned up {len(keys_to_remove)} old rate limit entries")


class RedisRateLimiter:
    """Redis-based rate limiter for distributed systems"""

    def __init__(self, redis_client):
        self.redis = redis_client

    async def check_rate_limit(self, key: str, rule: RateLimitRule) -> RateLimitResult:
        """Check rate limit using Redis sliding window"""
        current_time = time.time()
        window_start = current_time - rule.window

        # Use Redis pipeline for atomic operations
        pipe = self.redis.pipeline()

        try:
            # Remove expired entries
            pipe.zremrangebyscore(key, 0, window_start)

            # Count current requests in window
            pipe.zcard(key)

            # Add current request with score as timestamp
            pipe.zadd(key, {str(current_time): current_time})

            # Set expiry for the key
            pipe.expire(key, rule.window + 60)  # Extra buffer

            # Execute pipeline
            results = await pipe.execute()

            current_count = results[1]  # Count after cleanup, before adding

            if current_count >= rule.limit:
                # Get oldest request time for retry calculation
                oldest = await self.redis.zrange(key, 0, 0, withscores=True)
                if oldest:
                    oldest_time = oldest[0][1]
                    retry_after = int(oldest_time + rule.window - current_time)

                    return RateLimitResult(
                        allowed=False,
                        remaining=0,
                        reset_time=int(oldest_time + rule.window),
                        retry_after=max(1, retry_after),
                    )

            remaining = rule.limit - (current_count + 1)
            reset_time = int(current_time + rule.window)

            return RateLimitResult(
                allowed=True, remaining=remaining, reset_time=reset_time
            )

        except Exception as e:
            logger.error(f"Redis rate limiter error: {e}")
            # Fallback to allowing request if Redis fails
            return RateLimitResult(
                allowed=True,
                remaining=rule.limit - 1,
                reset_time=int(current_time + rule.window),
            )


class RateLimitMiddleware:
    """Comprehensive rate limiting middleware"""

    def __init__(self):
        self.config = get_config()
        self.memory_limiter = MemoryRateLimiter()
        self.redis_limiter = None

        # Rate limit rules
        self.rules = {
            RateLimitType.GLOBAL: RateLimitRule(
                limit=self.config.rate_limiting.global_requests_per_minute * 60,
                window=3600,  # 1 hour
            ),
            RateLimitType.IP: RateLimitRule(
                limit=self.config.rate_limiting.ip_requests_per_minute * 60,
                window=3600,  # 1 hour
            ),
            RateLimitType.USER: RateLimitRule(
                limit=self.config.rate_limiting.user_requests_per_hour,
                window=3600,  # 1 hour
            ),
            RateLimitType.API_KEY: RateLimitRule(
                limit=1000,  # Default, will be overridden by API key settings
                window=3600,  # 1 hour
            ),
        }

    async def initialize(self):
        """Initialize Redis connection if available"""
        try:
            if hasattr(self.config, "redis") and self.config.redis.enabled:
                import aioredis

                self.redis_client = aioredis.from_url(
                    self.config.redis.url, decode_responses=True
                )
                self.redis_limiter = RedisRateLimiter(self.redis_client)
                logger.info("Rate limiter initialized with Redis backend")
            else:
                logger.info("Rate limiter initialized with memory backend")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis for rate limiting: {e}")
            logger.info("Falling back to memory-based rate limiting")

    async def check_rate_limits(
        self,
        request: Request,
        user: User | None = None,
        api_key: APIKey | None = None,
    ) -> Response | None:
        """Check all applicable rate limits"""

        # Get client IP
        client_ip = self._get_client_ip(request)

        # Check global rate limit
        global_key = "global"
        global_result = await self._check_limit(
            global_key, self.rules[RateLimitType.GLOBAL]
        )
        if not global_result.allowed:
            return self._create_rate_limit_response(
                global_result, "Global rate limit exceeded"
            )

        # Check IP-based rate limit
        ip_key = f"ip:{client_ip}"
        ip_result = await self._check_limit(ip_key, self.rules[RateLimitType.IP])
        if not ip_result.allowed:
            return self._create_rate_limit_response(ip_result, "IP rate limit exceeded")

        # Check user-specific rate limit
        if user:
            user_key = f"user:{user.id}"
            user_result = await self._check_limit(
                user_key, self.rules[RateLimitType.USER]
            )
            if not user_result.allowed:
                return self._create_rate_limit_response(
                    user_result, "User rate limit exceeded"
                )

        # Check API key rate limit
        if api_key:
            api_key_rule = RateLimitRule(
                limit=api_key.rate_limit, window=3600  # 1 hour
            )
            api_key_result = await self._check_limit(
                f"api_key:{api_key.key_id}", api_key_rule
            )
            if not api_key_result.allowed:
                return self._create_rate_limit_response(
                    api_key_result, "API key rate limit exceeded"
                )

        # Add rate limit headers to successful requests
        headers = self._create_rate_limit_headers(user_result if user else ip_result)

        # Store headers for later addition to response
        request.state.rate_limit_headers = headers

        return None  # Allow request

    async def _check_limit(self, key: str, rule: RateLimitRule) -> RateLimitResult:
        """Check rate limit using available backend"""
        if self.redis_limiter:
            return await self.redis_limiter.check_rate_limit(key, rule)
        else:
            return await self.memory_limiter.check_rate_limit(key, rule)

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

    def _create_rate_limit_response(
        self, result: RateLimitResult, message: str
    ) -> Response:
        """Create rate limit exceeded response"""
        headers = {
            "X-RateLimit-Limit": str(self.rules[RateLimitType.USER].limit),
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(result.reset_time),
        }

        if result.retry_after:
            headers["Retry-After"] = str(result.retry_after)

        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "Rate limit exceeded",
                "message": message,
                "retry_after": result.retry_after,
            },
            headers=headers,
        )

    def _create_rate_limit_headers(self, result: RateLimitResult) -> dict[str, str]:
        """Create rate limit headers for successful responses"""
        return {
            "X-RateLimit-Limit": str(self.rules[RateLimitType.USER].limit),
            "X-RateLimit-Remaining": str(result.remaining),
            "X-RateLimit-Reset": str(result.reset_time),
        }

    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.redis_limiter and hasattr(self, "redis_client"):
                await self.redis_client.close()
        except Exception as e:
            logger.error(f"Error during rate limiter cleanup: {e}")


# Global rate limiter instance
rate_limiter = RateLimitMiddleware()


async def get_rate_limiter() -> RateLimitMiddleware:
    """Get global rate limiter instance"""
    return rate_limiter


# FastAPI dependency for rate limiting
async def check_rate_limits(
    request: Request, user: User | None = None, api_key: APIKey | None = None
) -> Response | None:
    """FastAPI dependency to check rate limits"""
    return await rate_limiter.check_rate_limits(request, user, api_key)
