"""
Memory Middleware for TradeKnowledge
Automatic memory integration for APIs and agent interactions
"""

import asyncio
import logging
import time
from collections.abc import Callable
from datetime import datetime
from functools import wraps
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from .memory_manager import MemoryEvent, get_memory_manager

logger = logging.getLogger(__name__)


class MemoryMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware that automatically captures API interactions in memory
    Only captures significant interactions to minimize overhead
    """

    def __init__(self, app, significance_threshold: float = 0.7):
        super().__init__(app)
        self.significance_threshold = significance_threshold

        # API endpoints worth tracking
        self.tracked_endpoints = {
            "/analyze/",
            "/search/",
            "/recommend/",
            "/backtest/",
            "/sparc/",
            "/strategy/",
            "/portfolio/",
        }

        # Response codes that indicate significant events
        self.significant_status_codes = {200, 201, 400, 500}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and capture significant interactions"""
        start_time = time.time()

        # Skip non-significant endpoints
        if not self._should_track_endpoint(request.url.path):
            return await call_next(request)

        # Capture request context
        request_context = await self._extract_request_context(request)

        # Process request
        response = await call_next(request)

        # Calculate significance and store if worthwhile
        duration = time.time() - start_time
        significance = self._calculate_significance(request, response, duration)

        if significance >= self.significance_threshold:
            await self._store_api_interaction(
                request_context, response, duration, significance
            )

        return response

    def _should_track_endpoint(self, path: str) -> bool:
        """Check if endpoint should be tracked"""
        return any(tracked in path for tracked in self.tracked_endpoints)

    async def _extract_request_context(self, request: Request) -> dict[str, Any]:
        """Extract relevant context from request"""
        context = {
            "method": request.method,
            "path": request.url.path,
            "user_agent": request.headers.get("user-agent", "unknown"),
            "timestamp": datetime.now().isoformat(),
        }

        # Extract user ID if available
        if hasattr(request.state, "user_id"):
            context["user_id"] = request.state.user_id

        # Extract query parameters for analysis endpoints
        if request.query_params:
            relevant_params = {}
            for key, value in request.query_params.items():
                if key in ["symbol", "strategy", "timeframe", "analysis_type"]:
                    relevant_params[key] = value
            context["query_params"] = relevant_params

        # Extract request body for POST requests (sample only)
        if request.method in [
            "POST",
            "PUT",
        ] and "application/json" in request.headers.get("content-type", ""):
            try:
                # Note: This is a simplified example. In real implementation,
                # you'd need to handle request body reading more carefully
                context["has_request_body"] = True
            except:
                context["has_request_body"] = False

        return context

    def _calculate_significance(
        self, request: Request, response: Response, duration: float
    ) -> float:
        """Calculate significance score for this API interaction"""
        significance = 0.5  # Base score

        # High significance for analysis endpoints
        if "/analyze/" in request.url.path:
            significance += 0.3

        # High significance for SPARC agent interactions
        if "/sparc/" in request.url.path:
            significance += 0.2

        # High significance for errors
        if response.status_code >= 400:
            significance += 0.4

        # High significance for slow requests (indicates complex operations)
        if duration > 2.0:
            significance += 0.2

        # High significance for successful analysis results
        if response.status_code == 200 and "/analyze/" in request.url.path:
            significance += 0.3

        return min(significance, 1.0)

    async def _store_api_interaction(
        self,
        request_context: dict[str, Any],
        response: Response,
        duration: float,
        significance: float,
    ):
        """Store significant API interaction in memory"""
        try:
            memory = await get_memory_manager()

            # Create memory event
            event = MemoryEvent(
                event_type="api_interaction",
                entity_id=f"api_{request_context['path'].replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}",
                context={
                    **request_context,
                    "status_code": response.status_code,
                    "duration_seconds": duration,
                    "related_entities": (
                        {
                            f"user_{request_context.get('user_id', 'anonymous')}": "performed_for"
                        }
                        if request_context.get("user_id")
                        else {}
                    ),
                },
                significance_score=significance,
                timestamp=datetime.now(),
            )

            # Store asynchronously to not block response
            asyncio.create_task(memory.store_significant_event(event))

        except Exception as e:
            logger.error(f"Failed to store API interaction in memory: {e}")


# Decorator for SPARC Agent Methods
def sparc_memory_aware(agent_name: str, significance_threshold: float = 0.8):
    """
    Decorator for SPARC agent methods to automatically capture interactions

    Usage:
    @sparc_memory_aware("RESEARCHER", 0.8)
    async def analyze_market(self, symbol: str):
        # Agent logic here
        return results
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                # Execute the original function
                result = await func(*args, **kwargs)

                # Calculate duration and significance
                duration = time.time() - start_time
                significance = _calculate_agent_significance(
                    agent_name, func.__name__, result, duration
                )

                if significance >= significance_threshold:
                    await _store_agent_interaction(
                        agent_name,
                        func.__name__,
                        args,
                        kwargs,
                        result,
                        duration,
                        significance,
                    )

                return result

            except Exception as e:
                # Store failed interactions too (they're often significant)
                duration = time.time() - start_time
                await _store_agent_interaction(
                    agent_name,
                    func.__name__,
                    args,
                    kwargs,
                    {"error": str(e)},
                    duration,
                    0.9,
                )
                raise

        return wrapper

    return decorator


def _calculate_agent_significance(
    agent_name: str, method_name: str, result: Any, duration: float
) -> float:
    """Calculate significance of agent interaction"""
    significance = 0.6  # Base score for agent interactions

    # High significance for analysis methods
    if "analyze" in method_name.lower():
        significance += 0.2

    # High significance for strategy methods
    if "strategy" in method_name.lower() or "recommend" in method_name.lower():
        significance += 0.2

    # High significance for RESEARCHER agent (knowledge gathering)
    if agent_name == "RESEARCHER":
        significance += 0.1

    # High significance for slow operations (complex analysis)
    if duration > 5.0:
        significance += 0.2

    # High significance if result contains confidence scores > 0.8
    if isinstance(result, dict) and "confidence" in result:
        if result["confidence"] > 0.8:
            significance += 0.3

    # High significance for multi-agent collaboration
    if isinstance(result, dict) and "agents_involved" in result:
        if len(result["agents_involved"]) > 1:
            significance += 0.2

    return min(significance, 1.0)


async def _store_agent_interaction(
    agent_name: str,
    method_name: str,
    args: tuple,
    kwargs: dict,
    result: Any,
    duration: float,
    significance: float,
):
    """Store agent interaction in memory"""
    try:
        memory = await get_memory_manager()

        # Extract relevant context
        context = {
            "agent_name": agent_name,
            "method_name": method_name,
            "duration_seconds": duration,
            "result_type": type(result).__name__,
            "args_count": len(args),
            "kwargs_keys": list(kwargs.keys()),
        }

        # Add specific context based on method
        if "analyze" in method_name:
            context["interaction_type"] = "analysis"
            if "symbol" in kwargs:
                context["symbol"] = kwargs["symbol"]
        elif "recommend" in method_name:
            context["interaction_type"] = "recommendation"
        elif "strategy" in method_name:
            context["interaction_type"] = "strategy"

        # Add result summary
        if isinstance(result, dict):
            if "confidence" in result:
                context["confidence"] = result["confidence"]
            if "recommendation" in result:
                context["has_recommendation"] = True
            if "error" in result:
                context["has_error"] = True
                context["error_type"] = type(result["error"]).__name__

        # Detect collaboration patterns
        related_entities = {f"agent_{agent_name}": "performed_by"}
        if isinstance(result, dict) and "agents_involved" in result:
            for other_agent in result["agents_involved"]:
                if other_agent != agent_name:
                    related_entities[f"agent_{other_agent}"] = "collaborated_with"

        context["related_entities"] = related_entities

        # Create and store event
        event = MemoryEvent(
            event_type="agent_interaction",
            entity_id=f"{agent_name}_{method_name}_{datetime.now().strftime('%Y%m%d_%H%M')}",
            context=context,
            significance_score=significance,
            timestamp=datetime.now(),
        )

        await memory.store_significant_event(event)

    except Exception as e:
        logger.error(f"Failed to store agent interaction: {e}")


# Context Manager for Complex Operations
class MemoryContextManager:
    """
    Context manager for tracking complex operations across multiple function calls

    Usage:
    async with MemoryContextManager("portfolio_analysis", significance=0.9) as ctx:
        researcher_result = await researcher.analyze_stocks(symbols)
        ctx.add_step("research_completed", researcher_result)

        strategy_result = await mastermind.create_strategy(researcher_result)
        ctx.add_step("strategy_created", strategy_result)

        # Context automatically stores the complete workflow
    """

    def __init__(self, operation_name: str, significance: float = 0.8):
        self.operation_name = operation_name
        self.significance = significance
        self.start_time = None
        self.steps = []
        self.context = {}

    async def __aenter__(self):
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            duration = time.time() - self.start_time

            # Calculate final significance
            final_significance = self.significance
            if exc_type is not None:  # Error occurred
                final_significance = min(final_significance + 0.2, 1.0)

            # Store complete operation
            memory = await get_memory_manager()

            event = MemoryEvent(
                event_type="complex_operation",
                entity_id=f"{self.operation_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                context={
                    "operation_name": self.operation_name,
                    "total_duration": duration,
                    "steps_count": len(self.steps),
                    "steps": self.steps,
                    "success": exc_type is None,
                    "error_type": exc_type.__name__ if exc_type else None,
                    **self.context,
                },
                significance_score=final_significance,
                timestamp=datetime.now(),
            )

            await memory.store_significant_event(event)

        except Exception as e:
            logger.error(f"Failed to store complex operation: {e}")

    def add_step(self, step_name: str, result: Any):
        """Add a step to the complex operation"""
        self.steps.append(
            {
                "step_name": step_name,
                "timestamp": datetime.now().isoformat(),
                "result_type": type(result).__name__,
                "has_result": result is not None,
            }
        )

    def add_context(self, key: str, value: Any):
        """Add additional context to the operation"""
        self.context[key] = value


# Helper function to setup memory middleware in FastAPI app
def setup_memory_middleware(app, significance_threshold: float = 0.7):
    """Setup memory middleware for FastAPI application"""
    app.add_middleware(MemoryMiddleware, significance_threshold=significance_threshold)
    logger.info("Memory middleware setup complete")


# Export key components
__all__ = [
    "MemoryMiddleware",
    "sparc_memory_aware",
    "MemoryContextManager",
    "setup_memory_middleware",
]
