"""
Degraded Search Service for TradeKnowledge.

This module provides search functionality with graceful degradation
when components are unavailable or performing poorly.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any

from ..resilience.graceful_degradation import (
    get_degradation_manager,
    update_component_health,
)

logger = logging.getLogger(__name__)


class DegradedSearchResult:
    """Result object for degraded search operations"""

    def __init__(
        self,
        results: list[dict[str, Any]] = None,
        total_count: int = 0,
        degraded: bool = False,
        degradation_level: str = "none",
        message: str = "",
        available_features: list[str] = None,
        disabled_features: list[str] = None,
    ):
        self.results = results or []
        self.total_count = total_count
        self.degraded = degraded
        self.degradation_level = degradation_level
        self.message = message
        self.available_features = available_features or []
        self.disabled_features = disabled_features or []
        self.timestamp = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "results": self.results,
            "total_count": self.total_count,
            "degraded": self.degraded,
            "degradation_level": self.degradation_level,
            "message": self.message,
            "available_features": self.available_features,
            "disabled_features": self.disabled_features,
            "timestamp": self.timestamp.isoformat(),
        }


class DegradedSearchService:
    """
    Search service with comprehensive graceful degradation capabilities.
    Provides fallback functionality when components are unavailable.
    """

    def __init__(self):
        self.degradation_manager = get_degradation_manager()
        self.simple_cache: dict[str, dict[str, Any]] = {}
        self.cache_ttl_minutes = 30
        self.fallback_results_limit = 10

        # Component health tracking
        self.vector_db_healthy = True
        self.embedding_service_healthy = True
        self.ai_service_healthy = True
        self.database_healthy = True

    async def vector_search(
        self,
        query: str,
        limit: int = 20,
        filters: dict[str, Any] | None = None,
        include_embeddings: bool = False,
        include_analytics: bool = True,
    ) -> DegradedSearchResult:
        """
        Perform vector search with graceful degradation.

        Args:
            query: Search query text
            limit: Maximum number of results
            filters: Optional search filters
            include_embeddings: Whether to include embedding vectors
            include_analytics: Whether to include analytics data

        Returns:
            DegradedSearchResult with search results or fallback data
        """
        start_time = time.time()

        try:
            # Check if we can use cached results
            cache_key = self._generate_cache_key(query, limit, filters)
            cached_result = self._get_cached_result(cache_key)

            if cached_result:
                await update_component_health("search_cache", True, 5.0)
                return DegradedSearchResult(
                    results=cached_result["results"],
                    total_count=cached_result["total_count"],
                    degraded=True,
                    degradation_level="reduced",
                    message="Serving cached search results",
                    available_features=["basic_search", "cached_results"],
                    disabled_features=["real_time_search", "advanced_analytics"],
                )

            # Try full vector search
            results = await self._perform_full_vector_search(
                query, limit, filters, include_embeddings, include_analytics
            )

            # Cache successful results
            self._cache_result(
                cache_key, {"results": results, "total_count": len(results)}
            )

            duration_ms = (time.time() - start_time) * 1000
            await update_component_health("vector_search", True, duration_ms)

            return DegradedSearchResult(
                results=results,
                total_count=len(results),
                degraded=False,
                message="Full vector search completed",
                available_features=[
                    "vector_search",
                    "embeddings",
                    "analytics",
                    "filters",
                ],
            )

        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            duration_ms = (time.time() - start_time) * 1000
            await update_component_health("vector_search", False, duration_ms)

            # Try fallback methods
            return await self._fallback_search(query, limit, filters)

    async def _perform_full_vector_search(
        self,
        query: str,
        limit: int,
        filters: dict[str, Any] | None,
        include_embeddings: bool,
        include_analytics: bool,
    ) -> list[dict[str, Any]]:
        """Perform full vector search (placeholder implementation)"""

        # This would normally call the actual vector database
        # For now, simulate with delay and potential failure
        await asyncio.sleep(0.1)  # Simulate processing time

        # Simulate occasional failures to test degradation
        import random

        if random.random() < 0.1:  # 10% chance of failure
            raise Exception("Vector database temporarily unavailable")

        # Return mock results
        return [
            {
                "id": f"doc_{i}",
                "title": f"Document {i} matching '{query}'",
                "content": f"This is content for document {i} that matches the query '{query}'",
                "score": 0.9 - (i * 0.1),
                "metadata": {
                    "source": f"source_{i}",
                    "created_at": (datetime.now() - timedelta(days=i)).isoformat(),
                },
                "embedding": [0.1] * 384 if include_embeddings else None,
                "analytics": (
                    {"view_count": 100 - (i * 10), "relevance_score": 0.9 - (i * 0.1)}
                    if include_analytics
                    else None
                ),
            }
            for i in range(min(limit, 5))  # Limit mock results
        ]

    async def _fallback_search(
        self, query: str, limit: int, filters: dict[str, Any] | None
    ) -> DegradedSearchResult:
        """Fallback search when vector search fails"""

        try:
            # Try text-based search
            results = await self._text_based_search(query, limit)

            return DegradedSearchResult(
                results=results,
                total_count=len(results),
                degraded=True,
                degradation_level="minimal",
                message="Using text-based search (vector search unavailable)",
                available_features=["text_search", "basic_results"],
                disabled_features=[
                    "vector_search",
                    "semantic_search",
                    "embeddings",
                    "analytics",
                ],
            )

        except Exception as e:
            logger.warning(f"Text search also failed: {e}")

            # Final fallback - simple keyword search
            return await self._simple_keyword_search(query, limit)

    async def _text_based_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Perform text-based search as fallback"""

        # Simulate text search
        await asyncio.sleep(0.05)

        # Mock text search results
        return [
            {
                "id": f"text_doc_{i}",
                "title": f"Text result {i} for '{query}'",
                "content": f"Basic text match for query '{query}' in document {i}",
                "score": 0.7 - (i * 0.1),
                "metadata": {"source": "text_search", "search_type": "text_based"},
            }
            for i in range(min(limit, 3))
        ]

    async def _simple_keyword_search(
        self, query: str, limit: int
    ) -> DegradedSearchResult:
        """Simple keyword search as final fallback"""

        # Most basic search - just return suggested searches or cached popular content
        suggested_results = [
            {
                "id": "fallback_1",
                "title": "Search Help",
                "content": "Try using simpler keywords or check your spelling",
                "score": 0.5,
                "metadata": {"type": "help", "source": "fallback"},
            },
            {
                "id": "fallback_2",
                "title": "Popular Content",
                "content": "Browse our most popular content while search is restored",
                "score": 0.4,
                "metadata": {"type": "popular", "source": "fallback"},
            },
        ]

        return DegradedSearchResult(
            results=suggested_results,
            total_count=len(suggested_results),
            degraded=True,
            degradation_level="emergency",
            message="Search temporarily limited. Showing help and popular content.",
            available_features=["basic_help"],
            disabled_features=["search", "filters", "analytics", "recommendations"],
        )

    async def get_recommendations(
        self,
        user_context: dict[str, Any] | None = None,
        content_id: str | None = None,
        limit: int = 10,
    ) -> DegradedSearchResult:
        """
        Get content recommendations with graceful degradation.

        Args:
            user_context: User context for personalization
            content_id: Specific content ID for related recommendations
            limit: Maximum number of recommendations

        Returns:
            DegradedSearchResult with recommendations or fallback content
        """
        start_time = time.time()

        try:
            # Try AI-powered recommendations
            recommendations = await self._ai_recommendations(
                user_context, content_id, limit
            )

            duration_ms = (time.time() - start_time) * 1000
            await update_component_health("ai_recommendations", True, duration_ms)

            return DegradedSearchResult(
                results=recommendations,
                total_count=len(recommendations),
                degraded=False,
                message="AI-powered recommendations",
                available_features=[
                    "ai_recommendations",
                    "personalization",
                    "content_analysis",
                ],
            )

        except Exception as e:
            logger.warning(f"AI recommendations failed: {e}")
            duration_ms = (time.time() - start_time) * 1000
            await update_component_health("ai_recommendations", False, duration_ms)

            # Fallback to simple recommendations
            return await self._simple_recommendations(limit)

    async def _ai_recommendations(
        self,
        user_context: dict[str, Any] | None,
        content_id: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """AI-powered recommendations (placeholder)"""

        await asyncio.sleep(0.2)  # Simulate AI processing

        # Simulate occasional AI service failures
        import random

        if random.random() < 0.15:  # 15% chance of failure
            raise Exception("AI recommendation service unavailable")

        return [
            {
                "id": f"ai_rec_{i}",
                "title": f"AI Recommended Content {i}",
                "content": f"AI-analyzed content recommendation {i} based on user preferences",
                "score": 0.95 - (i * 0.05),
                "reason": f"Recommended based on your interest in {content_id or 'similar topics'}",
                "metadata": {
                    "type": "ai_recommendation",
                    "personalized": bool(user_context),
                    "confidence": 0.9 - (i * 0.1),
                },
            }
            for i in range(min(limit, 4))
        ]

    async def _simple_recommendations(self, limit: int) -> DegradedSearchResult:
        """Simple popularity-based recommendations"""

        # Basic recommendations based on popularity
        popular_content = [
            {
                "id": f"popular_{i}",
                "title": f"Popular Content {i}",
                "content": f"This is popular content item {i}",
                "score": 0.8 - (i * 0.1),
                "metadata": {"type": "popular", "view_count": 1000 - (i * 100)},
            }
            for i in range(min(limit, 5))
        ]

        return DegradedSearchResult(
            results=popular_content,
            total_count=len(popular_content),
            degraded=True,
            degradation_level="reduced",
            message="Showing popular content (personalized recommendations unavailable)",
            available_features=["popular_content"],
            disabled_features=[
                "ai_recommendations",
                "personalization",
                "content_analysis",
            ],
        )

    async def analyze_content(
        self, content: str, analysis_type: str = "summary"
    ) -> dict[str, Any]:
        """
        Analyze content with graceful degradation.

        Args:
            content: Content to analyze
            analysis_type: Type of analysis to perform

        Returns:
            Analysis results or simplified fallback
        """
        start_time = time.time()

        try:
            # Try full AI analysis
            analysis = await self._full_content_analysis(content, analysis_type)

            duration_ms = (time.time() - start_time) * 1000
            await update_component_health("content_analysis", True, duration_ms)

            return {
                "degraded": False,
                "analysis": analysis,
                "analysis_type": analysis_type,
                "message": "Full AI content analysis completed",
            }

        except Exception as e:
            logger.warning(f"Content analysis failed: {e}")
            duration_ms = (time.time() - start_time) * 1000
            await update_component_health("content_analysis", False, duration_ms)

            # Fallback to basic analysis
            return await self._basic_content_analysis(content, analysis_type)

    async def _full_content_analysis(
        self, content: str, analysis_type: str
    ) -> dict[str, Any]:
        """Full AI-powered content analysis"""

        await asyncio.sleep(0.3)  # Simulate AI processing

        # Simulate occasional failures
        import random

        if random.random() < 0.2:  # 20% chance of failure
            raise Exception("Content analysis service unavailable")

        return {
            "summary": f"AI-generated summary of content ({len(content)} characters)",
            "key_topics": ["topic1", "topic2", "topic3"],
            "sentiment": "positive",
            "complexity_score": 0.7,
            "readability": "medium",
            "word_count": len(content.split()),
            "confidence": 0.95,
        }

    async def _basic_content_analysis(
        self, content: str, analysis_type: str
    ) -> dict[str, Any]:
        """Basic content analysis fallback"""

        # Simple text statistics
        words = content.split()
        word_count = len(words)
        char_count = len(content)

        return {
            "degraded": True,
            "analysis": {
                "summary": f"Basic analysis: {word_count} words, {char_count} characters",
                "word_count": word_count,
                "character_count": char_count,
                "estimated_reading_time": max(1, word_count // 200),  # Assume 200 WPM
                "analysis_type": "basic_statistics",
            },
            "message": "Advanced analysis unavailable. Showing basic statistics.",
            "available_features": ["word_count", "character_count", "reading_time"],
            "disabled_features": ["ai_analysis", "sentiment", "topics", "complexity"],
        }

    def _generate_cache_key(
        self, query: str, limit: int, filters: dict[str, Any] | None
    ) -> str:
        """Generate cache key for search results"""
        filter_str = str(sorted(filters.items())) if filters else ""
        return f"search:{query}:{limit}:{filter_str}"

    def _get_cached_result(self, cache_key: str) -> dict[str, Any] | None:
        """Get cached search result if still valid"""
        if cache_key not in self.simple_cache:
            return None

        cached_data, timestamp = self.simple_cache[cache_key]
        age_minutes = (datetime.now() - timestamp).total_seconds() / 60

        if age_minutes <= self.cache_ttl_minutes:
            return cached_data
        else:
            # Remove expired cache entry
            del self.simple_cache[cache_key]
            return None

    def _cache_result(self, cache_key: str, result: dict[str, Any]):
        """Cache search result"""
        self.simple_cache[cache_key] = (result, datetime.now())

        # Simple cache cleanup - remove oldest entries if cache gets too large
        if len(self.simple_cache) > 100:
            oldest_key = min(
                self.simple_cache.keys(), key=lambda k: self.simple_cache[k][1]
            )
            del self.simple_cache[oldest_key]

    async def get_service_status(self) -> dict[str, Any]:
        """Get current service status and degradation information"""
        return {
            "service": "degraded_search",
            "status": self.degradation_manager.get_service_status("search_service"),
            "cache_entries": len(self.simple_cache),
            "capabilities": {
                "vector_search": "available" if self.vector_db_healthy else "degraded",
                "ai_recommendations": (
                    "available" if self.ai_service_healthy else "degraded"
                ),
                "content_analysis": (
                    "available" if self.ai_service_healthy else "degraded"
                ),
                "text_search": "available",  # Always available as fallback
                "basic_cache": "available",  # Always available
            },
        }


# Global degraded search service instance
_global_search_service: DegradedSearchService | None = None


def get_degraded_search_service() -> DegradedSearchService:
    """Get or create global degraded search service"""
    global _global_search_service
    if _global_search_service is None:
        _global_search_service = DegradedSearchService()
    return _global_search_service
