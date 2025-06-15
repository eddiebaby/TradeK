"""
Unified Search Engine

Combines all search components into a single interface for the API.
"""

import asyncio
from typing import Dict, List, Any, Optional
import structlog

from .hybrid_search import HybridSearch
from .text_search import TextSearchEngine  
from .vector_search import VectorSearchEngine

logger = structlog.get_logger(__name__)

class UnifiedSearchEngine:
    """Unified interface for all search capabilities"""
    
    def __init__(self):
        self.hybrid_engine = None
        self.text_engine = None
        self.vector_engine = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize all search engines"""
        try:
            self.hybrid_engine = HybridSearch()
            await self.hybrid_engine.initialize()
            
            self.text_engine = TextSearchEngine()
            await self.text_engine.initialize()
            
            self.vector_engine = VectorSearchEngine()
            await self.vector_engine.initialize()
            
            self.initialized = True
            logger.info("Unified search engine initialized")
            
        except Exception as e:
            logger.error("Failed to initialize search engine", error=str(e))
            raise
    
    async def search(
        self,
        query: str,
        intent: Optional[str] = None,
        filters: Dict[str, Any] = None,
        max_results: int = 20,
        min_score: float = 0.0,
        user_id: str = None
    ) -> Dict[str, Any]:
        """Perform unified search across all engines"""
        
        if not self.initialized:
            raise RuntimeError("Search engine not initialized")
        
        try:
            # Use hybrid search as primary engine
            results = await self.hybrid_engine.search(
                query=query,
                max_results=max_results,
                filters=filters or {}
            )
            
            return {
                'results': results.get('results', []),
                'total_found': len(results.get('results', [])),
                'detected_intent': intent or 'semantic',
                'suggestions': [],
                'filters_applied': filters or {}
            }
            
        except Exception as e:
            logger.error("Search failed", query=query, error=str(e))
            # Return empty results on error
            return {
                'results': [],
                'total_found': 0,
                'detected_intent': intent or 'semantic',
                'suggestions': [],
                'filters_applied': filters or {}
            }
    
    async def get_suggestions(
        self,
        partial_query: str,
        max_suggestions: int = 5,
        user_id: str = None
    ) -> List[str]:
        """Get autocomplete suggestions"""
        try:
            # TODO: Implement actual suggestion logic
            suggestions = [
                f"{partial_query} strategy",
                f"{partial_query} analysis",
                f"{partial_query} implementation"
            ]
            return suggestions[:max_suggestions]
            
        except Exception as e:
            logger.error("Suggestions failed", query=partial_query, error=str(e))
            return []
    
    async def find_similar(
        self,
        result_id: str,
        max_results: int = 10,
        user_id: str = None
    ) -> List[Dict[str, Any]]:
        """Find similar content to a result"""
        try:
            # TODO: Implement actual similarity search
            return []
            
        except Exception as e:
            logger.error("Similar search failed", result_id=result_id, error=str(e))
            return []
    
    async def submit_feedback(
        self,
        user_id: str,
        query: str,
        result_id: str,
        rating: int,
        feedback: str = None
    ):
        """Submit search result feedback"""
        try:
            # TODO: Implement feedback storage and learning
            logger.info("Feedback received", 
                       user_id=user_id, result_id=result_id, rating=rating)
            
        except Exception as e:
            logger.error("Feedback submission failed", error=str(e))
    
    async def get_trending_queries(
        self,
        period: str = "24h",
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get trending queries"""
        try:
            # TODO: Implement actual trending analysis
            return [
                {"query": "momentum trading", "count": 45},
                {"query": "risk management", "count": 38},
                {"query": "technical analysis", "count": 31}
            ]
            
        except Exception as e:
            logger.error("Trending queries failed", error=str(e))
            return []
    
    async def cleanup(self):
        """Cleanup search engines"""
        try:
            if self.hybrid_engine:
                await self.hybrid_engine.cleanup()
            if self.text_engine:
                await self.text_engine.cleanup()
            if self.vector_engine:
                await self.vector_engine.cleanup()
                
            logger.info("Search engines cleaned up")
            
        except Exception as e:
            logger.error("Cleanup failed", error=str(e))