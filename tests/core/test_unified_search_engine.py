"""
Unified Search Engine Tests for TradeKnowledge.

This module tests the UnifiedSearchEngine class which coordinates
all search functionality across text, vector, and hybrid search engines.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.search.unified_search import UnifiedSearchEngine


class TestUnifiedSearchEngine:
    """Test the UnifiedSearchEngine coordinator"""
    
    @pytest.fixture
    def search_engine(self):
        """Create search engine instance for testing"""
        with patch('src.search.hybrid_search.HybridSearch') as mock_hybrid, \
             patch('src.search.text_search.TextSearchEngine') as mock_text, \
             patch('src.search.vector_search.VectorSearchEngine') as mock_vector:
            
            # Mock the underlying search engines
            mock_hybrid_instance = AsyncMock()
            mock_text_instance = AsyncMock()
            mock_vector_instance = AsyncMock()
            
            mock_hybrid.return_value = mock_hybrid_instance
            mock_text.return_value = mock_text_instance
            mock_vector.return_value = mock_vector_instance
            
            engine = UnifiedSearchEngine()
            engine.initialized = True  # Skip initialization for testing
            engine.hybrid_engine = mock_hybrid_instance
            engine.text_engine = mock_text_instance
            engine.vector_engine = mock_vector_instance
            
            return engine
    
    @pytest.mark.asyncio
    async def test_initialization_state(self):
        """Test search engine initialization state"""
        engine = UnifiedSearchEngine()
        
        # Initially should not be initialized
        assert not engine.initialized
        assert engine.hybrid_engine is None
        assert engine.text_engine is None
        assert engine.vector_engine is None
    
    @pytest.mark.asyncio
    async def test_search_basic_functionality(self, search_engine):
        """Test basic search functionality"""
        
        # Mock search results from hybrid engine
        mock_results = {
            'results': [
                {
                    'id': 'chunk_001',
                    'title': 'Algorithmic Trading Strategies',
                    'content': 'Moving average strategies are fundamental...',
                    'score': 0.92,
                    'book_title': 'Trading Handbook'
                },
                {
                    'id': 'chunk_002', 
                    'title': 'Risk Management',
                    'content': 'Position sizing is crucial for...',
                    'score': 0.85,
                    'book_title': 'Risk Control Guide'
                }
            ]
        }
        search_engine.hybrid_engine.search.return_value = mock_results
        
        # Test search
        result = await search_engine.search(
            query="trading strategies",
            max_results=10,
            filters={"book_id": "trading_book_1"}
        )
        
        # Verify result structure
        assert isinstance(result, dict)
        assert 'results' in result
        assert 'total_found' in result
        assert 'detected_intent' in result
        assert 'suggestions' in result
        assert 'filters_applied' in result
        
        # Verify result content
        assert result['total_found'] == 2
        assert len(result['results']) == 2
        assert result['results'][0]['id'] == 'chunk_001'
        assert result['filters_applied'] == {"book_id": "trading_book_1"}
        
        # Verify hybrid engine was called correctly
        search_engine.hybrid_engine.search.assert_called_once_with(
            query="trading strategies",
            max_results=10,
            filters={"book_id": "trading_book_1"}
        )
    
    @pytest.mark.asyncio
    async def test_search_with_different_parameters(self, search_engine):
        """Test search with various parameter combinations"""
        
        mock_results = {'results': []}
        search_engine.hybrid_engine.search.return_value = mock_results
        
        # Test with minimal parameters
        result1 = await search_engine.search(query="test")
        assert result1['total_found'] == 0
        
        # Test with all parameters
        result2 = await search_engine.search(
            query="machine learning",
            intent="research",
            filters={"author": "John Doe", "difficulty": "advanced"},
            max_results=25,
            min_score=0.7,
            user_id="user123"
        )
        
        assert result2['detected_intent'] == "research"
        assert result2['filters_applied'] == {"author": "John Doe", "difficulty": "advanced"}
    
    @pytest.mark.asyncio
    async def test_search_error_handling(self, search_engine):
        """Test search error handling"""
        
        # Mock an exception in the hybrid engine
        search_engine.hybrid_engine.search.side_effect = Exception("Search engine error")
        
        # Search should not raise but return empty results
        result = await search_engine.search(query="test query")
        
        assert result['results'] == []
        assert result['total_found'] == 0
        assert 'detected_intent' in result
        assert 'suggestions' in result
        assert 'filters_applied' in result
    
    @pytest.mark.asyncio
    async def test_search_without_initialization(self):
        """Test search fails when engine not initialized"""
        engine = UnifiedSearchEngine()
        
        with pytest.raises(RuntimeError, match="Search engine not initialized"):
            await engine.search(query="test")
    
    @pytest.mark.asyncio
    async def test_autocomplete_suggestions(self, search_engine):
        """Test autocomplete suggestion functionality"""
        
        # Test basic suggestions
        suggestions = await search_engine.get_suggestions(
            partial_query="trading",
            max_suggestions=3
        )
        
        assert isinstance(suggestions, list)
        assert len(suggestions) <= 3
        assert all("trading" in suggestion for suggestion in suggestions)
        
        # Test with different parameters
        suggestions2 = await search_engine.get_suggestions(
            partial_query="risk",
            max_suggestions=5,
            user_id="user123"
        )
        
        assert len(suggestions2) <= 5
        assert all("risk" in suggestion for suggestion in suggestions2)
    
    @pytest.mark.asyncio
    async def test_autocomplete_error_handling(self, search_engine):
        """Test autocomplete error handling"""
        
        # Test with empty query (current implementation returns suggestions with empty prefix)
        suggestions = await search_engine.get_suggestions(
            partial_query="",  # Empty query
            max_suggestions=5
        )
        
        # Current implementation returns suggestions even with empty query
        assert isinstance(suggestions, list)
        assert len(suggestions) <= 5
    
    @pytest.mark.asyncio
    async def test_similar_content_search(self, search_engine):
        """Test finding similar content"""
        
        # Test similar search (currently returns empty as per implementation)
        similar_results = await search_engine.find_similar(
            result_id="chunk_001",
            max_results=10,
            user_id="user123"
        )
        
        assert isinstance(similar_results, list)
        # Current implementation returns empty list
        assert similar_results == []
    
    @pytest.mark.asyncio
    async def test_similar_content_error_handling(self, search_engine):
        """Test similar content error handling"""
        
        # Test with invalid result ID
        similar_results = await search_engine.find_similar(
            result_id="invalid_id",
            max_results=5
        )
        
        assert similar_results == []
    
    @pytest.mark.asyncio
    async def test_feedback_submission(self, search_engine):
        """Test search result feedback submission"""
        
        # Test feedback submission
        await search_engine.submit_feedback(
            user_id="user123",
            query="trading strategies",
            result_id="chunk_001",
            rating=5,
            feedback="Very helpful content"
        )
        
        # Should not raise any exceptions
        # Current implementation just logs feedback
        
        # Test with minimal parameters
        await search_engine.submit_feedback(
            user_id="user456",
            query="risk management",
            result_id="chunk_002",
            rating=3
        )
    
    @pytest.mark.asyncio
    async def test_feedback_error_handling(self, search_engine):
        """Test feedback submission error handling"""
        
        # Test with invalid parameters (should not raise)
        await search_engine.submit_feedback(
            user_id="",
            query="",
            result_id="",
            rating=0
        )
    
    @pytest.mark.asyncio
    async def test_trending_queries(self, search_engine):
        """Test trending queries functionality"""
        
        # Test trending queries
        trending = await search_engine.get_trending_queries(
            period="24h",
            limit=10
        )
        
        assert isinstance(trending, list)
        assert len(trending) <= 10
        
        # Test with different parameters
        trending2 = await search_engine.get_trending_queries(
            period="7d",
            limit=5
        )
        
        assert len(trending2) <= 5
        
        # Verify structure of trending items
        if trending:
            for item in trending:
                assert 'query' in item
                assert 'count' in item
                assert isinstance(item['query'], str)
                assert isinstance(item['count'], int)


class TestSearchEngineIntegration:
    """Test search engine integration scenarios"""
    
    @pytest.fixture
    def search_engine(self):
        """Create search engine for integration tests"""
        with patch('src.search.hybrid_search.HybridSearch') as mock_hybrid, \
             patch('src.search.text_search.TextSearchEngine') as mock_text, \
             patch('src.search.vector_search.VectorSearchEngine') as mock_vector:
            
            mock_hybrid_instance = AsyncMock()
            mock_text_instance = AsyncMock()
            mock_vector_instance = AsyncMock()
            
            mock_hybrid.return_value = mock_hybrid_instance
            mock_text.return_value = mock_text_instance  
            mock_vector.return_value = mock_vector_instance
            
            engine = UnifiedSearchEngine()
            engine.initialized = True
            engine.hybrid_engine = mock_hybrid_instance
            engine.text_engine = mock_text_instance
            engine.vector_engine = mock_vector_instance
            
            return engine
    
    @pytest.mark.asyncio
    async def test_search_workflow_complete(self, search_engine):
        """Test complete search workflow"""
        
        # Mock complex search results
        search_engine.hybrid_engine.search.return_value = {
            'results': [
                {
                    'id': 'chunk_001',
                    'title': 'Advanced Trading Strategies',
                    'content': 'Momentum and mean reversion strategies combine...',
                    'score': 0.95,
                    'book_id': 'advanced_trading',
                    'book_title': 'Advanced Algorithmic Trading',
                    'author': 'Expert Trader',
                    'page_number': 127,
                    'metadata': {
                        'difficulty': 'advanced',
                        'topics': ['momentum', 'mean_reversion'],
                        'code_examples': True
                    }
                },
                {
                    'id': 'chunk_002',
                    'title': 'Risk Management Principles',
                    'content': 'Proper position sizing ensures capital preservation...',
                    'score': 0.87,
                    'book_id': 'risk_management',
                    'book_title': 'Trading Risk Control',
                    'author': 'Risk Expert',
                    'page_number': 45,
                    'metadata': {
                        'difficulty': 'intermediate',
                        'topics': ['position_sizing', 'risk_control'],
                        'code_examples': False
                    }
                }
            ]
        }
        
        # Perform search
        result = await search_engine.search(
            query="advanced trading strategies with risk management",
            intent="research",
            filters={
                "difficulty": "advanced",
                "has_code_examples": True
            },
            max_results=20,
            min_score=0.8,
            user_id="advanced_user"
        )
        
        # Verify comprehensive result
        assert result['total_found'] == 2
        assert len(result['results']) == 2
        assert result['detected_intent'] == "research"
        
        # Verify first result details
        first_result = result['results'][0]
        assert first_result['score'] == 0.95
        assert first_result['book_title'] == 'Advanced Algorithmic Trading'
        assert first_result['metadata']['difficulty'] == 'advanced'
        assert first_result['metadata']['code_examples'] is True
        
        # Verify filters were applied
        assert result['filters_applied']['difficulty'] == "advanced"
        assert result['filters_applied']['has_code_examples'] is True
    
    @pytest.mark.asyncio
    async def test_multiple_search_operations(self, search_engine):
        """Test multiple search operations in sequence"""
        
        # Mock different results for different queries
        def mock_search_side_effect(query, max_results, filters):
            if "trading" in query:
                return {'results': [{'id': '1', 'title': 'Trading Result'}]}
            elif "risk" in query:
                return {'results': [{'id': '2', 'title': 'Risk Result'}]}
            else:
                return {'results': []}
        
        search_engine.hybrid_engine.search.side_effect = mock_search_side_effect
        
        # Perform multiple searches
        result1 = await search_engine.search(query="trading strategies")
        result2 = await search_engine.search(query="risk management")
        result3 = await search_engine.search(query="unknown topic")
        
        # Verify each search worked independently
        assert result1['total_found'] == 1
        assert result1['results'][0]['title'] == 'Trading Result'
        
        assert result2['total_found'] == 1  
        assert result2['results'][0]['title'] == 'Risk Result'
        
        assert result3['total_found'] == 0
        assert result3['results'] == []
    
    @pytest.mark.asyncio
    async def test_concurrent_search_operations(self, search_engine):
        """Test concurrent search operations"""
        
        # Mock search results
        search_engine.hybrid_engine.search.return_value = {
            'results': [{'id': 'concurrent_result', 'title': 'Concurrent Test'}]
        }
        
        # Perform concurrent searches
        tasks = [
            search_engine.search(query=f"query_{i}", max_results=10)
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all searches completed
        assert len(results) == 5
        for result in results:
            assert result['total_found'] == 1
            assert result['results'][0]['title'] == 'Concurrent Test'
        
        # Verify hybrid engine was called for each search
        assert search_engine.hybrid_engine.search.call_count == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])