"""
Search Engine Core Logic Tests for TradeKnowledge.

This module tests the core search functionality including:
- Query processing and validation
- Search result ranking and scoring
- Intent detection algorithms
- Filter application
- Pagination and result limiting
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
from src.core.models import SearchRequest, SearchIntent, User
from src.core.input_validator import ValidationError


class TestSearchQueryProcessing:
    """Test search query processing and validation"""
    
    @pytest.fixture
    def search_engine(self):
        """Create search engine instance for testing"""
        with patch('src.search.hybrid_search.HybridSearch') as mock_hybrid, \
             patch('src.search.text_search.TextSearchEngine') as mock_text, \
             patch('src.search.vector_search.VectorSearchEngine') as mock_vector:
            
            # Mock the underlying search engines
            mock_hybrid.return_value = AsyncMock()
            mock_text.return_value = AsyncMock()  
            mock_vector.return_value = AsyncMock()
            
            engine = UnifiedSearchEngine()
            engine.initialized = True  # Skip initialization for testing
            engine.hybrid_engine = mock_hybrid.return_value
            engine.text_engine = mock_text.return_value
            engine.vector_engine = mock_vector.return_value
            
            return engine
    
    @pytest.mark.asyncio
    async def test_search_basic_functionality(self, search_engine):
        """Test basic search functionality"""
        
        # Mock search results
        mock_results = {
            'results': [
                {'id': '1', 'title': 'Test Result', 'score': 0.9},
                {'id': '2', 'title': 'Another Result', 'score': 0.8}
            ]
        }
        search_engine.hybrid_engine.search.return_value = mock_results
        
        # Test valid queries
        valid_queries = [
            "algorithmic trading strategies",
            "machine learning in finance", 
            "risk management techniques",
            "technical analysis indicators"
        ]
        
        for query in valid_queries:
            # Should not raise any exceptions
            result = await search_engine.search(query=query, max_results=10)
            
            assert isinstance(result, dict)
            assert 'results' in result
            assert 'total_found' in result
            assert result['total_found'] >= 0
    
    @pytest.mark.asyncio
    async def test_malicious_query_rejection(self, search_engine):
        """Test that malicious queries are rejected"""
        
        malicious_queries = [
            "' OR 1=1--",
            "<script>alert('xss')</script>",
            "UNION SELECT * FROM users",
            "../../../etc/passwd",
            "DROP TABLE chunks;",
            "'; DELETE FROM books; --"
        ]
        
        for query in malicious_queries:
            with pytest.raises((ValidationError, ValueError)):
                await search_engine._sanitize_query(query)
    
    @pytest.mark.asyncio
    async def test_query_length_limits(self, search_engine):
        """Test query length validation"""
        
        # Test empty query
        with pytest.raises((ValidationError, ValueError)):
            await search_engine._sanitize_query("")
        
        # Test extremely long query
        long_query = "a" * 1000
        with pytest.raises((ValidationError, ValueError)):
            await search_engine._sanitize_query(long_query)
        
        # Test normal length query
        normal_query = "test query with reasonable length"
        sanitized = await search_engine._sanitize_query(normal_query)
        assert sanitized == normal_query
    
    @pytest.mark.asyncio
    async def test_query_normalization(self, search_engine):
        """Test query normalization and cleaning"""
        
        test_cases = [
            ("  Trading   Strategies  ", "Trading Strategies"),
            ("MACHINE LEARNING", "machine learning"),
            ("Risk\tManagement\n", "Risk Management"),
            ("algorithmic-trading", "algorithmic-trading"),
            ("Python & Finance", "Python & Finance")
        ]
        
        for input_query, expected in test_cases:
            sanitized = await search_engine._sanitize_query(input_query)
            assert sanitized == expected


class TestSearchIntentDetection:
    """Test search intent detection algorithms"""
    
    @pytest.fixture
    def search_engine(self):
        """Create search engine with intent detection"""
        with patch('src.search.unified_search.get_config') as mock_config:
            mock_config.return_value.search.intent_detection.enabled = True
            mock_config.return_value.search.intent_detection.confidence_threshold = 0.7
            
            engine = UnifiedSearchEngine()
            return engine
    
    @pytest.mark.asyncio
    async def test_research_intent_detection(self, search_engine):
        """Test detection of research intent queries"""
        
        research_queries = [
            "comprehensive analysis of momentum trading strategies",
            "detailed study of market microstructure effects",
            "research on algorithmic trading performance",
            "investigate correlation between volatility and returns"
        ]
        
        for query in research_queries:
            intent = await search_engine._detect_intent(query)
            assert intent == SearchIntent.RESEARCH
    
    @pytest.mark.asyncio
    async def test_quick_lookup_intent_detection(self, search_engine):
        """Test detection of quick lookup intent"""
        
        lookup_queries = [
            "RSI formula",
            "what is MACD",
            "VaR definition",
            "Bollinger Bands calculation"
        ]
        
        for query in lookup_queries:
            intent = await search_engine._detect_intent(query)
            assert intent == SearchIntent.QUICK_LOOKUP
    
    @pytest.mark.asyncio
    async def test_learning_intent_detection(self, search_engine):
        """Test detection of learning intent"""
        
        learning_queries = [
            "how to implement moving averages",
            "learn about options pricing",
            "tutorial on backtesting strategies",
            "beginner guide to portfolio optimization"
        ]
        
        for query in learning_queries:
            intent = await search_engine._detect_intent(query)
            assert intent == SearchIntent.LEARNING
    
    @pytest.mark.asyncio
    async def test_comparison_intent_detection(self, search_engine):
        """Test detection of comparison intent"""
        
        comparison_queries = [
            "SMA vs EMA effectiveness",
            "compare momentum and mean reversion strategies",
            "LSTM vs ARIMA for time series prediction",
            "difference between value at risk and expected shortfall"
        ]
        
        for query in comparison_queries:
            intent = await search_engine._detect_intent(query)
            assert intent == SearchIntent.COMPARISON


class TestSearchResultRanking:
    """Test search result ranking and scoring algorithms"""
    
    @pytest.fixture
    def search_engine(self):
        """Create search engine for ranking tests"""
        with patch('src.search.unified_search.get_config') as mock_config:
            # Configure ranking weights
            mock_ranking_config = MagicMock()
            mock_ranking_config.weights.semantic = 0.6
            mock_ranking_config.weights.exact = 0.3
            mock_ranking_config.weights.recency = 0.1
            mock_ranking_config.boost_factors.title_match = 1.5
            mock_ranking_config.boost_factors.author_popularity = 1.2
            
            mock_config.return_value.search.ranking = mock_ranking_config
            
            engine = UnifiedSearchEngine()
            return engine
    
    def test_semantic_score_calculation(self, search_engine):
        """Test semantic similarity score calculation"""
        
        # Mock embedding similarities
        test_cases = [
            (0.95, "exact semantic match should score high"),
            (0.8, "high semantic similarity"),
            (0.6, "moderate semantic similarity"),
            (0.3, "low semantic similarity"),
            (0.1, "minimal semantic similarity")
        ]
        
        for similarity, description in test_cases:
            score = search_engine._calculate_semantic_score(similarity)
            
            assert 0.0 <= score <= 1.0, f"Score out of range for {description}"
            
            if similarity > 0.9:
                assert score > 0.8, f"High similarity should yield high score: {description}"
            elif similarity < 0.2:
                assert score < 0.3, f"Low similarity should yield low score: {description}"
    
    def test_exact_match_scoring(self, search_engine):
        """Test exact text match scoring"""
        
        query = "moving average crossover strategy"
        
        test_cases = [
            ("A comprehensive guide to moving average crossover strategy implementation", 1.0),
            ("Moving average crossover strategy for beginners", 0.9),
            ("Strategy using moving average crossover signals", 0.7),
            ("Moving averages and crossover techniques", 0.5),
            ("Technical analysis with various indicators", 0.1)
        ]
        
        for text, expected_min_score in test_cases:
            score = search_engine._calculate_exact_score(query, text)
            assert score >= expected_min_score - 0.2, f"Exact score too low for: {text}"
    
    def test_title_boost_application(self, search_engine):
        """Test title match boost factor application"""
        
        base_score = 0.6
        boost_factor = 1.5
        
        # Test with title match
        boosted_score = search_engine._apply_title_boost(base_score, has_title_match=True)
        expected_boosted = min(1.0, base_score * boost_factor)
        
        assert boosted_score == expected_boosted
        assert boosted_score > base_score
        
        # Test without title match
        unboosted_score = search_engine._apply_title_boost(base_score, has_title_match=False)
        assert unboosted_score == base_score
    
    def test_result_ranking_order(self, search_engine):
        """Test that results are properly ranked by combined score"""
        
        # Mock search results with different scores
        mock_results = [
            {"id": "1", "semantic_score": 0.9, "exact_score": 0.8, "title_match": True},
            {"id": "2", "semantic_score": 0.7, "exact_score": 0.9, "title_match": False},
            {"id": "3", "semantic_score": 0.8, "exact_score": 0.6, "title_match": True},
            {"id": "4", "semantic_score": 0.6, "exact_score": 0.7, "title_match": False}
        ]
        
        # Calculate combined scores
        for result in mock_results:
            result["combined_score"] = search_engine._calculate_combined_score(
                semantic_score=result["semantic_score"],
                exact_score=result["exact_score"],
                has_title_match=result["title_match"]
            )
        
        # Rank results
        ranked_results = sorted(mock_results, key=lambda x: x["combined_score"], reverse=True)
        
        # Verify ranking is correct
        scores = [r["combined_score"] for r in ranked_results]
        assert scores == sorted(scores, reverse=True), "Results not properly ranked"
        
        # High-scoring results should be first
        assert ranked_results[0]["combined_score"] >= 0.8


class TestSearchFilters:
    """Test search filter application"""
    
    @pytest.fixture
    def search_engine(self):
        """Create search engine for filter tests"""
        engine = UnifiedSearchEngine()
        return engine
    
    def test_book_filter_application(self, search_engine):
        """Test filtering results by book"""
        
        mock_results = [
            {"book_id": "book1", "title": "Result 1"},
            {"book_id": "book2", "title": "Result 2"}, 
            {"book_id": "book1", "title": "Result 3"},
            {"book_id": "book3", "title": "Result 4"}
        ]
        
        filters = {"book_id": "book1"}
        filtered_results = search_engine._apply_filters(mock_results, filters)
        
        assert len(filtered_results) == 2
        assert all(r["book_id"] == "book1" for r in filtered_results)
    
    def test_author_filter_application(self, search_engine):
        """Test filtering results by author"""
        
        mock_results = [
            {"author": "John Doe", "title": "Result 1"},
            {"author": "Jane Smith", "title": "Result 2"},
            {"author": "John Doe", "title": "Result 3"}
        ]
        
        filters = {"author": "John Doe"}
        filtered_results = search_engine._apply_filters(mock_results, filters)
        
        assert len(filtered_results) == 2
        assert all(r["author"] == "John Doe" for r in filtered_results)
    
    def test_date_range_filter(self, search_engine):
        """Test filtering results by date range"""
        
        mock_results = [
            {"created_at": "2023-01-15", "title": "Result 1"},
            {"created_at": "2023-06-20", "title": "Result 2"},
            {"created_at": "2023-12-10", "title": "Result 3"}
        ]
        
        filters = {
            "date_from": "2023-06-01",
            "date_to": "2023-12-31"
        }
        
        filtered_results = search_engine._apply_filters(mock_results, filters)
        
        assert len(filtered_results) == 2
        for result in filtered_results:
            assert result["created_at"] >= "2023-06-01"
            assert result["created_at"] <= "2023-12-31"
    
    def test_multiple_filters_combination(self, search_engine):
        """Test combining multiple filters"""
        
        mock_results = [
            {"book_id": "book1", "author": "John Doe", "category": "trading"},
            {"book_id": "book1", "author": "Jane Smith", "category": "finance"},
            {"book_id": "book2", "author": "John Doe", "category": "trading"},
            {"book_id": "book1", "author": "John Doe", "category": "analysis"}
        ]
        
        filters = {
            "book_id": "book1",
            "author": "John Doe"
        }
        
        filtered_results = search_engine._apply_filters(mock_results, filters)
        
        assert len(filtered_results) == 2
        for result in filtered_results:
            assert result["book_id"] == "book1"
            assert result["author"] == "John Doe"
    
    def test_invalid_filter_handling(self, search_engine):
        """Test handling of invalid or malicious filters"""
        
        mock_results = [{"title": "Result 1"}]
        
        # Test SQL injection attempt in filter
        malicious_filters = {
            "book_id": "'; DROP TABLE books; --",
            "author": "<script>alert('xss')</script>"
        }
        
        # Should not raise exception, should sanitize or ignore
        filtered_results = search_engine._apply_filters(mock_results, malicious_filters)
        assert isinstance(filtered_results, list)


class TestSearchPagination:
    """Test search result pagination"""
    
    @pytest.fixture
    def search_engine(self):
        """Create search engine for pagination tests"""
        engine = UnifiedSearchEngine()
        return engine
    
    def test_pagination_limits(self, search_engine):
        """Test pagination parameter validation"""
        
        # Test valid pagination
        offset, limit = search_engine._validate_pagination(0, 10)
        assert offset == 0
        assert limit == 10
        
        # Test maximum limit enforcement
        offset, limit = search_engine._validate_pagination(0, 200)
        assert limit <= 100  # Assuming max limit is 100
        
        # Test negative offset handling
        with pytest.raises(ValueError):
            search_engine._validate_pagination(-1, 10)
        
        # Test zero or negative limit handling
        with pytest.raises(ValueError):
            search_engine._validate_pagination(0, 0)
        
        with pytest.raises(ValueError):
            search_engine._validate_pagination(0, -5)
    
    def test_result_slicing(self, search_engine):
        """Test correct slicing of results for pagination"""
        
        # Create mock results
        mock_results = [{"id": f"result_{i}"} for i in range(25)]
        
        # Test first page
        page1 = search_engine._paginate_results(mock_results, offset=0, limit=10)
        assert len(page1) == 10
        assert page1[0]["id"] == "result_0"
        assert page1[9]["id"] == "result_9"
        
        # Test second page
        page2 = search_engine._paginate_results(mock_results, offset=10, limit=10)
        assert len(page2) == 10
        assert page2[0]["id"] == "result_10"
        assert page2[9]["id"] == "result_19"
        
        # Test partial last page
        page3 = search_engine._paginate_results(mock_results, offset=20, limit=10)
        assert len(page3) == 5
        assert page3[0]["id"] == "result_20"
        assert page3[4]["id"] == "result_24"
        
        # Test offset beyond results
        page4 = search_engine._paginate_results(mock_results, offset=30, limit=10)
        assert len(page4) == 0


class TestSearchPerformance:
    """Test search performance and optimization"""
    
    @pytest.fixture
    def search_engine(self):
        """Create search engine for performance tests"""
        engine = UnifiedSearchEngine()
        return engine
    
    @pytest.mark.asyncio
    async def test_search_timeout_handling(self, search_engine):
        """Test that search operations respect timeout limits"""
        
        with patch.object(search_engine, '_perform_vector_search') as mock_vector_search:
            # Mock a slow operation
            mock_vector_search.side_effect = asyncio.TimeoutError("Search timeout")
            
            with pytest.raises(asyncio.TimeoutError):
                await search_engine.search(
                    query="test query",
                    timeout=1.0  # 1 second timeout
                )
    
    @pytest.mark.asyncio
    async def test_cache_integration(self, search_engine):
        """Test search result caching"""
        
        with patch.object(search_engine, '_get_cached_results') as mock_cache_get, \
             patch.object(search_engine, '_cache_results') as mock_cache_set:
            
            # Test cache miss
            mock_cache_get.return_value = None
            
            # Mock search execution
            mock_results = [{"id": "1", "title": "Test Result"}]
            with patch.object(search_engine, '_execute_search', return_value=mock_results):
                results = await search_engine.search(query="test query")
                
                # Verify caching was attempted
                mock_cache_get.assert_called_once()
                mock_cache_set.assert_called_once()
                assert results == mock_results
            
            # Test cache hit
            mock_cache_get.return_value = mock_results
            
            cached_results = await search_engine.search(query="test query")
            assert cached_results == mock_results
    
    def test_result_limit_enforcement(self, search_engine):
        """Test that result limits are properly enforced"""
        
        # Create large result set
        large_results = [{"id": f"result_{i}"} for i in range(1000)]
        
        # Test default limit
        limited_results = search_engine._limit_results(large_results)
        assert len(limited_results) <= 50  # Assuming default limit is 50
        
        # Test custom limit
        limited_results = search_engine._limit_results(large_results, limit=25)
        assert len(limited_results) == 25
        
        # Test limit larger than results
        small_results = [{"id": f"result_{i}"} for i in range(5)]
        limited_results = search_engine._limit_results(small_results, limit=10)
        assert len(limited_results) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])