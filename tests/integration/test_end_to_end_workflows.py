"""
Integration Tests for TradeKnowledge End-to-End Workflows.

This module tests complete workflows that span multiple components:
- Book ingestion to search pipeline
- API authentication to search results
- Search engine integration with vector stores
- Complete user journeys
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.api.main import app
from src.core.models import Book, Chunk, User, SearchRequest, SearchIntent
from src.search.unified_search import UnifiedSearchEngine
from src.ingestion.enhanced_book_processor import EnhancedBookProcessor
from fastapi.testclient import TestClient


class TestBookIngestionToSearchWorkflow:
    """Test the complete flow from book ingestion to search"""
    
    @pytest.fixture
    async def mock_components(self):
        """Set up mocked components for integration testing"""
        
        # Mock book processor
        mock_processor = AsyncMock()
        mock_processor.process_book.return_value = {
            'chunks': [
                Chunk(
                    book_id="test_book_001",
                    chunk_index=0,
                    text="Algorithmic trading involves using computer programs to execute trades",
                    chapter="Introduction",
                    page_start=1
                ),
                Chunk(
                    book_id="test_book_001", 
                    chunk_index=1,
                    text="Risk management is crucial for successful trading strategies",
                    chapter="Risk Management",
                    page_start=25
                )
            ],
            'embeddings': [
                [0.1, 0.2, 0.3, 0.4] * 96,  # 384-dim embedding
                [0.5, 0.6, 0.7, 0.8] * 96   # 384-dim embedding
            ],
            'book': Book(
                id="test_book_001",
                title="Algorithmic Trading Fundamentals",
                author="Test Author",
                file_path="/tmp/test_book.pdf",
                file_type="pdf",
                file_hash="test_hash_123"
            )
        }
        
        # Mock search engine
        mock_search_engine = AsyncMock()
        mock_search_engine.search.return_value = {
            'results': [
                {
                    'id': 'test_book_001_chunk_00001',
                    'title': 'Algorithmic Trading Fundamentals',
                    'content': 'Algorithmic trading involves using computer programs to execute trades',
                    'score': 0.95,
                    'book_id': 'test_book_001',
                    'book_title': 'Algorithmic Trading Fundamentals',
                    'page_number': 1,
                    'metadata': {'chapter': 'Introduction'}
                }
            ],
            'total_found': 1,
            'detected_intent': 'research',
            'suggestions': [],
            'filters_applied': {}
        }
        
        return {
            'processor': mock_processor,
            'search_engine': mock_search_engine
        }
    
    @pytest.mark.asyncio
    async def test_complete_ingestion_to_search_workflow(self, mock_components):
        """Test the complete workflow from book upload to search results"""
        
        processor = mock_components['processor']
        search_engine = mock_components['search_engine']
        
        # Step 1: Ingest a book
        book_path = "/tmp/test_algorithmic_trading.pdf"
        processing_result = await processor.process_book(book_path)
        
        # Verify ingestion results
        assert 'chunks' in processing_result
        assert 'embeddings' in processing_result
        assert 'book' in processing_result
        assert len(processing_result['chunks']) == 2
        assert len(processing_result['embeddings']) == 2
        
        book = processing_result['book']
        assert book.title == "Algorithmic Trading Fundamentals"
        assert book.author == "Test Author"
        
        # Step 2: Verify chunks are properly created
        chunks = processing_result['chunks']
        assert chunks[0].text.startswith("Algorithmic trading involves")
        assert chunks[1].text.startswith("Risk management is crucial")
        
        # Step 3: Perform search on the ingested content
        search_query = "algorithmic trading strategies"
        search_result = await search_engine.search(
            query=search_query,
            max_results=10,
            user_id="test_user"
        )
        
        # Verify search results
        assert search_result['total_found'] > 0
        assert len(search_result['results']) > 0
        
        first_result = search_result['results'][0]
        assert first_result['book_id'] == book.id
        assert first_result['score'] > 0.8  # High relevance
        assert "algorithmic trading" in first_result['content'].lower()
        
        # Step 4: Verify end-to-end data consistency
        assert first_result['book_title'] == book.title
        assert first_result['page_number'] == chunks[0].page_start
    
    @pytest.mark.asyncio
    async def test_multiple_books_integration(self, mock_components):
        """Test integration with multiple books"""
        
        processor = mock_components['processor']
        search_engine = mock_components['search_engine']
        
        # Mock processing multiple books
        books_data = [
            {
                'id': 'trading_book_001',
                'title': 'Advanced Trading Strategies',
                'content': 'Momentum trading strategies for volatile markets'
            },
            {
                'id': 'risk_book_001', 
                'title': 'Risk Management in Trading',
                'content': 'Portfolio risk assessment and control methods'
            }
        ]
        
        # Update mock to return results from multiple books
        search_engine.search.return_value = {
            'results': [
                {
                    'id': f'{book["id"]}_chunk_00001',
                    'book_id': book['id'],
                    'book_title': book['title'],
                    'content': book['content'],
                    'score': 0.9 - i * 0.1  # Decreasing scores
                }
                for i, book in enumerate(books_data)
            ],
            'total_found': len(books_data),
            'detected_intent': 'research',
            'suggestions': [],
            'filters_applied': {}
        }
        
        # Search across all books
        search_result = await search_engine.search(
            query="trading risk management",
            max_results=10
        )
        
        # Verify multi-book results
        assert search_result['total_found'] == 2
        assert len(search_result['results']) == 2
        
        # Verify results are from different books
        book_ids = {result['book_id'] for result in search_result['results']}
        assert len(book_ids) == 2
        assert 'trading_book_001' in book_ids
        assert 'risk_book_001' in book_ids
    
    @pytest.mark.asyncio
    async def test_search_filtering_integration(self, mock_components):
        """Test search with filters integration"""
        
        search_engine = mock_components['search_engine']
        
        # Mock filtered search results
        search_engine.search.return_value = {
            'results': [
                {
                    'id': 'advanced_book_001_chunk_00001',
                    'book_id': 'advanced_book_001',
                    'book_title': 'Advanced Trading Algorithms',
                    'content': 'Machine learning applications in trading',
                    'score': 0.92,
                    'metadata': {'difficulty': 'advanced', 'topic': 'machine_learning'}
                }
            ],
            'total_found': 1,
            'detected_intent': 'research',
            'suggestions': [],
            'filters_applied': {'difficulty': 'advanced', 'topic': 'machine_learning'}
        }
        
        # Search with filters
        search_result = await search_engine.search(
            query="machine learning trading",
            filters={'difficulty': 'advanced', 'topic': 'machine_learning'},
            max_results=10
        )
        
        # Verify filtering worked
        assert search_result['total_found'] == 1
        assert search_result['filters_applied']['difficulty'] == 'advanced'
        assert search_result['filters_applied']['topic'] == 'machine_learning'
        
        result = search_result['results'][0]
        assert result['metadata']['difficulty'] == 'advanced'
        assert result['metadata']['topic'] == 'machine_learning'


class TestAPIIntegrationWorkflows:
    """Test API integration workflows"""
    
    @pytest.fixture
    def api_client(self):
        """Create API test client with mocked dependencies"""
        
        with patch('src.api.main.app_state') as mock_app_state:
            # Mock app state dependencies
            mock_dependencies = {
                'search_engine': AsyncMock(),
                'auth_manager': AsyncMock(),
                'metrics': AsyncMock(),
                'book_processor': AsyncMock()
            }
            
            mock_app_state.get.side_effect = lambda key: mock_dependencies.get(key)
            
            # Configure search engine mock
            mock_dependencies['search_engine'].search.return_value = {
                'results': [
                    {
                        'id': 'test_result_001',
                        'title': 'Test Trading Strategy',
                        'content': 'This is a test trading strategy description',
                        'score': 0.95,
                        'book_title': 'Trading Handbook'
                    }
                ],
                'total_found': 1,
                'detected_intent': 'research',
                'suggestions': [],
                'filters_applied': {}
            }
            
            # Configure auth manager mock
            test_user = User(
                id="test_user_001",
                username="testuser",
                email="test@example.com",
                role="user",
                created_at="2023-01-01T00:00:00Z"
            )
            mock_dependencies['auth_manager'].verify_token.return_value = test_user
            
            client = TestClient(app)
            return client, mock_dependencies
    
    def test_authenticated_search_workflow(self, api_client):
        """Test complete authenticated search workflow"""
        
        client, mock_deps = api_client
        
        # Make authenticated search request
        headers = {"Authorization": "Bearer valid_test_token"}
        response = client.post(
            "/api/v1/search/query",
            headers=headers,
            json={
                "query": "algorithmic trading strategies",
                "max_results": 10,
                "intent": "research"
            }
        )
        
        # Verify successful response
        assert response.status_code == 200
        
        response_data = response.json()
        assert response_data['query'] == "algorithmic trading strategies"
        assert response_data['total_found'] == 1
        assert len(response_data['results']) == 1
        
        # Verify result structure
        result = response_data['results'][0]
        assert result['id'] == 'test_result_001'
        assert result['title'] == 'Test Trading Strategy'
        assert result['score'] == 0.95
        
        # Verify search engine was called correctly
        mock_deps['search_engine'].search.assert_called_once()
        call_args = mock_deps['search_engine'].search.call_args
        assert call_args[1]['query'] == "algorithmic trading strategies"
        assert call_args[1]['max_results'] == 10
        
        # Verify authentication was checked
        mock_deps['auth_manager'].verify_token.assert_called_once_with("valid_test_token")
    
    def test_unauthenticated_request_rejection(self, api_client):
        """Test that unauthenticated requests are properly rejected"""
        
        client, mock_deps = api_client
        
        # Make request without authentication
        response = client.post(
            "/api/v1/search/query",
            json={
                "query": "test query",
                "max_results": 10
            }
        )
        
        # Should be rejected
        assert response.status_code in [401, 403, 422]
        
        # Search engine should not be called
        mock_deps['search_engine'].search.assert_not_called()
    
    def test_invalid_token_rejection(self, api_client):
        """Test rejection of invalid authentication tokens"""
        
        client, mock_deps = api_client
        
        # Configure auth manager to reject invalid token
        mock_deps['auth_manager'].verify_token.side_effect = Exception("Invalid token")
        
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.post(
            "/api/v1/search/query",
            headers=headers,
            json={
                "query": "test query",
                "max_results": 10
            }
        )
        
        # Should be rejected
        assert response.status_code == 401
        
        # Search engine should not be called
        mock_deps['search_engine'].search.assert_not_called()
    
    def test_search_with_various_parameters(self, api_client):
        """Test search with different parameter combinations"""
        
        client, mock_deps = api_client
        
        headers = {"Authorization": "Bearer valid_test_token"}
        
        # Test with filters
        response = client.post(
            "/api/v1/search/query",
            headers=headers,
            json={
                "query": "risk management",
                "max_results": 20,
                "intent": "learning",
                "filters": {"difficulty": "intermediate", "author": "John Doe"}
            }
        )
        
        assert response.status_code == 200
        
        # Verify parameters were passed correctly
        call_args = mock_deps['search_engine'].search.call_args
        assert call_args[1]['query'] == "risk management"
        assert call_args[1]['max_results'] == 20
        assert call_args[1]['filters']['difficulty'] == "intermediate"
        assert call_args[1]['filters']['author'] == "John Doe"


class TestSearchEngineIntegrationFlows:
    """Test search engine integration with various components"""
    
    @pytest.fixture
    def integrated_search_engine(self):
        """Create search engine with mocked component integrations"""
        
        with patch('src.search.hybrid_search.HybridSearch') as mock_hybrid, \
             patch('src.search.text_search.TextSearchEngine') as mock_text, \
             patch('src.search.vector_search.VectorSearchEngine') as mock_vector:
            
            # Set up mock engines
            mock_hybrid_instance = AsyncMock()
            mock_text_instance = AsyncMock()
            mock_vector_instance = AsyncMock()
            
            mock_hybrid.return_value = mock_hybrid_instance
            mock_text.return_value = mock_text_instance
            mock_vector.return_value = mock_vector_instance
            
            # Create unified search engine
            engine = UnifiedSearchEngine()
            engine.hybrid_engine = mock_hybrid_instance
            engine.text_engine = mock_text_instance
            engine.vector_engine = mock_vector_instance
            engine.initialized = True
            
            return engine, {
                'hybrid': mock_hybrid_instance,
                'text': mock_text_instance,
                'vector': mock_vector_instance
            }
    
    @pytest.mark.asyncio
    async def test_hybrid_search_coordination(self, integrated_search_engine):
        """Test coordination between different search engines"""
        
        engine, mock_engines = integrated_search_engine
        
        # Mock hybrid search results
        mock_engines['hybrid'].search.return_value = {
            'results': [
                {
                    'id': 'hybrid_result_001',
                    'content': 'Hybrid search result for algorithmic trading',
                    'score': 0.89,
                    'source': 'hybrid'
                }
            ]
        }
        
        # Perform search
        result = await engine.search(
            query="algorithmic trading optimization",
            max_results=15,
            filters={'complexity': 'high'}
        )
        
        # Verify hybrid engine was called
        mock_engines['hybrid'].search.assert_called_once_with(
            query="algorithmic trading optimization",
            max_results=15,
            filters={'complexity': 'high'}
        )
        
        # Verify result structure
        assert result['total_found'] == 1
        assert len(result['results']) == 1
        assert result['results'][0]['id'] == 'hybrid_result_001'
        assert result['results'][0]['score'] == 0.89
    
    @pytest.mark.asyncio
    async def test_search_error_handling_and_fallback(self, integrated_search_engine):
        """Test error handling and fallback mechanisms"""
        
        engine, mock_engines = integrated_search_engine
        
        # Mock hybrid search to fail
        mock_engines['hybrid'].search.side_effect = Exception("Hybrid search failure")
        
        # Search should handle error gracefully
        result = await engine.search(query="test query")
        
        # Should return empty results instead of failing
        assert result['results'] == []
        assert result['total_found'] == 0
        assert 'detected_intent' in result
    
    @pytest.mark.asyncio
    async def test_search_performance_under_load(self, integrated_search_engine):
        """Test search performance under concurrent load"""
        
        engine, mock_engines = integrated_search_engine
        
        # Mock consistent search results
        mock_engines['hybrid'].search.return_value = {
            'results': [{'id': 'perf_test_result', 'score': 0.8}]
        }
        
        # Perform concurrent searches
        search_tasks = [
            engine.search(query=f"test query {i}", max_results=10)
            for i in range(10)
        ]
        
        results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # All searches should complete successfully
        assert len(results) == 10
        for result in results:
            assert not isinstance(result, Exception)
            assert result['total_found'] == 1
            assert len(result['results']) == 1
        
        # Hybrid engine should be called for each search
        assert mock_engines['hybrid'].search.call_count == 10


class TestUserJourneyIntegration:
    """Test complete user journeys through the system"""
    
    @pytest.fixture
    def full_system_mock(self):
        """Mock complete system for user journey testing"""
        
        # Mock user data
        test_user = User(
            id="journey_user_001",
            username="journey_user",
            email="journey@example.com",
            role="user",
            created_at="2023-01-01T00:00:00Z"
        )
        
        # Mock search results for user journey
        search_results = {
            'results': [
                {
                    'id': 'journey_result_001',
                    'title': 'Introduction to Algorithmic Trading',
                    'content': 'Comprehensive guide to algorithmic trading strategies and implementation',
                    'score': 0.95,
                    'book_id': 'algo_trading_guide',
                    'book_title': 'Algorithmic Trading Handbook',
                    'page_number': 15,
                    'metadata': {'difficulty': 'beginner', 'estimated_read_time': 10}
                },
                {
                    'id': 'journey_result_002',
                    'title': 'Advanced Risk Management',
                    'content': 'Advanced techniques for managing trading risks and portfolio optimization',
                    'score': 0.87,
                    'book_id': 'risk_management_pro',
                    'book_title': 'Professional Risk Management',
                    'page_number': 42,
                    'metadata': {'difficulty': 'advanced', 'estimated_read_time': 25}
                }
            ],
            'total_found': 2,
            'detected_intent': 'learning',
            'suggestions': [
                'algorithmic trading basics',
                'trading strategy development',
                'risk management fundamentals'
            ],
            'filters_applied': {}
        }
        
        return {
            'user': test_user,
            'search_results': search_results
        }
    
    @pytest.mark.asyncio
    async def test_complete_learning_journey(self, full_system_mock):
        """Test a complete learning journey from search to content consumption"""
        
        user = full_system_mock['user']
        search_results = full_system_mock['search_results']
        
        # Step 1: User performs initial search
        query = "algorithmic trading for beginners"
        
        # Simulate search engine returning appropriate results
        with patch('src.search.unified_search.UnifiedSearchEngine') as mock_engine_class:
            mock_engine = AsyncMock()
            mock_engine.search.return_value = search_results
            mock_engine_class.return_value = mock_engine
            
            engine = UnifiedSearchEngine()
            search_result = await engine.search(
                query=query,
                user_id=user.id,
                max_results=10
            )
        
        # Verify initial search results
        assert search_result['total_found'] == 2
        assert search_result['detected_intent'] == 'learning'
        assert len(search_result['suggestions']) == 3
        
        # Step 2: User explores first result (beginner content)
        first_result = search_result['results'][0]
        assert first_result['metadata']['difficulty'] == 'beginner'
        assert first_result['title'] == 'Introduction to Algorithmic Trading'
        
        # Step 3: System suggests related content
        suggestions = search_result['suggestions']
        assert 'algorithmic trading basics' in suggestions
        assert 'trading strategy development' in suggestions
        
        # Step 4: User follows up with more specific search
        followup_query = "trading strategy development"
        
        # Mock follow-up search with more specific results
        followup_results = {
            'results': [
                {
                    'id': 'strategy_dev_001',
                    'title': 'Strategy Development Framework',
                    'content': 'Step-by-step guide to developing robust trading strategies',
                    'score': 0.92,
                    'metadata': {'difficulty': 'intermediate', 'related_to': first_result['id']}
                }
            ],
            'total_found': 1,
            'detected_intent': 'learning',
            'suggestions': [],
            'filters_applied': {}
        }
        
        mock_engine.search.return_value = followup_results
        followup_result = await engine.search(
            query=followup_query,
            user_id=user.id,
            max_results=10
        )
        
        # Verify progressive learning path
        assert followup_result['total_found'] == 1
        strategy_result = followup_result['results'][0]
        assert strategy_result['metadata']['difficulty'] == 'intermediate'
        assert strategy_result['metadata']['related_to'] == first_result['id']
    
    @pytest.mark.asyncio
    async def test_research_workflow_journey(self, full_system_mock):
        """Test a research-focused user journey"""
        
        user = full_system_mock['user']
        
        # Research journey: User looking for comprehensive analysis
        research_query = "comprehensive analysis of momentum trading strategies"
        
        research_results = {
            'results': [
                {
                    'id': 'research_001',
                    'title': 'Momentum Trading: A Comprehensive Analysis',
                    'content': 'Detailed academic analysis of momentum trading effectiveness across different market conditions',
                    'score': 0.96,
                    'book_id': 'academic_trading_research',
                    'book_title': 'Academic Perspectives on Trading',
                    'metadata': {
                        'content_type': 'research_paper',
                        'citations': 45,
                        'peer_reviewed': True
                    }
                },
                {
                    'id': 'research_002',
                    'title': 'Empirical Evidence for Momentum Strategies',
                    'content': 'Statistical analysis and backtesting results for various momentum approaches',
                    'score': 0.91,
                    'book_id': 'quantitative_methods',
                    'book_title': 'Quantitative Trading Methods',
                    'metadata': {
                        'content_type': 'empirical_study',
                        'data_period': '1990-2020',
                        'sample_size': 10000
                    }
                }
            ],
            'total_found': 2,
            'detected_intent': 'research',
            'suggestions': [
                'momentum factor models',
                'cross-sectional momentum',
                'time-series momentum'
            ],
            'filters_applied': {}
        }
        
        with patch('src.search.unified_search.UnifiedSearchEngine') as mock_engine_class:
            mock_engine = AsyncMock()
            mock_engine.search.return_value = research_results
            mock_engine_class.return_value = mock_engine
            
            engine = UnifiedSearchEngine()
            research_result = await engine.search(
                query=research_query,
                intent=SearchIntent.RESEARCH,
                user_id=user.id,
                max_results=20
            )
        
        # Verify research-appropriate results
        assert research_result['detected_intent'] == 'research'
        assert research_result['total_found'] == 2
        
        # Verify research-quality content
        first_paper = research_result['results'][0]
        assert first_paper['metadata']['content_type'] == 'research_paper'
        assert first_paper['metadata']['peer_reviewed'] is True
        assert first_paper['metadata']['citations'] == 45
        
        second_paper = research_result['results'][1]
        assert second_paper['metadata']['content_type'] == 'empirical_study'
        assert second_paper['metadata']['sample_size'] == 10000
        
        # Verify research-appropriate suggestions
        suggestions = research_result['suggestions']
        assert 'momentum factor models' in suggestions
        assert 'cross-sectional momentum' in suggestions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])