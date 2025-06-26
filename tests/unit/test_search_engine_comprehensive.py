"""
Comprehensive Search Engine Testing for TradeKnowledge
Tests core search functionality, embedding generation, vector operations, and edge cases
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from typing import List, Dict, Any
import tempfile
from pathlib import Path

from src.core.models import SearchResult, ChunkData
from src.core.unified_search_engine import UnifiedSearchEngine
from src.ingestion.local_embeddings import LocalEmbeddingGenerator
from src.core.qdrant_storage import QdrantStorage


class TestUnifiedSearchEngine:
    """Comprehensive tests for the unified search engine"""
    
    @pytest.fixture
    async def mock_qdrant_storage(self):
        """Mock Qdrant storage for testing"""
        storage = Mock(spec=QdrantStorage)
        storage.search = AsyncMock()
        storage.get_stats = AsyncMock()
        storage.count_points = AsyncMock()
        storage.initialize = AsyncMock()
        storage.close = AsyncMock()
        yield storage
    
    @pytest.fixture
    async def mock_embedding_generator(self):
        """Mock embedding generator for testing"""
        generator = Mock(spec=LocalEmbeddingGenerator)
        generator.generate_embeddings = AsyncMock()
        generator.__aenter__ = AsyncMock(return_value=generator)
        generator.__aexit__ = AsyncMock(return_value=None)
        yield generator
    
    @pytest.fixture
    async def search_engine(self, mock_qdrant_storage, mock_embedding_generator):
        """Create search engine with mocked dependencies"""
        with patch('src.core.unified_search_engine.QdrantStorage') as mock_storage_class:
            with patch('src.ingestion.local_embeddings.LocalEmbeddingGenerator') as mock_gen_class:
                mock_storage_class.return_value = mock_qdrant_storage
                mock_gen_class.return_value = mock_embedding_generator
                
                engine = UnifiedSearchEngine()
                await engine.initialize()
                yield engine
                await engine.close()
    
    @pytest.fixture
    def sample_search_results(self):
        """Sample search results for testing"""
        return [
            {
                "id": "chunk_1",
                "score": 0.95,
                "payload": {
                    "content": "Advanced trading strategies for forex markets",
                    "book_title": "Forex Master",
                    "page_number": 42,
                    "chapter": "Advanced Techniques"
                }
            },
            {
                "id": "chunk_2", 
                "score": 0.87,
                "payload": {
                    "content": "Risk management in volatile markets",
                    "book_title": "Risk Control",
                    "page_number": 15,
                    "chapter": "Market Volatility"
                }
            }
        ]


class TestSearchQueries:
    """Test various search query scenarios"""
    
    async def test_basic_search_query(self, search_engine, mock_qdrant_storage, mock_embedding_generator, sample_search_results):
        """Test basic search functionality"""
        # Setup mocks
        mock_embedding_generator.generate_embeddings.return_value = [[0.1] * 384]
        mock_qdrant_storage.search.return_value = sample_search_results
        
        # Execute search
        results = await search_engine.search("trading strategies", max_results=10)
        
        # Verify results
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].score == 0.95
        assert "trading strategies" in results[0].content.lower()
        
        # Verify mocks called correctly
        mock_embedding_generator.generate_embeddings.assert_called_once()
        mock_qdrant_storage.search.assert_called_once()
    
    async def test_empty_search_query(self, search_engine):
        """Test search with empty query"""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await search_engine.search("")
    
    async def test_whitespace_only_query(self, search_engine):
        """Test search with whitespace-only query"""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await search_engine.search("   \n\t   ")
    
    async def test_very_long_query(self, search_engine, mock_embedding_generator):
        """Test search with very long query"""
        long_query = "trading " * 1000
        mock_embedding_generator.generate_embeddings.return_value = [[0.1] * 384]
        
        # Should handle long queries gracefully
        results = await search_engine.search(long_query, max_results=5)
        assert isinstance(results, list)
    
    async def test_unicode_search_query(self, search_engine, mock_embedding_generator, mock_qdrant_storage):
        """Test search with Unicode characters"""
        unicode_query = "tradingλογικήстратегия 🚀📈"
        mock_embedding_generator.generate_embeddings.return_value = [[0.1] * 384]
        mock_qdrant_storage.search.return_value = []
        
        results = await search_engine.search(unicode_query)
        assert isinstance(results, list)
        
        # Verify embedding generator was called with Unicode query
        mock_embedding_generator.generate_embeddings.assert_called_once()
        call_args = mock_embedding_generator.generate_embeddings.call_args[0]
        assert unicode_query in call_args[0]
    
    async def test_search_with_special_characters(self, search_engine, mock_embedding_generator):
        """Test search with special characters"""
        special_query = "trading@#$%^&*()_+-=[]{}|;':\",./<>?"
        mock_embedding_generator.generate_embeddings.return_value = [[0.1] * 384]
        
        results = await search_engine.search(special_query)
        assert isinstance(results, list)


class TestSearchParameters:
    """Test search parameter validation and handling"""
    
    async def test_max_results_parameter(self, search_engine, mock_embedding_generator, mock_qdrant_storage):
        """Test max_results parameter handling"""
        mock_embedding_generator.generate_embeddings.return_value = [[0.1] * 384]
        mock_qdrant_storage.search.return_value = []
        
        # Test various max_results values
        test_values = [1, 5, 10, 50, 100]
        
        for max_results in test_values:
            await search_engine.search("test", max_results=max_results)
            # Verify Qdrant was called with correct limit
            call_args = mock_qdrant_storage.search.call_args
            assert call_args[1]["limit"] == max_results
    
    async def test_invalid_max_results(self, search_engine):
        """Test invalid max_results values"""
        with pytest.raises(ValueError):
            await search_engine.search("test", max_results=0)
        
        with pytest.raises(ValueError):
            await search_engine.search("test", max_results=-1)
        
        with pytest.raises(ValueError):
            await search_engine.search("test", max_results=1001)  # Assuming max limit
    
    async def test_min_score_parameter(self, search_engine, mock_embedding_generator, mock_qdrant_storage, sample_search_results):
        """Test min_score parameter filtering"""
        mock_embedding_generator.generate_embeddings.return_value = [[0.1] * 384]
        mock_qdrant_storage.search.return_value = sample_search_results
        
        # Test with min_score that should filter some results
        results = await search_engine.search("test", min_score=0.9)
        
        # Should only return results with score >= 0.9
        assert all(r.score >= 0.9 for r in results)
        assert len(results) == 1  # Only first result has score 0.95
    
    async def test_invalid_min_score(self, search_engine):
        """Test invalid min_score values"""
        with pytest.raises(ValueError):
            await search_engine.search("test", min_score=-0.1)
        
        with pytest.raises(ValueError):
            await search_engine.search("test", min_score=1.1)
    
    async def test_search_with_filters(self, search_engine, mock_embedding_generator, mock_qdrant_storage):
        """Test search with metadata filters"""
        mock_embedding_generator.generate_embeddings.return_value = [[0.1] * 384]
        mock_qdrant_storage.search.return_value = []
        
        filters = {
            "book_title": "Forex Master",
            "chapter": "Advanced Techniques"
        }
        
        await search_engine.search("test", filters=filters)
        
        # Verify filters were passed to Qdrant
        call_args = mock_qdrant_storage.search.call_args
        assert "filter" in call_args[1]


class TestEmbeddingGeneration:
    """Test embedding generation and handling"""
    
    async def test_embedding_generation_success(self, search_engine, mock_embedding_generator):
        """Test successful embedding generation"""
        expected_embedding = [0.1, 0.2, 0.3] * 128  # 384 dimensions
        mock_embedding_generator.generate_embeddings.return_value = [expected_embedding]
        
        # Call the internal method
        embeddings = await search_engine._generate_query_embedding("test query")
        
        assert embeddings == expected_embedding
        mock_embedding_generator.generate_embeddings.assert_called_once_with(["test query"])
    
    async def test_embedding_generation_failure(self, search_engine, mock_embedding_generator):
        """Test embedding generation failure handling"""
        mock_embedding_generator.generate_embeddings.side_effect = Exception("Embedding service unavailable")
        
        with pytest.raises(Exception, match="Embedding service unavailable"):
            await search_engine._generate_query_embedding("test query")
    
    async def test_embedding_dimension_validation(self, search_engine, mock_embedding_generator):
        """Test embedding dimension validation"""
        # Wrong dimension embedding
        wrong_embedding = [0.1] * 100  # Should be 384
        mock_embedding_generator.generate_embeddings.return_value = [wrong_embedding]
        
        with pytest.raises(ValueError, match="dimension"):
            await search_engine._generate_query_embedding("test query")
    
    async def test_multiple_embeddings_returned(self, search_engine, mock_embedding_generator):
        """Test handling when multiple embeddings are returned"""
        multiple_embeddings = [[0.1] * 384, [0.2] * 384]
        mock_embedding_generator.generate_embeddings.return_value = multiple_embeddings
        
        # Should use the first embedding
        embedding = await search_engine._generate_query_embedding("test query")
        assert embedding == multiple_embeddings[0]
    
    async def test_empty_embedding_returned(self, search_engine, mock_embedding_generator):
        """Test handling when no embeddings are returned"""
        mock_embedding_generator.generate_embeddings.return_value = []
        
        with pytest.raises(ValueError, match="No embeddings generated"):
            await search_engine._generate_query_embedding("test query")


class TestVectorSearchIntegration:
    """Test vector search integration with Qdrant"""
    
    async def test_vector_search_call(self, search_engine, mock_embedding_generator, mock_qdrant_storage):
        """Test vector search call to Qdrant"""
        query_embedding = [0.1] * 384
        mock_embedding_generator.generate_embeddings.return_value = [query_embedding]
        mock_qdrant_storage.search.return_value = []
        
        await search_engine.search("test query", max_results=10, min_score=0.7)
        
        # Verify Qdrant search was called with correct parameters
        mock_qdrant_storage.search.assert_called_once()
        call_args = mock_qdrant_storage.search.call_args
        
        assert call_args[0][0] == query_embedding  # Query vector
        assert call_args[1]["limit"] == 10
        assert call_args[1]["score_threshold"] == 0.7
    
    async def test_qdrant_connection_error(self, search_engine, mock_embedding_generator, mock_qdrant_storage):
        """Test Qdrant connection error handling"""
        mock_embedding_generator.generate_embeddings.return_value = [[0.1] * 384]
        mock_qdrant_storage.search.side_effect = ConnectionError("Cannot connect to Qdrant")
        
        with pytest.raises(ConnectionError, match="Cannot connect to Qdrant"):
            await search_engine.search("test query")
    
    async def test_qdrant_timeout_error(self, search_engine, mock_embedding_generator, mock_qdrant_storage):
        """Test Qdrant timeout error handling"""
        mock_embedding_generator.generate_embeddings.return_value = [[0.1] * 384]
        mock_qdrant_storage.search.side_effect = asyncio.TimeoutError("Qdrant search timeout")
        
        with pytest.raises(asyncio.TimeoutError, match="Qdrant search timeout"):
            await search_engine.search("test query")


class TestResultProcessing:
    """Test search result processing and formatting"""
    
    async def test_result_conversion(self, search_engine, mock_embedding_generator, mock_qdrant_storage):
        """Test conversion of Qdrant results to SearchResult objects"""
        qdrant_results = [
            {
                "id": "chunk_123",
                "score": 0.92,
                "payload": {
                    "content": "Test content about trading",
                    "book_title": "Trading Guide",
                    "page_number": 10,
                    "chapter": "Basics",
                    "author": "John Doe"
                }
            }
        ]
        
        mock_embedding_generator.generate_embeddings.return_value = [[0.1] * 384]
        mock_qdrant_storage.search.return_value = qdrant_results
        
        results = await search_engine.search("test")
        
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, SearchResult)
        assert result.chunk_id == "chunk_123"
        assert result.score == 0.92
        assert result.content == "Test content about trading"
        assert result.metadata["book_title"] == "Trading Guide"
        assert result.metadata["page_number"] == 10
    
    async def test_missing_content_handling(self, search_engine, mock_embedding_generator, mock_qdrant_storage):
        """Test handling of results with missing content"""
        qdrant_results = [
            {
                "id": "chunk_123",
                "score": 0.92,
                "payload": {
                    "book_title": "Trading Guide"
                    # Missing content field
                }
            }
        ]
        
        mock_embedding_generator.generate_embeddings.return_value = [[0.1] * 384]
        mock_qdrant_storage.search.return_value = qdrant_results
        
        results = await search_engine.search("test")
        
        # Should handle missing content gracefully
        assert len(results) == 1
        assert results[0].content == ""  # Default empty content
    
    async def test_result_sorting(self, search_engine, mock_embedding_generator, mock_qdrant_storage):
        """Test that results are properly sorted by score"""
        unsorted_results = [
            {"id": "chunk_1", "score": 0.7, "payload": {"content": "Content 1"}},
            {"id": "chunk_2", "score": 0.9, "payload": {"content": "Content 2"}},
            {"id": "chunk_3", "score": 0.8, "payload": {"content": "Content 3"}},
        ]
        
        mock_embedding_generator.generate_embeddings.return_value = [[0.1] * 384]
        mock_qdrant_storage.search.return_value = unsorted_results
        
        results = await search_engine.search("test")
        
        # Results should be sorted by score (descending)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert scores == [0.9, 0.8, 0.7]


class TestStatisticsAndMetrics:
    """Test search engine statistics and metrics"""
    
    async def test_get_stats(self, search_engine, mock_qdrant_storage):
        """Test getting search engine statistics"""
        mock_stats = {
            "total_vectors": 50000,
            "indexed_vectors": 50000,
            "disk_usage": 1024 * 1024 * 500,  # 500MB
            "ram_usage": 1024 * 1024 * 100,   # 100MB
        }
        mock_qdrant_storage.get_stats.return_value = mock_stats
        
        stats = await search_engine.get_stats()
        
        assert "total_documents" in stats
        assert "total_chunks" in stats
        assert "index_size" in stats
        assert stats["total_chunks"] == 50000
        
        mock_qdrant_storage.get_stats.assert_called_once()
    
    async def test_count_documents(self, search_engine, mock_qdrant_storage):
        """Test document counting functionality"""
        mock_qdrant_storage.count_points.return_value = 12345
        
        count = await search_engine.count_documents()
        
        assert count == 12345
        mock_qdrant_storage.count_points.assert_called_once()


class TestConcurrencyAndPerformance:
    """Test concurrent operations and performance characteristics"""
    
    @pytest.mark.performance
    async def test_concurrent_searches(self, search_engine, mock_embedding_generator, mock_qdrant_storage):
        """Test concurrent search operations"""
        mock_embedding_generator.generate_embeddings.return_value = [[0.1] * 384]
        mock_qdrant_storage.search.return_value = []
        
        # Create multiple concurrent search tasks
        search_tasks = [
            search_engine.search(f"query {i}")
            for i in range(10)
        ]
        
        # Execute all searches concurrently
        results = await asyncio.gather(*search_tasks)
        
        # Verify all searches completed successfully
        assert len(results) == 10
        assert all(isinstance(r, list) for r in results)
        
        # Verify embedding generator was called for each search
        assert mock_embedding_generator.generate_embeddings.call_count == 10
    
    @pytest.mark.performance
    async def test_search_performance_with_large_query(self, search_engine, mock_embedding_generator, mock_qdrant_storage):
        """Test search performance with large queries"""
        import time
        
        large_query = "trading strategy " * 500  # Large query
        mock_embedding_generator.generate_embeddings.return_value = [[0.1] * 384]
        mock_qdrant_storage.search.return_value = []
        
        start_time = time.time()
        await search_engine.search(large_query)
        end_time = time.time()
        
        # Should complete within reasonable time
        execution_time = end_time - start_time
        assert execution_time < 5.0  # Should complete in under 5 seconds


class TestErrorRecovery:
    """Test error recovery and resilience"""
    
    async def test_partial_failure_recovery(self, search_engine, mock_embedding_generator, mock_qdrant_storage):
        """Test recovery from partial failures"""
        # First call fails, second succeeds
        mock_embedding_generator.generate_embeddings.side_effect = [
            Exception("Temporary failure"),
            [[0.1] * 384]
        ]
        mock_qdrant_storage.search.return_value = []
        
        # First search should fail
        with pytest.raises(Exception, match="Temporary failure"):
            await search_engine.search("test query")
        
        # Second search should succeed
        results = await search_engine.search("test query")
        assert isinstance(results, list)
    
    async def test_resource_cleanup_on_error(self, search_engine, mock_embedding_generator):
        """Test proper resource cleanup when errors occur"""
        mock_embedding_generator.generate_embeddings.side_effect = Exception("Service error")
        
        try:
            await search_engine.search("test query")
        except Exception:
            pass
        
        # Verify embedding generator context was properly cleaned up
        mock_embedding_generator.__aexit__.assert_called()


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    async def test_search_with_all_zero_embedding(self, search_engine, mock_embedding_generator, mock_qdrant_storage):
        """Test search with all-zero embedding vector"""
        zero_embedding = [0.0] * 384
        mock_embedding_generator.generate_embeddings.return_value = [zero_embedding]
        mock_qdrant_storage.search.return_value = []
        
        # Should handle zero embedding without error
        results = await search_engine.search("test")
        assert isinstance(results, list)
    
    async def test_search_with_nan_embedding(self, search_engine, mock_embedding_generator):
        """Test search with NaN values in embedding"""
        nan_embedding = [float('nan')] * 384
        mock_embedding_generator.generate_embeddings.return_value = [nan_embedding]
        
        # Should detect and reject NaN embeddings
        with pytest.raises(ValueError, match="NaN"):
            await search_engine.search("test")
    
    async def test_search_with_infinite_embedding(self, search_engine, mock_embedding_generator):
        """Test search with infinite values in embedding"""
        inf_embedding = [float('inf')] * 384
        mock_embedding_generator.generate_embeddings.return_value = [inf_embedding]
        
        # Should detect and reject infinite embeddings
        with pytest.raises(ValueError, match="infinite"):
            await search_engine.search("test")
    
    async def test_empty_result_set(self, search_engine, mock_embedding_generator, mock_qdrant_storage):
        """Test handling of empty result sets"""
        mock_embedding_generator.generate_embeddings.return_value = [[0.1] * 384]
        mock_qdrant_storage.search.return_value = []
        
        results = await search_engine.search("nonexistent query")
        
        assert isinstance(results, list)
        assert len(results) == 0
    
    async def test_malformed_qdrant_results(self, search_engine, mock_embedding_generator, mock_qdrant_storage):
        """Test handling of malformed Qdrant results"""
        malformed_results = [
            {"id": "chunk_1"},  # Missing score and payload
            {"score": 0.8},     # Missing id and payload
            {                   # Missing everything
                "invalid_field": "value"
            }
        ]
        
        mock_embedding_generator.generate_embeddings.return_value = [[0.1] * 384]
        mock_qdrant_storage.search.return_value = malformed_results
        
        # Should handle malformed results gracefully
        results = await search_engine.search("test")
        assert isinstance(results, list)
        # Should filter out malformed results
        assert len(results) == 0  # All results are malformed


# Test configuration and utilities
@pytest.fixture(autouse=True)
async def cleanup_search_engine():
    """Ensure search engines are properly cleaned up after tests"""
    yield
    # Cleanup any lingering resources
    pass


@pytest.mark.asyncio
class TestAsyncContextManagement:
    """Test async context management for search engine"""
    
    async def test_context_manager_protocol(self):
        """Test search engine as async context manager"""
        with patch('src.core.unified_search_engine.QdrantStorage'):
            with patch('src.ingestion.local_embeddings.LocalEmbeddingGenerator'):
                async with UnifiedSearchEngine() as engine:
                    assert engine is not None
                    # Should be properly initialized
                    # Test some operation
                    pass
                # Should be properly closed