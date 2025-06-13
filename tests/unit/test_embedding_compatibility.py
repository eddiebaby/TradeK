"""
Test that LocalEmbeddingGenerator is compatible with original EmbeddingGenerator interface
and can pass the system tests that expect the original behavior.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from core.models import Chunk
from ingestion.local_embeddings import LocalEmbeddingGenerator


class TestOriginalEmbeddingGeneratorCompatibility:
    """Test compatibility with original EmbeddingGenerator expected behavior"""
    
    @pytest.mark.asyncio
    async def test_constructor_with_model_name_parameter(self):
        """Test that constructor accepts model_name parameter like original"""
        # This mimics how the original system test calls it:
        # generator = EmbeddingGenerator("text-embedding-ada-002")
        
        with patch('ingestion.local_embeddings.httpx.AsyncClient'):
            generator = LocalEmbeddingGenerator("nomic-embed-text")
            
            assert generator.model_name == "nomic-embed-text"
            assert hasattr(generator, 'embedding_dimension')
            assert hasattr(generator, 'cache')
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_with_test_chunks(self):
        """Test generate_embeddings with chunks like in system test"""
        # This mimics the system test pattern:
        test_chunks = [
            Chunk(
                book_id="test",
                chunk_index=0,
                text="Moving averages are technical indicators"
            ),
            Chunk(
                book_id="test",
                chunk_index=1,
                text="Python is used for algorithmic trading"
            )
        ]
        
        mock_client = AsyncMock()
        
        # Mock successful embedding responses
        def mock_embedding_response(url, json):
            response = Mock()
            response.status_code = 200
            text = json["prompt"]
            if "Moving averages" in text:
                embedding = [0.1] * 768
            else:
                embedding = [0.2] * 768
            response.json.return_value = {"embedding": embedding}
            return response
        
        mock_client.post = AsyncMock(side_effect=mock_embedding_response)
        
        with patch('ingestion.local_embeddings.httpx.AsyncClient', return_value=mock_client):
            generator = LocalEmbeddingGenerator("nomic-embed-text")
            
            embeddings = await generator.generate_embeddings(test_chunks)
            
            # Verify behavior expected by system test
            assert len(embeddings) == len(test_chunks)
            assert all(len(emb) == 768 for emb in embeddings)
            assert embeddings[0] == [0.1] * 768
            assert embeddings[1] == [0.2] * 768
    
    @pytest.mark.asyncio
    async def test_generate_query_embedding_compatibility(self):
        """Test query embedding generation like in system test"""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=Mock(
            status_code=200,
            json=Mock(return_value={"embedding": [0.5] * 768})
        ))
        
        with patch('ingestion.local_embeddings.httpx.AsyncClient', return_value=mock_client):
            generator = LocalEmbeddingGenerator("nomic-embed-text")
            
            # This mimics: query_embedding = await generator.generate_query_embedding("trading strategies")
            query_embedding = await generator.generate_query_embedding("trading strategies")
            
            # Verify behavior expected by system test
            assert len(query_embedding) > 0
            assert len(query_embedding) == 768  # Dimension expected by tests
    
    def test_get_stats_compatibility(self):
        """Test stats reporting like in system test"""
        with patch('ingestion.local_embeddings.httpx.AsyncClient'):
            generator = LocalEmbeddingGenerator("nomic-embed-text")
            
            # Add some mock cache data
            generator.cache = {"key1": [0.1] * 768, "key2": [0.2] * 768}
            generator.cache_hits = 5
            generator.cache_misses = 3
            
            # This mimics: stats = generator.get_stats()
            stats = generator.get_stats()
            
            # Verify stats structure expected by system test
            assert 'cache_size' in stats
            assert stats['cache_size'] > 0
            assert 'model_name' in stats
            assert 'embedding_dimension' in stats
            
            # System test checks: stats['cache_size'] > 0
            assert stats['cache_size'] == 2
    
    @pytest.mark.asyncio 
    async def test_caching_behavior_compatibility(self):
        """Test that caching works as expected by system tests"""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=Mock(
            status_code=200,
            json=Mock(return_value={"embedding": [0.3] * 768})
        ))
        
        with patch('ingestion.local_embeddings.httpx.AsyncClient', return_value=mock_client):
            generator = LocalEmbeddingGenerator("nomic-embed-text")
            
            # First call should generate embedding
            chunk = Chunk(book_id="test", chunk_index=0, text="test text")
            embeddings1 = await generator.generate_embeddings([chunk])
            
            # Second call should use cache
            embeddings2 = await generator.generate_embeddings([chunk])
            
            # Verify caching worked
            assert embeddings1 == embeddings2
            assert generator.cache_hits > 0
            assert len(generator.cache) > 0
            
            # Only one API call should have been made
            assert mock_client.post.call_count == 1
    
    @pytest.mark.asyncio
    async def test_error_handling_compatibility(self):
        """Test error handling doesn't break system test expectations"""
        # Test that errors don't crash, but return sensible defaults
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("Network error"))
        
        with patch('ingestion.local_embeddings.httpx.AsyncClient', return_value=mock_client):
            generator = LocalEmbeddingGenerator("nomic-embed-text")
            
            chunk = Chunk(book_id="test", chunk_index=0, text="test text")
            embeddings = await generator.generate_embeddings([chunk])
            
            # Should return zero vectors instead of crashing
            assert len(embeddings) == 1
            assert len(embeddings[0]) == 768
            assert all(x == 0.0 for x in embeddings[0])
    
    def test_api_key_error_simulation(self):
        """Test handling of missing API key scenario like original"""
        # The original system test expects a ValueError with "API key" message
        # when no API key is available. Our implementation doesn't need an API key,
        # but should handle the test gracefully.
        
        with patch('ingestion.local_embeddings.httpx.AsyncClient'):
            # Should not raise ValueError about API key
            try:
                generator = LocalEmbeddingGenerator("nomic-embed-text")
                # Constructor should succeed
                assert generator.model_name == "nomic-embed-text"
            except ValueError as e:
                # If it does raise ValueError, it should not be about API key
                assert "API key" not in str(e)
    
    @pytest.mark.asyncio
    async def test_dimension_compatibility(self):
        """Test that embedding dimension matches configuration"""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=Mock(
            status_code=200,
            json=Mock(return_value={"embedding": [0.1] * 768})
        ))
        
        with patch('ingestion.local_embeddings.httpx.AsyncClient', return_value=mock_client):
            generator = LocalEmbeddingGenerator("nomic-embed-text")
            
            # Test that embedding dimension is accessible like original
            assert hasattr(generator, 'embedding_dimension')
            assert generator.embedding_dimension == 768
            
            # Test that generated embeddings have correct dimension
            chunk = Chunk(book_id="test", chunk_index=0, text="test text")
            embeddings = await generator.generate_embeddings([chunk])
            
            assert len(embeddings[0]) == generator.embedding_dimension
    
    @pytest.mark.asyncio
    async def test_cleanup_compatibility(self):
        """Test cleanup method exists and works"""
        mock_client = AsyncMock()
        
        with patch('ingestion.local_embeddings.httpx.AsyncClient', return_value=mock_client):
            generator = LocalEmbeddingGenerator("nomic-embed-text")
            
            # Should have cleanup method
            assert hasattr(generator, 'cleanup')
            assert callable(generator.cleanup)
            
            # Should be able to call cleanup
            await generator.cleanup()
            
            # Should close the client
            mock_client.aclose.assert_called_once()


class TestSystemTestScenarios:
    """Test specific scenarios from the system test script"""
    
    @pytest.mark.asyncio
    async def test_system_test_embedding_scenario(self):
        """Replicate the exact embedding test scenario from system test"""
        # This replicates the test_embeddings() function from scripts/test_system.py
        
        mock_client = AsyncMock()
        
        # Mock responses for the specific test chunks
        def mock_embedding_response(url, json):
            response = Mock()
            response.status_code = 200
            text = json["prompt"]
            if "Moving averages" in text:
                embedding = [0.1] * 768
            elif "Python is used" in text:
                embedding = [0.2] * 768
            elif "trading strategies" in text:
                embedding = [0.3] * 768
            else:
                embedding = [0.4] * 768
            response.json.return_value = {"embedding": embedding}
            return response
        
        mock_client.post = AsyncMock(side_effect=mock_embedding_response)
        
        with patch('ingestion.local_embeddings.httpx.AsyncClient', return_value=mock_client):
            # This mimics: generator = EmbeddingGenerator("text-embedding-ada-002")
            generator = LocalEmbeddingGenerator("nomic-embed-text")
            
            # This mimics the exact test chunks from system test
            test_chunks = [
                Chunk(
                    book_id="test",
                    chunk_index=0,
                    text="Moving averages are technical indicators"
                ),
                Chunk(
                    book_id="test",
                    chunk_index=1,
                    text="Python is used for algorithmic trading"
                )
            ]
            
            # This mimics: embeddings = await generator.generate_embeddings(test_chunks)
            embeddings = await generator.generate_embeddings(test_chunks)
            
            # System test checks: len(embeddings) == len(test_chunks)
            assert len(embeddings) == len(test_chunks)
            
            # System test message: f"Generated {len(embeddings)} embeddings"
            print(f"Generated {len(embeddings)} embeddings")
            
            # This mimics: query_embedding = await generator.generate_query_embedding("trading strategies")
            query_embedding = await generator.generate_query_embedding("trading strategies")
            
            # System test checks: len(query_embedding) > 0
            assert len(query_embedding) > 0
            
            # System test message: f"Embedding dimension: {len(query_embedding)}"
            print(f"Embedding dimension: {len(query_embedding)}")
            
            # This mimics: stats = generator.get_stats()
            stats = generator.get_stats()
            
            # System test checks: stats['cache_size'] > 0
            assert stats['cache_size'] > 0
            
            # System test message: f"Cache size: {stats['cache_size']}"
            print(f"Cache size: {stats['cache_size']}")


if __name__ == "__main__":
    pytest.main([__file__])