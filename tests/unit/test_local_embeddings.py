"""
Tests for LocalEmbeddingGenerator class for Ollama integration
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime

# Import the models that will be used
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from core.models import Chunk, ChunkType
from core.config import EmbeddingConfig


class TestLocalEmbeddingGeneratorInterface:
    """Test the interface that LocalEmbeddingGenerator should implement"""
    
    def test_required_methods_exist(self):
        """Test that all required methods exist in the interface"""
        # This test will fail until we implement the class
        from ingestion.local_embeddings import LocalEmbeddingGenerator
        
        generator = LocalEmbeddingGenerator()
        
        # Check required methods exist
        assert hasattr(generator, 'generate_embeddings')
        assert hasattr(generator, 'generate_query_embedding')
        assert hasattr(generator, 'save_cache')
        assert hasattr(generator, 'load_cache')
        assert hasattr(generator, 'get_stats')
        assert hasattr(generator, 'cleanup')
        
        # Check required properties/attributes
        assert hasattr(generator, 'model_name')
        assert hasattr(generator, 'embedding_dimension')
        assert hasattr(generator, 'ollama_host')


class TestLocalEmbeddingGeneratorInitialization:
    """Test LocalEmbeddingGenerator initialization"""
    
    @patch('ingestion.local_embeddings.httpx.AsyncClient')
    def test_default_initialization(self, mock_client):
        """Test default initialization uses config values"""
        from ingestion.local_embeddings import LocalEmbeddingGenerator
        
        generator = LocalEmbeddingGenerator()
        
        assert generator.model_name == "nomic-embed-text"
        assert generator.embedding_dimension == 768
        assert generator.ollama_host == "http://localhost:11434"
        assert generator.timeout == 30
        assert isinstance(generator.cache, dict)
        assert generator.cache_hits == 0
        assert generator.cache_misses == 0
    
    @patch('ingestion.local_embeddings.httpx.AsyncClient')
    def test_custom_initialization(self, mock_client):
        """Test initialization with custom model name"""
        from ingestion.local_embeddings import LocalEmbeddingGenerator
        
        generator = LocalEmbeddingGenerator(model_name="custom-model")
        
        assert generator.model_name == "custom-model"
        assert generator.embedding_dimension == 768  # Should still use config default
    
    @patch('ingestion.local_embeddings.httpx.AsyncClient')
    def test_config_integration(self, mock_client):
        """Test that generator uses config properly"""
        from ingestion.local_embeddings import LocalEmbeddingGenerator
        
        config = EmbeddingConfig(
            model="test-model",
            dimension=512,
            ollama_host="http://test:11434",
            timeout=60
        )
        
        with patch('ingestion.local_embeddings.get_config', return_value=Mock(embedding=config)):
            generator = LocalEmbeddingGenerator()
            
            assert generator.model_name == "test-model"
            assert generator.embedding_dimension == 512
            assert generator.ollama_host == "http://test:11434"
            assert generator.timeout == 60


class TestOllamaConnection:
    """Test Ollama connection and verification"""
    
    @pytest.mark.asyncio
    async def test_ollama_verification_success(self):
        """Test successful Ollama verification"""
        from ingestion.local_embeddings import LocalEmbeddingGenerator
        
        mock_client = AsyncMock()
        
        # Mock version check
        version_response = Mock()
        version_response.status_code = 200
        version_response.json.return_value = {"version": "0.1.0"}
        
        # Mock tags check  
        tags_response = Mock()
        tags_response.status_code = 200
        tags_response.json.return_value = {
            "models": [{"name": "nomic-embed-text"}, {"name": "other-model"}]
        }
        
        mock_client.get = AsyncMock(side_effect=[version_response, tags_response])
        
        with patch('ingestion.local_embeddings.httpx.AsyncClient', return_value=mock_client):
            generator = LocalEmbeddingGenerator()
            await generator._verify_ollama()
            
            # Verify API calls were made
            assert mock_client.get.call_count == 2
            mock_client.get.assert_any_call("http://localhost:11434/api/version")
            mock_client.get.assert_any_call("http://localhost:11434/api/tags")
    
    @pytest.mark.asyncio
    async def test_ollama_verification_model_missing(self):
        """Test Ollama verification when model is missing"""
        from ingestion.local_embeddings import LocalEmbeddingGenerator
        
        mock_client = AsyncMock()
        
        # Mock version check success
        version_response = Mock()
        version_response.status_code = 200
        version_response.json.return_value = {"version": "0.1.0"}
        
        # Mock tags check with missing model
        tags_response = Mock()
        tags_response.status_code = 200
        tags_response.json.return_value = {
            "models": [{"name": "other-model"}]  # nomic-embed-text missing
        }
        
        mock_client.get = AsyncMock(side_effect=[version_response, tags_response])
        
        with patch('ingestion.local_embeddings.httpx.AsyncClient', return_value=mock_client):
            with patch('ingestion.local_embeddings.logger') as mock_logger:
                generator = LocalEmbeddingGenerator()
                await generator._verify_ollama()
                
                # Should log error about missing model
                mock_logger.error.assert_called()
                error_call = mock_logger.error.call_args[0][0]
                assert "not found" in error_call
    
    @pytest.mark.asyncio 
    async def test_ollama_verification_connection_error(self):
        """Test Ollama verification with connection error"""
        from ingestion.local_embeddings import LocalEmbeddingGenerator
        
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("Connection refused"))
        
        with patch('ingestion.local_embeddings.httpx.AsyncClient', return_value=mock_client):
            with patch('ingestion.local_embeddings.logger') as mock_logger:
                generator = LocalEmbeddingGenerator()
                await generator._verify_ollama()
                
                # Should log connection error
                mock_logger.error.assert_called()
                error_call = mock_logger.error.call_args[0][0]
                assert "Cannot connect to Ollama" in error_call


class TestEmbeddingGeneration:
    """Test embedding generation functionality"""
    
    @pytest.mark.asyncio
    async def test_generate_single_embedding_success(self):
        """Test successful single embedding generation"""
        from ingestion.local_embeddings import LocalEmbeddingGenerator
        
        mock_client = AsyncMock()
        
        # Mock successful embedding response
        embedding_response = Mock()
        embedding_response.status_code = 200
        embedding_response.json.return_value = {
            "embedding": [0.1, 0.2, 0.3] + [0.0] * 765  # 768 total
        }
        
        mock_client.post = AsyncMock(return_value=embedding_response)
        
        with patch('ingestion.local_embeddings.httpx.AsyncClient', return_value=mock_client):
            generator = LocalEmbeddingGenerator()
            
            embedding = await generator._generate_single_embedding("test text")
            
            assert len(embedding) == 768
            assert embedding[:3] == [0.1, 0.2, 0.3]
            
            # Verify API call
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "http://localhost:11434/api/embeddings"
            assert call_args[1]["json"]["model"] == "nomic-embed-text"
            assert call_args[1]["json"]["prompt"] == "test text"
    
    @pytest.mark.asyncio
    async def test_generate_single_embedding_error(self):
        """Test single embedding generation with API error"""
        from ingestion.local_embeddings import LocalEmbeddingGenerator
        
        mock_client = AsyncMock()
        
        # Mock error response
        error_response = Mock()
        error_response.status_code = 500
        error_response.text = "Internal server error"
        
        mock_client.post = AsyncMock(return_value=error_response)
        
        with patch('ingestion.local_embeddings.httpx.AsyncClient', return_value=mock_client):
            with patch('ingestion.local_embeddings.logger') as mock_logger:
                generator = LocalEmbeddingGenerator()
                
                embedding = await generator._generate_single_embedding("test text")
                
                # Should return zero vector on error
                assert len(embedding) == 768
                assert all(x == 0.0 for x in embedding)
                
                # Should log error
                mock_logger.error.assert_called()
    
    @pytest.mark.asyncio
    async def test_generate_single_embedding_dimension_mismatch(self):
        """Test handling of dimension mismatch"""
        from ingestion.local_embeddings import LocalEmbeddingGenerator
        
        mock_client = AsyncMock()
        
        # Mock response with wrong dimension
        embedding_response = Mock()
        embedding_response.status_code = 200
        embedding_response.json.return_value = {
            "embedding": [0.1, 0.2, 0.3]  # Only 3 dimensions instead of 768
        }
        
        mock_client.post = AsyncMock(return_value=embedding_response)
        
        with patch('ingestion.local_embeddings.httpx.AsyncClient', return_value=mock_client):
            with patch('ingestion.local_embeddings.logger') as mock_logger:
                generator = LocalEmbeddingGenerator()
                
                embedding = await generator._generate_single_embedding("test text")
                
                # Should still return the embedding
                assert len(embedding) == 3
                assert embedding == [0.1, 0.2, 0.3]
                
                # Should log warning about dimension mismatch
                mock_logger.warning.assert_called()
                warning_call = mock_logger.warning.call_args[0][0]
                assert "dimension mismatch" in warning_call


class TestBatchEmbeddingGeneration:
    """Test batch embedding generation"""
    
    @pytest.mark.asyncio
    async def test_generate_batch_embeddings(self):
        """Test generating embeddings for multiple texts"""
        from ingestion.local_embeddings import LocalEmbeddingGenerator
        
        mock_client = AsyncMock()
        
        # Mock embedding responses
        def mock_embedding_response(url, json):
            response = Mock()
            response.status_code = 200
            # Different embeddings for different texts
            text = json["prompt"]
            if "first" in text:
                embedding = [0.1] * 768
            elif "second" in text:
                embedding = [0.2] * 768
            else:
                embedding = [0.3] * 768
            
            response.json.return_value = {"embedding": embedding}
            return response
        
        mock_client.post = AsyncMock(side_effect=mock_embedding_response)
        
        with patch('ingestion.local_embeddings.httpx.AsyncClient', return_value=mock_client):
            generator = LocalEmbeddingGenerator()
            
            texts = ["first text", "second text", "third text"]
            embeddings = await generator._generate_batch_embeddings(texts)
            
            assert len(embeddings) == 3
            assert all(len(emb) == 768 for emb in embeddings)
            assert embeddings[0] == [0.1] * 768
            assert embeddings[1] == [0.2] * 768
            assert embeddings[2] == [0.3] * 768
            
            # Should make 3 API calls
            assert mock_client.post.call_count == 3
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_with_chunks(self):
        """Test generate_embeddings method with Chunk objects"""
        from ingestion.local_embeddings import LocalEmbeddingGenerator
        
        # Create test chunks
        chunks = [
            Chunk(id="1", book_id="test", chunk_index=0, text="first chunk"),
            Chunk(id="2", book_id="test", chunk_index=1, text="second chunk"),
        ]
        
        mock_client = AsyncMock()
        
        # Mock embedding responses
        def mock_embedding_response(url, json):
            response = Mock()
            response.status_code = 200
            text = json["prompt"]
            if "first" in text:
                embedding = [0.1] * 768
            else:
                embedding = [0.2] * 768
            response.json.return_value = {"embedding": embedding}
            return response
        
        mock_client.post = AsyncMock(side_effect=mock_embedding_response)
        
        with patch('ingestion.local_embeddings.httpx.AsyncClient', return_value=mock_client):
            generator = LocalEmbeddingGenerator()
            
            embeddings = await generator.generate_embeddings(chunks, show_progress=False)
            
            assert len(embeddings) == 2
            assert embeddings[0] == [0.1] * 768
            assert embeddings[1] == [0.2] * 768
            
            # Check cache was updated
            assert len(generator.cache) == 2
            assert generator.cache_misses == 2
            assert generator.cache_hits == 0


class TestEmbeddingCaching:
    """Test embedding caching functionality"""
    
    def test_cache_key_generation(self):
        """Test cache key generation"""
        from ingestion.local_embeddings import LocalEmbeddingGenerator
        
        with patch('ingestion.local_embeddings.httpx.AsyncClient'):
            generator = LocalEmbeddingGenerator()
            
            key1 = generator._get_cache_key("test text")
            key2 = generator._get_cache_key("test text")
            key3 = generator._get_cache_key("different text")
            
            # Same text should generate same key
            assert key1 == key2
            # Different text should generate different key
            assert key1 != key3
            # Key should include model name (tested by changing model)
            generator.model_name = "different-model"
            key4 = generator._get_cache_key("test text")
            assert key1 != key4
    
    @pytest.mark.asyncio
    async def test_cache_hits_and_misses(self):
        """Test cache hit and miss tracking"""
        from ingestion.local_embeddings import LocalEmbeddingGenerator
        
        chunks = [
            Chunk(id="1", book_id="test", chunk_index=0, text="cached text"),
            Chunk(id="2", book_id="test", chunk_index=1, text="new text"),
        ]
        
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=Mock(
            status_code=200,
            json=Mock(return_value={"embedding": [0.1] * 768})
        ))
        
        with patch('ingestion.local_embeddings.httpx.AsyncClient', return_value=mock_client):
            generator = LocalEmbeddingGenerator()
            
            # Pre-populate cache
            cache_key = generator._get_cache_key("cached text")
            generator.cache[cache_key] = [0.5] * 768
            
            embeddings = await generator.generate_embeddings(chunks, show_progress=False)
            
            assert len(embeddings) == 2
            assert embeddings[0] == [0.5] * 768  # From cache
            assert embeddings[1] == [0.1] * 768  # From API
            
            # Check stats
            assert generator.cache_hits == 1
            assert generator.cache_misses == 1
            
            # Only one API call should have been made
            assert mock_client.post.call_count == 1
    
    def test_save_and_load_cache(self):
        """Test saving and loading cache to/from disk"""
        from ingestion.local_embeddings import LocalEmbeddingGenerator
        
        with patch('ingestion.local_embeddings.httpx.AsyncClient'):
            generator = LocalEmbeddingGenerator()
            
            # Populate cache
            generator.cache = {
                "key1": [0.1] * 768,
                "key2": [0.2] * 768
            }
            generator.cache_hits = 5
            generator.cache_misses = 10
            
            # Save cache
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                cache_file = f.name
            
            try:
                generator.save_cache(cache_file)
                
                # Verify file was created and has correct content
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                assert cache_data['model_name'] == "nomic-embed-text"
                assert cache_data['embedding_dimension'] == 768
                assert len(cache_data['cache']) == 2
                assert cache_data['stats']['hits'] == 5
                assert cache_data['stats']['misses'] == 10
                assert 'saved_at' in cache_data['stats']
                
                # Test loading cache
                new_generator = LocalEmbeddingGenerator()
                new_generator.load_cache(cache_file)
                
                assert len(new_generator.cache) == 2
                assert new_generator.cache["key1"] == [0.1] * 768
                assert new_generator.cache["key2"] == [0.2] * 768
                
            finally:
                Path(cache_file).unlink()
    
    def test_load_cache_model_mismatch(self):
        """Test loading cache with model mismatch"""
        from ingestion.local_embeddings import LocalEmbeddingGenerator
        
        with patch('ingestion.local_embeddings.httpx.AsyncClient'):
            generator = LocalEmbeddingGenerator()
            
            # Create cache file with different model
            cache_data = {
                'model_name': 'different-model',
                'embedding_dimension': 512,
                'cache': {'key1': [0.1] * 512},
                'stats': {'hits': 0, 'misses': 0}
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(cache_data, f)
                cache_file = f.name
            
            try:
                with patch('ingestion.local_embeddings.logger') as mock_logger:
                    generator.load_cache(cache_file)
                    
                    # Should log warning about model mismatch
                    mock_logger.warning.assert_called()
                    warning_call = mock_logger.warning.call_args[0][0]
                    assert "model mismatch" in warning_call
                    
                    # Cache should remain empty
                    assert len(generator.cache) == 0
                    
            finally:
                Path(cache_file).unlink()


class TestQueryEmbedding:
    """Test query embedding generation"""
    
    @pytest.mark.asyncio
    async def test_generate_query_embedding(self):
        """Test generating embedding for search query"""
        from ingestion.local_embeddings import LocalEmbeddingGenerator
        
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=Mock(
            status_code=200,
            json=Mock(return_value={"embedding": [0.5] * 768})
        ))
        
        with patch('ingestion.local_embeddings.httpx.AsyncClient', return_value=mock_client):
            generator = LocalEmbeddingGenerator()
            
            embedding = await generator.generate_query_embedding("test query")
            
            assert len(embedding) == 768
            assert embedding == [0.5] * 768
            
            # Should update cache
            assert len(generator.cache) == 1
            assert generator.cache_misses == 1
            
            # Second call should use cache
            embedding2 = await generator.generate_query_embedding("test query")
            assert embedding2 == embedding
            assert generator.cache_hits == 1
            
            # Only one API call should have been made
            assert mock_client.post.call_count == 1


class TestStatistics:
    """Test statistics and monitoring"""
    
    def test_get_stats(self):
        """Test get_stats method"""
        from ingestion.local_embeddings import LocalEmbeddingGenerator
        
        with patch('ingestion.local_embeddings.httpx.AsyncClient'):
            generator = LocalEmbeddingGenerator()
            
            # Set up some test data
            generator.cache = {"key1": [0.1] * 768, "key2": [0.2] * 768}
            generator.cache_hits = 3
            generator.cache_misses = 7
            
            stats = generator.get_stats()
            
            expected_stats = {
                'model_name': 'nomic-embed-text',
                'embedding_dimension': 768,
                'ollama_host': 'http://localhost:11434',
                'cache_size': 2,
                'cache_hits': 3,
                'cache_misses': 7,
                'cache_hit_rate': 0.3,  # 3/(3+7)
                'total_requests': 10
            }
            
            assert stats == expected_stats
    
    def test_get_stats_no_requests(self):
        """Test get_stats with no requests made"""
        from ingestion.local_embeddings import LocalEmbeddingGenerator
        
        with patch('ingestion.local_embeddings.httpx.AsyncClient'):
            generator = LocalEmbeddingGenerator()
            
            stats = generator.get_stats()
            
            assert stats['cache_hit_rate'] == 0
            assert stats['total_requests'] == 0


class TestCleanup:
    """Test cleanup functionality"""
    
    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test cleanup method closes HTTP client"""
        from ingestion.local_embeddings import LocalEmbeddingGenerator
        
        mock_client = AsyncMock()
        
        with patch('ingestion.local_embeddings.httpx.AsyncClient', return_value=mock_client):
            generator = LocalEmbeddingGenerator()
            
            await generator.cleanup()
            
            # Should close the HTTP client
            mock_client.aclose.assert_called_once()


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_empty_list(self):
        """Test generating embeddings for empty chunk list"""
        from ingestion.local_embeddings import LocalEmbeddingGenerator
        
        with patch('ingestion.local_embeddings.httpx.AsyncClient'):
            generator = LocalEmbeddingGenerator()
            
            embeddings = await generator.generate_embeddings([])
            
            assert embeddings == []
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_network_error(self):
        """Test handling network errors during embedding generation"""
        from ingestion.local_embeddings import LocalEmbeddingGenerator
        
        chunks = [Chunk(id="1", book_id="test", chunk_index=0, text="test text")]
        
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("Network error"))
        
        with patch('ingestion.local_embeddings.httpx.AsyncClient', return_value=mock_client):
            with patch('ingestion.local_embeddings.logger') as mock_logger:
                generator = LocalEmbeddingGenerator()
                
                embedding = await generator._generate_single_embedding("test text")
                
                # Should return zero vector on network error
                assert len(embedding) == 768
                assert all(x == 0.0 for x in embedding)
                
                # Should log error
                mock_logger.error.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])