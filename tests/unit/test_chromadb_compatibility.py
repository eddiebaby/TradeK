"""
Test QdrantStorage compatibility with ChromaDBStorage interface

This ensures that QdrantStorage can be used as a drop-in replacement for ChromaDBStorage
by testing the same interfaces and expected behaviors.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from core.models import Chunk, ChunkType


class TestQdrantChromaCompatibility:
    """Test that QdrantStorage behaves identically to ChromaDBStorage for basic operations"""
    
    @pytest.mark.asyncio
    async def test_save_embeddings_chroma_compatibility(self):
        """Test that save_embeddings works identically to ChromaDB version"""
        # This test mimics how ChromaDBStorage save_embeddings is expected to work
        mock_client = MagicMock()
        
        # Mock successful upsert (Qdrant equivalent of ChromaDB add)
        mock_operation_info = Mock()
        mock_operation_info.status.name = "COMPLETED"
        mock_client.upsert.return_value = mock_operation_info
        
        with patch('core.qdrant_storage.QdrantClient', return_value=mock_client):
            mock_collections = Mock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            
            from core.qdrant_storage import QdrantStorage
            
            storage = QdrantStorage()
            
            # Create test chunks (same format that ChromaDB expects)
            chunks = [
                Chunk(
                    id="test_chunk_1",
                    book_id="test_book",
                    chunk_index=0,
                    text="First test chunk for compatibility",
                    chapter="Chapter 1",
                    page_start=1,
                    page_end=2
                ),
                Chunk(
                    id="test_chunk_2", 
                    book_id="test_book",
                    chunk_index=1,
                    text="Second test chunk for compatibility",
                    section="Introduction"
                )
            ]
            
            # Embeddings should be same format as ChromaDB (list of lists of floats)
            embeddings = [
                [0.1] * 768,  # nomic-embed-text dimension
                [0.2] * 768
            ]
            
            # Should behave identically to ChromaDB: return True on success
            success = await storage.save_embeddings(chunks, embeddings)
            
            assert success is True
            
            # Verify chunks were stored with all metadata (ChromaDB compatibility)
            mock_client.upsert.assert_called()
            upsert_call = mock_client.upsert.call_args
            points = upsert_call[1]['points']
            
            # Check that all ChromaDB metadata fields are preserved
            point1 = points[0]
            assert point1.payload['chunk_id'] == "test_chunk_1"
            assert point1.payload['book_id'] == "test_book"
            assert point1.payload['text'] == "First test chunk for compatibility"
            assert point1.payload['chunk_index'] == 0
            assert point1.payload['chunk_type'] == ChunkType.TEXT.value
            assert point1.payload['chapter'] == "Chapter 1"
            assert point1.payload['page_start'] == 1
            assert point1.payload['page_end'] == 2
            assert 'created_at' in point1.payload
    
    @pytest.mark.asyncio
    async def test_search_semantic_chroma_compatibility(self):
        """Test that search_semantic returns same format as ChromaDB"""
        mock_client = MagicMock()
        
        # Mock search response (similar to ChromaDB query response format)
        mock_search_result = [
            Mock(
                payload={
                    'chunk_id': 'chunk_1',
                    'text': 'ChromaDB compatible result text',
                    'book_id': 'book_1',
                    'chunk_index': 0,
                    'chunk_type': 'text',
                    'chapter': 'Chapter 1'
                },
                score=0.95  # Qdrant score (similarity)
            ),
            Mock(
                payload={
                    'chunk_id': 'chunk_2',
                    'text': 'Second compatible result',
                    'book_id': 'book_1',
                    'chunk_index': 1,
                    'chunk_type': 'text'
                },
                score=0.87
            )
        ]
        mock_client.search.return_value = mock_search_result
        
        with patch('core.qdrant_storage.QdrantClient', return_value=mock_client):
            mock_collections = Mock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            
            from core.qdrant_storage import QdrantStorage
            
            storage = QdrantStorage()
            
            query_embedding = [0.5] * 768
            
            # Test with ChromaDB-style filter (book_ids as list)
            filter_dict = {'book_ids': ['book_1', 'book_2']}
            
            results = await storage.search_semantic(query_embedding, filter_dict=filter_dict, limit=5)
            
            # Should return same format as ChromaDB
            assert len(results) == 2
            
            # Check ChromaDB-compatible result format
            result1 = results[0]
            assert 'chunk_id' in result1
            assert 'text' in result1
            assert 'metadata' in result1
            assert 'score' in result1
            assert 'distance' in result1
            
            # Score should be similarity (same as ChromaDB)
            assert result1['score'] == 0.95
            
            # Distance should be computed from score (compatibility)
            assert result1['distance'] == 1 - 0.95
            
            # Metadata should contain all fields except chunk_id and text
            metadata = result1['metadata']
            assert 'book_id' in metadata
            assert 'chunk_index' in metadata
            assert 'chunk_type' in metadata
            assert 'chapter' in metadata
            assert 'chunk_id' not in metadata  # Excluded from metadata (ChromaDB style)
            assert 'text' not in metadata  # Excluded from metadata (ChromaDB style)
    
    @pytest.mark.asyncio
    async def test_delete_embeddings_chroma_compatibility(self):
        """Test that delete_embeddings works like ChromaDB delete"""
        mock_client = MagicMock()
        
        with patch('core.qdrant_storage.QdrantClient', return_value=mock_client):
            mock_collections = Mock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            
            from core.qdrant_storage import QdrantStorage
            
            storage = QdrantStorage()
            
            # Delete by chunk IDs (same as ChromaDB delete by ids)
            chunk_ids = ["chunk_1", "chunk_2", "chunk_3"]
            success = await storage.delete_embeddings(chunk_ids)
            
            # Should return True on success (ChromaDB compatibility)
            assert success is True
            
            # Should have called delete with appropriate filter
            mock_client.delete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_collection_stats_chroma_compatibility(self):
        """Test that collection stats match ChromaDB format"""
        mock_client = MagicMock()
        
        # Mock collection info similar to ChromaDB collection.count()
        mock_collection_info = Mock()
        mock_collection_info.points_count = 1500  # equivalent to ChromaDB count
        mock_collection_info.config.params.vectors.size = 768
        mock_distance = Mock()
        mock_distance.name = "COSINE"
        mock_collection_info.config.params.vectors.distance = mock_distance
        mock_collection_info.segments_count = 3
        mock_collection_info.status = "green"
        
        mock_client.get_collection.return_value = mock_collection_info
        
        with patch('core.qdrant_storage.QdrantClient', return_value=mock_client):
            mock_collections = Mock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            
            from core.qdrant_storage import QdrantStorage
            
            storage = QdrantStorage()
            
            stats = await storage.get_collection_stats()
            
            # Should include ChromaDB-compatible stats
            assert 'collection_name' in stats
            assert 'total_embeddings' in stats  # equivalent to ChromaDB count
            assert stats['total_embeddings'] == 1500
            assert stats['collection_name'] == "tradeknowledge"
            
            # Additional Qdrant-specific stats are OK (superset of ChromaDB)
            assert 'vector_size' in stats
            assert 'distance_metric' in stats


class TestFilterCompatibility:
    """Test that QdrantStorage filter format is compatible with ChromaDB expectations"""
    
    @pytest.mark.asyncio
    async def test_book_id_filter_compatibility(self):
        """Test book_id filtering works like ChromaDB where clause"""
        mock_client = MagicMock()
        mock_client.search.return_value = []
        
        with patch('core.qdrant_storage.QdrantClient', return_value=mock_client):
            mock_collections = Mock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            
            from core.qdrant_storage import QdrantStorage
            
            storage = QdrantStorage()
            
            query_embedding = [0.5] * 768
            
            # Test ChromaDB-style filter: {'book_ids': ['book1', 'book2']}
            # ChromaDB equivalent: where={'book_id': {'$in': ['book1', 'book2']}}
            filter_dict = {'book_ids': ['book_1', 'book_2']}
            
            await storage.search_semantic(query_embedding, filter_dict=filter_dict)
            
            # Should have applied filter
            search_call = mock_client.search.call_args
            query_filter = search_call[1]['query_filter']
            
            assert query_filter is not None
            # Filter should handle multiple book IDs (ChromaDB $in equivalent)
            assert len(query_filter.must) >= 1
    
    @pytest.mark.asyncio
    async def test_single_book_filter_compatibility(self):
        """Test single book_id filtering"""
        mock_client = MagicMock()
        mock_client.search.return_value = []
        
        with patch('core.qdrant_storage.QdrantClient', return_value=mock_client):
            mock_collections = Mock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            
            from core.qdrant_storage import QdrantStorage
            
            storage = QdrantStorage()
            
            query_embedding = [0.5] * 768
            
            # Single book filter
            filter_dict = {'book_ids': ['book_1']}
            
            await storage.search_semantic(query_embedding, filter_dict=filter_dict)
            
            search_call = mock_client.search.call_args
            query_filter = search_call[1]['query_filter']
            
            assert query_filter is not None
    
    @pytest.mark.asyncio
    async def test_chunk_type_filter_compatibility(self):
        """Test chunk_type filtering compatibility"""
        mock_client = MagicMock()
        mock_client.search.return_value = []
        
        with patch('core.qdrant_storage.QdrantClient', return_value=mock_client):
            mock_collections = Mock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            
            from core.qdrant_storage import QdrantStorage
            
            storage = QdrantStorage()
            
            query_embedding = [0.5] * 768
            
            # ChromaDB-style chunk_type filter
            filter_dict = {'chunk_type': 'text'}
            
            await storage.search_semantic(query_embedding, filter_dict=filter_dict)
            
            search_call = mock_client.search.call_args
            query_filter = search_call[1]['query_filter']
            
            assert query_filter is not None


class TestEdgeCaseCompatibility:
    """Test edge cases that ChromaDB handles"""
    
    @pytest.mark.asyncio
    async def test_empty_embeddings_compatibility(self):
        """Test handling empty embeddings like ChromaDB"""
        mock_client = MagicMock()
        
        with patch('core.qdrant_storage.QdrantClient', return_value=mock_client):
            mock_collections = Mock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            
            from core.qdrant_storage import QdrantStorage
            
            storage = QdrantStorage()
            
            # ChromaDB allows empty saves and returns True
            success = await storage.save_embeddings([], [])
            assert success is True
            
            # Should not have called upsert
            mock_client.upsert.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_mismatched_chunks_embeddings_compatibility(self):
        """Test error handling for mismatched chunks/embeddings"""
        mock_client = MagicMock()
        
        with patch('core.qdrant_storage.QdrantClient', return_value=mock_client):
            mock_collections = Mock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            
            from core.qdrant_storage import QdrantStorage
            
            storage = QdrantStorage()
            
            chunks = [Chunk(id="1", book_id="test", chunk_index=0, text="test")]
            embeddings = [[0.1] * 768, [0.2] * 768]  # 2 embeddings for 1 chunk
            
            # Should return False on mismatch (like ChromaDB would)
            success = await storage.save_embeddings(chunks, embeddings)
            assert success is False
    
    @pytest.mark.asyncio
    async def test_no_filter_compatibility(self):
        """Test search without filters works like ChromaDB"""
        mock_client = MagicMock()
        mock_client.search.return_value = []
        
        with patch('core.qdrant_storage.QdrantClient', return_value=mock_client):
            mock_collections = Mock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            
            from core.qdrant_storage import QdrantStorage
            
            storage = QdrantStorage()
            
            query_embedding = [0.5] * 768
            
            # No filter (ChromaDB allows this)
            results = await storage.search_semantic(query_embedding)
            
            assert isinstance(results, list)
            
            # Should have called search without filter
            search_call = mock_client.search.call_args
            query_filter = search_call[1]['query_filter']
            
            assert query_filter is None
    
    @pytest.mark.asyncio
    async def test_empty_delete_compatibility(self):
        """Test deleting empty list works like ChromaDB"""
        mock_client = MagicMock()
        
        with patch('core.qdrant_storage.QdrantClient', return_value=mock_client):
            mock_collections = Mock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            
            from core.qdrant_storage import QdrantStorage
            
            storage = QdrantStorage()
            
            # Delete empty list (ChromaDB allows this)
            success = await storage.delete_embeddings([])
            assert success is True
            
            # Should not have called delete
            mock_client.delete.assert_not_called()


class TestBatchProcessingCompatibility:
    """Test that batch processing works like ChromaDB"""
    
    @pytest.mark.asyncio
    async def test_large_batch_compatibility(self):
        """Test that large batches are handled like ChromaDB batching"""
        mock_client = MagicMock()
        
        # Mock successful upsert for all batches
        mock_operation_info = Mock()
        mock_operation_info.status.name = "COMPLETED"
        mock_client.upsert.return_value = mock_operation_info
        
        with patch('core.qdrant_storage.QdrantClient', return_value=mock_client):
            mock_collections = Mock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            
            from core.qdrant_storage import QdrantStorage
            
            storage = QdrantStorage()
            
            # Create 150 chunks (triggers batching like ChromaDB)
            chunks = [
                Chunk(id=f"chunk_{i}", book_id="test", chunk_index=i, text=f"chunk {i}")
                for i in range(150)
            ]
            embeddings = [[0.1] * 768 for _ in range(150)]
            
            success = await storage.save_embeddings(chunks, embeddings)
            
            # Should succeed like ChromaDB batch processing
            assert success is True
            
            # Should have batched the calls (QdrantStorage uses 100 batch size)
            assert mock_client.upsert.call_count >= 2


if __name__ == "__main__":
    pytest.main([__file__])