"""
Tests for QdrantStorage class for vector storage and semantic search
"""

import pytest
import asyncio
import uuid
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime
from typing import List, Dict, Any

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from core.models import Chunk, ChunkType
from core.config import QdrantConfig


class TestQdrantStorageInterface:
    """Test that QdrantStorage implements the required VectorStorageInterface"""
    
    def test_required_methods_exist(self):
        """Test that all required methods exist in the interface"""
        from core.qdrant_storage import QdrantStorage
        
        storage = QdrantStorage()
        
        # Check required methods exist
        assert hasattr(storage, 'save_embeddings')
        assert hasattr(storage, 'search_semantic')
        assert hasattr(storage, 'delete_embeddings')
        assert hasattr(storage, 'get_collection_stats')
        
        # Check methods are callable
        assert callable(storage.save_embeddings)
        assert callable(storage.search_semantic)
        assert callable(storage.delete_embeddings)
        assert callable(storage.get_collection_stats)
        
        # Check required properties/attributes
        assert hasattr(storage, 'collection_name')
        assert hasattr(storage, 'client')
        assert hasattr(storage, 'vector_size')
        assert hasattr(storage, 'distance_metric')


class TestQdrantStorageInitialization:
    """Test QdrantStorage initialization"""
    
    @patch('core.qdrant_storage.QdrantClient')
    def test_default_initialization(self, mock_qdrant_client):
        """Test default initialization uses config values"""
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client
        
        # Mock collections response
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        
        from core.qdrant_storage import QdrantStorage
        
        storage = QdrantStorage()
        
        assert storage.collection_name == "tradeknowledge"
        assert storage.vector_size == 768
        assert storage.distance_metric.name == "COSINE"
        
        # Verify client was initialized
        mock_qdrant_client.assert_called_once()
    
    @patch('core.qdrant_storage.QdrantClient')
    def test_custom_collection_name(self, mock_qdrant_client):
        """Test initialization with custom collection name"""
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client
        
        # Mock collections response
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        
        from core.qdrant_storage import QdrantStorage
        
        storage = QdrantStorage(collection_name="custom_collection")
        
        assert storage.collection_name == "custom_collection"
    
    @patch('core.qdrant_storage.QdrantClient')
    def test_config_integration(self, mock_qdrant_client):
        """Test that storage uses QdrantConfig properly"""
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client
        
        # Mock collections response
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        
        config = QdrantConfig(
            host="custom-host",
            port=6334,
            collection_name="test_collection",
            api_key="test-key",
            https=True
        )
        
        with patch('core.qdrant_storage.get_config') as mock_get_config:
            mock_get_config.return_value = Mock(
                database=Mock(qdrant=config),
                embedding=Mock(dimension=512)
            )
            
            from core.qdrant_storage import QdrantStorage
            
            storage = QdrantStorage()
            
            assert storage.collection_name == "test_collection"
            assert storage.vector_size == 512
            
            # Verify client was initialized with correct parameters
            mock_qdrant_client.assert_called_once_with(
                host="custom-host",
                port=6334,
                api_key="test-key",
                https=True,
                prefer_grpc=False
            )


class TestQdrantCollectionManagement:
    """Test Qdrant collection creation and management"""
    
    @patch('core.qdrant_storage.QdrantClient')
    def test_collection_creation_new(self, mock_qdrant_client):
        """Test creating new collection when it doesn't exist"""
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client
        
        # Mock collections response - empty list (no existing collections)
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        
        from core.qdrant_storage import QdrantStorage
        
        storage = QdrantStorage()
        
        # Verify collection was created
        mock_client.create_collection.assert_called_once()
        create_call = mock_client.create_collection.call_args
        assert create_call[1]['collection_name'] == "tradeknowledge"
        assert create_call[1]['vectors_config'].size == 768
    
    @patch('core.qdrant_storage.QdrantClient')
    def test_collection_exists_compatible(self, mock_qdrant_client):
        """Test using existing collection with compatible configuration"""
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client
        
        # Mock existing collection
        mock_collection = Mock()
        mock_collection.name = "tradeknowledge"
        mock_collections = Mock()
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections
        
        # Mock collection info
        mock_collection_info = Mock()
        mock_collection_info.config.params.vectors.size = 768
        mock_client.get_collection.return_value = mock_collection_info
        
        from core.qdrant_storage import QdrantStorage
        
        storage = QdrantStorage()
        
        # Verify collection was not created (used existing)
        mock_client.create_collection.assert_not_called()
        mock_client.get_collection.assert_called_once_with("tradeknowledge")
    
    @patch('core.qdrant_storage.QdrantClient')
    def test_collection_exists_incompatible_dimension(self, mock_qdrant_client):
        """Test warning when existing collection has incompatible dimension"""
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client
        
        # Mock existing collection
        mock_collection = Mock()
        mock_collection.name = "tradeknowledge"
        mock_collections = Mock()
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections
        
        # Mock collection info with wrong dimension
        mock_collection_info = Mock()
        mock_collection_info.config.params.vectors.size = 1536  # Wrong dimension
        mock_client.get_collection.return_value = mock_collection_info
        
        with patch('core.qdrant_storage.logger') as mock_logger:
            from core.qdrant_storage import QdrantStorage
            
            storage = QdrantStorage()
            
            # Should log warning about dimension mismatch
            mock_logger.warning.assert_called()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "dimension mismatch" in warning_call.lower()
    
    @patch('core.qdrant_storage.QdrantClient')
    def test_collection_creation_error(self, mock_qdrant_client):
        """Test handling of collection creation errors"""
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client
        
        # Mock error during collection operations
        mock_client.get_collections.side_effect = Exception("Connection error")
        
        from core.qdrant_storage import QdrantStorage
        
        with pytest.raises(Exception):
            QdrantStorage()


class TestEmbeddingStorage:
    """Test embedding storage functionality"""
    
    @pytest.mark.asyncio
    async def test_save_embeddings_success(self):
        """Test successful embedding storage"""
        mock_client = MagicMock()
        
        # Mock successful upsert
        mock_operation_info = Mock()
        mock_operation_info.status.name = "COMPLETED"
        mock_client.upsert.return_value = mock_operation_info
        
        with patch('core.qdrant_storage.QdrantClient', return_value=mock_client):
            # Mock collection setup
            mock_collections = Mock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            
            from core.qdrant_storage import QdrantStorage
            
            storage = QdrantStorage()
            
            # Create test chunks
            chunks = [
                Chunk(
                    id="test_chunk_1",
                    book_id="test_book",
                    chunk_index=0,
                    text="First test chunk",
                    chapter="Chapter 1",
                    page_start=1
                ),
                Chunk(
                    id="test_chunk_2", 
                    book_id="test_book",
                    chunk_index=1,
                    text="Second test chunk",
                    section="Introduction"
                )
            ]
            
            embeddings = [
                [0.1] * 768,
                [0.2] * 768
            ]
            
            success = await storage.save_embeddings(chunks, embeddings)
            
            assert success is True
            
            # Verify upsert was called
            mock_client.upsert.assert_called()
            upsert_call = mock_client.upsert.call_args
            
            # Check collection name
            assert upsert_call[1]['collection_name'] == "tradeknowledge"
            assert upsert_call[1]['wait'] is True
            
            # Check points structure
            points = upsert_call[1]['points']
            assert len(points) == 2
            
            # Check first point
            point1 = points[0]
            assert point1.vector == [0.1] * 768
            assert point1.payload['chunk_id'] == "test_chunk_1"
            assert point1.payload['book_id'] == "test_book"
            assert point1.payload['text'] == "First test chunk"
            assert point1.payload['chapter'] == "Chapter 1"
            assert point1.payload['page_start'] == 1
            
            # Check second point
            point2 = points[1]
            assert point2.payload['section'] == "Introduction"
    
    @pytest.mark.asyncio
    async def test_save_embeddings_empty_input(self):
        """Test saving empty chunks and embeddings"""
        mock_client = MagicMock()
        
        with patch('core.qdrant_storage.QdrantClient', return_value=mock_client):
            mock_collections = Mock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            
            from core.qdrant_storage import QdrantStorage
            
            storage = QdrantStorage()
            
            # Test empty chunks
            success = await storage.save_embeddings([], [])
            assert success is True
            
            # Verify no upsert was called
            mock_client.upsert.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_save_embeddings_mismatch_length(self):
        """Test error handling for mismatched chunks and embeddings length"""
        mock_client = MagicMock()
        
        with patch('core.qdrant_storage.QdrantClient', return_value=mock_client):
            mock_collections = Mock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            
            from core.qdrant_storage import QdrantStorage
            
            storage = QdrantStorage()
            
            chunks = [Chunk(id="1", book_id="test", chunk_index=0, text="test")]
            embeddings = [[0.1] * 768, [0.2] * 768]  # Mismatch: 1 chunk, 2 embeddings
            
            with patch('core.qdrant_storage.logger') as mock_logger:
                success = await storage.save_embeddings(chunks, embeddings)
                
                assert success is False
                mock_logger.error.assert_called()
                error_call = mock_logger.error.call_args[0][0]
                assert "Mismatch" in error_call
    
    @pytest.mark.asyncio
    async def test_save_embeddings_batch_processing(self):
        """Test batch processing for large numbers of embeddings"""
        mock_client = MagicMock()
        
        # Mock successful upsert
        mock_operation_info = Mock()
        mock_operation_info.status.name = "COMPLETED"
        mock_client.upsert.return_value = mock_operation_info
        
        with patch('core.qdrant_storage.QdrantClient', return_value=mock_client):
            mock_collections = Mock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            
            from core.qdrant_storage import QdrantStorage
            
            storage = QdrantStorage()
            
            # Create 150 chunks (should trigger batching at 100)
            chunks = [
                Chunk(id=f"chunk_{i}", book_id="test", chunk_index=i, text=f"chunk {i}")
                for i in range(150)
            ]
            embeddings = [[0.1] * 768 for _ in range(150)]
            
            success = await storage.save_embeddings(chunks, embeddings)
            
            assert success is True
            
            # Should have made 2 upsert calls (batch size 100)
            assert mock_client.upsert.call_count == 2
            
            # Check first batch has 100 items
            first_call = mock_client.upsert.call_args_list[0]
            first_points = first_call[1]['points']
            assert len(first_points) == 100
            
            # Check second batch has 50 items
            second_call = mock_client.upsert.call_args_list[1]
            second_points = second_call[1]['points']
            assert len(second_points) == 50
    
    @pytest.mark.asyncio
    async def test_save_embeddings_upsert_failure(self):
        """Test handling of upsert operation failure"""
        mock_client = MagicMock()
        
        # Mock failed upsert
        mock_operation_info = Mock()
        mock_operation_info.status.name = "FAILED"
        mock_client.upsert.return_value = mock_operation_info
        
        with patch('core.qdrant_storage.QdrantClient', return_value=mock_client):
            mock_collections = Mock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            
            from core.qdrant_storage import QdrantStorage
            
            storage = QdrantStorage()
            
            chunks = [Chunk(id="1", book_id="test", chunk_index=0, text="test")]
            embeddings = [[0.1] * 768]
            
            with patch('core.qdrant_storage.logger') as mock_logger:
                success = await storage.save_embeddings(chunks, embeddings)
                
                assert success is False
                mock_logger.error.assert_called()


class TestSemanticSearch:
    """Test semantic search functionality"""
    
    @pytest.mark.asyncio
    async def test_search_semantic_success(self):
        """Test successful semantic search"""
        mock_client = MagicMock()
        
        # Mock search response
        mock_search_result = [
            Mock(
                payload={
                    'chunk_id': 'chunk_1',
                    'text': 'First result text',
                    'book_id': 'book_1',
                    'chunk_index': 0
                },
                score=0.95
            ),
            Mock(
                payload={
                    'chunk_id': 'chunk_2',
                    'text': 'Second result text',
                    'book_id': 'book_1',
                    'chunk_index': 1
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
            results = await storage.search_semantic(query_embedding, limit=5)
            
            assert len(results) == 2
            
            # Check first result
            assert results[0]['chunk_id'] == 'chunk_1'
            assert results[0]['text'] == 'First result text'
            assert results[0]['score'] == 0.95
            assert results[0]['distance'] == 0.05  # 1 - score
            assert 'book_id' in results[0]['metadata']
            
            # Check second result
            assert results[1]['score'] == 0.87
            
            # Verify search was called correctly
            mock_client.search.assert_called_once()
            search_call = mock_client.search.call_args
            assert search_call[1]['collection_name'] == "tradeknowledge"
            assert search_call[1]['query_vector'] == query_embedding
            assert search_call[1]['limit'] == 5
            assert search_call[1]['with_payload'] is True
            assert search_call[1]['with_vectors'] is False
    
    @pytest.mark.asyncio
    async def test_search_semantic_with_book_filter(self):
        """Test semantic search with book ID filtering"""
        mock_client = MagicMock()
        mock_client.search.return_value = []
        
        with patch('core.qdrant_storage.QdrantClient', return_value=mock_client):
            mock_collections = Mock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            
            from core.qdrant_storage import QdrantStorage
            
            storage = QdrantStorage()
            
            query_embedding = [0.5] * 768
            filter_dict = {'book_ids': ['book_1', 'book_2']}
            
            await storage.search_semantic(query_embedding, filter_dict=filter_dict, limit=10)
            
            # Verify filter was applied
            search_call = mock_client.search.call_args
            query_filter = search_call[1]['query_filter']
            
            assert query_filter is not None
            # Check that filter has the correct structure for book_id filtering
            assert len(query_filter.must) == 1
            condition = query_filter.must[0]
            assert condition.key == "book_id"
    
    @pytest.mark.asyncio
    async def test_search_semantic_with_chunk_type_filter(self):
        """Test semantic search with chunk type filtering"""
        mock_client = MagicMock()
        mock_client.search.return_value = []
        
        with patch('core.qdrant_storage.QdrantClient', return_value=mock_client):
            mock_collections = Mock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            
            from core.qdrant_storage import QdrantStorage
            
            storage = QdrantStorage()
            
            query_embedding = [0.5] * 768
            filter_dict = {'chunk_type': 'text'}
            
            await storage.search_semantic(query_embedding, filter_dict=filter_dict)
            
            # Verify filter was applied
            search_call = mock_client.search.call_args
            query_filter = search_call[1]['query_filter']
            
            assert query_filter is not None
            condition = query_filter.must[0]
            assert condition.key == "chunk_type"
    
    @pytest.mark.asyncio
    async def test_search_semantic_multiple_filters(self):
        """Test semantic search with multiple filters"""
        mock_client = MagicMock()
        mock_client.search.return_value = []
        
        with patch('core.qdrant_storage.QdrantClient', return_value=mock_client):
            mock_collections = Mock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            
            from core.qdrant_storage import QdrantStorage
            
            storage = QdrantStorage()
            
            query_embedding = [0.5] * 768
            filter_dict = {
                'book_ids': ['book_1'],
                'chunk_type': 'text'
            }
            
            await storage.search_semantic(query_embedding, filter_dict=filter_dict)
            
            # Verify both filters were applied
            search_call = mock_client.search.call_args
            query_filter = search_call[1]['query_filter']
            
            assert query_filter is not None
            assert len(query_filter.must) == 2
    
    @pytest.mark.asyncio
    async def test_search_semantic_no_filter(self):
        """Test semantic search without filters"""
        mock_client = MagicMock()
        mock_client.search.return_value = []
        
        with patch('core.qdrant_storage.QdrantClient', return_value=mock_client):
            mock_collections = Mock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            
            from core.qdrant_storage import QdrantStorage
            
            storage = QdrantStorage()
            
            query_embedding = [0.5] * 768
            
            await storage.search_semantic(query_embedding)
            
            # Verify no filter was applied
            search_call = mock_client.search.call_args
            query_filter = search_call[1]['query_filter']
            
            assert query_filter is None
    
    @pytest.mark.asyncio
    async def test_search_semantic_error_handling(self):
        """Test error handling during semantic search"""
        mock_client = MagicMock()
        mock_client.search.side_effect = Exception("Search error")
        
        with patch('core.qdrant_storage.QdrantClient', return_value=mock_client):
            mock_collections = Mock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            
            with patch('core.qdrant_storage.logger') as mock_logger:
                from core.qdrant_storage import QdrantStorage
                
                storage = QdrantStorage()
                
                query_embedding = [0.5] * 768
                results = await storage.search_semantic(query_embedding)
                
                # Should return empty list on error
                assert results == []
                
                # Should log error
                mock_logger.error.assert_called()


class TestEmbeddingDeletion:
    """Test embedding deletion functionality"""
    
    @pytest.mark.asyncio
    async def test_delete_embeddings_success(self):
        """Test successful embedding deletion"""
        mock_client = MagicMock()
        
        with patch('core.qdrant_storage.QdrantClient', return_value=mock_client):
            mock_collections = Mock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            
            from core.qdrant_storage import QdrantStorage
            
            storage = QdrantStorage()
            
            chunk_ids = ["chunk_1", "chunk_2", "chunk_3"]
            success = await storage.delete_embeddings(chunk_ids)
            
            assert success is True
            
            # Verify delete was called
            mock_client.delete.assert_called_once()
            delete_call = mock_client.delete.call_args
            
            assert delete_call[1]['collection_name'] == "tradeknowledge"
            
            # Check filter structure
            points_selector = delete_call[1]['points_selector']
            assert len(points_selector.must) == 1
            condition = points_selector.must[0]
            assert condition.key == "chunk_id"
    
    @pytest.mark.asyncio
    async def test_delete_embeddings_empty_list(self):
        """Test deleting empty chunk ID list"""
        mock_client = MagicMock()
        
        with patch('core.qdrant_storage.QdrantClient', return_value=mock_client):
            mock_collections = Mock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            
            from core.qdrant_storage import QdrantStorage
            
            storage = QdrantStorage()
            
            success = await storage.delete_embeddings([])
            
            assert success is True
            
            # Verify delete was not called
            mock_client.delete.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_delete_embeddings_error(self):
        """Test error handling during deletion"""
        mock_client = MagicMock()
        mock_client.delete.side_effect = Exception("Delete error")
        
        with patch('core.qdrant_storage.QdrantClient', return_value=mock_client):
            mock_collections = Mock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            
            with patch('core.qdrant_storage.logger') as mock_logger:
                from core.qdrant_storage import QdrantStorage
                
                storage = QdrantStorage()
                
                success = await storage.delete_embeddings(["chunk_1"])
                
                assert success is False
                mock_logger.error.assert_called()


class TestCollectionStats:
    """Test collection statistics functionality"""
    
    @pytest.mark.asyncio
    async def test_get_collection_stats_success(self):
        """Test successful collection stats retrieval"""
        mock_client = MagicMock()
        
        # Mock collection info
        mock_collection_info = Mock()
        mock_collection_info.points_count = 1500
        mock_collection_info.config.params.vectors.size = 768
        mock_collection_info.config.params.vectors.distance.name = "COSINE"
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
            
            expected_stats = {
                'collection_name': 'tradeknowledge',
                'total_embeddings': 1500,
                'vector_size': 768,
                'distance_metric': 'COSINE',
                'segments_count': 3,
                'status': 'green'
            }
            
            assert stats == expected_stats
            
            # Verify get_collection was called
            mock_client.get_collection.assert_called_once_with("tradeknowledge")
    
    @pytest.mark.asyncio
    async def test_get_collection_stats_error(self):
        """Test error handling during stats retrieval"""
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("Stats error")
        
        with patch('core.qdrant_storage.QdrantClient', return_value=mock_client):
            mock_collections = Mock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            
            with patch('core.qdrant_storage.logger') as mock_logger:
                from core.qdrant_storage import QdrantStorage
                
                storage = QdrantStorage()
                
                stats = await storage.get_collection_stats()
                
                # Should return error info
                assert 'collection_name' in stats
                assert 'error' in stats
                assert stats['collection_name'] == 'tradeknowledge'
                
                # Should log error
                mock_logger.error.assert_called()


class TestQdrantUtilities:
    """Test utility functions and methods"""
    
    @patch('core.qdrant_storage.QdrantClient')
    def test_create_snapshot_success(self, mock_qdrant_client):
        """Test successful snapshot creation"""
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client
        
        # Mock collections and snapshot
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        
        mock_snapshot_info = Mock()
        mock_snapshot_info.name = "snapshot_123"
        mock_client.create_snapshot.return_value = mock_snapshot_info
        
        from core.qdrant_storage import QdrantStorage
        
        storage = QdrantStorage()
        
        snapshot_name = storage.create_snapshot()
        
        assert snapshot_name == "snapshot_123"
        
        # Verify create_snapshot was called
        mock_client.create_snapshot.assert_called_once_with(
            collection_name="tradeknowledge"
        )
    
    @patch('core.qdrant_storage.QdrantClient')
    def test_create_snapshot_error(self, mock_qdrant_client):
        """Test error handling during snapshot creation"""
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client
        
        # Mock collections
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        
        mock_client.create_snapshot.side_effect = Exception("Snapshot error")
        
        with patch('core.qdrant_storage.logger') as mock_logger:
            from core.qdrant_storage import QdrantStorage
            
            storage = QdrantStorage()
            
            snapshot_name = storage.create_snapshot()
            
            assert snapshot_name == ""
            mock_logger.error.assert_called()


class TestBackwardCompatibility:
    """Test backward compatibility features"""
    
    @patch('core.qdrant_storage.QdrantClient')
    def test_chromadb_storage_alias(self, mock_qdrant_client):
        """Test that ChromaDBStorage alias works"""
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client
        
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        
        from core.qdrant_storage import ChromaDBStorage
        
        # Should be able to create via alias
        storage = ChromaDBStorage()
        
        # Should be QdrantStorage instance
        assert storage.__class__.__name__ == 'QdrantStorage'
        assert hasattr(storage, 'collection_name')


if __name__ == "__main__":
    pytest.main([__file__])