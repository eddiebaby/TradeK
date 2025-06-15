"""
Test that both QdrantStorage and ChromaDBStorage implement VectorStorageInterface correctly
and behave identically for the core interface methods.
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
from core.interfaces import VectorStorageInterface


class TestVectorStorageInterface:
    """Test that both storage implementations follow the interface correctly"""
    
    def test_qdrant_implements_interface(self):
        """Test that QdrantStorage implements VectorStorageInterface"""
        from core.qdrant_storage import QdrantStorage
        
        # Should be a subclass of VectorStorageInterface
        assert issubclass(QdrantStorage, VectorStorageInterface)
        
        # Create instance to verify interface methods
        with patch('core.qdrant_storage.QdrantClient'):
            mock_collections = Mock()
            mock_collections.collections = []
            
            with patch('core.qdrant_storage.QdrantClient') as mock_client_class:
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client
                mock_client.get_collections.return_value = mock_collections
                
                storage = QdrantStorage()
                
                # Verify all interface methods exist
                assert hasattr(storage, 'save_embeddings')
                assert hasattr(storage, 'search_semantic')
                assert hasattr(storage, 'delete_embeddings')
                assert hasattr(storage, 'get_collection_stats')
                
                # Verify methods are callable
                assert callable(storage.save_embeddings)
                assert callable(storage.search_semantic)
                assert callable(storage.delete_embeddings)
                assert callable(storage.get_collection_stats)
    
    def test_chromadb_implements_interface(self):
        """Test that ChromaDBStorage implements VectorStorageInterface"""
        from core.chroma_storage import ChromaDBStorage
        
        # Should be a subclass of VectorStorageInterface
        assert issubclass(ChromaDBStorage, VectorStorageInterface)
        
        # Create instance to verify interface methods
        with patch('core.chroma_storage.chromadb.PersistentClient'):
            storage = ChromaDBStorage()
            
            # Verify all interface methods exist
            assert hasattr(storage, 'save_embeddings')
            assert hasattr(storage, 'search_semantic')
            assert hasattr(storage, 'delete_embeddings')
            assert hasattr(storage, 'get_collection_stats')
            
            # Verify methods are callable
            assert callable(storage.save_embeddings)
            assert callable(storage.search_semantic)
            assert callable(storage.delete_embeddings)
            assert callable(storage.get_collection_stats)
    
    def test_qdrant_chromadb_alias_works(self):
        """Test that QdrantStorage can be used via ChromaDBStorage alias"""
        from core.qdrant_storage import ChromaDBStorage  # This should be the alias
        
        with patch('core.qdrant_storage.QdrantClient'):
            mock_collections = Mock()
            mock_collections.collections = []
            
            with patch('core.qdrant_storage.QdrantClient') as mock_client_class:
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client
                mock_client.get_collections.return_value = mock_collections
                
                # Should be able to create via alias
                storage = ChromaDBStorage()
                
                # Should actually be QdrantStorage instance
                assert storage.__class__.__name__ == 'QdrantStorage'
                
                # Should implement the interface
                assert isinstance(storage, VectorStorageInterface)


class TestInterfaceMethodSignatures:
    """Test that method signatures match the interface exactly"""
    
    @pytest.mark.asyncio
    async def test_save_embeddings_signature_compatibility(self):
        """Test save_embeddings method signature is identical"""
        from core.qdrant_storage import QdrantStorage
        from core.chroma_storage import ChromaDBStorage
        import inspect
        
        # Get method signatures
        qdrant_sig = inspect.signature(QdrantStorage.save_embeddings)
        chroma_sig = inspect.signature(ChromaDBStorage.save_embeddings)
        
        # Should have same parameter names and types
        assert list(qdrant_sig.parameters.keys()) == list(chroma_sig.parameters.keys())
        
        # Both should accept chunks and embeddings parameters
        assert 'chunks' in qdrant_sig.parameters
        assert 'embeddings' in qdrant_sig.parameters
        assert 'chunks' in chroma_sig.parameters
        assert 'embeddings' in chroma_sig.parameters
    
    @pytest.mark.asyncio
    async def test_search_semantic_signature_compatibility(self):
        """Test search_semantic method signature is identical"""
        from core.qdrant_storage import QdrantStorage
        from core.chroma_storage import ChromaDBStorage
        import inspect
        
        # Get method signatures
        qdrant_sig = inspect.signature(QdrantStorage.search_semantic)
        chroma_sig = inspect.signature(ChromaDBStorage.search_semantic)
        
        # Should have same parameter names
        assert list(qdrant_sig.parameters.keys()) == list(chroma_sig.parameters.keys())
        
        # Both should accept query_embedding, filter_dict, and limit
        assert 'query_embedding' in qdrant_sig.parameters
        assert 'filter_dict' in qdrant_sig.parameters
        assert 'limit' in qdrant_sig.parameters
    
    @pytest.mark.asyncio
    async def test_delete_embeddings_signature_compatibility(self):
        """Test delete_embeddings method signature is identical"""
        from core.qdrant_storage import QdrantStorage
        from core.chroma_storage import ChromaDBStorage
        import inspect
        
        # Get method signatures
        qdrant_sig = inspect.signature(QdrantStorage.delete_embeddings)
        chroma_sig = inspect.signature(ChromaDBStorage.delete_embeddings)
        
        # Should have same parameter names
        assert list(qdrant_sig.parameters.keys()) == list(chroma_sig.parameters.keys())
        
        # Both should accept chunk_ids parameter
        assert 'chunk_ids' in qdrant_sig.parameters
    
    @pytest.mark.asyncio
    async def test_get_collection_stats_signature_compatibility(self):
        """Test get_collection_stats method signature is identical"""
        from core.qdrant_storage import QdrantStorage
        from core.chroma_storage import ChromaDBStorage
        import inspect
        
        # Get method signatures
        qdrant_sig = inspect.signature(QdrantStorage.get_collection_stats)
        chroma_sig = inspect.signature(ChromaDBStorage.get_collection_stats)
        
        # Should have same parameter names (likely just self)
        assert list(qdrant_sig.parameters.keys()) == list(chroma_sig.parameters.keys())


class TestReturnValueCompatibility:
    """Test that return values have compatible formats"""
    
    @pytest.mark.asyncio
    async def test_save_embeddings_return_format(self):
        """Test that both implementations return bool for save_embeddings"""
        # Test QdrantStorage
        mock_client = MagicMock()
        mock_operation_info = Mock()
        mock_operation_info.status.name = "COMPLETED"
        mock_client.upsert.return_value = mock_operation_info
        
        with patch('core.qdrant_storage.QdrantClient', return_value=mock_client):
            mock_collections = Mock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            
            from core.qdrant_storage import QdrantStorage
            storage = QdrantStorage()
            
            result = await storage.save_embeddings([], [])
            assert isinstance(result, bool)
        
        # Test ChromaDBStorage
        with patch('core.chroma_storage.chromadb.PersistentClient'):
            from core.chroma_storage import ChromaDBStorage
            storage = ChromaDBStorage()
            
            result = await storage.save_embeddings([], [])
            assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_search_semantic_return_format(self):
        """Test that both implementations return List[Dict[str, Any]]"""
        # Test QdrantStorage
        mock_client = MagicMock()
        mock_client.search.return_value = []
        
        with patch('core.qdrant_storage.QdrantClient', return_value=mock_client):
            mock_collections = Mock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            
            from core.qdrant_storage import QdrantStorage
            storage = QdrantStorage()
            
            result = await storage.search_semantic([0.1] * 768)
            assert isinstance(result, list)
        
        # Test ChromaDBStorage  
        with patch('core.chroma_storage.chromadb.PersistentClient'):
            with patch('core.chroma_storage.asyncio.to_thread') as mock_to_thread:
                mock_to_thread.return_value = {
                    'ids': [[]],
                    'documents': [[]],
                    'metadatas': [[]],
                    'distances': [[]]
                }
                
                from core.chroma_storage import ChromaDBStorage
                storage = ChromaDBStorage()
                
                result = await storage.search_semantic([0.1] * 768)
                assert isinstance(result, list)
    
    @pytest.mark.asyncio
    async def test_delete_embeddings_return_format(self):
        """Test that both implementations return bool for delete_embeddings"""
        # Test QdrantStorage
        mock_client = MagicMock()
        
        with patch('core.qdrant_storage.QdrantClient', return_value=mock_client):
            mock_collections = Mock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            
            from core.qdrant_storage import QdrantStorage
            storage = QdrantStorage()
            
            result = await storage.delete_embeddings([])
            assert isinstance(result, bool)
        
        # Test ChromaDBStorage
        with patch('core.chroma_storage.chromadb.PersistentClient'):
            from core.chroma_storage import ChromaDBStorage
            storage = ChromaDBStorage()
            
            result = await storage.delete_embeddings([])
            assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_get_collection_stats_return_format(self):
        """Test that both implementations return Dict[str, Any]"""
        # Test QdrantStorage
        mock_client = MagicMock()
        mock_collection_info = Mock()
        mock_collection_info.points_count = 100
        mock_collection_info.config.params.vectors.size = 768
        mock_distance = Mock()
        mock_distance.name = "COSINE"
        mock_collection_info.config.params.vectors.distance = mock_distance
        mock_collection_info.segments_count = 1
        mock_collection_info.status = "green"
        mock_client.get_collection.return_value = mock_collection_info
        
        with patch('core.qdrant_storage.QdrantClient', return_value=mock_client):
            mock_collections = Mock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            
            from core.qdrant_storage import QdrantStorage
            storage = QdrantStorage()
            
            result = await storage.get_collection_stats()
            assert isinstance(result, dict)
            assert 'collection_name' in result
            assert 'total_embeddings' in result
        
        # Test ChromaDBStorage
        with patch('core.chroma_storage.chromadb.PersistentClient'):
            with patch('core.chroma_storage.asyncio.to_thread') as mock_to_thread:
                mock_to_thread.return_value = 100  # Mock collection.count()
                
                from core.chroma_storage import ChromaDBStorage
                storage = ChromaDBStorage()
                
                result = await storage.get_collection_stats()
                assert isinstance(result, dict)
                assert 'collection_name' in result
                assert 'total_embeddings' in result


if __name__ == "__main__":
    pytest.main([__file__])