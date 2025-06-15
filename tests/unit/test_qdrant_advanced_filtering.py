"""
Tests for advanced filtering capabilities in QdrantStorage

This tests the enhanced filtering options beyond basic book_id and chunk_type filtering,
including date ranges, numerical ranges, text matching, and complex combinations.
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


class TestAdvancedDateFiltering:
    """Test date-based filtering capabilities"""
    
    @pytest.mark.asyncio
    async def test_search_with_date_range_filter(self):
        """Test semantic search with date range filtering"""
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
                'date_range': {
                    'start': '2024-01-01',
                    'end': '2024-12-31'
                }
            }
            
            await storage.search_semantic(query_embedding, filter_dict=filter_dict)
            
            # Verify filter was applied
            search_call = mock_client.search.call_args
            query_filter = search_call[1]['query_filter']
            
            assert query_filter is not None
            # Should have date range conditions
            date_conditions = [c for c in query_filter.must if hasattr(c, 'range')]
            assert len(date_conditions) > 0
    
    @pytest.mark.asyncio
    async def test_search_with_created_after_filter(self):
        """Test filtering for chunks created after a specific date"""
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
                'created_after': '2024-06-01T00:00:00'
            }
            
            await storage.search_semantic(query_embedding, filter_dict=filter_dict)
            
            search_call = mock_client.search.call_args
            query_filter = search_call[1]['query_filter']
            
            assert query_filter is not None


class TestAdvancedPageFiltering:
    """Test page-based filtering capabilities"""
    
    @pytest.mark.asyncio
    async def test_search_with_page_range_filter(self):
        """Test semantic search with page range filtering"""
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
                'page_range': {
                    'start': 50,
                    'end': 150
                }
            }
            
            await storage.search_semantic(query_embedding, filter_dict=filter_dict)
            
            search_call = mock_client.search.call_args
            query_filter = search_call[1]['query_filter']
            
            assert query_filter is not None
    
    @pytest.mark.asyncio
    async def test_search_with_specific_page_filter(self):
        """Test filtering for content on specific pages"""
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
                'pages': [10, 25, 40]  # Specific pages
            }
            
            await storage.search_semantic(query_embedding, filter_dict=filter_dict)
            
            search_call = mock_client.search.call_args
            query_filter = search_call[1]['query_filter']
            
            assert query_filter is not None


class TestAdvancedTextFiltering:
    """Test text-based filtering capabilities"""
    
    @pytest.mark.asyncio
    async def test_search_with_chapter_filter(self):
        """Test semantic search with chapter filtering"""
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
                'chapters': ['Introduction', 'Chapter 1', 'Conclusion']
            }
            
            await storage.search_semantic(query_embedding, filter_dict=filter_dict)
            
            search_call = mock_client.search.call_args
            query_filter = search_call[1]['query_filter']
            
            assert query_filter is not None
    
    @pytest.mark.asyncio
    async def test_search_with_section_pattern_filter(self):
        """Test filtering by section pattern matching"""
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
                'section_pattern': 'Trading*'  # Wildcard pattern
            }
            
            await storage.search_semantic(query_embedding, filter_dict=filter_dict)
            
            search_call = mock_client.search.call_args
            query_filter = search_call[1]['query_filter']
            
            assert query_filter is not None


class TestCombinedAdvancedFiltering:
    """Test complex combinations of advanced filters"""
    
    @pytest.mark.asyncio
    async def test_search_with_multiple_advanced_filters(self):
        """Test semantic search with multiple advanced filters combined"""
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
                'book_ids': ['book_1', 'book_2'],
                'chunk_type': 'text',
                'page_range': {'start': 10, 'end': 100},
                'chapters': ['Chapter 1', 'Chapter 2'],
                'created_after': '2024-01-01T00:00:00'
            }
            
            await storage.search_semantic(query_embedding, filter_dict=filter_dict)
            
            search_call = mock_client.search.call_args
            query_filter = search_call[1]['query_filter']
            
            assert query_filter is not None
            # Should have multiple conditions
            assert len(query_filter.must) >= 4
    
    @pytest.mark.asyncio
    async def test_search_with_nested_filter_conditions(self):
        """Test semantic search with nested OR and AND conditions"""
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
                'complex_filter': {
                    'or': [
                        {'book_id': 'book_1', 'chapter': 'Introduction'},
                        {'book_id': 'book_2', 'page_range': {'start': 50, 'end': 100}}
                    ]
                }
            }
            
            await storage.search_semantic(query_embedding, filter_dict=filter_dict)
            
            search_call = mock_client.search.call_args
            query_filter = search_call[1]['query_filter']
            
            assert query_filter is not None
    
    @pytest.mark.asyncio
    async def test_search_with_exclusion_filters(self):
        """Test semantic search with exclusion filters (NOT conditions)"""
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
                'exclude_books': ['book_3', 'book_4'],
                'exclude_chapters': ['Appendix', 'Bibliography']
            }
            
            await storage.search_semantic(query_embedding, filter_dict=filter_dict)
            
            search_call = mock_client.search.call_args
            query_filter = search_call[1]['query_filter']
            
            assert query_filter is not None


class TestFilterValidationAndEdgeCases:
    """Test validation and edge cases for advanced filtering"""
    
    @pytest.mark.asyncio
    async def test_search_with_invalid_date_format(self):
        """Test handling of invalid date format in filters"""
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
                'created_after': 'invalid-date-format'
            }
            
            # Should handle gracefully, not crash
            results = await storage.search_semantic(query_embedding, filter_dict=filter_dict)
            
            # Should return empty results rather than crashing
            assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_search_with_empty_filter_values(self):
        """Test handling of empty filter values"""
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
                'book_ids': [],  # Empty list
                'chapters': None,  # None value
                'page_range': {}  # Empty dict
            }
            
            await storage.search_semantic(query_embedding, filter_dict=filter_dict)
            
            search_call = mock_client.search.call_args
            query_filter = search_call[1]['query_filter']
            
            # Empty filters should result in no filter being applied
            assert query_filter is None
    
    @pytest.mark.asyncio
    async def test_search_with_conflicting_filters(self):
        """Test handling of logically conflicting filters"""
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
                'page_range': {'start': 100, 'end': 50},  # Invalid range
                'chapters': ['Chapter 1'],
                'exclude_chapters': ['Chapter 1']  # Conflict: include and exclude same chapter
            }
            
            # Should handle gracefully
            results = await storage.search_semantic(query_embedding, filter_dict=filter_dict)
            
            assert isinstance(results, list)


class TestFilterPerformance:
    """Test performance aspects of advanced filtering"""
    
    @pytest.mark.asyncio
    async def test_search_with_many_filter_conditions(self):
        """Test performance with large number of filter conditions"""
        mock_client = MagicMock()
        mock_client.search.return_value = []
        
        with patch('core.qdrant_storage.QdrantClient', return_value=mock_client):
            mock_collections = Mock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            
            from core.qdrant_storage import QdrantStorage
            
            storage = QdrantStorage()
            
            query_embedding = [0.5] * 768
            
            # Create filter with many conditions
            many_book_ids = [f'book_{i}' for i in range(100)]
            many_chapters = [f'Chapter {i}' for i in range(50)]
            
            filter_dict = {
                'book_ids': many_book_ids,
                'chapters': many_chapters,
                'page_range': {'start': 1, 'end': 1000}
            }
            
            # Should complete without timeout or error
            results = await storage.search_semantic(query_embedding, filter_dict=filter_dict)
            
            assert isinstance(results, list)
            
            # Verify filter was constructed
            search_call = mock_client.search.call_args
            query_filter = search_call[1]['query_filter']
            
            assert query_filter is not None


if __name__ == "__main__":
    pytest.main([__file__])