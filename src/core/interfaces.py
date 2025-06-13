"""
Storage interfaces for TradeKnowledge

These abstract base classes define the contracts that our storage
implementations must follow. This allows us to swap implementations
without changing the rest of the code.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime

from core.models import Book, Chunk, SearchResult, SearchResponse

class BookStorageInterface(ABC):
    """
    Interface for book metadata storage.
    
    Any class that implements this interface can be used
    to store and retrieve book information.
    """
    
    @abstractmethod
    async def save_book(self, book: Book) -> bool:
        """Save a book's metadata"""
        pass
    
    @abstractmethod
    async def get_book(self, book_id: str) -> Optional[Book]:
        """Retrieve a book by ID"""
        pass
    
    @abstractmethod
    async def get_book_by_hash(self, file_hash: str) -> Optional[Book]:
        """Retrieve a book by file hash (for deduplication)"""
        pass
    
    @abstractmethod
    async def list_books(self, 
                        category: Optional[str] = None,
                        limit: int = 100,
                        offset: int = 0) -> List[Book]:
        """List books with optional filtering"""
        pass
    
    @abstractmethod
    async def update_book(self, book: Book) -> bool:
        """Update book metadata"""
        pass
    
    @abstractmethod
    async def delete_book(self, book_id: str) -> bool:
        """Delete a book and all its chunks"""
        pass

class ChunkStorageInterface(ABC):
    """
    Interface for chunk storage.
    
    This handles both the full text storage (for exact search)
    and metadata about chunks.
    """
    
    @abstractmethod
    async def save_chunks(self, chunks: List[Chunk]) -> bool:
        """Save multiple chunks efficiently"""
        pass
    
    @abstractmethod
    async def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Retrieve a single chunk"""
        pass
    
    @abstractmethod
    async def get_chunks_by_book(self, book_id: str) -> List[Chunk]:
        """Get all chunks for a book"""
        pass
    
    @abstractmethod
    async def get_chunk_context(self, 
                               chunk_id: str,
                               before: int = 1,
                               after: int = 1) -> Dict[str, Any]:
        """Get a chunk with surrounding context"""
        pass
    
    @abstractmethod
    async def search_exact(self,
                          query: str,
                          book_ids: Optional[List[str]] = None,
                          limit: int = 10) -> List[Dict[str, Any]]:
        """Perform exact text search"""
        pass
    
    @abstractmethod
    async def delete_chunks_by_book(self, book_id: str) -> bool:
        """Delete all chunks for a book"""
        pass

class VectorStorageInterface(ABC):
    """
    Interface for vector/embedding storage.
    
    This handles semantic search capabilities using
    vector embeddings.
    """
    
    @abstractmethod
    async def save_embeddings(self, 
                             chunks: List[Chunk],
                             embeddings: List[List[float]]) -> bool:
        """Save chunk embeddings"""
        pass
    
    @abstractmethod
    async def search_semantic(self,
                             query_embedding: List[float],
                             filter_dict: Optional[Dict[str, Any]] = None,
                             limit: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic search"""
        pass
    
    @abstractmethod
    async def delete_embeddings(self, chunk_ids: List[str]) -> bool:
        """Delete embeddings by chunk IDs"""
        pass
    
    @abstractmethod
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection"""
        pass

class CacheInterface(ABC):
    """
    Interface for caching frequently accessed data.
    
    This improves performance by storing recent search results
    and frequently accessed chunks.
    """
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete from cache"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear entire cache"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass