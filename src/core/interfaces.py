"""
Storage interfaces for TradeKnowledge

These abstract base classes define the contracts that our storage
implementations must follow. This allows us to swap implementations
without changing the rest of the code.
"""

from abc import ABC, abstractmethod
from typing import Any

from .models import Book, Chunk


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
    async def get_book(self, book_id: str) -> Book | None:
        """Retrieve a book by ID"""
        pass

    @abstractmethod
    async def get_book_by_hash(self, file_hash: str) -> Book | None:
        """Retrieve a book by file hash (for deduplication)"""
        pass

    @abstractmethod
    async def list_books(
        self, category: str | None = None, limit: int = 100, offset: int = 0
    ) -> list[Book]:
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
    async def save_chunks(self, chunks: list[Chunk]) -> bool:
        """Save multiple chunks efficiently"""
        pass

    @abstractmethod
    async def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Retrieve a single chunk"""
        pass

    @abstractmethod
    async def get_chunks_by_book(self, book_id: str) -> list[Chunk]:
        """Get all chunks for a book"""
        pass

    @abstractmethod
    async def get_chunk_context(
        self, chunk_id: str, before: int = 1, after: int = 1
    ) -> dict[str, Any]:
        """Get a chunk with surrounding context"""
        pass

    @abstractmethod
    async def search_exact(
        self, query: str, book_ids: list[str] | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
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
    async def save_embeddings(
        self, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> bool:
        """Save chunk embeddings"""
        pass

    @abstractmethod
    async def search_semantic(
        self,
        query_embedding: list[float],
        filter_dict: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Perform semantic search"""
        pass

    @abstractmethod
    async def delete_embeddings(self, chunk_ids: list[str]) -> bool:
        """Delete embeddings by chunk IDs"""
        pass

    @abstractmethod
    async def get_collection_stats(self) -> dict[str, Any]:
        """Get statistics about the vector collection"""
        pass


class CacheInterface(ABC):
    """
    Interface for caching frequently accessed data.

    This improves performance by storing recent search results
    and frequently accessed chunks.
    """

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Get value from cache"""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
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
