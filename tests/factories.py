"""
Test data factories for London School TDD

These factories generate realistic test data for our domain entities
and can be used across all test types (unit, integration, acceptance).
"""

import factory
import factory.fuzzy
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid
import hashlib

from src.core.models import (
    Book, Chunk, SearchResult, SearchResponse, IngestionStatus,
    FileType, ChunkType
)


class BookFactory(factory.Factory):
    """Factory for creating Book test instances"""
    
    class Meta:
        model = Book
    
    id = factory.LazyFunction(lambda: f"book_{uuid.uuid4().hex[:8]}")
    title = factory.Faker('sentence', nb_words=4)
    author = factory.Faker('name')
    file_path = factory.LazyAttribute(
        lambda obj: f"/tmp/test_books/{obj.title.replace(' ', '_')}.pdf"
    )
    file_type = FileType.PDF
    file_hash = factory.LazyFunction(lambda: hashlib.sha256(b"test_content").hexdigest())
    total_pages = factory.fuzzy.FuzzyInteger(10, 500)
    total_chunks = factory.fuzzy.FuzzyInteger(50, 2000)
    categories = factory.LazyFunction(
        lambda: factory.Faker('words', nb=factory.fuzzy.FuzzyInteger(1, 5).fuzz()).generate()
    )
    metadata = factory.LazyFunction(
        lambda: {
            'creation_date': datetime.now().isoformat(),
            'language': 'en',
            'subject': factory.Faker('word').generate()
        }
    )
    created_at = factory.LazyFunction(datetime.now)
    indexed_at = factory.LazyFunction(
        lambda: datetime.now() - timedelta(minutes=factory.fuzzy.FuzzyInteger(1, 60).fuzz())
    )
    # Remove processing_status field as it's not in the Book model
    
    @factory.post_generation
    def ensure_valid_path(obj, create, extracted, **kwargs):
        """Ensure the file path is valid for testing"""
        if create:
            # For tests, use /tmp paths
            obj.file_path = f"/tmp/test_books/{obj.id}.pdf"


class ChunkFactory(factory.Factory):
    """Factory for creating Chunk test instances"""
    
    class Meta:
        model = Chunk
    
    id = factory.LazyFunction(lambda: f"chunk_{uuid.uuid4().hex[:8]}")
    book_id = factory.SubFactory(BookFactory)
    chunk_index = factory.Sequence(lambda n: n)
    text = factory.Faker('text', max_nb_chars=1000)
    chunk_type = ChunkType.TEXT
    page_start = factory.fuzzy.FuzzyInteger(1, 100)
    page_end = factory.LazyAttribute(lambda obj: obj.page_start + factory.fuzzy.FuzzyInteger(0, 2).fuzz())
    metadata = factory.LazyFunction(
        lambda: {
            'font_info': {'size': 12, 'family': 'Arial'},
            'position': {'x': 100, 'y': 200}
        }
    )
    created_at = factory.LazyFunction(datetime.now)
    
    @factory.post_generation
    def set_book_reference(obj, create, extracted, **kwargs):
        """Set proper book reference for the chunk"""
        if create and hasattr(obj.book_id, 'id'):
            obj.book_id = obj.book_id.id


# EmbeddingVector not in models, removing this factory


class SearchResultFactory(factory.Factory):
    """Factory for creating SearchResult test instances"""
    
    class Meta:
        model = SearchResult
    
    chunk = factory.SubFactory(ChunkFactory)
    score = factory.fuzzy.FuzzyFloat(0.1, 1.0)
    match_type = factory.fuzzy.FuzzyChoice(['semantic', 'exact', 'hybrid'])
    highlights = factory.LazyFunction(
        lambda: [factory.Faker('sentence').generate() for _ in range(2)]
    )
    book_title = factory.Faker('sentence', nb_words=4)
    book_author = factory.Faker('name')
    chapter = factory.Faker('sentence', nb_words=3)
    page = factory.fuzzy.FuzzyInteger(1, 100)


class ConfigFactory:
    """Factory for creating configuration objects for testing"""
    
    @staticmethod
    def create_embedding_config(**overrides):
        """Create a test embedding configuration"""
        from src.core.config import EmbeddingConfig
        
        defaults = {
            'model': 'test-embed-model',
            'dimension': 768,
            'batch_size': 32,
            'ollama_host': 'http://localhost:11434',
            'timeout': 30
        }
        defaults.update(overrides)
        return EmbeddingConfig(**defaults)
    
    @staticmethod
    def create_qdrant_config(**overrides):
        """Create a test Qdrant configuration"""
        from src.core.config import QdrantConfig
        
        defaults = {
            'host': 'localhost',
            'port': 6333,
            'collection_name': 'test_collection',
            'use_grpc': False,
            'api_key': None,
            'https': False,
            'prefer_grpc': False
        }
        defaults.update(overrides)
        return QdrantConfig(**defaults)
    
    @staticmethod
    def create_database_config(**overrides):
        """Create a test database configuration"""
        from src.core.config import DatabaseConfig, QdrantConfig
        
        defaults = {
            'qdrant': ConfigFactory.create_qdrant_config(),
            'sqlite': {
                'path': ':memory:',
                'timeout': 30
            }
        }
        defaults.update(overrides)
        return DatabaseConfig(**defaults)


class TestDataBuilder:
    """Builder pattern for creating complex test scenarios"""
    
    def __init__(self):
        self.books = []
        self.chunks = []
        self.embeddings = []
        self.search_results = []
    
    def with_book(self, **kwargs) -> 'TestDataBuilder':
        """Add a book to the test scenario"""
        book = BookFactory(**kwargs)
        self.books.append(book)
        return self
    
    def with_books(self, count: int, **kwargs) -> 'TestDataBuilder':
        """Add multiple books to the test scenario"""
        for _ in range(count):
            self.books.append(BookFactory(**kwargs))
        return self
    
    def with_chunks_for_book(self, book_or_index: Any, count: int, **kwargs) -> 'TestDataBuilder':
        """Add chunks for a specific book"""
        if isinstance(book_or_index, int):
            book = self.books[book_or_index]
        else:
            book = book_or_index
        
        for i in range(count):
            chunk = ChunkFactory(book_id=book.id, chunk_index=i, **kwargs)
            self.chunks.append(chunk)
        return self
    
    def with_embeddings_for_chunks(self, **kwargs) -> 'TestDataBuilder':
        """Add embeddings for all chunks in scenario (placeholder for vector storage)"""
        # Since EmbeddingVector is not in models, we'll track this differently
        for chunk in self.chunks:
            embedding_data = {
                'chunk_id': chunk.id,
                'vector': [0.1] * 768,  # Mock embedding
                'model_name': 'test-model'
            }
            self.embeddings.append(embedding_data)
        return self
    
    def with_search_results(self, count: int, **kwargs) -> 'TestDataBuilder':
        """Add search results to the scenario"""
        for _ in range(count):
            if self.chunks:
                chunk = factory.fuzzy.FuzzyChoice(self.chunks).fuzz()
                book = next((b for b in self.books if b.id == chunk.book_id), None)
                result = SearchResultFactory(
                    chunk=chunk,
                    book_title=book.title if book else "Test Book",
                    book_author=book.author if book else "Test Author",
                    **kwargs
                )
            else:
                result = SearchResultFactory(**kwargs)
            self.search_results.append(result)
        return self
    
    def build(self) -> Dict[str, List[Any]]:
        """Build the complete test scenario"""
        return {
            'books': self.books,
            'chunks': self.chunks,
            'embeddings': self.embeddings,
            'search_results': self.search_results
        }


# Pre-built scenarios for common test cases
class TestScenarios:
    """Pre-built test scenarios for common use cases"""
    
    @staticmethod
    def single_book_with_chunks(chunk_count: int = 10):
        """Scenario: One book with multiple chunks"""
        return (TestDataBuilder()
                .with_book(title="Test Trading Book", author="Test Author")
                .with_chunks_for_book(0, chunk_count)
                .with_embeddings_for_chunks()
                .build())
    
    @staticmethod
    def multiple_books_library(book_count: int = 5, chunks_per_book: int = 10):
        """Scenario: Multiple books forming a library"""
        builder = TestDataBuilder().with_books(book_count)
        
        for i in range(book_count):
            builder.with_chunks_for_book(i, chunks_per_book)
        
        return builder.with_embeddings_for_chunks().build()
    
    @staticmethod
    def search_scenario(result_count: int = 5):
        """Scenario: Books with search results"""
        return (TestDataBuilder()
                .with_books(3)
                .with_chunks_for_book(0, 5)
                .with_chunks_for_book(1, 5)
                .with_chunks_for_book(2, 5)
                .with_embeddings_for_chunks()
                .with_search_results(result_count)
                .build())
    
    @staticmethod
    def empty_library():
        """Scenario: Empty library (no books)"""
        return TestDataBuilder().build()
    
    @staticmethod
    def processing_scenario():
        """Scenario: Books in various processing states"""
        return (TestDataBuilder()
                .with_book(title="Processing Book")
                .with_book(title="Completed Book", indexed_at=datetime.now())
                .with_book(title="Failed Book")
                .build())


# Utility functions for test setup
def create_temporary_pdf(content: str = "Test PDF content") -> Path:
    """Create a temporary PDF file for testing"""
    import tempfile
    
    temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
    temp_file.write(content.encode())
    temp_file.close()
    return Path(temp_file.name)


def cleanup_temporary_files(paths: List[Path]):
    """Clean up temporary files created during testing"""
    for path in paths:
        if path.exists():
            path.unlink()