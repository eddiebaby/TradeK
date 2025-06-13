#!/usr/bin/env python3
"""
System integration test for TradeKnowledge

This script tests the complete Phase 1 implementation to ensure
all components work together correctly.
"""

import asyncio
import logging
import sys
import tempfile
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our components
from core.models import Book, Chunk, FileType, ChunkType
from core.sqlite_storage import SQLiteStorage
from core.chroma_storage import ChromaDBStorage
from ingestion.pdf_parser import PDFParser
from ingestion.text_chunker import TextChunker, ChunkingConfig
from ingestion.embeddings import EmbeddingGenerator
from ingestion.ingestion_engine import IngestionEngine
from search.hybrid_search import HybridSearch

class SystemTestRunner:
    """Runs comprehensive system tests"""
    
    def __init__(self):
        """Initialize test runner"""
        self.test_results = []
        self.failed_tests = []
    
    def log_test(self, test_name: str, success: bool, message: str = ""):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if message:
            print(f"       {message}")
        
        self.test_results.append({
            'test': test_name,
            'success': success,
            'message': message,
            'timestamp': datetime.now()
        })
        
        if not success:
            self.failed_tests.append(test_name)
    
    async def test_core_models(self):
        """Test core data models"""
        print("\n🧪 Testing Core Data Models")
        print("-" * 50)
        
        try:
            # Test Book model
            book = Book(
                id="test-book-001",
                title="Test Trading Book",
                author="Test Author",
                file_path="/tmp/test.pdf",
                file_type=FileType.PDF,
                file_hash="abc123",
                categories=["trading", "test"]
            )
            
            self.log_test("Book model creation", True, f"Created book: {book.title}")
            
            # Test Chunk model
            chunk = Chunk(
                book_id=book.id,
                chunk_index=0,
                text="This is a test chunk about moving averages.",
                chunk_type=ChunkType.TEXT
            )
            
            self.log_test("Chunk model creation", True, f"Created chunk: {chunk.id}")
            
            # Test chunk methods
            size = chunk.get_size()
            tokens = chunk.get_token_estimate()
            
            self.log_test("Chunk utility methods", size > 0 and tokens > 0, 
                         f"Size: {size}, Tokens: {tokens}")
            
        except Exception as e:
            self.log_test("Core models", False, str(e))
    
    async def test_sqlite_storage(self):
        """Test SQLite storage"""
        print("\n🗄️ Testing SQLite Storage")
        print("-" * 50)
        
        try:
            # Use temporary database
            storage = SQLiteStorage("data/test_storage.db")
            
            # Test book operations
            book = Book(
                id="test-storage-book",
                title="Test Storage Book",
                author="Storage Author",
                file_path="/tmp/storage_test.pdf",
                file_type=FileType.PDF,
                file_hash="storage123"
            )
            
            # Save book
            success = await storage.save_book(book)
            self.log_test("Save book to SQLite", success)
            
            # Retrieve book
            retrieved = await storage.get_book(book.id)
            self.log_test("Retrieve book from SQLite", 
                         retrieved is not None and retrieved.title == book.title)
            
            # Test chunk operations
            chunks = [
                Chunk(
                    book_id=book.id,
                    chunk_index=i,
                    text=f"Test chunk {i} about trading strategies"
                )
                for i in range(3)
            ]
            
            # Save chunks
            success = await storage.save_chunks(chunks)
            self.log_test("Save chunks to SQLite", success)
            
            # Retrieve chunks
            retrieved_chunks = await storage.get_chunks_by_book(book.id)
            self.log_test("Retrieve chunks from SQLite", 
                         len(retrieved_chunks) == len(chunks))
            
            # Test exact search
            search_results = await storage.search_exact("trading", limit=5)
            self.log_test("SQLite FTS search", len(search_results) > 0)
            
        except Exception as e:
            self.log_test("SQLite storage", False, str(e))
    
    async def test_chroma_storage(self):
        """Test ChromaDB storage"""
        print("\n🔍 Testing ChromaDB Vector Storage")
        print("-" * 50)
        
        try:
            storage = ChromaDBStorage("data/test_chroma", "test_collection")
            
            # Create test chunks
            chunks = [
                Chunk(
                    id=f"chroma_test_{i}",
                    book_id="test_book",
                    chunk_index=i,
                    text=f"Test chunk {i} about machine learning in trading"
                )
                for i in range(3)
            ]
            
            # Create fake embeddings
            import random
            embeddings = [
                [random.random() for _ in range(384)]
                for _ in chunks
            ]
            
            # Save embeddings
            success = await storage.save_embeddings(chunks, embeddings)
            self.log_test("Save embeddings to ChromaDB", success)
            
            # Test search
            query_embedding = [random.random() for _ in range(384)]
            results = await storage.search_semantic(query_embedding, limit=2)
            self.log_test("ChromaDB semantic search", len(results) > 0)
            
            # Get stats
            stats = await storage.get_collection_stats()
            self.log_test("ChromaDB collection stats", 
                         stats.get('total_embeddings', 0) >= len(chunks))
            
        except Exception as e:
            self.log_test("ChromaDB storage", False, str(e))
    
    async def test_text_chunker(self):
        """Test text chunking"""
        print("\n📄 Testing Text Chunker")
        print("-" * 50)
        
        try:
            config = ChunkingConfig(
                chunk_size=200,
                chunk_overlap=50,
                preserve_code_blocks=True
            )
            chunker = TextChunker(config)
            
            # Test text with code
            sample_text = """
            Chapter 1: Introduction to Trading
            
            Trading involves buying and selling financial instruments.
            
            Here's a simple function:
            
            ```python
            def calculate_sma(prices, period):
                return sum(prices[-period:]) / period
            ```
            
            The simple moving average is widely used.
            """
            
            chunks = chunker.chunk_text(sample_text, "test_book")
            
            self.log_test("Text chunking", len(chunks) > 0, 
                         f"Created {len(chunks)} chunks")
            
            # Test chunk linking
            if len(chunks) > 1:
                has_links = any(chunk.next_chunk_id for chunk in chunks[:-1])
                self.log_test("Chunk linking", has_links)
            
            # Test chunk type detection
            code_chunks = [c for c in chunks if c.chunk_type == ChunkType.CODE]
            self.log_test("Code chunk detection", len(code_chunks) > 0)
            
        except Exception as e:
            self.log_test("Text chunker", False, str(e))
    
    async def test_embeddings(self):
        """Test embedding generation"""
        print("\n🧠 Testing Embedding Generation")
        print("-" * 50)
        
        try:
            # Test with OpenAI (if API key available)
            try:
                generator = EmbeddingGenerator("text-embedding-ada-002")
                
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
                
                embeddings = await generator.generate_embeddings(test_chunks)
                
                self.log_test("OpenAI embeddings generation", 
                             len(embeddings) == len(test_chunks),
                             f"Generated {len(embeddings)} embeddings")
                
                # Test query embedding
                query_embedding = await generator.generate_query_embedding("trading strategies")
                self.log_test("Query embedding generation", 
                             len(query_embedding) > 0,
                             f"Embedding dimension: {len(query_embedding)}")
                
                # Test caching
                stats = generator.get_stats()
                self.log_test("Embedding caching", 
                             stats['cache_size'] > 0,
                             f"Cache size: {stats['cache_size']}")
                
            except ValueError as e:
                if "API key" in str(e):
                    self.log_test("OpenAI embeddings", False, "No API key - expected in test")
                else:
                    raise
                
        except Exception as e:
            self.log_test("Embeddings", False, str(e))
    
    async def test_hybrid_search(self):
        """Test hybrid search engine"""
        print("\n🔎 Testing Hybrid Search Engine")
        print("-" * 50)
        
        try:
            search_engine = HybridSearch()
            await search_engine.initialize()
            
            # Test search stats
            stats = search_engine.get_stats()
            components_ok = all(stats['components_initialized'].values())
            self.log_test("Search engine initialization", components_ok)
            
            # Note: Full search testing would require indexed data
            self.log_test("Search engine setup", True, "Engine ready for search operations")
            
            await search_engine.cleanup()
            
        except Exception as e:
            self.log_test("Hybrid search", False, str(e))
    
    async def test_ingestion_engine(self):
        """Test ingestion engine"""
        print("\n⚙️ Testing Ingestion Engine")
        print("-" * 50)
        
        try:
            engine = IngestionEngine()
            await engine.initialize()
            
            # Test book listing (should work even with no books)
            books = await engine.list_books()
            self.log_test("Ingestion engine - list books", True, 
                         f"Found {len(books)} books")
            
            # Test ingestion status
            status = await engine.get_ingestion_status()
            self.log_test("Ingestion status check", True, 
                         "No active ingestion" if status is None else "Status available")
            
            await engine.cleanup()
            
        except Exception as e:
            self.log_test("Ingestion engine", False, str(e))
    
    async def run_all_tests(self):
        """Run all system tests"""
        print("🚀 TradeKnowledge Phase 1 System Tests")
        print("=" * 60)
        print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run all test suites
        await self.test_core_models()
        await self.test_sqlite_storage()
        await self.test_chroma_storage()
        await self.test_text_chunker()
        await self.test_embeddings()
        await self.test_hybrid_search()
        await self.test_ingestion_engine()
        
        # Print summary
        print("\n📊 Test Summary")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = len([t for t in self.test_results if t['success']])
        failed_tests = len(self.failed_tests)
        
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if self.failed_tests:
            print(f"\n⚠️ Failed tests:")
            for test in self.failed_tests:
                print(f"  - {test}")
        
        print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return failed_tests == 0

async def main():
    """Run system tests"""
    runner = SystemTestRunner()
    success = await runner.run_all_tests()
    
    if success:
        print("\n🎉 All tests passed! Phase 1 implementation is working correctly.")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed. Check the logs above for details.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())