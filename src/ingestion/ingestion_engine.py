"""
Book processing pipeline for TradeKnowledge

This orchestrates the entire process of ingesting a book:
1. Parse PDF/EPUB
2. Chunk the text
3. Generate embeddings  
4. Store everything
"""

import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib
from concurrent.futures import ProcessPoolExecutor

from ..core.models import Book, Chunk, FileType, IngestionStatus
from ..core.sqlite_storage import SQLiteStorage
from ..core.config import Config, get_config
from ..core.qdrant_storage import QdrantStorage
from .pdf_parser import PDFParser
from .text_chunker import TextChunker, ChunkingConfig
from .local_embeddings import LocalEmbeddingGenerator

logger = logging.getLogger(__name__)

class IngestionEngine:
    """
    Orchestrates the book ingestion pipeline.
    
    This class coordinates all the steps needed to ingest
    a book into our knowledge system.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize ingestion engine"""
        self.config = config or get_config()
        
        # Components
        self.pdf_parser = PDFParser()
        self.text_chunker = TextChunker(
            ChunkingConfig(
                chunk_size=self.config.ingestion.chunk_size,
                chunk_overlap=self.config.ingestion.chunk_overlap,
                min_chunk_size=self.config.ingestion.min_chunk_size,
                max_chunk_size=self.config.ingestion.max_chunk_size
            )
        )
        self.embedding_generator: Optional[LocalEmbeddingGenerator] = None
        self.sqlite_storage: Optional[SQLiteStorage] = None
        self.vector_storage: Optional[QdrantStorage] = None
        
        # Processing state
        self.current_status: Optional[IngestionStatus] = None
        
        # Process pool for CPU-intensive tasks
        self.process_pool: Optional[ProcessPoolExecutor] = None
    
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing ingestion engine...")
        
        # Initialize storage with proper configuration
        self.sqlite_storage = SQLiteStorage(self.config.database.sqlite.path)
        self.vector_storage = QdrantStorage(self.config.database.qdrant.collection_name)
        
        # Initialize embedding generator with configuration
        self.embedding_generator = LocalEmbeddingGenerator(self.config)
        
        # Initialize process pool for CPU-intensive tasks
        self.process_pool = ProcessPoolExecutor(max_workers=2)  # Limit to prevent system overload
        
        logger.info("Ingestion engine initialized")
    
    async def cleanup(self):
        """Cleanup resources"""
        # Save embedding cache - FIXED: Use configuration-driven path
        if self.embedding_generator:
            cache_dir = Path("data/embeddings")
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / "cache.json"
            self.embedding_generator.save_cache(str(cache_path))
        
        # Cleanup process pool
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
            logger.info("Process pool shut down")
    
    async def add_book(self,
                      file_path: str,
                      metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add a book to the knowledge base.
        
        This is the main entry point for ingesting books.
        It handles the entire pipeline from parsing to storage.
        
        Args:
            file_path: Path to the book file
            metadata: Optional metadata about the book
            
        Returns:
            Dictionary with ingestion results
        """
        path = Path(file_path)
        
        # Validate file
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return {'success': False, 'error': 'File not found'}
        
        if not path.suffix.lower() in ['.pdf', '.epub']:
            logger.error(f"Unsupported file type: {path.suffix}")
            return {'success': False, 'error': 'Unsupported file type'}
        
        # Calculate file hash for deduplication
        file_hash = self._calculate_file_hash(path)
        
        # Check if already processed
        existing_book = await self.sqlite_storage.get_book_by_hash(file_hash)
        if existing_book:
            logger.info(f"Book already exists: {existing_book.title}")
            return {
                'success': False,
                'error': 'Book already processed',
                'book_id': existing_book.id
            }
        
        # Start processing
        logger.info(f"Starting to process: {path.name}")
        
        try:
            # Step 1: Parse the file
            logger.info("Step 1: Parsing file...")
            parse_result = await self._parse_file(path)
            
            if parse_result['errors']:
                logger.error(f"Parse errors: {parse_result['errors']}")
                return {
                    'success': False,
                    'error': 'Failed to parse file',
                    'details': parse_result['errors']
                }
            
            # Step 2: Create book record
            logger.info("Step 2: Creating book record...")
            book = await self._create_book_record(
                path, file_hash, parse_result, metadata
            )
            
            # Initialize status tracking
            self.current_status = IngestionStatus(
                book_id=book.id,
                status='processing',
                total_pages=len(parse_result['pages'])
            )
            
            # Save book to database
            await self.sqlite_storage.save_book(book)
            
            # Step 3: Chunk the text
            logger.info("Step 3: Chunking text...")
            chunks = await self._chunk_book(parse_result['pages'], book.id)
            
            self.current_status.total_chunks = len(chunks)
            self.current_status.current_stage = 'chunking'
            
            # Step 4: Generate embeddings
            logger.info("Step 4: Generating embeddings...")
            self.current_status.current_stage = 'embedding'
            
            embeddings = await self.embedding_generator.generate_embeddings(chunks)
            
            # Step 5: Store everything
            logger.info("Step 5: Storing data...")
            self.current_status.current_stage = 'storing'
            
            # Store chunks in SQLite
            await self.sqlite_storage.save_chunks(chunks)
            
            # Store embeddings in Qdrant
            success = await self.vector_storage.save_embeddings(chunks, embeddings)
            
            if not success:
                logger.error("Failed to save embeddings")
                return {
                    'success': False,
                    'error': 'Failed to save embeddings'
                }
            
            # Update book record
            book.total_chunks = len(chunks)
            book.indexed_at = datetime.now()
            await self.sqlite_storage.update_book(book)
            
            # Complete!
            self.current_status.status = 'completed'
            self.current_status.completed_at = datetime.now()
            self.current_status.progress_percent = 100.0
            
            logger.info(f"Successfully processed book: {book.title}")
            
            return {
                'success': True,
                'book_id': book.id,
                'title': book.title,
                'chunks_created': len(chunks),
                'processing_time': (
                    self.current_status.completed_at - self.current_status.started_at
                ).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Error processing book: {e}", exc_info=True)
            
            # Update status if available
            if self.current_status:
                self.current_status.status = 'failed'
                self.current_status.error_message = str(e)
            
            # Cleanup partial state - try to remove any partial book data
            try:
                if 'book' in locals():
                    logger.warning(f"Cleaning up partial book data for: {book.id}")
                    await self.remove_book(book.id)
            except Exception as cleanup_error:
                logger.error(f"Failed to cleanup partial book data: {cleanup_error}")
            
            return {
                'success': False,
                'error': f'Processing failed: {str(e)}',
                'cleanup_attempted': True
            }
    
    async def remove_book(self, book_id: str) -> Dict[str, Any]:
        """Remove a book and all its data"""
        try:
            # Get book info first
            book = await self.sqlite_storage.get_book(book_id)
            if not book:
                return {'success': False, 'error': 'Book not found'}
            
            # Get chunk IDs for vector deletion
            chunks = await self.sqlite_storage.get_chunks_by_book(book_id)
            chunk_ids = [chunk.id for chunk in chunks]
            
            # Delete from vector storage
            if chunk_ids:
                await self.vector_storage.delete_embeddings(chunk_ids)
            
            # Delete from SQLite (cascades to chunks)
            await self.sqlite_storage.delete_book(book_id)
            
            logger.info(f"Removed book: {book.title}")
            
            return {
                'success': True,
                'message': f'Removed book: {book.title}',
                'chunks_deleted': len(chunk_ids)
            }
            
        except Exception as e:
            logger.error(f"Error removing book: {e}")
            return {
                'success': False,
                'error': f'Failed to remove book: {str(e)}'
            }
    
    async def list_books(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all books in the system"""
        books = await self.sqlite_storage.list_books(category=category)
        
        return [
            {
                'id': book.id,
                'title': book.title,
                'author': book.author,
                'total_chunks': book.total_chunks,
                'categories': book.categories,
                'file_path': book.file_path,
                'created_at': book.created_at.isoformat(),
                'indexed_at': book.indexed_at.isoformat() if book.indexed_at else None
            }
            for book in books
        ]
    
    async def get_book_details(self, book_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a book"""
        book = await self.sqlite_storage.get_book(book_id)
        if not book:
            return None
        
        # Get chunk statistics
        chunks = await self.sqlite_storage.get_chunks_by_book(book_id)
        
        chunk_types = {}
        for chunk in chunks:
            chunk_type = chunk.chunk_type.value
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        return {
            'id': book.id,
            'title': book.title,
            'author': book.author,
            'isbn': book.isbn,
            'file_path': book.file_path,
            'file_type': book.file_type.value,
            'total_pages': book.total_pages,
            'total_chunks': book.total_chunks,
            'categories': book.categories,
            'metadata': book.metadata,
            'created_at': book.created_at.isoformat(),
            'indexed_at': book.indexed_at.isoformat() if book.indexed_at else None,
            'chunk_statistics': {
                'total': len(chunks),
                'by_type': chunk_types
            }
        }
    
    async def get_ingestion_status(self) -> Optional[Dict[str, Any]]:
        """Get current ingestion status"""
        if not self.current_status:
            return None
        
        return {
            'book_id': self.current_status.book_id,
            'status': self.current_status.status,
            'progress_percent': self.current_status.progress_percent,
            'current_stage': self.current_status.current_stage,
            'total_pages': self.current_status.total_pages,
            'processed_pages': self.current_status.processed_pages,
            'total_chunks': self.current_status.total_chunks,
            'embedded_chunks': self.current_status.embedded_chunks,
            'started_at': self.current_status.started_at.isoformat(),
            'error_message': self.current_status.error_message,
            'warnings': self.current_status.warnings
        }
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    async def _parse_file(self, file_path: Path) -> Dict[str, Any]:
        """Parse file based on type"""
        if file_path.suffix.lower() == '.pdf':
            # PERFORMANCE FIX: Use process pool for CPU-intensive PDF parsing
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.process_pool, 
                self.pdf_parser.parse_file, 
                file_path
            )
        else:
            # TODO: Add EPUB parser
            raise NotImplementedError(f"Parser for {file_path.suffix} not implemented")
    
    async def _create_book_record(self,
                                 file_path: Path,
                                 file_hash: str,
                                 parse_result: Dict[str, Any],
                                 metadata: Optional[Dict[str, Any]]) -> Book:
        """Create book record from parse results"""
        book_metadata = parse_result['metadata']
        
        # Generate book ID (use ISBN if available)
        book_id = book_metadata.get('isbn')
        if not book_id:
            # Generate from title and author
            title = book_metadata.get('title', file_path.stem)
            author = book_metadata.get('author', 'Unknown')
            book_id = f"{title[:20]}_{author[:20]}_{file_hash[:8]}".replace(' ', '_')
        
        # Merge metadata
        if metadata:
            book_metadata.update(metadata)
        
        # Add statistics
        book_metadata['statistics'] = parse_result.get('statistics', {})
        
        # Create book object
        book = Book(
            id=book_id,
            title=book_metadata.get('title', file_path.stem),
            author=book_metadata.get('author'),
            isbn=book_metadata.get('isbn'),
            file_path=str(file_path),
            file_type=FileType.PDF if file_path.suffix.lower() == '.pdf' else FileType.EPUB,
            file_hash=file_hash,
            total_pages=book_metadata.get('total_pages', 0),
            categories=metadata.get('categories', []) if metadata else [],
            metadata=book_metadata
        )
        
        return book
    
    async def _chunk_book(self,
                         pages: List[Dict[str, Any]],
                         book_id: str) -> List[Chunk]:
        """Chunk book pages"""
        # Update status
        if self.current_status:
            self.current_status.processed_pages = len(pages)
        
        # Use page-aware chunking
        chunks = await asyncio.to_thread(
            self.text_chunker.chunk_pages,
            pages,
            book_id,
            {}
        )
        
        # Update chunk IDs for vector storage
        for chunk in chunks:
            chunk.embedding_id = chunk.id
        
        return chunks

# Test the ingestion engine
async def test_ingestion_engine():
    """Test the ingestion engine"""
    
    # Initialize
    engine = IngestionEngine()
    await engine.initialize()
    
    # Test with a sample PDF (you'll need to provide one)
    # Test with a sample PDF (you'll need to provide one)
    # FIXED: Use Path for better path handling
    test_file = Path("data/books/sample.pdf")
    
    if test_file.exists():
        print(f"Testing ingestion with: {test_file}")
        
        # Add book
        result = await engine.add_book(
            str(test_file),
            metadata={
                'categories': ['testing', 'sample'],
                'description': 'Test book for ingestion'
            }
        )
        
        print(f"Ingestion result: {result}")
        
        if result['success']:
            # List books
            books = await engine.list_books()
            print(f"\nBooks in system: {len(books)}")
            
            # Get book details
            book_details = await engine.get_book_details(result['book_id'])
            print(f"Book details: {book_details}")
        
    else:
        print(f"Test file not found: {test_file}")
        print("Please add a PDF file to test ingestion")
    
    # Cleanup
    await engine.cleanup()

if __name__ == "__main__":
    asyncio.run(test_ingestion_engine())