"""
Enhanced Book Processor for TradeKnowledge Phase 2

This is the unified interface that integrates all Phase 2 components:
- OCR-enabled PDF parser
- EPUB parser
- Content analyzer for code/formulas/tables
- Advanced caching
- Query suggestion integration
- Performance optimizations
"""

import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import hashlib
import mimetypes

from ..core.models import Book, Chunk, FileType, IngestionStatus
from ..core.sqlite_storage import SQLiteStorage
from ..core.qdrant_storage import QdrantStorage
from ..core.config import get_config
from .pdf_parser import PDFParser
from .epub_parser import EPUBParser
from .content_analyzer import ContentAnalyzer
from .text_chunker import TextChunker, ChunkingConfig
from .local_embeddings import LocalEmbeddingGenerator
from .resource_monitor import get_resource_monitor, monitor_processing
from ..utils.cache_manager import get_cache_manager
from ..search.query_suggester import QuerySuggester

logger = logging.getLogger(__name__)

class EnhancedBookProcessor:
    """
    Enhanced book processor that integrates all Phase 2 components.
    
    Features:
    - Multi-format support (PDF with OCR, EPUB)
    - Intelligent content analysis
    - Advanced caching for performance
    - Query suggestion integration
    - Comprehensive error handling
    """
    
    def __init__(self):
        """Initialize enhanced book processor"""
        
        # Core parsers
        self.pdf_parser = PDFParser(enable_ocr=True)
        self.epub_parser = EPUBParser()
        
        # Content analysis
        self.content_analyzer = ContentAnalyzer()
        
        # Text processing
        self.text_chunker = TextChunker(
            ChunkingConfig(
                chunk_size=1000,
                chunk_overlap=200,
                min_chunk_size=100,
                max_chunk_size=2000
            )
        )
        
        # Storage
        self.embedding_generator: Optional[LocalEmbeddingGenerator] = None
        self.sqlite_storage: Optional[SQLiteStorage] = None
        self.vector_storage: Optional[QdrantStorage] = None
        
        # Phase 2 components
        self.cache_manager = None
        self.query_suggester: Optional[QuerySuggester] = None
        
        # Processing state
        self.current_status: Optional[IngestionStatus] = None
        
        # Resource monitoring
        self.resource_monitor = get_resource_monitor()
        
        # Supported file types
        self.supported_extensions = {
            '.pdf': FileType.PDF,
            '.epub': FileType.EPUB
        }
    
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing enhanced book processor...")
        
        # Initialize storage with proper configuration  
        config = get_config()
        self.sqlite_storage = SQLiteStorage(config.database.sqlite.path)
        self.vector_storage = QdrantStorage(config.database.qdrant.collection_name)
        
        # Initialize embedding generator with configuration
        self.embedding_generator = LocalEmbeddingGenerator(config)
        
        # Initialize Phase 2 components
        self.cache_manager = await get_cache_manager()
        
        self.query_suggester = QuerySuggester()
        await self.query_suggester.initialize()
        
        logger.info("Enhanced book processor initialized")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.embedding_generator:
            self.embedding_generator.save_cache("data/embeddings/cache.json")
        
        if self.cache_manager:
            await self.cache_manager.cleanup()
    
    async def add_book(self,
                      file_path: str,
                      metadata: Optional[Dict[str, Any]] = None,
                      force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Add a book to the knowledge base with enhanced processing.
        
        Args:
            file_path: Path to the book file
            metadata: Optional metadata about the book
            force_reprocess: Force reprocessing even if book exists
            
        Returns:
            Dictionary with ingestion results
        """
        path = Path(file_path)
        
        # Validate file
        validation_result = await self._validate_file(path)
        if not validation_result['valid']:
            return {'success': False, 'error': validation_result['error']}
        
        # Calculate file hash for deduplication
        file_hash = await self._calculate_file_hash(path)
        
        # Check cache first
        cache_key = f"book_processing:{file_hash}"
        if not force_reprocess:
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result:
                logger.info(f"Found cached processing result for {path.name}")
                return cached_result
        
        # Check if already processed
        if not force_reprocess:
            existing_book = await self.sqlite_storage.get_book_by_hash(file_hash)
            if existing_book:
                logger.info(f"Book already exists: {existing_book.title}")
                result = {
                    'success': True,
                    'book_id': existing_book.id,
                    'title': existing_book.title,
                    'message': 'Book already processed',
                    'reprocessed': False
                }
                await self.cache_manager.set(cache_key, result, ttl=86400)  # Cache for 1 day
                return result
        
        # Start processing with resource monitoring
        logger.info(f"Starting enhanced processing: {path.name}")
        
        # Start resource monitoring
        await self.resource_monitor.start_monitoring()
        
        # Set up resource monitoring callback
        async def resource_callback(check):
            if check['memory_critical'] and self.current_status:
                self.current_status.current_stage = 'paused_memory'
                logger.warning("Processing paused due to memory constraints")
                await self.resource_monitor.wait_for_memory_available()
                if self.current_status:
                    self.current_status.current_stage = 'resumed'
        
        self.resource_monitor.add_callback(resource_callback)
        
        try:
            # Check initial memory availability
            if not await self.resource_monitor.wait_for_memory_available(timeout=30):
                return {
                    'success': False,
                    'error': 'Insufficient memory to start processing'
                }
            # Step 1: Parse the file with appropriate parser
            logger.info("Step 1: Parsing file with enhanced parsers...")
            parse_result = await self._parse_file_enhanced(path)
            
            if parse_result['errors']:
                logger.error(f"Parse errors: {parse_result['errors']}")
                return {
                    'success': False,
                    'error': 'Failed to parse file',
                    'details': parse_result['errors']
                }
            
            # Step 2: Analyze content for special elements
            logger.info("Step 2: Analyzing content...")
            content_analysis = await self._analyze_content(parse_result['pages'])
            
            # Step 3: Create enhanced book record
            logger.info("Step 3: Creating enhanced book record...")
            book = await self._create_enhanced_book_record(
                path, file_hash, parse_result, content_analysis, metadata
            )
            
            # Initialize status tracking
            self.current_status = IngestionStatus(
                book_id=book.id,
                status='processing',
                total_pages=len(parse_result['pages'])
            )
            
            # Save book to database
            if force_reprocess:
                # Delete existing book first
                existing_book = await self.sqlite_storage.get_book_by_hash(file_hash)
                if existing_book:
                    await self.remove_book(existing_book.id)
            
            await self.sqlite_storage.save_book(book)
            
            # Step 4: Enhanced chunking with content awareness
            logger.info("Step 4: Enhanced chunking...")
            chunks = await self._chunk_book_enhanced(
                parse_result['pages'], book.id, content_analysis
            )
            
            self.current_status.total_chunks = len(chunks)
            self.current_status.current_stage = 'chunking'
            
            # Step 5: Generate embeddings with caching
            logger.info("Step 5: Generating embeddings with caching...")
            self.current_status.current_stage = 'embedding'
            
            embeddings = await self._generate_embeddings_cached(chunks)
            
            # Step 6: Store everything
            logger.info("Step 6: Storing data...")
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
            
            # Step 7: Update search suggestions
            logger.info("Step 7: Updating search suggestions...")
            await self._update_search_suggestions(book, content_analysis)
            
            # Update book record
            book.total_chunks = len(chunks)
            book.indexed_at = datetime.now()
            await self.sqlite_storage.update_book(book)
            
            # Complete!
            self.current_status.status = 'completed'
            self.current_status.completed_at = datetime.now()
            self.current_status.progress_percent = 100.0
            
            processing_time = (
                self.current_status.completed_at - self.current_status.started_at
            ).total_seconds()
            
            result = {
                'success': True,
                'book_id': book.id,
                'title': book.title,
                'author': book.author,
                'file_type': book.file_type.value,
                'chunks_created': len(chunks),
                'content_analysis': {
                    'code_blocks': len(content_analysis.get('code', [])),
                    'formulas': len(content_analysis.get('formulas', [])),
                    'tables': len(content_analysis.get('tables', [])),
                    'strategies': len(content_analysis.get('strategies', []))
                },
                'processing_time': processing_time,
                'ocr_used': parse_result.get('metadata', {}).get('ocr_processed', False),
                'reprocessed': force_reprocess
            }
            
            # Cache the result
            await self.cache_manager.set(cache_key, result, ttl=86400)
            
            logger.info(f"Successfully processed book: {book.title}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing book: {e}", exc_info=True)
            
            if self.current_status:
                self.current_status.status = 'failed'
                self.current_status.error_message = str(e)
            
            return {
                'success': False,
                'error': f'Processing failed: {str(e)}'
            }
        
        finally:
            # Stop resource monitoring and get summary
            await self.resource_monitor.stop_monitoring()
            
            # Log resource usage summary
            try:
                summary = self.resource_monitor.get_usage_summary()
                logger.info(f"Resource usage summary: {summary}")
                
                if summary.get('peak_memory_exceeded'):
                    logger.warning("Peak memory usage exceeded limits during processing")
                    
            except Exception as e:
                logger.debug(f"Could not generate resource summary: {e}")
    
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
            
            # Clear related caches
            if self.cache_manager:
                await self.cache_manager.delete(f"book_processing:{book.file_hash}")
                await self.cache_manager.clear("search")  # Clear search cache
            
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
        """List all books in the system with enhanced information"""
        # Try cache first
        cache_key = f"book_list:{category or 'all'}"
        cached_books = await self.cache_manager.get(cache_key)
        if cached_books:
            return cached_books
        
        books = await self.sqlite_storage.list_books(category=category)
        
        result = []
        for book in books:
            book_info = {
                'id': book.id,
                'title': book.title,
                'author': book.author,
                'file_type': book.file_type.value,
                'total_chunks': book.total_chunks,
                'total_pages': book.total_pages,
                'categories': book.categories,
                'file_path': book.file_path,
                'created_at': book.created_at.isoformat(),
                'indexed_at': book.indexed_at.isoformat() if book.indexed_at else None,
                'has_code': book.metadata.get('content_analysis', {}).get('has_code', False),
                'has_formulas': book.metadata.get('content_analysis', {}).get('has_formulas', False),
                'ocr_processed': book.metadata.get('ocr_processed', False)
            }
            result.append(book_info)
        
        # Cache for 5 minutes
        await self.cache_manager.set(cache_key, result, ttl=300)
        
        return result
    
    async def get_book_details(self, book_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a book with enhanced metadata"""
        # Try cache first
        cache_key = f"book_details:{book_id}"
        cached_details = await self.cache_manager.get(cache_key)
        if cached_details:
            return cached_details
        
        book = await self.sqlite_storage.get_book(book_id)
        if not book:
            return None
        
        # Get chunk statistics
        chunks = await self.sqlite_storage.get_chunks_by_book(book_id)
        
        chunk_types = {}
        for chunk in chunks:
            chunk_type = chunk.chunk_type.value
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        result = {
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
            },
            'content_analysis': book.metadata.get('content_analysis', {}),
            'processing_info': {
                'ocr_processed': book.metadata.get('ocr_processed', False),
                'ocr_confidence': book.metadata.get('ocr_confidence'),
                'parse_method': book.metadata.get('parse_method', 'standard')
            }
        }
        
        # Cache for 10 minutes
        await self.cache_manager.set(cache_key, result, ttl=600)
        
        return result
    
    async def _validate_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate file for processing"""
        if not file_path.exists():
            return {'valid': False, 'error': 'File not found'}
        
        if file_path.suffix.lower() not in self.supported_extensions:
            supported = ', '.join(self.supported_extensions.keys())
            return {
                'valid': False, 
                'error': f'Unsupported file type: {file_path.suffix}. Supported: {supported}'
            }
        
        # Check file size (warn if very large)
        file_size = file_path.stat().st_size
        if file_size > 500 * 1024 * 1024:  # 500MB
            logger.warning(f"Large file detected: {file_size / 1024 / 1024:.1f}MB")
        
        return {'valid': True}
    
    async def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file with caching"""
        # Check if we have a cached hash
        cache_key = f"file_hash:{file_path}:{file_path.stat().st_mtime}"
        cached_hash = await self.cache_manager.get(cache_key)
        if cached_hash:
            return cached_hash
        
        # Calculate hash
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        file_hash = sha256_hash.hexdigest()
        
        # Cache for 24 hours
        await self.cache_manager.set(cache_key, file_hash, ttl=86400)
        
        return file_hash
    
    async def _parse_file_enhanced(self, file_path: Path) -> Dict[str, Any]:
        """Parse file with enhanced parsers based on type"""
        file_type = self.supported_extensions[file_path.suffix.lower()]
        
        if file_type == FileType.PDF:
            # Use enhanced PDF parser with OCR support
            return await self.pdf_parser.parse_file_async(file_path)
        elif file_type == FileType.EPUB:
            # Use EPUB parser
            return await self.epub_parser.parse_file_async(file_path)
        else:
            raise NotImplementedError(f"Parser for {file_path.suffix} not implemented")
    
    async def _analyze_content(self, pages: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze content for special elements"""
        # Combine all page text
        full_text = "\n\n".join(page.get('text', '') for page in pages if page.get('text'))
        
        # Run content analysis
        content_analysis = await asyncio.to_thread(
            self.content_analyzer.extract_special_content, full_text
        )
        
        return content_analysis
    
    async def _create_enhanced_book_record(self,
                                         file_path: Path,
                                         file_hash: str,
                                         parse_result: Dict[str, Any],
                                         content_analysis: Dict[str, Any],
                                         metadata: Optional[Dict[str, Any]]) -> Book:
        """Create enhanced book record with content analysis"""
        book_metadata = parse_result['metadata'].copy()
        
        # Add content analysis to metadata
        book_metadata['content_analysis'] = {
            'has_code': len(content_analysis.get('code', [])) > 0,
            'has_formulas': len(content_analysis.get('formulas', [])) > 0,
            'has_tables': len(content_analysis.get('tables', [])) > 0,
            'has_strategies': len(content_analysis.get('strategies', [])) > 0,
            'code_languages': list(set(
                item.get('metadata', {}).get('language', 'unknown')
                for item in content_analysis.get('code', [])
            )),
            'content_summary': {
                'code_blocks': len(content_analysis.get('code', [])),
                'formulas': len(content_analysis.get('formulas', [])),
                'tables': len(content_analysis.get('tables', [])),
                'strategies': len(content_analysis.get('strategies', []))
            }
        }
        
        # Generate book ID
        book_id = book_metadata.get('isbn')
        if not book_id:
            title = book_metadata.get('title', file_path.stem)
            author = book_metadata.get('author', 'Unknown')
            book_id = f"{title[:20]}_{author[:20]}_{file_hash[:8]}".replace(' ', '_')
        
        # Merge additional metadata
        if metadata:
            book_metadata.update(metadata)
        
        # Add statistics
        book_metadata['statistics'] = parse_result.get('statistics', {})
        
        # Determine file type
        file_type = self.supported_extensions[file_path.suffix.lower()]
        
        # Create book object
        book = Book(
            id=book_id,
            title=book_metadata.get('title', file_path.stem),
            author=book_metadata.get('author'),
            isbn=book_metadata.get('isbn'),
            file_path=str(file_path),
            file_type=file_type,
            file_hash=file_hash,
            total_pages=book_metadata.get('total_pages', 0),
            categories=metadata.get('categories', []) if metadata else [],
            metadata=book_metadata
        )
        
        return book
    
    async def _chunk_book_enhanced(self,
                                 pages: List[Dict[str, Any]],
                                 book_id: str,
                                 content_analysis: Dict[str, Any]) -> List[Chunk]:
        """Enhanced chunking with content awareness"""
        # Update status
        if self.current_status:
            self.current_status.processed_pages = len(pages)
        
        # Use page-aware chunking
        chunks = await asyncio.to_thread(
            self.text_chunker.chunk_pages,
            pages,
            book_id,
            {'content_analysis': content_analysis}
        )
        
        # Enhance chunks with content metadata
        for chunk in chunks:
            chunk.embedding_id = chunk.id
            # Add content analysis metadata to chunks
            if not chunk.metadata:
                chunk.metadata = {}
            chunk.metadata['content_analysis'] = content_analysis
        
        return chunks
    
    async def _generate_embeddings_cached(self, chunks: List[Chunk]) -> List[List[float]]:
        """Generate embeddings with caching and batching"""
        embeddings = []
        uncached_chunks = []
        uncached_indices = []
        
        logger.info(f"Processing embeddings for {len(chunks)} chunks...")
        
        # First pass: check cache for all chunks
        for i, chunk in enumerate(chunks):
            cache_key = f"embedding:{hashlib.md5(chunk.text.encode()).hexdigest()}"
            cached_embedding = await self.cache_manager.get(cache_key, 'embedding')
            
            if cached_embedding:
                embeddings.append(cached_embedding)
            else:
                embeddings.append(None)  # Placeholder
                uncached_chunks.append(chunk)
                uncached_indices.append(i)
        
        if uncached_chunks:
            logger.info(f"Generating embeddings for {len(uncached_chunks)} uncached chunks...")
            
            # Get dynamic batch size based on available memory
            recommendations = self.resource_monitor.get_processing_recommendations()
            EMBEDDING_BATCH_SIZE = recommendations['embedding_batch_size']
            
            logger.info(f"Using dynamic batch size: {EMBEDDING_BATCH_SIZE} (memory status: {recommendations['memory_status']})")
            
            for batch_start in range(0, len(uncached_chunks), EMBEDDING_BATCH_SIZE):
                batch_end = min(batch_start + EMBEDDING_BATCH_SIZE, len(uncached_chunks))
                batch_chunks = uncached_chunks[batch_start:batch_end]
                
                logger.debug(f"Processing embedding batch {batch_start + 1}-{batch_end}")
                
                # Generate embeddings for this batch
                batch_embeddings = await self.embedding_generator.generate_embeddings(batch_chunks)
                
                # Cache and store results
                for j, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
                    original_index = uncached_indices[batch_start + j]
                    embeddings[original_index] = embedding
                    
                    # Cache for 7 days
                    cache_key = f"embedding:{hashlib.md5(chunk.text.encode()).hexdigest()}"
                    await self.cache_manager.set(cache_key, embedding, 'embedding', ttl=604800)
                
                # Update status
                if self.current_status:
                    processed = batch_end
                    total = len(uncached_chunks)
                    self.current_status.progress_percent = (processed / total) * 100
                
                # Check if we need to pause between batches
                if recommendations['pause_between_batches']:
                    await asyncio.sleep(1)  # Brief pause to let system recover
                
                # Force garbage collection after each batch
                import gc
                if recommendations['enable_aggressive_gc']:
                    gc.collect()
                
                # Check memory status and adjust if needed
                check = self.resource_monitor.check_memory_limits()
                if check['memory_critical']:
                    logger.warning("Memory critical during embedding generation - waiting...")
                    await self.resource_monitor.wait_for_memory_available()
                
                if batch_end % 100 == 0:  # Log progress every 100 embeddings
                    logger.info(f"Generated {batch_end}/{len(uncached_chunks)} embeddings (Memory: {self.resource_monitor.current_memory_percent:.1f}%)")
        
        logger.info(f"Embedding generation complete. Cache hit rate: {(len(chunks) - len(uncached_chunks)) / len(chunks) * 100:.1f}%")
        return embeddings
    
    async def _update_search_suggestions(self, book: Book, content_analysis: Dict[str, Any]):
        """Update search suggestions based on book content"""
        if not self.query_suggester:
            return
        
        # Add book title and author as potential queries
        potential_queries = []
        
        if book.title:
            potential_queries.append(book.title)
        
        if book.author:
            potential_queries.append(book.author)
            potential_queries.append(f"{book.author} books")
        
        # Add content-based queries
        if content_analysis.get('code'):
            languages = set()
            for code_item in content_analysis['code']:
                lang = code_item.get('metadata', {}).get('language')
                if lang and lang != 'unknown':
                    languages.add(lang)
                    potential_queries.append(f"{lang} code")
                    potential_queries.append(f"{lang} examples")
            
        if content_analysis.get('formulas'):
            potential_queries.extend([
                "mathematical formulas",
                "trading formulas",
                "calculations"
            ])
        
        if content_analysis.get('strategies'):
            potential_queries.extend([
                "trading strategies",
                "investment strategies",
                "algorithmic trading"
            ])
        
        # Record these as potential successful searches
        for query in potential_queries:
            await self.query_suggester.record_search(query, 1)


# Example usage and testing
async def test_enhanced_processor():
    """Test the enhanced book processor"""
    processor = EnhancedBookProcessor()
    await processor.initialize()
    
    # Test with different file types
    test_files = [
        "data/books/sample.pdf",
        "data/books/sample.epub"
    ]
    
    for test_file in test_files:
        if Path(test_file).exists():
            print(f"\nTesting enhanced processing with: {test_file}")
            
            result = await processor.add_book(
                test_file,
                metadata={
                    'categories': ['testing', 'enhanced'],
                    'description': 'Test book for enhanced processing'
                }
            )
            
            print(f"Processing result: {result}")
            
            if result['success']:
                # Get detailed information
                details = await processor.get_book_details(result['book_id'])
                print(f"Content analysis: {details['content_analysis']}")
        else:
            print(f"Test file not found: {test_file}")
    
    # List all books
    books = await processor.list_books()
    print(f"\nTotal books: {len(books)}")
    
    await processor.cleanup()


if __name__ == "__main__":
    asyncio.run(test_enhanced_processor())