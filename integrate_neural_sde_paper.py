#!/usr/bin/env python3
"""
Integration script for Neural SDE paper into TradeKnowledge database.

This script properly integrates the structured Neural SDE paper data
into the TradeKnowledge system with async handling and vector embeddings.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.models import Book, Chunk, FileType, ChunkType
from src.core.sqlite_storage import SQLiteStorage
from src.core.qdrant_storage import QdrantStorage
from src.core.config import get_config
from src.ingestion.local_embeddings import LocalEmbeddingGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FixedSQLiteStorage(SQLiteStorage):
    """Fixed version of SQLiteStorage with proper async initialization"""
    
    def __init__(self, db_path: str = None):
        # Initialize required attributes before calling parent
        self._connection_pool = []
        self._connection_pool_size = 5
        self._pool_lock = asyncio.Lock()
        
        # Call parent initialization
        super().__init__(db_path)
        
        # Disable optimized connections to avoid async context manager issues
        self._use_optimized_connections = False

class NeuralSDEIntegrator:
    """Integrator for Neural SDE paper data"""
    
    def __init__(self):
        self.config = get_config()
        self.storage = None
        self.vector_storage = None
        self.embedding_generator = None
        
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing Neural SDE integrator...")
        
        # Initialize storage
        self.storage = FixedSQLiteStorage(self.config.database.sqlite.path)
        
        # Initialize vector storage
        try:
            self.vector_storage = QdrantStorage(self.config.database.qdrant.collection_name)
            # Test connection
            await self.vector_storage.ensure_collection_exists()
        except Exception as e:
            logger.warning(f"Qdrant not available: {e}")
            self.vector_storage = None
        
        # Initialize embedding generator
        try:
            self.embedding_generator = LocalEmbeddingGenerator(self.config)
            logger.info("Local embedding generator initialized")
        except Exception as e:
            logger.warning(f"Local embeddings not available: {e}")
            self.embedding_generator = None
        
        logger.info("Neural SDE integrator initialized successfully")
    
    async def integrate_paper(self, json_file_path: str, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Integrate the Neural SDE paper from structured JSON data.
        
        Args:
            json_file_path: Path to the neural_sde_tradeknowledge_entry.json file
            force_reprocess: Whether to reprocess if already exists
            
        Returns:
            Integration results
        """
        try:
            # Load the structured paper data
            with open(json_file_path, 'r', encoding='utf-8') as f:
                paper_data = json.load(f)
            
            logger.info(f"Loaded paper data: {paper_data['metadata']['title']}")
            
            # Check if book already exists
            if not force_reprocess:
                # Generate a consistent book ID
                book_id = paper_data['document_id']
                existing_book = await self.storage.get_book(book_id)
                if existing_book:
                    logger.info(f"Paper already exists: {existing_book.title}")
                    return {
                        'success': True,
                        'book_id': existing_book.id,
                        'title': existing_book.title,
                        'message': 'Paper already integrated',
                        'reprocessed': False
                    }
            
            # Create Book object from paper metadata
            book = await self._create_book_from_paper_data(paper_data)
            
            # Save book to database
            await self.storage.save_book(book)
            logger.info(f"Saved book: {book.title}")
            
            # Create Chunk objects from paper chunks
            chunks = await self._create_chunks_from_paper_data(paper_data, book.id)
            
            # Save chunks to database
            await self.storage.save_chunks(chunks)
            logger.info(f"Saved {len(chunks)} chunks")
            
            # Generate embeddings if available
            embeddings_generated = 0
            if self.embedding_generator and self.vector_storage:
                try:
                    logger.info("Generating embeddings...")
                    embeddings = await self._generate_embeddings_for_chunks(chunks)
                    
                    # Save embeddings to vector database
                    success = await self.vector_storage.save_embeddings(chunks, embeddings)
                    if success:
                        embeddings_generated = len(embeddings)
                        logger.info(f"Saved {embeddings_generated} embeddings to vector database")
                    else:
                        logger.warning("Failed to save embeddings to vector database")
                        
                except Exception as e:
                    logger.error(f"Error generating/saving embeddings: {e}")
            else:
                logger.info("Embedding generation skipped (services not available)")
            
            # Update book with final chunk count
            book.total_chunks = len(chunks)
            book.indexed_at = datetime.now()
            await self.storage.update_book(book)
            
            result = {
                'success': True,
                'book_id': book.id,
                'title': book.title,
                'author': ', '.join(paper_data['metadata']['authors']),
                'document_type': paper_data['document_type'],
                'chunks_created': len(chunks),
                'embeddings_generated': embeddings_generated,
                'processing_time': datetime.now().isoformat(),
                'vector_search_enabled': embeddings_generated > 0,
                'content_analysis': {
                    'total_chunks': paper_data['processing_stats']['total_chunks'],
                    'critical_chunks': paper_data['processing_stats']['critical_chunks'],
                    'high_priority_chunks': paper_data['processing_stats']['high_priority_chunks'],
                    'medium_priority_chunks': paper_data['processing_stats']['medium_priority_chunks']
                },
                'reprocessed': force_reprocess
            }
            
            logger.info(f"Successfully integrated Neural SDE paper: {book.title}")
            return result
            
        except Exception as e:
            logger.error(f"Error integrating paper: {e}", exc_info=True)
            return {
                'success': False,
                'error': f'Integration failed: {str(e)}'
            }
    
    async def _create_book_from_paper_data(self, paper_data: Dict[str, Any]) -> Book:
        """Create a Book object from paper metadata"""
        metadata = paper_data['metadata']
        
        # Generate file hash from document content
        import hashlib
        content_for_hash = json.dumps(paper_data, sort_keys=True)
        file_hash = hashlib.sha256(content_for_hash.encode()).hexdigest()
        
        # Create book metadata
        book_metadata = {
            'arxiv_id': metadata.get('arxiv_id'),
            'publication_date': metadata.get('publication_date'),
            'institutions': metadata.get('institutions', []),
            'subject': metadata.get('subject'),
            'keywords': metadata.get('keywords', []),
            'domain': metadata.get('domain'),
            'subdomain': metadata.get('subdomain'),
            'complexity_level': metadata.get('complexity_level'),
            'target_audience': metadata.get('target_audience', []),
            'abstract': metadata.get('abstract'),
            'key_contributions': metadata.get('key_contributions', []),
            'methodologies': metadata.get('methodologies', []),
            'applications': metadata.get('applications', []),
            'technical_details': metadata.get('technical_details', {}),
            'relevance_to_trading': metadata.get('relevance_to_trading', {}),
            'implementation_complexity': metadata.get('implementation_complexity', {}),
            'document_type': paper_data['document_type'],
            'processing_date': paper_data['processing_date'],
            'search_tags': paper_data.get('search_tags', [])
        }
        
        # Create Book object
        book = Book(
            id=paper_data['document_id'],
            title=metadata['title'],
            author=', '.join(metadata.get('authors', [])),
            isbn=metadata.get('arxiv_id'),  # Use arxiv_id as ISBN equivalent
            file_path=paper_data.get('source_file', 'neural_sde_paper.pdf'),
            file_type=FileType.PDF,
            file_hash=file_hash,
            total_pages=len(paper_data.get('chunks', [])),  # Use chunk count as page estimate
            categories=['academic_paper', 'quantitative_finance', 'neural_sde'],
            metadata=book_metadata,
            created_at=datetime.fromisoformat(paper_data['processing_date'].replace('Z', '+00:00')),
            indexed_at=None  # Will be set after processing
        )
        
        return book
    
    async def _create_chunks_from_paper_data(self, paper_data: Dict[str, Any], book_id: str) -> List[Chunk]:
        """Create Chunk objects from paper chunk data"""
        chunks = []
        
        for i, chunk_data in enumerate(paper_data['chunks']):
            # Determine chunk type based on section
            section = chunk_data.get('section', 'content')
            chunk_type = ChunkType.TEXT  # All academic paper chunks are text type
            
            # Create chunk metadata
            chunk_metadata = {
                'section': chunk_data.get('section'),
                'priority': chunk_data.get('priority'),
                'original_chunk_id': chunk_data.get('chunk_id'),
                'section_type': chunk_data.get('metadata', {}).get('section_type'),
                'importance': chunk_data.get('metadata', {}).get('importance'),
                'keywords': chunk_data.get('metadata', {}).get('keywords', []),
                'concepts': chunk_data.get('metadata', {}).get('concepts', []),
                'audience': chunk_data.get('metadata', {}).get('audience'),
                'topic': chunk_data.get('metadata', {}).get('topic'),
                'document_type': paper_data['document_type'],
                'source_document': paper_data['document_id']
            }
            
            # Create Chunk object
            chunk = Chunk(
                id=f"{book_id}_chunk_{i:04d}",
                book_id=book_id,
                chunk_index=i,
                text=chunk_data['content'],
                chunk_type=chunk_type,
                embedding_id=None,  # Will be set during embedding generation
                chapter=chunk_data.get('section'),
                section=chunk_data.get('metadata', {}).get('topic'),
                page_start=None,
                page_end=None,
                previous_chunk_id=f"{book_id}_chunk_{i-1:04d}" if i > 0 else None,
                next_chunk_id=f"{book_id}_chunk_{i+1:04d}" if i < len(paper_data['chunks']) - 1 else None,
                metadata=chunk_metadata,
                created_at=datetime.now()
            )
            
            chunks.append(chunk)
        
        return chunks
    
    async def _generate_embeddings_for_chunks(self, chunks: List[Chunk]) -> List[List[float]]:
        """Generate embeddings for chunks using the local embedding generator"""
        if not self.embedding_generator:
            raise ValueError("Embedding generator not available")
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        
        # Generate embeddings in batches for memory efficiency
        batch_size = self.config.embedding.batch_size
        all_embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            
            batch_embeddings = await self.embedding_generator.generate_embeddings(batch_chunks)
            all_embeddings.extend(batch_embeddings)
            
            # Update embedding IDs
            for j, chunk in enumerate(batch_chunks):
                chunk.embedding_id = chunk.id
        
        logger.info(f"Generated {len(all_embeddings)} embeddings")
        return all_embeddings
    
    async def test_integration(self, book_id: str) -> Dict[str, Any]:
        """Test the integration by performing searches"""
        logger.info(f"Testing integration for book: {book_id}")
        
        # Test text search
        text_results = await self.storage.search_exact(
            "neural SDE bayesian",
            book_ids=[book_id],
            limit=5
        )
        
        # Test vector search if available
        vector_results = []
        if self.vector_storage:
            try:
                vector_results = await self.vector_storage.search_similar(
                    "bayesian calibration neural networks",
                    limit=5,
                    book_ids=[book_id]
                )
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")
        
        # Get book details
        book = await self.storage.get_book(book_id)
        chunks = await self.storage.get_chunks_by_book(book_id)
        
        return {
            'book_found': book is not None,
            'book_title': book.title if book else None,
            'total_chunks': len(chunks),
            'text_search_results': len(text_results),
            'vector_search_results': len(vector_results),
            'sample_text_result': text_results[0]['chunk'].text[:200] + "..." if text_results else None,
            'sample_vector_result': vector_results[0]['chunk'].text[:200] + "..." if vector_results else None
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.storage:
            await self.storage.cleanup()
        if self.vector_storage:
            await self.vector_storage.cleanup()

async def main():
    """Main integration function"""
    integrator = NeuralSDEIntegrator()
    
    try:
        # Initialize
        await integrator.initialize()
        
        # Path to the Neural SDE paper JSON file
        json_file_path = "neural_sde_tradeknowledge_entry.json"
        
        if not Path(json_file_path).exists():
            logger.error(f"Neural SDE paper file not found: {json_file_path}")
            logger.info("Please ensure the neural_sde_tradeknowledge_entry.json file exists in the current directory")
            return
        
        # Integrate the paper
        logger.info("Starting Neural SDE paper integration...")
        result = await integrator.integrate_paper(json_file_path, force_reprocess=True)
        
        if result['success']:
            logger.info("Integration completed successfully!")
            logger.info(f"Book ID: {result['book_id']}")
            logger.info(f"Title: {result['title']}")
            logger.info(f"Chunks created: {result['chunks_created']}")
            logger.info(f"Embeddings generated: {result['embeddings_generated']}")
            logger.info(f"Vector search enabled: {result['vector_search_enabled']}")
            
            # Test the integration
            logger.info("\nTesting integration...")
            test_results = await integrator.test_integration(result['book_id'])
            
            logger.info(f"Book found: {test_results['book_found']}")
            logger.info(f"Total chunks: {test_results['total_chunks']}")
            logger.info(f"Text search results: {test_results['text_search_results']}")
            logger.info(f"Vector search results: {test_results['vector_search_results']}")
            
            if test_results['sample_text_result']:
                logger.info(f"Sample text result: {test_results['sample_text_result']}")
            
            if test_results['sample_vector_result']:
                logger.info(f"Sample vector result: {test_results['sample_vector_result']}")
            
            logger.info("\n✅ Neural SDE paper integration complete and tested!")
            
        else:
            logger.error(f"Integration failed: {result['error']}")
    
    except Exception as e:
        logger.error(f"Error during integration: {e}", exc_info=True)
    
    finally:
        await integrator.cleanup()

if __name__ == "__main__":
    asyncio.run(main())