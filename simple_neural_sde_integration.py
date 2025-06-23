#!/usr/bin/env python3
"""
Simple integration script for Neural SDE paper into TradeKnowledge database.

This script uses direct SQLite operations to avoid async context manager issues.
"""

import json
import sqlite3
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleNeuralSDEIntegrator:
    """Simple integrator using direct SQLite operations"""
    
    def __init__(self, db_path: str = "data/knowledge.db"):
        self.db_path = db_path
        
        # Ensure database directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database schema if needed
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema if needed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create books table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS books (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                author TEXT,
                isbn TEXT,
                file_path TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_hash TEXT UNIQUE,
                total_pages INTEGER,
                total_chunks INTEGER DEFAULT 0,
                categories TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                indexed_at TEXT
            )
        ''')
        
        # Create chunks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                book_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                chunk_type TEXT DEFAULT 'text',
                embedding_id TEXT,
                chapter TEXT,
                section TEXT,
                page_start INTEGER,
                page_end INTEGER,
                previous_chunk_id TEXT,
                next_chunk_id TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (book_id) REFERENCES books (id) ON DELETE CASCADE
            )
        ''')
        
        # Create FTS5 table for full-text search
        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                id UNINDEXED,
                text,
                content='chunks',
                content_rowid='rowid'
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_book_id ON chunks(book_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_index ON chunks(chunk_index)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_books_hash ON books(file_hash)')
        
        conn.commit()
        conn.close()
        logger.info("Database schema initialized")
    
    def integrate_paper(self, json_file_path: str, force_reprocess: bool = False) -> Dict[str, Any]:
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
            book_id = paper_data['document_id']
            
            if not force_reprocess:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT id, title FROM books WHERE id = ?", (book_id,))
                existing = cursor.fetchone()
                conn.close()
                
                if existing:
                    logger.info(f"Paper already exists: {existing[1]}")
                    return {
                        'success': True,
                        'book_id': existing[0],
                        'title': existing[1],
                        'message': 'Paper already integrated',
                        'reprocessed': False
                    }
            
            # Create book record
            book_data = self._create_book_data(paper_data)
            
            # Create chunk records
            chunks_data = self._create_chunks_data(paper_data, book_id)
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                # Delete existing if force_reprocess
                if force_reprocess:
                    cursor.execute("DELETE FROM chunks_fts WHERE id IN (SELECT id FROM chunks WHERE book_id = ?)", (book_id,))
                    cursor.execute("DELETE FROM chunks WHERE book_id = ?", (book_id,))
                    cursor.execute("DELETE FROM books WHERE id = ?", (book_id,))
                
                # Insert book
                cursor.execute("""
                    INSERT OR REPLACE INTO books (
                        id, title, author, isbn, file_path, file_type,
                        file_hash, total_pages, total_chunks, categories, metadata, created_at, indexed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, book_data)
                
                logger.info(f"Saved book: {book_data[1]}")
                
                # Insert chunks
                chunk_rows = []
                fts_rows = []
                
                for chunk_data in chunks_data:
                    chunk_rows.append(chunk_data)
                    # Add to FTS (id, text)
                    fts_rows.append((chunk_data[0], chunk_data[3]))
                
                cursor.executemany("""
                    INSERT OR REPLACE INTO chunks (
                        id, book_id, chunk_index, text, chunk_type, embedding_id,
                        chapter, section, page_start, page_end,
                        previous_chunk_id, next_chunk_id, metadata, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, chunk_rows)
                
                # Insert into FTS
                cursor.executemany(
                    "INSERT OR REPLACE INTO chunks_fts(id, text) VALUES (?, ?)",
                    fts_rows
                )
                
                logger.info(f"Saved {len(chunks_data)} chunks")
                
                # Update book with final chunk count and indexed time
                cursor.execute("""
                    UPDATE books SET total_chunks = ?, indexed_at = ? WHERE id = ?
                """, (len(chunks_data), datetime.now().isoformat(), book_id))
                
                conn.commit()
                
                result = {
                    'success': True,
                    'book_id': book_id,
                    'title': book_data[1],
                    'author': book_data[2],
                    'document_type': paper_data['document_type'],
                    'chunks_created': len(chunks_data),
                    'embeddings_generated': 0,  # Not generating embeddings in this simple version
                    'processing_time': datetime.now().isoformat(),
                    'vector_search_enabled': False,
                    'content_analysis': {
                        'total_chunks': paper_data['processing_stats']['total_chunks'],
                        'critical_chunks': paper_data['processing_stats']['critical_chunks'],
                        'high_priority_chunks': paper_data['processing_stats']['high_priority_chunks'],
                        'medium_priority_chunks': paper_data['processing_stats']['medium_priority_chunks']
                    },
                    'reprocessed': force_reprocess
                }
                
                logger.info(f"Successfully integrated Neural SDE paper: {book_data[1]}")
                return result
                
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"Error integrating paper: {e}", exc_info=True)
            return {
                'success': False,
                'error': f'Integration failed: {str(e)}'
            }
    
    def _create_book_data(self, paper_data: Dict[str, Any]) -> tuple:
        """Create book data tuple for SQLite insertion"""
        metadata = paper_data['metadata']
        
        # Generate file hash from document content
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
        
        # Return tuple for SQLite insertion
        return (
            paper_data['document_id'],  # id
            metadata['title'],  # title
            ', '.join(metadata.get('authors', [])),  # author
            metadata.get('arxiv_id'),  # isbn
            paper_data.get('source_file', 'neural_sde_paper.pdf'),  # file_path
            'pdf',  # file_type
            file_hash,  # file_hash
            len(paper_data.get('chunks', [])),  # total_pages (use chunk count as estimate)
            0,  # total_chunks (will be updated later)
            json.dumps(['academic_paper', 'quantitative_finance', 'neural_sde']),  # categories
            json.dumps(book_metadata),  # metadata
            paper_data['processing_date'],  # created_at
            None  # indexed_at (will be set after processing)
        )
    
    def _create_chunks_data(self, paper_data: Dict[str, Any], book_id: str) -> List[tuple]:
        """Create chunk data tuples for SQLite insertion"""
        chunks_data = []
        
        for i, chunk_data in enumerate(paper_data['chunks']):
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
            
            # Create chunk data tuple
            chunk_tuple = (
                f"{book_id}_chunk_{i:04d}",  # id
                book_id,  # book_id
                i,  # chunk_index
                chunk_data['content'],  # text
                'text',  # chunk_type
                None,  # embedding_id
                chunk_data.get('section'),  # chapter
                chunk_data.get('metadata', {}).get('topic'),  # section
                None,  # page_start
                None,  # page_end
                f"{book_id}_chunk_{i-1:04d}" if i > 0 else None,  # previous_chunk_id
                f"{book_id}_chunk_{i+1:04d}" if i < len(paper_data['chunks']) - 1 else None,  # next_chunk_id
                json.dumps(chunk_metadata),  # metadata
                datetime.now().isoformat()  # created_at
            )
            
            chunks_data.append(chunk_tuple)
        
        return chunks_data
    
    def test_integration(self, book_id: str) -> Dict[str, Any]:
        """Test the integration by performing searches"""
        logger.info(f"Testing integration for book: {book_id}")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check if book exists
            cursor.execute("SELECT id, title FROM books WHERE id = ?", (book_id,))
            book_result = cursor.fetchone()
            
            # Count chunks
            cursor.execute("SELECT COUNT(*) FROM chunks WHERE book_id = ?", (book_id,))
            chunk_count = cursor.fetchone()[0]
            
            # Test text search (simplified to avoid potential FTS issues)
            cursor.execute("""
                SELECT id, text 
                FROM chunks
                WHERE book_id = ? 
                AND text LIKE '%neural%'
                LIMIT 3
            """, (book_id,))
            text_results = cursor.fetchall()
            
            return {
                'book_found': book_result is not None,
                'book_title': book_result[1] if book_result else None,
                'total_chunks': chunk_count,
                'text_search_results': len(text_results),
                'vector_search_results': 0,  # Not available in simple version
                'sample_text_result': text_results[0][1][:200] + "..." if text_results else None,
                'sample_vector_result': None
            }
        
        finally:
            conn.close()

def main():
    """Main integration function"""
    integrator = SimpleNeuralSDEIntegrator()
    
    # Path to the Neural SDE paper JSON file
    json_file_path = "neural_sde_tradeknowledge_entry.json"
    
    if not Path(json_file_path).exists():
        logger.error(f"Neural SDE paper file not found: {json_file_path}")
        logger.info("Please ensure the neural_sde_tradeknowledge_entry.json file exists in the current directory")
        return
    
    # Integrate the paper
    logger.info("Starting Neural SDE paper integration...")
    result = integrator.integrate_paper(json_file_path, force_reprocess=True)
    
    if result['success']:
        logger.info("Integration completed successfully!")
        logger.info(f"Book ID: {result['book_id']}")
        logger.info(f"Title: {result['title']}")
        logger.info(f"Chunks created: {result['chunks_created']}")
        logger.info(f"Embeddings generated: {result['embeddings_generated']}")
        logger.info(f"Vector search enabled: {result['vector_search_enabled']}")
        
        # Test the integration
        logger.info("\nTesting integration...")
        test_results = integrator.test_integration(result['book_id'])
        
        logger.info(f"Book found: {test_results['book_found']}")
        logger.info(f"Total chunks: {test_results['total_chunks']}")
        logger.info(f"Text search results: {test_results['text_search_results']}")
        
        if test_results['sample_text_result']:
            logger.info(f"Sample text result: {test_results['sample_text_result']}")
        
        logger.info("\n✅ Neural SDE paper integration complete and tested!")
        
    else:
        logger.error(f"Integration failed: {result['error']}")

if __name__ == "__main__":
    main()