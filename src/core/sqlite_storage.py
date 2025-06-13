"""
SQLite storage implementation for TradeKnowledge

This provides persistent storage for books and chunks,
with full-text search capabilities using FTS5.
"""

import sqlite3
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import asyncio
from contextlib import asynccontextmanager

from core.interfaces import BookStorageInterface, ChunkStorageInterface
from core.models import Book, Chunk, FileType, ChunkType

logger = logging.getLogger(__name__)

class SQLiteStorage(BookStorageInterface, ChunkStorageInterface):
    """
    SQLite implementation of storage interfaces.
    
    This class provides:
    - Book metadata storage
    - Chunk text storage with FTS5 search
    - Transaction support
    - Connection pooling
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize SQLite storage"""
        self.db_path = db_path or "data/knowledge.db"
        
        # Ensure database directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
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
        logger.info("Database initialized")
    
    @asynccontextmanager
    async def _get_connection(self):
        """Get database connection (async context manager)"""
        # Create a new connection for each operation to avoid threading issues
        conn = await asyncio.to_thread(
            sqlite3.connect, 
            self.db_path,
            check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        
        try:
            yield conn
        finally:
            await asyncio.to_thread(conn.close)
    
    # Book Storage Methods
    
    async def save_book(self, book: Book) -> bool:
        """Save a book's metadata"""
        try:
            async with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Convert metadata to JSON
                metadata_json = json.dumps(book.metadata)
                categories_json = json.dumps(book.categories)
                
                # Insert or replace
                await asyncio.to_thread(
                    cursor.execute,
                    """
                    INSERT OR REPLACE INTO books (
                        id, title, author, isbn, file_path, file_type,
                        file_hash, total_pages, total_chunks, categories, metadata, created_at, indexed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        book.id,
                        book.title,
                        book.author,
                        book.isbn,
                        book.file_path,
                        book.file_type.value,
                        book.file_hash,
                        book.total_pages,
                        book.total_chunks,
                        categories_json,
                        metadata_json,
                        book.created_at.isoformat(),
                        book.indexed_at.isoformat() if book.indexed_at else None
                    )
                )
                
                await asyncio.to_thread(conn.commit)
                logger.info(f"Saved book: {book.id} - {book.title}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving book: {e}")
            return False
    
    async def get_book(self, book_id: str) -> Optional[Book]:
        """Retrieve a book by ID"""
        try:
            async with self._get_connection() as conn:
                cursor = conn.cursor()
                
                await asyncio.to_thread(
                    cursor.execute,
                    "SELECT * FROM books WHERE id = ?",
                    (book_id,)
                )
                
                row = await asyncio.to_thread(cursor.fetchone)
                if row:
                    return self._row_to_book(row)
                
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving book: {e}")
            return None
    
    async def get_book_by_hash(self, file_hash: str) -> Optional[Book]:
        """Retrieve a book by file hash"""
        try:
            async with self._get_connection() as conn:
                cursor = conn.cursor()
                
                await asyncio.to_thread(
                    cursor.execute,
                    "SELECT * FROM books WHERE file_hash = ?",
                    (file_hash,)
                )
                
                row = await asyncio.to_thread(cursor.fetchone)
                if row:
                    return self._row_to_book(row)
                
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving book by hash: {e}")
            return None
    
    async def list_books(self, 
                        category: Optional[str] = None,
                        limit: int = 100,
                        offset: int = 0) -> List[Book]:
        """List books with optional filtering"""
        try:
            async with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if category:
                    # Search in categories JSON
                    query = """
                        SELECT * FROM books 
                        WHERE categories LIKE ?
                        ORDER BY created_at DESC
                        LIMIT ? OFFSET ?
                    """
                    params = (f'%{category}%', limit, offset)
                else:
                    query = """
                        SELECT * FROM books
                        ORDER BY created_at DESC  
                        LIMIT ? OFFSET ?
                    """
                    params = (limit, offset)
                
                await asyncio.to_thread(cursor.execute, query, params)
                
                rows = await asyncio.to_thread(cursor.fetchall)
                return [self._row_to_book(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error listing books: {e}")
            return []
    
    async def update_book(self, book: Book) -> bool:
        """Update book metadata"""
        # Same as save_book with INSERT OR REPLACE
        return await self.save_book(book)
    
    async def delete_book(self, book_id: str) -> bool:
        """Delete a book and all its chunks"""
        try:
            async with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Delete chunks first (foreign key constraint)
                await asyncio.to_thread(
                    cursor.execute,
                    "DELETE FROM chunks WHERE book_id = ?",
                    (book_id,)
                )
                
                # Delete book
                await asyncio.to_thread(
                    cursor.execute,
                    "DELETE FROM books WHERE id = ?",
                    (book_id,)
                )
                
                await asyncio.to_thread(conn.commit)
                logger.info(f"Deleted book and chunks: {book_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting book: {e}")
            return False
    
    # Chunk Storage Methods
    
    async def save_chunks(self, chunks: List[Chunk]) -> bool:
        """Save multiple chunks efficiently"""
        if not chunks:
            return True
        
        try:
            async with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Prepare data
                chunk_data = []
                fts_data = []
                for chunk in chunks:
                    metadata_json = json.dumps(chunk.metadata)
                    chunk_data.append((
                        chunk.id,
                        chunk.book_id,
                        chunk.chunk_index,
                        chunk.text,
                        chunk.chunk_type.value,
                        chunk.embedding_id,
                        chunk.chapter,
                        chunk.section,
                        chunk.page_start,
                        chunk.page_end,
                        chunk.previous_chunk_id,
                        chunk.next_chunk_id,
                        metadata_json,
                        chunk.created_at.isoformat()
                    ))
                    
                    # FTS data
                    fts_data.append((chunk.id, chunk.text))
                
                # Batch insert chunks
                await asyncio.to_thread(
                    cursor.executemany,
                    """
                    INSERT OR REPLACE INTO chunks (
                        id, book_id, chunk_index, text, chunk_type, embedding_id,
                        chapter, section, page_start, page_end,
                        previous_chunk_id, next_chunk_id, metadata, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    chunk_data
                )
                
                # Update FTS index
                await asyncio.to_thread(
                    cursor.executemany,
                    "INSERT OR REPLACE INTO chunks_fts(id, text) VALUES (?, ?)",
                    fts_data
                )
                
                await asyncio.to_thread(conn.commit)
                logger.info(f"Saved {len(chunks)} chunks")
                return True
                
        except Exception as e:
            logger.error(f"Error saving chunks: {e}")
            return False
    
    async def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Retrieve a single chunk"""
        try:
            async with self._get_connection() as conn:
                cursor = conn.cursor()
                
                await asyncio.to_thread(
                    cursor.execute,
                    "SELECT * FROM chunks WHERE id = ?",
                    (chunk_id,)
                )
                
                row = await asyncio.to_thread(cursor.fetchone)
                if row:
                    return self._row_to_chunk(row)
                
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving chunk: {e}")
            return None
    
    async def get_chunks_by_book(self, book_id: str) -> List[Chunk]:
        """Get all chunks for a book"""
        try:
            async with self._get_connection() as conn:
                cursor = conn.cursor()
                
                await asyncio.to_thread(
                    cursor.execute,
                    """
                    SELECT * FROM chunks 
                    WHERE book_id = ?
                    ORDER BY chunk_index
                    """,
                    (book_id,)
                )
                
                rows = await asyncio.to_thread(cursor.fetchall)
                return [self._row_to_chunk(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error retrieving chunks by book: {e}")
            return []
    
    async def get_chunk_context(self, 
                               chunk_id: str,
                               before: int = 1,
                               after: int = 1) -> Dict[str, Any]:
        """Get a chunk with surrounding context"""
        try:
            async with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get the target chunk
                await asyncio.to_thread(
                    cursor.execute,
                    "SELECT * FROM chunks WHERE id = ?",
                    (chunk_id,)
                )
                
                target_row = await asyncio.to_thread(cursor.fetchone)
                if not target_row:
                    return {}
                
                target_chunk = self._row_to_chunk(target_row)
                
                # Get surrounding chunks
                await asyncio.to_thread(
                    cursor.execute,
                    """
                    SELECT * FROM chunks
                    WHERE book_id = ? 
                    AND chunk_index >= ? 
                    AND chunk_index <= ?
                    ORDER BY chunk_index
                    """,
                    (
                        target_chunk.book_id,
                        target_chunk.chunk_index - before,
                        target_chunk.chunk_index + after
                    )
                )
                
                rows = await asyncio.to_thread(cursor.fetchall)
                chunks = [self._row_to_chunk(row) for row in rows]
                
                # Build context
                context = {
                    'chunk': target_chunk,
                    'before': [],
                    'after': []
                }
                
                for chunk in chunks:
                    if chunk.chunk_index < target_chunk.chunk_index:
                        context['before'].append(chunk)
                    elif chunk.chunk_index > target_chunk.chunk_index:
                        context['after'].append(chunk)
                
                return context
                
        except Exception as e:
            logger.error(f"Error getting chunk context: {e}")
            return {}
    
    async def search_exact(self,
                          query: str,
                          book_ids: Optional[List[str]] = None,
                          limit: int = 10) -> List[Dict[str, Any]]:
        """Perform exact text search using FTS5"""
        try:
            async with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Build query
                if book_ids:
                    # Filter by book IDs
                    placeholders = ','.join('?' * len(book_ids))
                    fts_query = f"""
                        SELECT c.*, snippet(chunks_fts, 1, '<mark>', '</mark>', '...', 20) as snippet,
                               rank as score
                        FROM chunks_fts 
                        JOIN chunks c ON chunks_fts.id = c.id
                        WHERE chunks_fts MATCH ? 
                        AND c.book_id IN ({placeholders})
                        ORDER BY rank
                        LIMIT ?
                    """
                    params = [query] + book_ids + [limit]
                else:
                    fts_query = """
                        SELECT c.*, snippet(chunks_fts, 1, '<mark>', '</mark>', '...', 20) as snippet,
                               rank as score
                        FROM chunks_fts
                        JOIN chunks c ON chunks_fts.id = c.id
                        WHERE chunks_fts MATCH ?
                        ORDER BY rank
                        LIMIT ?
                    """
                    params = [query, limit]
                
                await asyncio.to_thread(cursor.execute, fts_query, params)
                
                rows = await asyncio.to_thread(cursor.fetchall)
                results = []
                
                for row in rows:
                    chunk = self._row_to_chunk(row)
                    results.append({
                        'chunk': chunk,
                        'score': -row['score'],  # FTS5 rank is negative
                        'snippet': row['snippet']
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Error in exact search: {e}")
            return []
    
    async def delete_chunks_by_book(self, book_id: str) -> bool:
        """Delete all chunks for a book"""
        try:
            async with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Delete from FTS first
                await asyncio.to_thread(
                    cursor.execute,
                    "DELETE FROM chunks_fts WHERE id IN (SELECT id FROM chunks WHERE book_id = ?)",
                    (book_id,)
                )
                
                # Delete chunks
                await asyncio.to_thread(
                    cursor.execute,
                    "DELETE FROM chunks WHERE book_id = ?",
                    (book_id,)
                )
                
                await asyncio.to_thread(conn.commit)
                return True
                
        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")
            return False
    
    # Helper methods
    
    def _row_to_book(self, row: sqlite3.Row) -> Book:
        """Convert database row to Book object"""
        return Book(
            id=row['id'],
            title=row['title'],
            author=row['author'],
            isbn=row['isbn'],
            file_path=row['file_path'],
            file_type=FileType(row['file_type']),
            file_hash=row['file_hash'],
            total_pages=row['total_pages'],
            total_chunks=row['total_chunks'],
            categories=json.loads(row['categories']) if row['categories'] else [],
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            created_at=datetime.fromisoformat(row['created_at']),
            indexed_at=datetime.fromisoformat(row['indexed_at']) if row['indexed_at'] else None
        )
    
    def _row_to_chunk(self, row: sqlite3.Row) -> Chunk:
        """Convert database row to Chunk object"""
        metadata = json.loads(row['metadata']) if row['metadata'] else {}
        
        return Chunk(
            id=row['id'],
            book_id=row['book_id'],
            chunk_index=row['chunk_index'],
            text=row['text'],
            chunk_type=ChunkType(row['chunk_type']) if row['chunk_type'] else ChunkType.TEXT,
            embedding_id=row['embedding_id'],
            chapter=row['chapter'],
            section=row['section'],
            page_start=row['page_start'],
            page_end=row['page_end'],
            previous_chunk_id=row['previous_chunk_id'],
            next_chunk_id=row['next_chunk_id'],
            metadata=metadata,
            created_at=datetime.fromisoformat(row['created_at'])
        )

# Test the storage
async def test_storage():
    """Test SQLite storage implementation"""
    storage = SQLiteStorage("data/test.db")
    
    # Test book operations
    book = Book(
        id="test-001",
        title="Test Book",
        author="Test Author",
        file_path="/tmp/test.pdf",
        file_type=FileType.PDF,
        file_hash="testhash123"
    )
    
    # Save book
    success = await storage.save_book(book)
    print(f"Save book: {success}")
    
    # Retrieve book
    retrieved = await storage.get_book("test-001")
    print(f"Retrieved: {retrieved.title if retrieved else 'Not found'}")
    
    # Test chunk operations
    chunks = [
        Chunk(
            book_id="test-001",
            chunk_index=i,
            text=f"This is test chunk {i} about trading strategies"
        )
        for i in range(5)
    ]
    
    # Save chunks
    success = await storage.save_chunks(chunks)
    print(f"Save chunks: {success}")
    
    # Search
    results = await storage.search_exact("trading", limit=3)
    print(f"Search results: {len(results)}")
    
    for result in results:
        print(f"  - {result['chunk'].text[:50]}... (score: {result['score']:.3f})")

if __name__ == "__main__":
    asyncio.run(test_storage())