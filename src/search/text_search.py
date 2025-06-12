"""
Full-Text Search Engine for TradeKnowledge
Handles exact text search using SQLite FTS5
"""

import logging
import sqlite3
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import json

from utils.logging import get_logger
from core.config import get_config

logger = get_logger(__name__)

class TextSearchEngine:
    """
    Full-text search engine using SQLite FTS5 for exact text matching
    
    Features:
    - FTS5 full-text search with advanced operators
    - Boolean search (AND, OR, NOT)
    - Phrase search with quotes
    - Wildcard search with *
    - Proximity search with NEAR
    - Fuzzy matching support
    """
    
    def __init__(self, db_path: Optional[str] = None):
        self.config = get_config()
        self.db_path = db_path or self.config.database.sqlite.path
        
        # Search configuration
        self.default_results = self.config.search.default_results
        self.max_results = self.config.search.max_results
        
        # Ensure database exists
        self._ensure_database()
    
    def _ensure_database(self):
        """Ensure database and tables exist"""
        try:
            db_dir = Path(self.db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if tables exist
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='chunks'
                """)
                
                if not cursor.fetchone():
                    logger.warning("Database tables don't exist. Run init_db.py first.")
                    
        except Exception as e:
            logger.error(f"Failed to check database: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Add documents to the text search database
        
        Args:
            documents: List of document dicts with:
                - chunk_id: unique identifier
                - book_id: book identifier
                - chunk_index: chunk position
                - text: text content
                - metadata: additional metadata
                - content_type: type of content
                - boundary_type: chunking boundary type
        
        Returns:
            Success status
        """
        if not documents:
            return True
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Prepare data for insertion
                for doc in documents:
                    # Serialize metadata as JSON
                    metadata_json = json.dumps(doc.get("metadata", {}))
                    
                    # Insert into chunks table
                    cursor.execute("""
                        INSERT OR REPLACE INTO chunks 
                        (id, book_id, chunk_index, text, metadata) 
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        doc["chunk_id"],
                        doc.get("book_id", ""),
                        doc.get("chunk_index", 0),
                        doc["text"],
                        metadata_json
                    ))
                
                conn.commit()
                logger.info(f"Added {len(documents)} documents to text search database")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add documents to text search: {e}")
            return False
    
    def search_exact(self, 
                    query: str,
                    num_results: int = None,
                    filter_books: Optional[List[str]] = None,
                    case_sensitive: bool = False) -> Dict[str, Any]:
        """
        Perform exact text search using FTS5
        
        Args:
            query: Search query with optional FTS5 operators
            num_results: Number of results to return
            filter_books: List of book IDs to filter by
            case_sensitive: Whether search should be case sensitive
            
        Returns:
            Search results with metadata
        """
        start_time = datetime.now()
        
        try:
            # Set defaults
            num_results = min(num_results or self.default_results, self.max_results)
            
            # Prepare the FTS5 query
            fts_query = self._prepare_fts_query(query, case_sensitive)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row  # Access columns by name
                cursor = conn.cursor()
                
                # Build the SQL query
                base_query = """
                    SELECT 
                        c.id as chunk_id,
                        c.book_id,
                        c.chunk_index,
                        c.text,
                        c.metadata,
                        bm25(chunks_fts) as relevance_score,
                        snippet(chunks_fts, 1, '<mark>', '</mark>', '...', 64) as snippet
                    FROM chunks_fts 
                    JOIN chunks c ON c.rowid = chunks_fts.rowid
                    WHERE chunks_fts MATCH ?
                """
                
                params = [fts_query]
                
                # Add book filtering
                if filter_books:
                    placeholders = ','.join(['?' for _ in filter_books])
                    base_query += f" AND c.book_id IN ({placeholders})"
                    params.extend(filter_books)
                
                # Order by relevance and limit results
                base_query += " ORDER BY relevance_score DESC LIMIT ?"
                params.append(num_results)
                
                # Execute search
                cursor.execute(base_query, params)
                rows = cursor.fetchall()
                
                # Process results
                search_results = []
                for row in rows:
                    # Parse metadata
                    try:
                        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                    except json.JSONDecodeError:
                        metadata = {}
                    
                    # Calculate normalized score (BM25 can be negative)
                    raw_score = row["relevance_score"]
                    normalized_score = max(0, min(1, (raw_score + 10) / 20))  # Rough normalization
                    
                    search_results.append({
                        "chunk_id": row["chunk_id"],
                        "text": row["text"],
                        "score": round(normalized_score, 4),
                        "snippet": row["snippet"],
                        "metadata": {
                            "book_id": row["book_id"],
                            "chunk_index": row["chunk_index"],
                            **metadata
                        }
                    })
                
                search_time = (datetime.now() - start_time).total_seconds() * 1000
                
                return {
                    "results": search_results,
                    "total_results": len(search_results),
                    "search_time_ms": int(search_time),
                    "query": query,
                    "fts_query": fts_query,
                    "search_type": "exact"
                }
                
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return {
                "results": [],
                "total_results": 0,
                "search_time_ms": 0,
                "query": query,
                "error": str(e),
                "search_type": "exact"
            }
    
    def _prepare_fts_query(self, query: str, case_sensitive: bool = False) -> str:
        """
        Prepare query for FTS5 with advanced operators
        
        Supports:
        - Phrase search: "exact phrase"
        - Boolean: word1 AND word2, word1 OR word2, NOT word
        - Wildcard: trade* (matches trading, trader, etc.)
        - Proximity: word1 NEAR/5 word2
        """
        # Clean the query
        query = query.strip()
        
        if not query:
            raise ValueError("Query cannot be empty")
        
        # Handle case sensitivity
        if not case_sensitive:
            query = query.lower()
        
        # If query already contains FTS5 operators, use as-is
        fts_operators = ['AND', 'OR', 'NOT', 'NEAR', '"', '*']
        if any(op in query.upper() for op in fts_operators):
            return query
        
        # For simple queries, add wildcard support for partial matching
        words = query.split()
        if len(words) == 1:
            # Single word - add wildcard
            return f"{words[0]}*"
        else:
            # Multiple words - treat as phrase or AND query
            if '"' in query:
                # Already a phrase query
                return query
            else:
                # Convert to AND query with wildcards
                wildcard_words = [f"{word}*" for word in words]
                return " AND ".join(wildcard_words)
    
    def search_phrase(self, phrase: str, num_results: int = None, 
                     filter_books: Optional[List[str]] = None) -> Dict[str, Any]:
        """Search for exact phrase"""
        phrase_query = f'"{phrase}"'
        return self.search_exact(phrase_query, num_results, filter_books)
    
    def search_boolean(self, terms: List[str], operator: str = "AND",
                      num_results: int = None, filter_books: Optional[List[str]] = None) -> Dict[str, Any]:
        """Search with boolean operators"""
        if operator.upper() not in ["AND", "OR"]:
            raise ValueError("Operator must be AND or OR")
        
        boolean_query = f" {operator.upper()} ".join(terms)
        return self.search_exact(boolean_query, num_results, filter_books)
    
    def search_proximity(self, word1: str, word2: str, distance: int = 5,
                        num_results: int = None, filter_books: Optional[List[str]] = None) -> Dict[str, Any]:
        """Search for words within specified distance"""
        proximity_query = f"{word1} NEAR/{distance} {word2}"
        return self.search_exact(proximity_query, num_results, filter_books)
    
    def search_wildcard(self, pattern: str, num_results: int = None,
                       filter_books: Optional[List[str]] = None) -> Dict[str, Any]:
        """Search with wildcard patterns"""
        if not pattern.endswith('*'):
            pattern += '*'
        return self.search_exact(pattern, num_results, filter_books)
    
    def get_chunk_context(self, chunk_id: str, before_chunks: int = 1, 
                         after_chunks: int = 1) -> Dict[str, Any]:
        """
        Get context around a specific chunk
        
        Args:
            chunk_id: ID of the target chunk
            before_chunks: Number of chunks before to include
            after_chunks: Number of chunks after to include
            
        Returns:
            Dict with target chunk and context
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get the target chunk
                cursor.execute("""
                    SELECT id, book_id, chunk_index, text, metadata
                    FROM chunks WHERE id = ?
                """, (chunk_id,))
                
                target_row = cursor.fetchone()
                if not target_row:
                    return {"error": f"Chunk {chunk_id} not found"}
                
                book_id = target_row["book_id"]
                chunk_index = target_row["chunk_index"]
                
                # Get context chunks
                cursor.execute("""
                    SELECT id, chunk_index, text, metadata
                    FROM chunks 
                    WHERE book_id = ? 
                    AND chunk_index BETWEEN ? AND ?
                    ORDER BY chunk_index
                """, (
                    book_id,
                    chunk_index - before_chunks,
                    chunk_index + after_chunks
                ))
                
                context_rows = cursor.fetchall()
                
                # Process results
                target_chunk = {
                    "chunk_id": target_row["id"],
                    "text": target_row["text"],
                    "metadata": json.loads(target_row["metadata"]) if target_row["metadata"] else {}
                }
                
                context_chunks = []
                for row in context_rows:
                    if row["id"] != chunk_id:  # Exclude target chunk from context
                        context_chunks.append({
                            "chunk_id": row["id"],
                            "chunk_index": row["chunk_index"],
                            "text": row["text"],
                            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                            "position": "before" if row["chunk_index"] < chunk_index else "after"
                        })
                
                return {
                    "target_chunk": target_chunk,
                    "context_chunks": context_chunks,
                    "book_id": book_id
                }
                
        except Exception as e:
            logger.error(f"Failed to get chunk context for {chunk_id}: {e}")
            return {"error": str(e)}
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the text search database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total chunks
                cursor.execute("SELECT COUNT(*) FROM chunks")
                total_chunks = cursor.fetchone()[0]
                
                # Unique books
                cursor.execute("SELECT COUNT(DISTINCT book_id) FROM chunks")
                unique_books = cursor.fetchone()[0]
                
                # Average chunk size
                cursor.execute("SELECT AVG(LENGTH(text)) FROM chunks")
                avg_chunk_size = cursor.fetchone()[0] or 0
                
                # Content type distribution (if stored in metadata)
                cursor.execute("""
                    SELECT book_id, COUNT(*) as chunk_count 
                    FROM chunks 
                    GROUP BY book_id 
                    ORDER BY chunk_count DESC
                """)
                book_distribution = cursor.fetchall()
                
                return {
                    "total_chunks": total_chunks,
                    "unique_books": unique_books,
                    "avg_chunk_size": round(avg_chunk_size, 1),
                    "book_distribution": [
                        {"book_id": row[0], "chunks": row[1]} 
                        for row in book_distribution[:10]  # Top 10
                    ],
                    "database_path": self.db_path
                }
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {
                "total_chunks": 0,
                "error": str(e),
                "database_path": self.db_path
            }
    
    def delete_book(self, book_id: str) -> bool:
        """Delete all chunks for a specific book"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete from main table (triggers will handle FTS)
                cursor.execute("DELETE FROM chunks WHERE book_id = ?", (book_id,))
                deleted_count = cursor.rowcount
                
                conn.commit()
                logger.info(f"Deleted {deleted_count} chunks for book {book_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete book {book_id}: {e}")
            return False

# Convenience functions
def search_text(query: str, num_results: int = 10, 
               filter_books: Optional[List[str]] = None) -> Dict[str, Any]:
    """Convenience function for text search"""
    engine = TextSearchEngine()
    return engine.search_exact(query, num_results, filter_books)