#!/usr/bin/env python3
"""
Initialize databases for TradeKnowledge
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import sqlite3
import logging
from datetime import datetime
import chromadb
from chromadb.config import Settings

from core.config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_sqlite():
    """Initialize SQLite database with FTS5"""
    config = get_config()
    db_path = Path(config.database.sqlite.path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Initializing SQLite database at {db_path}")
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Create main chunks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            book_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding_id TEXT,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(book_id, chunk_index)
        )
    """)
    
    # Create FTS5 virtual table
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            id UNINDEXED,
            text,
            content=chunks,
            content_rowid=rowid,
            tokenize='porter unicode61'
        )
    """)
    
    # Create triggers to keep FTS in sync
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS chunks_ai 
        AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, id, text) 
            VALUES (new.rowid, new.id, new.text);
        END
    """)
    
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS chunks_ad 
        AFTER DELETE ON chunks BEGIN
            DELETE FROM chunks_fts WHERE rowid = old.rowid;
        END
    """)
    
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS chunks_au 
        AFTER UPDATE ON chunks BEGIN
            DELETE FROM chunks_fts WHERE rowid = old.rowid;
            INSERT INTO chunks_fts(rowid, id, text) 
            VALUES (new.rowid, new.id, new.text);
        END
    """)
    
    # Create books table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS books (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            author TEXT,
            isbn TEXT,
            file_path TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            total_chunks INTEGER DEFAULT 0,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            indexed_at TIMESTAMP,
            UNIQUE(file_hash)
        )
    """)
    
    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_book_id ON chunks(book_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_embedding_id ON chunks(embedding_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_books_file_hash ON books(file_hash)")
    
    # Create search history table for analytics
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            query_type TEXT NOT NULL,
            results_count INTEGER,
            execution_time_ms INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
    
    logger.info("SQLite database initialized successfully")

def init_chromadb():
    """Initialize ChromaDB for vector storage"""
    config = get_config()
    persist_dir = Path(config.database.chroma.persist_directory)
    persist_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Initializing ChromaDB at {persist_dir}")
    
    # Create ChromaDB client
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    
    # Create or get collection
    collection = client.get_or_create_collection(
        name=config.database.chroma.collection_name,
        metadata={
            "description": "Trading and ML book embeddings",
            "created_at": datetime.now().isoformat()
        }
    )
    
    logger.info(f"ChromaDB collection '{config.database.chroma.collection_name}' ready")
    logger.info(f"Current document count: {collection.count()}")

def verify_installation():
    """Verify all components are working"""
    logger.info("Verifying installation...")
    
    # Test SQLite
    config = get_config()
    conn = sqlite3.connect(config.database.sqlite.path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    logger.info(f"SQLite tables: {[t[0] for t in tables]}")
    conn.close()
    
    # Test ChromaDB
    client = chromadb.PersistentClient(path=config.database.chroma.persist_directory)
    collections = client.list_collections()
    logger.info(f"ChromaDB collections: {[c.name for c in collections]}")
    
    logger.info("✅ All components verified successfully!")

def main():
    """Main initialization function"""
    logger.info("Starting database initialization...")
    
    try:
        init_sqlite()
        init_chromadb()
        verify_installation()
        logger.info("✅ Database initialization complete!")
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        raise

if __name__ == "__main__":
    main()