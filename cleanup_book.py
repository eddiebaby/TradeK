#!/usr/bin/env python3
"""
Clean up broken book ingestion
"""

import sys
import asyncio

# Add src to path
sys.path.append('src')

from src.core.sqlite_storage import SQLiteStorage
from src.core.qdrant_storage import QdrantStorage

async def cleanup_book(book_id: str):
    """Clean up all traces of a book"""
    
    print(f"🧹 Cleaning up book: {book_id}")
    
    sqlite_storage = SQLiteStorage()
    qdrant_storage = QdrantStorage()
    
    # Get book info first
    book = await sqlite_storage.get_book(book_id)
    if book:
        print(f"📚 Found book: {book.title}")
        
        # Delete chunks from SQLite
        print("🗑️ Deleting chunks from SQLite...")
        await sqlite_storage.delete_chunks_by_book(book_id)
        
        # Delete embeddings from Qdrant
        print("🗑️ Deleting embeddings from Qdrant...")
        chunks = await sqlite_storage.get_chunks_by_book(book_id)
        if chunks:
            chunk_ids = [chunk.id for chunk in chunks]
            await qdrant_storage.delete_embeddings(chunk_ids)
        
        # Delete book record
        print("🗑️ Deleting book record...")
        await sqlite_storage.delete_book(book_id)
        
        print(f"✅ {book.title} completely removed")
    else:
        print("❌ Book not found")
    
    return True

async def main():
    if len(sys.argv) < 2:
        print("Usage: python cleanup_book.py <book_id>")
        sys.exit(1)
    
    book_id = sys.argv[1]
    await cleanup_book(book_id)

if __name__ == "__main__":
    asyncio.run(main())