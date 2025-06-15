#!/usr/bin/env python3
"""
Debug chunk retrieval issue
"""

import sys
import asyncio
import sqlite3

# Add src to path
sys.path.append('src')

from src.core.sqlite_storage import SQLiteStorage

async def debug_chunks():
    """Debug chunk retrieval"""
    
    storage = SQLiteStorage()
    book_id = "academic_f353aa93"
    
    print(f"🔍 Debugging chunks for book: {book_id}")
    
    # Direct SQL query
    print("\n📊 Direct SQL count:")
    async with storage._get_connection() as conn:
        cursor = conn.cursor()
        
        await asyncio.to_thread(
            cursor.execute,
            "SELECT COUNT(*) FROM chunks WHERE book_id = ?",
            (book_id,)
        )
        
        count = await asyncio.to_thread(cursor.fetchone)
        print(f"  Total chunks in DB: {count[0]}")
        
        # Sample chunk IDs
        await asyncio.to_thread(
            cursor.execute,
            "SELECT id, chunk_index, length(text) as text_len FROM chunks WHERE book_id = ? ORDER BY chunk_index LIMIT 10",
            (book_id,)
        )
        
        rows = await asyncio.to_thread(cursor.fetchall)
        print(f"\n📄 First 10 chunks:")
        for row in rows:
            print(f"  {row[0]} (index: {row[1]}, len: {row[2]})")
    
    # Using storage method
    print(f"\n🔧 Using storage method:")
    chunks = await storage.get_chunks_by_book(book_id)
    print(f"  Retrieved chunks: {len(chunks)}")
    
    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"  Chunk {i}: {chunk.id} (index: {chunk.chunk_index}, len: {len(chunk.text)})")
    
    return True

if __name__ == "__main__":
    asyncio.run(debug_chunks())