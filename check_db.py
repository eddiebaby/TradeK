#!/usr/bin/env python3
"""
Check SQLite database contents
"""

import sys
import asyncio

# Add src to path
sys.path.append('src')

from src.core.sqlite_storage import SQLiteStorage

async def check_database():
    """Check what's in the SQLite database"""
    
    storage = SQLiteStorage()
    
    # Get all books
    print("📚 Books in database:")
    books = await storage.list_books()
    
    if not books:
        print("❌ No books found")
        return False
    
    for book in books:
        print(f"\n📖 {book.title}")
        print(f"  ID: {book.id}")
        print(f"  Author: {book.author}")
        print(f"  Pages: {book.total_pages}")
        print(f"  Chunks: {book.total_chunks}")
        print(f"  Categories: {', '.join(book.categories)}")
        print(f"  File: {book.file_path}")
        
        # Get some sample chunks
        chunks = await storage.get_chunks_by_book(book.id)
        print(f"  📦 Retrieved {len(chunks)} chunks from DB")
        
        if chunks:
            print(f"  📄 Sample chunk text (first 150 chars):")
            sample_text = chunks[0].text[:150] + "..." if len(chunks[0].text) > 150 else chunks[0].text
            print(f"    {sample_text}")
    
    return True

if __name__ == "__main__":
    asyncio.run(check_database())