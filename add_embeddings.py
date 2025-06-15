#!/usr/bin/env python3
"""
Add embeddings to existing book in database
"""

import sys
import asyncio
import time

# Add src to path
sys.path.append('src')

from src.core.sqlite_storage import SQLiteStorage
from src.core.qdrant_storage import QdrantStorage
from src.ingestion.local_embeddings import LocalEmbeddingGenerator

async def add_embeddings(book_id: str):
    """Add embeddings for a specific book"""
    
    print(f"🧠 Adding embeddings for book: {book_id}")
    
    # Initialize storage
    sqlite_storage = SQLiteStorage()
    qdrant_storage = QdrantStorage()
    
    # Get book info
    book = await sqlite_storage.get_book(book_id)
    if not book:
        print(f"❌ Book not found: {book_id}")
        return False
    
    print(f"📚 Book: {book.title}")
    print(f"👤 Author: {book.author}")
    
    # Get chunks
    chunks = await sqlite_storage.get_chunks_by_book(book_id)
    if not chunks:
        print("❌ No chunks found")
        return False
    
    print(f"📦 Found {len(chunks)} chunks")
    
    # Generate embeddings
    print("🧠 Generating embeddings...")
    embedding_generator = LocalEmbeddingGenerator()
    
    start_time = time.time()
    
    async with embedding_generator as gen:
        # Process in batches
        batch_size = 16  # Smaller batches for stability
        all_embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = await gen.generate_embeddings(batch_chunks)
            all_embeddings.extend(batch_embeddings)
            
            print(f"  📈 Generated {i+1}-{min(i+batch_size, len(chunks))} of {len(chunks)}")
    
    embedding_time = time.time() - start_time
    print(f"✅ Generated {len(all_embeddings)} embeddings in {embedding_time:.2f}s")
    
    # Store in Qdrant
    print("⚡ Storing in Qdrant...")
    
    # Prepare embeddings for Qdrant
    success = await qdrant_storage.save_embeddings(chunks, all_embeddings)
    
    if success:
        print(f"✅ {len(chunks)} vectors stored in Qdrant")
        print(f"🎉 {book.title} now has semantic search!")
        return True
    else:
        print("❌ Failed to store in Qdrant")
        return False

async def main():
    if len(sys.argv) < 2:
        print("Available books:")
        storage = SQLiteStorage()
        books = await storage.list_books()
        for book in books:
            print(f"  {book.id}: {book.title} ({book.total_chunks} chunks)")
        print("\nUsage: python add_embeddings.py <book_id>")
        sys.exit(1)
    
    book_id = sys.argv[1]
    success = await add_embeddings(book_id)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())