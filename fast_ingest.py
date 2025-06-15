#!/usr/bin/env python3
"""
Fast PDF ingestion script for TradeKnowledge
Optimized for large files with progress reporting
"""

import asyncio
import sys
import hashlib
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.append('src')

from src.core.sqlite_storage import SQLiteStorage
from src.core.models import Book, Chunk, FileType, ChunkType
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.text_chunker import TextChunker, ChunkingConfig

async def fast_ingest(pdf_path: str, categories: list = None):
    """Fast PDF ingestion with progress reporting"""
    
    print(f"🚀 Fast ingestion of: {pdf_path}")
    
    # Initialize components
    storage = SQLiteStorage()
    pdf_parser = PDFParser()
    chunker = TextChunker(ChunkingConfig(
        chunk_size=800,  # Smaller chunks for faster processing
        chunk_overlap=100,
        min_chunk_size=50,
        max_chunk_size=1500
    ))
    
    # Check if book already exists and has chunks
    with open(pdf_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    existing = await storage.get_book_by_hash(file_hash)
    if existing and existing.total_chunks > 0:
        print(f"✅ Book already fully processed: {existing.title} ({existing.total_chunks} chunks)")
        return True
    
    # Parse PDF with progress
    print("📖 Parsing PDF...")
    parse_result = pdf_parser.parse_file(Path(pdf_path))
    
    if parse_result['errors']:
        print(f"❌ Parse errors: {parse_result['errors']}")
        return False
    
    print(f"📄 Extracted {len(parse_result['pages'])} pages")
    
    # If book exists but no chunks, we can reuse the book record
    if existing:
        book = existing
        print(f"📚 Using existing book record: {book.title}")
    else:
        # Create new book record
        print("📝 Creating book record...")
        metadata = parse_result['metadata']
        
        book_id = f"{metadata.get('title', Path(pdf_path).stem)[:50]}_{file_hash[:8]}".replace(' ', '_')
        
        book = Book(
            id=book_id,
            title=metadata.get('title', Path(pdf_path).stem),
            author=metadata.get('author', 'Unknown'),
            file_path=pdf_path,
            file_type=FileType.PDF,
            file_hash=file_hash,
            total_pages=len(parse_result['pages']),
            categories=categories or [],
            metadata=metadata
        )
        
        # Save book
        success = await storage.save_book(book)
        if not success:
            print("❌ Failed to save book")
            return False
        
        print(f"✅ Book saved: {book.title}")
    
    # Chunk the text with progress reporting
    print("✂️  Chunking text...")
    
    # Process pages in batches for memory efficiency
    all_chunks = []
    batch_size = 50  # Process 50 pages at a time
    
    for i in range(0, len(parse_result['pages']), batch_size):
        batch_pages = parse_result['pages'][i:i + batch_size]
        print(f"📦 Processing pages {i+1}-{min(i+batch_size, len(parse_result['pages']))}...")
        
        # Chunk this batch
        batch_chunks = chunker.chunk_pages(batch_pages, book.id, {})
        all_chunks.extend(batch_chunks)
        
        # Save chunks in smaller batches to avoid memory issues
        if len(all_chunks) >= 100:  # Save every 100 chunks
            print(f"💾 Saving {len(all_chunks)} chunks...")
            success = await storage.save_chunks(all_chunks)
            if not success:
                print("❌ Failed to save chunk batch")
                return False
            all_chunks = []  # Clear for next batch
    
    # Save any remaining chunks
    if all_chunks:
        print(f"💾 Saving final {len(all_chunks)} chunks...")
        success = await storage.save_chunks(all_chunks)
        if not success:
            print("❌ Failed to save final chunks")
            return False
    
    # Count total chunks saved
    book_chunks = await storage.get_chunks_by_book(book.id)
    total_chunks = len(book_chunks)
    
    # Update book with final chunk count
    book.total_chunks = total_chunks
    book.indexed_at = datetime.now()
    await storage.update_book(book)
    
    print(f"🎉 Successfully ingested '{book.title}'")
    print(f"   📊 Pages: {book.total_pages}")
    print(f"   📦 Chunks: {total_chunks}")
    print(f"   🏷️  Categories: {', '.join(book.categories) if book.categories else 'None'}")
    print(f"   🆔 Book ID: {book.id}")
    
    return True

async def main():
    if len(sys.argv) < 2:
        print("Usage: python fast_ingest.py <pdf_path> [category1,category2,...]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    categories = []
    
    if len(sys.argv) > 2:
        categories = [cat.strip() for cat in sys.argv[2].split(',')]
    
    if not Path(pdf_path).exists():
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)
    
    success = await fast_ingest(pdf_path, categories)
    
    if success:
        print("🏆 Fast ingestion completed successfully!")
        
        # Show what we can do next
        print("\n🔍 Next steps:")
        print("   • Test search: sqlite3 data/knowledge.db \"SELECT COUNT(*) FROM chunks;\"")
        print("   • View books: sqlite3 data/knowledge.db \"SELECT title, total_chunks FROM books;\"")
        print("   • Search text: python -c \"import sys; sys.path.append('src'); from src.search.text_search import search_text; print(search_text('trading strategy'))\"")
    else:
        print("💥 Fast ingestion failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())