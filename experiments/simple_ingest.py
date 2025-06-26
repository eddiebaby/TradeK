#!/usr/bin/env python3
"""
Simple PDF ingestion script for TradeKnowledge
This bypasses the full pipeline to do a basic text-only ingestion
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

async def simple_ingest(pdf_path: str, categories: list = None):
    """Simple PDF ingestion without embeddings"""
    
    print(f"🔄 Starting ingestion of: {pdf_path}")
    
    # Initialize components
    storage = SQLiteStorage()
    pdf_parser = PDFParser()
    chunker = TextChunker(ChunkingConfig(
        chunk_size=1000,
        chunk_overlap=200,
        min_chunk_size=100,
        max_chunk_size=2000
    ))
    
    # Calculate file hash
    with open(pdf_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    # Check if already exists
    existing = await storage.get_book_by_hash(file_hash)
    if existing:
        print(f"⚠️  Book already exists: {existing.title}")
        return False
    
    # Parse PDF
    print("📖 Parsing PDF...")
    parse_result = pdf_parser.parse_file(Path(pdf_path))
    
    if parse_result['errors']:
        print(f"❌ Parse errors: {parse_result['errors']}")
        return False
    
    # Create book record
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
    
    # Chunk the text
    print("✂️  Chunking text...")
    chunks = chunker.chunk_pages(parse_result['pages'], book.id, {})
    
    print(f"📦 Created {len(chunks)} chunks")
    
    # Save chunks
    print("💾 Saving chunks...")
    success = await storage.save_chunks(chunks)
    
    if success:
        # Update book with chunk count
        book.total_chunks = len(chunks)
        book.indexed_at = datetime.now()
        await storage.update_book(book)
        
        print(f"✅ Successfully ingested '{book.title}'")
        print(f"   📊 Pages: {book.total_pages}")
        print(f"   📦 Chunks: {len(chunks)}")
        print(f"   🏷️  Categories: {', '.join(book.categories) if book.categories else 'None'}")
        return True
    else:
        print("❌ Failed to save chunks")
        return False

async def main():
    if len(sys.argv) < 2:
        print("Usage: python simple_ingest.py <pdf_path> [category1,category2,...]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    categories = []
    
    if len(sys.argv) > 2:
        categories = [cat.strip() for cat in sys.argv[2].split(',')]
    
    if not Path(pdf_path).exists():
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)
    
    success = await simple_ingest(pdf_path, categories)
    
    if success:
        print("🎉 Ingestion completed successfully!")
    else:
        print("💥 Ingestion failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())