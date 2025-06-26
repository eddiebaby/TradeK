#!/usr/bin/env python3
"""
Minimal PDF ingestion - SQLite only
"""

import sys
import hashlib
import time
import asyncio
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append('src')

from src.core.sqlite_storage import SQLiteStorage
from src.core.models import Book, Chunk, FileType, ChunkType
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.text_chunker import TextChunker, ChunkingConfig

async def minimal_ingest(pdf_path: str):
    """Minimal PDF ingestion"""
    
    print(f"📁 Ingesting: {pdf_path}")
    
    if not Path(pdf_path).exists():
        print(f"❌ File not found: {pdf_path}")
        return False
    
    file_size_mb = Path(pdf_path).stat().st_size / (1024*1024)
    print(f"📊 File size: {file_size_mb:.2f}MB")
    
    # Calculate file hash
    print("🔍 Calculating hash...")
    hash_sha256 = hashlib.sha256()
    with open(pdf_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    file_hash = hash_sha256.hexdigest()
    
    # Initialize storage
    storage = SQLiteStorage()
    
    # Check if already exists
    existing = await storage.get_book_by_hash(file_hash)
    if existing and existing.total_chunks > 0:
        print(f"✅ Already exists: {existing.title} ({existing.total_chunks} chunks)")
        return True
    
    # Parse PDF
    print("📖 Parsing PDF...")
    pdf_parser = PDFParser(enable_ocr=False)
    
    try:
        parse_result = pdf_parser.parse_file(Path(pdf_path))
        print(f"✅ Found {len(parse_result['pages'])} pages")
        
        if parse_result['errors']:
            print(f"⚠️  Errors: {parse_result['errors']}")
            
    except Exception as e:
        print(f"❌ Parse failed: {e}")
        return False
    
    # Create book
    filename = Path(pdf_path).stem
    title = filename.replace('_', ' ').replace('-', ' ').title()
    book_id = f"book_{file_hash[:8]}"
    
    book = Book(
        id=book_id,
        title=title,
        author="Unknown",
        file_path=pdf_path,
        file_type=FileType.PDF,
        file_hash=file_hash,
        total_pages=len(parse_result['pages']),
        categories=['finance'],
        metadata={
            'processing_date': datetime.now().isoformat(),
            'file_size_mb': file_size_mb
        }
    )
    
    print(f"📚 Book: {book.title}")
    
    # Save book
    success = await storage.save_book(book)
    if not success:
        print("❌ Failed to save book")
        return False
    
    # Chunk text
    print("✂️  Chunking...")
    chunker = TextChunker(ChunkingConfig(
        chunk_size=800,
        chunk_overlap=100
    ))
    
    chunks = chunker.chunk_pages(parse_result['pages'], book.id, {})
    print(f"✅ Created {len(chunks)} chunks")
    
    # Save chunks
    success = await storage.save_chunks(chunks)
    if not success:
        print("❌ Failed to save chunks")
        return False
    
    # Update book
    book.total_chunks = len(chunks)
    book.indexed_at = datetime.now()
    await storage.update_book(book)
    
    print(f"✅ Saved {len(chunks)} chunks")
    print(f"🎉 Complete! Book: {book.title}")
    
    return True

async def main():
    if len(sys.argv) < 2:
        print("Usage: python minimal_ingest.py <pdf_path>")
        sys.exit(1)
    
    success = await minimal_ingest(sys.argv[1])
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())