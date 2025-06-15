#!/usr/bin/env python3
"""
Safe PDF ingestion script with memory management and timeouts
Processes regime change PDF with resource limits and resume capability
"""

import asyncio
import sys
import hashlib
import time
import gc
import psutil
import os
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append('src')

from src.core.sqlite_storage import SQLiteStorage
from src.core.models import Book, Chunk, FileType, ChunkType
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.text_chunker import TextChunker, ChunkingConfig

class SafeIngestion:
    def __init__(self):
        self.max_memory_mb = 1024  # 1GB limit
        self.chunk_batch_size = 10  # Process chunks in small batches
        self.timeout_seconds = 300  # 5 minute timeout per operation
        
    def check_memory(self):
        """Check if memory usage is within limits"""
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        if memory_mb > self.max_memory_mb:
            print(f"⚠️  Memory usage {memory_mb:.1f}MB exceeds limit {self.max_memory_mb}MB")
            return False
        return True
    
    async def safe_ingest(self, pdf_path: str):
        """Safe PDF ingestion with memory and timeout limits"""
        
        print(f"🔒 Safe ingestion starting...")
        print(f"📁 File: {pdf_path}")
        print(f"💾 Memory limit: {self.max_memory_mb}MB")
        print(f"⏱️  Timeout: {self.timeout_seconds}s per operation")
        print()
        
        # Initialize storage
        sqlite_storage = SQLiteStorage()
        
        # Check if file exists
        if not Path(pdf_path).exists():
            print(f"❌ File not found: {pdf_path}")
            return False
        
        # Calculate file hash in chunks to avoid memory issues
        print("🔍 Calculating file hash safely...")
        hash_sha256 = hashlib.sha256()
        with open(pdf_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        file_hash = hash_sha256.hexdigest()
        
        # Check if already processed
        existing = await sqlite_storage.get_book_by_hash(file_hash)
        if existing and existing.total_chunks > 0:
            print(f"✅ Book already processed: {existing.title} ({existing.total_chunks} chunks)")
            return True
        
        # Parse PDF with timeout (disable OCR to avoid pytesseract dependency)
        print("📖 Parsing PDF with timeout...")
        pdf_parser = PDFParser(enable_ocr=False)
        
        try:
            start_time = time.time()
            parse_result = await asyncio.wait_for(
                asyncio.to_thread(pdf_parser.parse_file, Path(pdf_path)),
                timeout=self.timeout_seconds
            )
            parse_time = time.time() - start_time
            
            if parse_result['errors']:
                print(f"❌ Parse errors: {parse_result['errors']}")
                return False
                
            print(f"✅ PDF parsed in {parse_time:.2f}s")
            print(f"📄 Found {len(parse_result['pages'])} pages")
            
        except asyncio.TimeoutError:
            print(f"❌ PDF parsing timed out after {self.timeout_seconds}s")
            return False
        
        # Check memory after parsing
        if not self.check_memory():
            print("❌ Memory limit exceeded after parsing")
            return False
        
        # Create book record
        print("📝 Creating book record...")
        metadata = parse_result['metadata']
        title = "Detecting Regime Change in Computational Finance: Data Science, Machine Learning and Algorithmic Trading"
        author = "Jun Chen, Edward Tsang"
        book_id = f"regime_change_{file_hash[:8]}"
        
        book = Book(
            id=book_id,
            title=title,
            author=author,
            file_path=pdf_path,
            file_type=FileType.PDF,
            file_hash=file_hash,
            total_pages=len(parse_result['pages']),
            categories=['regime-change', 'machine-learning', 'computational-finance'],
            metadata={
                **metadata,
                'processing_date': datetime.now().isoformat(),
                'file_size_mb': round(Path(pdf_path).stat().st_size / (1024*1024), 2),
                'safe_ingestion': True
            }
        )
        
        # Save book
        success = await sqlite_storage.save_book(book)
        if not success:
            print("❌ Failed to save book")
            return False
        
        print(f"✅ Book saved: {book.title}")
        
        # Chunk text in batches
        print("✂️  Chunking text safely...")
        chunker = TextChunker(ChunkingConfig(
            chunk_size=800,  # Smaller chunks
            chunk_overlap=100,
            min_chunk_size=50,
            max_chunk_size=1200
        ))
        
        all_chunks = []
        pages = parse_result['pages']
        
        # Process pages in small batches
        page_batch_size = 5
        for i in range(0, len(pages), page_batch_size):
            batch_pages = pages[i:i + page_batch_size]
            batch_chunks = chunker.chunk_pages(batch_pages, book.id, {})
            all_chunks.extend(batch_chunks)
            
            print(f"  📦 Processed pages {i+1}-{min(i+page_batch_size, len(pages))} -> {len(batch_chunks)} chunks")
            
            # Force garbage collection
            gc.collect()
            
            # Check memory
            if not self.check_memory():
                print(f"❌ Memory limit exceeded at page batch {i}")
                return False
        
        print(f"✅ Created {len(all_chunks)} chunks total")
        
        # Save chunks in batches
        print("💾 Saving chunks to SQLite...")
        for i in range(0, len(all_chunks), self.chunk_batch_size):
            batch_chunks = all_chunks[i:i + self.chunk_batch_size]
            success = await sqlite_storage.save_chunks(batch_chunks)
            
            if not success:
                print(f"❌ Failed to save chunk batch {i}")
                return False
            
            print(f"  💾 Saved chunks {i+1}-{min(i+self.chunk_batch_size, len(all_chunks))}")
            
            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.1)
        
        # Update book with final chunk count
        book.total_chunks = len(all_chunks)
        book.indexed_at = datetime.now()
        await sqlite_storage.update_book(book)
        
        print()
        print("🎉 SAFE INGESTION COMPLETE!")
        print("=" * 40)
        print(f"📚 Book: {book.title}")
        print(f"📄 Pages: {book.total_pages}")
        print(f"📦 Chunks: {book.total_chunks}")
        print(f"💾 Storage: SQLite only (embeddings can be added later)")
        print()
        print("Next steps:")
        print("1. Run search tests to verify ingestion")
        print("2. Add embeddings later with separate script")
        
        return True

async def main():
    if len(sys.argv) < 2:
        print("Usage: python safe_ingest.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    ingestion = SafeIngestion()
    success = await ingestion.safe_ingest(pdf_path)
    
    if success:
        print("\n✅ Ingestion completed successfully!")
    else:
        print("\n❌ Ingestion failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())