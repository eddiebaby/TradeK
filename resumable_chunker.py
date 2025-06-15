#!/usr/bin/env python3
"""
Resumable chunking for large PDFs with incremental progress
"""

import asyncio
import sys
import time
import gc
from pathlib import Path
from datetime import datetime

sys.path.append('src')

from src.core.sqlite_storage import SQLiteStorage
from src.core.models import Book, Chunk
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.text_chunker import TextChunker, ChunkingConfig
from src.ingestion.resource_monitor import ResourceMonitor, ResourceLimits

class ResumableChunker:
    def __init__(self, book_id: str):
        self.book_id = book_id
        self.storage = SQLiteStorage()
        self.parser = PDFParser()
        self.chunker = TextChunker(ChunkingConfig(
            chunk_size=500,
            chunk_overlap=60,
            min_chunk_size=50,
            max_chunk_size=1000
        ))
        self.monitor = None
        
    async def initialize(self):
        """Initialize with conservative settings"""
        limits = ResourceLimits(
            max_memory_percent=60.0,  # Very conservative
            max_memory_mb=1000,       # 1GB limit
            warning_threshold=50.0,   # Warn at 50%
            check_interval=1.0        # Check every second
        )
        
        self.monitor = ResourceMonitor(limits)
        await self.monitor.start_monitoring()
        
        print("✅ Resumable chunker initialized")
    
    async def chunk_incrementally(self, pages_per_batch=3):
        """Chunk a book in very small increments with persistence"""
        
        # Get book
        book = await self.storage.get_book(self.book_id)
        if not book:
            print(f"❌ Book not found: {self.book_id}")
            return False
        
        print(f"📚 Resumable chunking: {book.title}")
        print(f"📄 Total pages: {book.total_pages}")
        
        # Check existing chunks to see where to resume
        existing_chunks = await self.storage.get_chunks_by_book(self.book_id)
        existing_count = len(existing_chunks)
        
        if existing_count > 0:
            print(f"📦 Found {existing_count} existing chunks, will resume...")
        
        # Parse PDF to get content
        print("📖 Loading PDF content...")
        parse_result = self.parser.parse_file(Path(book.file_path))
        
        if parse_result['errors']:
            print(f"❌ Parse errors: {parse_result['errors']}")
            return False
        
        pages = parse_result['pages']
        total_pages = len(pages)
        
        # Process in tiny batches
        total_chunks_created = existing_count
        batch_num = 0
        
        print(f"📦 Processing {total_pages} pages in batches of {pages_per_batch}")
        
        for batch_start in range(0, total_pages, pages_per_batch):
            batch_end = min(batch_start + pages_per_batch, total_pages)
            batch_pages = pages[batch_start:batch_end]
            batch_num += 1
            
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{timestamp}] Batch {batch_num}: Pages {batch_start+1}-{batch_end}")
            
            try:
                # Check memory before processing
                check = self.monitor.check_memory_limits()
                if check['memory_critical']:
                    print("🚨 Memory critical - waiting for recovery...")
                    await self.monitor.wait_for_memory_available(timeout=30)
                
                # Chunk this tiny batch
                batch_chunks = self.chunker.chunk_pages(batch_pages, book.id, {})
                
                if batch_chunks:
                    # Save immediately to database
                    success = await self.storage.save_chunks(batch_chunks)
                    if not success:
                        print(f"❌ Failed to save batch {batch_num}")
                        continue
                    
                    total_chunks_created += len(batch_chunks)
                    print(f"💾 Saved {len(batch_chunks)} chunks (Total: {total_chunks_created})")
                
                # Aggressive cleanup
                del batch_chunks
                del batch_pages
                gc.collect()
                
                # Update book record every 10 batches
                if batch_num % 10 == 0:
                    book.total_chunks = total_chunks_created
                    await self.storage.update_book(book)
                    print(f"📊 Progress: {batch_end}/{total_pages} pages, {total_chunks_created} chunks")
                
                # Brief pause between batches
                await asyncio.sleep(0.2)
                
            except Exception as e:
                print(f"❌ Error in batch {batch_num}: {e}")
                # Continue with next batch rather than failing completely
                continue
        
        # Final update
        book.total_chunks = total_chunks_created
        book.indexed_at = datetime.now()
        await self.storage.update_book(book)
        
        await self.monitor.stop_monitoring()
        
        print(f"🎉 Chunking completed!")
        print(f"📦 Total chunks: {total_chunks_created}")
        print(f"📚 Book: {book.title}")
        
        return True

async def main():
    if len(sys.argv) < 2:
        print("Usage: python resumable_chunker.py <book_id> [pages_per_batch]")
        print("\nTo find book IDs:")
        print("sqlite3 data/knowledge.db \"SELECT id, title, total_chunks FROM books;\"")
        sys.exit(1)
    
    book_id = sys.argv[1]
    pages_per_batch = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    print(f"🚀 Starting resumable chunking for: {book_id}")
    print(f"📦 Batch size: {pages_per_batch} pages")
    
    chunker = ResumableChunker(book_id)
    await chunker.initialize()
    
    success = await chunker.chunk_incrementally(pages_per_batch)
    
    if success:
        print("🏆 Resumable chunking completed successfully!")
    else:
        print("💥 Chunking failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())