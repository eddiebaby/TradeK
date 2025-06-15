#!/usr/bin/env python3
"""
Chunk an already-parsed book with memory optimization
"""

import asyncio
import sys
import time
import gc
from pathlib import Path
from datetime import datetime
import psutil
import os

# Add src to path
sys.path.append('src')

from src.core.sqlite_storage import SQLiteStorage
from src.core.models import Book, Chunk, FileType, ChunkType
from src.ingestion.text_chunker import TextChunker, ChunkingConfig
from src.ingestion.resource_monitor import ResourceMonitor, ResourceLimits

class ChunkingService:
    def __init__(self, book_id: str):
        self.book_id = book_id
        self.storage = SQLiteStorage()
        self.chunker = None
        self.resource_monitor = None
        
    async def initialize(self):
        """Initialize components"""
        print("🔧 Initializing chunking service...")
        
        # Configure resource limits for WSL2
        limits = ResourceLimits(
            max_memory_percent=70.0,  # Even more conservative
            max_memory_mb=1200,       # Limit to 1.2GB
            warning_threshold=55.0,   # Warn earlier
            check_interval=2.0        # Check more frequently
        )
        
        # Initialize resource monitor
        self.resource_monitor = ResourceMonitor(limits)
        await self.resource_monitor.start_monitoring()
        
        # Initialize chunker with smaller config
        self.chunker = TextChunker(ChunkingConfig(
            chunk_size=500,        # Even smaller chunks
            chunk_overlap=60,      # Reduced overlap
            min_chunk_size=50,
            max_chunk_size=1000
        ))
        
        print("✅ Chunking service initialized")
    
    async def chunk_book(self):
        """Chunk an existing book with minimal memory usage"""
        
        # Get book
        book = await self.storage.get_book(self.book_id)
        if not book:
            print(f"❌ Book not found: {self.book_id}")
            return False
        
        print(f"📚 Chunking book: {book.title}")
        print(f"📄 Pages: {book.total_pages}")
        
        # We need to re-parse to get page content, but do it super efficiently
        print("📖 Re-parsing PDF for chunking (memory-optimized)...")
        
        # Import parser here to save memory
        from src.ingestion.pdf_parser import PDFParser
        parser = PDFParser()
        
        # Parse with extreme memory conservation
        parse_result = parser.parse_file(Path(book.file_path))
        
        if parse_result['errors']:
            print(f"❌ Parse errors: {parse_result['errors']}")
            return False
        
        pages = parse_result['pages']
        total_pages = len(pages)
        
        # Process in extremely small batches to avoid memory issues
        batch_size = 5  # Very small batches
        total_chunks_created = 0
        
        print(f"📦 Processing {total_pages} pages in batches of {batch_size}")
        
        for batch_start in range(0, total_pages, batch_size):
            batch_end = min(batch_start + batch_size, total_pages)
            batch_pages = pages[batch_start:batch_end]
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing pages {batch_start+1}-{batch_end} of {total_pages}")
            
            try:
                # Check memory before processing
                check = self.resource_monitor.check_memory_limits()
                if check['memory_critical']:
                    print("🚨 Memory critical - waiting...")
                    await self.resource_monitor.wait_for_memory_available()
                
                # Chunk this small batch
                batch_chunks = self.chunker.chunk_pages(batch_pages, book.id, {})
                
                if batch_chunks:
                    # Save immediately
                    success = await self.storage.save_chunks(batch_chunks)
                    if not success:
                        print("❌ Failed to save batch chunks")
                        return False
                    
                    total_chunks_created += len(batch_chunks)
                    print(f"💾 Saved {len(batch_chunks)} chunks (Total: {total_chunks_created})")
                
                # Aggressive cleanup
                del batch_chunks
                del batch_pages
                gc.collect()
                
                # Brief pause
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"❌ Error processing batch {batch_start}-{batch_end}: {e}")
                continue
        
        # Update book record
        book.total_chunks = total_chunks_created
        book.indexed_at = datetime.now()
        await self.storage.update_book(book)
        
        # Stop monitoring
        await self.resource_monitor.stop_monitoring()
        
        print(f"🎉 Chunking completed!")
        print(f"📦 Total chunks created: {total_chunks_created}")
        
        return True

async def main():
    if len(sys.argv) < 2:
        print("Usage: python chunk_existing_book.py <book_id>")
        print("\nTo find book IDs:")
        print("sqlite3 data/knowledge.db \"SELECT id, title, total_chunks FROM books WHERE total_chunks = 0;\"")
        sys.exit(1)
    
    book_id = sys.argv[1]
    
    # Check memory
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    
    if available_gb < 0.8:
        print(f"⚠️  Warning: Very low memory available ({available_gb:.1f}GB)")
        print("   Consider closing other applications")
    
    print(f"🖥️  Available memory: {available_gb:.1f}GB")
    
    # Run chunking
    service = ChunkingService(book_id)
    await service.initialize()
    
    success = await service.chunk_book()
    
    if success:
        print("🏆 Chunking completed successfully!")
    else:
        print("💥 Chunking failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())