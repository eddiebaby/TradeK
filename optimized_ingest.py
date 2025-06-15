#!/usr/bin/env python3
"""
Memory-optimized PDF ingestion for TradeKnowledge
Designed for AMD GPU + 6.7GB RAM constraints
"""

import asyncio
import sys
import hashlib
import time
import gc
from pathlib import Path
from datetime import datetime
import json
import psutil
import os

# Add src to path
sys.path.append('src')

from src.core.sqlite_storage import SQLiteStorage
from src.core.models import Book, Chunk, FileType, ChunkType
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.text_chunker import TextChunker, ChunkingConfig
from src.ingestion.resource_monitor import ResourceMonitor, ResourceLimits

class OptimizedIngestion:
    def __init__(self):
        self.storage = None
        self.pdf_parser = None
        self.chunker = None
        self.resource_monitor = None
        self.stats = {
            'start_time': None,
            'pages_processed': 0,
            'chunks_created': 0,
            'memory_peak': 0,
            'processing_time': 0
        }
    
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def log_progress(self, message, force_gc=False):
        """Log progress with memory monitoring"""
        if force_gc:
            gc.collect()
        
        memory_mb = self.get_memory_usage()
        self.stats['memory_peak'] = max(self.stats['memory_peak'], memory_mb)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message} (Memory: {memory_mb:.1f}MB)")
    
    async def initialize(self):
        """Initialize components with memory optimization"""
        self.log_progress("🔧 Initializing components...")
        
        # Configure resource limits for WSL2
        limits = ResourceLimits(
            max_memory_percent=75.0,  # More conservative for WSL2
            max_memory_mb=1500,       # Limit to 1.5GB
            warning_threshold=60.0,   # Warn earlier
            check_interval=3.0        # Check more frequently
        )
        
        # Initialize resource monitor
        self.resource_monitor = ResourceMonitor(limits)
        await self.resource_monitor.start_monitoring()
        
        # Add callback for resource warnings
        async def resource_callback(check):
            if check['memory_warning']:
                self.log_progress(f"⚠️  Memory warning: {check['usage']['system_used_percent']:.1f}%")
            if check['memory_critical']:
                self.log_progress(f"🚨 Memory critical: {check['usage']['system_used_percent']:.1f}% - pausing...")
                await self.resource_monitor.wait_for_memory_available()
                self.log_progress("✅ Memory recovered, resuming...")
        
        self.resource_monitor.add_callback(resource_callback)
        
        # Initialize with smaller cache sizes for memory efficiency
        self.storage = SQLiteStorage()
        self.pdf_parser = PDFParser()
        
        # Optimized chunking config for your system
        self.chunker = TextChunker(ChunkingConfig(
            chunk_size=600,        # Smaller chunks = less memory
            chunk_overlap=80,      # Reduced overlap
            min_chunk_size=50,
            max_chunk_size=1200
        ))
        
        self.log_progress("✅ Components initialized", force_gc=True)
    
    async def process_pdf_batched(self, pdf_path: str, categories: list = None, batch_size: int = None):
        """Process PDF in memory-efficient batches with dynamic sizing"""
        
        self.stats['start_time'] = time.time()
        self.log_progress(f"🚀 Starting optimized ingestion: {pdf_path}")
        
        # Get dynamic batch size based on available memory
        if batch_size is None:
            recommendations = self.resource_monitor.get_processing_recommendations()
            batch_size = recommendations['pdf_batch_size']
            self.log_progress(f"📊 Using dynamic batch size: {batch_size} (Memory: {recommendations['memory_status']})")
        
        # Calculate file hash
        with open(pdf_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        # Check if already processed
        existing = await self.storage.get_book_by_hash(file_hash)
        if existing and existing.total_chunks > 0:
            self.log_progress(f"✅ Book already processed: {existing.title} ({existing.total_chunks} chunks)")
            return True
        elif existing and existing.total_chunks == 0:
            self.log_progress(f"📄 Book partially processed, continuing chunking: {existing.title}")
            # Will reuse existing book record and continue with chunking
        
        # Parse PDF
        self.log_progress("📖 Parsing PDF (this may take a few minutes)...")
        parse_result = self.pdf_parser.parse_file(Path(pdf_path))
        
        if parse_result['errors']:
            self.log_progress(f"❌ Parse errors: {parse_result['errors']}")
            return False
        
        total_pages = len(parse_result['pages'])
        self.log_progress(f"📄 Extracted {total_pages} pages", force_gc=True)
        
        # Create or reuse book record
        if existing:
            book = existing
            self.log_progress(f"📚 Using existing book: {book.title}")
        else:
            book = await self.create_book_record(pdf_path, file_hash, parse_result, categories)
            if not book:
                return False
        
        # Process pages in small batches to manage memory
        total_chunks_created = 0
        
        for batch_start in range(0, total_pages, batch_size):
            batch_end = min(batch_start + batch_size, total_pages)
            batch_pages = parse_result['pages'][batch_start:batch_end]
            
            self.log_progress(f"📦 Processing pages {batch_start+1}-{batch_end} of {total_pages}")
            
            # Chunk this batch
            try:
                batch_chunks = self.chunker.chunk_pages(batch_pages, book.id, {})
                
                if batch_chunks:
                    # Save immediately to avoid memory buildup
                    success = await self.storage.save_chunks(batch_chunks)
                    if not success:
                        self.log_progress("❌ Failed to save batch chunks")
                        return False
                    
                    total_chunks_created += len(batch_chunks)
                    self.stats['chunks_created'] = total_chunks_created
                    self.stats['pages_processed'] = batch_end
                    
                    self.log_progress(f"💾 Saved {len(batch_chunks)} chunks (Total: {total_chunks_created})")
                
                # Check memory status and manage resources
                check = self.resource_monitor.check_memory_limits()
                if check['memory_critical']:
                    self.log_progress("🚨 Memory critical - waiting for recovery...")
                    await self.resource_monitor.wait_for_memory_available()
                
                # Get updated recommendations
                recommendations = self.resource_monitor.get_processing_recommendations()
                if recommendations['enable_aggressive_gc']:
                    gc.collect()
                
                if recommendations['pause_between_batches']:
                    await asyncio.sleep(1)  # Longer pause if memory is constrained
                else:
                    await asyncio.sleep(0.1)  # Brief pause
                
                # Cleanup batch data
                del batch_chunks
                del batch_pages
                
            except Exception as e:
                self.log_progress(f"❌ Error processing batch {batch_start}-{batch_end}: {e}")
                continue
        
        # Update book record with final stats
        book.total_chunks = total_chunks_created
        book.indexed_at = datetime.now()
        await self.storage.update_book(book)
        
        self.stats['processing_time'] = time.time() - self.stats['start_time']
        
        # Stop resource monitoring and get summary
        await self.resource_monitor.stop_monitoring()
        resource_summary = self.resource_monitor.get_usage_summary()
        
        self.log_progress("🎉 PDF processing completed!", force_gc=True)
        self.print_final_stats(book, resource_summary)
        
        return True
    
    async def create_book_record(self, pdf_path, file_hash, parse_result, categories):
        """Create book record"""
        self.log_progress("📝 Creating book record...")
        
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
        
        success = await self.storage.save_book(book)
        if not success:
            self.log_progress("❌ Failed to save book")
            return None
        
        self.log_progress(f"✅ Book saved: {book.title}")
        return book
    
    def print_final_stats(self, book, resource_summary=None):
        """Print final processing statistics"""
        print("\n" + "="*60)
        print("📊 PROCESSING COMPLETE")
        print("="*60)
        print(f"📚 Book: {book.title}")
        print(f"👤 Author: {book.author}")
        print(f"📄 Pages: {book.total_pages}")
        print(f"📦 Chunks: {book.total_chunks}")
        print(f"🏷️  Categories: {', '.join(book.categories) if book.categories else 'None'}")
        print(f"⏱️  Processing time: {self.stats['processing_time']:.1f} seconds")
        print(f"🧠 Peak memory: {self.stats['memory_peak']:.1f} MB")
        print(f"📈 Processing rate: {book.total_pages / self.stats['processing_time']:.1f} pages/sec")
        print(f"🆔 Book ID: {book.id}")
        
        if resource_summary and 'error' not in resource_summary:
            print(f"\n🔧 Resource Usage:")
            memory_stats = resource_summary['memory_stats']
            print(f"   Memory range: {memory_stats['min_percent']:.1f}% - {memory_stats['max_percent']:.1f}%")
            print(f"   Average memory: {memory_stats['avg_percent']:.1f}%")
            print(f"   Final memory: {memory_stats['final_percent']:.1f}%")
            print(f"   Warnings issued: {resource_summary['warnings_issued']}")
            if resource_summary['peak_memory_exceeded']:
                print("   ⚠️  Peak memory limits were exceeded during processing")
        
        print("="*60)
        
        print("\n🔍 Next steps:")
        print("   • Test search: python test_search_real.py")
        print("   • View in DB: sqlite3 data/knowledge.db \"SELECT title, total_chunks FROM books;\"")
        print("   • Run embedding generation (if Ollama is ready)")

async def main():
    if len(sys.argv) < 2:
        print("Usage: python optimized_ingest.py <pdf_path> [category1,category2,...]")
        print("\nExample:")
        print("  python optimized_ingest.py 'data/books/Python_Trading.pdf' 'algorithmic-trading,python'")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    categories = []
    
    if len(sys.argv) > 2:
        categories = [cat.strip() for cat in sys.argv[2].split(',')]
    
    if not Path(pdf_path).exists():
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)
    
    # Check available memory
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    
    if available_gb < 1.0:
        print(f"⚠️  Warning: Low memory available ({available_gb:.1f}GB)")
        print("   Consider closing other applications")
    
    print(f"🖥️  System Info:")
    print(f"   Memory: {memory.total/(1024**3):.1f}GB total, {available_gb:.1f}GB available")
    print(f"   CPU cores: {psutil.cpu_count()}")
    
    # Run optimized ingestion
    ingestion = OptimizedIngestion()
    await ingestion.initialize()
    
    # Let the resource monitor determine the batch size dynamically
    success = await ingestion.process_pdf_batched(pdf_path, categories)
    
    if success:
        print("🏆 Optimized ingestion completed successfully!")
    else:
        print("💥 Ingestion failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())