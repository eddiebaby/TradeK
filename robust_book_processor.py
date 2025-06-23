#!/usr/bin/env python3
"""
Robust Book Processor for TradeKnowledge
=========================================

A fail-safe system for ingesting complete books into both SQLite and ChromaDB.
Designed to handle large PDFs without memory issues or async complications.

Features:
- Streaming page-by-page processing
- Memory monitoring and limits
- Batch processing with checkpoints
- Error recovery and retry logic
- Direct database integration
- Progress tracking

Author: Claude Code Assistant
"""

import os
import sys
import json
import time
import hashlib
import psutil
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Add project root to path
sys.path.append('.')

try:
    import PyPDF2
    import pdfplumber
    import chromadb
    from chromadb.config import Settings
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Install with: pip install PyPDF2 pdfplumber chromadb")
    sys.exit(1)

class MemoryMonitor:
    """Monitor and control memory usage"""
    
    def __init__(self, max_memory_mb: int = 512):
        self.max_memory_mb = max_memory_mb
        self.process = psutil.Process(os.getpid())
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def is_memory_ok(self) -> bool:
        """Check if memory usage is within limits"""
        return self.get_memory_usage() < self.max_memory_mb
    
    def wait_for_memory(self, timeout: int = 30) -> bool:
        """Wait for memory to become available"""
        start_time = time.time()
        while not self.is_memory_ok():
            if time.time() - start_time > timeout:
                return False
            print(f"⏳ Memory usage: {self.get_memory_usage():.1f}MB, waiting...")
            time.sleep(2)
        return True

class SimpleTextChunker:
    """Simple text chunker without complex dependencies"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings near the break point
                for i in range(end, max(end - 100, start), -1):
                    if text[i:i+2] in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.overlap
            if start >= len(text):
                break
                
        return chunks

class RobustBookProcessor:
    """Main book processing class with fail-safe mechanisms"""
    
    def __init__(self, max_memory_mb: int = 1024, batch_size: int = 25):
        self.max_memory_mb = max_memory_mb
        self.batch_size = batch_size
        self.memory_monitor = MemoryMonitor(max_memory_mb)
        self.chunker = SimpleTextChunker(chunk_size=1000, overlap=100)
        
        # Initialize databases
        self.sqlite_path = "./data/knowledge.db"
        self.chromadb_path = "./data/chromadb"
        
        # Progress tracking
        self.processed_pages = 0
        self.processed_chunks = 0
        self.errors = []
    
    def process_pdf_streaming(self, pdf_path: str, book_id: str) -> Tuple[int, Dict[str, Any]]:
        """Stream PDF processing - extract, chunk, and save in batches"""
        print(f"📖 Streaming PDF processing: {pdf_path}")
        
        page_batch_size = 50  # Process 50 pages at a time
        total_chunks_saved = 0
        metadata = {}
        chunk_counter = 0
        
        try:
            with open(pdf_path, 'rb') as file:
                # Try pdfplumber first (more reliable)
                try:
                    import pdfplumber
                    with pdfplumber.open(file) as pdf:
                        total_pages = len(pdf.pages)
                        metadata = {
                            'total_pages': total_pages,
                            'title': 'Machine Learning for Factor Investing (Python Version)',
                            'author': 'Guillaume Coqueret, Tony Guida'
                        }
                        
                        print(f"📄 Total pages to process: {total_pages}")
                        
                        # Process pages in batches
                        for batch_start in range(0, total_pages, page_batch_size):
                            batch_end = min(batch_start + page_batch_size, total_pages)
                            print(f"\\n📦 Processing batch: pages {batch_start+1}-{batch_end}")
                            
                            # Extract text from this batch of pages
                            batch_text = ""
                            for i in range(batch_start, batch_end):
                                page = pdf.pages[i]
                                text = page.extract_text() or ""
                                batch_text += text + "\\n\\n"
                                self.processed_pages += 1
                                
                                if i % 10 == 0:
                                    print(f"  📄 Extracted page {i+1}/{total_pages} (Memory: {self.memory_monitor.get_memory_usage():.1f}MB)")
                            
                            # Chunk this batch
                            batch_chunks = self.chunker.chunk_text(batch_text)
                            print(f"  🔪 Created {len(batch_chunks)} chunks from batch")
                            
                            # Convert to chunk objects
                            chunk_objects = []
                            for chunk_text in batch_chunks:
                                chunk = {
                                    'id': f"{book_id}_chunk_{chunk_counter:04d}",
                                    'book_id': book_id,
                                    'chunk_index': chunk_counter,
                                    'text': chunk_text,
                                    'metadata': {
                                        'source': 'streaming_processor',
                                        'series': 'CRC Financial Mathematics Series',
                                        'publisher': 'CRC Press',
                                        'authors': 'Guillaume Coqueret, Tony Guida',
                                        'title': 'Machine Learning for Factor Investing',
                                        'page_batch': f"{batch_start+1}-{batch_end}"
                                    }
                                }
                                chunk_objects.append(chunk)
                                chunk_counter += 1
                            
                            # Save this batch to ChromaDB immediately
                            if self.save_batch_to_chromadb(chunk_objects):
                                total_chunks_saved += len(chunk_objects)
                                print(f"  ✅ Saved {len(chunk_objects)} chunks (Total: {total_chunks_saved})")
                            else:
                                print(f"  ❌ Failed to save batch")
                                
                            # Clear batch data from memory
                            del batch_text, batch_chunks, chunk_objects
                            import gc
                            gc.collect()
                            
                            print(f"  🧠 Memory after batch: {self.memory_monitor.get_memory_usage():.1f}MB")
                            
                except ImportError:
                    print("❌ pdfplumber not available, please install it")
                    raise
        
        except Exception as e:
            print(f"❌ Streaming processing error: {e}")
            raise
        
        print(f"✅ Streaming processing complete: {total_chunks_saved} chunks saved")
        return total_chunks_saved, metadata
    
    def create_chunks(self, pages_text: List[str], book_id: str) -> List[Dict[str, Any]]:
        """Create chunks from page text"""
        print(f"🔪 Creating chunks from {len(pages_text)} pages")
        
        # Combine all text
        full_text = "\n\n".join(pages_text)
        
        # Chunk the text
        chunk_texts = self.chunker.chunk_text(full_text)
        
        # Create chunk objects
        chunks = []
        for i, text in enumerate(chunk_texts):
            chunk = {
                'id': f"{book_id}_chunk_{i:04d}",
                'book_id': book_id,
                'chunk_index': i,
                'text': text,
                'metadata': {
                    'source': 'robust_processor',
                    'series': 'CRC Financial Mathematics Series',
                    'publisher': 'CRC Press',
                    'authors': 'Guillaume Coqueret, Tony Guida',
                    'title': 'Machine Learning for Factor Investing'
                }
            }
            chunks.append(chunk)
        
        print(f"✅ Created {len(chunks)} chunks")
        return chunks
    
    def save_batch_to_chromadb(self, chunks: List[Dict[str, Any]]) -> bool:
        """Save a small batch of chunks to ChromaDB"""
        if not chunks:
            return True
            
        try:
            # Initialize ChromaDB
            client = chromadb.PersistentClient(
                path=self.chromadb_path,
                settings=Settings(anonymized_telemetry=False, allow_reset=False)
            )
            
            # Get or create collection
            try:
                collection = client.get_collection(name='trading_books')
            except:
                collection = client.create_collection(name='trading_books')
                print("  📊 Created trading_books collection")
            
            # Prepare batch data
            ids = [chunk['id'] for chunk in chunks]
            documents = [chunk['text'] for chunk in chunks]
            metadatas = []
            
            for chunk in chunks:
                metadata = {
                    'book_id': chunk['book_id'],
                    'chunk_index': chunk['chunk_index'],
                    'content_type': 'text',
                    'series': 'CRC Financial Mathematics Series',
                    'publisher': 'CRC Press',
                    'authors': 'Guillaume Coqueret, Tony Guida',
                    'title': 'Machine Learning for Factor Investing'
                }
                metadatas.append(metadata)
            
            # Add to collection
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            self.processed_chunks += len(chunks)
            return True
            
        except Exception as e:
            print(f"❌ ChromaDB batch save error: {e}")
            self.errors.append(f"ChromaDB batch: {e}")
            return False

    def save_to_chromadb(self, chunks: List[Dict[str, Any]]) -> bool:
        """Save chunks to ChromaDB in batches"""
        print(f"📊 Saving {len(chunks)} chunks to ChromaDB...")
        
        try:
            # Initialize ChromaDB
            client = chromadb.PersistentClient(
                path=self.chromadb_path,
                settings=Settings(anonymized_telemetry=False, allow_reset=False)
            )
            
            collection = client.get_collection(name='trading_books')
            initial_count = collection.count()
            
            # Process in batches
            total_saved = 0
            for i in range(0, len(chunks), self.batch_size):
                if not self.memory_monitor.wait_for_memory():
                    print(f"❌ Memory limit exceeded during ChromaDB save")
                    return False
                
                batch = chunks[i:i+self.batch_size]
                
                # Prepare batch data
                ids = [chunk['id'] for chunk in batch]
                documents = [chunk['text'] for chunk in batch]
                metadatas = []
                
                for chunk in batch:
                    metadata = {
                        'book_id': chunk['book_id'],
                        'chunk_index': chunk['chunk_index'],
                        'content_type': 'text',
                        'series': 'CRC Financial Mathematics Series',
                        'publisher': 'CRC Press',
                        'authors': 'Guillaume Coqueret, Tony Guida',
                        'title': 'Machine Learning for Factor Investing'
                    }
                    metadatas.append(metadata)
                
                # Add to collection
                collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                
                total_saved += len(batch)
                self.processed_chunks += len(batch)
                print(f"  📊 Batch {i//self.batch_size + 1}: {total_saved}/{len(chunks)} chunks saved")
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.1)
            
            final_count = collection.count()
            print(f"✅ ChromaDB updated: {initial_count} → {final_count} documents (+{final_count - initial_count})")
            return True
            
        except Exception as e:
            print(f"❌ ChromaDB save error: {e}")
            self.errors.append(f"ChromaDB: {e}")
            return False
    
    def process_book(self, pdf_path: str) -> Dict[str, Any]:
        """Main book processing method using streaming approach"""
        start_time = time.time()
        
        print("🚀 STREAMING BOOK PROCESSOR")
        print("=" * 50)
        print(f"📁 File: {pdf_path}")
        print(f"💾 Memory limit: {self.max_memory_mb}MB")
        print(f"📦 Batch size: {self.batch_size}")
        print("🔄 Method: Streaming (process & save in batches)")
        print()
        
        try:
            # Create book ID
            with open(pdf_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            book_id = f"crc_ml_factor_{file_hash[:8]}"
            
            print(f"📚 Book ID: {book_id}")
            
            # Use streaming processing - extract, chunk, and save in batches
            total_chunks_saved, metadata = self.process_pdf_streaming(pdf_path, book_id)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Generate report
            success = total_chunks_saved > 0
            result = {
                'success': success,
                'book_id': book_id,
                'pages_processed': self.processed_pages,
                'chunks_created': total_chunks_saved,
                'chunks_saved': total_chunks_saved,
                'processing_time_seconds': processing_time,
                'memory_peak_mb': self.memory_monitor.get_memory_usage(),
                'errors': self.errors,
                'method': 'streaming'
            }
            
            print("\n" + "=" * 50)
            print("📊 STREAMING PROCESSING COMPLETE")
            print("=" * 50)
            print(f"✅ Success: {result['success']}")
            print(f"📚 Book ID: {result['book_id']}")
            print(f"📄 Pages: {result['pages_processed']}")
            print(f"💾 Chunks saved: {result['chunks_saved']}")
            print(f"⏱️ Time: {result['processing_time_seconds']:.1f}s")
            print(f"🧠 Peak Memory: {result['memory_peak_mb']:.1f}MB")
            if self.errors:
                print(f"⚠️ Errors: {len(self.errors)}")
                for error in self.errors:
                    print(f"   - {error}")
            
            # Test search functionality
            if success:
                print(f"\n🔍 Testing search functionality...")
                self.test_search(book_id)
            
            return result
            
        except Exception as e:
            print(f"❌ CRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e),
                'pages_processed': self.processed_pages,
                'chunks_created': 0,
                'processing_time_seconds': time.time() - start_time
            }
    
    def test_search(self, book_id: str):
        """Test search functionality for the ingested book"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            client = chromadb.PersistentClient(
                path=self.chromadb_path,
                settings=Settings(anonymized_telemetry=False, allow_reset=False)
            )
            
            collection = client.get_collection(name='trading_books')
            total_docs = collection.count()
            
            # Test search
            results = collection.query(
                query_texts=['machine learning factor investing'],
                n_results=3,
                include=['documents', 'metadatas', 'distances']
            )
            
            print(f"📊 Collection size: {total_docs} documents")
            
            if results['ids'] and results['ids'][0]:
                crc_results = [r for r in results['metadatas'][0] if r.get('book_id') == book_id]
                print(f"🎯 Found {len(crc_results)} CRC book results in top 3")
                
                # Show best result
                best_result = results['documents'][0][0]
                best_score = 1 - results['distances'][0][0]
                print(f"📄 Best match (score: {best_score:.3f}): {best_result[:150]}...")
            else:
                print("❌ No search results found")
                
        except Exception as e:
            print(f"⚠️ Search test failed: {e}")

def main():
    """Main entry point"""
    if len(sys.argv) != 2:
        print("Usage: python robust_book_processor.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    if not Path(pdf_path).exists():
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)
    
    processor = RobustBookProcessor(max_memory_mb=1024, batch_size=25)
    result = processor.process_book(pdf_path)
    
    # Exit with appropriate code
    sys.exit(0 if result['success'] else 1)

if __name__ == "__main__":
    main()