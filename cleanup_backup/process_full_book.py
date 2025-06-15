#!/usr/bin/env python3
"""
Process the complete Python for Algorithmic Trading book
"""

import asyncio
import sys
import hashlib
import logging
from pathlib import Path
from datetime import datetime
import PyPDF2
import sqlite3
import json

# Add src to path
sys.path.append('src')

from src.core.models import Book, Chunk, FileType, ChunkType
from src.ingestion.text_chunker import TextChunker, ChunkingConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DirectBookProcessor:
    def __init__(self):
        self.book_path = "/home/scottschweizer/TradeKnowledge/data/books/Yves Hilpisch - Python for Algorithmic Trading_ From Idea to Cloud Deployment-O'Reilly Media (2020).pdf"
        self.db_path = "/home/scottschweizer/TradeKnowledge/data/knowledge.db"
        self.chunker = TextChunker()
        
    def extract_pdf_text(self):
        """Extract text from all pages of the PDF"""
        print("📖 Extracting text from PDF...")
        
        pages_text = []
        
        try:
            with open(self.book_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)
                
                print(f"📄 Processing {total_pages} pages...")
                
                for page_num in range(total_pages):
                    try:
                        page = reader.pages[page_num]
                        text = page.extract_text()
                        
                        # Clean and process text
                        text = self._clean_text(text)
                        
                        if text.strip():  # Only add non-empty pages
                            pages_text.append({
                                'page_number': page_num + 1,
                                'text': text,
                                'char_count': len(text),
                                'word_count': len(text.split())
                            })
                            
                        if (page_num + 1) % 5 == 0:
                            print(f"  ✅ Processed page {page_num + 1}/{total_pages}")
                            
                    except Exception as e:
                        print(f"  ⚠️  Error on page {page_num + 1}: {e}")
                        
        except Exception as e:
            print(f"❌ Error reading PDF: {e}")
            return []
        
        print(f"✅ Extracted text from {len(pages_text)} pages")
        total_chars = sum(p['char_count'] for p in pages_text)
        total_words = sum(p['word_count'] for p in pages_text)
        print(f"📊 Total: {total_chars:,} characters, {total_words:,} words")
        
        return pages_text
    
    def _clean_text(self, text):
        """Clean extracted text"""
        if not text:
            return ""
        
        # Basic cleaning
        text = text.replace('\x00', '')  # Remove null bytes
        text = ' '.join(text.split())    # Normalize whitespace
        
        return text
    
    def create_book_record(self, pages_text):
        """Create book record"""
        print("📚 Creating book record...")
        
        # Generate file hash
        with open(self.book_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        book = Book(
            id="Python_for_Algorithmic_Trading_full",
            title="Python for Algorithmic Trading",
            author="Yves Hilpisch",
            file_path=str(self.book_path),
            file_type=FileType.PDF,
            file_hash=file_hash,
            total_pages=len(pages_text),
            categories=["algorithmic trading", "python", "finance"],
            created_at=datetime.now()
        )
        
        print(f"✅ Book record created: {book.title}")
        return book
    
    def create_chunks(self, book, pages_text):
        """Create chunks from pages"""
        print("✂️  Creating chunks...")
        
        # Combine all page text
        full_text = " ".join(page['text'] for page in pages_text)
        
        # Configure chunking
        config = ChunkingConfig(
            chunk_size=1000,         # Reasonable chunk size
            chunk_overlap=200,       # Good overlap for context
            min_chunk_size=100,      # Minimum viable chunk
            max_chunk_size=2000,     # Maximum to prevent huge chunks
            respect_sentences=True,  # Keep sentences together
            respect_paragraphs=True  # Keep paragraphs together when possible
        )
        
        print(f"📝 Full text length: {len(full_text):,} characters")
        print(f"⚙️  Chunk config: size={config.chunk_size}, overlap={config.chunk_overlap}")
        
        # Create chunks
        raw_chunks = self.chunker.chunk_text(full_text, config)
        
        print(f"📦 Created {len(raw_chunks)} raw chunks")
        
        # Convert to Chunk objects
        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk = Chunk(
                id=f"{book.id}_chunk_{i:05d}",
                book_id=book.id,
                chunk_index=i,
                chunk_type=ChunkType.TEXT,
                text=chunk_text,
                created_at=datetime.now(),
                metadata={
                    "char_count": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                    "chunk_method": "sentence_aware"
                }
            )
            chunks.append(chunk)
        
        print(f"✅ Created {len(chunks)} final chunks")
        
        # Update book with chunk count
        book.total_chunks = len(chunks)
        
        return chunks
    
    def save_to_database(self, book, chunks):
        """Save book and chunks to SQLite database"""
        print("💾 Saving to database...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if book already exists
            cursor.execute("SELECT id FROM books WHERE id = ?", (book.id,))
            if cursor.fetchone():
                print("🔄 Book exists, deleting old data...")
                cursor.execute("DELETE FROM chunks WHERE book_id = ?", (book.id,))
                cursor.execute("DELETE FROM books WHERE id = ?", (book.id,))
                print("  ✅ Old data deleted")
            
            # Insert book
            cursor.execute("""
                INSERT INTO books (
                    id, title, author, isbn, file_path, file_type, file_hash,
                    total_pages, total_chunks, categories, metadata, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                book.id, book.title, book.author, book.isbn, book.file_path,
                book.file_type.value, book.file_hash, book.total_pages,
                book.total_chunks, json.dumps(book.categories),
                json.dumps(book.metadata), book.created_at.isoformat()
            ))
            
            print(f"✅ Book saved: {book.title}")
            
            # Insert chunks in batches
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                chunk_data = [
                    (
                        chunk.id, chunk.book_id, chunk.chunk_index,
                        chunk.chunk_type.value, chunk.text,
                        json.dumps(chunk.metadata), chunk.created_at.isoformat()
                    )
                    for chunk in batch
                ]
                
                cursor.executemany("""
                    INSERT INTO chunks (
                        id, book_id, chunk_index, chunk_type, text, metadata, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, chunk_data)
                
                print(f"  ✅ Saved batch {i//batch_size + 1} ({len(batch)} chunks)")
            
            conn.commit()
            conn.close()
            
            print(f"🎉 Successfully saved {len(chunks)} chunks to database!")
            
        except Exception as e:
            print(f"❌ Database error: {e}")
            raise
    
    def get_statistics(self):
        """Get processing statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get book info
            cursor.execute("""
                SELECT title, total_pages, total_chunks 
                FROM books 
                WHERE id = 'Python_for_Algorithmic_Trading_full'
            """)
            book_info = cursor.fetchone()
            
            if book_info:
                title, pages, chunks = book_info
                print(f"\n📊 Processing Statistics:")
                print(f"   Book: {title}")
                print(f"   Pages: {pages}")
                print(f"   Chunks: {chunks}")
                
                # Get chunk statistics
                cursor.execute("""
                    SELECT 
                        AVG(LENGTH(text)) as avg_length,
                        MIN(LENGTH(text)) as min_length,
                        MAX(LENGTH(text)) as max_length,
                        SUM(LENGTH(text)) as total_chars
                    FROM chunks 
                    WHERE book_id = 'Python_for_Algorithmic_Trading_full'
                """)
                stats = cursor.fetchone()
                
                if stats:
                    avg_len, min_len, max_len, total_chars = stats
                    print(f"   Average chunk size: {avg_len:.0f} chars")
                    print(f"   Size range: {min_len} - {max_len} chars")
                    print(f"   Total text: {total_chars:,} chars")
            
            conn.close()
            
        except Exception as e:
            print(f"Error getting statistics: {e}")

async def main():
    print("🚀 Processing complete Python for Algorithmic Trading book")
    
    processor = DirectBookProcessor()
    
    # Step 1: Extract text from PDF
    pages_text = processor.extract_pdf_text()
    if not pages_text:
        print("❌ Failed to extract text from PDF")
        return
    
    # Step 2: Create book record
    book = processor.create_book_record(pages_text)
    
    # Step 3: Create chunks
    chunks = processor.create_chunks(book, pages_text)
    
    # Step 4: Save to database
    processor.save_to_database(book, chunks)
    
    # Step 5: Show statistics
    processor.get_statistics()
    
    print("\n🏆 Book processing completed successfully!")
    print("📋 Next steps:")
    print("   1. Generate embeddings for all chunks")
    print("   2. Upload embeddings to vector database")
    print("   3. Test search functionality")

if __name__ == "__main__":
    asyncio.run(main())