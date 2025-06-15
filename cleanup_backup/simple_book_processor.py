#!/usr/bin/env python3
"""
Simple book processor - split by pages instead of complex chunking
"""

import sys
import sqlite3
import json
import hashlib
from datetime import datetime
import PyPDF2

# Add src to path
sys.path.append('src')

from src.core.models import FileType, ChunkType

def process_book_simple():
    """Process book by pages instead of complex chunking"""
    
    book_path = "/home/scottschweizer/TradeKnowledge/data/books/Yves Hilpisch - Python for Algorithmic Trading_ From Idea to Cloud Deployment-O'Reilly Media (2020).pdf"
    db_path = "/home/scottschweizer/TradeKnowledge/data/knowledge.db"
    
    print("📖 Simple book processing...")
    
    # Extract text page by page
    pages = []
    try:
        with open(book_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)
            
            print(f"📄 Processing {total_pages} pages...")
            
            for page_num in range(total_pages):  # Process ALL pages
                try:
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    
                    if text and len(text.strip()) > 50:  # Only meaningful pages
                        # Clean text and handle Unicode issues
                        clean_text = text.encode('utf-8', 'ignore').decode('utf-8')  # Remove bad Unicode
                        clean_text = ' '.join(clean_text.split())  # Clean whitespace
                        pages.append({
                            'page_number': page_num + 1,
                            'text': clean_text
                        })
                        
                    if (page_num + 1) % 20 == 0:
                        print(f"  ✅ Processed page {page_num + 1}")
                        
                except Exception as e:
                    print(f"  ⚠️  Error on page {page_num + 1}: {e}")
                    
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return False
    
    print(f"✅ Extracted {len(pages)} pages")
    
    # Calculate file hash
    with open(book_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    # Save to database
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        book_id = "Python_for_Algorithmic_Trading_complete"
        
        # Delete existing data
        cursor.execute("DELETE FROM chunks WHERE book_id = ?", (book_id,))
        cursor.execute("DELETE FROM books WHERE id = ?", (book_id,))
        
        # Insert book
        cursor.execute("""
            INSERT INTO books (
                id, title, author, file_path, file_type, file_hash,
                total_pages, total_chunks, categories, metadata, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            book_id,
            "Python for Algorithmic Trading (Complete)",
            "Yves Hilpisch",
            book_path,
            FileType.PDF.value,
            file_hash,
            len(pages),
            len(pages),
            json.dumps(["algorithmic trading", "python"]),
            json.dumps({"processing_method": "page_based"}),
            datetime.now().isoformat()
        ))
        
        print(f"✅ Book saved")
        
        # Insert chunks (one per page)
        for i, page in enumerate(pages):
            chunk_id = f"{book_id}_page_{page['page_number']:03d}"
            
            cursor.execute("""
                INSERT INTO chunks (
                    id, book_id, chunk_index, chunk_type, text, metadata, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk_id,
                book_id,
                i,
                ChunkType.TEXT.value,
                page['text'],
                json.dumps({
                    "page_number": page['page_number'],
                    "char_count": len(page['text']),
                    "word_count": len(page['text'].split())
                }),
                datetime.now().isoformat()
            ))
        
        conn.commit()
        conn.close()
        
        print(f"🎉 Successfully saved {len(pages)} page-based chunks!")
        
        # Show statistics
        total_chars = sum(len(p['text']) for p in pages)
        total_words = sum(len(p['text'].split()) for p in pages)
        print(f"📊 Statistics:")
        print(f"   Pages: {len(pages)}")
        print(f"   Total characters: {total_chars:,}")
        print(f"   Total words: {total_words:,}")
        print(f"   Average chars per page: {total_chars/len(pages):.0f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

if __name__ == "__main__":
    success = process_book_simple()
    if success:
        print("\n🏆 Simple book processing completed!")
        print("📋 Next: Generate embeddings for these chunks")
    else:
        print("\n❌ Processing failed")