#!/usr/bin/env python3
"""
Process the new PDF: 2505.12540v2.pdf (Embeddings paper)
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

def process_embeddings_paper():
    """Process the embeddings research paper"""
    
    pdf_path = "/home/scottschweizer/TradeKnowledge/data/books/2505.12540v2.pdf"
    db_path = "/home/scottschweizer/TradeKnowledge/data/knowledge.db"
    
    print("📖 Processing Embeddings Research Paper...")
    print("="*50)
    
    # Extract text page by page
    pages = []
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)
            
            print(f"📄 Processing {total_pages} pages...")
            
            # Get paper title and authors from first page
            first_page_text = reader.pages[0].extract_text()
            title_start = first_page_text.find("Harnessing the Universal Geometry of Embeddings")
            if title_start >= 0:
                paper_title = "Harnessing the Universal Geometry of Embeddings"
                # Extract authors (they appear after title)
                authors_section = first_page_text[title_start + len(paper_title):title_start + 200]
                if "Rishi Jha" in authors_section:
                    authors = "Rishi Jha, Collin Zhang, Vitaly Shmatikov, John X. Morris"
                else:
                    authors = "Unknown"
            else:
                paper_title = "Embeddings Research Paper"
                authors = "Unknown"
            
            print(f"📚 Title: {paper_title}")
            print(f"👥 Authors: {authors}")
            
            for page_num in range(total_pages):
                try:
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    
                    if text and len(text.strip()) > 50:  # Only meaningful pages
                        # Clean text and handle Unicode issues
                        clean_text = text.encode('utf-8', 'ignore').decode('utf-8')
                        clean_text = ' '.join(clean_text.split())  # Clean whitespace
                        pages.append({
                            'page_number': page_num + 1,
                            'text': clean_text
                        })
                        
                    if (page_num + 1) % 5 == 0:
                        print(f"  ✅ Processed page {page_num + 1}")
                        
                except Exception as e:
                    print(f"  ⚠️  Error on page {page_num + 1}: {e}")
                    
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return False
    
    print(f"✅ Extracted {len(pages)} pages")
    
    # Calculate file hash
    with open(pdf_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    # Save to database
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        book_id = "Embeddings_Paper_2505_12540v2"
        
        # Delete existing data if any
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
            paper_title,
            authors,
            pdf_path,
            FileType.PDF.value,
            file_hash,
            len(pages),
            len(pages),
            json.dumps(["machine learning", "embeddings", "vector spaces", "research paper"]),
            json.dumps({
                "paper_id": "2505.12540v2",
                "processing_method": "page_based",
                "document_type": "research_paper"
            }),
            datetime.now().isoformat()
        ))
        
        print(f"✅ Book saved: {paper_title}")
        
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
                    "word_count": len(page['text'].split()),
                    "document_type": "research_paper"
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
        print(f"   Document type: Research Paper")
        
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

if __name__ == "__main__":
    success = process_embeddings_paper()
    if success:
        print("\n🏆 Embeddings paper processing completed!")
        print("📋 Next: Generate embeddings for these chunks")
    else:
        print("\n❌ Processing failed")