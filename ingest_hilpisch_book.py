#!/usr/bin/env python3
"""
Ingest Yves Hilpisch Python for Algorithmic Trading book
"""

import sys
sys.path.append('.')

import subprocess
import sqlite3
import chromadb
import hashlib
import time
from pathlib import Path
from datetime import datetime

def extract_book_content(pdf_path):
    """Extract content using multiple methods"""
    
    print("📖 Extracting book content...")
    print(f"   📁 File: {Path(pdf_path).name}")
    print(f"   📊 Size: {Path(pdf_path).stat().st_size / (1024*1024):.1f} MB")
    
    # Try pdftotext first (most reliable for O'Reilly books)
    try:
        print("   🔧 Trying pdftotext extraction...")
        result = subprocess.run(['pdftotext', '-layout', str(pdf_path), '-'], 
                              capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0 and result.stdout:
            text = result.stdout.strip()
            print(f"   ✅ pdftotext success: {len(text)} characters")
            return text, "pdftotext"
        else:
            print(f"   ❌ pdftotext failed: {result.stderr}")
    except Exception as e:
        print(f"   ❌ pdftotext error: {e}")
    
    # Fallback to pdfplumber
    try:
        print("   🔧 Trying pdfplumber extraction...")
        import pdfplumber
        
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            print(f"      📄 Processing {len(pdf.pages)} pages...")
            
            for i, page in enumerate(pdf.pages):
                if i % 50 == 0:
                    print(f"      📄 Page {i+1}/{len(pdf.pages)}")
                
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {i+1} ---\n"
                    text += page_text
                    
        if text.strip():
            print(f"   ✅ pdfplumber success: {len(text)} characters")
            return text.strip(), "pdfplumber"
        else:
            print("   ❌ pdfplumber extracted no text")
            
    except Exception as e:
        print(f"   ❌ pdfplumber error: {e}")
    
    # Final fallback to PyPDF2
    try:
        print("   🔧 Trying PyPDF2 extraction...")
        import PyPDF2
        
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            print(f"      📄 Processing {len(reader.pages)} pages...")
            
            for i, page in enumerate(reader.pages):
                if i % 50 == 0:
                    print(f"      📄 Page {i+1}/{len(reader.pages)}")
                    
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {i+1} ---\n"
                    text += page_text
                    
        if text.strip():
            print(f"   ✅ PyPDF2 success: {len(text)} characters")
            return text.strip(), "PyPDF2"
        else:
            print("   ❌ PyPDF2 extracted no text")
            
    except Exception as e:
        print(f"   ❌ PyPDF2 error: {e}")
    
    return None, None

def create_intelligent_chunks(text, chunk_size=1500):
    """Create intelligent chunks for O'Reilly technical book"""
    
    print("✂️ Creating intelligent chunks...")
    
    if not text:
        return []
    
    # O'Reilly books have clear chapter/section structure
    # Split by major sections first
    sections = text.split('\n--- Page')
    chunks = []
    current_chunk = ""
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        # Split sections by paragraphs
        paragraphs = section.split('\n\n')
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Skip page headers/footers
            if len(para) < 30 or 'Chapter' in para[:50] or 'Page' in para[:20]:
                continue
            
            # If adding this paragraph exceeds chunk size
            if len(current_chunk) + len(para) > chunk_size:
                # Save current chunk if substantial
                if len(current_chunk.strip()) > 100:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk
                if len(para) > chunk_size:
                    # Split very long paragraphs by sentences
                    sentences = para.split('. ')
                    temp_chunk = ""
                    for sentence in sentences:
                        if len(temp_chunk) + len(sentence) < chunk_size:
                            temp_chunk += sentence + ". "
                        else:
                            if len(temp_chunk.strip()) > 100:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = sentence + ". "
                    current_chunk = temp_chunk
                else:
                    current_chunk = para + "\n\n"
            else:
                current_chunk += para + "\n\n"
    
    # Add final chunk
    if len(current_chunk.strip()) > 100:
        chunks.append(current_chunk.strip())
    
    # Filter out very short or header-like chunks
    filtered_chunks = []
    for chunk in chunks:
        clean_chunk = chunk.strip()
        if (len(clean_chunk) > 80 and 
            not clean_chunk.startswith('---') and
            'Table of Contents' not in clean_chunk and
            'Index' not in clean_chunk[:50]):
            filtered_chunks.append(clean_chunk)
    
    print(f"   📊 Created {len(filtered_chunks)} chunks")
    
    return filtered_chunks

def save_to_sqlite(book_id, title, author, chunks, file_path, processor):
    """Save to SQLite database"""
    
    print("💾 Saving to SQLite...")
    
    try:
        conn = sqlite3.connect("data/knowledge.db")
        cursor = conn.cursor()
        
        # Calculate file hash
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:8]
        
        # Insert book record
        cursor.execute("""
            INSERT OR REPLACE INTO books 
            (id, title, author, file_path, file_type, file_hash, total_chunks, created_at, processor_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            book_id,
            title,
            author,
            str(file_path),
            "PDF",
            file_hash,
            len(chunks),
            datetime.now().isoformat(),
            processor
        ))
        
        # Insert chunks in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            
            for j, chunk_text in enumerate(batch_chunks):
                chunk_index = i + j
                chunk_id = f"{book_id}_chunk_{chunk_index}"
                
                cursor.execute("""
                    INSERT OR REPLACE INTO chunks 
                    (id, book_id, chunk_index, text, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    chunk_id,
                    book_id,
                    chunk_index,
                    chunk_text,
                    datetime.now().isoformat()
                ))
            
            conn.commit()
            print(f"      Saved batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
        
        conn.close()
        print(f"   ✅ Saved {len(chunks)} chunks to SQLite")
        return True
        
    except Exception as e:
        print(f"   ❌ SQLite error: {e}")
        return False

def save_to_chromadb(book_id, title, chunks):
    """Save to ChromaDB for semantic search"""
    
    print("📚 Saving to ChromaDB...")
    
    try:
        client = chromadb.PersistentClient(path="./data/qdrant")
        collection = client.get_collection("books")
        
        ids = []
        documents = []
        metadatas = []
        
        print(f"   📝 Preparing {len(chunks)} chunks...")
        
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{book_id}_chunk_{i}"
            
            clean_text = chunk_text.strip()
            if len(clean_text) < 50:  # Skip very short chunks
                continue
                
            ids.append(chunk_id)
            documents.append(clean_text)
            
            metadata = {
                "book_id": book_id,
                "chunk_index": i,
                "title": title,
                "source": "hilpisch_algorithmic_trading.pdf"
            }
            metadatas.append(metadata)
        
        # Add in batches with progress tracking
        batch_size = 50  # Larger batches for efficiency
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in range(0, len(documents), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_docs = documents[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            
            collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta
            )
            
            batch_num = i//batch_size + 1
            print(f"      Added batch {batch_num}/{total_batches}")
            
            # Brief pause to prevent overwhelming ChromaDB
            time.sleep(0.1)
        
        print(f"   ✅ Saved {len(documents)} chunks to ChromaDB")
        return True
        
    except Exception as e:
        print(f"   ❌ ChromaDB error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_semantic_search(book_id, title):
    """Test semantic search on the new book"""
    
    print("\n🔍 Testing semantic search...")
    
    try:
        client = chromadb.PersistentClient(path="./data/qdrant")
        collection = client.get_collection("books")
        
        test_queries = [
            "Python algorithmic trading strategies",
            "backtesting trading algorithms",
            "cloud deployment trading systems",
            "financial data analysis Python",
            "risk management trading"
        ]
        
        for query in test_queries:
            results = collection.query(
                query_texts=[query],
                n_results=2,
                where={"book_id": book_id}
            )
            
            if results and results.get('documents') and results['documents'][0]:
                docs = results['documents'][0]
                distances = results.get('distances', [None])[0] or []
                
                print(f"   🎯 Query: '{query}'")
                
                for i, doc in enumerate(docs[:1]):  # Show top result
                    distance = distances[i] if i < len(distances) else None
                    score = 1 - distance if distance is not None else 0
                    
                    print(f"      Score: {score:.3f}")
                    print(f"      Text: {doc[:150]}...")
                print()
        
        return True
        
    except Exception as e:
        print(f"   ❌ Search test error: {e}")
        return False

def main():
    """Main ingestion function"""
    
    pdf_path = Path("/home/scottschweizer/TradeKnowledge/Knowledge/Yves Hilpisch - Python for Algorithmic Trading_ From Idea to Cloud Deployment-O'Reilly Media (2020).pdf")
    
    print("📚 INGESTING YVES HILPISCH ALGORITHMIC TRADING BOOK")
    print("=" * 60)
    print(f"📁 File: {pdf_path.name}")
    print(f"📊 Size: {pdf_path.stat().st_size / (1024*1024):.1f} MB")
    print()
    
    start_time = time.time()
    
    # Extract content
    content, processor = extract_book_content(pdf_path)
    if not content:
        print("❌ No content extracted!")
        return
    
    # Create chunks
    chunks = create_intelligent_chunks(content, chunk_size=1500)
    if not chunks:
        print("❌ No chunks created!")
        return
    
    # Generate book metadata
    book_id = f"hilpisch_trading_{hashlib.md5(str(pdf_path).encode()).hexdigest()[:8]}"
    title = "Python for Algorithmic Trading: From Idea to Cloud Deployment"
    author = "Yves Hilpisch"
    
    print(f"📚 Book ID: {book_id}")
    print(f"📝 Title: {title}")
    print(f"✍️ Author: {author}")
    print(f"🔧 Processor: {processor}")
    print()
    
    # Save to databases
    sqlite_success = save_to_sqlite(book_id, title, author, chunks, pdf_path, processor)
    chromadb_success = save_to_chromadb(book_id, title, chunks)
    
    processing_time = time.time() - start_time
    
    # Test semantic search
    if chromadb_success:
        test_semantic_search(book_id, title)
    
    print("\n🎉 INGESTION COMPLETED!")
    print(f"   📚 Book ID: {book_id}")
    print(f"   📝 Chunks: {len(chunks)}")
    print(f"   ⏱️ Time: {processing_time/60:.1f} minutes")
    print(f"   💾 SQLite: {'✅' if sqlite_success else '❌'}")
    print(f"   📚 ChromaDB: {'✅' if chromadb_success else '❌'}")
    print(f"   🔧 Method: {processor}")

if __name__ == "__main__":
    main()