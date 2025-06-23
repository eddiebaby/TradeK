#!/usr/bin/env python3
"""
Simple ingestion for dp439.pdf using PyPDF2
"""

import sys
sys.path.append('.')

import PyPDF2
import sqlite3
import chromadb
import hashlib
import time
from pathlib import Path
from datetime import datetime

def extract_pdf_content(pdf_path):
    """Extract content from PDF using PyPDF2"""
    
    print("📖 Extracting PDF content...")
    
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            print(f"   📄 Pages: {len(reader.pages)}")
            
            # Extract all text
            full_text = ""
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text.strip():
                    full_text += f"\n--- Page {i+1} ---\n"
                    full_text += page_text
            
            print(f"   📝 Total text length: {len(full_text)} characters")
            
            return full_text.strip()
            
    except Exception as e:
        print(f"❌ PDF extraction error: {e}")
        return None

def create_chunks(text, chunk_size=1000):
    """Create chunks from text"""
    
    if not text:
        return []
    
    print("✂️ Creating text chunks...")
    
    # Simple chunking by sentences
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    print(f"   📊 Created {len(chunks)} chunks")
    
    return chunks

def save_to_sqlite(book_id, title, chunks, file_path):
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
            "Jón Daníelsson, Jean-Pierre Zigrand",
            str(file_path),
            "PDF",
            file_hash,
            len(chunks),
            datetime.now().isoformat(),
            "PyPDF2_simple"
        ))
        
        # Insert chunks
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{book_id}_chunk_{i}"
            
            cursor.execute("""
                INSERT OR REPLACE INTO chunks 
                (id, book_id, chunk_index, text, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                chunk_id,
                book_id,
                i,
                chunk_text,
                datetime.now().isoformat()
            ))
        
        conn.commit()
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
        
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{book_id}_chunk_{i}"
            
            clean_text = chunk_text.strip()
            if len(clean_text) < 20:  # Skip very short chunks
                continue
                
            ids.append(chunk_id)
            documents.append(clean_text)
            
            metadata = {
                "book_id": book_id,
                "chunk_index": i,
                "title": title,
                "source": "dp439.pdf"
            }
            metadatas.append(metadata)
        
        # Add in batches
        batch_size = 25
        for i in range(0, len(documents), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_docs = documents[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            
            collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta
            )
        
        print(f"   ✅ Saved {len(documents)} chunks to ChromaDB")
        return True
        
    except Exception as e:
        print(f"   ❌ ChromaDB error: {e}")
        return False

def test_semantic_search(book_id, title):
    """Test semantic search"""
    
    print("🔍 Testing semantic search...")
    
    try:
        client = chromadb.PersistentClient(path="./data/qdrant")
        collection = client.get_collection("books")
        
        test_queries = [
            "time scaling risk square root rule",
            "financial risk management",
            "derivatives pricing"
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
                print(f"      Found {len(docs)} results")
                
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
    
    pdf_path = Path("/home/scottschweizer/TradeKnowledge/Knowledge/dp439.pdf")
    
    print("📄 SIMPLE DP439 INGESTION")
    print("=" * 40)
    print(f"📁 File: {pdf_path.name}")
    print(f"📊 Size: {pdf_path.stat().st_size / 1024:.1f} KB")
    print()
    
    # Extract content
    content = extract_pdf_content(pdf_path)
    if not content:
        print("❌ No content extracted!")
        return
    
    # Create chunks
    chunks = create_chunks(content, chunk_size=1000)
    if not chunks:
        print("❌ No chunks created!")
        return
    
    # Generate book ID and title
    book_id = f"dp439_{hashlib.md5(str(pdf_path).encode()).hexdigest()[:8]}"
    title = "On time-scaling of risk and the square-root-of-time rule"
    
    print(f"📚 Book ID: {book_id}")
    print(f"📝 Title: {title}")
    print()
    
    # Save to databases
    sqlite_success = save_to_sqlite(book_id, title, chunks, pdf_path)
    chromadb_success = save_to_chromadb(book_id, title, chunks)
    
    if chromadb_success:
        test_semantic_search(book_id, title)
    
    print("🎉 INGESTION COMPLETED!")
    print(f"   📚 Book ID: {book_id}")
    print(f"   📝 Chunks: {len(chunks)}")
    print(f"   💾 SQLite: {'✅' if sqlite_success else '❌'}")
    print(f"   📚 ChromaDB: {'✅' if chromadb_success else '❌'}")

if __name__ == "__main__":
    main()