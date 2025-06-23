#!/usr/bin/env python3
"""
Ingest dp439.pdf with graphics preservation
"""

import sys
sys.path.append('.')

import time
from pathlib import Path

def ingest_dp439():
    """Ingest dp439.pdf using enhanced processor"""
    
    pdf_path = "/home/scottschweizer/TradeKnowledge/Knowledge/dp439.pdf"
    pdf_file = Path(pdf_path)
    
    print("📄 INGESTING DP439.PDF")
    print("=" * 40)
    print(f"📁 File: {pdf_file.name}")
    print(f"📊 Size: {pdf_file.stat().st_size / 1024:.1f} KB")
    print()
    
    try:
        from enhanced_docling_processor import EnhancedDoclingProcessor
        
        print("🔧 Initializing enhanced processor...")
        processor = EnhancedDoclingProcessor()
        
        print("🔄 Processing PDF with graphics preservation...")
        start_time = time.time()
        
        result = processor.process_book(pdf_file)
        processing_time = time.time() - start_time
        
        if result.get('success'):
            print(f"✅ Processing completed in {processing_time:.1f}s")
            print()
            print("📊 RESULTS:")
            print(f"   📄 Processor: {result.get('processor_used', 'Unknown')}")
            print(f"   📝 Chunks: {result.get('total_chunks', 0)}")
            print(f"   🧠 Memory: {result.get('memory_usage_mb', 0):.1f} MB")
            print(f"   💾 SQLite: {'✅' if result.get('sqlite_success') else '❌'}")
            print(f"   📚 ChromaDB: {'✅' if result.get('chromadb_success') else '❌'}")
            
            if 'book_id' in result:
                print(f"   🆔 Book ID: {result['book_id']}")
            
            return result
        else:
            print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
            return None
            
    except ImportError:
        print("❌ Enhanced processor not available")
        return None
    except Exception as e:
        print(f"❌ Processing error: {e}")
        import traceback
        traceback.print_exc()
        return None

def add_to_chromadb(book_id):
    """Add the processed book to ChromaDB for semantic search"""
    
    print(f"\n📚 Adding {book_id} to ChromaDB...")
    
    try:
        import sqlite3
        import chromadb
        
        # Get chunks from SQLite
        conn = sqlite3.connect("data/knowledge.db")
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT chunk_index, text, metadata 
            FROM chunks 
            WHERE book_id = ? 
            ORDER BY chunk_index
        """, (book_id,))
        
        chunks = cursor.fetchall()
        conn.close()
        
        if not chunks:
            print("❌ No chunks found in SQLite")
            return False
        
        print(f"✅ Found {len(chunks)} chunks")
        
        # Add to ChromaDB
        client = chromadb.PersistentClient(path="./data/qdrant")
        collection = client.get_collection("books")
        
        ids = []
        documents = []
        metadatas = []
        
        for chunk_index, text, metadata_str in chunks:
            chunk_id = f"{book_id}_chunk_{chunk_index}"
            
            clean_text = text.strip()
            if not clean_text:
                continue
                
            ids.append(chunk_id)
            documents.append(clean_text)
            
            chunk_metadata = {
                "book_id": book_id,
                "chunk_index": chunk_index,
                "title": "DP439",
                "source": "dp439.pdf"
            }
            metadatas.append(chunk_metadata)
        
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
            
            print(f"   ✅ Added batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
        
        print("✅ ChromaDB ingestion completed!")
        return True
        
    except Exception as e:
        print(f"❌ ChromaDB ingestion error: {e}")
        return False

def test_semantic_search(book_id):
    """Test semantic search on the new paper"""
    
    print(f"\n🔍 Testing semantic search on {book_id}...")
    
    try:
        import chromadb
        
        client = chromadb.PersistentClient(path="./data/qdrant")
        collection = client.get_collection("books")
        
        # Test query
        test_query = "main contribution research paper"
        
        results = collection.query(
            query_texts=[test_query],
            n_results=2,
            where={"book_id": book_id}
        )
        
        if results and results.get('documents') and results['documents'][0]:
            docs = results['documents'][0]
            distances = results.get('distances', [None])[0] or []
            
            print(f"✅ Found {len(docs)} semantic results!")
            
            for i, doc in enumerate(docs):
                distance = distances[i] if i < len(distances) else None
                score = 1 - distance if distance is not None else 0
                
                print(f"\n   Result {i+1} (Score: {score:.3f}):")
                print(f"   {doc[:200]}...")
        else:
            print("❌ No semantic search results")
            
    except Exception as e:
        print(f"❌ Semantic search test failed: {e}")

def main():
    """Main ingestion function"""
    
    print("🚀 Starting DP439 ingestion with graphics preservation...")
    
    # Process the PDF
    result = ingest_dp439()
    
    if not result or not result.get('success'):
        print("\n❌ Ingestion failed!")
        return
    
    book_id = result.get('book_id')
    if not book_id:
        print("\n❌ No book ID returned!")
        return
    
    # Add to ChromaDB for semantic search
    chromadb_success = add_to_chromadb(book_id)
    
    if chromadb_success:
        # Test semantic search
        test_semantic_search(book_id)
    
    print(f"\n🎉 DP439 ingestion completed!")
    print(f"   Book ID: {book_id}")
    print(f"   Available for semantic search: {'✅' if chromadb_success else '❌'}")

if __name__ == "__main__":
    main()