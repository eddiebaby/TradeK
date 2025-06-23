#!/usr/bin/env python3
"""
Re-ingest the paper specifically into ChromaDB for semantic search
"""

import sys
sys.path.append('.')

import sqlite3
import chromadb
from pathlib import Path
import time

def get_paper_chunks_from_sqlite():
    """Get all chunks for our paper from SQLite"""
    
    db_path = Path("data/knowledge.db")
    if not db_path.exists():
        print("❌ SQLite database not found")
        return []
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    target_book_id = 'book_ca69c36d'
    
    cursor.execute("""
        SELECT chunk_index, text, metadata 
        FROM chunks 
        WHERE book_id = ? 
        ORDER BY chunk_index
    """, (target_book_id,))
    
    chunks = cursor.fetchall()
    conn.close()
    
    return chunks

def ingest_to_chromadb():
    """Ingest paper chunks into ChromaDB for semantic search"""
    
    print("📚 INGESTING PAPER INTO CHROMADB FOR SEMANTIC SEARCH")
    print("=" * 60)
    
    # Get chunks from SQLite
    print("🔍 Getting paper chunks from SQLite...")
    chunks = get_paper_chunks_from_sqlite()
    
    if not chunks:
        print("❌ No paper chunks found in SQLite")
        return False
    
    print(f"✅ Found {len(chunks)} chunks in SQLite")
    
    try:
        # Connect to ChromaDB
        print("🔗 Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path="./data/qdrant")
        
        # Create or get collection for books
        collection_name = "books"
        try:
            collection = client.get_collection(collection_name)
            print(f"✅ Using existing collection: {collection_name}")
        except:
            collection = client.create_collection(
                name=collection_name,
                metadata={"description": "Book chunks with embeddings"}
            )
            print(f"✅ Created new collection: {collection_name}")
        
        # Prepare data for ChromaDB
        print("📝 Preparing chunks for ChromaDB...")
        
        target_book_id = 'book_ca69c36d'
        ids = []
        documents = []
        metadatas = []
        
        for chunk_index, text, metadata_str in chunks:
            chunk_id = f"{target_book_id}_chunk_{chunk_index}"
            
            # Clean up text
            clean_text = text.strip()
            if not clean_text:
                continue
                
            ids.append(chunk_id)
            documents.append(clean_text)
            
            # Create metadata
            chunk_metadata = {
                "book_id": target_book_id,
                "chunk_index": chunk_index,
                "title": "Harnessing the Universal Geometry of Embeddings",
                "source": "2505.12540v2.pdf"
            }
            metadatas.append(chunk_metadata)
        
        print(f"📊 Prepared {len(documents)} chunks for ingestion")
        
        # Add to ChromaDB in batches
        print("💾 Adding chunks to ChromaDB...")
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
            time.sleep(0.1)  # Brief pause
        
        # Verify ingestion
        print("\n🔍 Verifying ChromaDB ingestion...")
        
        # Count total items in collection
        total_count = collection.count()
        print(f"   Total items in collection: {total_count}")
        
        # Count items from our paper
        paper_results = collection.get(
            where={"book_id": target_book_id},
            limit=1000
        )
        paper_count = len(paper_results.get('ids', []))
        print(f"   Items from our paper: {paper_count}")
        
        if paper_count == len(chunks):
            print("✅ All chunks successfully ingested into ChromaDB!")
            return True
        else:
            print(f"⚠️ Expected {len(chunks)} chunks, but found {paper_count}")
            return False
            
    except Exception as e:
        print(f"❌ ChromaDB ingestion error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_semantic_search():
    """Test semantic search after ingestion"""
    
    print("\n🔍 TESTING SEMANTIC SEARCH")
    print("=" * 30)
    
    try:
        client = chromadb.PersistentClient(path="./data/qdrant")
        collection = client.get_collection("books")
        
        target_book_id = 'book_ca69c36d'
        
        # Test query
        test_query = "vec2vec embedding translation method"
        print(f"🎯 Test query: '{test_query}'")
        
        results = collection.query(
            query_texts=[test_query],
            n_results=3,
            where={"book_id": target_book_id}
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
            print("❌ No semantic search results found")
            
    except Exception as e:
        print(f"❌ Semantic search test failed: {e}")

def main():
    """Main function"""
    
    success = ingest_to_chromadb()
    
    if success:
        test_semantic_search()
        print("\n🎉 ChromaDB ingestion completed!")
        print("   Semantic search is now available for the paper")
    else:
        print("\n❌ ChromaDB ingestion failed")

if __name__ == "__main__":
    main()