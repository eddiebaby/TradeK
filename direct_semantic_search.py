#!/usr/bin/env python3
"""
Direct semantic search using ChromaDB on the vec2vec paper
"""

import sys
sys.path.append('.')

try:
    import chromadb
    from pathlib import Path
    import time
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    sys.exit(1)

def semantic_search_paper():
    """Perform semantic search directly on ChromaDB"""
    
    print("🔍 DIRECT SEMANTIC SEARCH ON VEC2VEC PAPER")
    print("=" * 50)
    
    try:
        # Connect to ChromaDB
        client = chromadb.PersistentClient(path="./data/qdrant")
        collections = client.list_collections()
        
        if not collections:
            print("❌ No ChromaDB collections found")
            print("   Need to add paper embeddings to ChromaDB first")
            return
        
        print(f"📚 Found {len(collections)} collections")
        
        # Find collection with our paper
        target_collection = None
        target_book_id = 'book_ca69c36d'
        
        for collection_info in collections:
            collection = client.get_collection(collection_info.name)
            
            try:
                # Check if our paper is in this collection
                sample = collection.peek(5)
                if sample and sample.get('metadatas'):
                    for metadata in sample['metadatas']:
                        if metadata and metadata.get('book_id') == target_book_id:
                            target_collection = collection
                            print(f"✅ Found paper in collection: {collection_info.name}")
                            break
                    if target_collection:
                        break
            except Exception as e:
                print(f"   Error checking collection {collection_info.name}: {e}")
        
        if not target_collection:
            print("❌ Paper not found in any ChromaDB collection")
            print("   The paper may only be in SQLite, not ChromaDB")
            return
        
        # Test semantic queries
        queries = [
            "vec2vec translation method without paired data",
            "security implications embedding space translation", 
            "Platonic Representation Hypothesis universal latent space",
            "adversarial training cycle consistency embeddings",
            "information extraction from unknown embeddings",
            "T5 BERT GTR GTE embedding models tested"
        ]
        
        print(f"\n🎯 Testing {len(queries)} semantic queries...")
        print("-" * 40)
        
        for i, query in enumerate(queries, 1):
            print(f"\n🔍 Query {i}: '{query}'")
            
            start_time = time.time()
            
            try:
                # Perform semantic search
                results = target_collection.query(
                    query_texts=[query],
                    n_results=3,
                    where={"book_id": target_book_id}  # Filter to our paper only
                )
                
                search_time = time.time() - start_time
                
                if results and results.get('documents') and results['documents'][0]:
                    docs = results['documents'][0]
                    distances = results.get('distances', [None])[0] or []
                    metadatas = results.get('metadatas', [None])[0] or []
                    
                    print(f"   ✅ Found {len(docs)} results in {search_time:.3f}s")
                    
                    for j, doc in enumerate(docs[:2]):  # Show top 2 results
                        distance = distances[j] if j < len(distances) else None
                        metadata = metadatas[j] if j < len(metadatas) else {}
                        
                        score = 1 - distance if distance is not None else 0
                        chunk_idx = metadata.get('chunk_index', 'unknown')
                        
                        print(f"      Result {j+1} (Score: {score:.3f}, Chunk: {chunk_idx}):")
                        print(f"         {doc[:200]}...")
                else:
                    print(f"   ❌ No results found in {search_time:.3f}s")
                    
            except Exception as e:
                print(f"   ❌ Search error: {e}")
        
        # Show collection stats
        print(f"\n📊 Collection Statistics:")
        total_count = target_collection.count()
        print(f"   Total chunks: {total_count}")
        
        # Count chunks from our paper specifically
        try:
            paper_results = target_collection.get(
                where={"book_id": target_book_id},
                limit=1000  # Should be enough for our paper
            )
            paper_count = len(paper_results.get('ids', []))
            print(f"   Paper chunks: {paper_count}")
        except Exception as e:
            print(f"   Error counting paper chunks: {e}")
            
    except Exception as e:
        print(f"❌ ChromaDB connection error: {e}")

def check_if_paper_needs_chromadb_ingestion():
    """Check if we need to ingest the paper into ChromaDB"""
    
    print("\n🔍 CHECKING IF PAPER NEEDS CHROMADB INGESTION")
    print("=" * 50)
    
    try:
        client = chromadb.PersistentClient(path="./data/qdrant")
        collections = client.list_collections()
        
        target_book_id = 'book_ca69c36d'
        found = False
        
        for collection_info in collections:
            collection = client.get_collection(collection_info.name)
            try:
                results = collection.get(
                    where={"book_id": target_book_id},
                    limit=1
                )
                if results and results.get('ids'):
                    found = True
                    print(f"✅ Paper found in ChromaDB collection: {collection_info.name}")
                    break
            except:
                continue
        
        if not found:
            print("❌ Paper NOT found in ChromaDB")
            print("   ➡️ Need to re-run enhanced processor with ChromaDB enabled")
            print("   ➡️ Or the ChromaDB ingestion failed during processing")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking ChromaDB: {e}")
        return False

def main():
    """Main function"""
    
    # Check if paper is in ChromaDB
    if not check_if_paper_needs_chromadb_ingestion():
        print("\n💡 To enable semantic search:")
        print("   1. The paper needs to be ingested into ChromaDB")
        print("   2. Re-run the enhanced processor if needed")
        return
    
    # Perform semantic search
    semantic_search_paper()

if __name__ == "__main__":
    main()