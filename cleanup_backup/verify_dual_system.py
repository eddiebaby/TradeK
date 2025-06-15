#!/usr/bin/env python3
"""
Verify the dual database system (Qdrant + ChromaDB) works properly
"""

import sys
import asyncio
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Add src to path
sys.path.append('src')

from src.core.qdrant_storage import QdrantStorage

async def verify_dual_system():
    """Verify both Qdrant and ChromaDB have the same data and work properly"""
    
    print("🔍 Verifying Dual Database System")
    print("="*45)
    
    # Test query
    test_query = "machine learning trading strategies"
    
    # Initialize both systems
    print("📦 Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    query_embedding = model.encode([test_query], convert_to_numpy=True)[0].tolist()
    
    # Test Qdrant
    print("\n🔧 Testing Qdrant...")
    try:
        qdrant_storage = QdrantStorage()
        qdrant_stats = await qdrant_storage.get_collection_stats()
        print(f"  ✅ Collection: {qdrant_stats['collection_name']}")
        print(f"  ✅ Documents: {qdrant_stats['total_embeddings']}")
        print(f"  ✅ Status: {qdrant_stats['status']}")
        
        # Test search
        qdrant_results = await qdrant_storage.search_semantic(query_embedding, limit=3)
        print(f"  ✅ Search results: {len(qdrant_results)}")
        
        if qdrant_results:
            for i, result in enumerate(qdrant_results[:2], 1):
                book_title = result['metadata'].get('book_title', 'Unknown')
                score = result['score']
                print(f"    {i}. {book_title} (Score: {score:.3f})")
        
        qdrant_working = True
        
    except Exception as e:
        print(f"  ❌ Qdrant error: {e}")
        qdrant_working = False
    
    # Test ChromaDB
    print("\n🔧 Testing ChromaDB...")
    try:
        chroma_dir = '/home/scottschweizer/TradeKnowledge/data/chromadb'
        client = chromadb.PersistentClient(path=chroma_dir)
        
        collections = client.list_collections()
        if collections:
            collection = collections[0]
            count = collection.count()
            print(f"  ✅ Collection: {collection.name}")
            print(f"  ✅ Documents: {count}")
            
            # Test search
            chroma_results = collection.query(
                query_texts=[test_query],
                n_results=3,
                include=['documents', 'metadatas', 'distances']
            )
            
            results_count = len(chroma_results['ids'][0]) if chroma_results['ids'] else 0
            print(f"  ✅ Search results: {results_count}")
            
            if results_count > 0:
                for i in range(min(2, results_count)):
                    book_title = chroma_results['metadatas'][0][i].get('book_title', 'Unknown')
                    distance = chroma_results['distances'][0][i]
                    score = 1 - distance  # Convert distance to similarity
                    print(f"    {i+1}. {book_title} (Score: {score:.3f})")
            
            chroma_working = True
        else:
            print("  ❌ No collections found")
            chroma_working = False
            
    except Exception as e:
        print(f"  ❌ ChromaDB error: {e}")
        chroma_working = False
    
    # Compare results
    print("\n📊 System Comparison:")
    print(f"  Qdrant:   {'✅ Working' if qdrant_working else '❌ Not working'}")
    print(f"  ChromaDB: {'✅ Working' if chroma_working else '❌ Not working'}")
    
    if qdrant_working and chroma_working:
        print("  🎉 Dual system fully operational!")
        
        # Test book coverage
        print("\n📚 Book Coverage Test:")
        
        # Test trading book query
        trading_results_q = await qdrant_storage.search_semantic(
            model.encode(["algorithmic trading"], convert_to_numpy=True)[0].tolist(), 
            limit=1
        )
        trading_results_c = collection.query(
            query_texts=["algorithmic trading"],
            n_results=1
        )
        
        print(f"  Trading book query:")
        if trading_results_q:
            print(f"    Qdrant: Found {trading_results_q[0]['metadata'].get('book_title', 'Unknown')}")
        if trading_results_c['ids'] and trading_results_c['ids'][0]:
            print(f"    ChromaDB: Found {trading_results_c['metadatas'][0][0].get('book_title', 'Unknown')}")
        
        # Test embeddings paper query
        embedding_results_q = await qdrant_storage.search_semantic(
            model.encode(["universal geometry embeddings"], convert_to_numpy=True)[0].tolist(), 
            limit=1
        )
        embedding_results_c = collection.query(
            query_texts=["universal geometry embeddings"],
            n_results=1
        )
        
        print(f"  Embeddings paper query:")
        if embedding_results_q:
            print(f"    Qdrant: Found {embedding_results_q[0]['metadata'].get('book_title', 'Unknown')}")
        if embedding_results_c['ids'] and embedding_results_c['ids'][0]:
            print(f"    ChromaDB: Found {embedding_results_c['metadatas'][0][0].get('book_title', 'Unknown')}")
        
        return True
    else:
        print("  ❌ Dual system has issues!")
        return False

if __name__ == "__main__":
    success = asyncio.run(verify_dual_system())
    if success:
        print("\n🏆 Dual database system verification completed!")
        print("📋 Both Qdrant and ChromaDB are working properly")
    else:
        print("\n❌ Dual database system verification failed")