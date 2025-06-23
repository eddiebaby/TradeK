#!/usr/bin/env python3
"""
Search for numpy across all ingested documents using semantic search
"""

import sys
sys.path.append('.')

import chromadb

def search_numpy_semantic():
    """Search for numpy using semantic search across all documents"""
    
    print("🔍 SEMANTIC SEARCH FOR NUMPY")
    print("=" * 40)
    
    try:
        client = chromadb.PersistentClient(path="./data/qdrant")
        collection = client.get_collection("books")
        
        # Various numpy-related queries
        numpy_queries = [
            "numpy arrays numerical computing",
            "numpy mathematical operations",
            "numpy data structures",
            "import numpy as np",
            "numpy functions methods",
            "numerical Python arrays"
        ]
        
        all_results = {}
        
        for query in numpy_queries:
            print(f"\n🎯 Query: '{query}'")
            print("-" * 50)
            
            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=5,
                    # No book_id filter - search across all documents
                )
                
                if results and results.get('documents') and results['documents'][0]:
                    docs = results['documents'][0]
                    distances = results.get('distances', [None])[0] or []
                    metadatas = results.get('metadatas', [None])[0] or []
                    
                    for i, doc in enumerate(docs):
                        distance = distances[i] if i < len(distances) else None
                        metadata = metadatas[i] if i < len(metadatas) else {}
                        
                        score = 1 - distance if distance is not None else 0
                        book_id = metadata.get('book_id', 'unknown')
                        book_title = metadata.get('title', 'Unknown')
                        chunk_idx = metadata.get('chunk_index', 'unknown')
                        
                        # Track which books have numpy content
                        if book_id not in all_results:
                            all_results[book_id] = {'title': book_title, 'chunks': []}
                        
                        print(f"   📖 Match {i+1} (Score: {score:.3f})")
                        print(f"      Book: {book_title}")
                        print(f"      Book ID: {book_id}")
                        print(f"      Chunk: {chunk_idx}")
                        print(f"      Text: {doc[:200]}...")
                        print()
                        
                        # Store for summary
                        all_results[book_id]['chunks'].append({
                            'score': score,
                            'chunk': chunk_idx,
                            'text': doc[:300]
                        })
                
                else:
                    print("   ❌ No results found")
                    
            except Exception as e:
                print(f"   ❌ Search error: {e}")
        
        # Summary across all books
        print("\n📊 NUMPY CONTENT SUMMARY ACROSS ALL BOOKS")
        print("=" * 60)
        
        for book_id, book_data in all_results.items():
            unique_chunks = len(set(chunk['chunk'] for chunk in book_data['chunks']))
            avg_score = sum(chunk['score'] for chunk in book_data['chunks']) / len(book_data['chunks'])
            
            print(f"\n📚 {book_data['title']}")
            print(f"   Book ID: {book_id}")
            print(f"   Numpy chunks found: {unique_chunks}")
            print(f"   Average relevance: {avg_score:.3f}")
            
            # Show best match
            best_chunk = max(book_data['chunks'], key=lambda x: x['score'])
            print(f"   Best match (Score: {best_chunk['score']:.3f}):")
            print(f"      {best_chunk['text']}...")
        
    except Exception as e:
        print(f"❌ Search error: {e}")

def search_numpy_direct():
    """Direct search for specific numpy mentions"""
    
    print("\n🔍 DIRECT NUMPY SEARCHES")
    print("=" * 30)
    
    try:
        client = chromadb.PersistentClient(path="./data/qdrant")
        collection = client.get_collection("books")
        
        # Direct searches for numpy
        direct_queries = [
            "import numpy",
            "np.array",
            "numpy.array",
            "numpy library"
        ]
        
        for query in direct_queries:
            print(f"\n🎯 Direct search: '{query}'")
            
            results = collection.query(
                query_texts=[query],
                n_results=3
            )
            
            if results and results.get('documents') and results['documents'][0]:
                docs = results['documents'][0]
                distances = results.get('distances', [None])[0] or []
                metadatas = results.get('metadatas', [None])[0] or []
                
                for i, doc in enumerate(docs):
                    distance = distances[i] if i < len(distances) else None
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    
                    score = 1 - distance if distance is not None else 0
                    book_title = metadata.get('title', 'Unknown')
                    
                    print(f"   📖 Result {i+1} (Score: {score:.3f})")
                    print(f"      Book: {book_title}")
                    print(f"      Text: {doc[:150]}...")
                    print()
            else:
                print("   ❌ No direct matches")
        
    except Exception as e:
        print(f"❌ Direct search error: {e}")

def get_book_stats():
    """Get statistics about all books in the collection"""
    
    print("\n📊 COLLECTION STATISTICS")
    print("=" * 30)
    
    try:
        client = chromadb.PersistentClient(path="./data/qdrant")
        collection = client.get_collection("books")
        
        # Get total count
        total_count = collection.count()
        print(f"Total chunks in collection: {total_count}")
        
        # Get sample to see what books we have
        sample = collection.peek(limit=20)
        
        if sample and sample.get('metadatas'):
            books = {}
            for metadata in sample['metadatas']:
                if metadata:
                    book_id = metadata.get('book_id', 'unknown')
                    title = metadata.get('title', 'Unknown')
                    books[book_id] = title
            
            print(f"\nBooks in collection ({len(books)} found):")
            for book_id, title in books.items():
                print(f"   📚 {title} (ID: {book_id})")
        
    except Exception as e:
        print(f"❌ Stats error: {e}")

def main():
    """Main search function"""
    
    print("🚀 SEARCHING FOR NUMPY ACROSS ALL DOCUMENTS")
    print("=" * 50)
    
    # Get collection stats first
    get_book_stats()
    
    # Semantic search for numpy
    search_numpy_semantic()
    
    # Direct search for numpy
    search_numpy_direct()
    
    print("\n✅ Numpy search completed!")

if __name__ == "__main__":
    main()