#!/usr/bin/env python3
"""
Semantic search on the ingested paper using actual embeddings
"""

import sys
sys.path.append('.')

try:
    from src.search.unified_search import UnifiedSearchEngine
    from src.core.config import get_config
except ImportError as e:
    print(f"❌ Search engine import error: {e}")
    print("Let's try a simpler approach...")
    sys.exit(1)

def test_semantic_search():
    """Test semantic search on the vec2vec paper"""
    
    print("🔍 SEMANTIC SEARCH ON VEC2VEC PAPER")
    print("=" * 50)
    
    try:
        # Initialize search engine
        config = get_config()
        search_engine = UnifiedSearchEngine(config)
        
        # Test queries for semantic understanding
        queries = [
            "What is the main method proposed in this paper?",
            "How does vec2vec work without paired data?", 
            "What are the security implications of embedding translation?",
            "What embedding models were tested?",
            "What results did they achieve?",
            "How do they preserve semantic information?"
        ]
        
        print("Testing semantic search capabilities...\n")
        
        for i, query in enumerate(queries, 1):
            print(f"🔍 Query {i}: '{query}'")
            print("-" * 40)
            
            try:
                results = search_engine.search(
                    query=query,
                    top_k=3,
                    search_type="semantic"
                )
                
                if results and 'results' in results:
                    found_paper_results = False
                    
                    for j, result in enumerate(results['results'][:2]):
                        # Check if this result is from our paper
                        metadata = result.get('metadata', {})
                        book_id = metadata.get('book_id', '')
                        
                        if book_id == 'book_ca69c36d':  # Our paper's book_id
                            found_paper_results = True
                            score = result.get('score', 0)
                            text = result.get('text', '')
                            
                            print(f"   ✅ Result {j+1} (Score: {score:.3f}):")
                            print(f"      {text[:200]}...")
                            print()
                    
                    if not found_paper_results:
                        print("   ❌ No results from our paper found")
                        # Show what we did find
                        for j, result in enumerate(results['results'][:1]):
                            metadata = result.get('metadata', {})
                            book_id = metadata.get('book_id', '')
                            print(f"   Found result from: {book_id}")
                else:
                    print("   ❌ No search results returned")
                    
            except Exception as e:
                print(f"   ❌ Search error: {e}")
            
            print()
        
    except Exception as e:
        print(f"❌ Failed to initialize search engine: {e}")
        print("This suggests ChromaDB might not have our paper embeddings")

def check_chromadb_status():
    """Check if our paper is actually in ChromaDB"""
    
    print("\n🔍 CHECKING CHROMADB STATUS")
    print("=" * 30)
    
    try:
        import chromadb
        from pathlib import Path
        
        # Check if ChromaDB has any collections
        client = chromadb.PersistentClient(path="./data/qdrant")
        collections = client.list_collections()
        
        if not collections:
            print("❌ No ChromaDB collections found")
            print("   The paper embeddings may not be in ChromaDB yet")
            return False
        
        print(f"✅ Found {len(collections)} ChromaDB collections:")
        
        for collection_info in collections:
            collection = client.get_collection(collection_info.name)
            
            # Count total items
            try:
                data = collection.peek(1)
                if data and data.get('ids'):
                    total_count = collection.count()
                    print(f"   📚 {collection_info.name}: {total_count} items")
                    
                    # Check if our paper is in this collection
                    if data.get('metadatas') and len(data['metadatas']) > 0:
                        sample_metadata = data['metadatas'][0]
                        book_id = sample_metadata.get('book_id', 'unknown')
                        print(f"      Sample book_id: {book_id}")
                        
                        if book_id == 'book_ca69c36d':
                            print("      🎯 Found our paper!")
                            return True
                else:
                    print(f"   📚 {collection_info.name}: Empty")
            except Exception as e:
                print(f"   ❌ Error checking collection: {e}")
        
        return False
        
    except Exception as e:
        print(f"❌ ChromaDB check failed: {e}")
        return False

def main():
    """Main function"""
    
    # First check if our paper is in ChromaDB
    paper_in_chromadb = check_chromadb_status()
    
    if not paper_in_chromadb:
        print("\n⚠️ Our paper doesn't appear to be in ChromaDB")
        print("   Semantic search won't work without embeddings")
        print("   The paper was ingested into SQLite but not ChromaDB")
        return
    
    # Run semantic search tests
    test_semantic_search()

if __name__ == "__main__":
    main()