#!/usr/bin/env python3
"""
Test search functionality with optimized chunks
"""

import asyncio
import sys
from pathlib import Path

sys.path.append('src')

from src.core.sqlite_storage import SQLiteStorage
from src.search.text_search import TextSearchEngine
from src.core.config import get_config

async def test_search():
    """Test search functionality with existing chunks"""
    
    print("🔍 Testing search functionality...")
    
    # Initialize storage and search
    storage = SQLiteStorage()
    config = get_config()
    text_search = TextSearchEngine()
    
    # Check what books we have
    books = await storage.list_books()
    print(f"📚 Available books: {len(books)}")
    
    for book in books:
        print(f"   • {book.title} ({book.total_chunks} chunks)")
    
    # Find our test book
    test_book = None
    for book in books:
        if "Python for Algorithmic Trading" in book.title:
            test_book = book
            break
    
    if not test_book:
        print("❌ Test book not found")
        return False
    
    print(f"\n📖 Testing with: {test_book.title}")
    print(f"📦 Chunks available: {test_book.total_chunks}")
    
    # Get chunks to see what content we have
    chunks = await storage.get_chunks_by_book(test_book.id)
    print(f"📄 Retrieved {len(chunks)} chunks from database")
    
    if not chunks:
        print("❌ No chunks found for search testing")
        return False
    
    # Show sample content
    print(f"\n📝 Sample chunk content:")
    sample_chunk = chunks[0]
    print(f"   ID: {sample_chunk.id}")
    print(f"   Type: {sample_chunk.chunk_type.value}")
    print(f"   Length: {len(sample_chunk.text)} characters")
    print(f"   Preview: {sample_chunk.text[:150]}...")
    
    # Test different search queries
    test_queries = [
        "Python",
        "algorithmic trading",
        "O'Reilly",
        "educational",
        "business",
        "sales promotional"
    ]
    
    print(f"\n🔍 Testing search queries:")
    
    for query in test_queries:
        print(f"\n🎯 Query: '{query}'")
        
        try:
            # Test text search
            search_result = text_search.search_exact(query, num_results=3)
            
            results = search_result.get('results', [])
            
            print(f"   📊 Found {len(results)} results in {search_result.get('search_time_ms', 0)}ms")
            
            for i, result in enumerate(results, 1):
                print(f"   {i}. Score: {result['score']:.3f}")
                print(f"      Chunk: {result['chunk_id']}")
                print(f"      Preview: {result['text'][:100]}...")
                print(f"      Book: {result['metadata'].get('book_id', 'Unknown')}")
                if 'snippet' in result:
                    print(f"      Snippet: {result['snippet']}")
                print()
        
        except Exception as e:
            print(f"   ❌ Search error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"📊 Search testing completed for {len(test_queries)} queries")
    
    return True

async def test_hybrid_search():
    """Test hybrid search if available"""
    
    print(f"\n🔄 Testing hybrid search functionality...")
    
    try:
        from src.search.hybrid_search import HybridSearch
        
        hybrid_search = HybridSearch()
        await hybrid_search.initialize()
        
        # Test a simple query
        query = "Python algorithmic trading"
        print(f"🎯 Hybrid query: '{query}'")
        
        results = await hybrid_search.search(query, max_results=3)
        
        print(f"📊 Hybrid search found {len(results)} results")
        
        for i, result in enumerate(results, 1):
            print(f"   {i}. Score: {result.score:.3f}")
            print(f"      Type: {result.search_type}")
            print(f"      Preview: {result.content[:100]}...")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Hybrid search not available or failed: {e}")
        return False

async def main():
    """Main test function"""
    
    print("🚀 Starting search functionality tests")
    print("=" * 50)
    
    # Test basic search
    basic_success = await test_search()
    
    if basic_success:
        print("✅ Basic search functionality working!")
    else:
        print("❌ Basic search failed!")
        return
    
    # Test hybrid search
    hybrid_success = await test_hybrid_search()
    
    if hybrid_success:
        print("✅ Hybrid search functionality working!")
    else:
        print("⚠️  Hybrid search not available (may need embeddings)")
    
    print("\n" + "=" * 50)
    print("🏆 Search testing completed!")
    
    if basic_success:
        print("\n🔍 Next steps:")
        print("   • Basic text search is working with existing chunks")
        print("   • You can search for content within the processed book")
        print("   • Consider adding more books or completing the chunking")
        print("   • For semantic search, you may need to generate embeddings")

if __name__ == "__main__":
    asyncio.run(main())