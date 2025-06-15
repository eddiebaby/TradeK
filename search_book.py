#!/usr/bin/env python3
"""
Search the complete Python for Algorithmic Trading book
"""

import sys
import asyncio
from sentence_transformers import SentenceTransformer

# Add src to path
sys.path.append('src')

from src.core.qdrant_storage import QdrantStorage

class BookSearcher:
    def __init__(self):
        """Initialize the book searcher"""
        self.storage = QdrantStorage()
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        print("🔍 Book searcher initialized")
    
    async def search(self, query, limit=5):
        """Search the book with semantic similarity"""
        print(f"\n🔍 Searching for: \"{query}\"")
        print("="*50)
        
        # Generate embedding for the query
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0].tolist()
        
        # Search in vector database
        results = await self.storage.search_semantic(query_embedding, limit=limit)
        
        if not results:
            print("❌ No results found")
            return
        
        print(f"📊 Found {len(results)} results:\n")
        
        for i, result in enumerate(results, 1):
            score = result['score']
            chunk_id = result['chunk_id']
            text = result['text']
            metadata = result['metadata']
            
            # Extract page number from chunk ID
            page_num = "Unknown"
            if "page_" in chunk_id:
                try:
                    page_num = chunk_id.split("page_")[1]
                    page_num = f"Page {int(page_num)}"
                except:
                    pass
            
            print(f"🎯 Result {i} (Relevance: {score:.1%})")
            print(f"📄 {page_num}")
            print(f"📝 Text preview:")
            
            # Show first 300 characters with word boundary
            preview = text[:300]
            if len(text) > 300:
                # Find last complete word
                last_space = preview.rfind(' ')
                if last_space > 200:  # Only if we have a reasonable amount
                    preview = preview[:last_space] + "..."
                else:
                    preview += "..."
            
            print(f"   {preview}")
            print()
    
    async def interactive_search(self):
        """Interactive search mode"""
        print("🚀 Python for Algorithmic Trading - Interactive Search")
        print("Type your search queries. Press Ctrl+C to exit.")
        print("Examples:")
        print("  - 'machine learning strategies'")
        print("  - 'risk management techniques'")
        print("  - 'backtesting methods'")
        print("  - 'portfolio optimization'")
        print()
        
        try:
            while True:
                query = input("🔍 Enter search query: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                await self.search(query)
                print("\n" + "="*60 + "\n")
                
        except KeyboardInterrupt:
            print("\n\n👋 Search session ended")
    
    async def show_stats(self):
        """Show collection statistics"""
        print("📊 Book Collection Statistics")
        print("="*35)
        
        stats = await self.storage.get_collection_stats()
        
        print(f"📚 Total pages indexed: {stats['total_embeddings']}")
        print(f"🧠 Vector dimension: {stats['vector_size']}")
        print(f"📏 Distance metric: {stats['distance_metric']}")
        print(f"✅ Status: {stats['status']}")
        print()

async def main():
    """Main search interface"""
    if len(sys.argv) > 1:
        # Command line search
        query = " ".join(sys.argv[1:])
        searcher = BookSearcher()
        await searcher.show_stats()
        await searcher.search(query)
    else:
        # Interactive mode
        searcher = BookSearcher()
        await searcher.show_stats()
        await searcher.interactive_search()

if __name__ == "__main__":
    asyncio.run(main())