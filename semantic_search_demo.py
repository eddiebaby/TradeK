#!/usr/bin/env python3
"""
Semantic Search Demo for TradeKnowledge
Shows complete integration of vector embeddings
"""

import asyncio
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

sys.path.append('src')

class SemanticSearchDemo:
    def __init__(self):
        self.qdrant_client = None
        self.semantic_model = None
        self.collection_name = "tradeknowledge_demo"
        
    async def initialize(self):
        """Initialize semantic search system"""
        print("🚀 Initializing Semantic Search Demo")
        print("=" * 50)
        
        # Initialize Qdrant
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        
        print("📦 Setting up Qdrant vector database...")
        self.qdrant_client = QdrantClient(location=":memory:")
        
        # Create collection
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        
        # Load embeddings
        print("📂 Loading embeddings...")
        await self._load_embeddings()
        
        # Load semantic model
        print("🧠 Loading semantic model...")
        from sentence_transformers import SentenceTransformer
        self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
        
        print("✅ Semantic search system ready!")
    
    async def _load_embeddings(self):
        """Load embeddings into Qdrant"""
        # Load from files
        numpy_file = Path("data/embeddings/chunk_embeddings.npz")
        json_file = Path("data/embeddings/chunk_embeddings.json")
        
        if not numpy_file.exists():
            print("❌ No embeddings found. Run generate_embeddings.py first.")
            return
        
        # Load data
        numpy_data = np.load(numpy_file)
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        
        chunk_ids = numpy_data['chunk_ids']
        embeddings = numpy_data['embeddings']
        
        # Add chunk metadata
        sys.path.append('src')
        from src.core.sqlite_storage import SQLiteStorage
        storage = SQLiteStorage()
        
        points = []
        for i, chunk_id in enumerate(chunk_ids):
            # Get chunk details
            chunk = await storage.get_chunk(str(chunk_id))
            if chunk:
                point = {
                    'id': i,
                    'vector': embeddings[i].tolist(),
                    'payload': {
                        'chunk_id': str(chunk_id),
                        'book_id': chunk.book_id,
                        'text': chunk.text,
                        'preview': chunk.text[:150] + "..." if len(chunk.text) > 150 else chunk.text
                    }
                }
                points.append(point)
        
        # Upload to Qdrant
        from qdrant_client.models import PointStruct
        qdrant_points = [
            PointStruct(
                id=p['id'],
                vector=p['vector'],
                payload=p['payload']
            ) for p in points
        ]
        
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=qdrant_points
        )
        
        print(f"📤 Loaded {len(points)} embeddings into vector database")
    
    async def semantic_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search"""
        
        print(f"\n🔍 Semantic Search: '{query}'")
        print("-" * 40)
        
        # Generate query embedding
        query_embedding = self.semantic_model.encode([query], convert_to_numpy=True)[0]
        
        # Search in Qdrant
        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=limit
        )
        
        # Format and display results
        formatted_results = []
        for i, result in enumerate(results, 1):
            score = result.score
            payload = result.payload
            
            formatted_result = {
                'rank': i,
                'score': score,
                'chunk_id': payload['chunk_id'],
                'book_id': payload['book_id'],
                'text': payload['text'],
                'preview': payload['preview']
            }
            formatted_results.append(formatted_result)
            
            # Display result
            print(f"{i}. Score: {score:.4f}")
            print(f"   Chunk: {payload['chunk_id']}")
            print(f"   Book: {payload['book_id']}")
            print(f"   Preview: {payload['preview']}")
            print()
        
        return formatted_results
    
    async def demo_queries(self):
        """Run demo queries to showcase semantic search"""
        
        print("\n🎯 SEMANTIC SEARCH DEMONSTRATIONS")
        print("=" * 50)
        
        demo_queries = [
            {
                'query': 'machine learning for trading',
                'description': 'Conceptual search for ML trading content'
            },
            {
                'query': 'managing financial risk',
                'description': 'Risk management concepts'
            },
            {
                'query': 'portfolio optimization techniques',
                'description': 'Portfolio theory and optimization'
            },
            {
                'query': 'backtesting investment strategies',
                'description': 'Strategy testing methodologies'
            },
            {
                'query': 'quantitative analysis methods',
                'description': 'Quantitative finance approaches'
            }
        ]
        
        for demo in demo_queries:
            print(f"\n📋 Demo: {demo['description']}")
            results = await self.semantic_search(demo['query'])
            
            if results:
                best_match = results[0]
                print(f"💡 Best match: {best_match['chunk_id']} (Score: {best_match['score']:.4f})")
            
            print("\n" + "."*50)
    
    def get_system_info(self):
        """Display system information"""
        
        print("\n📊 SYSTEM INFORMATION")
        print("=" * 50)
        
        # Collection info
        try:
            info = self.qdrant_client.get_collection(self.collection_name)
            print(f"Vector Database:")
            print(f"  • Collection: {self.collection_name}")
            print(f"  • Vector dimension: {info.config.params.vectors.size}")
            print(f"  • Distance metric: {info.config.params.vectors.distance}")
            print(f"  • Total vectors: {info.points_count}")
        except Exception as e:
            print(f"  • Error getting collection info: {e}")
        
        # Model info
        print(f"\nSemantic Model:")
        print(f"  • Model: all-MiniLM-L6-v2")
        print(f"  • Type: Sentence Transformer")
        print(f"  • Embedding dimension: 384")
        print(f"  • Device: CPU")
        
        # Files info
        embeddings_dir = Path("data/embeddings")
        if embeddings_dir.exists():
            files = list(embeddings_dir.glob("*"))
            print(f"\nEmbedding Files:")
            for file in files:
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  • {file.name}: {size_mb:.2f}MB")

async def main():
    """Main demo function"""
    
    # Initialize system
    demo = SemanticSearchDemo()
    await demo.initialize()
    
    # Show system info
    demo.get_system_info()
    
    # Run demonstrations
    await demo.demo_queries()
    
    # Interactive mode
    print("\n🎮 INTERACTIVE MODE")
    print("=" * 50)
    print("Enter your semantic search queries (or 'quit' to exit):")
    
    while True:
        try:
            query = input("\n🔍 Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            await demo.semantic_search(query)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n🏆 Semantic Search Demo completed!")
    print("\n📈 Key Capabilities Demonstrated:")
    print("   ✅ Vector embeddings generation")
    print("   ✅ Semantic similarity search")
    print("   ✅ Conceptual query understanding")
    print("   ✅ Multi-book content search")
    print("   ✅ Real-time query processing")

if __name__ == "__main__":
    asyncio.run(main())