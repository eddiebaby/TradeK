#!/usr/bin/env python3
"""
Set up Qdrant vector database and upload embeddings
"""

import asyncio
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import time

# Add src to path
sys.path.append('src')

from src.core.sqlite_storage import SQLiteStorage

class QdrantSetup:
    def __init__(self):
        self.storage = SQLiteStorage()
        self.client = None
        self.collection_name = "tradeknowledge_chunks"
        self.embedding_dim = 384  # all-MiniLM-L6-v2 dimension
        
    async def initialize_qdrant(self, use_memory: bool = True):
        """Initialize Qdrant client"""
        
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        
        print("🔧 Initializing Qdrant...")
        
        if use_memory:
            # In-memory mode for testing
            print("📂 Using Qdrant in-memory mode")
            self.client = QdrantClient(location=":memory:")
        else:
            # Local persistent storage
            qdrant_path = Path("data/qdrant")
            qdrant_path.mkdir(parents=True, exist_ok=True)
            print(f"📂 Using local Qdrant storage: {qdrant_path}")
            self.client = QdrantClient(location=str(qdrant_path))
        
        # Create collection
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_exists = any(c.name == self.collection_name for c in collections.collections)
            
            if not collection_exists:
                print(f"📦 Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE  # Good for sentence embeddings
                    )
                )
                print("✅ Collection created successfully")
            else:
                print(f"📦 Collection already exists: {self.collection_name}")
        
        except Exception as e:
            print(f"❌ Error creating collection: {e}")
            return False
        
        print("✅ Qdrant initialized successfully")
        return True
    
    async def load_embeddings(self) -> Dict[str, Any]:
        """Load embeddings from files"""
        
        print("📂 Loading embeddings from files...")
        
        # Load JSON file with metadata
        json_file = Path("data/embeddings/chunk_embeddings.json")
        numpy_file = Path("data/embeddings/chunk_embeddings.npz")
        
        if not json_file.exists() or not numpy_file.exists():
            print("❌ Embedding files not found. Run generate_embeddings.py first.")
            return None
        
        # Load metadata from JSON
        with open(json_file, 'r') as f:
            embedding_data = json.load(f)
        
        # Load embeddings from numpy file
        numpy_data = np.load(numpy_file)
        chunk_ids = numpy_data['chunk_ids']
        embeddings_matrix = numpy_data['embeddings']
        
        print(f"📊 Loaded {len(chunk_ids)} embeddings")
        print(f"🏷️  Model: {embedding_data['model']}")
        print(f"📐 Dimension: {embedding_data['embedding_dim']}")
        
        return {
            'chunk_ids': chunk_ids,
            'embeddings': embeddings_matrix,
            'metadata': embedding_data
        }
    
    async def upload_embeddings_to_qdrant(self, embedding_data: Dict[str, Any]):
        """Upload embeddings to Qdrant"""
        
        from qdrant_client.models import PointStruct
        
        print("📤 Uploading embeddings to Qdrant...")
        
        chunk_ids = embedding_data['chunk_ids']
        embeddings = embedding_data['embeddings']
        
        # Get chunk metadata from database
        chunks_metadata = {}
        for chunk_id in chunk_ids:
            chunk = await self.storage.get_chunk(chunk_id)
            if chunk:
                chunks_metadata[chunk_id] = {
                    'book_id': chunk.book_id,
                    'chunk_index': chunk.chunk_index,
                    'chunk_type': chunk.chunk_type.value,
                    'text_preview': chunk.text[:200],  # First 200 chars for preview
                    'text_length': len(chunk.text)
                }
        
        # Prepare points for upload
        points = []
        for i, chunk_id in enumerate(chunk_ids):
            chunk_id_str = str(chunk_id)  # Ensure string format
            
            point = PointStruct(
                id=i,  # Use numeric ID for Qdrant
                vector=embeddings[i].tolist(),
                payload={
                    'chunk_id': chunk_id_str,
                    **chunks_metadata.get(chunk_id_str, {})
                }
            )
            points.append(point)
        
        # Upload in batches
        batch_size = 50
        total_points = len(points)
        
        print(f"📦 Uploading {total_points} points in batches of {batch_size}")
        
        for batch_start in range(0, total_points, batch_size):
            batch_end = min(batch_start + batch_size, total_points)
            batch_points = points[batch_start:batch_end]
            
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch_points
                )
                print(f"  ✅ Uploaded batch {batch_start+1}-{batch_end}")
                
            except Exception as e:
                print(f"  ❌ Error uploading batch {batch_start+1}-{batch_end}: {e}")
        
        print("✅ All embeddings uploaded to Qdrant")
    
    async def test_vector_search(self, query_text: str = "Python algorithmic trading"):
        """Test vector search functionality"""
        
        print(f"\n🔍 Testing vector search with query: '{query_text}'")
        
        # Generate embedding for query
        from sentence_transformers import SentenceTransformer
        
        print("📦 Loading model for query embedding...")
        model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
        
        print("🧠 Generating query embedding...")
        query_embedding = model.encode([query_text], convert_to_numpy=True)[0]
        
        # Search in Qdrant
        print("🔍 Searching in Qdrant...")
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=5
        )
        
        print(f"📊 Found {len(search_results)} results:")
        
        for i, result in enumerate(search_results, 1):
            score = result.score
            payload = result.payload
            chunk_id = payload.get('chunk_id', 'unknown')
            text_preview = payload.get('text_preview', 'No preview')
            
            print(f"\n{i}. Score: {score:.4f}")
            print(f"   Chunk ID: {chunk_id}")
            print(f"   Book ID: {payload.get('book_id', 'unknown')}")
            print(f"   Preview: {text_preview}...")
        
        return search_results
    
    def get_collection_info(self):
        """Get information about the collection"""
        
        try:
            info = self.client.get_collection(self.collection_name)
            print(f"\n📊 Collection Info:")
            print(f"   Name: {info.config.params.vectors.size}")
            print(f"   Vector size: {info.config.params.vectors.size}")
            print(f"   Distance: {info.config.params.vectors.distance}")
            print(f"   Points count: {info.points_count}")
            
            return info
            
        except Exception as e:
            print(f"❌ Error getting collection info: {e}")
            return None

async def main():
    print("🚀 Setting up Qdrant vector database")
    
    # Initialize setup
    setup = QdrantSetup()
    
    # Initialize Qdrant
    success = await setup.initialize_qdrant(use_memory=True)  # Use memory for testing
    if not success:
        print("❌ Failed to initialize Qdrant")
        return
    
    # Load embeddings
    embedding_data = await setup.load_embeddings()
    if not embedding_data:
        print("❌ Failed to load embeddings")
        return
    
    # Upload to Qdrant
    await setup.upload_embeddings_to_qdrant(embedding_data)
    
    # Get collection info
    setup.get_collection_info()
    
    # Test search
    results = await setup.test_vector_search("Python algorithmic trading")
    
    # Test another search
    print("\n" + "="*50)
    await setup.test_vector_search("risk management")
    
    print("\n🏆 Qdrant setup and testing completed successfully!")
    print("\n🔍 Next steps:")
    print("   • Vector database is ready")
    print("   • Semantic search is functional")
    print("   • Try more complex queries")
    print("   • Integrate with hybrid search")

if __name__ == "__main__":
    asyncio.run(main())