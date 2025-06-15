#!/usr/bin/env python3
"""
Set up persistent Qdrant vector database
"""

import asyncio
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

sys.path.append('src')

class PersistentQdrantSetup:
    def __init__(self):
        self.client = None
        self.collection_name = "tradeknowledge_persistent"
        self.qdrant_path = Path("data/qdrant")
        self.embedding_dim = 384
        
    async def setup_persistent_qdrant(self):
        """Set up persistent Qdrant vector database"""
        
        print("🚀 Setting up Persistent Qdrant Vector Database")
        print("=" * 60)
        
        # Create Qdrant directory
        self.qdrant_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created Qdrant storage directory: {self.qdrant_path}")
        
        # Initialize Qdrant client with persistent storage
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        
        print("🔧 Initializing persistent Qdrant client...")
        # Use file-based storage explicitly
        self.client = QdrantClient(path=str(self.qdrant_path))
        
        # Check if collection already exists
        try:
            collections = self.client.get_collections()
            collection_exists = any(c.name == self.collection_name for c in collections.collections)
            
            if collection_exists:
                print(f"📦 Collection '{self.collection_name}' already exists")
                
                # Get existing collection info
                info = self.client.get_collection(self.collection_name)
                print(f"   • Vectors: {info.points_count}")
                print(f"   • Dimension: {info.config.params.vectors.size}")
                print(f"   • Distance: {info.config.params.vectors.distance}")
                
                return True
            else:
                print(f"📦 Creating new collection: {self.collection_name}")
                
        except Exception as e:
            print(f"⚠️  Error checking collections: {e}")
            print("📦 Creating new collection...")
        
        # Create collection
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            print("✅ Collection created successfully")
            
        except Exception as e:
            print(f"❌ Error creating collection: {e}")
            return False
        
        return True
    
    async def load_embeddings_to_persistent_db(self):
        """Load embeddings into persistent database"""
        
        print("\n📤 Loading embeddings into persistent database...")
        
        # Load embeddings from files
        numpy_file = Path("data/embeddings/chunk_embeddings.npz")
        json_file = Path("data/embeddings/chunk_embeddings.json")
        
        if not numpy_file.exists() or not json_file.exists():
            print("❌ Embedding files not found. Run generate_embeddings.py first.")
            return False
        
        # Load data
        numpy_data = np.load(numpy_file)
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        
        chunk_ids = numpy_data['chunk_ids']
        embeddings = numpy_data['embeddings']
        
        print(f"📂 Loaded {len(chunk_ids)} embeddings from files")
        
        # Get chunk metadata from SQLite
        from src.core.sqlite_storage import SQLiteStorage
        storage = SQLiteStorage()
        
        # Prepare points for upload
        from qdrant_client.models import PointStruct
        
        points = []
        for i, chunk_id in enumerate(chunk_ids):
            try:
                # Get chunk details
                chunk = await storage.get_chunk(str(chunk_id))
                if chunk:
                    # Get book info
                    book = await storage.get_book(chunk.book_id)
                    book_title = book.title if book else "Unknown"
                    
                    point = PointStruct(
                        id=i,
                        vector=embeddings[i].tolist(),
                        payload={
                            'chunk_id': str(chunk_id),
                            'book_id': chunk.book_id,
                            'book_title': book_title,
                            'chunk_index': chunk.chunk_index,
                            'chunk_type': chunk.chunk_type.value,
                            'text': chunk.text,
                            'text_length': len(chunk.text),
                            'preview': chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
                        }
                    )
                    points.append(point)
                else:
                    print(f"⚠️  Chunk not found: {chunk_id}")
                    
            except Exception as e:
                print(f"❌ Error processing chunk {chunk_id}: {e}")
        
        if not points:
            print("❌ No valid points to upload")
            return False
        
        # Upload in batches
        batch_size = 50
        total_points = len(points)
        
        print(f"📦 Uploading {total_points} points in batches of {batch_size}")
        
        uploaded_count = 0
        for batch_start in range(0, total_points, batch_size):
            batch_end = min(batch_start + batch_size, total_points)
            batch_points = points[batch_start:batch_end]
            
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch_points
                )
                uploaded_count += len(batch_points)
                print(f"  ✅ Uploaded batch {batch_start+1}-{batch_end}")
                
            except Exception as e:
                print(f"  ❌ Error uploading batch {batch_start+1}-{batch_end}: {e}")
        
        print(f"✅ Successfully uploaded {uploaded_count}/{total_points} embeddings")
        return uploaded_count > 0
    
    async def test_persistent_search(self):
        """Test search functionality with persistent database"""
        
        print("\n🔍 Testing persistent vector search...")
        
        # Load model for query
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
        
        test_queries = [
            "Python algorithmic trading",
            "risk management strategies", 
            "portfolio optimization"
        ]
        
        for query in test_queries:
            print(f"\n🎯 Query: '{query}'")
            
            # Generate query embedding
            query_embedding = model.encode([query], convert_to_numpy=True)[0]
            
            # Search in persistent database
            try:
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding.tolist(),
                    limit=3
                )
                
                print(f"📊 Found {len(results)} results:")
                
                for i, result in enumerate(results, 1):
                    score = result.score
                    payload = result.payload
                    
                    print(f"  {i}. Score: {score:.4f}")
                    print(f"     Chunk: {payload.get('chunk_id', 'unknown')}")
                    print(f"     Book: {payload.get('book_title', 'unknown')}")
                    print(f"     Preview: {payload.get('preview', 'No preview')[:100]}...")
                    print()
                
            except Exception as e:
                print(f"❌ Search error: {e}")
    
    def get_persistent_database_info(self):
        """Get information about the persistent database"""
        
        print("\n📊 PERSISTENT DATABASE INFORMATION")
        print("=" * 60)
        
        try:
            # Collection info
            info = self.client.get_collection(self.collection_name)
            
            print(f"Collection Details:")
            print(f"  • Name: {self.collection_name}")
            print(f"  • Vector dimension: {info.config.params.vectors.size}")
            print(f"  • Distance metric: {info.config.params.vectors.distance}")
            print(f"  • Total vectors: {info.points_count}")
            print(f"  • Status: {info.status}")
            
            # Storage info
            storage_size = sum(
                f.stat().st_size for f in self.qdrant_path.rglob('*') if f.is_file()
            ) / (1024 * 1024)  # Convert to MB
            
            print(f"\nStorage Details:")
            print(f"  • Location: {self.qdrant_path}")
            print(f"  • Storage size: {storage_size:.2f} MB")
            print(f"  • Files: {len(list(self.qdrant_path.rglob('*')))}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error getting database info: {e}")
            return False
    
    async def verify_persistence(self):
        """Verify that the database persists across restarts"""
        
        print("\n🔄 Testing persistence...")
        
        # Close current client
        self.client = None
        
        # Reinitialize client
        from qdrant_client import QdrantClient
        self.client = QdrantClient(path=str(self.qdrant_path))
        
        try:
            # Check if collection still exists
            collections = self.client.get_collections()
            collection_exists = any(c.name == self.collection_name for c in collections.collections)
            
            if collection_exists:
                info = self.client.get_collection(self.collection_name)
                print(f"✅ Persistence verified!")
                print(f"   • Collection exists after restart")
                print(f"   • Vector count: {info.points_count}")
                return True
            else:
                print("❌ Persistence failed - collection not found")
                return False
                
        except Exception as e:
            print(f"❌ Persistence test failed: {e}")
            return False

async def main():
    """Main setup function"""
    
    setup = PersistentQdrantSetup()
    
    # Set up persistent database
    success = await setup.setup_persistent_qdrant()
    if not success:
        print("❌ Failed to set up persistent database")
        return
    
    # Load embeddings
    loaded = await setup.load_embeddings_to_persistent_db()
    if not loaded:
        print("❌ Failed to load embeddings")
        return
    
    # Get database info
    setup.get_persistent_database_info()
    
    # Test search functionality
    await setup.test_persistent_search()
    
    # Verify persistence
    await setup.verify_persistence()
    
    print("\n🏆 Persistent Qdrant setup completed successfully!")
    print("\n✅ Your vector database is now:")
    print("   • Persistent (survives restarts)")
    print("   • Ready for production use")
    print("   • Accessible at: data/qdrant/")
    print("   • Searchable via semantic queries")
    
    print("\n🔍 Next steps:")
    print("   • Integrate with main application")
    print("   • Add hybrid search combining text + vector")
    print("   • Scale up with more book content")

if __name__ == "__main__":
    asyncio.run(main())