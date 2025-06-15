#!/usr/bin/env python3
"""
Populate persistent Qdrant with existing embeddings
"""

import asyncio
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import uuid

# Add src to path
sys.path.append('src')

from src.core.qdrant_storage import QdrantStorage
from src.core.models import Chunk, ChunkType
from datetime import datetime

async def populate_persistent_qdrant():
    """Populate the persistent Qdrant service with existing embeddings"""
    
    print("🔧 Loading existing embeddings...")
    
    # Load embeddings
    json_file = Path("data/embeddings/chunk_embeddings.json")
    numpy_file = Path("data/embeddings/chunk_embeddings.npz")
    
    if not json_file.exists() or not numpy_file.exists():
        print("❌ Embedding files not found")
        return False
    
    # Load metadata from JSON
    with open(json_file, 'r') as f:
        metadata = json.load(f)
    
    # Load embeddings from numpy file
    numpy_data = np.load(numpy_file)
    chunk_ids = numpy_data['chunk_ids']
    embeddings_matrix = numpy_data['embeddings']
    
    print(f"📊 Found {len(chunk_ids)} embeddings")
    print(f"🏷️  Model: {metadata['model']}")
    print(f"📐 Dimension: {metadata['embedding_dim']}")
    
    # Initialize Qdrant storage (connects to persistent service)
    print("🔌 Connecting to persistent Qdrant...")
    storage = QdrantStorage()
    
    # Check current stats
    stats = await storage.get_collection_stats()
    print(f"📊 Current collection stats: {stats}")
    
    # Create dummy chunks for the embeddings
    chunks = []
    for i, chunk_id in enumerate(chunk_ids):
        chunk = Chunk(
            id=str(chunk_id),
            book_id="test_algo_trading_book" if chunk_id.startswith("chunk_") else "Python_for_Algorithmic_Trading_9b87c397",
            chunk_index=i,
            chunk_type=ChunkType.TEXT,
            text=f"Sample text for chunk {chunk_id}",  # This would come from database normally
            created_at=datetime.now(),
            metadata={"source": "existing_embeddings"}
        )
        chunks.append(chunk)
    
    # Convert embeddings to list format
    embeddings_list = [embedding.tolist() for embedding in embeddings_matrix]
    
    # Save to persistent Qdrant
    print("📤 Uploading to persistent Qdrant...")
    success = await storage.save_embeddings(chunks, embeddings_list)
    
    if success:
        print("✅ Successfully populated persistent Qdrant")
        
        # Get updated stats
        stats = await storage.get_collection_stats()
        print(f"📊 Updated collection stats: {stats}")
        
        # Test search
        print("\n🔍 Testing search...")
        test_embedding = embeddings_matrix[0].tolist()  # Use first embedding as test
        results = await storage.search_semantic(test_embedding, limit=3)
        
        print(f"📊 Search results: {len(results)} found")
        for i, result in enumerate(results[:3], 1):
            print(f"  {i}. Score: {result['score']:.4f}, Chunk: {result['chunk_id']}")
        
        return True
    else:
        print("❌ Failed to populate Qdrant")
        return False

if __name__ == "__main__":
    result = asyncio.run(populate_persistent_qdrant())
    if result:
        print("\n🏆 Persistent Qdrant population completed successfully!")
    else:
        print("\n❌ Failed to populate persistent Qdrant")