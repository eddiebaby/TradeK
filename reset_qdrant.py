#!/usr/bin/env python3
"""
Reset Qdrant collection with correct dimensions
"""

import sys
import asyncio

# Add src to path
sys.path.append('src')

from src.core.qdrant_storage import QdrantStorage
from src.core.config import get_config

async def reset_qdrant():
    """Reset Qdrant collection with correct dimensions"""
    
    config = get_config()
    
    print(f"🔧 Resetting Qdrant collection...")
    print(f"Collection: {config.database.qdrant.collection_name}")
    print(f"Dimensions: {config.embedding.dimension}")
    print(f"Host: {config.database.qdrant.host}:{config.database.qdrant.port}")
    
    storage = QdrantStorage()
    
    # Delete existing collection if it exists
    try:
        if storage.client.collection_exists(storage.collection_name):
            print(f"🗑️  Deleting existing collection...")
            storage.client.delete_collection(storage.collection_name)
            print("✅ Collection deleted")
        else:
            print("ℹ️  Collection doesn't exist yet")
    except Exception as e:
        print(f"⚠️  Error checking collection: {e}")
    
    # Recreate collection with correct dimensions
    print(f"🆕 Creating collection with {config.embedding.dimension} dimensions...")
    storage._ensure_collection()
    
    # Verify collection
    info = storage.client.get_collection(storage.collection_name)
    print(f"✅ Collection created successfully!")
    print(f"  Dimensions: {info.config.params.vectors.size}")
    print(f"  Distance: {info.config.params.vectors.distance}")
    
    return True

if __name__ == "__main__":
    asyncio.run(reset_qdrant())