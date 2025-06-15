#!/usr/bin/env python3
"""
Fix Qdrant collection with correct 384 dimensions
"""

import sys
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

def fix_qdrant():
    """Fix Qdrant collection with 384 dimensions"""
    
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "tradeknowledge"
    
    print(f"🔧 Fixing Qdrant collection: {collection_name}")
    
    # Delete existing collection
    try:
        if client.collection_exists(collection_name):
            print("🗑️  Deleting existing collection...")
            client.delete_collection(collection_name)
            print("✅ Collection deleted")
    except Exception as e:
        print(f"⚠️  Error: {e}")
    
    # Create new collection with 768 dimensions (actual embedding size)
    print("🆕 Creating collection with 768 dimensions...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=768,  # actual embedding dimension
            distance=Distance.COSINE
        )
    )
    
    # Verify
    info = client.get_collection(collection_name)
    print(f"✅ Collection created!")
    print(f"  Dimensions: {info.config.params.vectors.size}")
    print(f"  Distance: {info.config.params.vectors.distance}")
    
    return True

if __name__ == "__main__":
    fix_qdrant()