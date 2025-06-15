#!/usr/bin/env python3
"""
Debug configuration loading
"""

import sys
import os

# Add src to path
sys.path.append('src')

from src.core.config import get_config

def debug_config():
    """Debug configuration values"""
    
    config = get_config()
    
    print("🔍 Configuration Debug:")
    print(f"  EMBEDDING_DIMENSION env: {os.getenv('EMBEDDING_DIMENSION', 'NOT SET')}")
    print(f"  Config embedding dimension: {config.embedding.dimension}")
    print(f"  Embedding model: {config.embedding.model}")
    print(f"  Qdrant collection: {config.database.qdrant.collection_name}")
    
    # Check embedding config in detail
    print("\n📊 Embedding Config:")
    for attr in dir(config.embedding):
        if not attr.startswith('_'):
            value = getattr(config.embedding, attr)
            if not callable(value):
                print(f"  {attr}: {value}")

if __name__ == "__main__":
    debug_config()