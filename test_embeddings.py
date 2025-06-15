#!/usr/bin/env python3
"""
Test embedding generation dimensions
"""

import sys
import asyncio

# Add src to path
sys.path.append('src')

from src.ingestion.local_embeddings import LocalEmbeddingGenerator
from src.core.models import Chunk

async def test_embeddings():
    """Test embedding generation"""
    
    # Create a dummy chunk
    chunk = Chunk(
        id="test",
        book_id="test_book",
        text="This is a test chunk for embedding generation.",
        chunk_index=0,
        page_start=1,
        page_end=1,
        chunk_type="text"
    )
    
    print("🧠 Testing embedding generation...")
    
    generator = LocalEmbeddingGenerator()
    
    async with generator as gen:
        embeddings = await gen.generate_embeddings([chunk])
        
        if embeddings:
            print(f"✅ Generated {len(embeddings)} embeddings")
            print(f"📏 Dimension: {len(embeddings[0])}")
            print(f"📊 First 5 values: {embeddings[0][:5]}")
        else:
            print("❌ No embeddings generated")

if __name__ == "__main__":
    asyncio.run(test_embeddings())