#!/usr/bin/env python3
"""
Update vector database with all embeddings
"""

import sys
import sqlite3
import json
import numpy as np
import asyncio
from datetime import datetime

# Add src to path
sys.path.append('src')

from src.core.qdrant_storage import QdrantStorage
from src.core.models import Chunk, ChunkType

async def update_vector_database():
    """Update Qdrant with all embeddings"""
    
    print("🔄 Updating vector database with all embeddings...")
    
    # Load embeddings
    embedding_file = '/home/scottschweizer/TradeKnowledge/data/embeddings/chunk_embeddings.npz'
    db_path = '/home/scottschweizer/TradeKnowledge/data/knowledge.db'
    
    print("📂 Loading embeddings...")
    data = np.load(embedding_file)
    chunk_ids = data['chunk_ids']
    embeddings_matrix = data['embeddings']
    
    print(f"📊 Loaded {len(chunk_ids)} embeddings")
    
    # Get chunk metadata from database
    print("🔍 Loading chunk metadata from database...")
    chunk_metadata = {}
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all chunks
        cursor.execute("""
            SELECT c.id, c.book_id, c.chunk_index, c.chunk_type, c.text, c.metadata,
                   b.title, b.author
            FROM chunks c
            LEFT JOIN books b ON c.book_id = b.id
            ORDER BY c.book_id, c.chunk_index
        """)
        
        chunks_data = cursor.fetchall()
        
        for chunk_data in chunks_data:
            chunk_id, book_id, chunk_index, chunk_type, text, metadata_json, book_title, book_author = chunk_data
            
            # Parse metadata
            try:
                metadata = json.loads(metadata_json) if metadata_json else {}
            except:
                metadata = {}
            
            chunk_metadata[chunk_id] = {
                'book_id': book_id,
                'book_title': book_title or 'Unknown',
                'book_author': book_author or 'Unknown',
                'chunk_index': chunk_index,
                'chunk_type': chunk_type,
                'text': text,
                'metadata': metadata
            }
        
        conn.close()
        print(f"✅ Loaded metadata for {len(chunk_metadata)} chunks")
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False
    
    # Initialize Qdrant storage
    print("🔌 Connecting to Qdrant...")
    storage = QdrantStorage()
    
    # Clear existing collection
    print("🧹 Clearing existing collection...")
    try:
        storage.client.delete_collection('tradeknowledge')
        print("  ✅ Deleted old collection")
    except:
        pass
    
    # Recreate collection with correct dimensions
    from qdrant_client.models import Distance, VectorParams
    storage.client.create_collection(
        collection_name='tradeknowledge',
        vectors_config=VectorParams(
            size=embeddings_matrix.shape[1],
            distance=Distance.COSINE
        )
    )
    print(f"  ✅ Created new collection (dimension: {embeddings_matrix.shape[1]})")
    
    # Prepare chunks for upload
    print("📦 Preparing chunks for upload...")
    chunks = []
    embeddings_list = []
    
    for i, chunk_id in enumerate(chunk_ids):
        chunk_id_str = str(chunk_id)
        
        if chunk_id_str in chunk_metadata:
            meta = chunk_metadata[chunk_id_str]
            
            chunk = Chunk(
                id=chunk_id_str,
                book_id=meta['book_id'],
                chunk_index=meta['chunk_index'],
                chunk_type=ChunkType.TEXT,
                text=meta['text'],
                created_at=datetime.now(),
                metadata={
                    **meta['metadata'],
                    'book_title': meta['book_title'],
                    'book_author': meta['book_author']
                }
            )
            
            chunks.append(chunk)
            embeddings_list.append(embeddings_matrix[i].tolist())
        else:
            print(f"  ⚠️  No metadata found for chunk: {chunk_id_str}")
    
    print(f"📊 Prepared {len(chunks)} chunks for upload")
    
    # Upload to Qdrant
    print("📤 Uploading to Qdrant...")
    success = await storage.save_embeddings(chunks, embeddings_list)
    
    if success:
        print("✅ Successfully uploaded all embeddings!")
        
        # Get final stats
        stats = await storage.get_collection_stats()
        print(f"📊 Final collection stats: {stats}")
        
        return True
    else:
        print("❌ Failed to upload embeddings")
        return False

async def test_search():
    """Test search with the updated database"""
    print("\n🔍 Testing search functionality...")
    
    storage = QdrantStorage()
    
    # Load model for query embeddings
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    
    # Test queries
    test_queries = [
        "Python algorithmic trading strategies",
        "risk management in trading",
        "machine learning for finance",
        "backtesting trading algorithms",
        "portfolio optimization"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: \"{query}\"")
        
        # Generate embedding
        query_embedding = model.encode([query], convert_to_numpy=True)[0].tolist()
        
        # Search
        results = await storage.search_semantic(query_embedding, limit=3)
        
        print(f"📊 Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            score = result['score']
            chunk_id = result['chunk_id']
            book_title = result['metadata'].get('book_title', 'Unknown')
            text_preview = result['text'][:100] + '...' if len(result['text']) > 100 else result['text']
            
            print(f"  {i}. Score: {score:.4f}")
            print(f"     Book: {book_title}")
            print(f"     Chunk: {chunk_id}")
            print(f"     Text: {text_preview}")

async def main():
    print("🚀 Updating vector database with complete book data")
    
    # Update database
    success = await update_vector_database()
    
    if success:
        # Test search
        await test_search()
        
        print("\n🏆 Vector database update completed successfully!")
        print("📋 The book is now fully processed and searchable!")
    else:
        print("\n❌ Vector database update failed")

if __name__ == "__main__":
    asyncio.run(main())