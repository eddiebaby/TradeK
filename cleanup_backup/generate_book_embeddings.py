#!/usr/bin/env python3
"""
Generate embeddings for the processed book chunks
"""

import sys
import sqlite3
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import time

def generate_embeddings():
    """Generate embeddings for all book chunks"""
    
    db_path = '/home/scottschweizer/TradeKnowledge/data/knowledge.db'
    embedding_dir = Path('/home/scottschweizer/TradeKnowledge/data/embeddings')
    
    print("🧠 Generating embeddings for book chunks...")
    
    # Load model
    print("📦 Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    
    # Get chunks from database
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, text FROM chunks 
            WHERE book_id = 'Python_for_Algorithmic_Trading_complete'
            ORDER BY chunk_index
        """)
        chunks = cursor.fetchall()
        conn.close()
        
        if not chunks:
            print("❌ No chunks found!")
            return False
            
        print(f"📊 Found {len(chunks)} chunks to process")
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False
    
    # Generate embeddings in batches
    chunk_ids = [chunk[0] for chunk in chunks]
    texts = [chunk[1] for chunk in chunks]
    
    print("🔮 Generating embeddings...")
    start_time = time.time()
    
    # Process in smaller batches for memory efficiency
    batch_size = 16
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch_texts, convert_to_numpy=True)
        all_embeddings.extend(batch_embeddings)
        
        progress = min(i + batch_size, len(texts))
        print(f"  ✅ Processed {progress}/{len(texts)} chunks")
    
    processing_time = time.time() - start_time
    print(f"⏱️  Processing time: {processing_time:.1f} seconds")
    
    # Convert to numpy array
    embeddings_matrix = np.array(all_embeddings)
    
    print(f"📐 Embeddings shape: {embeddings_matrix.shape}")
    
    # Load existing embeddings if they exist
    json_file = embedding_dir / 'chunk_embeddings.json'
    npz_file = embedding_dir / 'chunk_embeddings.npz'
    
    existing_chunk_ids = []
    existing_embeddings = []
    
    if json_file.exists() and npz_file.exists():
        print("📂 Loading existing embeddings...")
        try:
            # Load existing data
            existing_data = np.load(npz_file)
            existing_chunk_ids = list(existing_data['chunk_ids'])
            existing_embeddings = existing_data['embeddings']
            
            print(f"  Found {len(existing_chunk_ids)} existing embeddings")
        except Exception as e:
            print(f"  ⚠️  Error loading existing embeddings: {e}")
    
    # Combine old and new embeddings
    final_chunk_ids = existing_chunk_ids + chunk_ids
    if len(existing_embeddings) > 0:
        final_embeddings = np.vstack([existing_embeddings, embeddings_matrix])
    else:
        final_embeddings = embeddings_matrix
    
    print(f"📊 Final dataset: {len(final_chunk_ids)} chunks, {final_embeddings.shape}")
    
    # Save embeddings
    embedding_dir.mkdir(exist_ok=True)
    
    # Save metadata
    metadata = {
        'model': 'all-MiniLM-L6-v2',
        'embedding_dim': int(final_embeddings.shape[1]),
        'total_embeddings': len(final_chunk_ids),
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'processing_time_seconds': processing_time
    }
    
    with open(json_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Saved metadata to {json_file}")
    
    # Save embeddings
    np.savez_compressed(
        npz_file,
        chunk_ids=np.array(final_chunk_ids),
        embeddings=final_embeddings,
        model=metadata['model'],
        embedding_dim=metadata['embedding_dim']
    )
    
    print(f"✅ Saved embeddings to {npz_file}")
    
    # Show statistics
    print(f"\n📊 Embedding Statistics:")
    print(f"   Total chunks: {len(final_chunk_ids)}")
    print(f"   Embedding dimension: {final_embeddings.shape[1]}")
    print(f"   Model: {metadata['model']}")
    print(f"   File size: {npz_file.stat().st_size / 1024:.1f} KB")
    
    return True

if __name__ == "__main__":
    success = generate_embeddings()
    if success:
        print("\n🏆 Embedding generation completed!")
        print("📋 Next: Update vector database with new embeddings")
    else:
        print("\n❌ Embedding generation failed")