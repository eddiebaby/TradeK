#!/usr/bin/env python3
"""
Populate ChromaDB with the same data that's in Qdrant
This fixes the inconsistent dual database state
"""

import sys
import sqlite3
import json
import numpy as np
import chromadb
from pathlib import Path
import time

# Add src to path
sys.path.append('src')

def populate_chromadb():
    """Populate ChromaDB with all book data"""
    
    print("🔧 Populating ChromaDB with both books...")
    print("="*50)
    
    db_path = '/home/scottschweizer/TradeKnowledge/data/knowledge.db'
    chroma_dir = '/home/scottschweizer/TradeKnowledge/data/chromadb'
    embedding_file = '/home/scottschweizer/TradeKnowledge/data/embeddings/chunk_embeddings.npz'
    
    # Load embeddings
    print("📂 Loading embeddings...")
    data = np.load(embedding_file)
    chunk_ids = data['chunk_ids']
    embeddings_matrix = data['embeddings']
    
    print(f"📊 Loaded {len(chunk_ids)} embeddings")
    
    # Load chunk metadata from database
    print("🔍 Loading chunk metadata...")
    chunk_metadata = {}
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
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
            
            try:
                metadata = json.loads(metadata_json) if metadata_json else {}
            except:
                metadata = {}
            
            chunk_metadata[chunk_id] = {
                'book_id': book_id,
                'book_title': book_title or 'Unknown',
                'book_author': book_author or 'Unknown',
                'chunk_index': chunk_index,
                'text': text,
                'metadata': metadata
            }
        
        conn.close()
        print(f"✅ Loaded metadata for {len(chunk_metadata)} chunks")
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False
    
    # Initialize ChromaDB
    print("🔌 Connecting to ChromaDB...")
    try:
        client = chromadb.PersistentClient(path=chroma_dir)
        
        # Delete existing collection if it exists
        try:
            client.delete_collection(name="trading_books")
            print("  ✅ Deleted old collection")
        except:
            pass
        
        # Create new collection
        collection = client.create_collection(
            name="trading_books",
            metadata={
                "description": "Trading and embeddings knowledge base",
                "created_at": time.strftime('%Y-%m-%d %H:%M:%S')
            }
        )
        print("  ✅ Created new ChromaDB collection")
        
    except Exception as e:
        print(f"❌ ChromaDB error: {e}")
        return False
    
    # Prepare data for ChromaDB
    print("📦 Preparing data for ChromaDB...")
    
    chromadb_ids = []
    chromadb_embeddings = []
    chromadb_documents = []
    chromadb_metadatas = []
    
    chunks_added = 0
    
    for i, chunk_id in enumerate(chunk_ids):
        chunk_id_str = str(chunk_id)
        
        if chunk_id_str in chunk_metadata:
            meta = chunk_metadata[chunk_id_str]
            
            # ChromaDB has strict metadata requirements
            safe_metadata = {
                'book_id': meta['book_id'],
                'book_title': meta['book_title'],
                'book_author': meta['book_author'],
                'chunk_index': int(meta['chunk_index']),
                'char_count': len(meta['text']),
                'word_count': len(meta['text'].split())
            }
            
            # Add additional metadata safely
            if isinstance(meta['metadata'], dict):
                for key, value in meta['metadata'].items():
                    # ChromaDB only supports certain types
                    if isinstance(value, (str, int, float, bool)):
                        safe_key = f"meta_{key}"
                        safe_metadata[safe_key] = value
            
            chromadb_ids.append(chunk_id_str)
            chromadb_embeddings.append(embeddings_matrix[i].tolist())
            chromadb_documents.append(meta['text'])
            chromadb_metadatas.append(safe_metadata)
            
            chunks_added += 1
        else:
            print(f"  ⚠️  No metadata found for chunk: {chunk_id_str}")
    
    print(f"📊 Prepared {chunks_added} chunks for ChromaDB")
    
    # Upload to ChromaDB in batches
    print("📤 Uploading to ChromaDB...")
    batch_size = 50
    
    for i in range(0, len(chromadb_ids), batch_size):
        batch_end = min(i + batch_size, len(chromadb_ids))
        
        batch_ids = chromadb_ids[i:batch_end]
        batch_embeddings = chromadb_embeddings[i:batch_end]
        batch_documents = chromadb_documents[i:batch_end]
        batch_metadatas = chromadb_metadatas[i:batch_end]
        
        try:
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas
            )
            
            print(f"  ✅ Uploaded batch {i//batch_size + 1} ({len(batch_ids)} chunks)")
            
        except Exception as e:
            print(f"  ❌ Error uploading batch {i//batch_size + 1}: {e}")
            return False
    
    # Verify ChromaDB
    print("🔍 Verifying ChromaDB...")
    try:
        count = collection.count()
        print(f"✅ ChromaDB now contains {count} documents")
        
        # Test a quick search
        if count > 0:
            test_results = collection.query(
                query_texts=["machine learning"],
                n_results=3
            )
            print(f"✅ Test search returned {len(test_results['ids'][0]) if test_results['ids'] else 0} results")
        
        return True
        
    except Exception as e:
        print(f"❌ ChromaDB verification error: {e}")
        return False

if __name__ == "__main__":
    success = populate_chromadb()
    if success:
        print("\n🏆 ChromaDB population completed!")
        print("📋 Both databases (Qdrant + ChromaDB) now have the same data")
    else:
        print("\n❌ ChromaDB population failed")