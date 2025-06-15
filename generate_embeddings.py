#!/usr/bin/env python3
"""
Memory-optimized embedding generation for TradeKnowledge
Uses sentence-transformers with careful memory management for WSL2
"""

import asyncio
import sys
import time
import gc
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import psutil

# Add src to path
sys.path.append('src')

from src.core.sqlite_storage import SQLiteStorage
from src.ingestion.resource_monitor import ResourceMonitor, ResourceLimits

class EmbeddingGenerator:
    def __init__(self):
        self.storage = SQLiteStorage()
        self.model = None
        self.resource_monitor = None
        self.model_name = "all-MiniLM-L6-v2"  # Small, fast model
        self.embedding_dim = 384  # Dimension for this model
        
        # Create embeddings directory
        self.embeddings_dir = Path("data/embeddings")
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Track progress
        self.stats = {
            'start_time': None,
            'chunks_processed': 0,
            'embeddings_generated': 0,
            'cache_hits': 0,
            'processing_time': 0,
            'average_time_per_chunk': 0
        }
    
    async def initialize(self):
        """Initialize with conservative memory settings"""
        print("🔧 Initializing embedding generator...")
        
        # Set up resource monitoring with strict limits
        limits = ResourceLimits(
            max_memory_percent=65.0,  # Very conservative for embedding generation
            max_memory_mb=1200,       # 1.2GB limit
            warning_threshold=50.0,   # Warn at 50%
            check_interval=2.0        # Check every 2 seconds
        )
        
        self.resource_monitor = ResourceMonitor(limits)
        await self.resource_monitor.start_monitoring()
        
        # Add callback for memory management
        async def memory_callback(check):
            if check['memory_warning']:
                print(f"⚠️  Memory warning: {check['usage']['system_used_percent']:.1f}%")
                # Force garbage collection on warning
                gc.collect()
            
            if check['memory_critical']:
                print(f"🚨 Memory critical: {check['usage']['system_used_percent']:.1f}% - pausing...")
                # Clear model cache if memory is critical
                if self.model:
                    self.model = None
                    gc.collect()
                await self.resource_monitor.wait_for_memory_available()
                print("✅ Memory recovered, will reload model...")
        
        self.resource_monitor.add_callback(memory_callback)
        
        print("✅ Embedding generator initialized")
    
    def _load_model(self):
        """Load the sentence transformer model (lazy loading)"""
        if self.model is None:
            print(f"📦 Loading model: {self.model_name}")
            
            from sentence_transformers import SentenceTransformer
            
            # Use CPU device to avoid GPU memory issues
            self.model = SentenceTransformer(self.model_name, device='cpu')
            
            print(f"✅ Model loaded: {self.model_name}")
        
        return self.model
    
    def _get_text_hash(self, text: str) -> str:
        """Get hash for text to enable caching"""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()
    
    async def _load_embedding_cache(self) -> Dict[str, List[float]]:
        """Load existing embedding cache"""
        cache_file = self.embeddings_dir / "embedding_cache.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
                print(f"📂 Loaded {len(cache)} cached embeddings")
                return cache
            except Exception as e:
                print(f"⚠️  Could not load cache: {e}")
        
        return {}
    
    async def _save_embedding_cache(self, cache: Dict[str, List[float]]):
        """Save embedding cache"""
        cache_file = self.embeddings_dir / "embedding_cache.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
            print(f"💾 Saved {len(cache)} embeddings to cache")
        except Exception as e:
            print(f"⚠️  Could not save cache: {e}")
    
    async def generate_embeddings_for_chunks(self, chunk_ids: Optional[List[str]] = None, batch_size: int = 5):
        """Generate embeddings for chunks with memory optimization"""
        
        self.stats['start_time'] = time.time()
        print("🚀 Starting embedding generation...")
        
        # Load embedding cache
        embedding_cache = await self._load_embedding_cache()
        
        # Get chunks to process
        if chunk_ids:
            chunks = []
            for chunk_id in chunk_ids:
                chunk = await self.storage.get_chunk(chunk_id)
                if chunk:
                    chunks.append(chunk)
        else:
            # Get all chunks
            books = await self.storage.list_books()
            chunks = []
            for book in books:
                book_chunks = await self.storage.get_chunks_by_book(book.id)
                chunks.extend(book_chunks)
        
        total_chunks = len(chunks)
        print(f"📦 Processing {total_chunks} chunks in batches of {batch_size}")
        
        if not chunks:
            print("❌ No chunks found to process")
            return False
        
        # Track embeddings to save
        embeddings_to_save = []
        new_embeddings = 0
        
        # Process in small batches to manage memory
        for batch_start in range(0, total_chunks, batch_size):
            batch_end = min(batch_start + batch_size, total_chunks)
            batch_chunks = chunks[batch_start:batch_end]
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing batch {batch_start+1}-{batch_end}")
            
            # Check memory before processing
            check = self.resource_monitor.check_memory_limits()
            if check['memory_critical']:
                print("🚨 Memory critical - waiting for recovery...")
                await self.resource_monitor.wait_for_memory_available()
            
            # Prepare texts for this batch
            batch_texts = []
            batch_chunk_data = []
            
            for chunk in batch_chunks:
                text_hash = self._get_text_hash(chunk.text)
                
                # Check cache first
                if text_hash in embedding_cache:
                    embedding = embedding_cache[text_hash]
                    embeddings_to_save.append({
                        'chunk_id': chunk.id,
                        'embedding': embedding,
                        'cached': True
                    })
                    self.stats['cache_hits'] += 1
                else:
                    # Need to generate embedding
                    batch_texts.append(chunk.text)
                    batch_chunk_data.append({
                        'chunk_id': chunk.id,
                        'text_hash': text_hash,
                        'cached': False
                    })
            
            # Generate embeddings for non-cached texts
            if batch_texts:
                try:
                    # Load model (lazy loading)
                    model = self._load_model()
                    
                    # Generate embeddings
                    print(f"  🧠 Generating {len(batch_texts)} new embeddings...")
                    batch_embeddings = model.encode(
                        batch_texts,
                        batch_size=2,  # Very small batch size for model
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
                    
                    # Process results
                    for i, chunk_data in enumerate(batch_chunk_data):
                        embedding = batch_embeddings[i].tolist()
                        
                        # Add to results
                        embeddings_to_save.append({
                            'chunk_id': chunk_data['chunk_id'],
                            'embedding': embedding,
                            'cached': False
                        })
                        
                        # Add to cache
                        embedding_cache[chunk_data['text_hash']] = embedding
                        new_embeddings += 1
                    
                    self.stats['embeddings_generated'] += len(batch_texts)
                    
                except Exception as e:
                    print(f"❌ Error generating embeddings for batch: {e}")
                    continue
            
            self.stats['chunks_processed'] = batch_end
            
            # Memory management
            del batch_chunks
            del batch_texts
            del batch_chunk_data
            gc.collect()
            
            # Progress update
            if batch_end % 20 == 0:
                progress = (batch_end / total_chunks) * 100
                print(f"📊 Progress: {batch_end}/{total_chunks} ({progress:.1f}%)")
                
                # Save cache periodically
                if new_embeddings > 0:
                    await self._save_embedding_cache(embedding_cache)
        
        # Save final cache
        await self._save_embedding_cache(embedding_cache)
        
        # Save embeddings to files (for vector database)
        await self._save_embeddings_to_files(embeddings_to_save)
        
        # Calculate final stats
        self.stats['processing_time'] = time.time() - self.stats['start_time']
        self.stats['average_time_per_chunk'] = self.stats['processing_time'] / total_chunks if total_chunks > 0 else 0
        
        await self.resource_monitor.stop_monitoring()
        
        self._print_final_stats(total_chunks, new_embeddings)
        
        return True
    
    async def _save_embeddings_to_files(self, embeddings_data: List[Dict[str, Any]]):
        """Save embeddings to files for vector database"""
        
        print("💾 Saving embeddings to files...")
        
        # Create embeddings file
        embeddings_file = self.embeddings_dir / "chunk_embeddings.json"
        
        # Prepare data
        embeddings_export = {
            'model': self.model_name,
            'embedding_dim': self.embedding_dim,
            'created_at': datetime.now().isoformat(),
            'total_embeddings': len(embeddings_data),
            'embeddings': {}
        }
        
        for item in embeddings_data:
            embeddings_export['embeddings'][item['chunk_id']] = {
                'embedding': item['embedding'],
                'cached': item['cached']
            }
        
        # Save to file
        with open(embeddings_file, 'w') as f:
            json.dump(embeddings_export, f, indent=2)
        
        print(f"💾 Saved {len(embeddings_data)} embeddings to {embeddings_file}")
        
        # Also save in numpy format for faster loading
        numpy_file = self.embeddings_dir / "chunk_embeddings.npz"
        
        chunk_ids = [item['chunk_id'] for item in embeddings_data]
        embeddings_matrix = np.array([item['embedding'] for item in embeddings_data])
        
        np.savez_compressed(
            numpy_file,
            chunk_ids=chunk_ids,
            embeddings=embeddings_matrix,
            model=self.model_name,
            embedding_dim=self.embedding_dim
        )
        
        print(f"💾 Saved embeddings in numpy format to {numpy_file}")
    
    def _print_final_stats(self, total_chunks: int, new_embeddings: int):
        """Print final generation statistics"""
        print("\n" + "="*60)
        print("📊 EMBEDDING GENERATION COMPLETE")
        print("="*60)
        print(f"📦 Total chunks processed: {total_chunks}")
        print(f"🧠 New embeddings generated: {new_embeddings}")
        print(f"📂 Cache hits: {self.stats['cache_hits']}")
        print(f"⏱️  Total processing time: {self.stats['processing_time']:.1f} seconds")
        print(f"⚡ Average time per chunk: {self.stats['average_time_per_chunk']:.3f} seconds")
        print(f"🏷️  Model used: {self.model_name}")
        print(f"📐 Embedding dimension: {self.embedding_dim}")
        
        cache_rate = (self.stats['cache_hits'] / total_chunks) * 100 if total_chunks > 0 else 0
        print(f"📈 Cache hit rate: {cache_rate:.1f}%")
        
        print("="*60)
        print("\n🔍 Next steps:")
        print("   • Embeddings saved to data/embeddings/")
        print("   • Start Qdrant vector database")
        print("   • Upload embeddings to Qdrant")
        print("   • Test semantic search functionality")

async def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage: python generate_embeddings.py [chunk_id1 chunk_id2 ...] [--batch-size N]")
        print("\nOptions:")
        print("  --batch-size N    Set batch size (default: 5)")
        print("  --help           Show this help")
        print("\nExamples:")
        print("  python generate_embeddings.py                    # Process all chunks")
        print("  python generate_embeddings.py chunk_001 chunk_002  # Process specific chunks")
        print("  python generate_embeddings.py --batch-size 3     # Use smaller batches")
        return
    
    # Parse arguments
    chunk_ids = []
    batch_size = 5
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--batch-size":
            if i + 1 < len(sys.argv):
                batch_size = int(sys.argv[i + 1])
                i += 2
            else:
                print("❌ --batch-size requires a number")
                return
        else:
            chunk_ids.append(arg)
            i += 1
    
    # Check memory
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    
    if available_gb < 1.0:
        print(f"⚠️  Warning: Low memory available ({available_gb:.1f}GB)")
        print("   Consider closing other applications")
        batch_size = min(batch_size, 3)  # Use smaller batches
    
    print(f"🖥️  Available memory: {available_gb:.1f}GB")
    print(f"📦 Using batch size: {batch_size}")
    
    # Initialize and run
    generator = EmbeddingGenerator()
    await generator.initialize()
    
    success = await generator.generate_embeddings_for_chunks(
        chunk_ids if chunk_ids else None,
        batch_size
    )
    
    if success:
        print("🏆 Embedding generation completed successfully!")
    else:
        print("💥 Embedding generation failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())