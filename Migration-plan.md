# Migration Guide: From OpenAI to Local Embeddings
## Transitioning TradeKnowledge to Ollama + Nomic + Qdrant

### Overview

This guide details how to migrate the existing TradeKnowledge implementation from OpenAI's API to a fully local setup using:
- **Ollama** - Local LLM runtime
- **nomic-embed-text** - Local embedding model (via Ollama)
- **Qdrant** - Vector database (replacing ChromaDB)

The migration preserves all existing functionality while eliminating API dependencies and costs.

---

## Pre-Migration Checklist

### 1. Install Required Components

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the nomic-embed-text model
ollama pull nomic-embed-text

# Verify installation
ollama list

# Install Qdrant (Docker method)
docker pull qdrant/qdrant
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

# OR Install Qdrant (Binary method)
wget https://github.com/qdrant/qdrant/releases/download/v1.7.4/qdrant-x86_64-unknown-linux-gnu.tar.gz
tar -xvf qdrant-x86_64-unknown-linux-gnu.tar.gz
./qdrant --config-path config/config.yaml
```

### 2. Update Python Dependencies

```bash
# Update requirements.txt - REMOVE these lines:
# openai>=1.0.0
# chromadb>=0.4.0

# ADD these lines:
pip install qdrant-client>=1.7.0
pip install ollama>=0.1.7
pip install httpx>=0.25.0  # For async HTTP requests

# Install updates
pip install -r requirements.txt
```

---

## Phase 1 Modifications

### 1. Update Configuration System

```python
# Update src/core/config.py
# Replace the embedding configuration section with:

@dataclass
class EmbeddingConfig:
    """Embedding configuration"""
    model: str = "nomic-embed-text"  # Changed from text-embedding-ada-002
    dimension: int = 768  # nomic-embed-text dimension (was 1536 for ada)
    batch_size: int = 32
    ollama_host: str = "http://localhost:11434"  # Ollama API endpoint
    timeout: int = 30  # Request timeout in seconds

@dataclass
class DatabaseConfig:
    """Database configuration"""
    sqlite: SQLiteConfig = field(default_factory=SQLiteConfig)
    qdrant: 'QdrantConfig' = field(default_factory=lambda: QdrantConfig())  # Replaced ChromaDB

@dataclass
class QdrantConfig:
    """Qdrant configuration"""
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "tradeknowledge"
    use_grpc: bool = False  # Use REST API by default
    api_key: Optional[str] = None  # For cloud deployment
    https: bool = False
    prefer_grpc: bool = False
    
    @property
    def url(self) -> str:
        protocol = "https" if self.https else "http"
        return f"{protocol}://{self.host}:{self.port}"
```

### 2. Replace Embedding Generator

Create a new embedding generator that uses Ollama:

```python
# Create src/ingestion/local_embeddings.py
cat > src/ingestion/local_embeddings.py << 'EOF'
"""
Local embedding generation using Ollama and nomic-embed-text

This replaces the OpenAI-based embedding generator with a fully local solution.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
import json
import hashlib
from datetime import datetime

import httpx
import numpy as np

from core.models import Chunk
from core.config import get_config

logger = logging.getLogger(__name__)

class LocalEmbeddingGenerator:
    """
    Generates embeddings locally using Ollama with nomic-embed-text.
    
    This provides the same interface as the OpenAI version but runs
    completely offline with no API costs.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize local embedding generator"""
        self.config = get_config()
        self.model_name = model_name or self.config.embedding.model
        self.ollama_host = self.config.embedding.ollama_host
        self.embedding_dimension = self.config.embedding.dimension
        self.timeout = self.config.embedding.timeout
        
        # HTTP client for Ollama API
        self.client = httpx.AsyncClient(timeout=self.timeout)
        
        # Cache for embeddings
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Verify Ollama is running
        asyncio.create_task(self._verify_ollama())
        
    async def _verify_ollama(self):
        """Verify Ollama is running and model is available"""
        try:
            # Check if Ollama is running
            response = await self.client.get(f"{self.ollama_host}/api/version")
            if response.status_code == 200:
                logger.info(f"Ollama is running: {response.json()}")
            
            # Check if model is available
            response = await self.client.get(f"{self.ollama_host}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                if self.model_name in model_names:
                    logger.info(f"Model {self.model_name} is available")
                else:
                    logger.error(f"Model {self.model_name} not found! Available: {model_names}")
                    logger.info(f"Pull the model with: ollama pull {self.model_name}")
                    
        except Exception as e:
            logger.error(f"Cannot connect to Ollama at {self.ollama_host}: {e}")
            logger.info("Make sure Ollama is running: ollama serve")
    
    async def generate_embeddings(self, 
                                  chunks: List[Chunk],
                                  show_progress: bool = True) -> List[List[float]]:
        """Generate embeddings for chunks using Ollama"""
        if not chunks:
            return []
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks using {self.model_name}")
        
        # Separate cached and uncached
        embeddings = []
        uncached_indices = []
        
        for i, chunk in enumerate(chunks):
            cache_key = self._get_cache_key(chunk.text)
            if cache_key in self.cache:
                embeddings.append(self.cache[cache_key])
                self.cache_hits += 1
            else:
                embeddings.append(None)  # Placeholder
                uncached_indices.append(i)
                self.cache_misses += 1
        
        logger.info(f"Cache hits: {self.cache_hits}, misses: {self.cache_misses}")
        
        # Generate embeddings for uncached chunks
        if uncached_indices:
            # Process in batches
            batch_size = self.config.embedding.batch_size
            
            for batch_start in range(0, len(uncached_indices), batch_size):
                batch_end = min(batch_start + batch_size, len(uncached_indices))
                batch_indices = uncached_indices[batch_start:batch_end]
                
                # Get texts for this batch
                batch_texts = [chunks[i].text for i in batch_indices]
                
                # Generate embeddings
                batch_embeddings = await self._generate_batch_embeddings(batch_texts)
                
                # Fill in results and update cache
                for idx, embedding in zip(batch_indices, batch_embeddings):
                    embeddings[idx] = embedding
                    cache_key = self._get_cache_key(chunks[idx].text)
                    self.cache[cache_key] = embedding
                
                if show_progress:
                    progress = (batch_end / len(uncached_indices)) * 100
                    logger.info(f"Embedding progress: {progress:.1f}%")
        
        return embeddings
    
    async def _generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        embeddings = []
        
        # Ollama doesn't support batch embedding, so we process one by one
        # but we can parallelize the requests
        tasks = []
        for text in texts:
            task = self._generate_single_embedding(text)
            tasks.append(task)
        
        # Run in parallel with semaphore to limit concurrent requests
        sem = asyncio.Semaphore(5)  # Max 5 concurrent requests
        
        async def bounded_task(task):
            async with sem:
                return await task
        
        bounded_tasks = [bounded_task(task) for task in tasks]
        embeddings = await asyncio.gather(*bounded_tasks)
        
        return embeddings
    
    async def _generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            # Prepare request
            data = {
                "model": self.model_name,
                "prompt": text
            }
            
            # Send request to Ollama
            response = await self.client.post(
                f"{self.ollama_host}/api/embeddings",
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                embedding = result['embedding']
                
                # Verify dimension
                if len(embedding) != self.embedding_dimension:
                    logger.warning(
                        f"Embedding dimension mismatch: "
                        f"expected {self.embedding_dimension}, got {len(embedding)}"
                    )
                
                return embedding
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                # Return zero vector on error
                return [0.0] * self.embedding_dimension
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * self.embedding_dimension
    
    async def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a search query"""
        # Check cache
        cache_key = self._get_cache_key(query)
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        self.cache_misses += 1
        
        # Generate embedding
        embedding = await self._generate_single_embedding(query)
        
        # Cache it
        self.cache[cache_key] = embedding
        
        return embedding
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        key_string = f"{self.model_name}:{text}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def save_cache(self, file_path: str):
        """Save embedding cache to disk"""
        cache_data = {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dimension,
            'cache': self.cache,
            'stats': {
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'saved_at': datetime.now().isoformat()
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(cache_data, f)
        
        logger.info(f"Saved {len(self.cache)} cached embeddings to {file_path}")
    
    def load_cache(self, file_path: str):
        """Load embedding cache from disk"""
        try:
            with open(file_path, 'r') as f:
                cache_data = json.load(f)
            
            # Verify model compatibility
            if cache_data['model_name'] != self.model_name:
                logger.warning(
                    f"Cache model mismatch: {cache_data['model_name']} != {self.model_name}"
                )
                return
            
            self.cache = cache_data['cache']
            logger.info(f"Loaded {len(self.cache)} cached embeddings")
            
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.client.aclose()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding generation statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dimension,
            'ollama_host': self.ollama_host,
            'cache_size': len(self.cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': hit_rate,
            'total_requests': total_requests
        }

# Compatibility layer for smooth migration
EmbeddingGenerator = LocalEmbeddingGenerator
EOF
```

### 3. Replace ChromaDB with Qdrant

Create a new Qdrant storage implementation:

```python
# Create src/core/qdrant_storage.py
cat > src/core/qdrant_storage.py << 'EOF'
"""
Qdrant vector storage implementation

This replaces ChromaDB with Qdrant for better performance and features.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    SearchRequest, SearchParams,
    UpdateStatus
)

from core.interfaces import VectorStorageInterface
from core.models import Chunk
from core.config import get_config

logger = logging.getLogger(__name__)

class QdrantStorage(VectorStorageInterface):
    """
    Qdrant implementation for vector storage.
    
    Qdrant provides:
    - Better performance than ChromaDB
    - More advanced filtering options
    - Payload storage alongside vectors
    - Snapshot and backup capabilities
    """
    
    def __init__(self, collection_name: Optional[str] = None):
        """Initialize Qdrant storage"""
        config = get_config()
        self.collection_name = collection_name or config.database.qdrant.collection_name
        
        # Initialize client
        self.client = QdrantClient(
            host=config.database.qdrant.host,
            port=config.database.qdrant.port,
            api_key=config.database.qdrant.api_key,
            https=config.database.qdrant.https,
            prefer_grpc=config.database.qdrant.prefer_grpc
        )
        
        # Vector configuration
        self.vector_size = config.embedding.dimension
        self.distance_metric = Distance.COSINE
        
        # Ensure collection exists
        self._ensure_collection()
        
        logger.info(f"Initialized Qdrant storage with collection: {self.collection_name}")
    
    def _ensure_collection(self):
        """Ensure collection exists with proper configuration"""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=self.distance_metric
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                # Verify configuration
                collection_info = self.client.get_collection(self.collection_name)
                current_size = collection_info.config.params.vectors.size
                
                if current_size != self.vector_size:
                    logger.warning(
                        f"Collection vector size mismatch: "
                        f"expected {self.vector_size}, got {current_size}"
                    )
                    
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
            raise
    
    async def save_embeddings(self, 
                             chunks: List[Chunk],
                             embeddings: List[List[float]]) -> bool:
        """Save chunk embeddings to Qdrant"""
        if not chunks or not embeddings:
            return True
        
        if len(chunks) != len(embeddings):
            logger.error(f"Mismatch: {len(chunks)} chunks, {len(embeddings)} embeddings")
            return False
        
        try:
            # Prepare points for Qdrant
            points = []
            
            for chunk, embedding in zip(chunks, embeddings):
                # Create payload
                payload = {
                    'chunk_id': chunk.id,
                    'book_id': chunk.book_id,
                    'chunk_index': chunk.chunk_index,
                    'chunk_type': chunk.chunk_type.value,
                    'text': chunk.text,
                    'created_at': chunk.created_at.isoformat()
                }
                
                # Add optional fields
                if chunk.chapter:
                    payload['chapter'] = chunk.chapter
                if chunk.section:
                    payload['section'] = chunk.section
                if chunk.page_start:
                    payload['page_start'] = chunk.page_start
                if chunk.page_end:
                    payload['page_end'] = chunk.page_end
                
                # Add chunk metadata
                payload.update(chunk.metadata)
                
                # Create point
                point = PointStruct(
                    id=str(uuid.uuid4()),  # Generate unique ID
                    vector=embedding,
                    payload=payload
                )
                points.append(point)
            
            # Upload in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                
                operation_info = self.client.upsert(
                    collection_name=self.collection_name,
                    wait=True,
                    points=batch
                )
                
                if operation_info.status != UpdateStatus.COMPLETED:
                    logger.error(f"Failed to upsert batch: {operation_info}")
                    return False
                
                logger.debug(f"Uploaded batch {i//batch_size + 1} ({len(batch)} points)")
            
            logger.info(f"Successfully saved {len(chunks)} embeddings to Qdrant")
            return True
            
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            return False
    
    async def search_semantic(self,
                             query_embedding: List[float],
                             filter_dict: Optional[Dict[str, Any]] = None,
                             limit: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic search using Qdrant"""
        try:
            # Build filter
            qdrant_filter = None
            if filter_dict:
                conditions = []
                
                # Book ID filter
                if 'book_ids' in filter_dict and filter_dict['book_ids']:
                    conditions.append(
                        FieldCondition(
                            key="book_id",
                            match=MatchValue(any=filter_dict['book_ids'])
                        )
                    )
                
                # Chunk type filter
                if 'chunk_type' in filter_dict:
                    conditions.append(
                        FieldCondition(
                            key="chunk_type",
                            match=MatchValue(value=filter_dict['chunk_type'])
                        )
                    )
                
                if conditions:
                    qdrant_filter = Filter(must=conditions)
            
            # Perform search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=qdrant_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False  # We don't need vectors back
            )
            
            # Format results
            search_results = []
            for hit in search_result:
                search_results.append({
                    'chunk_id': hit.payload['chunk_id'],
                    'text': hit.payload['text'],
                    'metadata': {
                        k: v for k, v in hit.payload.items()
                        if k not in ['chunk_id', 'text']
                    },
                    'score': hit.score,
                    'distance': 1 - hit.score  # Convert similarity to distance
                })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    async def delete_embeddings(self, chunk_ids: List[str]) -> bool:
        """Delete embeddings by chunk IDs"""
        if not chunk_ids:
            return True
        
        try:
            # Delete by filter
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="chunk_id",
                            match=MatchValue(any=chunk_ids)
                        )
                    ]
                )
            )
            
            logger.info(f"Deleted embeddings for {len(chunk_ids)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting embeddings: {e}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            return {
                'collection_name': self.collection_name,
                'total_embeddings': collection_info.points_count,
                'vector_size': collection_info.config.params.vectors.size,
                'distance_metric': str(collection_info.config.params.vectors.distance),
                'segments_count': collection_info.segments_count,
                'status': collection_info.status
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {
                'collection_name': self.collection_name,
                'error': str(e)
            }
    
    def create_snapshot(self) -> str:
        """Create a snapshot of the collection"""
        try:
            snapshot_info = self.client.create_snapshot(
                collection_name=self.collection_name
            )
            logger.info(f"Created snapshot: {snapshot_info}")
            return snapshot_info.name
            
        except Exception as e:
            logger.error(f"Error creating snapshot: {e}")
            return ""

# Compatibility layer
ChromaDBStorage = QdrantStorage  # For drop-in replacement
EOF
```

### 4. Update Import Statements

Update all files that import the old modules:

```python
# In all files that have:
# from ingestion.embeddings import EmbeddingGenerator
# Change to:
from ingestion.local_embeddings import LocalEmbeddingGenerator as EmbeddingGenerator

# In all files that have:
# from core.chroma_storage import ChromaDBStorage
# Change to:
from core.qdrant_storage import QdrantStorage as ChromaDBStorage
```

### 5. Remove OpenAI API Key Requirements

```python
# Update src/ingestion/embeddings.py (or delete it)
# Remove all OpenAI-related code and imports

# Update .env file - remove:
# OPENAI_API_KEY=your_key_here

# Update scripts/verify_environment.py
# Remove the OpenAI API key check
```

### 6. Update Database Initialization

```python
# Update scripts/init_db.py
# Add Qdrant initialization:

async def init_qdrant():
    """Initialize Qdrant collection"""
    from core.qdrant_storage import QdrantStorage
    
    logger.info("Initializing Qdrant...")
    storage = QdrantStorage()
    
    # Get stats to verify connection
    stats = await storage.get_collection_stats()
    logger.info(f"Qdrant stats: {stats}")
    
    return True

# Add to main():
# success = await init_qdrant()
```

---

## Phase 2 Modifications

### 1. Update C++ Modules

The C++ modules remain largely the same, but update the embedding dimension:

```cpp
// In src/cpp/similarity.cpp
// Update any hardcoded dimension values from 1536 to 768
// Or better, make it configurable:

class SimdSimilarity {
private:
    size_t embedding_dimension;
    
public:
    SimdSimilarity(size_t dim = 768) : embedding_dimension(dim) {}
    // ... rest of the code
};
```

### 2. Update Cache Manager

No changes needed for the cache manager - it's already storage-agnostic.

### 3. Update Tests

Update all test files to use the new components:

```python
# In test files, update embedding dimension expectations:
# test_embeddings = [[random.random() for _ in range(768)] for _ in test_chunks]
# Instead of range(1536)

# Update model names in tests:
# generator = EmbeddingGenerator("nomic-embed-text")
# Instead of "text-embedding-ada-002"
```

---

## Migration Script

Create a migration script to help users transition existing data:

```python
# Create scripts/migrate_to_local.py
cat > scripts/migrate_to_local.py << 'EOF'
#!/usr/bin/env python3
"""
Migrate existing TradeKnowledge data to local setup
"""

import asyncio
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

logger = logging.getLogger(__name__)

async def migrate_embeddings():
    """Re-generate embeddings with local model"""
    from core.sqlite_storage import SQLiteStorage
    from core.qdrant_storage import QdrantStorage
    from ingestion.local_embeddings import LocalEmbeddingGenerator
    
    logger.info("Starting embedding migration...")
    
    # Initialize components
    sqlite_storage = SQLiteStorage()
    qdrant_storage = QdrantStorage()
    embedding_generator = LocalEmbeddingGenerator()
    
    # Get all books
    books = await sqlite_storage.list_books()
    logger.info(f"Found {len(books)} books to migrate")
    
    for book in books:
        logger.info(f"Migrating book: {book.title}")
        
        # Get chunks
        chunks = await sqlite_storage.get_chunks_by_book(book.id)
        
        if chunks:
            # Generate new embeddings
            embeddings = await embedding_generator.generate_embeddings(chunks)
            
            # Save to Qdrant
            success = await qdrant_storage.save_embeddings(chunks, embeddings)
            
            if success:
                logger.info(f"✅ Migrated {len(chunks)} chunks")
            else:
                logger.error(f"❌ Failed to migrate chunks")
    
    # Save embedding cache
    embedding_generator.save_cache("data/embeddings/local_cache.json")
    
    # Cleanup
    await embedding_generator.cleanup()
    
    logger.info("Migration complete!")

async def verify_migration():
    """Verify the migration was successful"""
    from core.qdrant_storage import QdrantStorage
    from search.hybrid_search import HybridSearch
    
    # Check Qdrant
    storage = QdrantStorage()
    stats = await storage.get_collection_stats()
    logger.info(f"Qdrant stats: {stats}")
    
    # Test search
    search_engine = HybridSearch()
    await search_engine.initialize()
    
    test_query = "moving average"
    results = await search_engine.search_semantic(test_query, num_results=5)
    
    logger.info(f"Test search returned {results['total_results']} results")
    
    await search_engine.cleanup()

async def main():
    """Run migration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("TRADEKNOWLEDGE MIGRATION TO LOCAL SETUP")
    print("=" * 60)
    
    print("\nThis will:")
    print("1. Re-generate all embeddings using nomic-embed-text")
    print("2. Store them in Qdrant instead of ChromaDB")
    print("3. Verify the migration was successful")
    
    response = input("\nProceed? (y/n): ")
    if response.lower() != 'y':
        print("Migration cancelled")
        return
    
    # Run migration
    await migrate_embeddings()
    
    # Verify
    print("\nVerifying migration...")
    await verify_migration()
    
    print("\n✅ Migration complete!")
    print("\nNext steps:")
    print("1. Remove OpenAI API key from .env")
    print("2. Stop ChromaDB if running")
    print("3. Update any custom scripts to use new imports")

if __name__ == "__main__":
    asyncio.run(main())
EOF

chmod +x scripts/migrate_to_local.py
```

---

## Verification Script

Create a script to verify the local setup is working:

```python
# Create scripts/verify_local_setup.py
cat > scripts/verify_local_setup.py << 'EOF'
#!/usr/bin/env python3
"""
Verify local TradeKnowledge setup
"""

import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

async def check_ollama():
    """Check Ollama is running and model is available"""
    import httpx
    
    try:
        async with httpx.AsyncClient() as client:
            # Check Ollama
            response = await client.get("http://localhost:11434/api/version")
            print(f"✅ Ollama is running: {response.json()}")
            
            # Check model
            response = await client.get("http://localhost:11434/api/tags")
            models = [m['name'] for m in response.json()['models']]
            
            if 'nomic-embed-text' in models:
                print("✅ nomic-embed-text model is available")
            else:
                print("❌ nomic-embed-text model not found!")
                print("   Run: ollama pull nomic-embed-text")
                return False
                
        return True
        
    except Exception as e:
        print(f"❌ Ollama not accessible: {e}")
        print("   Make sure Ollama is running: ollama serve")
        return False

async def check_qdrant():
    """Check Qdrant is running"""
    from qdrant_client import QdrantClient
    
    try:
        client = QdrantClient(host="localhost", port=6333)
        collections = client.get_collections()
        print(f"✅ Qdrant is running: {len(collections.collections)} collections")
        return True
        
    except Exception as e:
        print(f"❌ Qdrant not accessible: {e}")
        print("   Start Qdrant with: docker run -p 6333:6333 qdrant/qdrant")
        return False

async def test_embeddings():
    """Test local embedding generation"""
    from ingestion.local_embeddings import LocalEmbeddingGenerator
    from core.models import Chunk
    
    print("\nTesting embedding generation...")
    
    try:
        generator = LocalEmbeddingGenerator()
        
        # Test single embedding
        test_text = "This is a test of local embeddings"
        embedding = await generator.generate_query_embedding(test_text)
        
        print(f"✅ Generated embedding dimension: {len(embedding)}")
        print(f"   Expected dimension: 768")
        
        if len(embedding) != 768:
            print("❌ Dimension mismatch!")
            return False
        
        # Test batch embeddings
        test_chunks = [
            Chunk(book_id="test", chunk_index=i, text=f"Test chunk {i}")
            for i in range(3)
        ]
        
        embeddings = await generator.generate_embeddings(test_chunks)
        print(f"✅ Generated {len(embeddings)} chunk embeddings")
        
        await generator.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return False

async def test_storage():
    """Test Qdrant storage"""
    from core.qdrant_storage import QdrantStorage
    from core.models import Chunk
    
    print("\nTesting Qdrant storage...")
    
    try:
        storage = QdrantStorage("test_collection")
        
        # Get stats
        stats = await storage.get_collection_stats()
        print(f"✅ Qdrant collection accessible: {stats['collection_name']}")
        
        # Test save and search
        test_chunk = Chunk(
            id="test_chunk_001",
            book_id="test_book",
            chunk_index=0,
            text="Test chunk for Qdrant storage"
        )
        
        test_embedding = [0.1] * 768  # Dummy embedding
        
        success = await storage.save_embeddings([test_chunk], [test_embedding])
        if success:
            print("✅ Successfully saved test embedding")
        else:
            print("❌ Failed to save embedding")
            return False
        
        # Test search
        results = await storage.search_semantic(test_embedding, limit=1)
        if results:
            print(f"✅ Search returned {len(results)} results")
        else:
            print("❌ Search failed")
            return False
        
        # Cleanup
        await storage.delete_embeddings([test_chunk.id])
        
        return True
        
    except Exception as e:
        print(f"❌ Storage test failed: {e}")
        return False

async def main():
    """Run all verification checks"""
    print("=" * 60)
    print("VERIFYING LOCAL TRADEKNOWLEDGE SETUP")
    print("=" * 60)
    
    checks = [
        ("Ollama", check_ollama),
        ("Qdrant", check_qdrant),
        ("Embeddings", test_embeddings),
        ("Storage", test_storage),
    ]
    
    all_passed = True
    for name, check_func in checks:
        print(f"\nChecking {name}...")
        result = await check_func()
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL CHECKS PASSED!")
        print("\nYour local TradeKnowledge setup is ready.")
        print("\nYou can now:")
        print("1. Run the migration script if you have existing data")
        print("2. Start processing books with the local setup")
        print("3. Enjoy free, fast, and private embeddings!")
    else:
        print("❌ SOME CHECKS FAILED!")
        print("\nPlease fix the issues above before proceeding.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
EOF

chmod +x scripts/verify_local_setup.py
```

---

## Summary of Changes

### What Changed

1. **Embedding Generation**
   - Replaced OpenAI API with Ollama + nomic-embed-text
   - Changed embedding dimension from 1536 to 768
   - No more API keys or costs
   - Parallel local generation for better performance

2. **Vector Storage**
   - Replaced ChromaDB with Qdrant
   - Better performance and more features
   - Proper filtering and metadata support
   - Snapshot/backup capabilities

3. **Configuration**
   - Removed OpenAI configuration
   - Added Ollama and Qdrant settings
   - Updated embedding dimensions throughout

4. **Dependencies**
   - Removed: openai, chromadb
   - Added: ollama, qdrant-client

### Migration Steps

1. **Install Local Components**
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull nomic-embed-text
   
   # Start Qdrant
   docker run -d -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
   ```

2. **Update Code**
   ```bash
   # Apply the changes described above
   # Or use provided migration scripts
   ```

3. **Verify Setup**
   ```bash
   python scripts/verify_local_setup.py
   ```

4. **Migrate Existing Data** (if applicable)
   ```bash
   python scripts/migrate_to_local.py
   ```

### Benefits of Local Setup

- **No API Costs**: Completely free after initial setup
- **Privacy**: Your data never leaves your machine
- **Speed**: No network latency for embeddings
- **Control**: Full control over models and parameters
- **Offline**: Works without internet connection

### Performance Comparison

| Metric | OpenAI API | Local Setup |
|--------|------------|-------------|
| Cost | ~$0.0001/1K tokens | Free |
| Speed | ~100-200ms/request | ~10-50ms/request |
| Batch Processing | Limited by rate limits | Limited by hardware |
| Privacy | Data sent to API | Completely local |
| Offline | No | Yes |

---

## Quick Start Commands

```bash
# 1. Install dependencies
curl -fsSL https://ollama.com/install.sh | sh
ollama pull nomic-embed-text
docker run -d -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

# 2. Update Python packages
pip install qdrant-client>=1.7.0 ollama>=0.1.7 httpx>=0.25.0
pip uninstall -y openai chromadb

# 3. Verify setup
python scripts/verify_local_setup.py

# 4. Migrate existing data (if needed)
python scripts/migrate_to_local.py

# 5. Test the system
python -c "
import asyncio
from src.ingestion.book_processor_v2 import EnhancedBookProcessor

async def test():
    processor = EnhancedBookProcessor()
    await processor.initialize()
    # Your test code here
    await processor.cleanup()

asyncio.run(test())
"
```

---

Now you're ready to create Phase 3 with the local-first approach!