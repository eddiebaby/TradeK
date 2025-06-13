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
        
        # Verify Ollama is running (will be called lazily)
        self._ollama_verified = False
        
    async def _verify_ollama(self):
        """Verify Ollama is running and model is available"""
        if self._ollama_verified:
            return
            
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
            
            self._ollama_verified = True
                    
        except Exception as e:
            logger.error(f"Cannot connect to Ollama at {self.ollama_host}: {e}")
            logger.info("Make sure Ollama is running: ollama serve")
    
    async def generate_embeddings(self, 
                                  chunks: List[Chunk],
                                  show_progress: bool = True) -> List[List[float]]:
        """Generate embeddings for chunks using Ollama"""
        if not chunks:
            return []
        
        # Verify Ollama connection
        await self._verify_ollama()
        
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
        
        # Verify Ollama connection
        await self._verify_ollama()
        
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