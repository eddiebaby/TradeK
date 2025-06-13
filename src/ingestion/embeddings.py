"""
Embedding generation for semantic search

This module handles converting text chunks into vector embeddings
that can be used for semantic similarity search.
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from datetime import datetime
import hashlib
import json

import openai
from openai import OpenAI
import numpy as np

from core.models import Chunk
from core.config import get_config

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Generates embeddings for text chunks.
    
    This class supports multiple embedding models:
    1. OpenAI embeddings (requires API key)
    2. Local sentence transformers (no API needed)
    
    The embeddings capture semantic meaning, allowing us to find
    similar content even when different words are used.
    """
    
    def __init__(self, config=None, model_name: Optional[str] = None):
        """
        Initialize the embedding generator.
        
        Args:
            config: Configuration object
            model_name: Name of the embedding model to use
        """
        self.config = config or get_config()
        self.model_name = model_name or self.config.embedding.model
        
        # Initialize the appropriate model
        if self.model_name.startswith("text-embedding"):
            # OpenAI model
            self._init_openai()
        else:
            # Local sentence transformer
            self._init_sentence_transformer()
        
        # Cache for embeddings (avoid regenerating)
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def _init_openai(self):
        """Initialize OpenAI client"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key.strip() == "" or api_key.lower().startswith("your"):
            raise ValueError(
                "OpenAI API key not found! Please set OPENAI_API_KEY in .env file"
            )
        
        self.client = OpenAI(api_key=api_key)
        self.embedding_dimension = 1536  # for ada-002
        self.is_local = False
        logger.info(f"Initialized OpenAI embeddings with model: {self.model_name}")
        
    def _init_sentence_transformer(self):
        """Initialize local sentence transformer"""
        logger.info(f"Loading sentence transformer: {self.model_name}")
        
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Check if CUDA is available for GPU acceleration
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            self.model = SentenceTransformer(self.model_name, device=device)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            self.is_local = True
            
            logger.info(f"Loaded model with dimension: {self.embedding_dimension}")
        except ImportError:
            logger.error("sentence-transformers not installed. Attempting OpenAI fallback.")
            try:
                self._init_openai()
            except ValueError as e:
                raise RuntimeError(f"Neither local embeddings nor OpenAI available: {e}") from e
        
    async def generate_embeddings(self, 
                                  chunks: List[Chunk],
                                  show_progress: bool = True) -> List[List[float]]:
        """
        Generate embeddings for a list of chunks.
        
        This is the main method for generating embeddings. It handles:
        - Batching for efficiency
        - Caching to avoid regeneration
        - Progress tracking
        - Error handling and retries
        
        Args:
            chunks: List of chunks to embed
            show_progress: Whether to show progress bar
            
        Returns:
            List of embedding vectors
        """
        if not chunks:
            return []
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        # Group chunks by whether they're cached
        cached_chunks = []
        uncached_chunks = []
        
        for chunk in chunks:
            cache_key = self._get_cache_key(chunk.text)
            if cache_key in self.cache:
                cached_chunks.append((chunk, self.cache[cache_key]))
                self.cache_hits += 1
            else:
                uncached_chunks.append(chunk)
                self.cache_misses += 1
        
        logger.info(f"Cache hits: {len(cached_chunks)}, misses: {len(uncached_chunks)}")
        
        # Generate embeddings for uncached chunks
        if uncached_chunks:
            if self.is_local:
                new_embeddings = await self._generate_local_embeddings(uncached_chunks)
            else:
                new_embeddings = await self._generate_openai_embeddings(uncached_chunks)
            
            # Add to cache
            for chunk, embedding in zip(uncached_chunks, new_embeddings):
                cache_key = self._get_cache_key(chunk.text)
                self.cache[cache_key] = embedding
        else:
            new_embeddings = []
        
        # Combine cached and new embeddings in original order
        result = []
        cached_dict = {chunk.id: embedding for chunk, embedding in cached_chunks}
        new_iter = iter(new_embeddings)
        
        for chunk in chunks:
            if chunk.id in cached_dict:
                result.append(cached_dict[chunk.id])
            else:
                result.append(next(new_iter))
        
        return result
    
    async def _generate_openai_embeddings(self, chunks: List[Chunk]) -> List[List[float]]:
        """Generate embeddings using OpenAI API"""
        embeddings = []
        batch_size = 100  # OpenAI batch size limit
        
        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [chunk.text for chunk in batch]
            
            try:
                # Make API call
                response = await asyncio.to_thread(
                    self.client.embeddings.create,
                    input=texts,
                    model=self.model_name
                )
                
                # Extract embeddings
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                logger.debug(f"Generated {len(batch_embeddings)} embeddings")
                
                # Rate limiting - be nice to the API
                if i + batch_size < len(chunks):
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error generating OpenAI embeddings: {e}")
                # Re-raise the exception instead of silently failing
                raise RuntimeError(f"Failed to generate embeddings for batch {i//batch_size + 1}: {e}") from e
        
        return embeddings
    
    async def _generate_local_embeddings(self, chunks: List[Chunk]) -> List[List[float]]:
        """Generate embeddings using local model"""
        texts = [chunk.text for chunk in chunks]
        
        try:
            # Generate embeddings
            # Run in thread to avoid blocking
            embeddings = await asyncio.to_thread(
                self.model.encode,
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Convert to list format
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Error generating local embeddings: {e}")
            # Return zero vectors for failed chunks
            return [[0.0] * self.embedding_dimension] * len(chunks)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        # Include model name in cache key
        key_string = f"{self.model_name}:{text}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        
        Queries are handled separately because they might need
        different processing than document chunks.
        """
        # Check cache first
        cache_key = self._get_cache_key(query)
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        self.cache_misses += 1
        
        # Generate embedding
        if self.is_local:
            embedding = await asyncio.to_thread(
                self.model.encode,
                [query],
                convert_to_numpy=True
            )
            result = embedding[0].tolist()
        else:
            response = await asyncio.to_thread(
                self.client.embeddings.create,
                input=[query],
                model=self.model_name
            )
            result = response.data[0].embedding
        
        # Cache it
        self.cache[cache_key] = result
        
        return result
    
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
        if not os.path.exists(file_path):
            logger.warning(f"Cache file not found: {file_path}")
            return
        
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding generation statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dimension,
            'is_local': self.is_local,
            'cache_size': len(self.cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': hit_rate,
            'total_requests': total_requests
        }

# Utility functions for testing and validation

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

async def test_embedding_generator():
    """Test the embedding generator"""
    
    # Create test chunks
    test_chunks = [
        Chunk(
            book_id="test",
            chunk_index=0,
            text="Moving averages are technical indicators used in trading."
        ),
        Chunk(
            book_id="test",
            chunk_index=1,
            text="The simple moving average calculates the mean of prices."
        ),
        Chunk(
            book_id="test", 
            chunk_index=2,
            text="Python is a programming language used for data analysis."
        )
    ]
    
    # Test with OpenAI model
    print("Testing with OpenAI model...")
    try:
        generator = EmbeddingGenerator("text-embedding-ada-002")
        
        # Generate embeddings
        embeddings = await generator.generate_embeddings(test_chunks)
        
        print(f"Generated {len(embeddings)} embeddings")
        print(f"Embedding dimension: {len(embeddings[0])}")
        
        # Test similarity
        print("\nTesting semantic similarity:")
        query = "What are moving averages in trading?"
        query_embedding = await generator.generate_query_embedding(query)
        
        for i, (chunk, embedding) in enumerate(zip(test_chunks, embeddings)):
            similarity = cosine_similarity(query_embedding, embedding)
            print(f"Chunk {i}: {similarity:.3f} - {chunk.text[:50]}...")
        
        # Show stats
        print(f"\nStats: {generator.get_stats()}")
        
    except ValueError as e:
        print(f"OpenAI test failed: {e}")
        print("Make sure to set OPENAI_API_KEY in .env file")

if __name__ == "__main__":
    asyncio.run(test_embedding_generator())