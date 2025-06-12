"""
Embedding Generator for TradeKnowledge
Handles vector embedding generation for text chunks
"""

import logging
import asyncio
from typing import List, Optional, Dict, Any, Union
import hashlib
import pickle
from pathlib import Path
import os

import openai
from openai import AsyncOpenAI

from utils.logging import get_logger
from core.config import get_config
from .text_chunker import TextChunk

logger = get_logger(__name__)

class EmbeddingGenerator:
    """
    Embedding generator supporting multiple models
    
    Features:
    - OpenAI embeddings (primary)
    - Local sentence-transformers (fallback)
    - Caching for performance
    - Batch processing
    - Error handling and retries
    """
    
    def __init__(self, model_name: Optional[str] = None, cache_dir: Optional[str] = None):
        self.config = get_config()
        self.model_name = model_name or self.config.embedding.model
        self.batch_size = self.config.embedding.batch_size
        self.cache_embeddings = self.config.embedding.cache_embeddings
        
        # Setup cache directory
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize clients
        self.openai_client = None
        self.local_model = None
        
        # Model capabilities
        self.is_openai_model = "text-embedding" in self.model_name
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        
        self._initialize_embedding_model()
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model"""
        try:
            if self.is_openai_model:
                self._initialize_openai()
            else:
                self._initialize_local_model()
                
        except Exception as e:
            logger.error(f"Failed to initialize embedding model {self.model_name}: {e}")
            raise
    
    def _initialize_openai(self):
        """Initialize OpenAI client"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.openai_client = AsyncOpenAI(api_key=api_key)
        logger.info(f"Initialized OpenAI embedding model: {self.model_name}")
    
    def _initialize_local_model(self):
        """Initialize local sentence-transformers model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.local_model = SentenceTransformer(self.model_name)
            logger.info(f"Initialized local embedding model: {self.model_name}")
        except ImportError:
            raise ImportError("sentence-transformers not installed. Install with: pip install sentence-transformers")
        except Exception as e:
            logger.error(f"Failed to load local model {self.model_name}: {e}")
            raise
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        return f"{self.model_name}_{content_hash}"
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path"""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _load_from_cache(self, cache_key: str) -> Optional[List[float]]:
        """Load embedding from cache"""
        if not self.cache_embeddings:
            return None
            
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_path}: {e}")
                return None
        return None
    
    def _save_to_cache(self, cache_key: str, embedding: List[float]):
        """Save embedding to cache"""
        if not self.cache_embeddings:
            return
            
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_path}: {e}")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Check cache first
        cache_key = self._get_cache_key(text)
        cached_embedding = self._load_from_cache(cache_key)
        if cached_embedding is not None:
            return cached_embedding
        
        # Generate new embedding
        if self.is_openai_model:
            embedding = await self._generate_openai_embedding(text)
        else:
            embedding = await self._generate_local_embedding(text)
        
        # Cache the result
        self._save_to_cache(cache_key, embedding)
        
        return embedding
    
    async def _generate_openai_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API"""
        for attempt in range(self.max_retries):
            try:
                response = await self.openai_client.embeddings.create(
                    input=text,
                    model=self.model_name
                )
                return response.data[0].embedding
                
            except Exception as e:
                logger.warning(f"OpenAI embedding attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                else:
                    raise
    
    async def _generate_local_embedding(self, text: str) -> List[float]:
        """Generate embedding using local model"""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(None, self.local_model.encode, text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Local embedding generation failed: {e}")
            raise
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        if not texts:
            return []
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Check cache for all texts
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            cached_embedding = self._load_from_cache(cache_key)
            
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
            else:
                embeddings.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            if self.is_openai_model:
                new_embeddings = await self._generate_openai_embeddings_batch(uncached_texts)
            else:
                new_embeddings = await self._generate_local_embeddings_batch(uncached_texts)
            
            # Fill in the new embeddings and cache them
            for idx, embedding in zip(uncached_indices, new_embeddings):
                embeddings[idx] = embedding
                cache_key = self._get_cache_key(texts[idx])
                self._save_to_cache(cache_key, embedding)
        
        logger.info(f"Generated {len(uncached_texts)} new embeddings, {len(texts) - len(uncached_texts)} from cache")
        return embeddings
    
    async def _generate_openai_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API in batches"""
        all_embeddings = []
        
        # Process in batches to respect API limits
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            for attempt in range(self.max_retries):
                try:
                    response = await self.openai_client.embeddings.create(
                        input=batch,
                        model=self.model_name
                    )
                    
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    break
                    
                except Exception as e:
                    logger.warning(f"OpenAI batch embedding attempt {attempt + 1} failed: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    else:
                        raise
        
        return all_embeddings
    
    async def _generate_local_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local model in batches"""
        try:
            # Process in batches
            all_embeddings = []
            
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                
                # Run in thread pool
                loop = asyncio.get_event_loop()
                batch_embeddings = await loop.run_in_executor(None, self.local_model.encode, batch)
                all_embeddings.extend([emb.tolist() for emb in batch_embeddings])
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Local batch embedding generation failed: {e}")
            raise
    
    async def generate_chunk_embeddings(self, chunks: List[TextChunk]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for text chunks
        
        Returns:
            List of dicts with chunk metadata and embeddings
        """
        if not chunks:
            return []
        
        # Extract texts from chunks
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings
        embeddings = await self.generate_embeddings_batch(texts)
        
        # Combine with chunk metadata
        results = []
        for chunk, embedding in zip(chunks, embeddings):
            results.append({
                "chunk_id": chunk.chunk_id,
                "book_id": chunk.book_id,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "embedding": embedding,
                "metadata": chunk.metadata,
                "content_type": chunk.content_type,
                "boundary_type": chunk.boundary_type.value
            })
        
        return results
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for this model"""
        if self.is_openai_model:
            # Known dimensions for OpenAI models
            dimensions = {
                "text-embedding-ada-002": 1536,
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072
            }
            return dimensions.get(self.model_name, 1536)  # Default
        else:
            if self.local_model:
                return self.local_model.get_sentence_embedding_dimension()
            return 768  # Common default for sentence transformers
    
    def clear_cache(self):
        """Clear embedding cache"""
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {cache_file}: {e}")
            logger.info("Embedding cache cleared")

async def generate_embeddings_for_chunks(chunks: List[TextChunk], 
                                       model_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Convenience function to generate embeddings for chunks"""
    generator = EmbeddingGenerator(model_name=model_name)
    return await generator.generate_chunk_embeddings(chunks)