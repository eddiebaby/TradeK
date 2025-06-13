"""
Advanced caching system for TradeKnowledge

This implements a multi-level cache with Redis and in-memory storage
for optimal performance.
"""

import logging
import json
import pickle
import hashlib
import asyncio
from typing import Any, Optional, Dict, List, Callable
from datetime import datetime, timedelta
from functools import wraps
from cachetools import TTLCache, LRUCache
import zlib

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Multi-level cache manager with Redis and memory caching.
    
    Features:
    - Two-level caching (memory -> Redis)
    - Compression for large values
    - TTL support
    - Cache warming
    - Statistics tracking
    """
    
    def __init__(self, redis_enabled: bool = True):
        """Initialize cache manager"""
        # Memory caches
        self.memory_cache = TTLCache(maxsize=1000, ttl=1800)  # 30 minutes
        
        # Specialized caches
        self.embedding_cache = LRUCache(maxsize=10000)
        self.search_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour
        
        # Redis connection
        self.redis_client: Optional[Any] = None
        self.redis_enabled = redis_enabled
        
        # Statistics
        self.stats = {
            'memory_hits': 0,
            'memory_misses': 0,
            'redis_hits': 0,
            'redis_misses': 0,
            'total_requests': 0
        }
        
        # Compression threshold (compress if larger than 1KB)
        self.compression_threshold = 1024
    
    async def initialize(self):
        """Initialize Redis connection"""
        if not self.redis_enabled:
            logger.info("Redis caching disabled, using memory cache only")
            return
            
        try:
            import redis.asyncio as redis
            
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                db=0,
                decode_responses=False  # We'll handle encoding
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Redis cache connected successfully")
            
        except ImportError:
            logger.warning("Redis not available. Install redis-py for Redis caching.")
            self.redis_client = None
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using memory cache only.")
            self.redis_client = None
    
    async def get(self, 
                  key: str, 
                  cache_type: str = 'general') -> Optional[Any]:
        """
        Get value from cache (memory first, then Redis).
        
        Args:
            key: Cache key
            cache_type: Type of cache to use
            
        Returns:
            Cached value or None
        """
        self.stats['total_requests'] += 1
        
        # Select appropriate memory cache
        memory_cache = self._get_cache_by_type(cache_type)
        
        # Try memory cache first
        if key in memory_cache:
            self.stats['memory_hits'] += 1
            logger.debug(f"Memory cache hit: {key}")
            return memory_cache[key]
        
        self.stats['memory_misses'] += 1
        
        # Try Redis if available
        if self.redis_client:
            try:
                redis_key = self._make_redis_key(key, cache_type)
                data = await self.redis_client.get(redis_key)
                
                if data:
                    self.stats['redis_hits'] += 1
                    logger.debug(f"Redis cache hit: {key}")
                    
                    # Deserialize
                    value = self._deserialize(data)
                    
                    # Store in memory cache for faster access
                    memory_cache[key] = value
                    
                    return value
                else:
                    self.stats['redis_misses'] += 1
                    
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.error(f"Redis get error: {e}")
        
        return None
    
    async def set(self,
                  key: str,
                  value: Any,
                  cache_type: str = 'general',
                  ttl: Optional[int] = None) -> bool:
        """
        Set value in cache (both memory and Redis).
        
        Args:
            key: Cache key
            value: Value to cache
            cache_type: Type of cache to use
            ttl: Time to live in seconds
            
        Returns:
            Success status
        """
        try:
            # Store in memory cache
            memory_cache = self._get_cache_by_type(cache_type)
            memory_cache[key] = value
            
            # Store in Redis if available
            if self.redis_client:
                redis_key = self._make_redis_key(key, cache_type)
                serialized = self._serialize(value)
                
                # Set with TTL
                if ttl is None:
                    ttl = 3600  # Default 1 hour
                
                await self.redis_client.setex(
                    redis_key,
                    ttl,
                    serialized
                )
                
                logger.debug(f"Cached {key} (size: {len(serialized)} bytes)")
            
            return True
            
        except (ConnectionError, TimeoutError, OSError, MemoryError) as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str, cache_type: str = 'general') -> bool:
        """Delete value from cache"""
        try:
            # Remove from memory
            memory_cache = self._get_cache_by_type(cache_type)
            memory_cache.pop(key, None)
            
            # Remove from Redis
            if self.redis_client:
                redis_key = self._make_redis_key(key, cache_type)
                await self.redis_client.delete(redis_key)
            
            return True
            
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def clear(self, cache_type: Optional[str] = None) -> bool:
        """Clear cache (optionally by type)"""
        try:
            if cache_type:
                # Clear specific cache type
                memory_cache = self._get_cache_by_type(cache_type)
                memory_cache.clear()
                
                if self.redis_client:
                    pattern = f"{cache_type}:*"
                    async for key in self.redis_client.scan_iter(match=pattern):
                        await self.redis_client.delete(key)
            else:
                # Clear all caches
                self.memory_cache.clear()
                self.embedding_cache.clear()
                self.search_cache.clear()
                
                if self.redis_client:
                    await self.redis_client.flushdb()
            
            logger.info(f"Cleared cache: {cache_type or 'all'}")
            return True
            
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    def _get_cache_by_type(self, cache_type: str):
        """Get appropriate cache by type"""
        if cache_type == 'embedding':
            return self.embedding_cache
        elif cache_type == 'search':
            return self.search_cache
        else:
            return self.memory_cache
    
    def _make_redis_key(self, key: str, cache_type: str) -> str:
        """Create Redis key with namespace"""
        return f"tradeknowledge:{cache_type}:{key}"
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage"""
        # Pickle the value
        data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Compress if large
        if len(data) > self.compression_threshold:
            data = b'Z' + zlib.compress(data, level=6)
        else:
            data = b'U' + data  # Uncompressed marker
        
        return data
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        if not data:
            return None
        
        # Check compression marker
        if data[0:1] == b'Z':
            # Decompress
            data = zlib.decompress(data[1:])
        else:
            # Remove marker
            data = data[1:]
        
        # Unpickle
        return pickle.loads(data)
    
    def cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        # Create a string representation
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key_string = ":".join(key_parts)
        
        # Hash for consistent length
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_hits = self.stats['memory_hits'] + self.stats['redis_hits']
        total_misses = self.stats['memory_misses']  # Redis miss counted only after memory miss
        
        hit_rate = total_hits / self.stats['total_requests'] if self.stats['total_requests'] > 0 else 0
        
        return {
            'total_requests': self.stats['total_requests'],
            'memory_hits': self.stats['memory_hits'],
            'memory_misses': self.stats['memory_misses'],
            'redis_hits': self.stats['redis_hits'],
            'redis_misses': self.stats['redis_misses'],
            'hit_rate': hit_rate,
            'memory_cache_size': len(self.memory_cache),
            'embedding_cache_size': len(self.embedding_cache),
            'search_cache_size': len(self.search_cache)
        }
    
    async def warm_cache(self, keys_and_values: List[tuple]):
        """Warm cache with predefined key-value pairs"""
        logger.info(f"Warming cache with {len(keys_and_values)} items...")
        
        for key, value, cache_type in keys_and_values:
            await self.set(key, value, cache_type)
        
        logger.info("Cache warming completed")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.redis_client:
            await self.redis_client.close()


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None

async def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance"""
    global _cache_manager
    
    if _cache_manager is None:
        _cache_manager = CacheManager()
        await _cache_manager.initialize()
    
    return _cache_manager


def cached(cache_type: str = 'general', ttl: Optional[int] = None):
    """
    Decorator for caching function results.
    
    Args:
        cache_type: Type of cache to use
        ttl: Time to live in seconds
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_manager = await get_cache_manager()
            
            # Generate cache key
            key = cache_manager.cache_key(func.__name__, *args, **kwargs)
            
            # Try to get from cache
            result = await cache_manager.get(key, cache_type)
            if result is not None:
                return result
            
            # Call function and cache result
            result = await func(*args, **kwargs)
            await cache_manager.set(key, result, cache_type, ttl)
            
            return result
        
        return wrapper
    return decorator


# Example usage
async def example_usage():
    """Example of using the cache manager"""
    # Get cache manager
    cache = await get_cache_manager()
    
    # Basic caching
    await cache.set("user:123", {"name": "John", "books": 42})
    user = await cache.get("user:123")
    print(f"User: {user}")
    
    # Embedding caching
    embeddings = [0.1, 0.2, 0.3, 0.4, 0.5]
    await cache.set("embeddings:doc123", embeddings, cache_type='embedding')
    
    # Search result caching
    search_results = ["result1", "result2", "result3"]
    await cache.set("search:trading", search_results, cache_type='search', ttl=1800)
    
    # Get statistics
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")
    
    # Using decorator
    @cached(cache_type='search', ttl=3600)
    async def expensive_search(query: str):
        # Simulate expensive operation
        await asyncio.sleep(1)
        return f"Results for: {query}"
    
    # First call will execute function
    result1 = await expensive_search("trading strategies")
    
    # Second call will use cache
    result2 = await expensive_search("trading strategies")
    
    print(f"Results: {result1} == {result2}")


if __name__ == "__main__":
    asyncio.run(example_usage())