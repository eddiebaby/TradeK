"""
Advanced Caching Layer for TradeKnowledge
Provides intelligent caching for search results, embeddings, and API responses
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

import redis.asyncio as redis

from src.core.config import get_config
from src.core.models import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata"""

    key: str
    value: Any
    created_at: float
    expires_at: float | None
    hit_count: int = 0
    size_bytes: int = 0
    tags: list[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds"""
        return time.time() - self.created_at


class IntelligentCache:
    """
    Intelligent caching system with multiple backends and strategies
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or self._get_default_config()
        self.memory_cache: dict[str, CacheEntry] = {}
        self.redis_client: redis.Redis | None = None
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size_bytes": 0,
            "total_requests": 0,
        }
        self._cleanup_task: asyncio.Task | None = None

    def _get_default_config(self) -> dict[str, Any]:
        """Get default cache configuration"""
        app_config = get_config()
        return {
            "memory_max_size": 100 * 1024 * 1024,  # 100MB
            "memory_max_entries": 10000,
            "default_ttl": 3600,  # 1 hour
            "redis_url": f"redis://{app_config.redis.host}:{app_config.redis.port}",
            "redis_db": 0,
            "enable_redis": True,
            "enable_memory": True,
            "cleanup_interval": 300,  # 5 minutes
            "prefetch_enabled": True,
            "compression_enabled": True,
        }

    async def initialize(self):
        """Initialize cache backends"""
        try:
            if self.config["enable_redis"]:
                self.redis_client = redis.from_url(
                    self.config["redis_url"],
                    db=self.config["redis_db"],
                    encoding="utf-8",
                    decode_responses=True,
                )
                await self.redis_client.ping()
                logger.info("Redis cache backend initialized")
        except Exception as e:
            logger.warning(f"Redis not available, using memory-only cache: {e}")
            self.redis_client = None

        # Start cleanup task
        if self.config["enable_memory"]:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info("Intelligent cache initialized")

    async def close(self):
        """Close cache connections"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        if self.redis_client:
            await self.redis_client.close()

    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = f"{prefix}:{':'.join(str(arg) for arg in args)}"
        if kwargs:
            sorted_kwargs = sorted(kwargs.items())
            kwargs_str = ":".join(f"{k}={v}" for k, v in sorted_kwargs)
            key_data += f":{kwargs_str}"

        # Hash long keys
        if len(key_data) > 200:
            key_hash = hashlib.sha256(key_data.encode()).hexdigest()[:16]
            return f"{prefix}:hash:{key_hash}"

        return key_data

    async def get(self, key: str) -> Any | None:
        """Get value from cache"""
        self.stats["total_requests"] += 1

        # Try memory cache first
        if self.config["enable_memory"] and key in self.memory_cache:
            entry = self.memory_cache[key]
            if not entry.is_expired:
                entry.hit_count += 1
                self.stats["hits"] += 1
                logger.debug(f"Memory cache hit: {key}")
                return entry.value
            else:
                # Remove expired entry
                del self.memory_cache[key]

        # Try Redis cache
        if self.redis_client:
            try:
                redis_value = await self.redis_client.get(key)
                if redis_value:
                    value = json.loads(redis_value)
                    self.stats["hits"] += 1

                    # Populate memory cache for faster access
                    if self.config["enable_memory"]:
                        await self._store_in_memory(key, value)

                    logger.debug(f"Redis cache hit: {key}")
                    return value
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")

        self.stats["misses"] += 1
        logger.debug(f"Cache miss: {key}")
        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        tags: list[str] | None = None,
    ):
        """Set value in cache"""
        if ttl is None:
            ttl = self.config["default_ttl"]

        tags = tags or []
        expires_at = time.time() + ttl if ttl > 0 else None

        # Store in memory cache
        if self.config["enable_memory"]:
            await self._store_in_memory(key, value, expires_at, tags)

        # Store in Redis cache
        if self.redis_client:
            try:
                serialized_value = json.dumps(value, default=str)
                await self.redis_client.setex(key, ttl, serialized_value)
                logger.debug(f"Stored in Redis cache: {key}")
            except Exception as e:
                logger.warning(f"Redis cache storage error: {e}")

    async def _store_in_memory(
        self,
        key: str,
        value: Any,
        expires_at: float | None = None,
        tags: list[str] | None = None,
    ):
        """Store value in memory cache"""
        try:
            serialized_value = json.dumps(value, default=str)
            size_bytes = len(serialized_value.encode("utf-8"))

            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                expires_at=expires_at,
                size_bytes=size_bytes,
                tags=tags or [],
            )

            # Check memory limits
            await self._enforce_memory_limits(size_bytes)

            self.memory_cache[key] = entry
            self.stats["size_bytes"] += size_bytes

        except Exception as e:
            logger.warning(f"Memory cache storage error: {e}")

    async def _enforce_memory_limits(self, new_entry_size: int):
        """Enforce memory cache size limits"""
        # Check entry count limit
        if len(self.memory_cache) >= self.config["memory_max_entries"]:
            await self._evict_lru_entries(1)

        # Check size limit
        while (self.stats["size_bytes"] + new_entry_size) > self.config[
            "memory_max_size"
        ]:
            await self._evict_lru_entries(max(1, len(self.memory_cache) // 10))

    async def _evict_lru_entries(self, count: int):
        """Evict least recently used entries"""
        if not self.memory_cache:
            return

        # Sort by hit count and age (LRU)
        sorted_entries = sorted(
            self.memory_cache.items(), key=lambda x: (x[1].hit_count, -x[1].age_seconds)
        )

        evicted = 0
        for key, entry in sorted_entries:
            if evicted >= count:
                break

            del self.memory_cache[key]
            self.stats["size_bytes"] -= entry.size_bytes
            self.stats["evictions"] += 1
            evicted += 1

        logger.debug(f"Evicted {evicted} cache entries")

    async def delete(self, key: str):
        """Delete entry from cache"""
        # Remove from memory
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            self.stats["size_bytes"] -= entry.size_bytes
            del self.memory_cache[key]

        # Remove from Redis
        if self.redis_client:
            try:
                await self.redis_client.delete(key)
            except Exception as e:
                logger.warning(f"Redis cache deletion error: {e}")

    async def clear_by_tags(self, tags: list[str]):
        """Clear cache entries by tags"""
        keys_to_delete = []

        # Find keys with matching tags in memory cache
        for key, entry in self.memory_cache.items():
            if any(tag in entry.tags for tag in tags):
                keys_to_delete.append(key)

        # Delete found keys
        for key in keys_to_delete:
            await self.delete(key)

        logger.info(f"Cleared {len(keys_to_delete)} cache entries with tags: {tags}")

    async def clear_all(self):
        """Clear all cache entries"""
        self.memory_cache.clear()
        self.stats["size_bytes"] = 0

        if self.redis_client:
            try:
                await self.redis_client.flushdb()
            except Exception as e:
                logger.warning(f"Redis cache clear error: {e}")

        logger.info("Cleared all cache entries")

    async def _cleanup_loop(self):
        """Background cleanup task"""
        while True:
            try:
                await asyncio.sleep(self.config["cleanup_interval"])
                await self._cleanup_expired_entries()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")

    async def _cleanup_expired_entries(self):
        """Remove expired entries from memory cache"""
        expired_keys = []

        for key, entry in self.memory_cache.items():
            if entry.is_expired:
                expired_keys.append(key)

        for key in expired_keys:
            entry = self.memory_cache[key]
            self.stats["size_bytes"] -= entry.size_bytes
            del self.memory_cache[key]

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        hit_rate = 0.0
        if self.stats["total_requests"] > 0:
            hit_rate = self.stats["hits"] / self.stats["total_requests"]

        return {
            **self.stats,
            "hit_rate": hit_rate,
            "memory_entries": len(self.memory_cache),
            "memory_size_mb": self.stats["size_bytes"] / (1024 * 1024),
            "redis_connected": self.redis_client is not None,
        }


class SearchCache:
    """Specialized cache for search operations"""

    def __init__(self, cache: IntelligentCache):
        self.cache = cache

    def _search_key(
        self,
        query: str,
        max_results: int = 10,
        min_score: float = 0.0,
        filters: dict[str, Any] | None = None,
    ) -> str:
        """Generate cache key for search query"""
        return self.cache._generate_key(
            "search",
            query=query,
            max_results=max_results,
            min_score=min_score,
            filters=filters or {},
        )

    async def get_search_results(
        self,
        query: str,
        max_results: int = 10,
        min_score: float = 0.0,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult] | None:
        """Get cached search results"""
        key = self._search_key(query, max_results, min_score, filters)
        cached_data = await self.cache.get(key)

        if cached_data:
            # Convert back to SearchResult objects
            return [SearchResult(**result) for result in cached_data]

        return None

    async def cache_search_results(
        self,
        query: str,
        results: list[SearchResult],
        max_results: int = 10,
        min_score: float = 0.0,
        filters: dict[str, Any] | None = None,
        ttl: int = 1800,
    ):
        """Cache search results"""
        key = self._search_key(query, max_results, min_score, filters)

        # Convert SearchResult objects to dict for serialization
        serializable_results = [asdict(result) for result in results]

        await self.cache.set(
            key, serializable_results, ttl=ttl, tags=["search", "results"]
        )

    async def invalidate_search_cache(self):
        """Invalidate all search-related cache entries"""
        await self.cache.clear_by_tags(["search"])


class EmbeddingCache:
    """Specialized cache for embedding operations"""

    def __init__(self, cache: IntelligentCache):
        self.cache = cache

    def _embedding_key(self, text: str, model: str = "default") -> str:
        """Generate cache key for embedding"""
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return self.cache._generate_key("embedding", model, text_hash)

    async def get_embedding(
        self, text: str, model: str = "default"
    ) -> list[float] | None:
        """Get cached embedding"""
        key = self._embedding_key(text, model)
        return await self.cache.get(key)

    async def cache_embedding(
        self,
        text: str,
        embedding: list[float],
        model: str = "default",
        ttl: int = 86400,
    ):
        """Cache embedding (24 hour default TTL)"""
        key = self._embedding_key(text, model)
        await self.cache.set(key, embedding, ttl=ttl, tags=["embedding", model])

    async def invalidate_model_cache(self, model: str = "default"):
        """Invalidate cache for specific model"""
        await self.cache.clear_by_tags([model])


# Global cache instances
_cache_instance: IntelligentCache | None = None
_search_cache_instance: SearchCache | None = None
_embedding_cache_instance: EmbeddingCache | None = None


async def get_cache() -> IntelligentCache:
    """Get global cache instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = IntelligentCache()
        await _cache_instance.initialize()
    return _cache_instance


async def get_search_cache() -> SearchCache:
    """Get global search cache instance"""
    global _search_cache_instance
    if _search_cache_instance is None:
        cache = await get_cache()
        _search_cache_instance = SearchCache(cache)
    return _search_cache_instance


async def get_embedding_cache() -> EmbeddingCache:
    """Get global embedding cache instance"""
    global _embedding_cache_instance
    if _embedding_cache_instance is None:
        cache = await get_cache()
        _embedding_cache_instance = EmbeddingCache(cache)
    return _embedding_cache_instance


async def cleanup_cache():
    """Cleanup all cache instances"""
    global _cache_instance, _search_cache_instance, _embedding_cache_instance

    if _cache_instance:
        await _cache_instance.close()
        _cache_instance = None

    _search_cache_instance = None
    _embedding_cache_instance = None


# Cache decorators
def cache_search_results(ttl: int = 1800):
    """Decorator to cache search results"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            search_cache = await get_search_cache()

            # Extract search parameters
            query = kwargs.get("query") or (args[1] if len(args) > 1 else None)
            max_results = kwargs.get("max_results", 10)
            min_score = kwargs.get("min_score", 0.0)
            filters = kwargs.get("filters")

            if query:
                # Try to get from cache
                cached_results = await search_cache.get_search_results(
                    query, max_results, min_score, filters
                )
                if cached_results is not None:
                    logger.debug(f"Search cache hit for query: {query}")
                    return cached_results

            # Execute original function
            results = await func(*args, **kwargs)

            # Cache the results
            if query and results:
                await search_cache.cache_search_results(
                    query, results, max_results, min_score, filters, ttl
                )
                logger.debug(f"Cached search results for query: {query}")

            return results

        return wrapper

    return decorator


def cache_embeddings(ttl: int = 86400):
    """Decorator to cache embeddings"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            embedding_cache = await get_embedding_cache()

            # Extract text parameter
            texts = kwargs.get("texts") or (args[1] if len(args) > 1 else None)
            model = kwargs.get("model", "default")

            if texts and isinstance(texts, list) and len(texts) == 1:
                text = texts[0]

                # Try to get from cache
                cached_embedding = await embedding_cache.get_embedding(text, model)
                if cached_embedding is not None:
                    logger.debug(f"Embedding cache hit for text: {text[:50]}...")
                    return [cached_embedding]

            # Execute original function
            embeddings = await func(*args, **kwargs)

            # Cache the embeddings
            if texts and embeddings and isinstance(texts, list) and len(texts) == 1:
                await embedding_cache.cache_embedding(
                    texts[0], embeddings[0], model, ttl
                )
                logger.debug(f"Cached embedding for text: {texts[0][:50]}...")

            return embeddings

        return wrapper

    return decorator
