"""
Comprehensive tests for the intelligent caching layer
Tests all caching functionality including Redis, memory cache, and specialized caches
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch
from typing import List, Dict, Any

from src.core.caching import (
    IntelligentCache, SearchCache, EmbeddingCache, CacheEntry,
    get_cache, get_search_cache, get_embedding_cache, cleanup_cache,
    cache_search_results, cache_embeddings
)
from src.core.models import SearchResult


class TestCacheEntry:
    """Test CacheEntry data structure"""
    
    def test_cache_entry_creation(self):
        """Test CacheEntry creation and properties"""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=time.time(),
            expires_at=time.time() + 3600,
            size_bytes=100,
            tags=["test", "cache"]
        )
        
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert not entry.is_expired
        assert entry.age_seconds >= 0
        assert "test" in entry.tags
    
    def test_cache_entry_expiration(self):
        """Test cache entry expiration logic"""
        # Expired entry
        expired_entry = CacheEntry(
            key="expired",
            value="value",
            created_at=time.time() - 7200,
            expires_at=time.time() - 3600  # Expired 1 hour ago
        )
        assert expired_entry.is_expired
        
        # Non-expiring entry
        persistent_entry = CacheEntry(
            key="persistent",
            value="value",
            created_at=time.time(),
            expires_at=None
        )
        assert not persistent_entry.is_expired


class TestIntelligentCache:
    """Test the core IntelligentCache functionality"""
    
    @pytest.fixture
    async def cache(self):
        """Create test cache instance"""
        config = {
            "memory_max_size": 1024 * 1024,  # 1MB
            "memory_max_entries": 100,
            "default_ttl": 3600,
            "enable_redis": False,  # Disable Redis for unit tests
            "enable_memory": True,
            "cleanup_interval": 1,
            "redis_url": "redis://localhost:6379"
        }
        
        cache = IntelligentCache(config)
        await cache.initialize()
        yield cache
        await cache.close()
    
    async def test_cache_initialization(self, cache):
        """Test cache initialization"""
        stats = cache.get_stats()
        assert stats["memory_entries"] == 0
        assert stats["memory_size_mb"] == 0.0
        assert not stats["redis_connected"]  # Redis disabled for tests
    
    async def test_basic_get_set(self, cache):
        """Test basic get/set operations"""
        # Set a value
        await cache.set("test_key", "test_value", ttl=3600)
        
        # Get the value
        value = await cache.get("test_key")
        assert value == "test_value"
        
        # Get non-existent key
        value = await cache.get("nonexistent")
        assert value is None
    
    async def test_ttl_expiration(self, cache):
        """Test TTL-based expiration"""
        # Set value with short TTL
        await cache.set("short_ttl", "value", ttl=1)
        
        # Should be available immediately
        value = await cache.get("short_ttl")
        assert value == "value"
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Should be expired and cleaned up
        value = await cache.get("short_ttl")
        assert value is None
    
    async def test_cache_tags(self, cache):
        """Test cache tagging and clearing by tags"""
        # Set values with tags
        await cache.set("tagged1", "value1", tags=["group1", "test"])
        await cache.set("tagged2", "value2", tags=["group1"])
        await cache.set("tagged3", "value3", tags=["group2"])
        
        # Verify values are set
        assert await cache.get("tagged1") == "value1"
        assert await cache.get("tagged2") == "value2"
        assert await cache.get("tagged3") == "value3"
        
        # Clear by tag
        await cache.clear_by_tags(["group1"])
        
        # Check results
        assert await cache.get("tagged1") is None  # Cleared (has group1 tag)
        assert await cache.get("tagged2") is None  # Cleared (has group1 tag)
        assert await cache.get("tagged3") == "value3"  # Not cleared (group2 tag)
    
    async def test_memory_limits(self, cache):
        """Test memory limit enforcement"""
        # Fill cache near limit
        for i in range(50):
            large_value = "x" * 10000  # 10KB each
            await cache.set(f"large_{i}", large_value)
        
        stats = cache.get_stats()
        assert stats["memory_entries"] <= cache.config["memory_max_entries"]
        assert stats["size_bytes"] <= cache.config["memory_max_size"]
    
    async def test_lru_eviction(self, cache):
        """Test LRU eviction policy"""
        # Set many values to trigger eviction
        for i in range(150):  # More than max_entries (100)
            await cache.set(f"key_{i}", f"value_{i}")
        
        stats = cache.get_stats()
        assert stats["memory_entries"] <= cache.config["memory_max_entries"]
        assert stats["evictions"] > 0
        
        # First entries should be evicted
        value = await cache.get("key_0")
        assert value is None
        
        # Last entries should still exist
        value = await cache.get("key_149")
        assert value == "value_149"
    
    async def test_cache_statistics(self, cache):
        """Test cache statistics tracking"""
        # Generate some hits and misses
        await cache.set("exists", "value")
        
        # Hit
        await cache.get("exists")
        
        # Miss
        await cache.get("nonexistent")
        
        stats = cache.get_stats()
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1
        assert stats["total_requests"] >= 2
        assert stats["hit_rate"] > 0
    
    async def test_key_generation(self, cache):
        """Test cache key generation"""
        key1 = cache._generate_key("prefix", "arg1", "arg2", param1="value1")
        key2 = cache._generate_key("prefix", "arg1", "arg2", param1="value1")
        key3 = cache._generate_key("prefix", "arg1", "arg2", param1="value2")
        
        # Same parameters should generate same key
        assert key1 == key2
        
        # Different parameters should generate different key
        assert key1 != key3
    
    async def test_clear_all(self, cache):
        """Test clearing all cache entries"""
        # Add some entries
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        
        assert await cache.get("key1") == "value1"
        assert await cache.get("key2") == "value2"
        
        # Clear all
        await cache.clear_all()
        
        # All should be gone
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
        
        stats = cache.get_stats()
        assert stats["memory_entries"] == 0


class TestSearchCache:
    """Test SearchCache specialized functionality"""
    
    @pytest.fixture
    async def search_cache(self):
        """Create test search cache"""
        base_cache = IntelligentCache({
            "enable_redis": False,
            "enable_memory": True,
            "memory_max_entries": 100
        })
        await base_cache.initialize()
        
        search_cache = SearchCache(base_cache)
        yield search_cache
        await base_cache.close()
    
    @pytest.fixture
    def sample_search_results(self):
        """Sample search results for testing"""
        return [
            SearchResult(
                chunk_id="chunk_1",
                content="Trading strategies for beginners",
                score=0.95,
                metadata={"book": "Trading 101", "page": 5}
            ),
            SearchResult(
                chunk_id="chunk_2",
                content="Advanced trading techniques",
                score=0.87,
                metadata={"book": "Pro Trading", "page": 42}
            )
        ]
    
    async def test_search_result_caching(self, search_cache, sample_search_results):
        """Test caching and retrieval of search results"""
        query = "trading strategies"
        
        # Cache search results
        await search_cache.cache_search_results(
            query, sample_search_results, max_results=10, ttl=1800
        )
        
        # Retrieve cached results
        cached_results = await search_cache.get_search_results(
            query, max_results=10
        )
        
        assert cached_results is not None
        assert len(cached_results) == 2
        assert cached_results[0].chunk_id == "chunk_1"
        assert cached_results[0].content == "Trading strategies for beginners"
    
    async def test_search_cache_key_generation(self, search_cache):
        """Test search cache key generation"""
        key1 = search_cache._search_key("trading", max_results=10, min_score=0.7)
        key2 = search_cache._search_key("trading", max_results=10, min_score=0.7)
        key3 = search_cache._search_key("trading", max_results=20, min_score=0.7)
        
        # Same parameters should generate same key
        assert key1 == key2
        
        # Different parameters should generate different key
        assert key1 != key3
    
    async def test_search_cache_with_filters(self, search_cache, sample_search_results):
        """Test search caching with filters"""
        query = "trading"
        filters = {"book": "Trading 101", "chapter": "Basics"}
        
        # Cache with filters
        await search_cache.cache_search_results(
            query, sample_search_results, filters=filters
        )
        
        # Should retrieve with same filters
        results = await search_cache.get_search_results(query, filters=filters)
        assert results is not None
        assert len(results) == 2
        
        # Should not retrieve with different filters
        results = await search_cache.get_search_results(
            query, filters={"book": "Different Book"}
        )
        assert results is None
    
    async def test_search_cache_invalidation(self, search_cache, sample_search_results):
        """Test search cache invalidation"""
        # Cache some results
        await search_cache.cache_search_results("query1", sample_search_results)
        await search_cache.cache_search_results("query2", sample_search_results)
        
        # Verify cached
        assert await search_cache.get_search_results("query1") is not None
        assert await search_cache.get_search_results("query2") is not None
        
        # Invalidate search cache
        await search_cache.invalidate_search_cache()
        
        # Should be cleared
        assert await search_cache.get_search_results("query1") is None
        assert await search_cache.get_search_results("query2") is None


class TestEmbeddingCache:
    """Test EmbeddingCache specialized functionality"""
    
    @pytest.fixture
    async def embedding_cache(self):
        """Create test embedding cache"""
        base_cache = IntelligentCache({
            "enable_redis": False,
            "enable_memory": True,
            "memory_max_entries": 100
        })
        await base_cache.initialize()
        
        embedding_cache = EmbeddingCache(base_cache)
        yield embedding_cache
        await base_cache.close()
    
    async def test_embedding_caching(self, embedding_cache):
        """Test caching and retrieval of embeddings"""
        text = "This is a test text for embedding"
        embedding = [0.1, 0.2, 0.3] * 128  # 384-dimensional
        model = "test-model"
        
        # Cache embedding
        await embedding_cache.cache_embedding(text, embedding, model)
        
        # Retrieve cached embedding
        cached_embedding = await embedding_cache.get_embedding(text, model)
        
        assert cached_embedding is not None
        assert cached_embedding == embedding
    
    async def test_embedding_cache_key_generation(self, embedding_cache):
        """Test embedding cache key generation"""
        text = "test text"
        
        key1 = embedding_cache._embedding_key(text, "model1")
        key2 = embedding_cache._embedding_key(text, "model1")
        key3 = embedding_cache._embedding_key(text, "model2")
        
        # Same text and model should generate same key
        assert key1 == key2
        
        # Different model should generate different key
        assert key1 != key3
    
    async def test_embedding_model_invalidation(self, embedding_cache):
        """Test invalidation by model"""
        text1 = "text one"
        text2 = "text two"
        embedding = [0.1] * 384
        
        # Cache embeddings for different models
        await embedding_cache.cache_embedding(text1, embedding, "model1")
        await embedding_cache.cache_embedding(text2, embedding, "model1")
        await embedding_cache.cache_embedding(text1, embedding, "model2")
        
        # Verify cached
        assert await embedding_cache.get_embedding(text1, "model1") is not None
        assert await embedding_cache.get_embedding(text2, "model1") is not None
        assert await embedding_cache.get_embedding(text1, "model2") is not None
        
        # Invalidate model1 cache
        await embedding_cache.invalidate_model_cache("model1")
        
        # model1 embeddings should be cleared
        assert await embedding_cache.get_embedding(text1, "model1") is None
        assert await embedding_cache.get_embedding(text2, "model1") is None
        
        # model2 embeddings should remain
        assert await embedding_cache.get_embedding(text1, "model2") is not None


class TestCacheDecorators:
    """Test cache decorators"""
    
    @pytest.fixture
    async def mock_search_function(self):
        """Mock search function for decorator testing"""
        async def mock_search(self, query: str, max_results: int = 10, 
                            min_score: float = 0.0, filters=None):
            # Simulate search results
            return [
                SearchResult(
                    chunk_id="test_chunk",
                    content=f"Result for {query}",
                    score=0.9,
                    metadata={}
                )
            ]
        return mock_search
    
    @pytest.fixture
    async def mock_embedding_function(self):
        """Mock embedding function for decorator testing"""
        async def mock_generate_embeddings(self, texts: List[str], model: str = "default"):
            # Simulate embedding generation
            return [[0.1] * 384 for _ in texts]
        return mock_embedding_function
    
    async def test_search_cache_decorator(self, mock_search_function):
        """Test search cache decorator"""
        # Apply decorator
        cached_search = cache_search_results(ttl=3600)(mock_search_function)
        
        # Mock object with the method
        search_engine = Mock()
        search_engine.cached_search = cached_search.__get__(search_engine, Mock)
        
        # First call should execute function
        with patch('src.core.caching.get_search_cache') as mock_get_cache:
            mock_cache = AsyncMock()
            mock_cache.get_search_results.return_value = None  # Cache miss
            mock_cache.cache_search_results = AsyncMock()
            mock_get_cache.return_value = mock_cache
            
            results1 = await search_engine.cached_search(query="test")
            
            # Should have called cache methods
            mock_cache.get_search_results.assert_called_once()
            mock_cache.cache_search_results.assert_called_once()
    
    async def test_embedding_cache_decorator(self, mock_embedding_function):
        """Test embedding cache decorator"""
        # Apply decorator
        cached_embeddings = cache_embeddings(ttl=86400)(mock_embedding_function)
        
        # Mock object with the method
        generator = Mock()
        generator.cached_generate = cached_embeddings.__get__(generator, Mock)
        
        # Test with single text (should use cache)
        with patch('src.core.caching.get_embedding_cache') as mock_get_cache:
            mock_cache = AsyncMock()
            mock_cache.get_embedding.return_value = None  # Cache miss
            mock_cache.cache_embedding = AsyncMock()
            mock_get_cache.return_value = mock_cache
            
            results = await generator.cached_generate(texts=["test text"])
            
            # Should have called cache methods
            mock_cache.get_embedding.assert_called_once()
            mock_cache.cache_embedding.assert_called_once()


class TestConcurrency:
    """Test cache performance under concurrent access"""
    
    @pytest.fixture
    async def concurrent_cache(self):
        """Create cache for concurrency testing"""
        cache = IntelligentCache({
            "enable_redis": False,
            "enable_memory": True,
            "memory_max_entries": 1000
        })
        await cache.initialize()
        yield cache
        await cache.close()
    
    @pytest.mark.performance
    async def test_concurrent_read_write(self, concurrent_cache):
        """Test concurrent read/write operations"""
        async def writer(cache, prefix, count):
            for i in range(count):
                await cache.set(f"{prefix}_{i}", f"value_{i}")
        
        async def reader(cache, prefix, count):
            results = []
            for i in range(count):
                value = await cache.get(f"{prefix}_{i}")
                results.append(value)
            return results
        
        # Run concurrent writers and readers
        tasks = []
        for i in range(5):
            tasks.append(writer(concurrent_cache, f"writer_{i}", 20))
            tasks.append(reader(concurrent_cache, f"writer_{i}", 20))
        
        await asyncio.gather(*tasks)
        
        # Verify cache is in consistent state
        stats = concurrent_cache.get_stats()
        assert stats["memory_entries"] > 0
        assert stats["total_requests"] > 0
    
    @pytest.mark.performance
    async def test_cache_performance(self, concurrent_cache):
        """Test cache performance characteristics"""
        import time
        
        # Measure write performance
        start_time = time.time()
        for i in range(1000):
            await concurrent_cache.set(f"perf_key_{i}", f"value_{i}")
        write_time = time.time() - start_time
        
        # Measure read performance (cache hits)
        start_time = time.time()
        for i in range(1000):
            await concurrent_cache.get(f"perf_key_{i}")
        read_time = time.time() - start_time
        
        # Performance assertions
        assert write_time < 5.0  # Should complete writes in under 5 seconds
        assert read_time < 1.0   # Should complete reads in under 1 second
        
        stats = concurrent_cache.get_stats()
        assert stats["hit_rate"] > 0.9  # Should have high hit rate


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.fixture
    async def error_cache(self):
        """Create cache for error testing"""
        cache = IntelligentCache({
            "enable_redis": False,
            "enable_memory": True
        })
        await cache.initialize()
        yield cache
        await cache.close()
    
    async def test_serialization_errors(self, error_cache):
        """Test handling of non-serializable values"""
        # Try to cache non-serializable object
        class NonSerializable:
            def __init__(self):
                self.func = lambda x: x
        
        obj = NonSerializable()
        
        # Should handle serialization error gracefully
        try:
            await error_cache.set("bad_obj", obj)
        except Exception:
            pass  # Expected to fail
        
        # Cache should still be functional
        await error_cache.set("good_key", "good_value")
        assert await error_cache.get("good_key") == "good_value"
    
    async def test_large_value_handling(self, error_cache):
        """Test handling of very large values"""
        # Create very large value
        large_value = "x" * (10 * 1024 * 1024)  # 10MB
        
        # Should handle large values appropriately
        await error_cache.set("large_key", large_value)
        
        # May or may not be stored depending on limits
        # but cache should remain functional
        await error_cache.set("normal_key", "normal_value")
        assert await error_cache.get("normal_key") == "normal_value"
    
    async def test_invalid_ttl_handling(self, error_cache):
        """Test handling of invalid TTL values"""
        # Should handle negative TTL gracefully
        await error_cache.set("key1", "value1", ttl=-1)
        
        # Should handle very large TTL gracefully
        await error_cache.set("key2", "value2", ttl=999999999)
        
        # Cache should remain functional
        value = await error_cache.get("key2")
        assert value == "value2"


# Global cache instance tests
class TestGlobalInstances:
    """Test global cache instance management"""
    
    async def test_get_cache_singleton(self):
        """Test global cache singleton behavior"""
        cache1 = await get_cache()
        cache2 = await get_cache()
        
        # Should return same instance
        assert cache1 is cache2
    
    async def test_get_search_cache_singleton(self):
        """Test global search cache singleton"""
        search_cache1 = await get_search_cache()
        search_cache2 = await get_search_cache()
        
        # Should return same instance
        assert search_cache1 is search_cache2
    
    async def test_get_embedding_cache_singleton(self):
        """Test global embedding cache singleton"""
        embedding_cache1 = await get_embedding_cache()
        embedding_cache2 = await get_embedding_cache()
        
        # Should return same instance
        assert embedding_cache1 is embedding_cache2
    
    async def test_cleanup_cache_function(self):
        """Test cache cleanup function"""
        # Initialize caches
        await get_cache()
        await get_search_cache()
        await get_embedding_cache()
        
        # Cleanup
        await cleanup_cache()
        
        # Should reset global instances
        # Next calls should create new instances
        new_cache = await get_cache()
        assert new_cache is not None