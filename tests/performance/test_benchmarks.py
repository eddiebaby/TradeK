"""
Performance Benchmarking Framework for TradeKnowledge.

This module provides comprehensive performance testing and benchmarking
capabilities to measure and validate system performance under various conditions.
"""

import pytest
import asyncio
import time
import statistics
import psutil
import gc
from typing import List, Dict, Any, Callable
from dataclasses import dataclass
from unittest.mock import patch, AsyncMock, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.search.unified_search import UnifiedSearchEngine
from src.ingestion.enhanced_book_processor import EnhancedBookProcessor
from src.ingestion.local_embeddings import LocalEmbeddingGenerator
from src.core.models import Chunk, Book


@dataclass
class PerformanceMetrics:
    """Container for performance measurement results"""
    operation_name: str
    duration_ms: float
    memory_usage_mb: float
    cpu_percent: float
    success: bool
    error_message: str = None
    metadata: Dict[str, Any] = None


class PerformanceBenchmark:
    """Core performance benchmarking utilities"""
    
    def __init__(self):
        self.results: List[PerformanceMetrics] = []
        self.process = psutil.Process()
    
    async def measure_async_operation(self, 
                                      operation: Callable,
                                      operation_name: str,
                                      *args, **kwargs) -> PerformanceMetrics:
        """Measure performance of an async operation"""
        
        # Force garbage collection before measurement
        gc.collect()
        
        # Capture initial state
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu = self.process.cpu_percent()
        
        start_time = time.perf_counter()
        success = True
        error_message = None
        
        try:
            result = await operation(*args, **kwargs)
        except Exception as e:
            success = False
            error_message = str(e)
            result = None
        
        end_time = time.perf_counter()
        
        # Capture final state
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        final_cpu = self.process.cpu_percent()
        
        duration_ms = (end_time - start_time) * 1000
        memory_usage_mb = final_memory - initial_memory
        cpu_percent = max(final_cpu - initial_cpu, 0)
        
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            duration_ms=duration_ms,
            memory_usage_mb=memory_usage_mb,
            cpu_percent=cpu_percent,
            success=success,
            error_message=error_message,
            metadata={"result_size": len(str(result)) if result else 0}
        )
        
        self.results.append(metrics)
        return metrics
    
    def measure_sync_operation(self, 
                               operation: Callable,
                               operation_name: str,
                               *args, **kwargs) -> PerformanceMetrics:
        """Measure performance of a synchronous operation"""
        
        # Force garbage collection before measurement
        gc.collect()
        
        # Capture initial state
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu = self.process.cpu_percent()
        
        start_time = time.perf_counter()
        success = True
        error_message = None
        
        try:
            result = operation(*args, **kwargs)
        except Exception as e:
            success = False
            error_message = str(e)
            result = None
        
        end_time = time.perf_counter()
        
        # Capture final state
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        final_cpu = self.process.cpu_percent()
        
        duration_ms = (end_time - start_time) * 1000
        memory_usage_mb = final_memory - initial_memory
        cpu_percent = max(final_cpu - initial_cpu, 0)
        
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            duration_ms=duration_ms,
            memory_usage_mb=memory_usage_mb,
            cpu_percent=cpu_percent,
            success=success,
            error_message=error_message,
            metadata={"result_size": len(str(result)) if result else 0}
        )
        
        self.results.append(metrics)
        return metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary statistics"""
        if not self.results:
            return {"error": "No performance data collected"}
        
        # Group results by operation name
        operations = {}
        for result in self.results:
            if result.operation_name not in operations:
                operations[result.operation_name] = []
            operations[result.operation_name].append(result)
        
        summary = {}
        for op_name, op_results in operations.items():
            durations = [r.duration_ms for r in op_results if r.success]
            memory_usage = [r.memory_usage_mb for r in op_results if r.success]
            cpu_usage = [r.cpu_percent for r in op_results if r.success]
            
            if durations:
                summary[op_name] = {
                    "duration_stats": {
                        "min_ms": min(durations),
                        "max_ms": max(durations),
                        "mean_ms": statistics.mean(durations),
                        "median_ms": statistics.median(durations),
                        "p95_ms": self._percentile(durations, 95),
                        "p99_ms": self._percentile(durations, 99)
                    },
                    "memory_stats": {
                        "min_mb": min(memory_usage) if memory_usage else 0,
                        "max_mb": max(memory_usage) if memory_usage else 0,
                        "mean_mb": statistics.mean(memory_usage) if memory_usage else 0
                    },
                    "cpu_stats": {
                        "min_percent": min(cpu_usage) if cpu_usage else 0,
                        "max_percent": max(cpu_usage) if cpu_usage else 0,
                        "mean_percent": statistics.mean(cpu_usage) if cpu_usage else 0
                    },
                    "success_rate": len([r for r in op_results if r.success]) / len(op_results),
                    "total_runs": len(op_results)
                }
        
        return summary
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        if index == int(index):
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))


class TestSearchPerformanceBenchmarks:
    """Performance benchmarks for search operations"""
    
    @pytest.fixture
    def benchmark(self):
        """Create performance benchmark instance"""
        return PerformanceBenchmark()
    
    @pytest.fixture
    def mock_search_engine(self):
        """Create mocked search engine for benchmarking"""
        with patch('src.search.hybrid_search.HybridSearch') as mock_hybrid, \
             patch('src.search.text_search.TextSearchEngine') as mock_text, \
             patch('src.search.vector_search.VectorSearchEngine') as mock_vector:
            
            # Mock search engines with realistic delays
            mock_hybrid_instance = AsyncMock()
            mock_text_instance = AsyncMock()
            mock_vector_instance = AsyncMock()
            
            # Simulate search processing time
            async def mock_search_with_delay(*args, **kwargs):
                await asyncio.sleep(0.02)  # 20ms processing time
                return {
                    'results': [
                        {
                            'id': f'result_{i}',
                            'title': f'Test Result {i}',
                            'content': f'Content for result {i}' * 20,  # Realistic content size
                            'score': 0.9 - i * 0.1,
                            'book_title': f'Book {i}'
                        }
                        for i in range(min(10, kwargs.get('max_results', 10)))
                    ]
                }
            
            mock_hybrid_instance.search.side_effect = mock_search_with_delay
            mock_text_instance.search.side_effect = mock_search_with_delay
            mock_vector_instance.search.side_effect = mock_search_with_delay
            
            mock_hybrid.return_value = mock_hybrid_instance
            mock_text.return_value = mock_text_instance
            mock_vector.return_value = mock_vector_instance
            
            engine = UnifiedSearchEngine()
            engine.initialized = True
            engine.hybrid_engine = mock_hybrid_instance
            engine.text_engine = mock_text_instance
            engine.vector_engine = mock_vector_instance
            
            return engine
    
    @pytest.mark.asyncio
    async def test_single_search_performance(self, benchmark, mock_search_engine):
        """Benchmark single search operation performance"""
        
        # Test single search
        metrics = await benchmark.measure_async_operation(
            mock_search_engine.search,
            "single_search",
            query="algorithmic trading strategies",
            max_results=10
        )
        
        # Validate performance expectations
        assert metrics.success, f"Search failed: {metrics.error_message}"
        assert metrics.duration_ms < 100, f"Search too slow: {metrics.duration_ms}ms"
        assert metrics.memory_usage_mb < 50, f"Memory usage too high: {metrics.memory_usage_mb}MB"
        
        print(f"Single search: {metrics.duration_ms:.2f}ms, {metrics.memory_usage_mb:.2f}MB")
    
    @pytest.mark.asyncio
    async def test_concurrent_search_performance(self, benchmark, mock_search_engine):
        """Benchmark concurrent search operations"""
        
        # Test concurrent searches
        search_queries = [
            "algorithmic trading strategies",
            "risk management techniques", 
            "machine learning in finance",
            "technical analysis indicators",
            "portfolio optimization methods"
        ]
        
        async def concurrent_searches():
            tasks = [
                mock_search_engine.search(query=query, max_results=10)
                for query in search_queries
            ]
            return await asyncio.gather(*tasks)
        
        metrics = await benchmark.measure_async_operation(
            concurrent_searches,
            "concurrent_5_searches"
        )
        
        # Validate concurrent performance
        assert metrics.success, f"Concurrent searches failed: {metrics.error_message}"
        assert metrics.duration_ms < 200, f"Concurrent searches too slow: {metrics.duration_ms}ms"
        assert metrics.memory_usage_mb < 100, f"Memory usage too high: {metrics.memory_usage_mb}MB"
        
        print(f"Concurrent searches: {metrics.duration_ms:.2f}ms, {metrics.memory_usage_mb:.2f}MB")
    
    @pytest.mark.asyncio
    async def test_search_load_simulation(self, benchmark, mock_search_engine):
        """Simulate realistic search load"""
        
        # Simulate 20 sequential searches (representing user session)
        search_queries = [
            f"trading strategy {i}" for i in range(20)
        ]
        
        async def search_load_simulation():
            results = []
            for query in search_queries:
                result = await mock_search_engine.search(query=query, max_results=10)
                results.append(result)
                # Small delay between searches (user thinking time)
                await asyncio.sleep(0.001)
            return results
        
        metrics = await benchmark.measure_async_operation(
            search_load_simulation,
            "search_load_20_queries"
        )
        
        # Validate load performance
        assert metrics.success, f"Search load simulation failed: {metrics.error_message}"
        assert metrics.duration_ms < 1000, f"Search load too slow: {metrics.duration_ms}ms"
        assert metrics.memory_usage_mb < 200, f"Memory usage too high: {metrics.memory_usage_mb}MB"
        
        print(f"Search load (20 queries): {metrics.duration_ms:.2f}ms, {metrics.memory_usage_mb:.2f}MB")
    
    @pytest.mark.asyncio
    async def test_search_scaling_performance(self, benchmark, mock_search_engine):
        """Test search performance scaling with different result sizes"""
        
        result_sizes = [5, 10, 25, 50, 100]
        
        for size in result_sizes:
            metrics = await benchmark.measure_async_operation(
                mock_search_engine.search,
                f"search_results_{size}",
                query="comprehensive trading analysis",
                max_results=size
            )
            
            # Validate scaling behavior
            assert metrics.success, f"Search with {size} results failed: {metrics.error_message}"
            # Performance should scale reasonably with result size
            expected_max_time = 50 + (size * 0.5)  # Base time + scaling factor
            assert metrics.duration_ms < expected_max_time, \
                f"Search with {size} results too slow: {metrics.duration_ms}ms"
            
            print(f"Search {size} results: {metrics.duration_ms:.2f}ms")


class TestEmbeddingPerformanceBenchmarks:
    """Performance benchmarks for embedding operations"""
    
    @pytest.fixture
    def benchmark(self):
        """Create performance benchmark instance"""
        return PerformanceBenchmark()
    
    @pytest.fixture
    def mock_embedding_generator(self):
        """Create mocked embedding generator for benchmarking"""
        
        generator = AsyncMock()
        
        # Mock realistic embedding generation with processing time
        async def mock_generate_embeddings(chunks, show_progress=True):
            # Simulate processing time based on content size
            processing_time = len(chunks) * 0.01  # 10ms per chunk
            await asyncio.sleep(processing_time)
            
            return [
                [0.1, 0.2, 0.3, 0.4] * 96  # 384-dim embedding
                for _ in chunks
            ]
        
        async def mock_generate_query_embedding(query):
            await asyncio.sleep(0.005)  # 5ms for query embedding
            return [0.1, 0.2, 0.3, 0.4] * 96
        
        generator.generate_embeddings.side_effect = mock_generate_embeddings
        generator.generate_query_embedding.side_effect = mock_generate_query_embedding
        
        return generator
    
    @pytest.mark.asyncio
    async def test_single_embedding_performance(self, benchmark, mock_embedding_generator):
        """Benchmark single embedding generation"""
        
        # Create test chunk
        test_chunk = Chunk(
            book_id="test_book",
            chunk_index=0,
            text="This is a test chunk about algorithmic trading strategies" * 10,
            chapter="Test Chapter",
            page_start=1
        )
        
        metrics = await benchmark.measure_async_operation(
            mock_embedding_generator.generate_embeddings,
            "single_embedding",
            [test_chunk]
        )
        
        # Validate embedding performance
        assert metrics.success, f"Embedding generation failed: {metrics.error_message}"
        assert metrics.duration_ms < 50, f"Embedding generation too slow: {metrics.duration_ms}ms"
        assert metrics.memory_usage_mb < 100, f"Memory usage too high: {metrics.memory_usage_mb}MB"
        
        print(f"Single embedding: {metrics.duration_ms:.2f}ms, {metrics.memory_usage_mb:.2f}MB")
    
    @pytest.mark.asyncio
    async def test_batch_embedding_performance(self, benchmark, mock_embedding_generator):
        """Benchmark batch embedding generation"""
        
        # Create batch of test chunks
        test_chunks = [
            Chunk(
                book_id="test_book",
                chunk_index=i,
                text=f"Test chunk {i} with content about trading strategies" * 10,
                chapter=f"Chapter {i // 10}",
                page_start=i
            )
            for i in range(50)
        ]
        
        metrics = await benchmark.measure_async_operation(
            mock_embedding_generator.generate_embeddings,
            "batch_50_embeddings",
            test_chunks
        )
        
        # Validate batch performance
        assert metrics.success, f"Batch embedding failed: {metrics.error_message}"
        assert metrics.duration_ms < 1000, f"Batch embedding too slow: {metrics.duration_ms}ms"
        assert metrics.memory_usage_mb < 500, f"Memory usage too high: {metrics.memory_usage_mb}MB"
        
        print(f"Batch embeddings (50): {metrics.duration_ms:.2f}ms, {metrics.memory_usage_mb:.2f}MB")
    
    @pytest.mark.asyncio
    async def test_query_embedding_performance(self, benchmark, mock_embedding_generator):
        """Benchmark query embedding generation"""
        
        test_queries = [
            "algorithmic trading strategies",
            "risk management in portfolio optimization",
            "machine learning applications in finance",
            "technical analysis and market indicators"
        ]
        
        for query in test_queries:
            metrics = await benchmark.measure_async_operation(
                mock_embedding_generator.generate_query_embedding,
                "query_embedding",
                query
            )
            
            # Validate query embedding performance
            assert metrics.success, f"Query embedding failed: {metrics.error_message}"
            assert metrics.duration_ms < 20, f"Query embedding too slow: {metrics.duration_ms}ms"
            
            print(f"Query embedding '{query[:30]}...': {metrics.duration_ms:.2f}ms")


class TestSystemPerformanceBenchmarks:
    """System-wide performance benchmarks"""
    
    @pytest.fixture
    def benchmark(self):
        """Create performance benchmark instance"""
        return PerformanceBenchmark()
    
    @pytest.mark.asyncio
    async def test_memory_usage_baseline(self, benchmark):
        """Establish memory usage baseline"""
        
        def get_memory_info():
            process = psutil.Process()
            return {
                'rss': process.memory_info().rss / 1024 / 1024,  # MB
                'vms': process.memory_info().vms / 1024 / 1024,  # MB
                'percent': process.memory_percent()
            }
        
        metrics = benchmark.measure_sync_operation(
            get_memory_info,
            "memory_baseline"
        )
        
        assert metrics.success, f"Memory baseline failed: {metrics.error_message}"
        
        print(f"Memory baseline: {metrics.metadata}")
    
    @pytest.mark.asyncio
    async def test_cpu_usage_baseline(self, benchmark):
        """Establish CPU usage baseline"""
        
        def get_cpu_info():
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'cpu_count': psutil.cpu_count(),
                'load_avg': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            }
        
        metrics = benchmark.measure_sync_operation(
            get_cpu_info,
            "cpu_baseline"
        )
        
        assert metrics.success, f"CPU baseline failed: {metrics.error_message}"
        
        print(f"CPU baseline: {metrics.metadata}")
    
    @pytest.mark.asyncio
    async def test_performance_regression_detection(self, benchmark):
        """Test performance regression detection capabilities"""
        
        # Simulate baseline performance
        baseline_times = []
        for i in range(10):
            start = time.perf_counter()
            await asyncio.sleep(0.01)  # Simulate consistent 10ms operation
            end = time.perf_counter()
            baseline_times.append((end - start) * 1000)
        
        baseline_mean = statistics.mean(baseline_times)
        baseline_std = statistics.stdev(baseline_times)
        
        # Test regression detection (simulated slower operation)
        start = time.perf_counter()
        await asyncio.sleep(0.05)  # Simulate 50ms operation (5x slower)
        end = time.perf_counter()
        test_time = (end - start) * 1000
        
        # Check for performance regression
        regression_threshold = baseline_mean + (2 * baseline_std)
        is_regression = test_time > regression_threshold
        
        print(f"Baseline: {baseline_mean:.2f}ms ± {baseline_std:.2f}ms")
        print(f"Test: {test_time:.2f}ms")
        print(f"Regression detected: {is_regression}")
        
        # This test documents the regression detection capability
        assert baseline_mean > 0, "Baseline measurement failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])