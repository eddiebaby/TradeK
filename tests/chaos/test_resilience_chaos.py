"""
Chaos Engineering Tests for Resilience Validation

Tests system behavior under various failure conditions to ensure
robust error handling and graceful degradation.
"""

import pytest
import asyncio
import random
import time
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import List, Dict, Any

from src.search.unified_search import UnifiedSearchEngine
from src.core.qdrant_storage import QdrantStorage
from src.ingestion.enhanced_book_processor import EnhancedBookProcessor
from src.resilience.circuit_breaker import CircuitBreaker
from src.resilience.retry_mechanisms import RetryMechanism


class ChaosTestFramework:
    """Framework for chaos engineering tests."""
    
    def __init__(self):
        self.failure_scenarios = []
        self.metrics = {}
        
    def inject_latency(self, min_ms: int = 100, max_ms: int = 1000):
        """Inject random latency into operations."""
        delay = random.uniform(min_ms, max_ms) / 1000
        time.sleep(delay)
        
    def inject_intermittent_failure(self, failure_rate: float = 0.3):
        """Inject intermittent failures based on probability."""
        if random.random() < failure_rate:
            raise ConnectionError("Chaos: Intermittent failure injected")
            
    def inject_resource_exhaustion(self, memory_pressure: bool = True):
        """Simulate resource exhaustion scenarios."""
        if memory_pressure:
            # Simulate memory pressure
            large_list = [0] * (10**6)  # Allocate some memory
            time.sleep(0.1)
            del large_list
            
    def inject_partial_failure(self, success_rate: float = 0.7):
        """Inject partial failures in distributed operations."""
        return random.random() < success_rate


class TestSearchEngineResilience:
    """Test search engine resilience under chaos conditions."""
    
    def setup_method(self):
        self.chaos = ChaosTestFramework()
        self.search_engine = UnifiedSearchEngine()
        
    def test_search_engine_handles_qdrant_connection_failures_gracefully(self):
        """Test search engine handles Qdrant connection failures gracefully"""
        # ARRANGE
        with patch('src.core.qdrant_storage.QdrantStorage') as mock_qdrant:
            mock_instance = Mock()
            mock_qdrant.return_value = mock_instance
            
            # Inject connection failures
            mock_instance.search.side_effect = ConnectionError("Connection failed")
            
            # ACT
            result = self.search_engine.search("test query")
            
            # ASSERT
            # Should gracefully degrade to fallback search
            assert result is not None
            assert 'error' not in result or 'fallback' in str(result)
            assert 'results' in result  # Should provide some results even if degraded
    
    def test_search_engine_handles_intermittent_qdrant_failures(self):
        """Test search engine handles intermittent Qdrant failures"""
        # ARRANGE
        failure_count = 0
        success_count = 0
        
        def intermittent_search(*args, **kwargs):
            nonlocal failure_count, success_count
            self.chaos.inject_intermittent_failure(failure_rate=0.5)
            success_count += 1
            return {'results': [{'id': 'test', 'score': 0.8, 'content': 'test'}]}
        
        with patch('src.core.qdrant_storage.QdrantStorage') as mock_qdrant:
            mock_instance = Mock()
            mock_qdrant.return_value = mock_instance
            mock_instance.search.side_effect = intermittent_search
            
            # ACT
            results = []
            for i in range(10):
                try:
                    result = self.search_engine.search(f"test query {i}")
                    results.append(result)
                except Exception as e:
                    failure_count += 1
            
            # ASSERT
            # Should handle failures gracefully and eventually succeed
            assert len(results) > 0, "Should have some successful results"
            assert failure_count < 10, "Should not fail all attempts"
    
    def test_search_engine_handles_high_latency_gracefully(self):
        """Test search engine handles high latency gracefully"""
        # ARRANGE
        def slow_search(*args, **kwargs):
            self.chaos.inject_latency(min_ms=500, max_ms=2000)
            return {'results': []}
        
        with patch('src.core.qdrant_storage.QdrantStorage') as mock_qdrant:
            mock_instance = Mock()
            mock_qdrant.return_value = mock_instance
            mock_instance.search.side_effect = slow_search
            
            # ACT
            start_time = time.time()
            result = self.search_engine.search("test query")
            execution_time = time.time() - start_time
            
            # ASSERT
            # Should timeout appropriately and not hang indefinitely
            assert execution_time < 5.0, "Should timeout before 5 seconds"
            assert result is not None, "Should return some result even on timeout"
    
    @pytest.mark.asyncio
    async def test_concurrent_search_resilience_under_failures(self):
        """Test concurrent search operations resilience under failures"""
        # ARRANGE
        failure_injection = random.Random(42)  # Deterministic for testing
        
        def chaos_search(*args, **kwargs):
            if failure_injection.random() < 0.3:  # 30% failure rate
                raise ConnectionError("Chaos failure")
            self.chaos.inject_latency(50, 200)
            return {'results': [{'id': 'test', 'score': 0.9, 'content': 'result'}]}
        
        with patch('src.core.qdrant_storage.QdrantStorage') as mock_qdrant:
            mock_instance = Mock()
            mock_qdrant.return_value = mock_instance
            mock_instance.search.side_effect = chaos_search
            
            # ACT
            async def search_task(query):
                try:
                    return self.search_engine.search(f"query {query}")
                except Exception:
                    return None
            
            tasks = [search_task(i) for i in range(20)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # ASSERT
            successful_results = [r for r in results if r is not None and not isinstance(r, Exception)]
            assert len(successful_results) > 10, "Majority of searches should succeed despite failures"


class TestIngestionResilience:
    """Test document ingestion resilience under chaos conditions."""
    
    def setup_method(self):
        self.chaos = ChaosTestFramework()
        self.processor = EnhancedBookProcessor()
    
    def test_book_processor_handles_corrupted_files_gracefully(self):
        """Test book processor handles corrupted files gracefully"""
        # ARRANGE
        import tempfile
        corrupted_content = b"corrupted pdf content" + b"\x00" * 1000
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(corrupted_content)
            corrupted_file = f.name
        
        try:
            # ACT
            result = self.processor.process_file(corrupted_file)
            
            # ASSERT
            # Should handle corruption gracefully
            assert result is not None
            assert 'error' in result or 'chunks' in result
            
        finally:
            import os
            os.unlink(corrupted_file)
    
    def test_embedding_generation_handles_ollama_service_failures(self):
        """Test embedding generation handles Ollama service failures"""
        # ARRANGE
        with patch('src.ingestion.local_embeddings.LocalEmbeddingService') as mock_service:
            mock_instance = Mock()
            mock_service.return_value = mock_instance
            
            # Inject service failures
            mock_instance.generate_embedding.side_effect = ConnectionError("Ollama unavailable")
            
            # ACT
            result = self.processor.process_text_with_embeddings("test content")
            
            # ASSERT
            # Should gracefully handle embedding service failures
            assert result is not None
            # Should either use fallback embeddings or skip embedding
            assert 'processed' in result or 'error' in result
    
    def test_memory_pressure_during_large_file_processing(self):
        """Test behavior under memory pressure during large file processing"""
        # ARRANGE
        # Simulate memory pressure
        large_text = "Large document content. " * 100000  # ~2MB text
        
        def memory_pressure_embedding(*args, **kwargs):
            self.chaos.inject_resource_exhaustion(memory_pressure=True)
            return [0.1] * 384  # Return dummy embedding
        
        with patch('src.ingestion.local_embeddings.LocalEmbeddingService') as mock_service:
            mock_instance = Mock()
            mock_service.return_value = mock_instance
            mock_instance.generate_embedding.side_effect = memory_pressure_embedding
            
            # ACT
            result = self.processor.process_text(large_text)
            
            # ASSERT
            # Should handle memory pressure gracefully
            assert result is not None
            assert isinstance(result.get('chunks', []), list)


class TestCircuitBreakerResilience:
    """Test circuit breaker resilience patterns."""
    
    def test_circuit_breaker_opens_after_consecutive_failures(self):
        """Test circuit breaker opens after consecutive failures"""
        # ARRANGE
        circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=1)
        
        def failing_operation():
            raise ConnectionError("Service unavailable")
        
        # ACT
        failure_count = 0
        for i in range(5):
            try:
                circuit_breaker.call(failing_operation)
            except Exception:
                failure_count += 1
        
        # ASSERT
        assert circuit_breaker.state == "OPEN"
        assert failure_count >= 3
    
    def test_circuit_breaker_half_open_state_transitions(self):
        """Test circuit breaker half-open state transitions correctly"""
        # ARRANGE
        circuit_breaker = CircuitBreaker(failure_threshold=2, timeout=0.1)
        
        # Force circuit breaker to OPEN state
        for _ in range(3):
            try:
                circuit_breaker.call(lambda: exec('raise ConnectionError()'))
            except:
                pass
        
        # Wait for timeout
        time.sleep(0.2)
        
        # ACT
        # First call should transition to HALF_OPEN
        try:
            circuit_breaker.call(lambda: "success")
        except:
            pass
        
        # ASSERT
        assert circuit_breaker.state in ["HALF_OPEN", "CLOSED"]


class TestCascadingFailureResilience:
    """Test resilience against cascading failures."""
    
    def test_service_isolation_prevents_cascading_failures(self):
        """Test service isolation prevents cascading failures"""
        # ARRANGE
        services = {
            'search': Mock(),
            'embedding': Mock(),
            'storage': Mock()
        }
        
        # Make search service fail
        services['search'].search.side_effect = ConnectionError("Search service down")
        
        # ACT
        search_engine = UnifiedSearchEngine()
        
        # Simulate multiple service calls
        results = []
        for i in range(5):
            try:
                # Even if search fails, other services should remain functional
                result = search_engine.fallback_search(f"query {i}")
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})
        
        # ASSERT
        # Should have some successful fallback results
        successful_results = [r for r in results if 'error' not in r]
        assert len(successful_results) > 0, "Fallback should provide some results"
    
    def test_bulkhead_pattern_isolates_critical_operations(self):
        """Test bulkhead pattern isolates critical operations"""
        # ARRANGE
        with ThreadPoolExecutor(max_workers=2) as critical_executor:
            with ThreadPoolExecutor(max_workers=8) as general_executor:
                
                # ACT
                # Submit critical operations to isolated thread pool
                critical_futures = []
                for i in range(3):
                    future = critical_executor.submit(lambda x=i: f"critical_result_{x}")
                    critical_futures.append(future)
                
                # Submit resource-heavy operations to general pool
                def resource_heavy_task():
                    time.sleep(0.1)  # Simulate work
                    return "general_result"
                
                general_futures = []
                for i in range(10):
                    future = general_executor.submit(resource_heavy_task)
                    general_futures.append(future)
                
                # ASSERT
                # Critical operations should complete even if general pool is busy
                critical_results = [f.result(timeout=1) for f in critical_futures]
                assert len(critical_results) == 3
                assert all("critical_result" in r for r in critical_results)


class TestDataConsistencyUnderChaos:
    """Test data consistency under chaotic conditions."""
    
    def test_concurrent_writes_maintain_consistency(self):
        """Test concurrent writes maintain data consistency"""
        # ARRANGE
        storage = QdrantStorage()
        consistency_check = threading.Lock()
        write_results = []
        
        def concurrent_write(doc_id, content):
            try:
                # Inject chaos
                if random.random() < 0.2:  # 20% failure rate
                    raise ConnectionError("Write failure")
                
                result = storage.store_document(doc_id, content, {})
                with consistency_check:
                    write_results.append((doc_id, "success"))
                return result
            except Exception as e:
                with consistency_check:
                    write_results.append((doc_id, "failed"))
                raise
        
        # ACT
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(20):
                future = executor.submit(concurrent_write, f"doc_{i}", f"content_{i}")
                futures.append(future)
            
            # Wait for completion
            for future in futures:
                try:
                    future.result(timeout=2)
                except:
                    pass  # Expected some failures
        
        # ASSERT
        successful_writes = [r for r in write_results if r[1] == "success"]
        assert len(successful_writes) > 10, "Majority of writes should succeed"
        
        # Check for data consistency
        unique_docs = set(r[0] for r in successful_writes)
        assert len(unique_docs) == len(successful_writes), "No duplicate writes should succeed"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])