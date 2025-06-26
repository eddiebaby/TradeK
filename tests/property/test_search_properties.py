"""
Property-Based Testing for Search Algorithms

Uses Hypothesis to generate test data and validate algorithmic properties
that should always hold true regardless of input.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant
import numpy as np
from typing import List, Dict, Any

from src.search.unified_search import UnifiedSearchEngine
from src.search.vector_search import VectorSearchEngine
from src.search.text_search import TextSearchEngine
from src.ingestion.text_chunker import TextChunker


class TestSearchAlgorithmProperties:
    """Property-based tests for search algorithms."""
    
    @given(st.text(min_size=1, max_size=1000))
    def test_search_results_deterministic_property(self, query: str):
        """Property: Same query should always return same results"""
        # ARRANGE
        assume(query.strip())  # Non-empty query
        search_engine = UnifiedSearchEngine()
        
        # ACT
        result1 = search_engine.search(query)
        result2 = search_engine.search(query)
        
        # ASSERT
        assert result1 == result2, "Search results should be deterministic"
    
    @given(st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=50))
    def test_search_relevance_ordering_property(self, queries: List[str]):
        """Property: More specific queries should return more relevant results"""
        # ARRANGE
        search_engine = UnifiedSearchEngine()
        
        for query in queries:
            assume(query.strip())
            
            # ACT
            results = search_engine.search(query)
            
            # ASSERT - Relevance scores should be in descending order
            if 'results' in results and len(results['results']) > 1:
                scores = [r.get('score', 0) for r in results['results']]
                assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1)), \
                    "Results should be ordered by relevance score"
    
    @given(st.lists(st.floats(min_value=-1.0, max_value=1.0), min_size=384, max_size=384))
    def test_vector_similarity_properties(self, vector: List[float]):
        """Property: Vector similarity should satisfy mathematical properties"""
        # ARRANGE
        assume(all(not np.isnan(v) and not np.isinf(v) for v in vector))
        vector_search = VectorSearchEngine()
        
        # ACT
        similarity_self = vector_search.cosine_similarity(vector, vector)
        
        # ASSERT
        # Self-similarity should be 1.0 (or very close due to floating point)
        assert abs(similarity_self - 1.0) < 1e-6, "Vector should have perfect self-similarity"
        
        # Similarity should be symmetric
        other_vector = [v * 0.5 for v in vector]  # Create related vector
        sim_ab = vector_search.cosine_similarity(vector, other_vector)
        sim_ba = vector_search.cosine_similarity(other_vector, vector)
        assert abs(sim_ab - sim_ba) < 1e-6, "Similarity should be symmetric"
    
    @given(st.text(min_size=10, max_size=10000))
    def test_text_chunking_properties(self, text: str):
        """Property: Text chunking should preserve all content"""
        # ARRANGE
        assume(text.strip())
        chunker = TextChunker(chunk_size=500, overlap=50)
        
        # ACT
        chunks = chunker.chunk_text(text)
        
        # ASSERT
        # All chunks should be non-empty
        assert all(chunk.strip() for chunk in chunks), "All chunks should be non-empty"
        
        # Total length should be preserved (accounting for overlaps)
        total_chars = sum(len(chunk) for chunk in chunks)
        assert total_chars >= len(text), "No text should be lost in chunking"
        
        # Each chunk should respect size limits
        for chunk in chunks:
            assert len(chunk) <= chunker.chunk_size + 100, "Chunks should respect size limits"
    
    @given(st.dictionaries(
        st.text(min_size=1, max_size=20),
        st.one_of(st.text(), st.integers(), st.floats(), st.booleans()),
        min_size=1, max_size=10
    ))
    def test_metadata_handling_properties(self, metadata: Dict[str, Any]):
        """Property: Metadata should be preserved through search operations"""
        # ARRANGE
        search_engine = UnifiedSearchEngine()
        
        # Filter out invalid values for JSON serialization
        clean_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                continue
            clean_metadata[k] = v
        
        assume(clean_metadata)  # Non-empty after cleaning
        
        # ACT
        # Simulate storing and retrieving document with metadata
        doc_id = "test_doc"
        search_engine._store_document_metadata(doc_id, clean_metadata)
        retrieved_metadata = search_engine._get_document_metadata(doc_id)
        
        # ASSERT
        assert retrieved_metadata == clean_metadata, "Metadata should be preserved exactly"


class SearchEngineStateMachine(RuleBasedStateMachine):
    """Stateful property-based testing for search engine operations."""
    
    def __init__(self):
        super().__init__()
        self.search_engine = UnifiedSearchEngine()
        self.stored_documents = {}
        self.search_history = []
    
    @rule(doc_id=st.text(min_size=1, max_size=50),
          content=st.text(min_size=10, max_size=1000),
          metadata=st.dictionaries(st.text(min_size=1, max_size=20), st.text(), min_size=0, max_size=5))
    def store_document(self, doc_id: str, content: str, metadata: Dict[str, str]):
        """Store a document in the search engine."""
        assume(doc_id.strip() and content.strip())
        
        self.search_engine.store_document(doc_id, content, metadata)
        self.stored_documents[doc_id] = {'content': content, 'metadata': metadata}
    
    @rule(query=st.text(min_size=1, max_size=100))
    def search_documents(self, query: str):
        """Search for documents."""
        assume(query.strip())
        
        results = self.search_engine.search(query)
        self.search_history.append((query, results))
        
        # Basic invariants
        assert 'results' in results
        assert isinstance(results['results'], list)
        assert 'total_found' in results
    
    @rule(doc_id=st.text(min_size=1, max_size=50))
    def delete_document(self, doc_id: str):
        """Delete a document from the search engine."""
        if doc_id in self.stored_documents:
            self.search_engine.delete_document(doc_id)
            del self.stored_documents[doc_id]
    
    @invariant()
    def search_results_consistent(self):
        """Invariant: Search results should be consistent with stored documents."""
        if not self.stored_documents:
            return
        
        # Repeat last search to check consistency
        if self.search_history:
            last_query, last_results = self.search_history[-1]
            current_results = self.search_engine.search(last_query)
            
            # Results should be identical for same query
            assert current_results == last_results, "Search results should be consistent"
    
    @invariant()
    def document_count_consistent(self):
        """Invariant: Document count should match stored documents."""
        stored_count = len(self.stored_documents)
        engine_count = self.search_engine.get_document_count()
        
        assert engine_count == stored_count, "Document counts should match"


class TestSearchEngineStateful:
    """Run stateful property-based tests."""
    
    @settings(max_examples=50, stateful_step_count=20)
    def test_search_engine_state_machine(self):
        """Run the search engine state machine."""
        SearchEngineStateMachine.TestCase().runTest()


class TestPerformanceProperties:
    """Property-based tests for performance characteristics."""
    
    @given(st.integers(min_value=1, max_value=1000))
    def test_search_performance_scales_linearly(self, num_docs: int):
        """Property: Search performance should scale reasonably with document count."""
        # ARRANGE
        search_engine = UnifiedSearchEngine()
        
        # Store documents
        for i in range(num_docs):
            search_engine.store_document(f"doc_{i}", f"content {i} test document", {})
        
        # ACT
        import time
        start_time = time.time()
        results = search_engine.search("test")
        search_time = time.time() - start_time
        
        # ASSERT
        # Search time should be reasonable (under 1 second for 1000 docs)
        expected_max_time = 0.001 * num_docs + 0.1  # Linear + constant
        assert search_time < expected_max_time, f"Search took {search_time:.3f}s for {num_docs} docs"
    
    @given(st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=20))
    def test_batch_operations_efficient(self, queries: List[str]):
        """Property: Batch operations should be more efficient than individual ops."""
        # ARRANGE
        search_engine = UnifiedSearchEngine()
        valid_queries = [q for q in queries if q.strip()]
        assume(len(valid_queries) >= 2)
        
        # ACT
        # Individual searches
        start_time = time.time()
        individual_results = []
        for query in valid_queries:
            individual_results.append(search_engine.search(query))
        individual_time = time.time() - start_time
        
        # Batch search
        start_time = time.time()
        batch_results = search_engine.batch_search(valid_queries)
        batch_time = time.time() - start_time
        
        # ASSERT
        # Batch should be faster (or at least not significantly slower)
        efficiency_ratio = batch_time / individual_time if individual_time > 0 else 1
        assert efficiency_ratio < 1.5, f"Batch search should be efficient (ratio: {efficiency_ratio:.2f})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])