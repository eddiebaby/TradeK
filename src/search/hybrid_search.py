"""
Hybrid search engine combining semantic and exact search

This is where the magic happens - we combine vector similarity
with traditional text search for the best results.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import time

from core.models import SearchResult, SearchResponse, Chunk
from core.sqlite_storage import SQLiteStorage
from core.config import Config, get_config
from core.chroma_storage import ChromaDBStorage
from ingestion.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)

class HybridSearch:
    """
    Hybrid search engine combining semantic and exact search.
    
    This class orchestrates:
    - Semantic search through ChromaDB
    - Exact text search through SQLite FTS5
    - Result merging and ranking
    - Context retrieval
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize search engine"""
        self.config = config or get_config()
        
        # Storage backends
        self.sqlite_storage: Optional[SQLiteStorage] = None
        self.chroma_storage: Optional[ChromaDBStorage] = None
        
        # Embedding generator
        self.embedding_generator: Optional[EmbeddingGenerator] = None
        
        # Search statistics
        self.search_count = 0
        self.total_search_time = 0
    
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing hybrid search engine...")
        
        # Initialize storage
        self.sqlite_storage = SQLiteStorage(self.config)
        self.chroma_storage = ChromaDBStorage(self.config)
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(self.config)
        
        # Load embedding cache if available
        cache_path = "data/embeddings/cache.json"
        self.embedding_generator.load_cache(cache_path)
        
        logger.info("Search engine initialized")
    
    async def cleanup(self):
        """Cleanup resources"""
        # Save embedding cache
        if self.embedding_generator:
            self.embedding_generator.save_cache("data/embeddings/cache.json")
    
    async def search_semantic(self,
                            query: str,
                            num_results: int = 10,
                            filter_books: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform semantic search only.
        
        This searches based on meaning similarity, finding content
        that's conceptually related even if different words are used.
        """
        start_time = time.time()
        
        try:
            # Generate query embedding
            logger.debug(f"Generating embedding for query: {query[:50]}...")
            query_embedding = await self.embedding_generator.generate_query_embedding(query)
            
            # Build filter
            filter_dict = {}
            if filter_books:
                filter_dict['book_ids'] = filter_books
            
            # Search in ChromaDB
            logger.debug("Searching in vector database...")
            results = await self.chroma_storage.search_semantic(
                query_embedding=query_embedding,
                filter_dict=filter_dict,
                limit=num_results
            )
            
            # Convert to SearchResponse
            search_results = []
            for result in results:
                # Get full chunk data
                chunk = await self.sqlite_storage.get_chunk(result['chunk_id'])
                if not chunk:
                    continue
                
                # Get book info
                book = await self.sqlite_storage.get_book(chunk.book_id)
                if not book:
                    continue
                
                search_result = SearchResult(
                    chunk=chunk,
                    score=result['score'],
                    match_type='semantic',
                    highlights=[self._extract_highlight(chunk.text, query)],
                    book_title=book.title,
                    book_author=book.author,
                    chapter=result['metadata'].get('chapter'),
                    page=result['metadata'].get('page_start')
                )
                
                search_results.append(search_result)
            
            # Build response
            search_time = int((time.time() - start_time) * 1000)
            
            response = SearchResponse(
                query=query,
                results=search_results,
                total_results=len(results),
                returned_results=len(search_results),
                search_time_ms=search_time,
                search_type='semantic',
                filters_applied={'book_ids': filter_books} if filter_books else {}
            )
            
            # Update statistics
            self.search_count += 1
            self.total_search_time += search_time
            
            return response.dict()
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                returned_results=0,
                search_time_ms=int((time.time() - start_time) * 1000),
                search_type='semantic'
            ).dict()
    
    async def search_exact(self,
                          query: str,
                          num_results: int = 10,
                          filter_books: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform exact text search.
        
        This finds exact matches of words or phrases,
        useful for finding specific terms or code snippets.
        """
        start_time = time.time()
        
        try:
            # Search in SQLite FTS
            logger.debug(f"Performing exact search for: {query}")
            results = await self.sqlite_storage.search_exact(
                query=query,
                book_ids=filter_books,
                limit=num_results
            )
            
            # Convert to SearchResponse
            search_results = []
            for result in results:
                chunk = result['chunk']
                
                # Get book info
                book = await self.sqlite_storage.get_book(chunk.book_id)
                if not book:
                    continue
                
                search_result = SearchResult(
                    chunk=chunk,
                    score=result['score'],
                    match_type='exact',
                    highlights=[result.get('snippet', '')],
                    book_title=book.title,
                    book_author=book.author,
                    chapter=chunk.metadata.get('chapter'),
                    page=chunk.page_start
                )
                
                search_results.append(search_result)
            
            # Build response
            search_time = int((time.time() - start_time) * 1000)
            
            response = SearchResponse(
                query=query,
                results=search_results,
                total_results=len(results),
                returned_results=len(search_results),
                search_time_ms=search_time,
                search_type='exact',
                filters_applied={'book_ids': filter_books} if filter_books else {}
            )
            
            return response.dict()
            
        except Exception as e:
            logger.error(f"Error in exact search: {e}")
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                returned_results=0,
                search_time_ms=int((time.time() - start_time) * 1000),
                search_type='exact'
            ).dict()
    
    async def search_hybrid(self,
                           query: str,
                           num_results: int = 10,
                           filter_books: Optional[List[str]] = None,
                           semantic_weight: float = 0.7) -> Dict[str, Any]:
        """
        Perform hybrid search combining semantic and exact.
        
        This is our secret sauce - we run both searches and
        intelligently combine the results for best relevance.
        
        Args:
            query: Search query
            num_results: Number of results to return
            filter_books: Optional book IDs to search within
            semantic_weight: Weight for semantic results (0-1)
        """
        start_time = time.time()
        
        try:
            # Run both searches in parallel
            logger.debug(f"Running hybrid search for: {query}")
            
            semantic_task = self.search_semantic(query, num_results * 2, filter_books)
            exact_task = self.search_exact(query, num_results * 2, filter_books)
            
            semantic_response, exact_response = await asyncio.gather(
                semantic_task, exact_task
            )
            
            # Merge results
            merged_results = self._merge_results(
                semantic_response['results'],
                exact_response['results'],
                semantic_weight
            )
            
            # Take top N results
            final_results = merged_results[:num_results]
            
            # Build response
            search_time = int((time.time() - start_time) * 1000)
            
            response = SearchResponse(
                query=query,
                results=final_results,
                total_results=len(merged_results),
                returned_results=len(final_results),
                search_time_ms=search_time,
                search_type='hybrid',
                filters_applied={
                    'book_ids': filter_books,
                    'semantic_weight': semantic_weight
                } if filter_books else {'semantic_weight': semantic_weight}
            )
            
            return response.dict()
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                returned_results=0,
                search_time_ms=int((time.time() - start_time) * 1000),
                search_type='hybrid'
            ).dict()
    
    async def get_chunk_context(self,
                               chunk_id: str,
                               before_chunks: int = 1,
                               after_chunks: int = 1) -> Dict[str, Any]:
        """
        Get expanded context for a chunk.
        
        This is useful for showing more context around
        a search result when the user wants to see more.
        """
        try:
            context = await self.sqlite_storage.get_chunk_context(
                chunk_id=chunk_id,
                before=before_chunks,
                after=after_chunks
            )
            
            if not context:
                return {'error': 'Chunk not found'}
            
            # Format response
            response = {
                'chunk_id': chunk_id,
                'chunk': context['chunk'].dict() if context.get('chunk') else None,
                'context': {
                    'before': [c.dict() for c in context.get('before', [])],
                    'after': [c.dict() for c in context.get('after', [])]
                }
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting chunk context: {e}")
            return {'error': str(e)}
    
    def _merge_results(self,
                      semantic_results: List[Dict],
                      exact_results: List[Dict],
                      semantic_weight: float) -> List[SearchResult]:
        """
        Merge and re-rank results from both search types.
        
        This is a simple weighted combination, but could be
        made more sophisticated with learning-to-rank models.
        """
        # Create a map of chunk_id to result
        result_map = {}
        
        # Add semantic results
        for result in semantic_results:
            chunk_id = result['chunk']['id']
            result_map[chunk_id] = {
                'result': result,
                'semantic_score': result['score'],
                'exact_score': 0.0
            }
        
        # Add/update with exact results
        for result in exact_results:
            chunk_id = result['chunk']['id']
            if chunk_id in result_map:
                result_map[chunk_id]['exact_score'] = result['score']
            else:
                result_map[chunk_id] = {
                    'result': result,
                    'semantic_score': 0.0,
                    'exact_score': result['score']
                }
        
        # Calculate combined scores
        exact_weight = 1 - semantic_weight
        for chunk_id, data in result_map.items():
            # Normalize scores to 0-1 range
            semantic_score = min(data['semantic_score'], 1.0)
            exact_score = min(data['exact_score'], 1.0)
            
            # Calculate weighted score
            combined_score = (
                semantic_score * semantic_weight +
                exact_score * exact_weight
            )
            
            # Update the result
            data['result']['score'] = combined_score
            data['result']['match_type'] = 'hybrid'
        
        # Sort by combined score
        sorted_results = sorted(
            result_map.values(),
            key=lambda x: x['result']['score'],
            reverse=True
        )
        
        # Return just the result objects as SearchResult instances
        final_results = []
        for item in sorted_results:
            result_dict = item['result']
            # Convert dict back to SearchResult if needed
            if isinstance(result_dict, dict):
                chunk_dict = result_dict['chunk']
                chunk = Chunk(**chunk_dict) if isinstance(chunk_dict, dict) else chunk_dict
                
                search_result = SearchResult(
                    chunk=chunk,
                    score=result_dict['score'],
                    match_type=result_dict['match_type'],
                    highlights=result_dict.get('highlights', []),
                    context_before=result_dict.get('context_before'),
                    context_after=result_dict.get('context_after'),
                    book_title=result_dict['book_title'],
                    book_author=result_dict.get('book_author'),
                    chapter=result_dict.get('chapter'),
                    page=result_dict.get('page')
                )
                final_results.append(search_result)
            else:
                final_results.append(result_dict)
        
        return final_results
    
    def _extract_highlight(self, text: str, query: str, context_length: int = 100) -> str:
        """
        Extract a relevant highlight from the text.
        
        This finds the most relevant snippet to show in search results.
        """
        # Simple implementation - find first occurrence
        query_lower = query.lower()
        text_lower = text.lower()
        
        pos = text_lower.find(query_lower)
        if pos == -1:
            # Query not found, return beginning
            return text[:context_length * 2] + '...' if len(text) > context_length * 2 else text
        
        # Extract context around match
        start = max(0, pos - context_length)
        end = min(len(text), pos + len(query) + context_length)
        
        highlight = text[start:end]
        
        # Add ellipsis if needed
        if start > 0:
            highlight = '...' + highlight
        if end < len(text):
            highlight = highlight + '...'
        
        return highlight

    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        avg_time = self.total_search_time / self.search_count if self.search_count > 0 else 0
        
        return {
            'total_searches': self.search_count,
            'total_search_time_ms': self.total_search_time,
            'average_search_time_ms': avg_time,
            'components_initialized': {
                'sqlite_storage': self.sqlite_storage is not None,
                'chroma_storage': self.chroma_storage is not None,
                'embedding_generator': self.embedding_generator is not None
            }
        }

# Test the search engine
async def test_search_engine():
    """Test the hybrid search engine"""
    
    # Initialize
    search_engine = HybridSearch()
    await search_engine.initialize()
    
    # Test queries
    test_queries = [
        "moving average trading strategy",
        "def calculate_sma",
        "momentum indicators"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        # Semantic search
        print("\nSemantic Search:")
        results = await search_engine.search_semantic(query, num_results=3)
        print(f"Found {results['total_results']} results in {results['search_time_ms']}ms")
        
        # Exact search
        print("\nExact Search:")
        results = await search_engine.search_exact(query, num_results=3)
        print(f"Found {results['total_results']} results in {results['search_time_ms']}ms")
        
        # Hybrid search
        print("\nHybrid Search:")
        results = await search_engine.search_hybrid(query, num_results=3)
        print(f"Found {results['total_results']} results in {results['search_time_ms']}ms")
        
        if results['results']:
            print("\nTop result:")
            top = results['results'][0]
            print(f"Book: {top['book_title']}")
            print(f"Score: {top['score']:.3f}")
            print(f"Preview: {top['highlights'][0] if top['highlights'] else 'N/A'}")
    
    # Show stats
    print(f"\nSearch Engine Stats: {search_engine.get_stats()}")
    
    # Cleanup
    await search_engine.cleanup()

if __name__ == "__main__":
    asyncio.run(test_search_engine())