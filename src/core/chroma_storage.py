"""
ChromaDB vector storage implementation

DEPRECATED: This module is deprecated in favor of QdrantStorage.
Use src.core.qdrant_storage.QdrantStorage instead for better performance and features.

This handles semantic search using vector embeddings.
"""

import logging
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
from pathlib import Path

import chromadb
from chromadb.config import Settings

from .interfaces import VectorStorageInterface
from .models import Chunk

logger = logging.getLogger(__name__)

class ChromaDBStorage(VectorStorageInterface):
    """
    ChromaDB implementation for vector storage.
    
    This provides semantic search capabilities by storing
    and searching through vector embeddings.
    """
    
    def __init__(self, persist_directory: Optional[str] = None, collection_name: Optional[str] = None):
        """Initialize ChromaDB storage"""
        self.persist_directory = persist_directory or "data/chromadb"
        self.collection_name = collection_name or "tradeknowledge"
        
        # Ensure directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
        logger.info(f"Initialized ChromaDB with collection: {self.collection_name}")
    
    def _get_or_create_collection(self):
        """Get or create the main collection"""
        try:
            collection = self.client.get_collection(self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
        except Exception:
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Trading and ML book embeddings",
                    "created_at": datetime.now().isoformat()
                }
            )
            logger.info(f"Created new collection: {self.collection_name}")
        
        return collection
    
    async def save_embeddings(self, 
                             chunks: List[Chunk],
                             embeddings: List[List[float]]) -> bool:
        """Save chunk embeddings to ChromaDB"""
        if not chunks or not embeddings:
            return True
        
        if len(chunks) != len(embeddings):
            logger.error(f"Mismatch: {len(chunks)} chunks, {len(embeddings)} embeddings")
            return False
        
        try:
            # Prepare data for ChromaDB
            ids = []
            documents = []
            metadatas = []
            
            for chunk in chunks:
                ids.append(chunk.id)
                documents.append(chunk.text)
                
                # Prepare metadata
                metadata = {
                    'book_id': chunk.book_id,
                    'chunk_index': chunk.chunk_index,
                    'chunk_type': chunk.chunk_type.value,
                    'created_at': chunk.created_at.isoformat()
                }
                
                # Add optional fields
                if chunk.chapter:
                    metadata['chapter'] = chunk.chapter
                if chunk.section:
                    metadata['section'] = chunk.section
                if chunk.page_start:
                    metadata['page_start'] = chunk.page_start
                if chunk.page_end:
                    metadata['page_end'] = chunk.page_end
                
                metadatas.append(metadata)
            
            # Add to collection in batches
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_docs = documents[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                batch_metadata = metadatas[i:i + batch_size]
                
                # Use asyncio to avoid blocking
                await asyncio.to_thread(
                    self.collection.add,
                    ids=batch_ids,
                    documents=batch_docs,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadata
                )
                
                logger.debug(f"Added batch {i//batch_size + 1} ({len(batch_ids)} chunks)")
            
            logger.info(f"Successfully saved {len(chunks)} embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            return False
    
    async def search_semantic(self,
                             query_embedding: List[float],
                             filter_dict: Optional[Dict[str, Any]] = None,
                             limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform semantic search using vector similarity.
        
        Args:
            query_embedding: Vector embedding of the query
            filter_dict: Metadata filters (e.g., {'book_id': 'xyz'})
            limit: Maximum number of results
            
        Returns:
            List of search results with chunks and scores
        """
        try:
            # Build where clause for filtering
            where = None
            if filter_dict:
                # ChromaDB expects specific filter format
                where = {}
                if 'book_ids' in filter_dict and filter_dict['book_ids']:
                    where['book_id'] = {'$in': filter_dict['book_ids']}
                if 'chunk_type' in filter_dict:
                    where['chunk_type'] = filter_dict['chunk_type']
            
            # Perform search
            results = await asyncio.to_thread(
                self.collection.query,
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            search_results = []
            if results['ids'] and results['ids'][0]:
                for i, chunk_id in enumerate(results['ids'][0]):
                    # Convert distance to similarity score (1 - normalized_distance)
                    # ChromaDB uses L2 distance by default
                    distance = results['distances'][0][i]
                    score = 1 / (1 + distance)  # Convert distance to similarity
                    
                    search_results.append({
                        'chunk_id': chunk_id,
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'score': score,
                        'distance': distance
                    })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    async def delete_embeddings(self, chunk_ids: List[str]) -> bool:
        """Delete embeddings by chunk IDs"""
        if not chunk_ids:
            return True
        
        try:
            await asyncio.to_thread(
                self.collection.delete,
                ids=chunk_ids
            )
            logger.info(f"Deleted {len(chunk_ids)} embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting embeddings: {e}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection"""
        try:
            count = await asyncio.to_thread(self.collection.count)
            
            # Get collection metadata
            metadata = getattr(self.collection, 'metadata', None) or {}
            
            return {
                'collection_name': self.collection_name,
                'total_embeddings': count,
                'persist_directory': self.persist_directory,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {
                'collection_name': self.collection_name,
                'error': str(e)
            }

# Test ChromaDB storage
async def test_chroma_storage():
    """Test ChromaDB storage implementation"""
    storage = ChromaDBStorage()
    
    # Get stats
    stats = await storage.get_collection_stats()
    print(f"Collection stats: {stats}")
    
    # Create test data
    test_chunks = [
        Chunk(
            id=f"test_chunk_{i}",
            book_id="test_book",
            chunk_index=i,
            text=f"Test chunk {i} about trading strategies"
        )
        for i in range(3)
    ]
    
    # Create fake embeddings (normally from embedding generator)
    import random
    test_embeddings = [
        [random.random() for _ in range(384)]  # 384-dim embeddings
        for _ in test_chunks
    ]
    
    # Save embeddings
    success = await storage.save_embeddings(test_chunks, test_embeddings)
    print(f"Save embeddings: {success}")
    
    # Test search
    query_embedding = [random.random() for _ in range(384)]
    results = await storage.search_semantic(query_embedding, limit=2)
    
    print(f"\nSearch results ({len(results)} found):")
    for result in results:
        print(f"  - ID: {result['chunk_id']}")
        print(f"    Score: {result['score']:.3f}")
        print(f"    Text: {result['text'][:50]}...")

if __name__ == "__main__":
    asyncio.run(test_chroma_storage())