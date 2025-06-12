"""
Vector Search Engine for TradeKnowledge
Handles semantic search using ChromaDB
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid

import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection

from utils.logging import get_logger
from core.config import get_config
from ingestion.embeddings import EmbeddingGenerator

logger = get_logger(__name__)

class VectorSearchEngine:
    """
    Vector search engine using ChromaDB for semantic similarity search
    """
    
    def __init__(self, collection_name: Optional[str] = None, persist_directory: Optional[str] = None):
        self.config = get_config()
        self.collection_name = collection_name or self.config.database.chroma.collection_name
        self.persist_directory = persist_directory or self.config.database.chroma.persist_directory
        
        # ChromaDB client and collection
        self.client = None
        self.collection = None
        
        # Embedding generator for query embeddings
        self.embedding_generator = None
        
        # Search configuration
        self.default_results = self.config.search.default_results
        self.max_results = self.config.search.max_results
        self.min_score = self.config.search.min_score
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client"""
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=False
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except Exception:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={
                        "description": "Trading and ML book embeddings",
                        "created_at": datetime.now().isoformat()
                    }
                )
                logger.info(f"Created new collection: {self.collection_name}")
            
            # Initialize embedding generator
            self.embedding_generator = EmbeddingGenerator()
            
            logger.info(f"Vector search engine initialized with {self.collection.count()} documents")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector search engine: {e}")
            raise
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Add documents to the vector store
        
        Args:
            documents: List of document dicts with:
                - chunk_id: unique identifier
                - text: text content
                - embedding: vector embedding
                - metadata: additional metadata
        
        Returns:
            Success status
        """
        if not documents:
            return True
        
        try:
            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            metadatas = []
            documents_text = []
            
            for doc in documents:
                ids.append(doc["chunk_id"])
                embeddings.append(doc["embedding"])
                documents_text.append(doc["text"])
                
                # Prepare metadata (ChromaDB has limitations on metadata types)
                metadata = {
                    "book_id": doc.get("book_id", ""),
                    "chunk_index": doc.get("chunk_index", 0),
                    "content_type": doc.get("content_type", "text"),
                    "boundary_type": doc.get("boundary_type", "paragraph"),
                }
                
                # Add safe metadata fields
                if "metadata" in doc and isinstance(doc["metadata"], dict):
                    for key, value in doc["metadata"].items():
                        # ChromaDB only supports certain types
                        if isinstance(value, (str, int, float, bool)):
                            metadata[f"meta_{key}"] = value
                        elif isinstance(value, dict):
                            # Flatten dict metadata
                            for subkey, subvalue in value.items():
                                if isinstance(subvalue, (str, int, float, bool)):
                                    metadata[f"meta_{key}_{subkey}"] = subvalue
                
                metadatas.append(metadata)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents_text
            )
            
            logger.info(f"Added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            return False
    
    async def search_semantic(self, 
                            query: str,
                            num_results: int = None,
                            filter_books: Optional[List[str]] = None,
                            min_score: Optional[float] = None) -> Dict[str, Any]:
        """
        Perform semantic search using vector similarity
        
        Args:
            query: Search query text
            num_results: Number of results to return
            filter_books: List of book IDs to filter by
            min_score: Minimum similarity score threshold
            
        Returns:
            Search results with metadata
        """
        start_time = datetime.now()
        
        try:
            # Set defaults
            num_results = min(num_results or self.default_results, self.max_results)
            min_score = min_score or self.min_score
            
            # Generate query embedding
            query_embedding = await self.embedding_generator.generate_embedding(query)
            
            # Prepare where clause for filtering
            where_clause = None
            if filter_books:
                where_clause = {"book_id": {"$in": filter_books}}
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=num_results,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            search_results = []
            if results["ids"] and results["ids"][0]:
                for i, chunk_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i]
                    similarity_score = 1 - distance  # Convert distance to similarity
                    
                    # Filter by minimum score
                    if similarity_score < min_score:
                        continue
                    
                    # Extract metadata
                    metadata = results["metadatas"][0][i]
                    processed_metadata = self._process_metadata(metadata)
                    
                    search_results.append({
                        "chunk_id": chunk_id,
                        "text": results["documents"][0][i],
                        "score": round(similarity_score, 4),
                        "metadata": processed_metadata
                    })
            
            # Sort by score (highest first)
            search_results.sort(key=lambda x: x["score"], reverse=True)
            
            search_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                "results": search_results,
                "total_results": len(search_results),
                "search_time_ms": int(search_time),
                "query": query,
                "search_type": "semantic"
            }
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return {
                "results": [],
                "total_results": 0,
                "search_time_ms": 0,
                "query": query,
                "error": str(e),
                "search_type": "semantic"
            }
    
    def _process_metadata(self, raw_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process and clean metadata from ChromaDB"""
        processed = {}
        
        # Extract standard fields
        for field in ["book_id", "chunk_index", "content_type", "boundary_type"]:
            if field in raw_metadata:
                processed[field] = raw_metadata[field]
        
        # Extract and unflatten meta_ fields
        for key, value in raw_metadata.items():
            if key.startswith("meta_"):
                clean_key = key[5:]  # Remove "meta_" prefix
                
                # Handle nested keys (meta_chapter_title -> chapter.title)
                if "_" in clean_key:
                    parts = clean_key.split("_", 1)
                    if parts[0] not in processed:
                        processed[parts[0]] = {}
                    if isinstance(processed[parts[0]], dict):
                        processed[parts[0]][parts[1]] = value
                    else:
                        processed[clean_key] = value
                else:
                    processed[clean_key] = value
        
        return processed
    
    async def get_similar_chunks(self, chunk_id: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Get chunks similar to a given chunk"""
        try:
            # Get the chunk to find its embedding
            result = self.collection.get(
                ids=[chunk_id],
                include=["embeddings", "documents", "metadatas"]
            )
            
            if not result["ids"]:
                return []
            
            # Use its embedding to find similar chunks
            embedding = result["embeddings"][0]
            
            similar_results = self.collection.query(
                query_embeddings=[embedding],
                n_results=num_results + 1,  # +1 to exclude the original
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results (skip the first one which should be the original)
            similar_chunks = []
            for i, similar_id in enumerate(similar_results["ids"][0]):
                if similar_id == chunk_id:
                    continue  # Skip the original chunk
                
                distance = similar_results["distances"][0][i]
                similarity_score = 1 - distance
                
                similar_chunks.append({
                    "chunk_id": similar_id,
                    "text": similar_results["documents"][0][i],
                    "score": round(similarity_score, 4),
                    "metadata": self._process_metadata(similar_results["metadatas"][0][i])
                })
            
            return similar_chunks[:num_results]
            
        except Exception as e:
            logger.error(f"Failed to get similar chunks for {chunk_id}: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection"""
        try:
            count = self.collection.count()
            
            # Get a sample to analyze
            sample_size = min(100, count)
            if sample_size > 0:
                sample = self.collection.peek(limit=sample_size)
                
                # Analyze content types
                content_types = {}
                book_ids = set()
                
                for metadata in sample.get("metadatas", []):
                    content_type = metadata.get("content_type", "unknown")
                    content_types[content_type] = content_types.get(content_type, 0) + 1
                    
                    book_id = metadata.get("book_id")
                    if book_id:
                        book_ids.add(book_id)
                
                return {
                    "total_chunks": count,
                    "unique_books": len(book_ids),
                    "content_types": content_types,
                    "embedding_dimension": len(sample["embeddings"][0]) if sample.get("embeddings") else 0,
                    "collection_name": self.collection_name
                }
            else:
                return {
                    "total_chunks": 0,
                    "unique_books": 0,
                    "content_types": {},
                    "embedding_dimension": 0,
                    "collection_name": self.collection_name
                }
                
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {
                "total_chunks": 0,
                "error": str(e),
                "collection_name": self.collection_name
            }
    
    def delete_book(self, book_id: str) -> bool:
        """Delete all chunks for a specific book"""
        try:
            # Get all chunk IDs for this book
            results = self.collection.get(
                where={"book_id": book_id},
                include=["ids"]
            )
            
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} chunks for book {book_id}")
                return True
            else:
                logger.info(f"No chunks found for book {book_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete book {book_id}: {e}")
            return False
    
    def reset_collection(self) -> bool:
        """Reset the entire collection (delete all data)"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Trading and ML book embeddings",
                    "created_at": datetime.now().isoformat()
                }
            )
            logger.info(f"Reset collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            return False

# Convenience functions
async def search_vectors(query: str, num_results: int = 10, 
                        filter_books: Optional[List[str]] = None) -> Dict[str, Any]:
    """Convenience function for vector search"""
    engine = VectorSearchEngine()
    return await engine.search_semantic(query, num_results, filter_books)