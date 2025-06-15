"""
Qdrant vector storage implementation

This replaces ChromaDB with Qdrant for better performance and features.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, MatchAny, MatchText,
    Range, DatetimeRange,
    SearchRequest, SearchParams,
    UpdateStatus
)

from .interfaces import VectorStorageInterface
from .models import Chunk
from .config import get_config

logger = logging.getLogger(__name__)

class QdrantStorage(VectorStorageInterface):
    """
    Qdrant implementation for vector storage.
    
    Qdrant provides:
    - Better performance than ChromaDB
    - More advanced filtering options
    - Payload storage alongside vectors
    - Snapshot and backup capabilities
    """
    
    def __init__(self, collection_name: Optional[str] = None):
        """Initialize Qdrant storage"""
        config = get_config()
        self.collection_name = collection_name or config.database.qdrant.collection_name
        
        # Initialize client
        self.client = QdrantClient(
            host=config.database.qdrant.host,
            port=config.database.qdrant.port,
            api_key=config.database.qdrant.api_key,
            https=config.database.qdrant.https,
            prefer_grpc=config.database.qdrant.prefer_grpc
        )
        
        # Vector configuration
        self.vector_size = config.embedding.dimension
        self.distance_metric = Distance.COSINE
        
        # Ensure collection exists
        self._ensure_collection()
        
        logger.info(f"Initialized Qdrant storage with collection: {self.collection_name}")
    
    def _build_advanced_filter(self, filter_dict: Dict[str, Any]) -> Optional[Filter]:
        """Build advanced Qdrant filter from filter dictionary"""
        if not filter_dict:
            return None
        
        conditions = []
        must_not_conditions = []
        
        # Basic filters (existing functionality)
        if 'book_ids' in filter_dict and filter_dict['book_ids']:
            book_ids = filter_dict['book_ids']
            if len(book_ids) == 1:
                conditions.append(
                    FieldCondition(
                        key="book_id",
                        match=MatchValue(value=book_ids[0])
                    )
                )
            else:
                conditions.append(
                    FieldCondition(
                        key="book_id",
                        match=MatchAny(any=book_ids)
                    )
                )
        
        if 'chunk_type' in filter_dict:
            conditions.append(
                FieldCondition(
                    key="chunk_type",
                    match=MatchValue(value=filter_dict['chunk_type'])
                )
            )
        
        # Advanced date filtering
        if 'date_range' in filter_dict and filter_dict['date_range']:
            date_range = filter_dict['date_range']
            if 'start' in date_range and 'end' in date_range:
                try:
                    start_date = datetime.fromisoformat(date_range['start'].replace('Z', '+00:00'))
                    end_date = datetime.fromisoformat(date_range['end'].replace('Z', '+00:00'))
                    conditions.append(
                        FieldCondition(
                            key="created_at",
                            range=DatetimeRange(
                                gte=start_date,
                                lte=end_date
                            )
                        )
                    )
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid date range format: {e}")
        
        if 'created_after' in filter_dict:
            try:
                after_date = datetime.fromisoformat(filter_dict['created_after'].replace('Z', '+00:00'))
                conditions.append(
                    FieldCondition(
                        key="created_at",
                        range=DatetimeRange(gte=after_date)
                    )
                )
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid created_after date format: {e}")
        
        if 'created_before' in filter_dict:
            try:
                before_date = datetime.fromisoformat(filter_dict['created_before'].replace('Z', '+00:00'))
                conditions.append(
                    FieldCondition(
                        key="created_at",
                        range=DatetimeRange(lte=before_date)
                    )
                )
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid created_before date format: {e}")
        
        # Advanced page filtering
        if 'page_range' in filter_dict and filter_dict['page_range']:
            page_range = filter_dict['page_range']
            if 'start' in page_range and 'end' in page_range:
                try:
                    start_page = int(page_range['start'])
                    end_page = int(page_range['end'])
                    if start_page <= end_page:  # Validate range
                        conditions.append(
                            FieldCondition(
                                key="page_start",
                                range=Range(gte=start_page, lte=end_page)
                            )
                        )
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid page range format: {e}")
        
        if 'pages' in filter_dict and filter_dict['pages']:
            try:
                page_numbers = [int(p) for p in filter_dict['pages']]
                conditions.append(
                    FieldCondition(
                        key="page_start",
                        match=MatchAny(any=page_numbers)
                    )
                )
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid page numbers format: {e}")
        
        # Advanced text filtering  
        if 'chapters' in filter_dict and filter_dict['chapters']:
            chapters = filter_dict['chapters']
            if len(chapters) == 1:
                conditions.append(
                    FieldCondition(
                        key="chapter",
                        match=MatchValue(value=chapters[0])
                    )
                )
            else:
                conditions.append(
                    FieldCondition(
                        key="chapter",
                        match=MatchAny(any=chapters)
                    )
                )
        
        if 'sections' in filter_dict and filter_dict['sections']:
            sections = filter_dict['sections']
            if len(sections) == 1:
                conditions.append(
                    FieldCondition(
                        key="section",
                        match=MatchValue(value=sections[0])
                    )
                )
            else:
                conditions.append(
                    FieldCondition(
                        key="section",
                        match=MatchAny(any=sections)
                    )
                )
        
        if 'section_pattern' in filter_dict:
            # Use text matching for pattern-based search
            conditions.append(
                FieldCondition(
                    key="section",
                    match=MatchText(text=filter_dict['section_pattern'])
                )
            )
        
        # Exclusion filters (must_not conditions)
        if 'exclude_books' in filter_dict and filter_dict['exclude_books']:
            exclude_books = filter_dict['exclude_books']
            if len(exclude_books) == 1:
                must_not_conditions.append(
                    FieldCondition(
                        key="book_id",
                        match=MatchValue(value=exclude_books[0])
                    )
                )
            else:
                must_not_conditions.append(
                    FieldCondition(
                        key="book_id",
                        match=MatchAny(any=exclude_books)
                    )
                )
        
        if 'exclude_chapters' in filter_dict and filter_dict['exclude_chapters']:
            exclude_chapters = filter_dict['exclude_chapters']
            if len(exclude_chapters) == 1:
                must_not_conditions.append(
                    FieldCondition(
                        key="chapter",
                        match=MatchValue(value=exclude_chapters[0])
                    )
                )
            else:
                must_not_conditions.append(
                    FieldCondition(
                        key="chapter",
                        match=MatchAny(any=exclude_chapters)
                    )
                )
        
        # Handle complex nested filters
        if 'complex_filter' in filter_dict:
            complex_filter = filter_dict['complex_filter']
            if 'or' in complex_filter:
                # Create OR conditions (should clause)
                or_conditions = []
                for or_condition in complex_filter['or']:
                    sub_filter = self._build_advanced_filter(or_condition)
                    if sub_filter and sub_filter.must:
                        or_conditions.extend(sub_filter.must)
                
                if or_conditions:
                    conditions.append(Filter(should=or_conditions))
        
        # Return filter if any conditions exist
        filter_args = {}
        if conditions:
            filter_args['must'] = conditions
        if must_not_conditions:
            filter_args['must_not'] = must_not_conditions
        
        return Filter(**filter_args) if filter_args else None
    
    def _ensure_collection(self):
        """Ensure collection exists with proper configuration"""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=self.distance_metric
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                # Verify configuration
                collection_info = self.client.get_collection(self.collection_name)
                current_size = collection_info.config.params.vectors.size
                
                if current_size != self.vector_size:
                    logger.warning(
                        f"Collection dimension mismatch: "
                        f"expected {self.vector_size}, got {current_size}"
                    )
                    
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
            raise
    
    async def save_embeddings(self, 
                             chunks: List[Chunk],
                             embeddings: List[List[float]]) -> bool:
        """Save chunk embeddings to Qdrant"""
        if not chunks or not embeddings:
            return True
        
        if len(chunks) != len(embeddings):
            logger.error(f"Mismatch: {len(chunks)} chunks, {len(embeddings)} embeddings")
            return False
        
        try:
            # Prepare points for Qdrant
            points = []
            
            for chunk, embedding in zip(chunks, embeddings):
                # Create payload
                payload = {
                    'chunk_id': chunk.id,
                    'book_id': chunk.book_id,
                    'chunk_index': chunk.chunk_index,
                    'chunk_type': chunk.chunk_type.value,
                    'text': chunk.text,
                    'created_at': chunk.created_at.isoformat()
                }
                
                # Add optional fields
                if chunk.chapter:
                    payload['chapter'] = chunk.chapter
                if chunk.section:
                    payload['section'] = chunk.section
                if chunk.page_start:
                    payload['page_start'] = chunk.page_start
                if chunk.page_end:
                    payload['page_end'] = chunk.page_end
                
                # Add chunk metadata
                payload.update(chunk.metadata)
                
                # Create point
                point = PointStruct(
                    id=str(uuid.uuid4()),  # Generate unique ID
                    vector=embedding,
                    payload=payload
                )
                points.append(point)
            
            # Upload in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                
                operation_info = self.client.upsert(
                    collection_name=self.collection_name,
                    wait=True,
                    points=batch
                )
                
                if operation_info.status.name != "COMPLETED":
                    logger.error(f"Failed to upsert batch: {operation_info}")
                    return False
                
                logger.debug(f"Uploaded batch {i//batch_size + 1} ({len(batch)} points)")
            
            logger.info(f"Successfully saved {len(chunks)} embeddings to Qdrant")
            return True
            
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            return False
    
    async def search_semantic(self,
                             query_embedding: List[float],
                             filter_dict: Optional[Dict[str, Any]] = None,
                             limit: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic search using Qdrant"""
        try:
            # Build advanced filter
            qdrant_filter = self._build_advanced_filter(filter_dict)
            
            # Perform search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=qdrant_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False  # We don't need vectors back
            )
            
            # Format results
            search_results = []
            for hit in search_result:
                search_results.append({
                    'chunk_id': hit.payload['chunk_id'],
                    'text': hit.payload['text'],
                    'metadata': {
                        k: v for k, v in hit.payload.items()
                        if k not in ['chunk_id', 'text']
                    },
                    'score': hit.score,
                    'distance': 1 - hit.score  # Convert similarity to distance
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
            # Delete by filter using MatchAny for multiple values
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="chunk_id",
                            match=MatchAny(any=chunk_ids)
                        )
                    ]
                )
            )
            
            logger.info(f"Deleted embeddings for {len(chunk_ids)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting embeddings: {e}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            return {
                'collection_name': self.collection_name,
                'total_embeddings': collection_info.points_count,
                'vector_size': collection_info.config.params.vectors.size,
                'distance_metric': collection_info.config.params.vectors.distance.name,
                'segments_count': collection_info.segments_count,
                'status': collection_info.status
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {
                'collection_name': self.collection_name,
                'error': str(e)
            }
    
    def create_snapshot(self) -> str:
        """Create a snapshot of the collection"""
        try:
            snapshot_info = self.client.create_snapshot(
                collection_name=self.collection_name
            )
            logger.info(f"Created snapshot: {snapshot_info}")
            return snapshot_info.name
            
        except Exception as e:
            logger.error(f"Error creating snapshot: {e}")
            return ""
    
    async def cleanup(self):
        """Cleanup Qdrant client resources"""
        try:
            # Close any open connections
            if hasattr(self.client, 'close'):
                await self.client.close()
            logger.info("Qdrant client cleaned up")
        except Exception as e:
            logger.error(f"Error during Qdrant cleanup: {e}")

# Note: ChromaDBStorage compatibility removed - use QdrantStorage directly
# If you need ChromaDB, import from core.chroma_storage instead