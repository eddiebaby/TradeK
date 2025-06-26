#!/usr/bin/env python3
"""
Persistent Memory System for InfluxDB Blackboard
Advanced Long-Term Memory with Intelligent Storage and Retrieval

This system provides:
- Long-term persistent memory across agent sessions
- Intelligent memory indexing and retrieval
- Memory consolidation and pattern recognition
- Cross-agent memory sharing and learning
- Memory decay and importance scoring
- Efficient storage using InfluxDB time-series
"""

import os
import json
import asyncio
import time
import hashlib
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
from enum import Enum

try:
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import SYNCHRONOUS
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False
    print("⚠️  InfluxDB client not available. Run: pip install influxdb-client")

# Import existing blackboard if available
try:
    from influx_blackboard import get_blackboard, InfluxBlackboard
    BLACKBOARD_AVAILABLE = True
except ImportError:
    BLACKBOARD_AVAILABLE = False
    print("⚠️  Blackboard not available - using standalone mode")

class MemoryType(Enum):
    """Types of persistent memory"""
    WORKING = "working"           # Short-term working memory (hours)
    EPISODIC = "episodic"        # Event/experience memory (days)
    SEMANTIC = "semantic"        # Fact/knowledge memory (months)
    PROCEDURAL = "procedural"    # Skill/process memory (permanent)
    CONTEXTUAL = "contextual"    # Context/environment memory (days)

class MemoryImportance(Enum):
    """Memory importance levels for decay calculations"""
    CRITICAL = 10    # Never decay
    HIGH = 8        # Very slow decay
    MEDIUM = 5      # Normal decay
    LOW = 3         # Fast decay
    TEMPORARY = 1   # Very fast decay

@dataclass
class MemoryEntry:
    """Persistent memory entry"""
    id: str
    agent: str
    memory_type: MemoryType
    importance: MemoryImportance
    content: Dict[str, Any]
    tags: List[str]
    context: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None
    created_at: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0
    decay_score: float = 1.0
    linked_memories: List[str] = None
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.last_accessed == 0.0:
            self.last_accessed = self.created_at
        if self.linked_memories is None:
            self.linked_memories = []

@dataclass
class MemoryQuery:
    """Memory retrieval query"""
    agent: str
    query_text: Optional[str] = None
    memory_types: List[MemoryType] = None
    tags: List[str] = None
    importance_threshold: int = 1
    time_range: Optional[Tuple[datetime, datetime]] = None
    limit: int = 10
    include_decay: bool = True
    semantic_similarity: bool = True

class MemoryConsolidator:
    """Consolidates and optimizes memory storage"""
    
    def __init__(self):
        self.consolidation_patterns = {}
        self.importance_weights = {
            "access_frequency": 0.3,
            "recency": 0.2,
            "cross_references": 0.2,
            "agent_importance": 0.2,
            "content_richness": 0.1
        }
    
    def calculate_importance_score(self, memory: MemoryEntry) -> float:
        """Calculate dynamic importance score"""
        now = time.time()
        age_hours = (now - memory.created_at) / 3600
        recency_hours = (now - memory.last_accessed) / 3600
        
        # Base importance
        base_score = memory.importance.value / 10.0
        
        # Access frequency bonus
        access_score = min(1.0, memory.access_count / 10.0)
        
        # Recency bonus (higher for recently accessed)
        recency_score = max(0.1, 1.0 / (1 + recency_hours / 24))
        
        # Content richness (based on data complexity)
        content_score = min(1.0, len(str(memory.content)) / 1000)
        
        # Cross-reference bonus
        link_score = min(1.0, len(memory.linked_memories) / 5.0)
        
        # Weighted combination
        weights = self.importance_weights
        total_score = (
            base_score * weights["agent_importance"] +
            access_score * weights["access_frequency"] +
            recency_score * weights["recency"] +
            link_score * weights["cross_references"] +
            content_score * weights["content_richness"]
        )
        
        return min(1.0, total_score)
    
    def calculate_decay_score(self, memory: MemoryEntry) -> float:
        """Calculate memory decay based on type and importance"""
        if memory.importance == MemoryImportance.CRITICAL:
            return 1.0  # No decay
        
        now = time.time()
        age_hours = (now - memory.created_at) / 3600
        
        # Memory type decay rates (half-life in hours)
        decay_rates = {
            MemoryType.WORKING: 24,      # 1 day
            MemoryType.EPISODIC: 168,    # 1 week
            MemoryType.SEMANTIC: 2160,   # 3 months
            MemoryType.PROCEDURAL: float('inf'),  # No decay
            MemoryType.CONTEXTUAL: 72    # 3 days
        }
        
        half_life = decay_rates.get(memory.memory_type, 168)
        if half_life == float('inf'):
            return 1.0
        
        # Exponential decay with importance modifier
        importance_modifier = memory.importance.value / 10.0
        adjusted_half_life = half_life * (1 + importance_modifier)
        
        decay_score = 0.5 ** (age_hours / adjusted_half_life)
        
        # Boost for recently accessed memories
        if memory.last_accessed > now - 3600:  # Accessed in last hour
            decay_score = min(1.0, decay_score * 1.5)
        
        return max(0.01, decay_score)  # Minimum decay score

class PersistentMemorySystem:
    """Advanced persistent memory system using InfluxDB"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path(__file__).parent / "config" / "blackboard_influx.yaml"
        self.consolidator = MemoryConsolidator()
        self.embedding_cache = {}
        
        # Initialize InfluxDB connection
        if BLACKBOARD_AVAILABLE:
            self.blackboard = get_blackboard()
            self.client = self.blackboard.client
            self.write_api = self.blackboard.write_api
            self.query_api = self.blackboard.query_api
        else:
            self.blackboard = None
            self.client = None
            self.write_api = None
            self.query_api = None
            
        # Memory buckets for different types
        self.memory_buckets = {
            MemoryType.WORKING: "working_memory",
            MemoryType.EPISODIC: "episodic_memory", 
            MemoryType.SEMANTIC: "semantic_memory",
            MemoryType.PROCEDURAL: "procedural_memory",
            MemoryType.CONTEXTUAL: "contextual_memory"
        }
        
        # Start background tasks
        self._consolidation_task = None
        self._running = False
    
    async def start(self):
        """Start the persistent memory system"""
        self._running = True
        
        # Start background consolidation
        self._consolidation_task = asyncio.create_task(self._background_consolidation())
        
        print("✅ Persistent memory system started")
    
    async def stop(self):
        """Stop the persistent memory system"""
        self._running = False
        
        if self._consolidation_task:
            self._consolidation_task.cancel()
            try:
                await self._consolidation_task
            except asyncio.CancelledError:
                pass
        
        print("✅ Persistent memory system stopped")
    
    async def store_memory(self, 
                          agent: str,
                          memory_type: MemoryType,
                          content: Dict[str, Any],
                          importance: MemoryImportance = MemoryImportance.MEDIUM,
                          tags: List[str] = None,
                          context: Dict[str, Any] = None) -> str:
        """Store a new memory entry"""
        
        # Generate memory ID
        memory_id = hashlib.md5(f"{agent}{memory_type.value}{time.time()}".encode()).hexdigest()[:12]
        
        # Create memory entry
        memory = MemoryEntry(
            id=memory_id,
            agent=agent,
            memory_type=memory_type,
            importance=importance,
            content=content,
            tags=tags or [],
            context=context,
            created_at=time.time(),
            last_accessed=time.time()
        )
        
        # Generate embedding for semantic search
        if memory_type in [MemoryType.SEMANTIC, MemoryType.EPISODIC]:
            memory.embedding = await self._generate_embedding(content)
        
        # Store in InfluxDB
        await self._write_memory_to_influx(memory)
        
        # Update linked memories
        await self._update_memory_links(memory)
        
        return memory_id
    
    async def retrieve_memories(self, query: MemoryQuery) -> List[MemoryEntry]:
        """Retrieve memories based on query"""
        
        memories = []
        
        # Query each relevant memory type
        memory_types = query.memory_types or list(MemoryType)
        
        for memory_type in memory_types:
            bucket = self.memory_buckets[memory_type]
            type_memories = await self._query_memories_from_influx(query, bucket, memory_type)
            memories.extend(type_memories)
        
        # Apply semantic similarity if requested
        if query.semantic_similarity and query.query_text:
            memories = await self._apply_semantic_filtering(memories, query.query_text)
        
        # Update access statistics
        for memory in memories:
            await self._update_memory_access(memory)
        
        # Sort by relevance and importance
        memories = self._rank_memories(memories, query)
        
        # Apply limit
        return memories[:query.limit]
    
    async def get_memory_by_id(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve specific memory by ID"""
        
        # Search across all buckets
        for memory_type, bucket in self.memory_buckets.items():
            memory = await self._get_memory_from_bucket(memory_id, bucket, memory_type)
            if memory:
                await self._update_memory_access(memory)
                return memory
        
        return None
    
    async def update_memory(self, memory_id: str, 
                           content: Dict[str, Any] = None,
                           tags: List[str] = None,
                           importance: MemoryImportance = None) -> bool:
        """Update existing memory entry"""
        
        memory = await self.get_memory_by_id(memory_id)
        if not memory:
            return False
        
        # Update fields
        if content is not None:
            memory.content.update(content)
            # Regenerate embedding if semantic memory
            if memory.memory_type in [MemoryType.SEMANTIC, MemoryType.EPISODIC]:
                memory.embedding = await self._generate_embedding(memory.content)
        
        if tags is not None:
            memory.tags = tags
        
        if importance is not None:
            memory.importance = importance
        
        memory.last_accessed = time.time()
        
        # Write updated memory
        await self._write_memory_to_influx(memory)
        
        return True
    
    async def link_memories(self, memory_id1: str, memory_id2: str, 
                           relationship: str = "related") -> bool:
        """Create bidirectional link between memories"""
        
        memory1 = await self.get_memory_by_id(memory_id1)
        memory2 = await self.get_memory_by_id(memory_id2)
        
        if not memory1 or not memory2:
            return False
        
        # Add links
        if memory_id2 not in memory1.linked_memories:
            memory1.linked_memories.append(memory_id2)
        if memory_id1 not in memory2.linked_memories:
            memory2.linked_memories.append(memory_id1)
        
        # Update both memories
        await self._write_memory_to_influx(memory1)
        await self._write_memory_to_influx(memory2)
        
        # Log the relationship
        await self._log_memory_relationship(memory_id1, memory_id2, relationship)
        
        return True
    
    async def consolidate_memories(self, agent: str = None) -> Dict[str, int]:
        """Consolidate and optimize memories"""
        
        consolidation_stats = {
            "processed": 0,
            "consolidated": 0,
            "archived": 0,
            "expired": 0
        }
        
        # Process each memory type
        for memory_type in MemoryType:
            bucket = self.memory_buckets[memory_type]
            memories = await self._get_all_memories_from_bucket(bucket, memory_type, agent)
            
            for memory in memories:
                consolidation_stats["processed"] += 1
                
                # Update decay score
                memory.decay_score = self.consolidator.calculate_decay_score(memory)
                
                # Archive very old, low-importance memories
                if memory.decay_score < 0.1 and memory.importance.value < 5:
                    await self._archive_memory(memory)
                    consolidation_stats["archived"] += 1
                
                # Expire temporary memories
                elif memory.importance == MemoryImportance.TEMPORARY and memory.decay_score < 0.5:
                    await self._expire_memory(memory)
                    consolidation_stats["expired"] += 1
                
                # Consolidate similar memories
                elif memory.memory_type == MemoryType.EPISODIC:
                    consolidated = await self._try_consolidate_memory(memory)
                    if consolidated:
                        consolidation_stats["consolidated"] += 1
                
                # Update memory with new scores
                await self._write_memory_to_influx(memory)
        
        return consolidation_stats
    
    async def get_memory_statistics(self, agent: str = None) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        
        stats = {
            "total_memories": 0,
            "by_type": {},
            "by_importance": {},
            "by_agent": {},
            "decay_distribution": {},
            "storage_efficiency": 0.0,
            "oldest_memory": None,
            "newest_memory": None
        }
        
        for memory_type in MemoryType:
            bucket = self.memory_buckets[memory_type]
            memories = await self._get_all_memories_from_bucket(bucket, memory_type, agent)
            
            type_count = len(memories)
            stats["by_type"][memory_type.value] = type_count
            stats["total_memories"] += type_count
            
            for memory in memories:
                # By importance
                imp_key = memory.importance.name
                stats["by_importance"][imp_key] = stats["by_importance"].get(imp_key, 0) + 1
                
                # By agent
                stats["by_agent"][memory.agent] = stats["by_agent"].get(memory.agent, 0) + 1
                
                # Decay distribution
                decay_bucket = f"{int(memory.decay_score * 10) * 10}%"
                stats["decay_distribution"][decay_bucket] = stats["decay_distribution"].get(decay_bucket, 0) + 1
                
                # Oldest/newest
                if not stats["oldest_memory"] or memory.created_at < stats["oldest_memory"]:
                    stats["oldest_memory"] = memory.created_at
                if not stats["newest_memory"] or memory.created_at > stats["newest_memory"]:
                    stats["newest_memory"] = memory.created_at
        
        # Calculate storage efficiency
        if stats["total_memories"] > 0:
            active_memories = sum(1 for decay in stats["decay_distribution"] 
                                if int(decay.replace('%', '')) >= 50)
            stats["storage_efficiency"] = active_memories / stats["total_memories"]
        
        return stats
    
    async def _write_memory_to_influx(self, memory: MemoryEntry):
        """Write memory entry to InfluxDB"""
        if not self.write_api:
            return
        
        bucket = self.memory_buckets[memory.memory_type]
        
        # Create InfluxDB point
        point = Point("persistent_memory") \
            .tag("memory_id", memory.id) \
            .tag("agent", memory.agent) \
            .tag("memory_type", memory.memory_type.value) \
            .tag("importance", memory.importance.name) \
            .field("content", json.dumps(memory.content)) \
            .field("tags", ",".join(memory.tags)) \
            .field("context", json.dumps(memory.context or {})) \
            .field("embedding", json.dumps(memory.embedding or [])) \
            .field("created_at", memory.created_at) \
            .field("last_accessed", memory.last_accessed) \
            .field("access_count", memory.access_count) \
            .field("decay_score", memory.decay_score) \
            .field("linked_memories", ",".join(memory.linked_memories)) \
            .time(datetime.utcnow())
        
        try:
            self.write_api.write(bucket=bucket, record=point)
        except Exception as e:
            print(f"❌ Error writing memory to InfluxDB: {e}")
    
    async def _query_memories_from_influx(self, query: MemoryQuery, bucket: str, 
                                        memory_type: MemoryType) -> List[MemoryEntry]:
        """Query memories from InfluxDB bucket"""
        if not self.query_api:
            return []
        
        # Build Flux query
        flux_query = f'''
        from(bucket: "{bucket}")
          |> range(start: -30d)
          |> filter(fn: (r) => r._measurement == "persistent_memory")
          |> filter(fn: (r) => r.agent == "{query.agent}")
        '''
        
        if query.tags:
            # Add tag filtering (simplified - would need more sophisticated tag matching)
            tag_filter = " or ".join([f'contains(value: r.tags, substring: "{tag}")' 
                                    for tag in query.tags])
            flux_query += f'|> filter(fn: (r) => {tag_filter})'
        
        flux_query += '''
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> sort(columns: ["_time"], desc: true)
        '''
        
        try:
            tables = self.query_api.query(flux_query)
            memories = []
            
            for table in tables:
                for record in table.records:
                    memory = self._convert_record_to_memory(record, memory_type)
                    if memory and self._matches_query(memory, query):
                        memories.append(memory)
            
            return memories
            
        except Exception as e:
            print(f"❌ Error querying memories: {e}")
            return []
    
    def _convert_record_to_memory(self, record, memory_type: MemoryType) -> Optional[MemoryEntry]:
        """Convert InfluxDB record to MemoryEntry"""
        try:
            return MemoryEntry(
                id=record.values.get("memory_id", ""),
                agent=record.values.get("agent", ""),
                memory_type=memory_type,
                importance=MemoryImportance[record.values.get("importance", "MEDIUM")],
                content=json.loads(record.values.get("content", "{}")),
                tags=record.values.get("tags", "").split(",") if record.values.get("tags") else [],
                context=json.loads(record.values.get("context", "{}")),
                embedding=json.loads(record.values.get("embedding", "[]")),
                created_at=record.values.get("created_at", 0.0),
                last_accessed=record.values.get("last_accessed", 0.0),
                access_count=record.values.get("access_count", 0),
                decay_score=record.values.get("decay_score", 1.0),
                linked_memories=record.values.get("linked_memories", "").split(",") if record.values.get("linked_memories") else []
            )
        except Exception as e:
            print(f"❌ Error converting record to memory: {e}")
            return None
    
    def _matches_query(self, memory: MemoryEntry, query: MemoryQuery) -> bool:
        """Check if memory matches query criteria"""
        
        # Importance threshold
        if memory.importance.value < query.importance_threshold:
            return False
        
        # Decay score check
        if query.include_decay and memory.decay_score < 0.1:
            return False
        
        # Text matching (simple keyword search)
        if query.query_text:
            query_lower = query.query_text.lower()
            content_text = json.dumps(memory.content).lower()
            if query_lower not in content_text:
                return False
        
        return True
    
    async def _generate_embedding(self, content: Dict[str, Any]) -> List[float]:
        """Generate simple embedding for content (placeholder for real embedding)"""
        # This is a simplified embedding - in production, use proper embedding models
        content_str = json.dumps(content)
        
        # Create a simple hash-based embedding
        embedding = []
        for i in range(0, min(len(content_str), 384), 8):
            chunk = content_str[i:i+8]
            chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
            embedding.append(int(chunk_hash[:8], 16) / (16**8))
        
        # Pad to fixed size
        while len(embedding) < 48:
            embedding.append(0.0)
        
        return embedding[:48]
    
    async def _apply_semantic_filtering(self, memories: List[MemoryEntry], 
                                      query_text: str) -> List[MemoryEntry]:
        """Apply semantic similarity filtering"""
        if not query_text:
            return memories
        
        # Generate query embedding
        query_embedding = await self._generate_embedding({"query": query_text})
        
        # Calculate similarities
        scored_memories = []
        for memory in memories:
            if memory.embedding:
                similarity = self._cosine_similarity(query_embedding, memory.embedding)
                scored_memories.append((similarity, memory))
        
        # Sort by similarity and return top matches
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scored_memories]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            v1 = np.array(vec1[:min(len(vec1), len(vec2))])
            v2 = np.array(vec2[:min(len(vec1), len(vec2))])
            
            dot_product = np.dot(v1, v2)
            norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
            
            if norm_product == 0:
                return 0.0
            
            return float(dot_product / norm_product)
        except:
            return 0.0
    
    def _rank_memories(self, memories: List[MemoryEntry], query: MemoryQuery) -> List[MemoryEntry]:
        """Rank memories by relevance and importance"""
        
        def score_memory(memory: MemoryEntry) -> float:
            # Base importance score
            importance_score = memory.importance.value / 10.0
            
            # Decay penalty
            decay_score = memory.decay_score
            
            # Access frequency bonus
            access_score = min(1.0, memory.access_count / 10.0)
            
            # Recency bonus
            hours_since_access = (time.time() - memory.last_accessed) / 3600
            recency_score = 1.0 / (1 + hours_since_access / 24)
            
            # Combined score
            total_score = (
                importance_score * 0.4 +
                decay_score * 0.3 +
                access_score * 0.2 +
                recency_score * 0.1
            )
            
            return total_score
        
        # Sort by score
        memories.sort(key=score_memory, reverse=True)
        return memories
    
    async def _update_memory_access(self, memory: MemoryEntry):
        """Update memory access statistics"""
        memory.last_accessed = time.time()
        memory.access_count += 1
        
        # Write back to InfluxDB
        await self._write_memory_to_influx(memory)
    
    async def _background_consolidation(self):
        """Background task for memory consolidation"""
        while self._running:
            try:
                # Run consolidation every hour
                await asyncio.sleep(3600)
                
                if self._running:
                    stats = await self.consolidate_memories()
                    print(f"📊 Memory consolidation: {stats}")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ Error in background consolidation: {e}")
    
    async def _get_memory_from_bucket(self, memory_id: str, bucket: str, 
                                    memory_type: MemoryType) -> Optional[MemoryEntry]:
        """Get specific memory from bucket"""
        if not self.query_api:
            return None
        
        query = f'''
        from(bucket: "{bucket}")
          |> range(start: -30d)
          |> filter(fn: (r) => r._measurement == "persistent_memory")
          |> filter(fn: (r) => r.memory_id == "{memory_id}")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> last()
        '''
        
        try:
            tables = self.query_api.query(query)
            
            for table in tables:
                for record in table.records:
                    return self._convert_record_to_memory(record, memory_type)
            
            return None
            
        except Exception as e:
            print(f"❌ Error getting memory by ID: {e}")
            return None
    
    async def _get_all_memories_from_bucket(self, bucket: str, memory_type: MemoryType, 
                                          agent: str = None) -> List[MemoryEntry]:
        """Get all memories from bucket for consolidation"""
        if not self.query_api:
            return []
        
        query = f'''
        from(bucket: "{bucket}")
          |> range(start: -30d)
          |> filter(fn: (r) => r._measurement == "persistent_memory")
        '''
        
        if agent:
            query += f'|> filter(fn: (r) => r.agent == "{agent}")'
        
        query += '''
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        
        try:
            tables = self.query_api.query(query)
            memories = []
            
            for table in tables:
                for record in table.records:
                    memory = self._convert_record_to_memory(record, memory_type)
                    if memory:
                        memories.append(memory)
            
            return memories
            
        except Exception as e:
            print(f"❌ Error getting all memories: {e}")
            return []
    
    async def _update_memory_links(self, memory: MemoryEntry):
        """Update memory links based on content similarity"""
        # This is a simplified implementation
        # In production, you'd use more sophisticated similarity matching
        pass
    
    async def _log_memory_relationship(self, memory_id1: str, memory_id2: str, relationship: str):
        """Log relationship between memories"""
        if not self.write_api:
            return
        
        point = Point("memory_relationships") \
            .tag("memory_1", memory_id1) \
            .tag("memory_2", memory_id2) \
            .tag("relationship", relationship) \
            .field("created", True) \
            .time(datetime.utcnow())
        
        try:
            self.write_api.write(bucket="relationships", record=point)
        except Exception as e:
            print(f"❌ Error logging memory relationship: {e}")
    
    async def _archive_memory(self, memory: MemoryEntry):
        """Archive old memory to long-term storage"""
        # Move to archive bucket
        archive_bucket = f"archive_{self.memory_buckets[memory.memory_type]}"
        
        point = Point("archived_memory") \
            .tag("original_id", memory.id) \
            .tag("agent", memory.agent) \
            .tag("memory_type", memory.memory_type.value) \
            .field("archived_content", json.dumps(asdict(memory))) \
            .time(datetime.utcnow())
        
        if self.write_api:
            try:
                self.write_api.write(bucket=archive_bucket, record=point)
            except Exception as e:
                print(f"❌ Error archiving memory: {e}")
    
    async def _expire_memory(self, memory: MemoryEntry):
        """Expire temporary memory"""
        # Log expiration
        point = Point("expired_memory") \
            .tag("memory_id", memory.id) \
            .tag("agent", memory.agent) \
            .tag("reason", "temporary_expired") \
            .field("expired", True) \
            .time(datetime.utcnow())
        
        if self.write_api:
            try:
                self.write_api.write(bucket="expired", record=point)
            except Exception as e:
                print(f"❌ Error logging memory expiration: {e}")
    
    async def _try_consolidate_memory(self, memory: MemoryEntry) -> bool:
        """Try to consolidate similar episodic memories"""
        # This is a placeholder for memory consolidation logic
        # In production, you'd implement sophisticated pattern matching
        return False

# Global persistent memory instance
_persistent_memory = None

def get_persistent_memory() -> PersistentMemorySystem:
    """Get global persistent memory instance"""
    global _persistent_memory
    if _persistent_memory is None:
        _persistent_memory = PersistentMemorySystem()
    return _persistent_memory

# Convenience functions for agent usage
async def store_agent_memory(agent: str, memory_type: str, content: Dict[str, Any], 
                           importance: str = "medium", tags: List[str] = None) -> str:
    """Store agent memory with simplified interface"""
    pm = get_persistent_memory()
    
    mem_type = MemoryType(memory_type.lower())
    mem_importance = MemoryImportance[importance.upper()]
    
    return await pm.store_memory(agent, mem_type, content, mem_importance, tags)

async def recall_agent_memories(agent: str, query_text: str = None, 
                              memory_types: List[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
    """Recall agent memories with simplified interface"""
    pm = get_persistent_memory()
    
    types = None
    if memory_types:
        types = [MemoryType(t.lower()) for t in memory_types]
    
    query = MemoryQuery(
        agent=agent,
        query_text=query_text,
        memory_types=types,
        limit=limit
    )
    
    memories = await pm.retrieve_memories(query)
    return [asdict(memory) for memory in memories]

async def get_memory_stats(agent: str = None) -> Dict[str, Any]:
    """Get memory statistics"""
    pm = get_persistent_memory()
    return await pm.get_memory_statistics(agent)