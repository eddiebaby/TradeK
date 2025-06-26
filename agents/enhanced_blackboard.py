#!/usr/bin/env python3
"""
Enhanced Blackboard with Persistent Memory Integration
Seamless integration between InfluxDB blackboard and persistent memory system

This enhanced blackboard provides:
- Automatic memory persistence for all agent interactions
- Intelligent memory retrieval and context loading
- Cross-agent memory sharing and learning
- Memory-enhanced task execution
- Persistent agent state and context
"""

import asyncio
import time
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import asdict

try:
    from influx_blackboard import InfluxBlackboard, get_blackboard
    from persistent_memory import (
        PersistentMemorySystem, MemoryType, MemoryImportance, 
        MemoryQuery, get_persistent_memory
    )
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print("⚠️  Required dependencies not available")

class EnhancedBlackboard:
    """Enhanced blackboard with persistent memory integration"""
    
    def __init__(self):
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("Required dependencies not available")
        
        # Initialize core systems
        self.blackboard = get_blackboard()
        self.persistent_memory = get_persistent_memory()
        
        # Memory integration settings
        self.auto_memory_capture = True
        self.memory_context_window = 2  # hours
        self.max_context_memories = 10
        
        # Agent state tracking
        self.agent_states = {}
        self.active_sessions = {}
    
    async def start(self):
        """Start the enhanced blackboard system"""
        await self.persistent_memory.start()
        print("✅ Enhanced blackboard with persistent memory started")
    
    async def stop(self):
        """Stop the enhanced blackboard system"""
        await self.persistent_memory.stop()
        print("✅ Enhanced blackboard stopped")
    
    # Enhanced Task Management with Memory
    
    async def write_task_with_memory(self, agent: str, task_type: str, data: Any,
                                   priority: int = 1, dependencies: List[str] = None,
                                   memory_importance: MemoryImportance = MemoryImportance.MEDIUM) -> str:
        """Write task and automatically store in persistent memory"""
        
        # Write to blackboard
        task_id = await self.blackboard.write_task(agent, task_type, data, priority, dependencies)
        
        # Store in persistent memory if enabled
        if self.auto_memory_capture:
            memory_content = {
                "task_id": task_id,
                "task_type": task_type,
                "data": data,
                "priority": priority,
                "dependencies": dependencies or [],
                "timestamp": time.time()
            }
            
            await self.persistent_memory.store_memory(
                agent=agent,
                memory_type=MemoryType.WORKING,
                content=memory_content,
                importance=memory_importance,
                tags=["task", task_type, f"priority_{priority}"]
            )
        
        return task_id
    
    async def read_tasks_with_context(self, agent: str, status: str = None, 
                                    limit: int = 10, include_memory_context: bool = True) -> List[Dict[str, Any]]:
        """Read tasks with enhanced memory context"""
        
        # Get tasks from blackboard
        tasks = await self.blackboard.read_tasks(agent, status, limit)
        
        # Enhance with memory context
        if include_memory_context:
            for task in tasks:
                task["memory_context"] = await self._get_task_memory_context(agent, task)
        
        return tasks
    
    async def complete_task_with_memory(self, task_id: str, agent: str, 
                                      result: Any, lessons_learned: List[str] = None):
        """Complete task and store results in persistent memory"""
        
        # Update task status
        await self.blackboard.update_task_status(task_id, "completed", agent)
        
        # Store completion memory
        memory_content = {
            "task_id": task_id,
            "result": result,
            "completion_time": time.time(),
            "lessons_learned": lessons_learned or []
        }
        
        # Determine importance based on result complexity
        importance = MemoryImportance.MEDIUM
        if lessons_learned or isinstance(result, dict) and len(result) > 5:
            importance = MemoryImportance.HIGH
        
        await self.persistent_memory.store_memory(
            agent=agent,
            memory_type=MemoryType.EPISODIC,
            content=memory_content,
            importance=importance,
            tags=["task_completion", "result", task_id]
        )
    
    # Enhanced Agent Context Management
    
    async def load_agent_context(self, agent: str, context_hours: int = None) -> Dict[str, Any]:
        """Load comprehensive agent context including memory"""
        
        hours = context_hours or self.memory_context_window
        
        # Get blackboard context
        blackboard_context = await self.blackboard.get_agent_context(agent, hours)
        
        # Get memory context
        memory_query = MemoryQuery(
            agent=agent,
            memory_types=[MemoryType.WORKING, MemoryType.EPISODIC, MemoryType.SEMANTIC],
            limit=self.max_context_memories,
            time_range=(datetime.now() - timedelta(hours=hours), datetime.now())
        )
        
        memories = await self.persistent_memory.retrieve_memories(memory_query)
        
        # Get agent state
        agent_state = self.agent_states.get(agent, {})
        
        # Combine context
        enhanced_context = {
            "agent": agent,
            "blackboard_context": blackboard_context,
            "memory_context": {
                "recent_memories": [asdict(memory) for memory in memories],
                "memory_count": len(memories),
                "oldest_memory": min([m.created_at for m in memories]) if memories else None,
                "newest_memory": max([m.last_accessed for m in memories]) if memories else None
            },
            "agent_state": agent_state,
            "context_loaded_at": time.time(),
            "context_window_hours": hours
        }
        
        return enhanced_context
    
    async def save_agent_state(self, agent: str, state: Dict[str, Any], 
                             importance: MemoryImportance = MemoryImportance.MEDIUM):
        """Save agent state to persistent memory"""
        
        # Update local state
        self.agent_states[agent] = state
        
        # Store in persistent memory
        await self.persistent_memory.store_memory(
            agent=agent,
            memory_type=MemoryType.CONTEXTUAL,
            content={"agent_state": state, "saved_at": time.time()},
            importance=importance,
            tags=["agent_state", "context"]
        )
    
    async def start_agent_session(self, agent: str, session_context: Dict[str, Any] = None):
        """Start agent session with memory-enhanced context"""
        
        session_id = f"{agent}_{int(time.time())}"
        
        # Load agent context
        context = await self.load_agent_context(agent)
        
        # Store session start
        session_data = {
            "session_id": session_id,
            "started_at": time.time(),
            "context": session_context or {},
            "loaded_context": context
        }
        
        self.active_sessions[agent] = session_data
        
        # Store session memory
        await self.persistent_memory.store_memory(
            agent=agent,
            memory_type=MemoryType.EPISODIC,
            content=session_data,
            importance=MemoryImportance.MEDIUM,
            tags=["session_start", "context_load"]
        )
        
        return session_id, context
    
    async def end_agent_session(self, agent: str, session_summary: Dict[str, Any] = None):
        """End agent session and store session summary"""
        
        if agent not in self.active_sessions:
            return
        
        session_data = self.active_sessions[agent]
        session_data["ended_at"] = time.time()
        session_data["duration"] = session_data["ended_at"] - session_data["started_at"]
        session_data["summary"] = session_summary or {}
        
        # Store session end memory
        await self.persistent_memory.store_memory(
            agent=agent,
            memory_type=MemoryType.EPISODIC,
            content=session_data,
            importance=MemoryImportance.HIGH if session_data["duration"] > 3600 else MemoryImportance.MEDIUM,
            tags=["session_end", "session_summary"]
        )
        
        # Clean up
        del self.active_sessions[agent]
    
    # Enhanced Data Storage with Memory
    
    async def store_data_with_memory(self, key: str, data: Any, agent: str,
                                   bucket: str = "data", ttl: int = 3600,
                                   memory_type: MemoryType = MemoryType.WORKING,
                                   importance: MemoryImportance = MemoryImportance.MEDIUM):
        """Store data in blackboard and persistent memory"""
        
        # Store in blackboard
        await self.blackboard.write_data(key, data, bucket, ttl)
        
        # Store in persistent memory
        memory_content = {
            "key": key,
            "data": data,
            "bucket": bucket,
            "ttl": ttl,
            "stored_at": time.time()
        }
        
        await self.persistent_memory.store_memory(
            agent=agent,
            memory_type=memory_type,
            content=memory_content,
            importance=importance,
            tags=["data_storage", bucket, key]
        )
    
    async def retrieve_data_with_memory(self, key: str, agent: str, bucket: str = "data") -> Any:
        """Retrieve data from blackboard with memory fallback"""
        
        # Try blackboard first
        data = await self.blackboard.read_data(key, bucket)
        
        if data is not None:
            return data
        
        # Fallback to memory
        memory_query = MemoryQuery(
            agent=agent,
            query_text=key,
            tags=["data_storage", bucket],
            limit=1
        )
        
        memories = await self.persistent_memory.retrieve_memories(memory_query)
        
        if memories:
            memory = memories[0]
            return memory.content.get("data")
        
        return None
    
    # Enhanced Reflection and Learning
    
    async def write_reflection_with_memory(self, agent: str, category: str, severity: str,
                                         note: str, action: str, impact_score: float,
                                         related_memories: List[str] = None):
        """Write reflection with memory linking"""
        
        # Write to blackboard
        await self.blackboard.write_reflection(agent, category, severity, note, action, impact_score)
        
        # Store detailed reflection in memory
        reflection_content = {
            "category": category,
            "severity": severity,
            "note": note,
            "action": action,
            "impact_score": impact_score,
            "related_memories": related_memories or [],
            "reflection_time": time.time()
        }
        
        # Determine importance based on impact score
        importance = MemoryImportance.LOW
        if impact_score >= 0.8:
            importance = MemoryImportance.HIGH
        elif impact_score >= 0.5:
            importance = MemoryImportance.MEDIUM
        
        memory_id = await self.persistent_memory.store_memory(
            agent=agent,
            memory_type=MemoryType.SEMANTIC,
            content=reflection_content,
            importance=importance,
            tags=["reflection", category, severity]
        )
        
        # Link to related memories
        if related_memories:
            for related_id in related_memories:
                await self.persistent_memory.link_memories(memory_id, related_id, "reflection")
    
    async def get_learning_patterns(self, agent: str, days: int = 7) -> Dict[str, Any]:
        """Analyze learning patterns from memory"""
        
        # Get reflections and completed tasks
        memory_query = MemoryQuery(
            agent=agent,
            tags=["reflection", "task_completion"],
            time_range=(datetime.now() - timedelta(days=days), datetime.now()),
            limit=100
        )
        
        memories = await self.persistent_memory.retrieve_memories(memory_query)
        
        # Analyze patterns
        patterns = {
            "reflection_count": 0,
            "task_completion_count": 0,
            "improvement_areas": {},
            "success_patterns": {},
            "learning_velocity": 0.0,
            "knowledge_areas": {}
        }
        
        for memory in memories:
            if "reflection" in memory.tags:
                patterns["reflection_count"] += 1
                category = memory.content.get("category", "unknown")
                patterns["improvement_areas"][category] = patterns["improvement_areas"].get(category, 0) + 1
            
            if "task_completion" in memory.tags:
                patterns["task_completion_count"] += 1
                if memory.content.get("lessons_learned"):
                    patterns["learning_velocity"] += 1
        
        # Calculate learning velocity (learnings per day)
        if days > 0:
            patterns["learning_velocity"] = patterns["learning_velocity"] / days
        
        return patterns
    
    # Memory-Enhanced Agent Handoffs
    
    async def prepare_agent_handoff(self, from_agent: str, to_agent: str, 
                                  task_context: Dict[str, Any],
                                  shared_memories: List[str] = None) -> Dict[str, Any]:
        """Prepare memory-enhanced agent handoff"""
        
        # Get source agent context
        source_context = await self.load_agent_context(from_agent)
        
        # Create handoff memory
        handoff_content = {
            "from_agent": from_agent,
            "to_agent": to_agent,
            "task_context": task_context,
            "source_context": source_context,
            "shared_memories": shared_memories or [],
            "handoff_time": time.time()
        }
        
        # Store handoff memory for both agents
        handoff_id = await self.persistent_memory.store_memory(
            agent=from_agent,
            memory_type=MemoryType.EPISODIC,
            content=handoff_content,
            importance=MemoryImportance.HIGH,
            tags=["handoff", "transfer", to_agent]
        )
        
        await self.persistent_memory.store_memory(
            agent=to_agent,
            memory_type=MemoryType.EPISODIC,
            content=handoff_content,
            importance=MemoryImportance.HIGH,
            tags=["handoff", "received", from_agent]
        )
        
        # Share specified memories
        if shared_memories:
            for memory_id in shared_memories:
                source_memory = await self.persistent_memory.get_memory_by_id(memory_id)
                if source_memory:
                    # Create shared copy for target agent
                    shared_content = source_memory.content.copy()
                    shared_content["shared_from"] = from_agent
                    shared_content["original_memory_id"] = memory_id
                    
                    shared_id = await self.persistent_memory.store_memory(
                        agent=to_agent,
                        memory_type=source_memory.memory_type,
                        content=shared_content,
                        importance=source_memory.importance,
                        tags=source_memory.tags + ["shared_memory"]
                    )
                    
                    # Link memories
                    await self.persistent_memory.link_memories(memory_id, shared_id, "shared")
        
        return {
            "handoff_id": handoff_id,
            "source_context": source_context,
            "shared_memory_count": len(shared_memories or [])
        }
    
    # Utility Methods
    
    async def _get_task_memory_context(self, agent: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Get memory context for a specific task"""
        
        task_type = task.get("type", "")
        task_id = task.get("id", "")
        
        # Search for related memories
        memory_query = MemoryQuery(
            agent=agent,
            query_text=f"{task_type} {task_id}",
            memory_types=[MemoryType.WORKING, MemoryType.EPISODIC],
            limit=5
        )
        
        memories = await self.persistent_memory.retrieve_memories(memory_query)
        
        return {
            "related_memories": [asdict(memory) for memory in memories],
            "memory_count": len(memories),
            "context_relevance": len(memories) > 0
        }
    
    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        # Get blackboard stats
        blackboard_report = await self.blackboard.generate_efficiency_report(24)
        
        # Get memory stats
        memory_stats = await self.persistent_memory.get_memory_statistics()
        
        # Get agent session stats
        session_stats = {
            "active_sessions": len(self.active_sessions),
            "agents_with_state": len(self.agent_states)
        }
        
        return {
            "blackboard": blackboard_report,
            "persistent_memory": memory_stats,
            "sessions": session_stats,
            "system_health": {
                "memory_enabled": self.auto_memory_capture,
                "context_window_hours": self.memory_context_window,
                "max_context_memories": self.max_context_memories
            }
        }

# Global enhanced blackboard instance
_enhanced_blackboard = None

def get_enhanced_blackboard() -> EnhancedBlackboard:
    """Get global enhanced blackboard instance"""
    global _enhanced_blackboard
    if _enhanced_blackboard is None:
        _enhanced_blackboard = EnhancedBlackboard()
    return _enhanced_blackboard

# Convenience functions for agent usage
async def start_memory_session(agent: str, context: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
    """Start memory-enhanced agent session"""
    ebb = get_enhanced_blackboard()
    return await ebb.start_agent_session(agent, context)

async def end_memory_session(agent: str, summary: Dict[str, Any] = None):
    """End memory-enhanced agent session"""
    ebb = get_enhanced_blackboard()
    await ebb.end_agent_session(agent, summary)

async def store_agent_experience(agent: str, experience_type: str, data: Dict[str, Any],
                               importance: str = "medium", tags: List[str] = None) -> str:
    """Store agent experience in persistent memory"""
    ebb = get_enhanced_blackboard()
    
    importance_map = {
        "critical": MemoryImportance.CRITICAL,
        "high": MemoryImportance.HIGH,
        "medium": MemoryImportance.MEDIUM,
        "low": MemoryImportance.LOW,
        "temporary": MemoryImportance.TEMPORARY
    }
    
    memory_type = MemoryType.EPISODIC
    if experience_type in ["skill", "process", "procedure"]:
        memory_type = MemoryType.PROCEDURAL
    elif experience_type in ["fact", "knowledge", "concept"]:
        memory_type = MemoryType.SEMANTIC
    
    return await ebb.persistent_memory.store_memory(
        agent=agent,
        memory_type=memory_type,
        content=data,
        importance=importance_map.get(importance, MemoryImportance.MEDIUM),
        tags=(tags or []) + [experience_type]
    )

async def recall_agent_experience(agent: str, query: str, experience_types: List[str] = None,
                                limit: int = 10) -> List[Dict[str, Any]]:
    """Recall agent experiences from persistent memory"""
    ebb = get_enhanced_blackboard()
    
    memory_query = MemoryQuery(
        agent=agent,
        query_text=query,
        tags=experience_types,
        limit=limit
    )
    
    memories = await ebb.persistent_memory.retrieve_memories(memory_query)
    return [asdict(memory) for memory in memories]