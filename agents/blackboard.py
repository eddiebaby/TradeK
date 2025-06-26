"""
Enhanced Agent Blackboard System with Token Optimization
Token-efficient inter-agent communication and self-reflection system
"""

import json
import hashlib
import asyncio
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
import zlib
import base64

# Blackboard file paths
BLACKBOARD_DIR = Path(__file__).parent
BLACKBOARD_FILE = BLACKBOARD_DIR / "blackboard.md"
DATA_CACHE_FILE = BLACKBOARD_DIR / "data_cache.json"

# Token optimization constants
KEY_MAP = {
    # Research domain
    "technical_analysis": "TA",
    "security_intelligence": "SI", 
    "performance_benchmarking": "PB",
    "best_practices": "BP",
    "trend_analysis": "TR",
    "market_intelligence": "MI",
    
    # Strategy domain
    "architectural_analysis": "AA",
    "quality_strategy": "QS",
    "risk_assessment": "RA",
    "strategic_analysis": "SA",
    "technology_fit": "TF",
    
    # Implementation domain
    "tdd_implementation": "TDD",
    "test_creation": "TC",
    "quality_validation": "QV",
    "deployment_pipeline": "DP",
    "monitoring_setup": "MS",
    
    # Context7 MCP
    "library_documentation": "LD",
    "api_reference": "AR",
    "code_examples": "CE",
    "implementation_patterns": "IP"
}

@dataclass
class BlackboardEntry:
    """Compressed blackboard entry for token efficiency"""
    id: str
    agent: str  # R, M, E (Researcher, Mastermind, Executor)
    type: str   # Abbreviated type
    data: str   # Compressed data reference or inline
    ts: float   # Timestamp
    deps: List[str] = None  # Dependencies
    status: str = "new"     # new, proc, done
    priority: int = 1       # 1=high, 2=med, 3=low

@dataclass
class AgentMetrics:
    """Token usage and performance metrics"""
    agent: str
    operation: str
    tokens_used: int
    exec_time: float
    success: bool
    timestamp: float

class TokenOptimizer:
    """Optimizes token usage across agent communications"""
    
    def __init__(self):
        self.operation_history = defaultdict(list)
        self.compression_ratios = {}
        
    def compress_data(self, data: Any) -> str:
        """Compress data for token efficiency"""
        if isinstance(data, dict):
            # Apply key mapping
            compressed = {}
            for key, value in data.items():
                short_key = KEY_MAP.get(key, key[:3])  # Use mapping or first 3 chars
                if isinstance(value, (dict, list)):
                    compressed[short_key] = self._compress_complex(value)
                else:
                    compressed[short_key] = value
            data = compressed
            
        # Serialize and compress
        json_str = json.dumps(data, separators=(',', ':'))
        compressed = zlib.compress(json_str.encode('utf-8'))
        encoded = base64.b64encode(compressed).decode('utf-8')
        
        # Calculate compression ratio
        original_size = len(json_str)
        compressed_size = len(encoded)
        ratio = compressed_size / original_size
        
        return encoded
    
    def decompress_data(self, compressed: str) -> Any:
        """Decompress data from blackboard"""
        try:
            decoded = base64.b64decode(compressed.encode('utf-8'))
            decompressed = zlib.decompress(decoded)
            return json.loads(decompressed.decode('utf-8'))
        except Exception:
            # Fallback for uncompressed data
            try:
                return json.loads(compressed)
            except:
                return compressed
    
    def _compress_complex(self, obj):
        """Compress complex nested objects"""
        if isinstance(obj, dict):
            return {KEY_MAP.get(k, k[:3]): v for k, v in obj.items()}
        elif isinstance(obj, list):
            return obj[:5]  # Limit list size for token efficiency
        return obj

class EnhancedBlackboard:
    """Token-optimized blackboard system for agent communication"""
    
    def __init__(self):
        self.optimizer = TokenOptimizer()
        self.data_cache = {}
        self.entries = []
        self.metrics = []
        self._load_blackboard()
        self._load_cache()
    
    def _load_blackboard(self):
        """Load existing blackboard entries"""
        if BLACKBOARD_FILE.exists():
            try:
                content = BLACKBOARD_FILE.read_text()
                # Parse markdown format entries
                self.entries = self._parse_markdown_entries(content)
            except Exception as e:
                print(f"Error loading blackboard: {e}")
                self.entries = []
    
    def _load_cache(self):
        """Load data cache"""
        if DATA_CACHE_FILE.exists():
            try:
                with open(DATA_CACHE_FILE, 'r') as f:
                    self.data_cache = json.load(f)
            except Exception:
                self.data_cache = {}
    
    def _save_blackboard(self):
        """Save blackboard in markdown format"""
        content = self._generate_markdown()
        BLACKBOARD_FILE.write_text(content)
        
        # Save data cache
        with open(DATA_CACHE_FILE, 'w') as f:
            json.dump(self.data_cache, f)
    
    def _generate_markdown(self) -> str:
        """Generate markdown format blackboard"""
        lines = [
            "# Agent Blackboard - Token Optimized Communication",
            "",
            f"**Last Updated**: {datetime.now().isoformat()}",
            f"**Active Entries**: {len([e for e in self.entries if e.status != 'done'])}",
            f"**Total Entries**: {len(self.entries)}",
            "",
            "## Current Tasks",
            ""
        ]
        
        # Group by agent
        by_agent = defaultdict(list)
        for entry in self.entries:
            if entry.status != 'done':
                by_agent[entry.agent].append(entry)
        
        for agent, entries in by_agent.items():
            agent_name = {"R": "🔍 RESEARCHER", "M": "🧠 MASTERMIND", "E": "⚡ EXECUTOR"}.get(agent, agent)
            lines.append(f"### {agent_name}")
            lines.append("")
            
            for entry in entries:
                status_icon = {"new": "🆕", "proc": "⏳", "done": "✅"}.get(entry.status, "❓")
                priority_icon = {1: "🔴", 2: "🟡", 3: "🟢"}.get(entry.priority, "⚪")
                
                lines.append(f"- {status_icon} {priority_icon} **{entry.id}** ({entry.type})")
                
                # Show compressed data reference or preview
                if entry.data.startswith("ref:"):
                    lines.append(f"  - Data: {entry.data}")
                else:
                    preview = entry.data[:50] + "..." if len(entry.data) > 50 else entry.data
                    lines.append(f"  - Preview: {preview}")
                
                lines.append(f"  - Timestamp: {datetime.fromtimestamp(entry.ts).strftime('%H:%M:%S')}")
                
                if entry.deps:
                    lines.append(f"  - Dependencies: {', '.join(entry.deps)}")
                
                lines.append("")
        
        # Recent metrics
        if self.metrics:
            lines.extend([
                "## Recent Performance",
                "",
            ])
            
            recent_metrics = sorted(self.metrics, key=lambda x: x.timestamp, reverse=True)[:10]
            for metric in recent_metrics:
                agent_name = {"R": "🔍", "M": "🧠", "E": "⚡"}.get(metric.agent, metric.agent)
                success_icon = "✅" if metric.success else "❌"
                lines.append(f"- {agent_name} {metric.operation}: {metric.tokens_used}t, {metric.exec_time:.2f}s {success_icon}")
        
        return "\n".join(lines)
    
    def _parse_markdown_entries(self, content: str) -> List[BlackboardEntry]:
        """Parse entries from markdown format"""
        # For now, return empty list - implement parsing if needed
        return []
    
    async def write_task(self, agent: str, task_type: str, data: Any, 
                        priority: int = 1, dependencies: List[str] = None) -> str:
        """Write a task to the blackboard with token optimization"""
        
        # Generate compressed ID
        task_id = hashlib.md5(f"{agent}{task_type}{time.time()}".encode()).hexdigest()[:8]
        
        # Compress data
        if isinstance(data, (dict, list)) and len(str(data)) > 100:
            # Store large data in cache with reference
            cache_key = f"data_{task_id}"
            compressed_data = self.optimizer.compress_data(data)
            self.data_cache[cache_key] = compressed_data
            data_ref = f"ref:{cache_key}"
        else:
            # Store small data inline
            data_ref = str(data)[:200]  # Limit inline data size
        
        # Create entry
        entry = BlackboardEntry(
            id=task_id,
            agent=agent[0].upper(),  # R, M, E
            type=KEY_MAP.get(task_type, task_type[:3]),
            data=data_ref,
            ts=time.time(),
            deps=dependencies or [],
            priority=priority
        )
        
        self.entries.append(entry)
        self._save_blackboard()
        
        return task_id
    
    async def read_tasks(self, agent: str, status: str = None) -> List[BlackboardEntry]:
        """Read tasks for specific agent"""
        agent_code = agent[0].upper()
        
        tasks = [e for e in self.entries if e.agent == agent_code]
        
        if status:
            tasks = [e for e in tasks if e.status == status]
        
        return tasks
    
    async def update_task_status(self, task_id: str, status: str):
        """Update task status"""
        for entry in self.entries:
            if entry.id == task_id:
                entry.status = status
                break
        
        self._save_blackboard()
    
    async def get_task_data(self, task_id: str) -> Any:
        """Get full task data, decompressing if needed"""
        entry = next((e for e in self.entries if e.id == task_id), None)
        if not entry:
            return None
        
        if entry.data.startswith("ref:"):
            cache_key = entry.data[4:]  # Remove "ref:" prefix
            compressed_data = self.data_cache.get(cache_key)
            if compressed_data:
                return self.optimizer.decompress_data(compressed_data)
        
        return entry.data
    
    async def write_data(self, key: str, data: Any, ttl: int = 3600):
        """Write data with TTL and compression"""
        compressed = self.optimizer.compress_data(data)
        self.data_cache[key] = {
            "data": compressed,
            "expires": time.time() + ttl
        }
        self._save_blackboard()
    
    async def read_data(self, key: str) -> Any:
        """Read data with expiration check"""
        cached = self.data_cache.get(key)
        if not cached:
            return None
        
        if cached.get("expires", 0) < time.time():
            del self.data_cache[key]
            return None
        
        data = cached.get("data", cached)  # Backward compatibility
        return self.optimizer.decompress_data(data)
    
    async def log_metrics(self, agent: str, operation: str, tokens_used: int, 
                         exec_time: float, success: bool = True):
        """Log performance metrics"""
        metric = AgentMetrics(
            agent=agent[0].upper(),
            operation=operation,
            tokens_used=tokens_used,
            exec_time=exec_time,
            success=success,
            timestamp=time.time()
        )
        
        self.metrics.append(metric)
        
        # Keep only recent metrics (last 24 hours)
        cutoff = time.time() - 86400
        self.metrics = [m for m in self.metrics if m.timestamp > cutoff]
        
        self._save_blackboard()
    
    async def get_agent_context(self, agent: str, lookback_hours: int = 2) -> Dict[str, Any]:
        """Get compressed context for agent handoffs"""
        agent_code = agent[0].upper()
        cutoff = time.time() - (lookback_hours * 3600)
        
        # Recent tasks
        recent_tasks = [
            e for e in self.entries 
            if e.agent == agent_code and e.ts > cutoff
        ]
        
        # Recent metrics
        recent_metrics = [
            m for m in self.metrics 
            if m.agent == agent_code and m.timestamp > cutoff
        ]
        
        # Compressed context
        context = {
            "tasks": len(recent_tasks),
            "completed": len([t for t in recent_tasks if t.status == "done"]),
            "avg_tokens": sum(m.tokens_used for m in recent_metrics) / max(len(recent_metrics), 1),
            "success_rate": sum(m.success for m in recent_metrics) / max(len(recent_metrics), 1),
            "last_activity": max([t.ts for t in recent_tasks], default=0)
        }
        
        return context

# Global blackboard instance
blackboard = EnhancedBlackboard()

# Convenience functions
async def write_task(agent: str, task_type: str, data: Any, priority: int = 1) -> str:
    """Write task to blackboard"""
    return await blackboard.write_task(agent, task_type, data, priority)

async def read_tasks(agent: str, status: str = None) -> List[BlackboardEntry]:
    """Read tasks from blackboard"""
    return await blackboard.read_tasks(agent, status)

async def update_status(task_id: str, status: str):
    """Update task status"""
    await blackboard.update_task_status(task_id, status)

async def get_data(task_id: str) -> Any:
    """Get task data"""
    return await blackboard.get_task_data(task_id)

async def log_performance(agent: str, operation: str, tokens: int, time_sec: float, success: bool = True):
    """Log performance metrics"""
    await blackboard.log_metrics(agent, operation, tokens, time_sec, success)

async def get_context(agent: str) -> Dict[str, Any]:
    """Get agent context for handoffs"""
    return await blackboard.get_agent_context(agent)