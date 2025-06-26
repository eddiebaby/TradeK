#!/usr/bin/env python3
"""
Enhanced Agent Blackboard System with InfluxDB 2.7
Token-Optimized Inter-Agent Communication & Self-Reflection System

This implementation provides:
- Token-first design for minimal usage
- Self-improving agent communication
- Time-series native data storage
- Expandable agent architecture
"""

import os
import json
import hashlib
import asyncio
import time
import zlib
import base64
import yaml
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np

try:
    from influxdb_client import InfluxDBClient, Point, WriteOptions
    from influxdb_client.client.write_api import SYNCHRONOUS
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False
    print("⚠️  InfluxDB client not available. Run: pip install influxdb-client")

# Configuration
CONFIG_PATH = Path(__file__).parent / "config" / "blackboard_influx.yaml"

# Token optimization key mappings
KEY_MAP = {
    # Common operations
    "technical_analysis": "TA",
    "security_intelligence": "SI", 
    "performance_benchmarking": "PB",
    "best_practices": "BP",
    "trend_analysis": "TR",
    "market_intelligence": "MI",
    "architectural_analysis": "AA",
    "quality_strategy": "QS",
    "risk_assessment": "RA",
    "strategic_analysis": "SA",
    "technology_fit": "TF",
    "tdd_implementation": "TDD",
    "test_creation": "TC",
    "quality_validation": "QV",
    "deployment_pipeline": "DP",
    "monitoring_setup": "MS",
    "library_documentation": "LD",
    "api_reference": "AR",
    "code_examples": "CE",
    "implementation_patterns": "IP",
    
    # Data types
    "SPX_OPTION_CHAIN": "SOC",
    "VOLATILITY_SURFACE": "VS",
    "MARKET_SENTIMENT": "MS",
    "TRADING_SIGNALS": "TS",
}

@dataclass
class TokenMetric:
    """Token usage metric for optimization"""
    agent: str
    operation: str
    tokens_used: int
    exec_time: float
    success: bool
    timestamp: float
    data_size: int = 0
    compression_ratio: float = 1.0

@dataclass
class ReflectionEntry:
    """Agent reflection for self-improvement"""
    agent: str
    category: str  # performance, optimization, error, pattern
    severity: str  # low, medium, high
    note: str
    action: str
    impact_score: float
    timestamp: float

@dataclass
class OptimizationSuggestion:
    """Optimization suggestion from analysis"""
    target_agent: str
    category: str
    suggestion: str
    expected_savings: int  # tokens
    confidence: float
    auto_approve: bool = False
    implemented: bool = False

class TokenOptimizer:
    """Advanced token optimization with compression and caching"""
    
    def __init__(self):
        self.operation_history = defaultdict(list)
        self.compression_ratios = {}
        self.cache_hit_rates = defaultdict(float)
        
    def compress_data(self, data: Any) -> tuple[str, float]:
        """Compress data and return (compressed_str, compression_ratio)"""
        if isinstance(data, dict):
            # Apply key mapping for better compression
            compressed = {}
            for key, value in data.items():
                short_key = KEY_MAP.get(key, key[:3] if len(key) > 3 else key)
                if isinstance(value, (dict, list)):
                    compressed[short_key] = self._compress_complex(value)
                else:
                    compressed[short_key] = value
            data = compressed
        
        # Serialize and compress
        json_str = json.dumps(data, separators=(',', ':'))
        original_size = len(json_str)
        
        if original_size > 50:  # Only compress if worthwhile
            compressed_bytes = zlib.compress(json_str.encode('utf-8'), level=9)
            encoded = base64.b64encode(compressed_bytes).decode('utf-8')
            compression_ratio = len(encoded) / original_size
        else:
            encoded = json_str
            compression_ratio = 1.0
        
        return encoded, compression_ratio
    
    def decompress_data(self, compressed: str) -> Any:
        """Decompress data from storage"""
        try:
            # Try base64 decode first (compressed)
            decoded = base64.b64decode(compressed.encode('utf-8'))
            decompressed = zlib.decompress(decoded)
            return json.loads(decompressed.decode('utf-8'))
        except:
            # Fallback to direct JSON (uncompressed)
            try:
                return json.loads(compressed)
            except:
                return compressed
    
    def _compress_complex(self, obj):
        """Compress complex nested objects"""
        if isinstance(obj, dict):
            # Limit dict size and apply key mapping
            items = list(obj.items())[:10]  # Limit to 10 items
            return {KEY_MAP.get(k, k[:3]): v for k, v in items}
        elif isinstance(obj, list):
            return obj[:5]  # Limit to 5 items
        return obj
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        return max(1, len(text) // 4)  # Rough estimate: 4 chars per token
    
    def analyze_operation_efficiency(self, operation: str) -> Dict[str, Any]:
        """Analyze operation efficiency and suggest optimizations"""
        history = self.operation_history.get(operation, [])
        if len(history) < 3:
            return {"status": "insufficient_data"}
        
        # Calculate metrics
        token_usage = [h["tokens"] for h in history[-10:]]  # Last 10 operations
        exec_times = [h["exec_time"] for h in history[-10:]]
        data_sizes = [h.get("data_size", 0) for h in history[-10:]]
        
        avg_tokens = np.mean(token_usage)
        avg_time = np.mean(exec_times)
        token_efficiency = avg_tokens / max(np.mean(data_sizes), 1)
        
        suggestions = []
        
        # High token usage
        if avg_tokens > 500:
            suggestions.append({
                "type": "compression",
                "message": f"High token usage ({avg_tokens:.0f}). Consider data compression.",
                "expected_savings": int(avg_tokens * 0.3)
            })
        
        # Poor token efficiency
        if token_efficiency > 0.1:
            suggestions.append({
                "type": "optimization",
                "message": f"Poor token efficiency ({token_efficiency:.3f}). Consider abbreviated keys.",
                "expected_savings": int(avg_tokens * 0.2)
            })
        
        # Slow execution
        if avg_time > 2.0:
            suggestions.append({
                "type": "performance",
                "message": f"Slow execution ({avg_time:.2f}s). Consider caching.",
                "expected_savings": 0  # Time, not tokens
            })
        
        return {
            "status": "analyzed",
            "avg_tokens": avg_tokens,
            "avg_time": avg_time,
            "efficiency": token_efficiency,
            "suggestions": suggestions
        }

class InfluxBlackboard:
    """Token-optimized InfluxDB blackboard for agent communication"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or CONFIG_PATH
        self.config = self._load_config()
        self.optimizer = TokenOptimizer()
        self.client = None
        self.write_api = None
        self.query_api = None
        
        if INFLUXDB_AVAILABLE:
            self._initialize_client()
        else:
            print("⚠️  InfluxDB not available - running in fallback mode")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            # Return default config
            return {
                'influxdb': {
                    'url': 'http://localhost:8087',
                    'org': 'AgentBlackboard',
                    'bucket': 'blackboard',
                    'token_file': str(Path(__file__).parent / 'config' / '.influx_token')
                },
                'token_optimization': {
                    'compression_threshold': 100,
                    'cache_ttl': 3600,
                    'max_inline_data': 200,
                    'batch_size': 10
                },
                'monitoring': {
                    'high_token_usage': 1000,
                    'slow_execution': 5.0,
                    'error_threshold': 3,
                    'low_confidence': 0.7
                }
            }
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _initialize_client(self):
        """Initialize InfluxDB client"""
        try:
            influx_config = self.config['influxdb']
            token_file = Path(influx_config['token_file'])
            
            if token_file.exists():
                token = token_file.read_text().strip()
            else:
                print(f"⚠️  Token file not found: {token_file}")
                print("   Run setup script: python scripts/setup_blackboard_influxdb.py")
                return
            
            self.client = InfluxDBClient(
                url=influx_config['url'],
                token=token,
                org=influx_config['org']
            )
            
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            self.query_api = self.client.query_api()
            
            # Test connection
            health = self.client.health()
            if health.status == "pass":
                print(f"✅ Connected to InfluxDB at {influx_config['url']}")
            else:
                print(f"⚠️  InfluxDB health check failed: {health.message}")
                
        except Exception as e:
            print(f"❌ Failed to connect to InfluxDB: {e}")
            self.client = None
    
    async def write_task(self, agent: str, task_type: str, data: Any, 
                        priority: int = 1, dependencies: List[str] = None) -> str:
        """Write a task with token optimization"""
        start_time = time.time()
        
        # Generate compressed task ID
        task_id = hashlib.md5(f"{agent}{task_type}{time.time()}".encode()).hexdigest()[:8]
        
        # Compress data and get compression ratio
        compressed_data, compression_ratio = self.optimizer.compress_data(data)
        data_size = len(str(data))
        
        # Decide storage strategy based on size
        optimization_config = self.config['token_optimization']
        threshold = optimization_config['compression_threshold']
        
        if len(compressed_data) > threshold:
            # Store in data bucket with reference
            await self._write_data_reference(task_id, compressed_data, "tasks")
            stored_data = f"ref:{task_id}"
        else:
            # Store inline
            stored_data = compressed_data[:optimization_config['max_inline_data']]
        
        # Create InfluxDB point
        point = Point("tasks") \
            .tag("agent", agent[0].upper()) \
            .tag("status", "new") \
            .tag("priority", str(priority)) \
            .tag("type", KEY_MAP.get(task_type, task_type[:3])) \
            .field("id", task_id) \
            .field("data", stored_data) \
            .field("deps", ",".join(dependencies or [])) \
            .field("compression_ratio", compression_ratio) \
            .time(datetime.utcnow())
        
        # Write to InfluxDB
        if self.write_api:
            try:
                self.write_api.write(bucket="tasks", record=point)
            except Exception as e:
                print(f"❌ Error writing task to InfluxDB: {e}")
        
        # Log performance metrics
        exec_time = time.time() - start_time
        tokens_used = self.optimizer.estimate_tokens(stored_data)
        
        await self.log_metrics(
            agent=agent,
            operation=f"write_task:{task_type}",
            tokens_used=tokens_used,
            exec_time=exec_time,
            success=True,
            data_size=data_size,
            compression_ratio=compression_ratio
        )
        
        return task_id
    
    async def read_tasks(self, agent: str, status: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Read tasks for specific agent with token optimization"""
        start_time = time.time()
        
        # Build query
        agent_code = agent[0].upper()
        query_parts = [
            f'from(bucket: "tasks")',
            f'|> range(start: -24h)',  # Last 24 hours
            f'|> filter(fn: (r) => r._measurement == "tasks")',
            f'|> filter(fn: (r) => r.agent == "{agent_code}")'
        ]
        
        if status:
            query_parts.append(f'|> filter(fn: (r) => r.status == "{status}")')
        
        query_parts.extend([
            f'|> sort(columns: ["_time"], desc: true)',
            f'|> limit(n: {limit})'
        ])
        
        query = ' '.join(query_parts)
        
        tasks = []
        if self.query_api:
            try:
                tables = self.query_api.query(query)
                
                for table in tables:
                    for record in table.records:
                        task_data = {
                            'id': record.values.get('id'),
                            'agent': record.values.get('agent'),
                            'type': record.values.get('type'),
                            'status': record.values.get('status'),
                            'priority': record.values.get('priority'),
                            'data': record.values.get('data'),
                            'deps': record.values.get('deps', '').split(',') if record.values.get('deps') else [],
                            'timestamp': record.get_time()
                        }
                        tasks.append(task_data)
                        
            except Exception as e:
                print(f"❌ Error reading tasks from InfluxDB: {e}")
        
        # Log performance
        exec_time = time.time() - start_time
        tokens_used = len(tasks) * 20  # Estimate
        
        await self.log_metrics(
            agent=agent,
            operation="read_tasks",
            tokens_used=tokens_used,
            exec_time=exec_time,
            success=len(tasks) >= 0
        )
        
        return tasks
    
    async def update_task_status(self, task_id: str, status: str, agent: str = None):
        """Update task status"""
        if not self.write_api:
            return
        
        # Write status update point
        point = Point("task_updates") \
            .tag("task_id", task_id) \
            .tag("status", status) \
            .field("updated", True) \
            .time(datetime.utcnow())
        
        if agent:
            point = point.tag("agent", agent[0].upper())
        
        try:
            self.write_api.write(bucket="tasks", record=point)
        except Exception as e:
            print(f"❌ Error updating task status: {e}")
    
    async def get_task_data(self, task_id: str) -> Any:
        """Get full task data, decompressing if needed"""
        if not self.query_api:
            return None
        
        # Query for task data
        query = f'''
        from(bucket: "tasks")
          |> range(start: -24h)
          |> filter(fn: (r) => r._measurement == "tasks")
          |> filter(fn: (r) => r.id == "{task_id}")
          |> last()
        '''
        
        try:
            tables = self.query_api.query(query)
            
            for table in tables:
                for record in table.records:
                    data = record.values.get('data')
                    
                    if data and data.startswith('ref:'):
                        # Get data from reference
                        ref_id = data[4:]  # Remove 'ref:' prefix
                        return await self._read_data_reference(ref_id, "data")
                    else:
                        # Decompress inline data
                        return self.optimizer.decompress_data(data)
                        
        except Exception as e:
            print(f"❌ Error getting task data: {e}")
        
        return None
    
    async def write_data(self, key: str, data: Any, bucket: str = "data", ttl: int = 3600):
        """Write data with compression and TTL"""
        compressed, compression_ratio = self.optimizer.compress_data(data)
        
        point = Point("data") \
            .tag("key", key) \
            .tag("bucket", bucket) \
            .field("value", compressed) \
            .field("compression_ratio", compression_ratio) \
            .field("ttl", ttl) \
            .time(datetime.utcnow())
        
        if self.write_api:
            try:
                self.write_api.write(bucket=bucket, record=point)
            except Exception as e:
                print(f"❌ Error writing data: {e}")
    
    async def read_data(self, key: str, bucket: str = "data") -> Any:
        """Read data with expiration check"""
        if not self.query_api:
            return None
        
        query = f'''
        from(bucket: "{bucket}")
          |> range(start: -24h)
          |> filter(fn: (r) => r._measurement == "data")
          |> filter(fn: (r) => r.key == "{key}")
          |> last()
        '''
        
        try:
            tables = self.query_api.query(query)
            
            for table in tables:
                for record in table.records:
                    # Check TTL
                    timestamp = record.get_time().timestamp()
                    ttl = record.values.get('ttl', 3600)
                    
                    if time.time() - timestamp > ttl:
                        return None  # Expired
                    
                    compressed_value = record.values.get('value')
                    return self.optimizer.decompress_data(compressed_value)
                    
        except Exception as e:
            print(f"❌ Error reading data: {e}")
        
        return None
    
    async def log_metrics(self, agent: str, operation: str, tokens_used: int,
                         exec_time: float, success: bool = True, 
                         data_size: int = 0, compression_ratio: float = 1.0):
        """Log performance metrics for optimization"""
        
        # Store in optimizer history
        self.optimizer.operation_history[operation].append({
            "tokens": tokens_used,
            "exec_time": exec_time,
            "success": success,
            "data_size": data_size,
            "compression_ratio": compression_ratio,
            "timestamp": time.time()
        })
        
        # Keep only recent history (last 100 operations)
        if len(self.optimizer.operation_history[operation]) > 100:
            self.optimizer.operation_history[operation] = \
                self.optimizer.operation_history[operation][-100:]
        
        # Write to InfluxDB
        point = Point("metrics") \
            .tag("agent", agent[0].upper() if agent else "UNKNOWN") \
            .tag("operation", operation) \
            .tag("success", str(success).lower()) \
            .field("tokens_used", tokens_used) \
            .field("exec_time", exec_time) \
            .field("data_size", data_size) \
            .field("compression_ratio", compression_ratio) \
            .time(datetime.utcnow())
        
        if self.write_api:
            try:
                self.write_api.write(bucket="metrics", record=point)
            except Exception as e:
                print(f"❌ Error logging metrics: {e}")
        
        # Check for optimization opportunities
        await self._check_optimization_triggers(agent, operation, tokens_used, exec_time)
    
    async def _check_optimization_triggers(self, agent: str, operation: str, 
                                         tokens_used: int, exec_time: float):
        """Check if optimization is needed based on thresholds"""
        monitoring = self.config['monitoring']
        
        suggestions = []
        
        # High token usage
        if tokens_used > monitoring['high_token_usage']:
            suggestions.append(OptimizationSuggestion(
                target_agent=agent,
                category="high_token_usage",
                suggestion=f"Operation '{operation}' used {tokens_used} tokens. Consider compression or caching.",
                expected_savings=int(tokens_used * 0.3),
                confidence=0.8
            ))
        
        # Slow execution
        if exec_time > monitoring['slow_execution']:
            suggestions.append(OptimizationSuggestion(
                target_agent=agent,
                category="slow_execution",
                suggestion=f"Operation '{operation}' took {exec_time:.2f}s. Consider optimization.",
                expected_savings=0,  # Time, not tokens
                confidence=0.7
            ))
        
        # Write suggestions to InfluxDB
        for suggestion in suggestions:
            await self._write_optimization_suggestion(suggestion)
    
    async def _write_optimization_suggestion(self, suggestion: OptimizationSuggestion):
        """Write optimization suggestion to InfluxDB"""
        point = Point("optimizations") \
            .tag("target_agent", suggestion.target_agent) \
            .tag("category", suggestion.category) \
            .tag("implemented", str(suggestion.implemented).lower()) \
            .field("suggestion", suggestion.suggestion) \
            .field("expected_savings", suggestion.expected_savings) \
            .field("confidence", suggestion.confidence) \
            .field("auto_approve", suggestion.auto_approve) \
            .time(datetime.utcnow())
        
        if self.write_api:
            try:
                self.write_api.write(bucket="optimizations", record=point)
            except Exception as e:
                print(f"❌ Error writing optimization suggestion: {e}")
    
    async def write_reflection(self, agent: str, category: str, severity: str,
                             note: str, action: str, impact_score: float):
        """Write agent reflection for self-improvement"""
        point = Point("reflections") \
            .tag("agent", agent[0].upper()) \
            .tag("category", category) \
            .tag("severity", severity) \
            .field("note", note) \
            .field("action", action) \
            .field("impact_score", impact_score) \
            .time(datetime.utcnow())
        
        if self.write_api:
            try:
                self.write_api.write(bucket="reflections", record=point)
            except Exception as e:
                print(f"❌ Error writing reflection: {e}")
    
    async def get_agent_context(self, agent: str, lookback_hours: int = 2) -> Dict[str, Any]:
        """Get compressed context for agent handoffs"""
        if not self.query_api:
            return {"error": "InfluxDB not available"}
        
        agent_code = agent[0].upper()
        start_time = f"-{lookback_hours}h"
        
        # Query recent metrics
        metrics_query = f'''
        from(bucket: "metrics")
          |> range(start: {start_time})
          |> filter(fn: (r) => r._measurement == "metrics")
          |> filter(fn: (r) => r.agent == "{agent_code}")
        '''
        
        # Query recent tasks
        tasks_query = f'''
        from(bucket: "tasks")
          |> range(start: {start_time})
          |> filter(fn: (r) => r._measurement == "tasks")
          |> filter(fn: (r) => r.agent == "{agent_code}")
        '''
        
        try:
            # Get metrics
            metrics_tables = self.query_api.query(metrics_query)
            metrics = []
            for table in metrics_tables:
                for record in table.records:
                    metrics.append({
                        'tokens_used': record.values.get('tokens_used', 0),
                        'exec_time': record.values.get('exec_time', 0),
                        'success': record.values.get('success') == 'true'
                    })
            
            # Get tasks
            tasks_tables = self.query_api.query(tasks_query)
            tasks = []
            for table in tasks_tables:
                for record in table.records:
                    tasks.append({
                        'id': record.values.get('id'),
                        'status': record.values.get('status'),
                        'type': record.values.get('type')
                    })
            
            # Calculate compressed context
            context = {
                "agent": agent_code,
                "period_hours": lookback_hours,
                "tasks_total": len(tasks),
                "tasks_completed": len([t for t in tasks if t.get('status') == 'done']),
                "avg_tokens": sum(m['tokens_used'] for m in metrics) / max(len(metrics), 1),
                "avg_exec_time": sum(m['exec_time'] for m in metrics) / max(len(metrics), 1),
                "success_rate": sum(m['success'] for m in metrics) / max(len(metrics), 1),
                "last_activity": datetime.utcnow().isoformat(),
                "efficiency_score": self._calculate_efficiency_score(metrics)
            }
            
            return context
            
        except Exception as e:
            print(f"❌ Error getting agent context: {e}")
            return {"error": str(e)}
    
    def _calculate_efficiency_score(self, metrics: List[Dict]) -> float:
        """Calculate efficiency score (0-1, higher is better)"""
        if not metrics:
            return 0.5
        
        # Factor in success rate, token efficiency, and execution speed
        success_rate = sum(m['success'] for m in metrics) / len(metrics)
        avg_tokens = sum(m['tokens_used'] for m in metrics) / len(metrics)
        avg_time = sum(m['exec_time'] for m in metrics) / len(metrics)
        
        # Normalize and combine (lower tokens and time are better)
        token_efficiency = max(0, 1 - (avg_tokens / 1000))  # Normalize to 0-1
        time_efficiency = max(0, 1 - (avg_time / 10))       # Normalize to 0-1
        
        # Weighted combination
        efficiency = (success_rate * 0.4) + (token_efficiency * 0.3) + (time_efficiency * 0.3)
        return min(1.0, max(0.0, efficiency))
    
    async def _write_data_reference(self, ref_id: str, data: str, bucket: str):
        """Write data reference for large data"""
        point = Point("data_refs") \
            .tag("ref_id", ref_id) \
            .field("data", data) \
            .time(datetime.utcnow())
        
        if self.write_api:
            try:
                self.write_api.write(bucket=bucket, record=point)
            except Exception as e:
                print(f"❌ Error writing data reference: {e}")
    
    async def _read_data_reference(self, ref_id: str, bucket: str) -> Any:
        """Read data from reference"""
        if not self.query_api:
            return None
        
        query = f'''
        from(bucket: "{bucket}")
          |> range(start: -24h)
          |> filter(fn: (r) => r._measurement == "data_refs")
          |> filter(fn: (r) => r.ref_id == "{ref_id}")
          |> last()
        '''
        
        try:
            tables = self.query_api.query(query)
            
            for table in tables:
                for record in table.records:
                    compressed_data = record.values.get('data')
                    return self.optimizer.decompress_data(compressed_data)
                    
        except Exception as e:
            print(f"❌ Error reading data reference: {e}")
        
        return None
    
    async def generate_efficiency_report(self, period_hours: int = 24) -> Dict[str, Any]:
        """Generate efficiency report for all agents"""
        if not self.query_api:
            return {"error": "InfluxDB not available"}
        
        start_time = f"-{period_hours}h"
        
        # Query all metrics in period
        query = f'''
        from(bucket: "metrics")
          |> range(start: {start_time})
          |> filter(fn: (r) => r._measurement == "metrics")
          |> group(columns: ["agent"])
        '''
        
        try:
            tables = self.query_api.query(query)
            
            agent_stats = defaultdict(lambda: {
                'tokens_used': 0,
                'operations': 0,
                'exec_time': 0,
                'successes': 0
            })
            
            for table in tables:
                for record in table.records:
                    agent = record.values.get('agent')
                    tokens = record.values.get('tokens_used', 0)
                    exec_time = record.values.get('exec_time', 0)
                    success = record.values.get('success') == 'true'
                    
                    agent_stats[agent]['tokens_used'] += tokens
                    agent_stats[agent]['operations'] += 1
                    agent_stats[agent]['exec_time'] += exec_time
                    if success:
                        agent_stats[agent]['successes'] += 1
            
            # Generate report
            report = {
                "period_hours": period_hours,
                "generated_at": datetime.utcnow().isoformat(),
                "total_tokens": sum(stats['tokens_used'] for stats in agent_stats.values()),
                "total_operations": sum(stats['operations'] for stats in agent_stats.values()),
                "agents": {}
            }
            
            for agent, stats in agent_stats.items():
                if stats['operations'] > 0:
                    report["agents"][agent] = {
                        "tokens_used": stats['tokens_used'],
                        "operations": stats['operations'],
                        "avg_tokens_per_op": stats['tokens_used'] / stats['operations'],
                        "avg_exec_time": stats['exec_time'] / stats['operations'],
                        "success_rate": stats['successes'] / stats['operations'],
                        "efficiency_score": self._calculate_efficiency_score([{
                            'tokens_used': stats['tokens_used'] / stats['operations'],
                            'exec_time': stats['exec_time'] / stats['operations'],
                            'success': stats['successes'] / stats['operations'] > 0.5
                        }])
                    }
            
            return report
            
        except Exception as e:
            print(f"❌ Error generating efficiency report: {e}")
            return {"error": str(e)}
    
    def close(self):
        """Close InfluxDB client"""
        if self.client:
            self.client.close()

# Global blackboard instance
blackboard = None

def get_blackboard() -> InfluxBlackboard:
    """Get global blackboard instance"""
    global blackboard
    if blackboard is None:
        blackboard = InfluxBlackboard()
    return blackboard

# Convenience functions for backward compatibility
async def write_task(agent: str, task_type: str, data: Any, priority: int = 1) -> str:
    """Write task to blackboard"""
    bb = get_blackboard()
    return await bb.write_task(agent, task_type, data, priority)

async def read_tasks(agent: str, status: str = None) -> List[Dict[str, Any]]:
    """Read tasks from blackboard"""
    bb = get_blackboard()
    return await bb.read_tasks(agent, status)

async def update_status(task_id: str, status: str, agent: str = None):
    """Update task status"""
    bb = get_blackboard()
    await bb.update_task_status(task_id, status, agent)

async def get_data(task_id: str) -> Any:
    """Get task data"""
    bb = get_blackboard()
    return await bb.get_task_data(task_id)

async def log_performance(agent: str, operation: str, tokens: int, time_sec: float, 
                         success: bool = True, data_size: int = 0):
    """Log performance metrics"""
    bb = get_blackboard()
    await bb.log_metrics(agent, operation, tokens, time_sec, success, data_size)

async def get_context(agent: str) -> Dict[str, Any]:
    """Get agent context for handoffs"""
    bb = get_blackboard()
    return await bb.get_agent_context(agent)

async def write_reflection(agent: str, category: str, severity: str, note: str, action: str, impact: float):
    """Write agent reflection"""
    bb = get_blackboard()
    await bb.write_reflection(agent, category, severity, note, action, impact)

async def generate_report(hours: int = 24) -> Dict[str, Any]:
    """Generate efficiency report"""
    bb = get_blackboard()
    return await bb.generate_efficiency_report(hours)