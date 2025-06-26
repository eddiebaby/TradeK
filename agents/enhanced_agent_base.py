#!/usr/bin/env python3
"""
Enhanced Agent Base Class with Token Optimization and InfluxDB Blackboard Integration
Provides foundation for SPARC framework agents with self-reflection capabilities
"""

import asyncio
import time
import json
import hashlib
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from pathlib import Path

from influx_blackboard import (
    get_blackboard, write_task, read_tasks, update_status, 
    log_performance, get_context, write_reflection
)

@dataclass
class AgentCapability:
    """Agent capability definition"""
    name: str
    description: str
    token_budget: int
    success_threshold: float = 0.8

@dataclass
class TaskResult:
    """Task execution result"""
    success: bool
    data: Any
    tokens_used: int
    exec_time: float
    confidence: float = 1.0
    error_message: Optional[str] = None

class TokenTracker:
    """Advanced token usage tracking and optimization"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.operation_costs = {}
        self.session_tokens = 0
        self.optimization_suggestions = []
        
    def start_operation(self, operation_name: str) -> str:
        """Start tracking an operation"""
        op_id = hashlib.md5(f"{operation_name}{time.time()}".encode()).hexdigest()[:8]
        self.operation_costs[op_id] = {
            "name": operation_name,
            "start_time": time.time(),
            "tokens_start": self.session_tokens
        }
        return op_id
    
    def end_operation(self, op_id: str, tokens_used: int, success: bool = True) -> Dict[str, Any]:
        """End operation tracking and return metrics"""
        if op_id not in self.operation_costs:
            return {}
        
        op_data = self.operation_costs[op_id]
        exec_time = time.time() - op_data["start_time"]
        
        self.session_tokens += tokens_used
        
        metrics = {
            "operation": op_data["name"],
            "tokens_used": tokens_used,
            "exec_time": exec_time,
            "success": success,
            "tokens_per_second": tokens_used / max(exec_time, 0.001)
        }
        
        # Check for optimization opportunities
        self._check_optimization_opportunity(metrics)
        
        # Clean up
        del self.operation_costs[op_id]
        
        return metrics
    
    def _check_optimization_opportunity(self, metrics: Dict[str, Any]):
        """Check if operation needs optimization"""
        operation = metrics["operation"]
        tokens_used = metrics["tokens_used"]
        exec_time = metrics["exec_time"]
        
        suggestions = []
        
        # High token usage
        if tokens_used > 500:
            suggestions.append({
                "type": "high_token_usage",
                "message": f"Operation '{operation}' used {tokens_used} tokens. Consider compression.",
                "priority": "high" if tokens_used > 1000 else "medium"
            })
        
        # Slow execution with high tokens
        if exec_time > 3.0 and tokens_used > 200:
            suggestions.append({
                "type": "slow_heavy_operation", 
                "message": f"Operation '{operation}' is slow ({exec_time:.2f}s) and token-heavy ({tokens_used}). Consider caching.",
                "priority": "high"
            })
        
        # Poor token efficiency
        token_efficiency = tokens_used / max(exec_time, 0.001)
        if token_efficiency > 200:  # More than 200 tokens per second
            suggestions.append({
                "type": "poor_efficiency",
                "message": f"Operation '{operation}' has poor token efficiency ({token_efficiency:.1f} t/s). Consider optimization.",
                "priority": "medium"
            })
        
        self.optimization_suggestions.extend(suggestions)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get session token usage summary"""
        return {
            "total_tokens": self.session_tokens,
            "active_operations": len(self.operation_costs),
            "optimization_suggestions": len(self.optimization_suggestions),
            "suggestions": self.optimization_suggestions[-5:]  # Last 5 suggestions
        }

class EnhancedAgentBase(ABC):
    """Enhanced base class for SPARC framework agents with token optimization"""
    
    def __init__(self, agent_name: str, capabilities: List[AgentCapability] = None):
        self.agent_name = agent_name
        self.agent_code = agent_name[0].upper()  # R, M, E
        self.capabilities = capabilities or []
        self.blackboard = get_blackboard()
        self.token_tracker = TokenTracker(agent_name)
        self.logger = self._setup_logger()
        
        # Performance tracking
        self.success_count = 0
        self.failure_count = 0
        self.total_tokens = 0
        self.session_start = time.time()
        
        # Self-reflection settings
        self.reflection_interval = 300  # 5 minutes
        self.last_reflection = time.time()
        self.performance_threshold = 0.8
        
        self.logger.info(f"🚀 {agent_name} agent initialized with {len(capabilities)} capabilities")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup agent-specific logger"""
        logger = logging.getLogger(f"agent.{self.agent_name.lower()}")
        logger.setLevel(logging.INFO)
        
        # Create handler if not exists
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - {self.agent_name} - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    @abstractmethod
    async def process_task(self, task_data: Dict[str, Any]) -> TaskResult:
        """Process a task - must be implemented by subclasses"""
        pass
    
    async def execute_with_tracking(self, operation_name: str, operation_func: Callable, *args, **kwargs) -> TaskResult:
        """Execute operation with automatic token tracking"""
        op_id = self.token_tracker.start_operation(operation_name)
        
        try:
            # Execute the operation
            result = await operation_func(*args, **kwargs)
            
            # Handle different result types
            if isinstance(result, TaskResult):
                task_result = result
            elif isinstance(result, dict) and 'success' in result:
                task_result = TaskResult(**result)
            else:
                # Assume successful operation with data
                task_result = TaskResult(
                    success=True,
                    data=result,
                    tokens_used=self._estimate_tokens(str(result)),
                    exec_time=0  # Will be calculated by tracker
                )
            
            # Track performance
            metrics = self.token_tracker.end_operation(
                op_id, task_result.tokens_used, task_result.success
            )
            
            # Update task result with actual timing
            task_result.exec_time = metrics.get("exec_time", 0)
            
            # Log performance to blackboard
            await log_performance(
                agent=self.agent_name,
                operation=operation_name,
                tokens=task_result.tokens_used,
                time_sec=task_result.exec_time,
                success=task_result.success
            )
            
            # Update counters
            if task_result.success:
                self.success_count += 1
            else:
                self.failure_count += 1
            
            self.total_tokens += task_result.tokens_used
            
            # Check if reflection is needed
            await self._check_reflection_trigger()
            
            return task_result
            
        except Exception as e:
            # Handle errors
            metrics = self.token_tracker.end_operation(op_id, 0, False)
            
            error_result = TaskResult(
                success=False,
                data=None,
                tokens_used=0,
                exec_time=metrics.get("exec_time", 0),
                error_message=str(e)
            )
            
            self.failure_count += 1
            
            # Log error
            await log_performance(
                agent=self.agent_name,
                operation=operation_name,
                tokens=0,
                time_sec=error_result.exec_time,
                success=False
            )
            
            self.logger.error(f"Operation '{operation_name}' failed: {e}")
            
            return error_result
    
    async def run_task_loop(self, max_iterations: int = None, poll_interval: float = 1.0):
        """Main task processing loop"""
        self.logger.info(f"🔄 Starting task loop (max_iterations: {max_iterations})")
        
        iteration = 0
        
        while max_iterations is None or iteration < max_iterations:
            try:
                # Get pending tasks
                tasks = await read_tasks(self.agent_name, status="new")
                
                if tasks:
                    self.logger.info(f"📋 Found {len(tasks)} pending tasks")
                    
                    for task in tasks:
                        # Mark task as processing
                        await update_status(task['id'], "proc", self.agent_name)
                        
                        # Get full task data
                        task_data = await self.blackboard.get_task_data(task['id'])
                        
                        self.logger.info(f"🔨 Processing task {task['id']} ({task['type']})")
                        
                        # Process task with tracking
                        result = await self.execute_with_tracking(
                            f"task_{task['type']}", 
                            self.process_task,
                            task_data or {}
                        )
                        
                        # Update task status based on result
                        final_status = "done" if result.success else "error"
                        await update_status(task['id'], final_status, self.agent_name)
                        
                        # Log result
                        if result.success:
                            self.logger.info(f"✅ Task {task['id']} completed successfully")
                        else:
                            self.logger.error(f"❌ Task {task['id']} failed: {result.error_message}")
                
                else:
                    # No tasks - brief idle period
                    await asyncio.sleep(poll_interval)
                
                iteration += 1
                
            except Exception as e:
                self.logger.error(f"Error in task loop: {e}")
                await asyncio.sleep(poll_interval)
    
    async def create_task_for_agent(self, target_agent: str, task_type: str, 
                                  data: Any, priority: int = 1, dependencies: List[str] = None) -> str:
        """Create a task for another agent"""
        task_id = await write_task(target_agent, task_type, data, priority, dependencies)
        
        self.logger.info(f"📤 Created task {task_id} for {target_agent} ({task_type})")
        
        return task_id
    
    async def handoff_to_agent(self, target_agent: str, handoff_data: Dict[str, Any]) -> str:
        """Perform intelligent handoff to another agent"""
        
        # Get context for intelligent handoff
        my_context = await get_context(self.agent_name)
        target_context = await get_context(target_agent)
        
        # Create compressed handoff package
        handoff_package = {
            "from_agent": self.agent_name,
            "from_context": my_context,
            "handoff_data": handoff_data,
            "timestamp": datetime.utcnow().isoformat(),
            "priority_suggestion": self._calculate_handoff_priority(my_context, target_context)
        }
        
        # Determine task type based on target agent
        task_type = self._determine_task_type(target_agent, handoff_data)
        
        # Create task
        task_id = await self.create_task_for_agent(
            target_agent, 
            task_type, 
            handoff_package,
            priority=handoff_package["priority_suggestion"]
        )
        
        self.logger.info(f"🤝 Handed off to {target_agent}: {task_id}")
        
        return task_id
    
    def _calculate_handoff_priority(self, my_context: Dict, target_context: Dict) -> int:
        """Calculate priority for handoff based on agent contexts"""
        
        # High priority if target agent has low workload
        target_tasks = target_context.get("tasks_total", 0)
        if target_tasks == 0:
            return 1  # High priority
        elif target_tasks < 3:
            return 2  # Medium priority
        else:
            return 3  # Low priority
    
    def _determine_task_type(self, target_agent: str, data: Dict[str, Any]) -> str:
        """Determine appropriate task type for target agent"""
        
        # Map common handoff patterns
        handoff_patterns = {
            "Researcher": ["intelligence_gathering", "security_analysis", "market_research"],
            "Mastermind": ["strategic_analysis", "architectural_design", "quality_strategy"],
            "Executor": ["implementation", "testing", "deployment"]
        }
        
        agent_tasks = handoff_patterns.get(target_agent, ["general_task"])
        
        # Try to match data content to appropriate task type
        data_str = str(data).lower()
        
        for task_type in agent_tasks:
            if any(keyword in data_str for keyword in task_type.split("_")):
                return task_type
        
        # Default to first task type for agent
        return agent_tasks[0]
    
    async def _check_reflection_trigger(self):
        """Check if self-reflection should be triggered"""
        
        current_time = time.time()
        
        # Time-based reflection
        if current_time - self.last_reflection > self.reflection_interval:
            await self._perform_self_reflection("scheduled")
            self.last_reflection = current_time
        
        # Performance-based reflection
        total_operations = self.success_count + self.failure_count
        if total_operations > 0 and total_operations % 10 == 0:  # Every 10 operations
            success_rate = self.success_count / total_operations
            if success_rate < self.performance_threshold:
                await self._perform_self_reflection("performance")
    
    async def _perform_self_reflection(self, trigger_type: str):
        """Perform self-reflection and generate insights"""
        
        self.logger.info(f"🤔 Performing self-reflection (trigger: {trigger_type})")
        
        # Gather performance data
        total_operations = self.success_count + self.failure_count
        success_rate = self.success_count / max(total_operations, 1)
        avg_tokens = self.total_tokens / max(total_operations, 1)
        session_duration = time.time() - self.session_start
        
        # Get token tracker summary
        token_summary = self.token_tracker.get_session_summary()
        
        # Generate reflection insights
        insights = []
        action_items = []
        impact_score = 0.0
        
        # Performance analysis
        if success_rate < 0.8:
            insights.append(f"Success rate ({success_rate:.1%}) below target (80%)")
            action_items.append("Investigate failure patterns and improve error handling")
            impact_score += 0.3
        
        # Token efficiency analysis
        if avg_tokens > 300:
            insights.append(f"High average token usage ({avg_tokens:.0f} per operation)")
            action_items.append("Implement data compression and caching strategies")
            impact_score += 0.2
        
        # Optimization suggestions
        if token_summary["optimization_suggestions"] > 0:
            insights.append(f"Found {token_summary['optimization_suggestions']} optimization opportunities")
            action_items.append("Review and implement optimization suggestions")
            impact_score += 0.1
        
        # Productivity analysis
        operations_per_minute = total_operations / max(session_duration / 60, 1)
        if operations_per_minute < 1.0:
            insights.append(f"Low productivity ({operations_per_minute:.2f} operations/min)")
            action_items.append("Optimize task processing pipeline")
            impact_score += 0.2
        
        # Positive insights
        if success_rate > 0.95:
            insights.append(f"Excellent success rate ({success_rate:.1%})")
            impact_score += 0.1
        
        if avg_tokens < 100:
            insights.append(f"Efficient token usage ({avg_tokens:.0f} avg per operation)")
            impact_score += 0.1
        
        # Compile reflection
        reflection_note = f"Session analysis: {len(insights)} insights identified. " + "; ".join(insights[:3])
        action_summary = "; ".join(action_items[:2]) if action_items else "Continue current approach"
        
        # Write reflection to blackboard
        await write_reflection(
            agent=self.agent_name,
            category=trigger_type,
            severity="high" if impact_score > 0.5 else "medium" if impact_score > 0.2 else "low",
            note=reflection_note,
            action=action_summary,
            impact_score=impact_score
        )
        
        self.logger.info(f"📝 Reflection completed - Impact score: {impact_score:.2f}")
        
        # Reset some counters for next reflection cycle
        if trigger_type == "scheduled":
            self.success_count = 0
            self.failure_count = 0
            self.total_tokens = 0
            self.session_start = time.time()
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        return max(1, len(text) // 4)  # Rough estimate
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        total_operations = self.success_count + self.failure_count
        session_duration = time.time() - self.session_start
        
        return {
            "agent_name": self.agent_name,
            "status": "active",
            "session_duration": session_duration,
            "operations_completed": total_operations,
            "success_rate": self.success_count / max(total_operations, 1),
            "total_tokens_used": self.total_tokens,
            "avg_tokens_per_operation": self.total_tokens / max(total_operations, 1),
            "capabilities": [cap.name for cap in self.capabilities],
            "last_reflection": datetime.fromtimestamp(self.last_reflection).isoformat(),
            "token_summary": self.token_tracker.get_session_summary()
        }
    
    async def shutdown(self):
        """Graceful shutdown with final reflection"""
        self.logger.info("🛑 Shutting down agent...")
        
        # Perform final reflection
        await self._perform_self_reflection("shutdown")
        
        # Get final status
        final_status = await self.get_agent_status()
        self.logger.info(f"📊 Final status: {json.dumps(final_status, indent=2)}")
        
        # Close blackboard connection
        self.blackboard.close()
        
        self.logger.info("✅ Agent shutdown complete")

# Convenience decorators for token tracking
def track_tokens(operation_name: str = None):
    """Decorator to automatically track token usage for methods"""
    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            if not hasattr(self, 'execute_with_tracking'):
                # Fallback for non-agent classes
                return await func(self, *args, **kwargs)
            
            op_name = operation_name or f"{func.__name__}"
            return await self.execute_with_tracking(op_name, func, *args, **kwargs)
        
        return wrapper
    return decorator

def compress_data(func):
    """Decorator to automatically compress large data returns"""
    async def wrapper(*args, **kwargs):
        result = await func(*args, **kwargs)
        
        # If result is large, consider compression
        if isinstance(result, (dict, list)) and len(str(result)) > 1000:
            # Could implement automatic compression here
            pass
        
        return result
    
    return wrapper