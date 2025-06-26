"""
Base Agent Architecture for MASTERMIND & EXECUTOR

This module provides the foundational architecture for both agents,
including shared communication protocols, memory systems, and tool integration.
Enhanced with persistent memory capabilities for crash recovery and state continuity.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import logging
from pathlib import Path

# Import persistent memory capabilities (avoid circular import)
try:
    from .persistent_mixin import PersistentAgentMixin, EventType, DocumentType
except ImportError:
    # Fallback definitions to avoid circular imports
    from enum import Enum
    
    class EventType(Enum):
        TASK_START = "task_start"
        TASK_UPDATE = "task_update"
        TASK_COMPLETE = "task_complete"
        AGENT_MESSAGE = "agent_message"
        SYSTEM_EVENT = "system_event"
    
    class DocumentType(Enum):
        RESEARCH = "research"
        CODE = "code"
        PLAN = "plan"
        REPORT = "report"
    
    class PersistentAgentMixin:
        """Fallback mixin for when persistence isn't available"""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
        def save_state(self, **kwargs):
            pass
            
        def restore_state(self):
            pass


class AgentRole(Enum):
    """Agent role definitions."""
    MASTERMIND = "strategic_architect"
    EXECUTOR = "implementation_virtuoso"
    RESEARCHER = "knowledge_architect"


class MessageType(Enum):
    """Inter-agent message types."""
    STRATEGIC_HANDOFF = "strategic_to_tactical"
    TACTICAL_FEEDBACK = "tactical_to_strategic"
    COLLABORATIVE_SESSION = "joint_problem_solving"
    STATUS_UPDATE = "status_notification"
    CONTEXT_SYNC = "context_synchronization"


@dataclass
class AgentMessage:
    """Structured message format for agent communication."""
    sender: AgentRole
    recipient: AgentRole
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: f"msg_{int(time.time() * 1000)}")
    priority: int = 1  # 1-5, where 5 is highest priority
    requires_response: bool = False
    context_id: Optional[str] = None


@dataclass
class TaskContext:
    """Context preservation for task handoffs."""
    task_id: str
    description: str
    requirements: Dict[str, Any]
    constraints: Dict[str, Any]
    quality_gates: Dict[str, Any]
    success_criteria: Dict[str, Any]
    architectural_context: Dict[str, Any]
    performance_targets: Dict[str, Any]
    security_requirements: Dict[str, Any]
    current_status: str = "pending"
    start_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SharedMemory:
    """Persistent memory system shared between agents."""
    
    def __init__(self, memory_file: str = "agents/data/shared_memory.json"):
        self.memory_file = Path(memory_file)
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        self._memory: Dict[str, Any] = self._load_memory()
        
    def _load_memory(self) -> Dict[str, Any]:
        """Load memory from persistent storage."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load memory: {e}")
        
        return {
            "project_state": {},
            "quality_trajectory": [],
            "architectural_decisions": [],
            "performance_baseline": {},
            "technical_debt_map": {},
            "team_velocity": {},
            "collaboration_history": [],
            "optimization_opportunities": [],
            "risk_factors": []
        }
    
    def _save_memory(self):
        """Save memory to persistent storage."""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self._memory, f, indent=2, default=str)
        except Exception as e:
            logging.error(f"Failed to save memory: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from shared memory."""
        return self._memory.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set value in shared memory."""
        self._memory[key] = value
        self._save_memory()
    
    def update(self, key: str, updates: Dict[str, Any]):
        """Update nested dictionary in memory."""
        if key not in self._memory:
            self._memory[key] = {}
        self._memory[key].update(updates)
        self._save_memory()
    
    def append(self, key: str, item: Any):
        """Append item to list in memory."""
        if key not in self._memory:
            self._memory[key] = []
        self._memory[key].append(item)
        self._save_memory()
    
    def get_context_snapshot(self) -> Dict[str, Any]:
        """Get complete context snapshot for handoffs."""
        return {
            "timestamp": time.time(),
            "project_state": self._memory.get("project_state", {}),
            "current_quality": self._memory.get("quality_trajectory", [])[-1:],
            "active_decisions": self._memory.get("architectural_decisions", [])[-5:],
            "performance_status": self._memory.get("performance_baseline", {}),
            "priority_risks": self._memory.get("risk_factors", [])[:3]
        }


class BaseAgent(ABC, PersistentAgentMixin):
    """
    Base class for both MASTERMIND and EXECUTOR agents.
    Enhanced with persistent memory capabilities for crash recovery and state continuity.
    """
    
    def __init__(self, role: AgentRole, name: str):
        # Initialize traditional agent components
        self.role = role
        self.name = name
        self.shared_memory = SharedMemory()
        self.message_queue: List[AgentMessage] = []
        self.active_tasks: Dict[str, TaskContext] = {}
        self.tools: Dict[str, Callable] = {}
        self.capabilities: List[str] = []
        self.thinking_modes: Dict[str, str] = {}
        self.performance_metrics: Dict[str, Any] = {}
        
        # Setup logger early so it's available for all operations
        self.logger = self._setup_logger()
        
        # Initialize persistent memory capabilities
        PersistentAgentMixin.__init__(self, agent_name=name.lower())
        
        # Restore state if available (now that logger exists)
        try:
            if hasattr(self, 'restore_state'):
                self.restore_state()
        except Exception as e:
            self.logger.error(f"Error restoring state for agent {self.name}: {e}")
        
        # Record agent initialization
        try:
            self.signal_event(
                EventType.AGENT_STATE_CHANGE,
                f"Agent {self.name} ({self.role.value}) initialized",
                context={"role": self.role.value, "capabilities": self._get_agent_capabilities()},
                metadata={"initialization": True}
            )
        except Exception as e:
            self.logger.error(f"Error recording initialization event for agent {self.name}: {e}")
        
    def _setup_logger(self) -> logging.Logger:
        """Setup agent-specific logging."""
        logger = logging.getLogger(f"agent.{self.name.lower()}")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = Path(f"agents/logs/{self.name.lower()}.log")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities."""
        pass
    
    @abstractmethod
    def get_thinking_modes(self) -> Dict[str, str]:
        """Return available thinking modes for this agent."""
        pass
    
    @abstractmethod
    async def process_task(self, task: TaskContext) -> Dict[str, Any]:
        """Process a task according to agent specialization."""
        pass
    
    def register_tool(self, tool_name: str, tool_func: Callable):
        """Register an MCP tool with the agent."""
        self.tools[tool_name] = tool_func
        self.logger.info(f"Registered tool: {tool_name}")
    
    def get_available_tools(self) -> List[str]:
        """Get list of available MCP tools."""
        return list(self.tools.keys())
    
    async def use_tool(self, tool_name: str, **kwargs) -> Any:
        """Use an MCP tool with provided arguments."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not available")
        
        self.logger.info(f"Using tool: {tool_name} with args: {kwargs}")
        
        try:
            result = await self.tools[tool_name](**kwargs)
            self.logger.info(f"Tool {tool_name} completed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Tool {tool_name} failed: {e}")
            raise
    
    def send_message(self, recipient: AgentRole, message_type: MessageType, 
                    payload: Dict[str, Any], **kwargs) -> str:
        """Send message to another agent."""
        message = AgentMessage(
            sender=self.role,
            recipient=recipient,
            message_type=message_type,
            payload=payload,
            **kwargs
        )
        
        # Store in shared memory for recipient
        messages_key = f"messages_{recipient.value}"
        self.shared_memory.append(messages_key, message.__dict__)
        
        # Record in persistent memory
        self.signal_event(
            EventType.HANDOFF,
            f"Sent {message_type.value} message to {recipient.value}",
            context={
                "message_id": message.message_id,
                "recipient": recipient.value,
                "message_type": message_type.value,
                "payload_keys": list(payload.keys())
            },
            metadata={"communication": True}
        )
        
        self.logger.info(f"Sent {message_type.value} message to {recipient.value}")
        return message.message_id
    
    def receive_messages(self) -> List[AgentMessage]:
        """Receive pending messages for this agent."""
        messages_key = f"messages_{self.role.value}"
        message_dicts = self.shared_memory.get(messages_key, [])
        
        messages = []
        for msg_dict in message_dicts:
            message = AgentMessage(**msg_dict)
            messages.append(message)
        
        # Clear received messages
        self.shared_memory.set(messages_key, [])
        
        return messages
    
    def create_task_context(self, description: str, **kwargs) -> TaskContext:
        """Create a new task context."""
        task_id = f"task_{int(time.time() * 1000)}"
        
        context = TaskContext(
            task_id=task_id,
            description=description,
            requirements=kwargs.get('requirements', {}),
            constraints=kwargs.get('constraints', {}),
            quality_gates=kwargs.get('quality_gates', {}),
            success_criteria=kwargs.get('success_criteria', {}),
            architectural_context=kwargs.get('architectural_context', {}),
            performance_targets=kwargs.get('performance_targets', {}),
            security_requirements=kwargs.get('security_requirements', {}),
            metadata=kwargs.get('metadata', {})
        )
        
        self.active_tasks[task_id] = context
        self.logger.info(f"Created task context: {task_id}")
        
        return context
    
    def update_task_status(self, task_id: str, status: str, **updates):
        """Update task status and context."""
        if task_id in self.active_tasks:
            self.active_tasks[task_id].current_status = status
            for key, value in updates.items():
                setattr(self.active_tasks[task_id], key, value)
            
            # Update persistent task tracking if this is the current task
            if hasattr(self, 'current_task') and self.current_task and self.current_task.task_id == task_id:
                progress = getattr(self.active_tasks[task_id], 'progress', 0.0)
                self.update_task_progress(
                    progress=progress,
                    phase=status,
                    context={"status_update": updates}
                )
            
            # Record status change event
            self.signal_event(
                EventType.TASK_COMPLETION,
                f"Task {task_id} status updated to {status}",
                context={
                    "task_id": task_id,
                    "new_status": status,
                    "updates": updates
                },
                metadata={"status_change": True}
            )
            
            self.logger.info(f"Updated task {task_id} status to: {status}")
    
    def get_context_for_handoff(self, task_id: str) -> Dict[str, Any]:
        """Prepare context for task handoff to another agent."""
        task = self.active_tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        return {
            "task_context": task.__dict__,
            "shared_context": self.shared_memory.get_context_snapshot(),
            "agent_insights": self.get_insights_for_handoff(task),
            "recommended_approach": self.recommend_approach(task),
            "quality_requirements": self.get_quality_requirements(task),
            "handoff_timestamp": time.time()
        }
    
    @abstractmethod
    def get_insights_for_handoff(self, task: TaskContext) -> Dict[str, Any]:
        """Get agent-specific insights for task handoff."""
        pass
    
    @abstractmethod
    def recommend_approach(self, task: TaskContext) -> Dict[str, Any]:
        """Recommend approach for task execution."""
        pass
    
    @abstractmethod
    def get_quality_requirements(self, task: TaskContext) -> Dict[str, Any]:
        """Get quality requirements for task."""
        pass
    
    def record_performance_metric(self, metric_name: str, value: Any):
        """Record performance metric."""
        timestamp = time.time()
        if metric_name not in self.performance_metrics:
            self.performance_metrics[metric_name] = []
        
        self.performance_metrics[metric_name].append({
            "value": value,
            "timestamp": timestamp
        })
        
        # Also store in shared memory
        self.shared_memory.append(f"performance_{self.role.value}", {
            "metric": metric_name,
            "value": value,
            "timestamp": timestamp,
            "agent": self.role.value
        })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for this agent."""
        summary = {}
        for metric_name, values in self.performance_metrics.items():
            if values:
                recent_values = [v["value"] for v in values[-10:]]  # Last 10 values
                summary[metric_name] = {
                    "current": values[-1]["value"],
                    "average": sum(recent_values) / len(recent_values),
                    "trend": "improving" if len(values) > 1 and values[-1]["value"] > values[-2]["value"] else "stable",
                    "count": len(values)
                }
        
        return summary
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform agent health check."""
        health_info = {
            "agent": self.name,
            "role": self.role.value,
            "status": "healthy",
            "active_tasks": len(self.active_tasks),
            "available_tools": len(self.tools),
            "capabilities": len(self.capabilities),
            "performance_metrics": len(self.performance_metrics),
            "memory_usage": len(str(self.shared_memory._memory)),
            "timestamp": time.time()
        }
        
        # Add persistent state health info
        if hasattr(self, 'current_task') and self.current_task:
            health_info["current_task"] = {
                "task_id": self.current_task.task_id,
                "progress": self.current_task.progress,
                "phase": self.current_task.phase
            }
        
        if hasattr(self, 'workflow_context') and self.workflow_context:
            health_info["workflow_active"] = True
            health_info["workflow_context_size"] = len(self.workflow_context)
        
        return health_info
    
    def _get_persistent_state(self) -> Dict[str, Any]:
        """Get agent-specific persistent state for saving."""
        return {
            "active_tasks": {k: v.__dict__ for k, v in self.active_tasks.items()},
            "tools": list(self.tools.keys()),
            "capabilities": self.capabilities,
            "performance_metrics": self.performance_metrics,
            "thinking_modes": self.thinking_modes,
            "shared_memory_snapshot": self.shared_memory.get_context_snapshot()
        }
    
    def _restore_persistent_state(self, state_data: Dict[str, Any]) -> None:
        """Restore agent-specific state from persistent storage."""
        try:
            # Restore active tasks
            if "active_tasks" in state_data:
                for task_id, task_data in state_data["active_tasks"].items():
                    # Reconstruct TaskContext objects
                    task_context = TaskContext(**task_data)
                    self.active_tasks[task_id] = task_context
            
            # Restore capabilities
            if "capabilities" in state_data:
                self.capabilities = state_data["capabilities"]
            
            # Restore performance metrics
            if "performance_metrics" in state_data:
                self.performance_metrics = state_data["performance_metrics"]
            
            # Restore thinking modes
            if "thinking_modes" in state_data:
                self.thinking_modes = state_data["thinking_modes"]
            
            # Note: Tools are not restored as they need to be re-registered
            # Shared memory is managed separately
            
            self.logger.info(f"Restored persistent state for agent {self.name}")
            
        except Exception as e:
            self.logger.error(f"Error restoring persistent state for {self.name}: {e}")
    
    def _handle_received_handoff(self, handoff_context) -> None:
        """Handle received handoff from another agent."""
        try:
            # Extract task context if available
            task_context = handoff_context.task_context
            if task_context and "task_id" in task_context:
                # Start tracking the handoff task
                if hasattr(self, 'start_task'):
                    self.start_task(
                        task_id=task_context.get("task_id", "handoff_task"),
                        description=task_context.get("description", "Received handoff task"),
                        phase="handoff_received"
                    )
            
            # Update agent's understanding of the workflow
            if "workflow_context" in task_context:
                self.set_workflow_context(task_context["workflow_context"])
            
            self.logger.info(f"Processed handoff from {handoff_context.from_agent}")
            
        except Exception as e:
            self.logger.error(f"Error handling received handoff: {e}")
    
    def prepare_for_shutdown(self) -> None:
        """Prepare agent for clean shutdown."""
        try:
            # Save current state
            if hasattr(self, 'save_state'):
                self.save_state(force=True)
            
            # Record shutdown event
            if hasattr(self, 'signal_event'):
                self.signal_event(
                    EventType.AGENT_STATE_CHANGE,
                    f"Agent {self.name} preparing for shutdown",
                    context={"shutdown_preparation": True},
                    metadata={"clean_shutdown": True}
                )
            
            self.logger.info(f"Agent {self.name} prepared for shutdown")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown preparation for {self.name}: {e}")


class AgentCommunicationBus:
    """Communication bus for agent coordination."""
    
    def __init__(self):
        self.agents: Dict[AgentRole, BaseAgent] = {}
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.collaboration_sessions: Dict[str, Dict[str, Any]] = {}
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the communication bus."""
        self.agents[agent.role] = agent
    
    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """Register handler for specific message type."""
        self.message_handlers[message_type] = handler
    
    async def facilitate_handoff(self, from_agent: AgentRole, to_agent: AgentRole, 
                                task_id: str) -> str:
        """Facilitate task handoff between agents."""
        if from_agent not in self.agents or to_agent not in self.agents:
            raise ValueError("Both agents must be registered")
        
        sender = self.agents[from_agent]
        recipient = self.agents[to_agent]
        
        # Get handoff context
        handoff_context = sender.get_context_for_handoff(task_id)
        
        # Send handoff message
        message_id = sender.send_message(
            recipient=to_agent,
            message_type=MessageType.STRATEGIC_HANDOFF,
            payload=handoff_context,
            requires_response=True,
            context_id=task_id
        )
        
        # Update task status
        sender.update_task_status(task_id, "handed_off", handoff_to=to_agent.value)
        
        return message_id
    
    async def start_collaboration_session(self, agents: List[AgentRole], 
                                        problem_definition: str) -> str:
        """Start collaborative problem-solving session."""
        session_id = f"collab_{int(time.time() * 1000)}"
        
        session = {
            "session_id": session_id,
            "participants": [agent.value for agent in agents],
            "problem_definition": problem_definition,
            "start_time": time.time(),
            "status": "active",
            "shared_context": {},
            "decisions": [],
            "action_items": []
        }
        
        self.collaboration_sessions[session_id] = session
        
        # Notify all participating agents
        for agent_role in agents:
            if agent_role in self.agents:
                self.agents[agent_role].send_message(
                    recipient=agent_role,
                    message_type=MessageType.COLLABORATIVE_SESSION,
                    payload={
                        "session_id": session_id,
                        "problem_definition": problem_definition,
                        "participants": [a.value for a in agents]
                    }
                )
        
        return session_id


# Initialize the communication bus
communication_bus = AgentCommunicationBus()