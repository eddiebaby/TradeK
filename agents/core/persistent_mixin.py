"""
Persistent Agent Mixin for TradeKnowledge Agents

This mixin provides persistent state capabilities to agents, allowing them to:
- Save and restore their state across crashes/restarts
- Record events and signals in persistent memory
- Track workflow progress and context
- Maintain continuity during handoffs

Usage:
    class MyAgent(PersistentAgentMixin, BaseAgent):
        def __init__(self):
            super().__init__()
            self.agent_name = "my_agent"
            self.restore_state()
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from pathlib import Path

import sys
from pathlib import Path

# Add project root to path for src imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from src.core.persistent_state import (
        PersistentStateManager, 
        get_state_manager,
        EventType,
        DocumentType
    )
except ImportError:
    # Fallback if src module not available
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
        
    class PersistentStateManager:
        def __init__(self):
            pass
        def record_event(self, *args, **kwargs):
            pass
        def save_agent_state(self, *args, **kwargs):
            pass
        def restore_agent_state(self, *args, **kwargs):
            return {}
            
    def get_state_manager():
        return PersistentStateManager()

logger = logging.getLogger(__name__)


@dataclass
class TaskProgress:
    """Track progress for a specific task"""
    task_id: str
    description: str
    started_at: str
    phase: str
    progress: float  # 0.0 to 1.0
    substeps_completed: List[str]
    next_action: str
    context: Dict[str, Any]
    
    @classmethod
    def create(cls, task_id: str, description: str, phase: str = "started") -> 'TaskProgress':
        return cls(
            task_id=task_id,
            description=description,
            started_at=datetime.now(timezone.utc).isoformat(),
            phase=phase,
            progress=0.0,
            substeps_completed=[],
            next_action="",
            context={}
        )


@dataclass
class HandoffContext:
    """Context for agent handoffs"""
    from_agent: str
    to_agent: str
    reason: str
    timestamp: str
    task_context: Dict[str, Any]
    completion_status: Dict[str, Any]
    
    @classmethod
    def create(cls, from_agent: str, to_agent: str, reason: str,
               task_context: Dict[str, Any] = None,
               completion_status: Dict[str, Any] = None) -> 'HandoffContext':
        return cls(
            from_agent=from_agent,
            to_agent=to_agent,
            reason=reason,
            timestamp=datetime.now(timezone.utc).isoformat(),
            task_context=task_context or {},
            completion_status=completion_status or {}
        )


class PersistentAgentMixin:
    """
    Mixin class that adds persistent state capabilities to agents.
    
    This mixin provides:
    - Automatic state saving and restoration
    - Event/signal recording
    - Task progress tracking
    - Handoff context management
    - Workflow continuity
    """
    
    def __init__(self, agent_name: str = None):
        """
        Initialize persistent capabilities.
        
        Args:
            agent_name: Name of the agent (e.g., "mastermind", "executor", "researcher")
        """
        # Agent identification
        if not hasattr(self, 'agent_name') and agent_name:
            self.agent_name = agent_name
        elif not hasattr(self, 'agent_name'):
            self.agent_name = self.__class__.__name__.lower()
        
        # State manager
        self.state_manager = get_state_manager()
        
        # Current state
        self.current_task: Optional[TaskProgress] = None
        self.workflow_context: Dict[str, Any] = {}
        self.memory_context: Dict[str, Any] = {}
        
        # Auto-save configuration
        self.auto_save_enabled = True
        self.save_interval_seconds = 30
        self._last_save_time = 0
        
        # Event callbacks
        self._event_callbacks: Dict[EventType, List[Callable]] = {}
        
        logger.info(f"Initialized persistent capabilities for agent: {self.agent_name}")

    def save_state(self, force: bool = False) -> None:
        """
        Save current agent state to persistent storage.
        
        Args:
            force: Force save even if auto-save interval hasn't elapsed
        """
        try:
            current_time = datetime.now().timestamp()
            
            # Check if we should save (respecting interval unless forced)
            if not force and self.auto_save_enabled:
                if current_time - self._last_save_time < self.save_interval_seconds:
                    return
            
            # Prepare state data
            state_data = {
                "current_task": asdict(self.current_task) if self.current_task else None,
                "workflow_context": self.workflow_context,
                "memory_context": self.memory_context,
                "last_active": datetime.now(timezone.utc).isoformat(),
                "class_name": self.__class__.__name__,
                "capabilities": self._get_agent_capabilities(),
                "performance_metrics": self._get_performance_metrics()
            }
            
            # Add agent-specific state if method exists
            if hasattr(self, '_get_persistent_state'):
                agent_specific_state = self._get_persistent_state()
                if agent_specific_state:
                    state_data["agent_specific"] = agent_specific_state
            
            # Save to persistent storage
            self.state_manager.update_agent_state(self.agent_name, state_data)
            self._last_save_time = current_time
            
            logger.debug(f"Saved state for agent {self.agent_name}")
            
        except Exception as e:
            logger.error(f"Error saving state for agent {self.agent_name}: {e}")

    def restore_state(self) -> bool:
        """
        Restore agent state from persistent storage.
        
        Returns:
            bool: True if state was successfully restored
        """
        try:
            # Get saved state
            saved_state = self.state_manager.get_agent_state(self.agent_name)
            
            if not saved_state:
                logger.info(f"No saved state found for agent {self.agent_name}")
                return False
            
            # Restore current task
            if saved_state.get("current_task"):
                task_data = saved_state["current_task"]
                self.current_task = TaskProgress(**task_data)
                logger.info(f"Restored current task: {self.current_task.task_id}")
            
            # Restore contexts
            self.workflow_context = saved_state.get("workflow_context", {})
            self.memory_context = saved_state.get("memory_context", {})
            
            # Restore agent-specific state if method exists
            if hasattr(self, '_restore_persistent_state'):
                agent_specific_state = saved_state.get("agent_specific", {})
                if agent_specific_state:
                    self._restore_persistent_state(agent_specific_state)
            
            # Record restoration event
            self.signal_event(
                EventType.AGENT_STATE_CHANGE,
                f"Agent {self.agent_name} state restored from persistent storage",
                context={"restored_at": datetime.now(timezone.utc).isoformat()},
                metadata={"has_current_task": self.current_task is not None}
            )
            
            logger.info(f"Successfully restored state for agent {self.agent_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring state for agent {self.agent_name}: {e}")
            return False

    def start_task(self, task_id: str, description: str, phase: str = "started") -> TaskProgress:
        """
        Start a new task and save progress.
        
        Args:
            task_id: Unique identifier for the task
            description: Human-readable task description
            phase: Current phase of the task
            
        Returns:
            TaskProgress: The created task progress object
        """
        # Complete previous task if exists
        if self.current_task:
            self.complete_task("interrupted_by_new_task")
        
        # Create new task
        self.current_task = TaskProgress.create(task_id, description, phase)
        
        # Record event
        self.signal_event(
            EventType.WORKFLOW_START,
            f"Started task: {description}",
            context={
                "task_id": task_id,
                "phase": phase,
                "agent": self.agent_name
            },
            metadata={"task_type": "agent_task"}
        )
        
        # Save state
        self.save_state(force=True)
        
        logger.info(f"Agent {self.agent_name} started task: {task_id}")
        return self.current_task

    def update_task_progress(self, progress: float, phase: str = None, 
                           next_action: str = None, context: Dict[str, Any] = None) -> None:
        """
        Update progress for the current task.
        
        Args:
            progress: Progress value between 0.0 and 1.0
            phase: New phase name
            next_action: Description of next action
            context: Additional context data
        """
        if not self.current_task:
            logger.warning(f"No current task to update for agent {self.agent_name}")
            return
        
        # Update task progress
        self.current_task.progress = max(0.0, min(1.0, progress))
        
        if phase:
            self.current_task.phase = phase
        
        if next_action:
            self.current_task.next_action = next_action
        
        if context:
            self.current_task.context.update(context)
        
        # Record progress event
        self.signal_event(
            EventType.TASK_COMPLETION,
            f"Task progress updated: {progress:.1%} ({self.current_task.phase})",
            context={
                "task_id": self.current_task.task_id,
                "progress": progress,
                "phase": self.current_task.phase,
                "next_action": next_action
            }
        )
        
        # Auto-save if significant progress
        if progress > 0 and progress % 0.25 < 0.1:  # Save at 25%, 50%, 75%, 100%
            self.save_state(force=True)

    def complete_task(self, completion_reason: str = "completed",
                     artifacts: List[str] = None,
                     quality_score: float = None) -> None:
        """
        Complete the current task.
        
        Args:
            completion_reason: Reason for completion
            artifacts: List of files/artifacts created
            quality_score: Quality assessment (0.0 to 1.0)
        """
        if not self.current_task:
            logger.warning(f"No current task to complete for agent {self.agent_name}")
            return
        
        # Update task
        self.current_task.progress = 1.0
        self.current_task.next_action = "completed"
        
        # Record completion event
        completion_context = {
            "task_id": self.current_task.task_id,
            "completion_reason": completion_reason,
            "duration_seconds": self._calculate_task_duration(),
            "artifacts_created": artifacts or []
        }
        
        if quality_score is not None:
            completion_context["quality_score"] = quality_score
        
        self.signal_event(
            EventType.TASK_COMPLETION,
            f"Completed task: {self.current_task.description}",
            context=completion_context,
            metadata={
                "completion_reason": completion_reason,
                "success": completion_reason == "completed"
            }
        )
        
        # Clear current task
        self.current_task = None
        
        # Save state
        self.save_state(force=True)
        
        logger.info(f"Agent {self.agent_name} completed task: {completion_reason}")

    def handoff_to_agent(self, target_agent: str, reason: str,
                        task_context: Dict[str, Any] = None,
                        completion_status: Dict[str, Any] = None) -> HandoffContext:
        """
        Initiate handoff to another agent.
        
        Args:
            target_agent: Name of the target agent
            reason: Reason for handoff
            task_context: Context to pass to target agent
            completion_status: Status of current work
            
        Returns:
            HandoffContext: The handoff context object
        """
        # Create handoff context
        handoff_context = HandoffContext.create(
            from_agent=self.agent_name,
            to_agent=target_agent,
            reason=reason,
            task_context=task_context,
            completion_status=completion_status
        )
        
        # Record handoff event
        self.signal_event(
            EventType.HANDOFF,
            f"Handoff from {self.agent_name} to {target_agent}: {reason}",
            context=asdict(handoff_context),
            metadata={
                "handoff_type": "agent_to_agent",
                "reason_category": self._categorize_handoff_reason(reason)
            }
        )
        
        # Update workflow context
        self.workflow_context["last_handoff"] = asdict(handoff_context)
        
        # Save state before handoff
        self.save_state(force=True)
        
        logger.info(f"Initiated handoff from {self.agent_name} to {target_agent}: {reason}")
        return handoff_context

    def receive_handoff(self, handoff_context: HandoffContext) -> bool:
        """
        Receive handoff from another agent.
        
        Args:
            handoff_context: Context from the handoff
            
        Returns:
            bool: True if handoff was successfully received
        """
        try:
            # Update workflow context
            self.workflow_context["received_handoff"] = asdict(handoff_context)
            self.workflow_context["handoff_chain"] = self.workflow_context.get("handoff_chain", [])
            self.workflow_context["handoff_chain"].append(asdict(handoff_context))
            
            # Extract task context if provided
            if handoff_context.task_context:
                self.memory_context.update(handoff_context.task_context)
            
            # Record received handoff event
            self.signal_event(
                EventType.HANDOFF,
                f"Received handoff from {handoff_context.from_agent}: {handoff_context.reason}",
                context=asdict(handoff_context),
                metadata={
                    "handoff_type": "received",
                    "from_agent": handoff_context.from_agent
                }
            )
            
            # Call agent-specific handoff handler if exists
            if hasattr(self, '_handle_received_handoff'):
                self._handle_received_handoff(handoff_context)
            
            # Save state
            self.save_state(force=True)
            
            logger.info(f"Agent {self.agent_name} received handoff from {handoff_context.from_agent}")
            return True
            
        except Exception as e:
            logger.error(f"Error receiving handoff for agent {self.agent_name}: {e}")
            return False

    def signal_event(self, event_type: EventType, summary: str,
                    context: Dict[str, Any] = None,
                    metadata: Dict[str, Any] = None) -> str:
        """
        Record an event/signal in persistent memory.
        
        Args:
            event_type: Type of event
            summary: Human-readable event summary
            context: Event context data
            metadata: Additional metadata
            
        Returns:
            str: Signal ID
        """
        try:
            # Add agent context to metadata
            if metadata is None:
                metadata = {}
            metadata.update({
                "agent_name": self.agent_name,
                "agent_class": self.__class__.__name__,
                "current_task_id": self.current_task.task_id if self.current_task else None
            })
            
            # Record the signal
            signal_id = self.state_manager.add_signal(
                source_agent=self.agent_name,
                event_type=event_type,
                summary=summary,
                context=context or {},
                metadata=metadata
            )
            
            # Trigger event callbacks
            self._trigger_event_callbacks(event_type, summary, context, metadata)
            
            # Auto-save after important events
            if event_type in [EventType.TASK_COMPLETION, EventType.HANDOFF, EventType.ERROR]:
                self.save_state(force=True)
            
            return signal_id
            
        except Exception as e:
            logger.error(f"Error recording event for agent {self.agent_name}: {e}")
            return ""

    def register_document(self, path: str, description: str, doc_type: DocumentType,
                         status: str = "draft", ai_verifiable_outcome: str = "") -> bool:
        """
        Register a document created by this agent.
        
        Args:
            path: Path to the document
            description: Document description
            doc_type: Type of document
            status: Document status
            ai_verifiable_outcome: AI-verifiable completion criteria
            
        Returns:
            bool: True if document was newly registered
        """
        return self.state_manager.register_document(
            path=path,
            description=description,
            doc_type=doc_type,
            created_by=self.agent_name,
            status=status,
            ai_verifiable_outcome=ai_verifiable_outcome
        )

    def get_recent_signals(self, limit: int = 10,
                          event_type: Optional[EventType] = None) -> List[Dict[str, Any]]:
        """
        Get recent signals for this agent.
        
        Args:
            limit: Maximum number of signals to return
            event_type: Optional filter by event type
            
        Returns:
            List of signal dictionaries
        """
        return self.state_manager.get_signals(
            limit=limit,
            source_agent=self.agent_name,
            event_type=event_type
        )

    def get_workflow_history(self) -> List[Dict[str, Any]]:
        """
        Get workflow history from memory context.
        
        Returns:
            List of workflow events
        """
        return self.workflow_context.get("handoff_chain", [])

    def register_event_callback(self, event_type: EventType, callback: Callable) -> None:
        """
        Register a callback for specific event types.
        
        Args:
            event_type: Type of event to listen for
            callback: Function to call when event occurs
        """
        if event_type not in self._event_callbacks:
            self._event_callbacks[event_type] = []
        self._event_callbacks[event_type].append(callback)

    def set_workflow_context(self, context: Dict[str, Any]) -> None:
        """
        Set workflow context for this agent.
        
        Args:
            context: Workflow context data
        """
        self.workflow_context.update(context)
        self.save_state()

    def add_memory_context(self, key: str, value: Any) -> None:
        """
        Add information to agent's memory context.
        
        Args:
            key: Context key
            value: Context value
        """
        self.memory_context[key] = value
        self.save_state()

    def cleanup_completed_tasks(self, days_old: int = 7) -> int:
        """
        Clean up old completed task data.
        
        Args:
            days_old: Remove task data older than this many days
            
        Returns:
            int: Number of tasks cleaned up
        """
        # This would be implemented to clean up old task data
        # For now, just return 0
        return 0

    def _get_agent_capabilities(self) -> List[str]:
        """Get list of agent capabilities for state storage"""
        capabilities = []
        
        # Check for common agent methods
        if hasattr(self, 'conduct_research'):
            capabilities.append("research")
        if hasattr(self, 'execute_implementation'):
            capabilities.append("implementation")
        if hasattr(self, 'coordinate_workflow'):
            capabilities.append("coordination")
        if hasattr(self, 'process_document'):
            capabilities.append("document_processing")
        
        return capabilities

    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for state storage"""
        metrics = {}
        
        if self.current_task:
            metrics["current_task_progress"] = self.current_task.progress
            metrics["current_task_duration"] = self._calculate_task_duration()
        
        # Get recent signal count as activity metric
        recent_signals = len(self.get_recent_signals(limit=10))
        metrics["recent_activity_level"] = recent_signals
        
        return metrics

    def _calculate_task_duration(self) -> float:
        """Calculate duration of current task in seconds"""
        if not self.current_task:
            return 0.0
        
        try:
            start_time = datetime.fromisoformat(self.current_task.started_at.replace('Z', '+00:00'))
            current_time = datetime.now(timezone.utc)
            return (current_time - start_time).total_seconds()
        except:
            return 0.0

    def _categorize_handoff_reason(self, reason: str) -> str:
        """Categorize handoff reason for better tracking"""
        reason_lower = reason.lower()
        
        if "complete" in reason_lower or "finish" in reason_lower:
            return "completion"
        elif "research" in reason_lower or "investigate" in reason_lower:
            return "research_needed"
        elif "implement" in reason_lower or "code" in reason_lower:
            return "implementation_needed"
        elif "error" in reason_lower or "fail" in reason_lower:
            return "error_recovery"
        elif "coordinate" in reason_lower or "plan" in reason_lower:
            return "coordination_needed"
        else:
            return "other"

    def _trigger_event_callbacks(self, event_type: EventType, summary: str,
                                context: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        """Trigger registered callbacks for an event type"""
        if event_type in self._event_callbacks:
            for callback in self._event_callbacks[event_type]:
                try:
                    callback(event_type, summary, context, metadata)
                except Exception as e:
                    logger.error(f"Error in event callback for {event_type}: {e}")

    def __del__(self):
        """Ensure state is saved when agent is destroyed"""
        try:
            if hasattr(self, 'state_manager') and hasattr(self, 'agent_name'):
                self.save_state(force=True)
        except:
            pass  # Ignore errors during cleanup