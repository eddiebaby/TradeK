"""
Persistent State Management System for TradeKnowledge

This module provides comprehensive persistent storage capabilities to prevent
data loss during crashes, VSCode restarts, or system failures. Based on
successful patterns from Roo-orchestrated projects.

Key Features:
- Event/signal history with auto-pruning (.memory)
- Document registry tracking (.docsregistry)
- Agent-specific state persistence (.agentstate)
- Workflow context preservation (.workflowstack)
- Automatic backup and recovery
- Atomic operations and data integrity
"""

import json
import logging
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types for memory signals"""

    TASK_COMPLETION = "task_completion"
    HANDOFF = "handoff"
    ERROR = "error"
    WORKFLOW_START = "workflow_start"
    WORKFLOW_STARTED = "workflow_started"  # Alias for WORKFLOW_START
    WORKFLOW_END = "workflow_end"
    QUALITY_GATE = "quality_gate"
    AGENT_STATE_CHANGE = "agent_state_change"
    DOCUMENT_CREATED = "document_created"
    SYSTEM_EVENT = "system_event"
    SYSTEM_CHECKPOINT = "system_checkpoint"
    RECOVERY_COMPLETED = "recovery_completed"


class DocumentType(Enum):
    """Document types for registry"""

    FEATURE_SPECIFICATION = "Feature Specification"
    TEST_PLAN = "Test Plan"
    ARCHITECTURE = "Architecture"
    RESEARCH_REPORT = "Research Report"
    IMPLEMENTATION_REPORT = "Implementation Report"
    CONFIGURATION = "Configuration"
    GENERAL_DOCUMENT = "General Document"


@dataclass
class Signal:
    """Individual memory signal/event"""

    id: str
    timestamp: str
    source_agent: str
    event_type: str
    summary: str
    context: dict[str, Any]
    metadata: dict[str, Any]

    @classmethod
    def create(
        cls,
        source_agent: str,
        event_type,
        summary: str,
        context: dict[str, Any] = None,
        metadata: dict[str, Any] = None,
    ) -> "Signal":
        """Create a new signal with auto-generated ID and timestamp"""
        # Handle both string and enum event types
        if hasattr(event_type, "value"):
            event_type_str = event_type.value
        else:
            event_type_str = str(event_type)

        return cls(
            id=f"uuid-{datetime.now(UTC).isoformat()}-{uuid.uuid4().hex[:10]}",
            timestamp=datetime.now(UTC).isoformat(),
            source_agent=source_agent,
            event_type=event_type_str,
            summary=summary,
            context=context or {},
            metadata=metadata or {},
        )


@dataclass
class DocumentEntry:
    """Document registry entry"""

    path: str
    description: str
    type: str
    timestamp: str
    created_by: str
    status: str = "draft"
    related_tasks: list[str] = None
    ai_verifiable_outcome: str = ""

    def __post_init__(self):
        if self.related_tasks is None:
            self.related_tasks = []

    @classmethod
    def create(
        cls,
        path: str,
        description: str,
        doc_type: DocumentType,
        created_by: str,
        status: str = "draft",
        ai_verifiable_outcome: str = "",
    ) -> "DocumentEntry":
        """Create a new document entry"""
        return cls(
            path=path,
            description=description,
            type=doc_type.value,
            timestamp=datetime.now(UTC).isoformat(),
            created_by=created_by,
            status=status,
            related_tasks=[],
            ai_verifiable_outcome=ai_verifiable_outcome,
        )


class PersistentStateManager:
    """
    Central manager for all persistent state operations.

    Handles memory signals, document registry, agent states, and workflow context
    with automatic backup, recovery, and data integrity features.
    """

    def __init__(self, base_path: str = "data/persistent"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Core state files
        self.memory_file = self.base_path / ".memory"
        self.docs_registry_file = self.base_path / ".docsregistry"
        self.agent_state_file = self.base_path / ".agentstate"
        self.workflow_stack_file = self.base_path / ".workflowstack"

        # Backup directory
        self.backup_dir = self.base_path / "backups"
        self.backup_dir.mkdir(exist_ok=True)

        # Configuration
        self.max_memory_signals = 300
        self.max_memory_lines = 300
        self.backup_interval_minutes = 10
        self.backup_retention_hours = 24

        # Thread safety
        self._lock = threading.RLock()
        self._backup_thread = None
        self._shutdown = False

        # Initialize files if they don't exist
        self._initialize_files()

        # Start automatic backup
        self._start_backup_thread()

        logger.info(f"PersistentStateManager initialized at {self.base_path}")

    def _initialize_files(self):
        """Initialize state files with empty structures if they don't exist"""
        with self._lock:
            # Initialize memory file
            if not self.memory_file.exists():
                self._write_json(self.memory_file, {"signals": []})
                logger.info("Initialized empty .memory file")

            # Initialize docs registry file
            if not self.docs_registry_file.exists():
                self._write_json(
                    self.docs_registry_file, {"documentation_registry": []}
                )
                logger.info("Initialized empty .docsregistry file")

            # Initialize agent state file
            if not self.agent_state_file.exists():
                self._write_json(self.agent_state_file, {"agents": {}})
                logger.info("Initialized empty .agentstate file")

            # Initialize workflow stack file
            if not self.workflow_stack_file.exists():
                empty_workflow = {
                    "current_workflow": None,
                    "task_context": None,
                    "workflow_history": [],
                }
                self._write_json(self.workflow_stack_file, empty_workflow)
                logger.info("Initialized empty .workflowstack file")

    def _read_json(self, file_path: Path) -> dict[str, Any]:
        """Safely read JSON file with error handling"""
        try:
            if not file_path.exists():
                return {}

            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
                return data
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Error reading {file_path}: {e}")
            # Try to restore from backup
            backup_data = self._restore_from_backup(file_path.name)
            if backup_data:
                logger.info(f"Restored {file_path} from backup")
                return backup_data
            return {}

    def _write_json(self, file_path: Path, data: dict[str, Any]):
        """Safely write JSON file with atomic operations"""
        try:
            # Write to temporary file first
            temp_file = file_path.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Atomic move
            temp_file.replace(file_path)

        except OSError as e:
            logger.error(f"Error writing {file_path}: {e}")
            raise

    def add_signal(
        self,
        source_agent: str,
        event_type,
        summary: str,
        context: dict[str, Any] = None,
        metadata: dict[str, Any] = None,
    ) -> str:
        """
        Add a new signal to memory with auto-pruning.

        Returns:
            str: The signal ID
        """
        with self._lock:
            signal = Signal.create(source_agent, event_type, summary, context, metadata)

            # Read current memory
            memory_data = self._read_json(self.memory_file)
            signals = memory_data.get("signals", [])

            # Add new signal
            signals.append(asdict(signal))

            # Check if pruning is needed
            memory_data["signals"] = signals

            # Convert to string to check line count
            test_content = json.dumps(memory_data, indent=2)
            line_count = len(test_content.split("\n"))

            if line_count > self.max_memory_lines:
                # Remove 3 oldest signals
                pruned_signals = signals[3:]
                logger.info(
                    f"Pruned 3 old signals from memory (was {len(signals)}, now {len(pruned_signals)})"
                )
                memory_data["signals"] = pruned_signals

            # Write updated memory
            self._write_json(self.memory_file, memory_data)

            logger.debug(
                f"Added signal {signal.id} from {source_agent}: {summary[:50]}..."
            )
            return signal.id

    def get_signals(
        self,
        limit: int | None = None,
        source_agent: str | None = None,
        event_type: EventType | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve signals with optional filtering.

        Args:
            limit: Maximum number of signals to return (most recent first)
            source_agent: Filter by source agent
            event_type: Filter by event type

        Returns:
            List of signal dictionaries
        """
        memory_data = self._read_json(self.memory_file)
        signals = memory_data.get("signals", [])

        # Apply filters
        if source_agent:
            signals = [s for s in signals if s.get("source_agent") == source_agent]

        if event_type:
            signals = [s for s in signals if s.get("event_type") == event_type.value]

        # Sort by timestamp (most recent first)
        signals.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        # Apply limit
        if limit:
            signals = signals[:limit]

        return signals

    def register_document(
        self,
        path: str,
        description: str,
        doc_type: DocumentType,
        created_by: str,
        status: str = "draft",
        ai_verifiable_outcome: str = "",
    ) -> bool:
        """
        Register a new document in the registry.

        Returns:
            bool: True if document was added, False if it already exists
        """
        with self._lock:
            registry_data = self._read_json(self.docs_registry_file)
            registry = registry_data.get("documentation_registry", [])

            # Check if document already exists
            existing = next((doc for doc in registry if doc.get("path") == path), None)
            if existing:
                # Update existing document
                existing.update(
                    {
                        "description": description,
                        "type": doc_type.value,
                        "timestamp": datetime.now(UTC).isoformat(),
                        "status": status,
                        "ai_verifiable_outcome": ai_verifiable_outcome,
                    }
                )
                logger.info(f"Updated document registry entry: {path}")
                was_new = False
            else:
                # Add new document
                doc_entry = DocumentEntry.create(
                    path,
                    description,
                    doc_type,
                    created_by,
                    status,
                    ai_verifiable_outcome,
                )
                registry.append(asdict(doc_entry))
                logger.info(f"Added new document to registry: {path}")
                was_new = True

            # Write updated registry
            registry_data["documentation_registry"] = registry
            self._write_json(self.docs_registry_file, registry_data)

            # Also add a signal about the document
            signal_summary = f"Document {'registered' if was_new else 'updated'}: {path} ({doc_type.value})"
            self.add_signal(
                source_agent=created_by,
                event_type=EventType.DOCUMENT_CREATED,
                summary=signal_summary,
                context={"document_path": path, "document_type": doc_type.value},
                metadata={"new_document": was_new},
            )

            return was_new

    def get_documents(
        self, doc_type: DocumentType | None = None, created_by: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Retrieve documents from registry with optional filtering.

        Args:
            doc_type: Filter by document type
            created_by: Filter by creator

        Returns:
            List of document dictionaries
        """
        registry_data = self._read_json(self.docs_registry_file)
        documents = registry_data.get("documentation_registry", [])

        # Apply filters
        if doc_type:
            documents = [d for d in documents if d.get("type") == doc_type.value]

        if created_by:
            documents = [d for d in documents if d.get("created_by") == created_by]

        # Sort by timestamp (most recent first)
        documents.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        return documents

    def update_agent_state(self, agent_name: str, state_update: dict[str, Any]) -> None:
        """
        Update agent-specific state.

        Args:
            agent_name: Name of the agent (mastermind, executor, researcher)
            state_update: State data to update/merge
        """
        with self._lock:
            state_data = self._read_json(self.agent_state_file)
            agents = state_data.get("agents", {})

            if agent_name not in agents:
                agents[agent_name] = {}

            # Deep merge the state update
            self._deep_merge(agents[agent_name], state_update)

            # Update timestamp
            agents[agent_name]["last_updated"] = datetime.now(UTC).isoformat()

            # Write updated state
            state_data["agents"] = agents
            self._write_json(self.agent_state_file, state_data)

            logger.debug(f"Updated state for agent {agent_name}")

    def get_agent_state(self, agent_name: str) -> dict[str, Any]:
        """
        Get current state for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Agent state dictionary
        """
        state_data = self._read_json(self.agent_state_file)
        agents = state_data.get("agents", {})
        return agents.get(agent_name, {})

    def save_workflow_context(self, workflow_data: dict[str, Any]) -> None:
        """
        Save current workflow context.

        Args:
            workflow_data: Workflow context data
        """
        with self._lock:
            workflow_stack = self._read_json(self.workflow_stack_file)

            # Update current workflow
            workflow_stack["current_workflow"] = workflow_data
            workflow_stack["last_updated"] = datetime.now(UTC).isoformat()

            # Add to history
            if "workflow_history" not in workflow_stack:
                workflow_stack["workflow_history"] = []

            # Keep last 10 workflows in history
            history = workflow_stack["workflow_history"]
            if len(history) >= 10:
                history = history[-9:]  # Keep last 9, will add current as 10th

            history.append(
                {
                    "workflow": workflow_data.copy(),
                    "saved_at": datetime.now(UTC).isoformat(),
                }
            )
            workflow_stack["workflow_history"] = history

            # Write updated workflow stack
            self._write_json(self.workflow_stack_file, workflow_stack)

            logger.debug("Saved workflow context")

    def get_workflow_context(self) -> dict[str, Any]:
        """
        Get current workflow context.

        Returns:
            Current workflow context dictionary
        """
        workflow_stack = self._read_json(self.workflow_stack_file)
        return workflow_stack.get("current_workflow", {})

    def set_task_context(self, task_context: dict[str, Any]) -> None:
        """
        Set current task context.

        Args:
            task_context: Task context data
        """
        with self._lock:
            workflow_stack = self._read_json(self.workflow_stack_file)
            workflow_stack["task_context"] = task_context
            workflow_stack["task_updated"] = datetime.now(UTC).isoformat()

            self._write_json(self.workflow_stack_file, workflow_stack)
            logger.debug("Updated task context")

    def get_task_context(self) -> dict[str, Any]:
        """
        Get current task context.

        Returns:
            Current task context dictionary
        """
        workflow_stack = self._read_json(self.workflow_stack_file)
        return workflow_stack.get("task_context", {})

    def create_backup(self, backup_name: str | None = None) -> str:
        """
        Create a backup of all state files.

        Args:
            backup_name: Optional custom backup name

        Returns:
            str: Backup directory name
        """
        with self._lock:
            if backup_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"backup_{timestamp}"

            backup_path = self.backup_dir / backup_name
            backup_path.mkdir(exist_ok=True)

            # Copy all state files
            state_files = [
                self.memory_file,
                self.docs_registry_file,
                self.agent_state_file,
                self.workflow_stack_file,
            ]

            for file_path in state_files:
                if file_path.exists():
                    dest_path = backup_path / file_path.name
                    shutil.copy2(file_path, dest_path)

            logger.info(f"Created backup: {backup_name}")
            return backup_name

    def restore_from_backup(self, backup_name: str) -> bool:
        """
        Restore state from a backup.

        Args:
            backup_name: Name of the backup to restore from

        Returns:
            bool: True if restoration was successful
        """
        with self._lock:
            backup_path = self.backup_dir / backup_name
            if not backup_path.exists():
                logger.error(f"Backup not found: {backup_name}")
                return False

            try:
                # Restore all state files
                state_files = [
                    self.memory_file,
                    self.docs_registry_file,
                    self.agent_state_file,
                    self.workflow_stack_file,
                ]

                for file_path in state_files:
                    backup_file = backup_path / file_path.name
                    if backup_file.exists():
                        shutil.copy2(backup_file, file_path)

                logger.info(f"Restored from backup: {backup_name}")
                return True

            except Exception as e:
                logger.error(f"Error restoring from backup {backup_name}: {e}")
                return False

    def _restore_from_backup(self, filename: str) -> dict[str, Any] | None:
        """
        Try to restore a specific file from the most recent backup.

        Args:
            filename: Name of the file to restore

        Returns:
            Restored data or None if not found
        """
        try:
            # Find most recent backup
            backups = sorted(
                [d for d in self.backup_dir.iterdir() if d.is_dir()],
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )

            for backup_dir in backups:
                backup_file = backup_dir / filename
                if backup_file.exists():
                    with open(backup_file, encoding="utf-8") as f:
                        return json.load(f)

            return None

        except Exception as e:
            logger.error(f"Error restoring {filename} from backup: {e}")
            return None

    def cleanup_old_backups(self) -> None:
        """Remove backups older than retention period"""
        try:
            cutoff_time = time.time() - (self.backup_retention_hours * 3600)

            for backup_dir in self.backup_dir.iterdir():
                if backup_dir.is_dir() and backup_dir.stat().st_mtime < cutoff_time:
                    shutil.rmtree(backup_dir)
                    logger.debug(f"Removed old backup: {backup_dir.name}")

        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")

    def _start_backup_thread(self) -> None:
        """Start automatic backup thread"""

        def backup_loop():
            while not self._shutdown:
                try:
                    time.sleep(self.backup_interval_minutes * 60)
                    if not self._shutdown:
                        self.create_backup()
                        self.cleanup_old_backups()
                except Exception as e:
                    logger.error(f"Error in backup thread: {e}")

        self._backup_thread = threading.Thread(target=backup_loop, daemon=True)
        self._backup_thread.start()
        logger.info(
            f"Started automatic backup thread (interval: {self.backup_interval_minutes} minutes)"
        )

    def shutdown(self) -> None:
        """Shutdown the state manager and create final backup"""
        logger.info("Shutting down PersistentStateManager")
        self._shutdown = True

        # Create final backup
        self.create_backup("shutdown_backup")

        # Wait for backup thread
        if self._backup_thread and self._backup_thread.is_alive():
            self._backup_thread.join(timeout=5)

    def get_system_status(self) -> dict[str, Any]:
        """
        Get current system status and health information.

        Returns:
            System status dictionary
        """
        with self._lock:
            memory_data = self._read_json(self.memory_file)
            registry_data = self._read_json(self.docs_registry_file)
            state_data = self._read_json(self.agent_state_file)
            workflow_data = self._read_json(self.workflow_stack_file)

            # Count backups
            backup_count = len([d for d in self.backup_dir.iterdir() if d.is_dir()])

            status = {
                "memory_signals_count": len(memory_data.get("signals", [])),
                "documents_count": len(registry_data.get("documentation_registry", [])),
                "active_agents_count": len(state_data.get("agents", {})),
                "has_active_workflow": workflow_data.get("current_workflow")
                is not None,
                "backup_count": backup_count,
                "last_backup": None,
                "data_integrity": True,
                "storage_path": str(self.base_path),
                "uptime_seconds": time.time() - self.base_path.stat().st_ctime,
            }

            # Find most recent backup
            try:
                backups = sorted(
                    [d for d in self.backup_dir.iterdir() if d.is_dir()],
                    key=lambda x: x.stat().st_mtime,
                    reverse=True,
                )
                if backups:
                    status["last_backup"] = backups[0].name
            except:
                pass

            return status

    @staticmethod
    def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
        """Deep merge source dictionary into target dictionary"""
        for key, value in source.items():
            if (
                key in target
                and isinstance(target[key], dict)
                and isinstance(value, dict)
            ):
                PersistentStateManager._deep_merge(target[key], value)
            else:
                target[key] = value


# Global instance
_global_state_manager: PersistentStateManager | None = None


def get_state_manager() -> PersistentStateManager:
    """Get the global state manager instance"""
    global _global_state_manager
    if _global_state_manager is None:
        _global_state_manager = PersistentStateManager()
    return _global_state_manager


def shutdown_state_manager() -> None:
    """Shutdown the global state manager"""
    global _global_state_manager
    if _global_state_manager:
        _global_state_manager.shutdown()
        _global_state_manager = None
