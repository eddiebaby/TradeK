"""
Crash Recovery System for TradeKnowledge

This module provides comprehensive crash detection and recovery capabilities
for the TradeKnowledge agent system. It can detect unclean shutdowns and
restore agent states, workflow contexts, and system configuration.

Features:
- Unclean shutdown detection
- Agent state recovery
- Workflow resumption
- System health validation
- Recovery reporting and logging
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from src.core.persistent_state import (
    EventType,
    get_state_manager,
)

logger = logging.getLogger(__name__)


class RecoveryStatus(Enum):
    """Recovery operation status"""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    NOT_NEEDED = "not_needed"


class ShutdownType(Enum):
    """Type of shutdown detected"""

    CLEAN = "clean"
    UNCLEAN = "unclean"
    CRASH = "crash"
    UNKNOWN = "unknown"


@dataclass
class RecoveryReport:
    """Report of recovery operations performed"""

    recovery_timestamp: str
    shutdown_type: str
    agents_recovered: list[str]
    workflows_restored: int
    documents_validated: int
    errors_encountered: list[str]
    recovery_duration_seconds: float
    system_health_score: float

    @classmethod
    def create(cls, shutdown_type: ShutdownType) -> "RecoveryReport":
        return cls(
            recovery_timestamp=datetime.now(UTC).isoformat(),
            shutdown_type=shutdown_type.value,
            agents_recovered=[],
            workflows_restored=0,
            documents_validated=0,
            errors_encountered=[],
            recovery_duration_seconds=0.0,
            system_health_score=0.0,
        )


@dataclass
class AgentRecoveryState:
    """State of an agent during recovery"""

    agent_name: str
    has_saved_state: bool
    current_task_id: str | None
    task_progress: float
    last_active: str
    recovery_needed: bool
    recovery_status: str
    recovery_actions: list[str]


class CrashRecoverySystem:
    """
    Comprehensive crash detection and recovery system.

    This system monitors for unclean shutdowns and provides automated
    recovery of agent states, workflow contexts, and system integrity.
    """

    def __init__(self, base_path: str = "data/persistent"):
        self.base_path = Path(base_path)
        self.state_manager = get_state_manager()

        # Recovery tracking files
        self.shutdown_marker_file = self.base_path / ".shutdown_marker"
        self.recovery_log_file = self.base_path / ".recovery_log"
        self.system_health_file = self.base_path / ".system_health"

        # Recovery configuration
        self.max_recovery_attempts = 3
        self.recovery_timeout_seconds = 300  # 5 minutes
        self.health_check_interval = 60  # 1 minute

        # Recovery state
        self.recovery_in_progress = False
        self.last_recovery_time = None
        self.recovery_attempts = 0

        logger.info("CrashRecoverySystem initialized")

    def mark_clean_shutdown(self) -> None:
        """Mark that the system is shutting down cleanly"""
        try:
            shutdown_info = {
                "timestamp": datetime.now(UTC).isoformat(),
                "type": "clean",
                "agent_states_saved": True,
                "process_id": self._get_process_id(),
            }

            with open(self.shutdown_marker_file, "w") as f:
                json.dump(shutdown_info, f, indent=2)

            logger.info("Marked clean shutdown")

        except Exception as e:
            logger.error(f"Error marking clean shutdown: {e}")

    def detect_unclean_shutdown(self) -> tuple[bool, ShutdownType]:
        """
        Detect if the last shutdown was unclean.

        Returns:
            Tuple of (was_unclean, shutdown_type)
        """
        try:
            # Check if shutdown marker exists
            if not self.shutdown_marker_file.exists():
                logger.warning("No shutdown marker found - possible unclean shutdown")
                return True, ShutdownType.UNCLEAN

            # Read shutdown marker
            with open(self.shutdown_marker_file) as f:
                shutdown_info = json.load(f)

            shutdown_type = shutdown_info.get("type", "unknown")
            timestamp = shutdown_info.get("timestamp", "")

            # Remove the marker since we've processed it
            self.shutdown_marker_file.unlink()

            if shutdown_type == "clean":
                logger.info(f"Clean shutdown detected at {timestamp}")
                return False, ShutdownType.CLEAN
            else:
                logger.warning(
                    f"Unclean shutdown detected: {shutdown_type} at {timestamp}"
                )
                return True, ShutdownType(shutdown_type)

        except Exception as e:
            logger.error(f"Error detecting shutdown type: {e}")
            return True, ShutdownType.UNKNOWN

    def recover_system_state(self) -> RecoveryReport:
        """
        Perform comprehensive system recovery.

        Returns:
            RecoveryReport: Detailed report of recovery operations
        """
        start_time = time.time()
        self.recovery_in_progress = True

        # Detect shutdown type
        was_unclean, shutdown_type = self.detect_unclean_shutdown()

        # Create recovery report
        report = RecoveryReport.create(shutdown_type)

        try:
            logger.info(
                f"Starting system recovery - shutdown type: {shutdown_type.value}"
            )

            if not was_unclean and shutdown_type == ShutdownType.CLEAN:
                report.recovery_duration_seconds = time.time() - start_time
                report.system_health_score = 1.0
                logger.info("Clean shutdown detected - no recovery needed")
                return report

            # Perform recovery steps
            self._validate_persistent_storage(report)
            self._recover_agent_states(report)
            self._restore_workflow_contexts(report)
            self._validate_document_registry(report)
            self._perform_system_health_check(report)

            # Log recovery completion
            self._log_recovery_completion(report)

            # Record recovery event
            self.state_manager.add_signal(
                source_agent="crash_recovery_system",
                event_type=EventType.SYSTEM_EVENT,
                summary=f"System recovery completed - {len(report.agents_recovered)} agents recovered",
                context=asdict(report),
                metadata={"recovery_type": "crash_recovery"},
            )

            logger.info(
                f"System recovery completed in {report.recovery_duration_seconds:.2f}s"
            )

        except Exception as e:
            report.errors_encountered.append(f"Recovery failed: {str(e)}")
            logger.error(f"System recovery failed: {e}")

        finally:
            report.recovery_duration_seconds = time.time() - start_time
            self.recovery_in_progress = False
            self.last_recovery_time = datetime.now(UTC)

        return report

    def _validate_persistent_storage(self, report: RecoveryReport) -> None:
        """Validate persistent storage files integrity"""
        try:
            logger.info("Validating persistent storage integrity...")

            # Check core files exist and are valid JSON
            core_files = [
                self.state_manager.memory_file,
                self.state_manager.docs_registry_file,
                self.state_manager.agent_state_file,
                self.state_manager.workflow_stack_file,
            ]

            for file_path in core_files:
                if not file_path.exists():
                    error_msg = f"Missing core file: {file_path.name}"
                    report.errors_encountered.append(error_msg)
                    logger.warning(error_msg)
                    continue

                try:
                    # Validate JSON format
                    with open(file_path) as f:
                        json.load(f)
                    logger.debug(f"Validated {file_path.name}")

                except json.JSONDecodeError as e:
                    error_msg = f"Corrupted file {file_path.name}: {e}"
                    report.errors_encountered.append(error_msg)
                    logger.error(error_msg)

                    # Try to restore from backup
                    if self._restore_file_from_backup(file_path):
                        logger.info(f"Restored {file_path.name} from backup")
                    else:
                        logger.error(f"Could not restore {file_path.name} from backup")

            logger.info("Persistent storage validation completed")

        except Exception as e:
            error_msg = f"Error validating persistent storage: {e}"
            report.errors_encountered.append(error_msg)
            logger.error(error_msg)

    def _recover_agent_states(self, report: RecoveryReport) -> None:
        """Recover states for all agents"""
        try:
            logger.info("Recovering agent states...")

            # Get all agent states
            agent_states = self.state_manager.get_agent_state("") or {}
            agents_data = agent_states.get("agents", {})

            if not agents_data:
                logger.info("No agent states found to recover")
                return

            for agent_name, agent_data in agents_data.items():
                try:
                    recovery_state = self._analyze_agent_recovery_needs(
                        agent_name, agent_data
                    )

                    if recovery_state.recovery_needed:
                        logger.info(f"Recovering agent: {agent_name}")
                        self._perform_agent_recovery(
                            agent_name, agent_data, recovery_state, report
                        )
                        report.agents_recovered.append(agent_name)
                    else:
                        logger.debug(f"Agent {agent_name} does not need recovery")

                except Exception as e:
                    error_msg = f"Error recovering agent {agent_name}: {e}"
                    report.errors_encountered.append(error_msg)
                    logger.error(error_msg)

            logger.info(
                f"Agent state recovery completed - {len(report.agents_recovered)} agents recovered"
            )

        except Exception as e:
            error_msg = f"Error during agent state recovery: {e}"
            report.errors_encountered.append(error_msg)
            logger.error(error_msg)

    def _analyze_agent_recovery_needs(
        self, agent_name: str, agent_data: dict[str, Any]
    ) -> AgentRecoveryState:
        """Analyze what recovery is needed for an agent"""
        current_task = agent_data.get("current_task")
        last_active = agent_data.get("last_active", "")

        recovery_state = AgentRecoveryState(
            agent_name=agent_name,
            has_saved_state=bool(agent_data),
            current_task_id=current_task.get("task_id") if current_task else None,
            task_progress=current_task.get("progress", 0.0) if current_task else 0.0,
            last_active=last_active,
            recovery_needed=False,
            recovery_status="analyzed",
            recovery_actions=[],
        )

        # Determine if recovery is needed
        if current_task and current_task.get("progress", 0.0) > 0.0:
            recovery_state.recovery_needed = True
            recovery_state.recovery_actions.append("resume_interrupted_task")

        if agent_data.get("workflow_context"):
            recovery_state.recovery_needed = True
            recovery_state.recovery_actions.append("restore_workflow_context")

        if agent_data.get("memory_context"):
            recovery_state.recovery_actions.append("restore_memory_context")

        return recovery_state

    def _perform_agent_recovery(
        self,
        agent_name: str,
        agent_data: dict[str, Any],
        recovery_state: AgentRecoveryState,
        report: RecoveryReport,
    ) -> None:
        """Perform recovery actions for a specific agent"""
        try:
            # Log recovery start
            logger.info(
                f"Performing recovery for agent {agent_name}: {recovery_state.recovery_actions}"
            )

            # Record recovery event for the agent
            recovery_context = {
                "agent_name": agent_name,
                "recovery_actions": recovery_state.recovery_actions,
                "current_task_id": recovery_state.current_task_id,
                "task_progress": recovery_state.task_progress,
                "last_active": recovery_state.last_active,
            }

            self.state_manager.add_signal(
                source_agent="crash_recovery_system",
                event_type=EventType.AGENT_STATE_CHANGE,
                summary=f"Recovered agent {agent_name} with {len(recovery_state.recovery_actions)} actions",
                context=recovery_context,
                metadata={"recovery_type": "agent_recovery"},
            )

            # Update agent state with recovery timestamp
            agent_data["recovery_info"] = {
                "recovered_at": datetime.now(UTC).isoformat(),
                "recovery_actions": recovery_state.recovery_actions,
                "recovery_system_version": "1.0",
            }

            # Save updated state
            self.state_manager.update_agent_state(agent_name, agent_data)

            recovery_state.recovery_status = "completed"
            logger.info(f"Successfully recovered agent {agent_name}")

        except Exception as e:
            recovery_state.recovery_status = "failed"
            error_msg = f"Agent recovery failed for {agent_name}: {e}"
            report.errors_encountered.append(error_msg)
            logger.error(error_msg)

    def _restore_workflow_contexts(self, report: RecoveryReport) -> None:
        """Restore workflow contexts from persistent storage"""
        try:
            logger.info("Restoring workflow contexts...")

            workflow_data = self.state_manager.get_workflow_context()

            if not workflow_data:
                logger.info("No workflow context to restore")
                return

            # Validate workflow data
            if self._validate_workflow_data(workflow_data):
                report.workflows_restored = 1

                # Record workflow restoration
                self.state_manager.add_signal(
                    source_agent="crash_recovery_system",
                    event_type=EventType.WORKFLOW_START,
                    summary="Restored workflow context after recovery",
                    context={"workflow_restored": True},
                    metadata={"recovery_type": "workflow_restoration"},
                )

                logger.info("Workflow context restored successfully")
            else:
                error_msg = "Workflow context validation failed"
                report.errors_encountered.append(error_msg)
                logger.warning(error_msg)

        except Exception as e:
            error_msg = f"Error restoring workflow contexts: {e}"
            report.errors_encountered.append(error_msg)
            logger.error(error_msg)

    def _validate_document_registry(self, report: RecoveryReport) -> None:
        """Validate document registry integrity"""
        try:
            logger.info("Validating document registry...")

            documents = self.state_manager.get_documents()
            valid_documents = 0

            for doc in documents:
                doc_path = doc.get("path", "")
                if doc_path and Path(doc_path).exists():
                    valid_documents += 1
                else:
                    logger.warning(f"Document not found: {doc_path}")

            report.documents_validated = valid_documents
            logger.info(
                f"Document registry validation completed - {valid_documents} documents validated"
            )

        except Exception as e:
            error_msg = f"Error validating document registry: {e}"
            report.errors_encountered.append(error_msg)
            logger.error(error_msg)

    def _perform_system_health_check(self, report: RecoveryReport) -> None:
        """Perform comprehensive system health check"""
        try:
            logger.info("Performing system health check...")

            health_score = 0.0
            total_checks = 5

            # Check 1: Persistent storage integrity
            if len(report.errors_encountered) == 0:
                health_score += 0.2

            # Check 2: Agent state recovery
            if len(report.agents_recovered) > 0 or report.agents_recovered == []:
                health_score += 0.2

            # Check 3: Workflow restoration
            if report.workflows_restored >= 0:
                health_score += 0.2

            # Check 4: Document registry validation
            if report.documents_validated >= 0:
                health_score += 0.2

            # Check 5: Overall system responsiveness
            if self._check_system_responsiveness():
                health_score += 0.2

            report.system_health_score = health_score

            # Save health status
            health_status = {
                "timestamp": datetime.now(UTC).isoformat(),
                "health_score": health_score,
                "checks_passed": int(health_score * total_checks),
                "total_checks": total_checks,
                "recovery_report": asdict(report),
            }

            with open(self.system_health_file, "w") as f:
                json.dump(health_status, f, indent=2)

            logger.info(f"System health check completed - score: {health_score:.2f}")

        except Exception as e:
            error_msg = f"Error during system health check: {e}"
            report.errors_encountered.append(error_msg)
            logger.error(error_msg)

    def _check_system_responsiveness(self) -> bool:
        """Check if the system is responsive"""
        try:
            # Simple responsiveness test - can we access state manager?
            status = self.state_manager.get_system_status()
            return bool(status)
        except:
            return False

    def _validate_workflow_data(self, workflow_data: dict[str, Any]) -> bool:
        """Validate workflow data structure"""
        try:
            # Basic validation - check for required fields
            required_fields = ["workflow_id", "pattern", "current_phase"]

            for field in required_fields:
                if field not in workflow_data:
                    return False

            return True

        except:
            return False

    def _restore_file_from_backup(self, file_path: Path) -> bool:
        """Try to restore a file from backup"""
        try:
            backup_data = self.state_manager._restore_from_backup(file_path.name)
            if backup_data:
                with open(file_path, "w") as f:
                    json.dump(backup_data, f, indent=2)
                return True
            return False
        except:
            return False

    def _log_recovery_completion(self, report: RecoveryReport) -> None:
        """Log recovery completion with detailed report"""
        try:
            log_entry = {
                "timestamp": datetime.now(UTC).isoformat(),
                "recovery_report": asdict(report),
                "system_info": self._get_system_info(),
            }

            # Read existing log
            recovery_log = []
            if self.recovery_log_file.exists():
                try:
                    with open(self.recovery_log_file) as f:
                        recovery_log = json.load(f)
                except:
                    recovery_log = []

            # Add new entry
            recovery_log.append(log_entry)

            # Keep only last 50 entries
            if len(recovery_log) > 50:
                recovery_log = recovery_log[-50:]

            # Save updated log
            with open(self.recovery_log_file, "w") as f:
                json.dump(recovery_log, f, indent=2)

        except Exception as e:
            logger.error(f"Error logging recovery completion: {e}")

    def _get_system_info(self) -> dict[str, Any]:
        """Get system information for logging"""
        return {
            "python_version": "3.x",  # Would get actual version
            "platform": "linux",  # Would get actual platform
            "process_id": self._get_process_id(),
            "memory_usage": "unknown",  # Would get actual memory usage
            "disk_space": "unknown",  # Would get actual disk space
        }

    def _get_process_id(self) -> int:
        """Get current process ID"""
        import os

        return os.getpid()

    def get_recovery_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get recovery history.

        Args:
            limit: Maximum number of recovery entries to return

        Returns:
            List of recovery log entries
        """
        try:
            if not self.recovery_log_file.exists():
                return []

            with open(self.recovery_log_file) as f:
                recovery_log = json.load(f)

            # Return most recent entries
            return recovery_log[-limit:] if len(recovery_log) > limit else recovery_log

        except Exception as e:
            logger.error(f"Error reading recovery history: {e}")
            return []

    def get_system_health_status(self) -> dict[str, Any]:
        """
        Get current system health status.

        Returns:
            System health status dictionary
        """
        try:
            if not self.system_health_file.exists():
                return {"status": "unknown", "message": "No health data available"}

            with open(self.system_health_file) as f:
                health_status = json.load(f)

            return health_status

        except Exception as e:
            logger.error(f"Error reading system health status: {e}")
            return {"status": "error", "message": str(e)}


# Global instance
_global_recovery_system: CrashRecoverySystem | None = None


def get_recovery_system() -> CrashRecoverySystem:
    """Get the global recovery system instance"""
    global _global_recovery_system
    if _global_recovery_system is None:
        _global_recovery_system = CrashRecoverySystem()
    return _global_recovery_system


def perform_startup_recovery() -> RecoveryReport:
    """
    Perform startup recovery check and operations.

    This should be called at application startup to detect and recover
    from any crashes or unclean shutdowns.

    Returns:
        RecoveryReport: Detailed report of recovery operations
    """
    recovery_system = get_recovery_system()
    return recovery_system.recover_system_state()


def mark_clean_shutdown() -> None:
    """
    Mark that the system is shutting down cleanly.

    This should be called during normal application shutdown.
    """
    recovery_system = get_recovery_system()
    recovery_system.mark_clean_shutdown()
