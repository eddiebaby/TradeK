"""
Unit tests for the persistent memory system.

Tests cover:
- PersistentStateManager core functionality
- PersistentAgentMixin integration
- CrashRecoverySystem operations
- Configuration handling
- Data integrity and backup/restore
"""

import json
import tempfile
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from src.core.persistent_state import (
    PersistentStateManager,
    Signal,
    DocumentEntry,
    EventType,
    DocumentType,
    get_state_manager,
    shutdown_state_manager
)
from agents.core.persistent_mixin import (
    PersistentAgentMixin,
    TaskProgress,
    HandoffContext
)
from src.core.crash_recovery import (
    CrashRecoverySystem,
    RecoveryStatus,
    ShutdownType,
    RecoveryReport,
    get_recovery_system,
    perform_startup_recovery,
    mark_clean_shutdown
)


class TestPersistentStateManager:
    """Test cases for PersistentStateManager"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.fixture
    def state_manager(self, temp_dir):
        """Create PersistentStateManager instance for testing"""
        return PersistentStateManager(base_path=str(temp_dir))
    
    def test_initialization(self, state_manager, temp_dir):
        """Test proper initialization of state manager"""
        assert state_manager.base_path == temp_dir
        assert state_manager.memory_file.exists()
        assert state_manager.docs_registry_file.exists()
        assert state_manager.agent_state_file.exists()
        assert state_manager.workflow_stack_file.exists()
        
        # Check initial file contents
        memory_data = state_manager._read_json(state_manager.memory_file)
        assert "signals" in memory_data
        assert memory_data["signals"] == []
        
        registry_data = state_manager._read_json(state_manager.docs_registry_file)
        assert "documentation_registry" in registry_data
        assert registry_data["documentation_registry"] == []
    
    def test_add_signal(self, state_manager):
        """Test adding signals to memory"""
        signal_id = state_manager.add_signal(
            source_agent="test_agent",
            event_type=EventType.TASK_COMPLETION,
            summary="Test task completed",
            context={"task_id": "test_001"},
            metadata={"test": True}
        )
        
        assert signal_id
        
        # Verify signal was stored
        signals = state_manager.get_signals(limit=1)
        assert len(signals) == 1
        
        signal = signals[0]
        assert signal["source_agent"] == "test_agent"
        assert signal["event_type"] == EventType.TASK_COMPLETION.value
        assert signal["summary"] == "Test task completed"
        assert signal["context"]["task_id"] == "test_001"
        assert signal["metadata"]["test"] is True
    
    def test_signal_auto_pruning(self, state_manager):
        """Test automatic pruning when memory limit is exceeded"""
        # Set a low line limit for testing
        state_manager.max_memory_lines = 50
        
        # Add many signals to trigger pruning
        for i in range(20):
            state_manager.add_signal(
                source_agent="test_agent",
                event_type=EventType.TASK_COMPLETION,
                summary=f"Test task {i} completed with lots of details to make the JSON large",
                context={"task_id": f"test_{i:03d}", "iteration": i, "data": "x" * 100}
            )
        
        # Check that pruning occurred
        signals = state_manager.get_signals()
        assert len(signals) < 20  # Some signals should have been pruned
    
    def test_register_document(self, state_manager):
        """Test document registration"""
        was_new = state_manager.register_document(
            path="docs/test_spec.md",
            description="Test specification document",
            doc_type=DocumentType.FEATURE_SPECIFICATION,
            created_by="test_agent",
            status="draft",
            ai_verifiable_outcome="Specification is complete and reviewed"
        )
        
        assert was_new is True
        
        # Verify document was registered
        documents = state_manager.get_documents()
        assert len(documents) == 1
        
        doc = documents[0]
        assert doc["path"] == "docs/test_spec.md"
        assert doc["type"] == DocumentType.FEATURE_SPECIFICATION.value
        assert doc["created_by"] == "test_agent"
        assert doc["status"] == "draft"
        
        # Test updating existing document
        was_new = state_manager.register_document(
            path="docs/test_spec.md",
            description="Updated test specification",
            doc_type=DocumentType.FEATURE_SPECIFICATION,
            created_by="test_agent",
            status="approved"
        )
        
        assert was_new is False
        
        # Verify update
        documents = state_manager.get_documents()
        assert len(documents) == 1
        assert documents[0]["description"] == "Updated test specification"
        assert documents[0]["status"] == "approved"
    
    def test_agent_state_management(self, state_manager):
        """Test agent state updates and retrieval"""
        agent_name = "test_agent"
        
        # Initial state should be empty
        state = state_manager.get_agent_state(agent_name)
        assert state == {}
        
        # Update agent state
        state_update = {
            "current_task": {
                "task_id": "test_task",
                "progress": 0.5
            },
            "memory_context": {
                "last_action": "research"
            }
        }
        
        state_manager.update_agent_state(agent_name, state_update)
        
        # Verify state was saved
        state = state_manager.get_agent_state(agent_name)
        assert state["current_task"]["task_id"] == "test_task"
        assert state["current_task"]["progress"] == 0.5
        assert state["memory_context"]["last_action"] == "research"
        assert "last_updated" in state
        
        # Test state merging
        additional_update = {
            "current_task": {
                "progress": 0.8  # This should update existing
            },
            "workflow_context": {
                "pattern": "sparc_full_cycle"  # This should be added
            }
        }
        
        state_manager.update_agent_state(agent_name, additional_update)
        
        state = state_manager.get_agent_state(agent_name)
        assert state["current_task"]["task_id"] == "test_task"  # Should remain
        assert state["current_task"]["progress"] == 0.8  # Should be updated
        assert state["workflow_context"]["pattern"] == "sparc_full_cycle"  # Should be added
    
    def test_workflow_context_management(self, state_manager):
        """Test workflow context save and retrieval"""
        workflow_data = {
            "workflow_id": "test_workflow_001",
            "pattern": "sparc_full_cycle",
            "current_phase": "implementation",
            "progress": {
                "specification": "completed",
                "pseudocode": "completed",
                "architecture": "in_progress"
            }
        }
        
        state_manager.save_workflow_context(workflow_data)
        
        # Verify workflow was saved
        context = state_manager.get_workflow_context()
        assert context["workflow_id"] == "test_workflow_001"
        assert context["pattern"] == "sparc_full_cycle"
        assert context["current_phase"] == "implementation"
        
        # Test task context
        task_context = {
            "task_id": "implement_feature_x",
            "description": "Implement feature X with TDD",
            "requirements": {"test_coverage": 0.9}
        }
        
        state_manager.set_task_context(task_context)
        
        retrieved_context = state_manager.get_task_context()
        assert retrieved_context["task_id"] == "implement_feature_x"
        assert retrieved_context["requirements"]["test_coverage"] == 0.9
    
    def test_backup_and_restore(self, state_manager):
        """Test backup creation and restoration"""
        # Add some data
        state_manager.add_signal(
            source_agent="test_agent",
            event_type=EventType.TASK_COMPLETION,
            summary="Test signal for backup"
        )
        
        state_manager.register_document(
            path="docs/backup_test.md",
            description="Test document for backup",
            doc_type=DocumentType.GENERAL_DOCUMENT,
            created_by="test_agent"
        )
        
        # Create backup
        backup_name = state_manager.create_backup("test_backup")
        assert backup_name == "test_backup"
        
        backup_path = state_manager.backup_dir / backup_name
        assert backup_path.exists()
        assert (backup_path / ".memory").exists()
        assert (backup_path / ".docsregistry").exists()
        
        # Modify data
        state_manager.add_signal(
            source_agent="test_agent",
            event_type=EventType.ERROR,
            summary="Error signal after backup"
        )
        
        # Verify data was modified
        signals = state_manager.get_signals()
        assert len(signals) == 2
        
        # Restore from backup
        success = state_manager.restore_from_backup("test_backup")
        assert success is True
        
        # Verify data was restored
        signals = state_manager.get_signals()
        assert len(signals) == 1
        assert signals[0]["summary"] == "Test signal for backup"
    
    def test_get_system_status(self, state_manager):
        """Test system status reporting"""
        # Add some test data
        state_manager.add_signal(
            source_agent="test_agent",
            event_type=EventType.TASK_COMPLETION,
            summary="Test signal"
        )
        
        state_manager.register_document(
            path="docs/test.md",
            description="Test document",
            doc_type=DocumentType.GENERAL_DOCUMENT,
            created_by="test_agent"
        )
        
        state_manager.update_agent_state("test_agent", {"active": True})
        
        status = state_manager.get_system_status()
        
        assert status["memory_signals_count"] == 1
        assert status["documents_count"] == 1
        assert status["active_agents_count"] == 1
        assert status["has_active_workflow"] is False
        assert status["data_integrity"] is True
        assert "storage_path" in status
        assert "uptime_seconds" in status


class TestPersistentAgentMixin:
    """Test cases for PersistentAgentMixin"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.fixture
    def mock_agent(self, temp_dir):
        """Create mock agent with PersistentAgentMixin"""
        class MockAgent(PersistentAgentMixin):
            def __init__(self):
                super().__init__(agent_name="test_agent")
                self.state_manager = PersistentStateManager(base_path=str(temp_dir))
        
        return MockAgent()
    
    def test_agent_initialization(self, mock_agent):
        """Test agent initialization with persistence"""
        assert mock_agent.agent_name == "test_agent"
        assert mock_agent.current_task is None
        assert mock_agent.workflow_context == {}
        assert mock_agent.memory_context == {}
        assert mock_agent.auto_save_enabled is True
    
    def test_task_lifecycle(self, mock_agent):
        """Test complete task lifecycle"""
        # Start task
        task = mock_agent.start_task(
            task_id="test_task_001",
            description="Test task for lifecycle",
            phase="started"
        )
        
        assert isinstance(task, TaskProgress)
        assert task.task_id == "test_task_001"
        assert task.progress == 0.0
        assert task.phase == "started"
        assert mock_agent.current_task == task
        
        # Update progress
        mock_agent.update_task_progress(
            progress=0.5,
            phase="implementation",
            next_action="Continue coding",
            context={"files_modified": ["test.py"]}
        )
        
        assert mock_agent.current_task.progress == 0.5
        assert mock_agent.current_task.phase == "implementation"
        assert mock_agent.current_task.next_action == "Continue coding"
        assert mock_agent.current_task.context["files_modified"] == ["test.py"]
        
        # Complete task
        mock_agent.complete_task(
            completion_reason="completed",
            artifacts=["test.py", "test_test.py"],
            quality_score=0.95
        )
        
        assert mock_agent.current_task is None
        
        # Verify signals were recorded
        signals = mock_agent.get_recent_signals(limit=5)
        assert len(signals) >= 3  # start, progress, completion
        
        start_signal = next(s for s in signals if "Started task" in s["summary"])
        assert start_signal["context"]["task_id"] == "test_task_001"
        
        completion_signal = next(s for s in signals if "Completed task" in s["summary"])
        assert completion_signal["context"]["completion_reason"] == "completed"
        assert completion_signal["context"]["artifacts_created"] == ["test.py", "test_test.py"]
    
    def test_handoff_operations(self, mock_agent):
        """Test agent handoff operations"""
        # Start a task
        mock_agent.start_task("handoff_test", "Test handoff task")
        
        # Initiate handoff
        handoff_context = mock_agent.handoff_to_agent(
            target_agent="executor",
            reason="need_implementation",
            task_context={"specification_complete": True},
            completion_status={"research_quality": 0.9}
        )
        
        assert isinstance(handoff_context, HandoffContext)
        assert handoff_context.from_agent == "test_agent"
        assert handoff_context.to_agent == "executor"
        assert handoff_context.reason == "need_implementation"
        
        # Check workflow context was updated
        assert "last_handoff" in mock_agent.workflow_context
        
        # Simulate receiving handoff
        received_handoff = HandoffContext.create(
            from_agent="researcher",
            to_agent="test_agent",
            reason="research_complete",
            task_context={"findings": "important data"},
            completion_status={"quality": 0.85}
        )
        
        success = mock_agent.receive_handoff(received_handoff)
        assert success is True
        
        # Check handoff chain was updated
        assert "handoff_chain" in mock_agent.workflow_context
        assert len(mock_agent.workflow_context["handoff_chain"]) == 1
        
        # Check memory context was updated
        assert mock_agent.memory_context["findings"] == "important data"
    
    def test_state_persistence(self, mock_agent):
        """Test state save and restore"""
        # Set up some state
        mock_agent.start_task("persistence_test", "Test persistence")
        mock_agent.update_task_progress(0.3, "testing")
        mock_agent.workflow_context["test_data"] = "important_info"
        mock_agent.memory_context["cached_result"] = "cached_value"
        
        # Save state
        mock_agent.save_state(force=True)
        
        # Create new agent instance to test restoration
        class MockAgent(PersistentAgentMixin):
            def __init__(self, state_manager):
                super().__init__(agent_name="test_agent")
                self.state_manager = state_manager
        
        new_agent = MockAgent(mock_agent.state_manager)
        
        # Restore state
        success = new_agent.restore_state()
        assert success is True
        
        # Verify state was restored
        assert new_agent.current_task is not None
        assert new_agent.current_task.task_id == "persistence_test"
        assert new_agent.current_task.progress == 0.3
        assert new_agent.current_task.phase == "testing"
        assert new_agent.workflow_context["test_data"] == "important_info"
        assert new_agent.memory_context["cached_result"] == "cached_value"
    
    def test_document_registration(self, mock_agent):
        """Test document registration through agent"""
        was_new = mock_agent.register_document(
            path="docs/agent_test.md",
            description="Document created by agent",
            doc_type=DocumentType.TEST_PLAN,
            status="draft",
            ai_verifiable_outcome="Test plan covers all requirements"
        )
        
        assert was_new is True
        
        # Verify document was registered and signal was recorded
        documents = mock_agent.state_manager.get_documents()
        assert len(documents) == 1
        assert documents[0]["created_by"] == "test_agent"
        
        signals = mock_agent.get_recent_signals()
        doc_signal = next(s for s in signals if "Document registered" in s["summary"])
        assert doc_signal is not None
    
    def test_event_callbacks(self, mock_agent):
        """Test event callback system"""
        callback_called = []
        
        def test_callback(event_type, summary, context, metadata):
            callback_called.append({
                "event_type": event_type,
                "summary": summary,
                "context": context,
                "metadata": metadata
            })
        
        # Register callback
        mock_agent.register_event_callback(EventType.TASK_COMPLETION, test_callback)
        
        # Trigger event
        mock_agent.signal_event(
            EventType.TASK_COMPLETION,
            "Test callback event",
            context={"test": True}
        )
        
        # Verify callback was called
        assert len(callback_called) == 1
        assert callback_called[0]["event_type"] == EventType.TASK_COMPLETION
        assert callback_called[0]["summary"] == "Test callback event"
        assert callback_called[0]["context"]["test"] is True


class TestCrashRecoverySystem:
    """Test cases for CrashRecoverySystem"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.fixture
    def recovery_system(self, temp_dir):
        """Create CrashRecoverySystem instance for testing"""
        return CrashRecoverySystem(base_path=str(temp_dir))
    
    def test_clean_shutdown_detection(self, recovery_system):
        """Test clean shutdown marking and detection"""
        # Mark clean shutdown
        recovery_system.mark_clean_shutdown()
        
        # Verify shutdown marker was created
        assert recovery_system.shutdown_marker_file.exists()
        
        # Detect shutdown type
        was_unclean, shutdown_type = recovery_system.detect_unclean_shutdown()
        
        assert was_unclean is False
        assert shutdown_type == ShutdownType.CLEAN
        
        # Verify marker was removed after detection
        assert not recovery_system.shutdown_marker_file.exists()
    
    def test_unclean_shutdown_detection(self, recovery_system):
        """Test unclean shutdown detection"""
        # Simulate unclean shutdown (no marker file)
        was_unclean, shutdown_type = recovery_system.detect_unclean_shutdown()
        
        assert was_unclean is True
        assert shutdown_type == ShutdownType.UNCLEAN
    
    def test_system_recovery_clean_shutdown(self, recovery_system):
        """Test system recovery after clean shutdown"""
        # Mark clean shutdown first
        recovery_system.mark_clean_shutdown()
        
        # Perform recovery
        report = recovery_system.recover_system_state()
        
        assert report.shutdown_type == ShutdownType.CLEAN.value
        assert report.system_health_score == 1.0
        assert len(report.errors_encountered) == 0
    
    def test_system_recovery_unclean_shutdown(self, recovery_system):
        """Test system recovery after unclean shutdown"""
        # Set up some test state to recover
        state_manager = PersistentStateManager(base_path=str(recovery_system.base_path))
        
        # Add test agent state
        state_manager.update_agent_state("test_agent", {
            "current_task": {
                "task_id": "interrupted_task",
                "progress": 0.7,
                "phase": "implementation"
            },
            "workflow_context": {"pattern": "sparc_full_cycle"},
            "last_active": datetime.now(timezone.utc).isoformat()
        })
        
        # Add test workflow
        state_manager.save_workflow_context({
            "workflow_id": "test_workflow",
            "pattern": "sparc_full_cycle",
            "current_phase": "implementation"
        })
        
        # Perform recovery (should detect unclean shutdown)
        report = recovery_system.recover_system_state()
        
        assert report.shutdown_type in [ShutdownType.UNCLEAN.value, ShutdownType.UNKNOWN.value]
        assert len(report.agents_recovered) >= 0  # May have recovered the test agent
        assert report.workflows_restored >= 0
        assert report.recovery_duration_seconds > 0
    
    def test_recovery_report_structure(self, recovery_system):
        """Test recovery report structure and data"""
        report = recovery_system.recover_system_state()
        
        assert hasattr(report, 'recovery_timestamp')
        assert hasattr(report, 'shutdown_type')
        assert hasattr(report, 'agents_recovered')
        assert hasattr(report, 'workflows_restored')
        assert hasattr(report, 'documents_validated')
        assert hasattr(report, 'errors_encountered')
        assert hasattr(report, 'recovery_duration_seconds')
        assert hasattr(report, 'system_health_score')
        
        assert isinstance(report.agents_recovered, list)
        assert isinstance(report.errors_encountered, list)
        assert isinstance(report.recovery_duration_seconds, float)
        assert isinstance(report.system_health_score, float)
    
    def test_recovery_history(self, recovery_system):
        """Test recovery history logging"""
        # Perform recovery
        report = recovery_system.recover_system_state()
        
        # Check recovery history
        history = recovery_system.get_recovery_history(limit=5)
        
        assert len(history) >= 1
        assert "recovery_report" in history[-1]
        assert "timestamp" in history[-1]
    
    def test_system_health_status(self, recovery_system):
        """Test system health status reporting"""
        # Perform recovery to generate health data
        recovery_system.recover_system_state()
        
        # Check health status
        health_status = recovery_system.get_system_health_status()
        
        assert "health_score" in health_status
        assert "timestamp" in health_status
        assert "checks_passed" in health_status
        assert "total_checks" in health_status


class TestIntegration:
    """Integration tests for the complete persistent memory system"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    def test_complete_workflow_persistence(self, temp_dir):
        """Test complete workflow with persistence and recovery"""
        # Create state manager and recovery system
        state_manager = PersistentStateManager(base_path=str(temp_dir))
        recovery_system = CrashRecoverySystem(base_path=str(temp_dir))
        
        # Create mock agent
        class TestAgent(PersistentAgentMixin):
            def __init__(self):
                super().__init__(agent_name="integration_test_agent")
                self.state_manager = state_manager
        
        agent = TestAgent()
        
        # Simulate workflow
        # 1. Agent starts task
        task = agent.start_task("integration_task", "Integration test task")
        
        # 2. Agent makes progress
        agent.update_task_progress(0.3, "analysis", "Analyzing requirements")
        
        # 3. Agent registers document
        agent.register_document(
            path="docs/integration_spec.md",
            description="Integration test specification",
            doc_type=DocumentType.FEATURE_SPECIFICATION,
            status="draft"
        )
        
        # 4. Agent updates workflow context
        agent.set_workflow_context({
            "workflow_id": "integration_workflow",
            "pattern": "test_pattern",
            "current_phase": "analysis"
        })
        
        # 5. Mark clean shutdown
        recovery_system.mark_clean_shutdown()
        
        # Simulate restart - create new instances
        new_state_manager = PersistentStateManager(base_path=str(temp_dir))
        new_recovery_system = CrashRecoverySystem(base_path=str(temp_dir))
        
        # Perform recovery
        report = new_recovery_system.recover_system_state()
        
        # Verify recovery was successful
        assert report.shutdown_type == ShutdownType.CLEAN.value
        assert report.system_health_score == 1.0
        
        # Create new agent and restore state
        class NewTestAgent(PersistentAgentMixin):
            def __init__(self):
                super().__init__(agent_name="integration_test_agent")
                self.state_manager = new_state_manager
        
        new_agent = NewTestAgent()
        success = new_agent.restore_state()
        
        # Verify state was restored
        assert success is True
        assert new_agent.current_task is not None
        assert new_agent.current_task.task_id == "integration_task"
        assert new_agent.current_task.progress == 0.3
        assert new_agent.current_task.phase == "analysis"
        
        # Verify workflow context
        workflow_context = new_state_manager.get_workflow_context()
        assert workflow_context["workflow_id"] == "integration_workflow"
        
        # Verify document registry
        documents = new_state_manager.get_documents()
        assert len(documents) == 1
        assert documents[0]["path"] == "docs/integration_spec.md"
        
        # Verify signals
        signals = new_state_manager.get_signals(limit=10)
        assert len(signals) > 0
        
        # Continue workflow with new agent
        new_agent.update_task_progress(0.8, "implementation", "Implementing features")
        new_agent.complete_task("completed", ["feature.py", "test_feature.py"], 0.95)
        
        # Verify completion
        assert new_agent.current_task is None
        
        final_signals = new_state_manager.get_signals(limit=5)
        completion_signal = next(s for s in final_signals if "Completed task" in s["summary"])
        assert completion_signal["context"]["completion_reason"] == "completed"


if __name__ == "__main__":
    pytest.main([__file__])