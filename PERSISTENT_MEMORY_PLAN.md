# TradeKnowledge Persistent Memory & State Management Plan

## 🎯 **Executive Summary**

Based on analysis of a successful Roo-orchestrated project, this plan implements persistent memory and state management to prevent data loss during crashes, VSCode restarts, or WSL issues.

## 🏗️ **Architecture Overview**

### **Core Components**
1. **`.memory`** - Event/signal history with automatic pruning
2. **`.docsregistry`** - Documentation and artifact tracking  
3. **`.agentstate`** - Agent-specific persistent state
4. **`.workflowstack`** - Current workflow and task context

## 📁 **File Structure**
```
data/persistent/
├── .memory                 # Event history (JSON, auto-pruned at 300 lines)
├── .docsregistry          # Document registry (JSON, never pruned)
├── .agentstate            # Agent states (JSON)
├── .workflowstack         # Current workflow context (JSON)
└── backups/               # Timestamped backups
    ├── memory_backup_YYYYMMDD_HHMMSS.json
    ├── docsregistry_backup_YYYYMMDD_HHMMSS.json
    └── agentstate_backup_YYYYMMDD_HHMMSS.json
```

## 🧠 **1. Memory System (.memory)**

### **Purpose**
Track chronological events, agent handoffs, task completions, and system state changes.

### **Data Structure**
```json
{
  "signals": [
    {
      "id": "uuid-YYYY-MM-DDTHH:mm:ss.sssZ-randomstring",
      "timestamp": "2025-06-18T10:30:45.123Z",
      "source_agent": "mastermind|executor|researcher",
      "event_type": "task_completion|handoff|error|workflow_start|quality_gate",
      "summary": "Detailed description of what happened",
      "context": {
        "task_id": "string",
        "quality_score": 0.95,
        "artifacts_created": ["file1.py", "test_file.py"],
        "handoff_reason": "task_completed_successfully"
      },
      "metadata": {
        "workflow_pattern": "sparc_full_cycle",
        "phase": "implementation",
        "success": true
      }
    }
  ]
}
```

### **Auto-Pruning Logic**
- Maximum 300 lines when serialized
- Remove 3 oldest signals when limit exceeded
- Backup pruned signals before removal

## 📚 **2. Documentation Registry (.docsregistry)**

### **Purpose**
Track all formal project documentation, specifications, test plans, and artifacts.

### **Data Structure**
```json
{
  "documentation_registry": [
    {
      "path": "docs/specifications/task_xyz_spec.md",
      "description": "Feature specification for XYZ functionality",
      "type": "Feature Specification|Test Plan|Architecture|Research Report",
      "timestamp": "2025-06-18T10:30:45.123Z",
      "created_by": "mastermind|executor|researcher",
      "status": "draft|reviewed|approved|implemented",
      "related_tasks": ["task_xyz"],
      "ai_verifiable_outcome": "Description of measurable completion criteria"
    }
  ]
}
```

### **Document Types**
- **Specifications**: Feature and technical specifications
- **Test Plans**: Granular and acceptance test plans  
- **Architecture**: System design documents
- **Research Reports**: Investigation and analysis documents
- **Implementation Reports**: Code quality and completion reports

## 🤖 **3. Agent State (.agentstate)**

### **Purpose** 
Maintain persistent state for each agent to survive crashes and enable seamless recovery.

### **Data Structure**
```json
{
  "agents": {
    "mastermind": {
      "current_task": {
        "task_id": "implement_feature_xyz",
        "started_at": "2025-06-18T10:00:00.000Z",
        "phase": "specification|pseudocode|architecture|refinement|completion",
        "progress": 0.65,
        "context": {}
      },
      "workflow_stack": [
        {
          "pattern": "sparc_full_cycle",
          "step": "architecture",
          "substeps_completed": ["specification", "pseudocode"],
          "next_handoff": "executor"
        }
      ],
      "memory": {
        "recent_decisions": [],
        "quality_gates_status": {},
        "coordination_context": {}
      },
      "tools_state": {
        "openai_agents": {
          "last_used": "2025-06-18T10:15:00.000Z",
          "context_maintained": true
        }
      }
    },
    "executor": {
      "current_implementation": {
        "files_in_progress": ["src/feature_xyz.py"],
        "test_status": "red|green|refactor",
        "quality_metrics": {
          "test_coverage": 0.85,
          "complexity_score": 2.3
        }
      },
      "tdd_cycle": {
        "phase": "red|green|refactor", 
        "iteration": 3,
        "last_test_run": "2025-06-18T10:25:00.000Z"
      }
    },
    "researcher": {
      "research_context": {
        "current_domains": ["technical_analysis", "market_intelligence"],
        "sources_consulted": [],
        "synthesis_progress": 0.40
      },
      "web_search_state": {
        "last_query": "algorithmic trading patterns",
        "results_cached": true
      }
    }
  }
}
```

## 📋 **4. Workflow Stack (.workflowstack)**

### **Purpose**
Maintain current workflow context and coordination state for seamless recovery.

### **Data Structure**
```json
{
  "current_workflow": {
    "workflow_id": "uuid-workflow-id",
    "pattern": "sparc_full_cycle|research_driven|implementation_focused",
    "started_at": "2025-06-18T09:00:00.000Z",
    "current_phase": "architecture",
    "progress": {
      "specification": "completed",
      "pseudocode": "completed", 
      "architecture": "in_progress",
      "refinement": "pending",
      "completion": "pending"
    },
    "handoff_chain": [
      {
        "from": "mastermind",
        "to": "researcher", 
        "timestamp": "2025-06-18T09:15:00.000Z",
        "reason": "need_market_research",
        "completed": true
      },
      {
        "from": "researcher",
        "to": "mastermind",
        "timestamp": "2025-06-18T09:45:00.000Z", 
        "reason": "research_complete",
        "completed": true
      }
    ],
    "quality_gates": {
      "specification_quality": 0.95,
      "test_coverage": 0.88,
      "coordination_quality": 0.92
    }
  },
  "task_context": {
    "task_id": "implement_advanced_search",
    "description": "Implement advanced semantic search with filters",
    "requirements": {},
    "success_criteria": [],
    "ai_verifiable_outcome": "Search API returns filtered results with >0.8 relevance"
  }
}
```

## 🔄 **5. Implementation Components**

### **A. Persistent State Manager**
```python
# src/core/persistent_state.py
class PersistentStateManager:
    def __init__(self, base_path: str = "data/persistent"):
        self.base_path = Path(base_path)
        self.memory_file = self.base_path / ".memory"
        self.docs_registry_file = self.base_path / ".docsregistry"
        self.agent_state_file = self.base_path / ".agentstate"
        self.workflow_stack_file = self.base_path / ".workflowstack"
        
    def add_signal(self, source_agent: str, event_type: str, summary: str, context: dict):
        """Add event to memory with auto-pruning"""
        
    def register_document(self, path: str, description: str, doc_type: str):
        """Register new document in registry"""
        
    def update_agent_state(self, agent_name: str, state_update: dict):
        """Update agent-specific state"""
        
    def save_workflow_context(self, workflow_data: dict):
        """Save current workflow state"""
        
    def restore_from_crash(self) -> dict:
        """Restore all state after crash"""
```

### **B. Agent Integration Mixin**
```python
# agents/core/persistent_mixin.py
class PersistentAgentMixin:
    def __init__(self):
        self.state_manager = PersistentStateManager()
        
    def save_state(self):
        """Save current agent state to persistent storage"""
        
    def restore_state(self):
        """Restore agent state from persistent storage"""
        
    def signal_event(self, event_type: str, summary: str, context: dict = None):
        """Record event in persistent memory"""
```

### **C. Crash Recovery System**
```python
# src/core/crash_recovery.py
class CrashRecoverySystem:
    def detect_unclean_shutdown(self) -> bool:
        """Detect if last shutdown was unclean"""
        
    def recover_workflow_state(self) -> dict:
        """Recover workflow state after crash"""
        
    def resume_agent_tasks(self):
        """Resume interrupted agent tasks"""
```

## 🛡️ **6. Backup & Recovery Strategy**

### **Automatic Backups**
- **Frequency**: Every agent handoff, every 10 minutes during active work
- **Retention**: Keep last 24 hours of backups
- **Location**: `data/persistent/backups/`

### **Recovery Scenarios**
1. **VSCode Crash**: Restore from latest agent state
2. **WSL Restart**: Resume workflow from last checkpoint  
3. **System Crash**: Full state recovery from backups
4. **Corrupted State**: Rollback to last known good state

## ⚡ **7. Performance Optimizations**

### **Lazy Loading**
- Load state files only when needed
- Cache frequently accessed data
- Async background saves

### **Compression**
- Compress old backups  
- Efficient JSON serialization
- Binary storage for large datasets

## 🔐 **8. Data Integrity**

### **Validation**
- JSON schema validation for all state files
- Checksums for backup verification
- Atomic file operations

### **Consistency**
- Transaction-like updates across multiple files
- Rollback capability for failed operations
- State consistency checks on startup

## 🚀 **9. Integration with Existing System**

### **Enhanced Agent Base Classes**
- Add PersistentAgentMixin to all agents
- Automatic state saving on handoffs
- Recovery hooks in agent initialization

### **OpenAI Agents SDK Integration**  
- Persist OpenAI agent contexts
- Save tool usage state
- Restore conversation history

### **Configuration Updates**
```yaml
# config/config.yaml
persistence:
  enabled: true
  base_path: "data/persistent"
  auto_backup_interval_minutes: 10
  max_memory_signals: 300
  backup_retention_hours: 24
  compression_enabled: true
```

## 📊 **10. Success Metrics**

### **Reliability**
- Zero data loss during crashes
- <5 second recovery time
- 99.9% state consistency

### **Performance** 
- <100ms state save operations
- <50MB memory footprint
- Minimal impact on workflow speed

## 🎯 **Next Steps**

1. **Implement Core Persistence Manager** (2-3 hours)
2. **Add Agent Integration Mixins** (1-2 hours)  
3. **Create Crash Recovery System** (2-3 hours)
4. **Add Backup Automation** (1 hour)
5. **Integration Testing** (1-2 hours)

This persistence system will ensure your TradeKnowledge agents can survive any crash and seamlessly resume work exactly where they left off!