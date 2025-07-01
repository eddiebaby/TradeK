# SuperClaude Integration Guide

## Overview
SuperClaude is a Chrome extension that enhances the Claude.ai web interface with additional features like conversation saving, enhanced UI/UX, and custom prompt templates. This document outlines how to integrate SuperClaude with the TradeKnowledge SPARC trio system.

## SuperClaude Features
- **Auto-save conversations**: Automatically saves Claude conversations to local storage
- **Enhanced UI/UX**: Improved interface for better user experience  
- **Custom prompts/templates**: Create and manage reusable prompt templates
- **Conversation organization**: Better organization and search of conversation history
- **Export capabilities**: Export conversations in various formats

## Integration Architecture

### 1. Installation and Setup
```bash
# Install SuperClaude Chrome extension from Chrome Web Store
# Configure export directory in extension settings:
Export Directory: ~/Downloads/superclaude_exports
```

### 2. Sync Bridge Components

#### A. SuperClaude Sync Service
**Location**: `src/integrations/superclaude_sync.py`

```python
"""
SuperClaude Integration Service
Syncs conversations and templates between SuperClaude and TradeKnowledge
"""

class SuperClaudeSync:
    def __init__(self):
        self.export_dir = Path("~/Downloads/superclaude_exports").expanduser()
        self.conversations_dir = Path("~/TradeKnowledge/agents/conversations").expanduser()
        self.templates_dir = Path("~/TradeKnowledge/agents/templates").expanduser()
        
    async def sync_conversations(self):
        """Import SuperClaude conversations to SPARC agents"""
        
    async def export_templates(self):
        """Export SPARC agent prompts as SuperClaude templates"""
        
    async def auto_import_loop(self):
        """Continuous monitoring for new exports"""
```

#### B. Conversation Processing Pipeline
1. **Detection**: Monitor export directory for new files
2. **Classification**: Identify conversation type (RESEARCHER, MASTERMIND, EXECUTOR)
3. **Extraction**: Parse conversation content and metadata
4. **Integration**: Add to appropriate agent's conversation history
5. **Indexing**: Update knowledge graph with conversation insights

### 3. Auto-Import Configuration

#### Systemd Service (Linux)
**Create**: `/etc/systemd/user/superclaude-sync.service`
```ini
[Unit]
Description=SuperClaude Sync Service
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 /home/scott/TradeKnowledge/src/integrations/superclaude_sync.py --daemon
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
```

#### Cron Job Alternative
```bash
# Add to crontab (crontab -e):
*/5 * * * * /usr/bin/python3 /home/scott/TradeKnowledge/src/integrations/superclaude_sync.py --sync-once
```

### 4. Template Export System

#### SPARC Agent Template Export
```python
class SPARCTemplateExporter:
    def export_researcher_templates(self):
        """Export RESEARCHER agent prompts as SuperClaude templates"""
        templates = {
            "Market Research": "Analyze market conditions for...",
            "Technical Analysis": "Perform technical analysis on...",
            "Data Investigation": "Research and investigate..."
        }
        
    def export_mastermind_templates(self):
        """Export MASTERMIND agent prompts as SuperClaude templates"""
        
    def export_executor_templates(self):
        """Export EXECUTOR agent prompts as SuperClaude templates"""
```

#### SuperClaude Template Format
```json
{
    "name": "SPARC Research Template",
    "category": "TradeKnowledge",
    "description": "RESEARCHER agent template for market analysis",
    "template": "🔍 RESEARCHER Agent - Market Analysis\n\nAnalyze the following market conditions:\n{market_data}\n\nFocus areas:\n- Trend analysis\n- Volume patterns\n- Support/resistance levels\n- Risk assessment",
    "variables": ["market_data"],
    "tags": ["research", "market", "sparc", "tradeknowledge"]
}
```

### 5. Workflow Integration

#### Daily Workflow
1. **Morning Setup**: 
   - SuperClaude auto-imports overnight conversations
   - Templates sync with latest SPARC agent updates
   - Knowledge graph updates with new insights

2. **Active Work**:
   - Use SuperClaude for web-based Claude interactions
   - Conversations automatically saved and categorized
   - Templates provide quick access to SPARC workflows

3. **Evening Sync**:
   - Final conversation import and processing
   - Export new templates created during the day
   - Update agent knowledge with web conversation insights

#### Integration with SPARC Trio
```python
# Example: Auto-classify conversations by agent type
async def classify_conversation(conversation_text: str) -> str:
    """Classify conversation by SPARC agent type"""
    
    # Keywords for classification
    researcher_keywords = ["research", "analyze", "investigate", "data", "market"]
    mastermind_keywords = ["strategy", "architecture", "design", "plan", "structure"]
    executor_keywords = ["implement", "code", "test", "deploy", "build"]
    
    # Use LLMLingua for efficient classification
    compressed = await llmlingua_service.compress_prompt(conversation_text)
    
    # Classification logic...
    return agent_type
```

### 6. Configuration Management

#### SuperClaude Extension Settings
```json
{
    "exportSettings": {
        "autoExport": true,
        "exportFormat": "json",
        "exportDirectory": "~/Downloads/superclaude_exports",
        "includeMetadata": true,
        "compressionEnabled": true
    },
    "integrationSettings": {
        "tradeKnowledgeSync": true,
        "sparcAgentClassification": true,
        "knowledgeGraphUpdate": true
    }
}
```

#### TradeKnowledge Configuration
Add to `CLAUDE.md`:
```markdown
# SuperClaude Integration
- Auto-import conversations: Check ~/Downloads/superclaude_exports every 5 minutes
- Convert templates to SPARC agent prompts  
- Sync with knowledge graph for context preservation
- Classification: RESEARCHER/MASTERMIND/EXECUTOR based on content analysis
```

### 7. Monitoring and Metrics

#### Sync Metrics
- **Import Rate**: Conversations imported per day
- **Classification Accuracy**: Correct agent type assignment
- **Template Usage**: Most used SuperClaude templates
- **Knowledge Graph Growth**: New nodes/edges from conversations

#### Health Checks
```python
async def health_check():
    """Check SuperClaude integration health"""
    checks = {
        "export_directory_accessible": check_export_dir(),
        "recent_imports": check_recent_activity(),
        "classification_service": check_classifier(),
        "knowledge_graph_sync": check_kg_sync()
    }
    return checks
```

### 8. Security Considerations

#### Data Privacy
- **Local Processing**: All conversation data processed locally
- **Encryption**: Sensitive data encrypted at rest
- **Access Control**: Restrict file system permissions
- **Audit Logging**: Log all import/export activities

#### Content Filtering
```python
def sanitize_conversation(conversation: dict) -> dict:
    """Remove sensitive information from conversations"""
    # Remove API keys, passwords, personal information
    # Redact financial account numbers
    # Filter out proprietary trading strategies
    return sanitized_conversation
```

### 9. Troubleshooting

#### Common Issues
1. **Export Directory Not Found**
   - Verify SuperClaude export path settings
   - Check file system permissions
   - Ensure directory exists and is writable

2. **Classification Errors**
   - Review keyword lists for agent classification
   - Check LLMLingua service availability
   - Validate conversation format

3. **Import Failures**
   - Check file format compatibility
   - Verify JSON structure
   - Review error logs

#### Debug Commands
```bash
# Test SuperClaude sync
python -m src.integrations.superclaude_sync --test

# Manual import
python -m src.integrations.superclaude_sync --import ~/Downloads/superclaude_exports/conversation_123.json

# Health check
python -m src.integrations.superclaude_sync --health
```

### 10. Future Enhancements

#### Planned Features
- **Real-time Sync**: WebSocket connection for instant synchronization
- **Advanced Classification**: ML-based conversation categorization
- **Template Recommendations**: AI-suggested templates based on conversation patterns
- **Cross-Device Sync**: Synchronize across laptop and desktop installations
- **Analytics Dashboard**: Visual analytics for conversation patterns and insights

#### Integration Roadmap
1. **Phase 1**: Basic import/export functionality
2. **Phase 2**: Auto-classification and template sync
3. **Phase 3**: Real-time synchronization
4. **Phase 4**: Advanced analytics and ML features

## Setup Instructions

### Quick Start
1. Install SuperClaude Chrome extension
2. Configure export directory: `~/Downloads/superclaude_exports`
3. Run: `python -m src.integrations.superclaude_sync --setup`
4. Enable auto-sync: Add to startup services

### Verification
```bash
# Test the integration
~/.claude/startup_services.sh
curl http://localhost:8765/health
python -m src.integrations.superclaude_sync --test
```

This integration bridges the gap between web-based Claude interactions and the local TradeKnowledge SPARC trio system, providing seamless workflow continuity across different interfaces.