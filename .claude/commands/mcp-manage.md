# MCP Server Management

Manage Model Context Protocol (MCP) servers for enhanced Claude Code functionality.

## Active MCP Servers

Current enabled servers in TradeKnowledge:
- **filesystem**: File operations and codebase management
- **sqlite**: Database operations and query execution
- **github**: Repository management and CI/CD integration  
- **perplexity**: Real-time web research and intelligence
- **puppeteer**: Web automation and scraping
- **zen-mcp-server**: Multi-model AI provider management
- **context7**: Advanced context management
- **sequential-thinking**: Advanced reasoning capabilities
- **memory**: Persistent agent memory and learning

## Health Check Commands

### Check MCP Server Status
```bash
# Check which MCP servers are running
claude mcp list

# Test specific server connectivity
claude mcp test filesystem
claude mcp test sequential-thinking
claude mcp test memory
```

### Zen MCP Server Management
```bash
# Check Zen server status and available models
cd /home/scott/TradeKnowledge/zen-mcp-server
.zen_venv/bin/python -c "
from providers.gemini import GeminiModelProvider
from providers.openai_provider import OpenAIModelProvider
from providers.xai import XAIModelProvider
print('=== Available Models ===')
print('Gemini:', list(GeminiModelProvider.SUPPORTED_MODELS.keys()))
print('OpenAI:', list(OpenAIModelProvider.SUPPORTED_MODELS.keys()))
print('X.AI:', list(XAIModelProvider.SUPPORTED_MODELS.keys()))
"

# Start/restart Zen MCP server
/home/scott/.local/bin/mcp-start.sh
```

## Configuration Management

### View Current Configuration
```bash
# Check enabled MCP servers
cat /home/scott/TradeKnowledge/.claude/settings.local.json | grep -A 20 "enabledMcpjsonServers"

# Check permissions for MCP tools
cat /home/scott/TradeKnowledge/.claude/settings.local.json | grep -A 10 "mcp__"
```

### Add New MCP Server
To add a new MCP server:
1. Install the MCP server package
2. Add server config to Claude settings
3. Add necessary permissions
4. Test connectivity

Example for adding a new server:
```json
{
  "enabledMcpjsonServers": [
    "filesystem",
    "memory", 
    "new-server-name"
  ]
}
```

## Server-Specific Operations

### Filesystem MCP Operations
```bash
# List directory trees efficiently
# Uses: mcp__filesystem__directory_tree
# Uses: mcp__filesystem__list_directory
# Uses: mcp__filesystem__read_multiple_files
```

### Memory MCP Operations  
```bash
# Agent memory management
# Uses: mcp__memory__create_entities
# Uses: mcp__memory__create_relations
# Uses: mcp__memory__search_nodes
```

### Sequential Thinking MCP Operations
```bash
# Advanced reasoning workflows
# Uses: mcp__sequential-thinking__sequentialthinking
```

## Troubleshooting

### Common Issues

**Server Not Responding:**
```bash
# Check server process
ps aux | grep mcp

# Check logs
journalctl --user -u mcp-servers -f

# Restart MCP services
systemctl --user restart mcp-servers
```

**Permission Denied:**
```bash
# Check permissions in settings.local.json
grep -A 5 -B 5 "permission_name" /home/scott/TradeKnowledge/.claude/settings.local.json

# Add missing permissions to allow list
```

**Model Provider Issues (Zen Server):**
```bash
# Check API key configuration
cd /home/scott/TradeKnowledge/zen-mcp-server
.zen_venv/bin/python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('GEMINI_API_KEY:', 'Found' if os.getenv('GEMINI_API_KEY') else 'Missing')
print('OPENAI_API_KEY:', 'Found' if os.getenv('OPENAI_API_KEY') else 'Missing')
print('XAI_API_KEY:', 'Found' if os.getenv('XAI_API_KEY') else 'Missing')
"

# Test model availability
.zen_venv/bin/python -c "
from providers.gemini import GeminiModelProvider
provider = GeminiModelProvider()
models = provider.get_available_models()
print(f'Available Gemini models: {len(models)}')
"
```

## Performance Optimization

### Server Resource Monitoring
```bash
# Monitor MCP server resource usage
top -p $(pgrep -f mcp)

# Check memory usage by server
ps aux | grep mcp | awk '{print $4, $11}' | sort -nr
```

### Connection Pooling
- Keep MCP connections alive between requests
- Implement connection retry logic for transient failures
- Monitor connection health and automatically reconnect

## Security Considerations

### MCP Server Security
- **Authentication**: Ensure all MCP servers require proper authentication
- **Permissions**: Follow principle of least privilege for tool permissions
- **Network Security**: Restrict MCP server network access as needed
- **Audit Logging**: Log all MCP tool usage for security auditing

### Safe Tool Usage
```bash
# Review potentially dangerous tools before use
grep -E "(Bash|file|delete|remove)" /home/scott/TradeKnowledge/.claude/settings.local.json

# Validate tool parameters before execution
# Always verify file paths and command arguments
```

## Integration with TradeKnowledge

### Financial Data Processing
- Use **filesystem** MCP for reading market data files
- Use **memory** MCP for storing analysis results
- Use **perplexity** MCP for real-time market research

### Agent Coordination  
- Use **memory** MCP for agent-to-agent communication
- Use **sequential-thinking** MCP for complex analysis workflows
- Use **filesystem** MCP for sharing analysis artifacts

### Development Workflow
- Use **github** MCP for repository operations
- Use **filesystem** MCP for code generation and editing
- Use **sqlite** MCP for database operations and testing