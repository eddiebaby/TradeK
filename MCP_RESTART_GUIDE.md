# MCP Server Restart Guide

This guide provides comprehensive instructions for reinitializing MCP servers after a restart or system reboot.

## Quick Start (TL;DR)

```bash
# Run the initialization script
./initialize_mcp_servers.sh

# Check server health
./check_mcp_health.sh

# Set missing API keys (if needed)
export PERPLEXITY_API_KEY='your_key_here'
export GITHUB_PERSONAL_ACCESS_TOKEN='your_key_here'

# Restart Claude Code session
```

## Complete Restart Procedure

### 1. Initialize All MCP Servers

```bash
cd /home/scott/TradeKnowledge
./initialize_mcp_servers.sh
```

This script will:
- ✅ Check all environment variables
- ✅ Initialize Zen MCP Server virtual environment
- ✅ Install all Python dependencies
- ✅ Create necessary directories
- ✅ Test server connectivity
- ✅ Report server status

### 2. Verify Server Health

```bash
./check_mcp_health.sh
```

This will show you:
- Environment variable status
- Server file integrity
- Dependency availability
- Basic connectivity tests
- Overall health summary

### 3. Configure Missing API Keys (Optional)

If you want full functionality, set these environment variables:

```bash
# For Perplexity search functionality
export PERPLEXITY_API_KEY='your_perplexity_api_key'

# For GitHub integration
export GITHUB_PERSONAL_ACCESS_TOKEN='your_github_token'

# Make permanent by adding to ~/.bashrc or ~/.zshrc
echo 'export PERPLEXITY_API_KEY="your_key"' >> ~/.bashrc
echo 'export GITHUB_PERSONAL_ACCESS_TOKEN="your_token"' >> ~/.bashrc
```

### 4. Restart Claude Code

After initialization, restart your Claude Code session to reload the MCP configuration.

## Server Status Overview

| Server | Status | Requirements | Functionality |
|--------|--------|--------------|---------------|
| Filesystem | ✅ Ready | Built-in | File operations |
| SQLite | ✅ Ready | Built-in | Database queries |
| GitHub | ⚠️ Needs API Key | GITHUB_PERSONAL_ACCESS_TOKEN | Repository access |
| Perplexity | ⚠️ Needs API Key | PERPLEXITY_API_KEY | Web search |
| Zen | ✅ Ready | GEMINI_API_KEY (set) | AI conversations |
| Context7 | ✅ Ready | Built-in | Documentation |
| Sequential Thinking | ✅ Ready | Built-in | Problem solving |
| Memory | ✅ Ready | Built-in | Knowledge graph |

## Troubleshooting

### Common Issues

1. **"MCP server not found"**
   - Run `./initialize_mcp_servers.sh`
   - Check if virtual environment exists
   - Verify file paths in `.mcp.json`

2. **"API key not set"**
   - Set required environment variables
   - Add to shell profile for persistence
   - Restart Claude Code session

3. **"Python dependencies missing"**
   - Zen server: `cd mcp-management/servers/zen-mcp-server && source .zen_venv/bin/activate && pip install -r requirements.txt`
   - Perplexity server: `cd perplexity-mcp && pip install -r requirements.txt`

4. **"Permission denied"**
   - Make scripts executable: `chmod +x *.sh`
   - Check file ownership: `ls -la`

### Manual Server Testing

Test individual servers manually:

```bash
# Test Zen MCP Server
cd /home/scott/TradeKnowledge/mcp-management/servers/zen-mcp-server
source .zen_venv/bin/activate
python server.py --version

# Test Perplexity MCP Server
cd /home/scott/TradeKnowledge/perplexity-mcp
python3 src/perplexity_mcp/server.py --version

# Test standard MCP servers
npx -y @modelcontextprotocol/server-filesystem --version
npx -y @modelcontextprotocol/server-sqlite --version
```

### Logs and Debugging

Check server logs for issues:

```bash
# Zen server logs
tail -f /home/scott/TradeKnowledge/mcp-management/servers/zen-mcp-server/logs/mcp_server.log

# Claude Code logs (if available)
# Check Claude Code documentation for log locations
```

## Files Created/Modified

This restart procedure creates these files:

- `initialize_mcp_servers.sh` - Main initialization script
- `check_mcp_health.sh` - Health check and verification
- `MCP_RESTART_GUIDE.md` - This guide
- `.mcp.json` - MCP server configuration (verified/updated)

## Environment Variables Reference

### Required for Full Functionality
```bash
export GEMINI_API_KEY="your_gemini_key"              # ✅ Already set
export PERPLEXITY_API_KEY="your_perplexity_key"     # ⚠️ Optional
export GITHUB_PERSONAL_ACCESS_TOKEN="your_gh_token" # ⚠️ Optional
```

### Auto-configured Defaults
```bash
export GITHUB_TOOLSETS="repository,issues,pull_requests"
export GITHUB_READ_ONLY="true"
export PERPLEXITY_MODEL="llama-3.1-sonar-large-128k-online"
```

## Next Steps After Restart

1. **Test MCP Integration**: Try using MCP tools in Claude Code
2. **Verify Agent Functionality**: Test the SPARC trio agents
3. **Check Database Access**: Verify SQLite database connectivity
4. **Test Web Search**: Use Perplexity search (if API key set)
5. **GitHub Integration**: Test repository access (if token set)

## Regular Maintenance

- Run health check weekly: `./check_mcp_health.sh`
- Update dependencies monthly: `./initialize_mcp_servers.sh`
- Monitor server logs for errors
- Keep API keys current and secure

---

**Note**: This guide assumes you're running the TradeKnowledge project setup. Adjust paths and configurations as needed for different environments.