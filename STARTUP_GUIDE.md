# TradeKnowledge Startup Guide

Quick reference for starting up the TradeKnowledge project with all MCP servers properly configured.

## Quick Start

```bash
# Navigate to project directory
cd /home/scott/TradeKnowledge

# Load environment variables (recommended)
source ./load_env.sh

# Check MCP server health
./check_mcp_health.sh

# Start Claude Code (if environment already loaded)
claude-code
```

## Environment Configuration

### ✅ Configured API Keys
- **GEMINI_API_KEY**: Ready for Zen MCP Server
- **PERPLEXITY_API_KEY**: Ready for web search functionality
- **GITHUB_PERSONAL_ACCESS_TOKEN**: Ready for repository access
- **OPENAI_API_KEY**: Ready for embeddings and AI functions
- **ANTHROPIC_API_KEY**: Ready for Claude integration

### 📁 Key Files
- `.env` - Main environment configuration (contains all API keys)
- `.env.example` - Template for new setups
- `load_env.sh` - Environment variable loader script
- `check_mcp_health.sh` - MCP server health verification
- `.mcp.json` - MCP server configuration for Claude Code

## MCP Server Status (8/8 Ready)

| Server | Status | Functionality |
|--------|--------|---------------|
| ✅ Filesystem | Ready | File operations and directory access |
| ✅ SQLite | Ready | Database queries and data management |
| ✅ GitHub | Ready | Repository access and code management |
| ✅ Perplexity | Ready | Web search and research capabilities |
| ✅ Zen | Ready | AI conversations and multi-model access |
| ✅ Context7 | Ready | Documentation and library assistance |
| ✅ Sequential Thinking | Ready | Complex problem-solving workflows |
| ✅ Memory | Ready | Knowledge graph and persistent memory |

## Startup Options

### Option 1: Environment Pre-loaded (Recommended)
```bash
source ./load_env.sh && claude-code
```

### Option 2: Manual Environment Loading
```bash
# Load environment first
source ./load_env.sh

# Then start Claude Code in a new command
claude-code
```

### Option 3: Direct Start (Environment loads automatically via .env)
```bash
claude-code
```

## Verification Commands

```bash
# Check all MCP servers
./check_mcp_health.sh

# Verify environment variables are loaded
source ./load_env.sh

# Test individual MCP servers
cd mcp-management/servers/zen-mcp-server && source .zen_venv/bin/activate && python server.py --version
```

## Agent Integration

The project includes specialized AI agents with isolated contexts:

```bash
# Access individual agents
cd agents && python ask_researcher.py    # Research & intelligence
cd agents && python ask_mastermind.py    # Strategy & architecture  
cd agents && python ask_executor.py      # Implementation & testing

# SPARC trio collaboration
cd agents && python sparc_trio_demo.py
```

## Troubleshooting

### If MCP servers aren't working:
1. Run `./check_mcp_health.sh` to identify issues
2. Reload environment: `source ./load_env.sh`
3. Restart Claude Code session

### If environment variables aren't loading:
1. Verify `.env` file exists and contains the keys
2. Run `source ./load_env.sh` to manually load
3. Check for syntax errors in `.env` file

### If API keys are missing:
1. Check `.env` file for the required keys
2. Copy missing keys from `.env.example` template
3. Restart Claude Code after updating

## File Locations

- **Project Root**: `/home/scott/TradeKnowledge/`
- **Environment**: `/home/scott/TradeKnowledge/.env`
- **MCP Config**: `/home/scott/TradeKnowledge/.mcp.json`
- **Agents**: `/home/scott/TradeKnowledge/agents/`
- **Zen Server**: `/home/scott/TradeKnowledge/mcp-management/servers/zen-mcp-server/`
- **Perplexity Server**: `/home/scott/TradeKnowledge/perplexity-mcp/`

---

**Ready to go!** All MCP servers are configured and ready for use with Claude Code.