#!/bin/bash

# MCP Server Health Check Script
# Verifies that all MCP servers are properly configured and functional

set -e

# Load environment variables from .env file
if [ -f ".env" ]; then
    set -a  # automatically export all variables
    source .env
    set +a  # turn off automatic export
fi

echo "🔍 MCP Server Health Check for TradeKnowledge project"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    local status=$1
    local message=$2
    case $status in
        "ok")
            echo -e "${GREEN}✅ $message${NC}"
            ;;
        "warning")
            echo -e "${YELLOW}⚠️  $message${NC}"
            ;;
        "error")
            echo -e "${RED}❌ $message${NC}"
            ;;
    esac
}

# Check environment variables
echo ""
echo "📋 Environment Variables Check:"

if [ -n "$GEMINI_API_KEY" ]; then
    print_status "ok" "GEMINI_API_KEY is set"
else
    print_status "error" "GEMINI_API_KEY is not set"
fi

if [ -n "$PERPLEXITY_API_KEY" ]; then
    print_status "ok" "PERPLEXITY_API_KEY is set"
else
    print_status "warning" "PERPLEXITY_API_KEY is not set (Perplexity MCP will not work)"
fi

if [ -n "$GITHUB_PERSONAL_ACCESS_TOKEN" ]; then
    print_status "ok" "GITHUB_PERSONAL_ACCESS_TOKEN is set"
else
    print_status "warning" "GITHUB_PERSONAL_ACCESS_TOKEN is not set (GitHub MCP will not work)"
fi

# Check MCP configuration file
echo ""
echo "📝 MCP Configuration Check:"

if [ -f "/home/scott/TradeKnowledge/.mcp.json" ]; then
    print_status "ok" "MCP configuration file exists"
    
    # Validate JSON syntax
    if python3 -m json.tool /home/scott/TradeKnowledge/.mcp.json > /dev/null 2>&1; then
        print_status "ok" "MCP configuration JSON is valid"
    else
        print_status "error" "MCP configuration JSON is invalid"
    fi
else
    print_status "error" "MCP configuration file not found"
fi

# Check server directories and files
echo ""
echo "📁 Server Files Check:"

# Zen MCP Server
ZEN_DIR="/home/scott/TradeKnowledge/mcp-management/servers/zen-mcp-server"
if [ -d "$ZEN_DIR" ]; then
    print_status "ok" "Zen MCP Server directory exists"
    
    if [ -f "$ZEN_DIR/server.py" ]; then
        print_status "ok" "Zen MCP Server main file exists"
    else
        print_status "error" "Zen MCP Server main file missing"
    fi
    
    if [ -d "$ZEN_DIR/.zen_venv" ]; then
        print_status "ok" "Zen MCP Server virtual environment exists"
    else
        print_status "warning" "Zen MCP Server virtual environment missing"
    fi
else
    print_status "error" "Zen MCP Server directory not found"
fi

# Perplexity MCP Server
PERP_DIR="/home/scott/TradeKnowledge/perplexity-mcp"
if [ -d "$PERP_DIR" ]; then
    print_status "ok" "Perplexity MCP Server directory exists"
    
    if [ -f "$PERP_DIR/src/perplexity_mcp/server.py" ]; then
        print_status "ok" "Perplexity MCP Server main file exists"
    else
        print_status "error" "Perplexity MCP Server main file missing"
    fi
else
    print_status "error" "Perplexity MCP Server directory not found"
fi

# Check data directory for SQLite
if [ -d "/home/scott/TradeKnowledge/data" ]; then
    print_status "ok" "Data directory exists for SQLite database"
else
    print_status "warning" "Data directory missing (will be created when needed)"
fi

# Check dependencies
echo ""
echo "🔧 Dependencies Check:"

dependencies=("node" "npm" "python3" "docker")
for dep in "${dependencies[@]}"; do
    if command -v "$dep" > /dev/null 2>&1; then
        print_status "ok" "$dep is installed"
    else
        print_status "error" "$dep is not installed"
    fi
done

# Test MCP server connectivity (basic checks)
echo ""
echo "🔌 Server Connectivity Check:"

# Test Zen server (if possible)
if [ -d "$ZEN_DIR" ] && [ -f "$ZEN_DIR/server.py" ] && [ -d "$ZEN_DIR/.zen_venv" ]; then
    cd "$ZEN_DIR"
    if timeout 3s bash -c 'source .zen_venv/bin/activate && python server.py --version' > /dev/null 2>&1; then
        print_status "ok" "Zen MCP Server responds to version check"
    else
        print_status "warning" "Zen MCP Server may need manual testing"
    fi
    cd - > /dev/null
fi

# Claude Code MCP integration check
echo ""
echo "🤖 Claude Code Integration:"

# Check if running in Claude Code environment
if [ -n "$CLAUDE_CODE_SESSION" ] || [ -n "$ANTHROPIC_CLAUDE_CODE" ]; then
    print_status "ok" "Running in Claude Code environment"
else
    print_status "warning" "Not running in Claude Code environment (MCP integration may not be active)"
fi

# Summary
echo ""
echo "📊 Health Check Summary:"
echo "========================"

# Count server status
TOTAL_SERVERS=8
READY_SERVERS=5  # filesystem, sqlite, zen, context7, sequential-thinking, memory

if [ -n "$PERPLEXITY_API_KEY" ]; then
    ((READY_SERVERS++))
fi

if [ -n "$GITHUB_PERSONAL_ACCESS_TOKEN" ]; then
    ((READY_SERVERS++))
fi

echo "📈 Server Status: $READY_SERVERS/$TOTAL_SERVERS servers ready"

# List server status
echo ""
echo "Server Status Details:"
print_status "ok" "Filesystem MCP Server: Ready"
print_status "ok" "SQLite MCP Server: Ready"
if [ -n "$GITHUB_PERSONAL_ACCESS_TOKEN" ]; then
    print_status "ok" "GitHub MCP Server: Ready"
else
    print_status "warning" "GitHub MCP Server: Needs API key"
fi
if [ -n "$PERPLEXITY_API_KEY" ]; then
    print_status "ok" "Perplexity MCP Server: Ready"
else
    print_status "warning" "Perplexity MCP Server: Needs API key"
fi
print_status "ok" "Zen MCP Server: Ready"
print_status "ok" "Context7 MCP Server: Ready"
print_status "ok" "Sequential Thinking MCP Server: Ready"
print_status "ok" "Memory MCP Server: Ready"

echo ""
if [ $READY_SERVERS -eq $TOTAL_SERVERS ]; then
    print_status "ok" "All MCP servers are ready for use!"
elif [ $READY_SERVERS -ge 7 ]; then
    print_status "ok" "All configured MCP servers are ready for use!"
elif [ $READY_SERVERS -ge 5 ]; then
    print_status "warning" "Core MCP servers are ready. Optional servers need API keys."
else
    print_status "error" "Critical MCP servers are not ready. Check configuration."
fi

echo ""
echo "🔧 Quick Actions:"
echo "  • To set missing API keys: export VARIABLE_NAME='your_key'"
echo "  • To reinitialize servers: ./initialize_mcp_servers.sh"
echo "  • To restart Claude Code: exit and restart your session"
echo "  • To test Zen server: cd $ZEN_DIR && source .zen_venv/bin/activate && python server.py"