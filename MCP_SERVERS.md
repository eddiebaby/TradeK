# MCP Servers Configuration for TradeKnowledge

## Overview
This document tracks the MCP (Model Context Protocol) servers configured for the TradeKnowledge project. These servers provide additional capabilities for research, documentation, and development assistance.

## Configured MCP Servers

### Perplexity MCP Server
**Status**: ✅ Configured and Ready  
**Location**: `/home/scottschweizer/perplexity-mcp/`  
**Repository**: https://github.com/pashpashpash/perplexity-mcp  
**Configuration File**: `~/.claude/claude_desktop_config.json`

#### Available Tools:
- `search` - Search the web with Perplexity AI for real-time information
- `get_documentation` - Retrieve documentation for programming languages, frameworks, and tools
- `chat_perplexity` - Interactive chat with Perplexity AI for complex questions
- `find_apis` - Discover APIs and integration options
- `check_deprecated_code` - Verify if code patterns or libraries are deprecated

#### Usage in TradeKnowledge Development:
1. **Research Best Practices**: Use for finding current algorithmic trading best practices
2. **Documentation Lookup**: Get up-to-date documentation for Python libraries (pandas, numpy, scikit-learn, etc.)
3. **Code Validation**: Check if trading algorithms or patterns are still current
4. **API Discovery**: Find new financial data APIs or services
5. **Technology Updates**: Stay current with ML/AI developments in finance

#### Configuration:
```json
{
  "mcpServers": {
    "perplexity": {
      "command": "node",
      "args": ["/home/scottschweizer/perplexity-mcp/build/index.js"],
      "env": {
        "PERPLEXITY_API_KEY": "pplx-YOUR-API-KEY"
      },
      "disabled": false,
      "autoApprove": ["search", "get_documentation", "chat_perplexity"],
      "alwaysAllow": ["search", "get_documentation", "chat_perplexity"]
    }
  }
}
```

#### Security Notes:
- API key is stored in environment variables
- Auto-approval is limited to read-only operations
- No sensitive data is sent to external services

## Usage Examples

### For TradeKnowledge Development:
```
# Research current algorithmic trading practices
search("latest algorithmic trading strategies 2024 machine learning")

# Get documentation for financial libraries
get_documentation("pandas-ta technical analysis library")

# Check if libraries are current
check_deprecated_code("PyPDF2 vs pypdf alternatives")

# Find new APIs
find_apis("real-time stock market data free APIs")
```

### For Code Review and Optimization:
```
# Validate performance patterns
search("Python async SQLite connection pooling best practices")

# Check vector database comparisons
search("Qdrant vs ChromaDB vs Weaviate performance 2024")

# Research embedding models
search("nomic-embed-text vs sentence-transformers performance")
```

## Integration with TradeKnowledge

The Perplexity MCP server enhances our development workflow by providing:

1. **Real-time Research**: Stay updated on financial ML/AI developments
2. **Code Validation**: Verify that our approaches align with current best practices
3. **Documentation Access**: Quick access to library documentation during development
4. **Pattern Discovery**: Find new algorithmic trading patterns and strategies
5. **Technology Evaluation**: Research new tools before integration

## Maintenance

- **Updates**: Run `npm update` in `/home/scottschweizer/perplexity-mcp/` periodically
- **API Key Rotation**: Update key in config file when needed
- **Monitoring**: Check logs if MCP tools become unavailable

---

*Last Updated: 2025-06-14*  
*Project: TradeKnowledge - AI-Powered Trading Knowledge Assistant*