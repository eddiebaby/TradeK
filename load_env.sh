#!/bin/bash

# Environment Variable Loader for TradeKnowledge Project
# This script loads environment variables from .env file and exports them
# to make them available for MCP servers and other project components

echo "🔧 Loading TradeKnowledge environment variables..."

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found in current directory"
    echo "   Make sure you're in the TradeKnowledge project root"
    exit 1
fi

# Function to export variables from .env file
load_env() {
    set -a  # automatically export all variables
    source .env
    set +a  # turn off automatic export
}

# Load environment variables
load_env

# Verify critical API keys are loaded
echo "📋 Verifying environment variables:"

if [ -n "$GEMINI_API_KEY" ]; then
    echo "✅ GEMINI_API_KEY is loaded"
else
    echo "❌ GEMINI_API_KEY is not set"
fi

if [ -n "$PERPLEXITY_API_KEY" ]; then
    echo "✅ PERPLEXITY_API_KEY is loaded"
else
    echo "❌ PERPLEXITY_API_KEY is not set"
fi

if [ -n "$GITHUB_PERSONAL_ACCESS_TOKEN" ]; then
    echo "✅ GITHUB_PERSONAL_ACCESS_TOKEN is loaded"
else
    echo "❌ GITHUB_PERSONAL_ACCESS_TOKEN is not set"
fi

if [ -n "$OPENAI_API_KEY" ]; then
    echo "✅ OPENAI_API_KEY is loaded"
else
    echo "❌ OPENAI_API_KEY is not set"
fi

if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo "✅ ANTHROPIC_API_KEY is loaded"
else
    echo "❌ ANTHROPIC_API_KEY is not set"
fi

# Set default values for GitHub MCP if not set
if [ -z "$GITHUB_TOOLSETS" ]; then
    export GITHUB_TOOLSETS="repository,issues,pull_requests"
    echo "✅ Set GITHUB_TOOLSETS to default: $GITHUB_TOOLSETS"
fi

if [ -z "$GITHUB_READ_ONLY" ]; then
    export GITHUB_READ_ONLY="true"
    echo "✅ Set GITHUB_READ_ONLY to default: $GITHUB_READ_ONLY"
fi

if [ -z "$PERPLEXITY_MODEL" ]; then
    export PERPLEXITY_MODEL="sonar-deep-research"
    echo "✅ Set PERPLEXITY_MODEL to default: $PERPLEXITY_MODEL"
fi

echo ""
echo "🎉 Environment variables loaded successfully!"
echo ""
echo "📝 To use in current shell session:"
echo "   source ./load_env.sh"
echo ""
echo "📝 To run MCP health check:"
echo "   source ./load_env.sh && ./check_mcp_health.sh"
echo ""
echo "📝 To start Claude Code with environment loaded:"
echo "   source ./load_env.sh && claude-code"