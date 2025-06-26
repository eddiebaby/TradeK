#!/bin/bash
# Test script for TradeKnowledge Local LLM setup

echo "🧪 Testing TradeKnowledge Local LLM Setup..."
echo "================================================"

# Source configuration
source ~/TradeKnowledge/llm_config.env

# Test 1: Check CUDA
echo "🔧 CUDA Status:"
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release"
    echo "✅ CUDA available"
else
    echo "❌ CUDA not found in PATH"
fi

# Test 2: Check Ollama
echo -e "\n🦙 Ollama Status:"
if command -v snap &> /dev/null && snap list | grep -q ollama; then
    snap run ollama --version
    echo "✅ Ollama installed via snap"
else
    echo "❌ Ollama not found"
fi

# Test 3: Check if service is running
echo -e "\n🔄 Service Status:"
if pgrep -f "ollama serve" > /dev/null; then
    echo "✅ Ollama service is running"
else
    echo "⚠️  Ollama service not running - starting..."
    snap run ollama serve &
    sleep 3
fi

# Test 4: List models
echo -e "\n📦 Available Models:"
snap run ollama list

# Test 5: Quick inference test
echo -e "\n🤖 Quick Inference Test:"
echo "Testing with: 'Write hello world in Python'"
timeout 15s snap run ollama run qwen2.5-coder:7b "Write a simple hello world program in Python" 2>&1 | head -10

echo -e "\n✅ Test complete! Your local LLM is ready for TradeKnowledge development."