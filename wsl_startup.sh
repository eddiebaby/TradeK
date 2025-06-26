#!/bin/bash
# WSL Startup Script for TradeKnowledge Local LLM
# Run this script when WSL starts to initialize the LLM environment

echo "🔧 Initializing TradeKnowledge LLM environment..."

# Set CUDA environment
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Configure Ollama
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_ORIGINS="*"

# Start Ollama if not running
if ! pgrep -f "ollama serve" > /dev/null; then
    echo "🚀 Starting Ollama service..."
    snap run ollama serve > /dev/null 2>&1 &
    sleep 5
    echo "✅ Ollama service started"
fi

# Warm up the model (optional)
echo "🔥 Warming up Qwen2.5-Coder model..."
snap run ollama run qwen2.5-coder:7b "Hello" --timeout 5s > /dev/null 2>&1 &

echo "✅ TradeKnowledge LLM environment ready!"
echo "💡 Use 'qwen \"your question\"' for quick coding assistance"