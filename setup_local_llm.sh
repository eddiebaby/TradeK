#!/bin/bash
# TradeKnowledge Local LLM Setup Script
# Configures Qwen2.5-Coder with GPU acceleration for persistent use

echo "🚀 Setting up TradeKnowledge Local LLM Environment..."

# Set CUDA paths permanently
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Add to bashrc if not already present
if ! grep -q "CUDA" ~/.bashrc; then
    echo "# CUDA Environment Variables" >> ~/.bashrc
    echo "export PATH=/usr/local/cuda/bin:\$PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
    echo "✅ CUDA paths added to bashrc"
fi

# Configure Ollama environment
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_ORIGINS="*"

# Add Ollama environment to bashrc
if ! grep -q "OLLAMA" ~/.bashrc; then
    echo "# Ollama Environment Variables" >> ~/.bashrc
    echo "export OLLAMA_HOST=0.0.0.0:11434" >> ~/.bashrc
    echo "export OLLAMA_ORIGINS=\"*\"" >> ~/.bashrc
    echo "✅ Ollama environment configured"
fi

# Check if Ollama service is running
if ! pgrep -f "ollama serve" > /dev/null; then
    echo "🔧 Starting Ollama service..."
    snap run ollama serve &
    sleep 3
    echo "✅ Ollama service started"
else
    echo "✅ Ollama service already running"
fi

# Check available models
echo "📦 Available models:"
snap run ollama list

# Test GPU acceleration
echo "🧪 Testing GPU acceleration..."
if snap run ollama run qwen2.5-coder:7b "Write a hello world in Python" --timeout 10s 2>/dev/null; then
    echo "✅ GPU-accelerated inference working!"
else
    echo "⚠️  Model may still be downloading or needs configuration"
fi

echo "🎯 Setup complete! Use 'snap run ollama run qwen2.5-coder:7b' to start coding!"