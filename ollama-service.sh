#!/bin/bash
# Ollama Service Startup Script for TradeKnowledge

# Wait for system to be ready
sleep 10

# Set environment variables
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_ORIGINS="*"

# Start Ollama service
exec snap run ollama serve