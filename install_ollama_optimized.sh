#!/bin/bash
# Optimized Ollama Installation for AMD GPU + TradeKnowledge

set -e

echo "🚀 Installing Ollama with AMD GPU optimization..."

# Install Ollama
echo "📥 Downloading and installing Ollama..."
curl -fsSL https://ollama.ai/install.sh | sh

# Configure Ollama for AMD GPU
echo "⚙️  Configuring Ollama for AMD Radeon..."
sudo mkdir -p /etc/systemd/system/ollama.service.d/

# Create optimized configuration for your hardware
cat << EOF | sudo tee /etc/systemd/system/ollama.service.d/gpu-config.conf
[Service]
Environment="ROCM_PATH=/opt/rocm"
Environment="HIP_VISIBLE_DEVICES=0"
Environment="OLLAMA_GPU_LAYERS=35"
Environment="OLLAMA_NUM_PARALLEL=4"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_HOST=0.0.0.0:11434"
# Memory optimization for 6.7GB system
Environment="OLLAMA_MAX_VRAM=2GB"
Environment="OLLAMA_FLASH_ATTENTION=1"
EOF

# Start Ollama service
echo "🔄 Starting Ollama service..."
sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl start ollama

# Wait for service to start
echo "⏳ Waiting for Ollama to start..."
sleep 5

# Pull the embedding model optimized for your system
echo "📦 Pulling nomic-embed-text model..."
ollama pull nomic-embed-text

# Test the installation
echo "🧪 Testing Ollama installation..."
curl -s http://localhost:11434/api/version || echo "⚠️  Ollama not responding on expected port"

# Show status
echo "📊 Ollama service status:"
sudo systemctl status ollama --no-pager

echo ""
echo "✅ Ollama installation completed!"
echo ""
echo "🔍 Verify installation:"
echo "   • Service status: sudo systemctl status ollama"
echo "   • Test embedding: ollama run nomic-embed-text"
echo "   • Check GPU usage: rocm-smi"
echo ""
echo "📝 Configuration optimized for:"
echo "   • AMD Ryzen 7 7730U with Radeon Graphics"
echo "   • 6.7GB RAM with shared graphics memory"
echo "   • 4 parallel processing threads"
echo "   • 2GB VRAM allocation"