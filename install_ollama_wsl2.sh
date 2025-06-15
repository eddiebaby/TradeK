#!/bin/bash
# Ollama Installation optimized for WSL2 + AMD CPU
# Since ROCm doesn't work in WSL2, we optimize for CPU performance

set -e

echo "🚀 Installing Ollama optimized for WSL2 + AMD CPU..."

# Install Ollama
echo "📥 Downloading and installing Ollama..."
curl -fsSL https://ollama.ai/install.sh | sh

# Configure Ollama for optimal CPU performance on your Ryzen 7 7730U
echo "⚙️  Configuring Ollama for AMD Ryzen 7 7730U (CPU-only)..."
sudo mkdir -p /etc/systemd/system/ollama.service.d/

# Create CPU-optimized configuration
cat << EOF | sudo tee /etc/systemd/system/ollama.service.d/cpu-optimized.conf
[Service]
# CPU optimization for AMD Ryzen 7 7730U (8 cores, 16 threads)
Environment="OLLAMA_NUM_PARALLEL=8"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_HOST=0.0.0.0:11434"
# Memory optimization for 6.7GB system
Environment="OLLAMA_MAX_VRAM=0"
Environment="OLLAMA_FLASH_ATTENTION=1"
# CPU threading optimization
Environment="OMP_NUM_THREADS=8"
Environment="MALLOC_ARENA_MAX=2"
# Disable GPU attempts
Environment="CUDA_VISIBLE_DEVICES="
Environment="HIP_VISIBLE_DEVICES="
EOF

# Start Ollama service
echo "🔄 Starting Ollama service..."
sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl start ollama

# Wait for service to start
echo "⏳ Waiting for Ollama to start..."
sleep 10

# Test connectivity
echo "🧪 Testing Ollama connectivity..."
max_retries=5
for i in $(seq 1 $max_retries); do
    if curl -s http://localhost:11434/api/version > /dev/null; then
        echo "✅ Ollama is responding!"
        break
    else
        echo "⏳ Attempt $i/$max_retries - waiting for Ollama..."
        sleep 3
    fi
done

# Pull the embedding model
echo "📦 Pulling nomic-embed-text model (274MB)..."
ollama pull nomic-embed-text

# Test the model
echo "🧪 Testing embedding model..."
echo "test embedding" | ollama run nomic-embed-text > /dev/null 2>&1 && echo "✅ Model working!" || echo "⚠️ Model test failed"

echo ""
echo "✅ Ollama WSL2 installation completed!"
echo ""
echo "📊 Configuration optimized for:"
echo "   • AMD Ryzen 7 7730U (8 cores, CPU-only)"
echo "   • 6.7GB RAM constraint"
echo "   • WSL2 environment limitations"
echo "   • 8 parallel processing threads"
echo ""
echo "🔍 Verify installation:"
echo "   • Service: sudo systemctl status ollama"
echo "   • Version: curl http://localhost:11434/api/version"
echo "   • Models: ollama list"
echo ""
echo "⚡ Expected performance:"
echo "   • ~1-2 seconds per embedding (CPU-only)"
echo "   • ~15-30 minutes for full book vectorization"
EOF