#!/bin/bash
# AMD GPU Setup Script for TradeKnowledge
# Run this after adding the ROCm GPG key

set -e

echo "🔧 Setting up AMD GPU acceleration for Ollama..."

# Add ROCm repository
echo "📦 Adding ROCm repository..."
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list

# Update package list
echo "🔄 Updating package lists..."
sudo apt update

# Install ROCm packages for WSL2
echo "📥 Installing ROCm packages..."
sudo apt install -y rocm-dev rocm-libs rccl

# Install OpenCL support
echo "🎯 Installing OpenCL support..."
sudo apt install -y rocm-opencl rocm-opencl-dev

# Add user to render group for GPU access
echo "👤 Adding user to render group..."
sudo usermod -a -G render $USER

# Set up environment variables
echo "🌍 Setting up environment variables..."
echo 'export PATH=$PATH:/opt/rocm/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib' >> ~/.bashrc
echo 'export ROCM_PATH=/opt/rocm' >> ~/.bashrc

# Create systemd override for WSL2 (if needed)
echo "⚙️  Configuring for WSL2..."
sudo mkdir -p /etc/systemd/system/ollama.service.d/
cat << EOF | sudo tee /etc/systemd/system/ollama.service.d/override.conf
[Service]
Environment="ROCM_PATH=/opt/rocm"
Environment="HIP_VISIBLE_DEVICES=0"
Environment="OLLAMA_GPU_LAYERS=35"
EOF

echo "✅ AMD GPU setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Logout and login again (or run: newgrp render)"
echo "2. Test with: rocm-smi"
echo "3. Test OpenCL: clinfo"
echo "4. Install Ollama with GPU support"
echo ""
echo "⚠️  Note: You may need to restart WSL2 for changes to take effect"
echo "   Run in PowerShell: wsl --shutdown && wsl"