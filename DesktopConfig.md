# TradeKnowledge Desktop Configuration for NVIDIA 1080 GPU

## Overview
This configuration guide is specifically for running TradeKnowledge on your Windows desktop with NVIDIA GTX 1080 card for GPU-accelerated vectorization processing.

## 🖥️ **System Setup - Windows Desktop**

### Hardware Specifications
- **System**: Windows Desktop
- **GPU**: NVIDIA GTX 1080 (4GB VRAM) 
- **Purpose**: High-performance vectorization and embedding generation
- **Advantage**: GPU acceleration for much faster embedding processing

### Prerequisites Installation

#### 1. Python Environment
```powershell
# Install Python 3.11+ from python.org
# Recommended: Python 3.11.x for best compatibility

# Install pip packages
pip install --upgrade pip
pip install virtualenv
```

#### 2. NVIDIA GPU Setup
```powershell
# Ensure NVIDIA drivers are up to date
# Download from: https://www.nvidia.com/drivers/

# Install CUDA Toolkit 11.8 or 12.x
# Download from: https://developer.nvidia.com/cuda-toolkit

# Verify CUDA installation
nvcc --version
nvidia-smi
```

#### 3. Docker Desktop (Optional but Recommended)
```powershell
# Download Docker Desktop for Windows
# https://docs.docker.com/desktop/install/windows-wsl/

# Enable WSL2 backend for better performance
# This allows running Linux containers efficiently
```

## 🚀 **Installation Instructions**

### Step 1: Clone Project
```powershell
# Navigate to desired directory
cd "C:\Users\scott\OneDrive\Documents"

# Clone from GitHub
git clone https://github.com/eddiebaby/TradeK.git
cd TradeK
```

### Step 2: Virtual Environment Setup
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install additional GPU-optimized packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install sentence-transformers[gpu]
pip install cupy-cuda11x  # or cupy-cuda12x depending on CUDA version
```

### Step 3: GPU-Optimized Configuration

#### Environment Variables (.env file)
```bash
# GPU Configuration
CUDA_VISIBLE_DEVICES=0
OLLAMA_NUM_GPU=1
OLLAMA_CUDA_COMPUTE_CAP=6.1  # GTX 1080 compute capability

# Embedding Configuration for GPU
EMBEDDING_BATCH_SIZE=64      # Increased for GPU
EMBEDDING_DIMENSION=384
OLLAMA_MODEL=nomic-embed-text
OLLAMA_HOST=http://localhost:11434

# Database Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
REDIS_HOST=localhost
REDIS_PORT=6379

# Performance Settings
MAX_CONCURRENT_REQUESTS=8    # Increased for desktop
THREAD_POOL_SIZE=12         # More threads for desktop CPU
```

## 🔧 **GPU-Accelerated Ollama Setup**

### Install Ollama with GPU Support
```powershell
# Download Ollama for Windows with CUDA support
# From: https://ollama.ai/download/windows

# Install Ollama
# The installer will automatically detect CUDA

# Pull the embedding model
ollama pull nomic-embed-text

# Verify GPU usage
ollama run nomic-embed-text "test embedding"
```

### Ollama GPU Configuration
```powershell
# Set environment variables for Ollama
setx OLLAMA_HOST "0.0.0.0"
setx OLLAMA_NUM_GPU "1"
setx OLLAMA_MAX_LOADED_MODELS "2"
setx OLLAMA_GPU_MEMORY_FRACTION "0.8"  # Use 80% of GPU memory
```

## 📊 **Vector Database Setup**

### Option 1: Docker-based Qdrant (Recommended)
```powershell
# Start Qdrant in Docker
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant:latest

# Start Redis for caching
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

### Option 2: Native Windows Installation
```powershell
# Download Qdrant Windows binary
# From: https://github.com/qdrant/qdrant/releases

# Extract and run
qdrant.exe
```

## 🚀 **High-Performance Vectorization Scripts**

### GPU-Optimized Batch Processing
```python
# File: scripts/gpu_vectorization.py
"""
GPU-optimized vectorization for Windows desktop with NVIDIA 1080
"""

import asyncio
import os
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer

class GPUVectorizer:
    def __init__(self):
        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model on GPU
        self.model = SentenceTransformer('nomic-ai/nomic-embed-text-v1')
        self.model = self.model.to(self.device)
        
        # Optimized batch size for GTX 1080 (4GB VRAM)
        self.batch_size = 128  # Adjust based on available memory
    
    async def process_books(self, book_folder: str):
        """Process all books with GPU acceleration"""
        # Implementation for high-speed processing
        pass
```

### Recommended Batch Processing Settings
```python
# Optimized for GTX 1080 (4GB VRAM)
BATCH_SIZES = {
    "small_chunks": 256,    # < 200 characters
    "medium_chunks": 128,   # 200-500 characters  
    "large_chunks": 64,     # 500+ characters
}

# Memory management
MEMORY_SETTINGS = {
    "gpu_memory_fraction": 0.8,
    "cpu_workers": 8,
    "prefetch_factor": 4,
}
```

## 📁 **Project Structure on Windows**

```
C:\Users\scott\OneDrive\Documents\From WSL\TradeKnowledge\
├── venv/                          # Python virtual environment
├── src/                          # Source code
├── Knowledge/                    # Books to process (18+ books)
├── data/                         # Databases and cache
│   ├── knowledge.db             # SQLite database (5 books processed)
│   ├── qdrant/                  # Vector database storage
│   └── embeddings/              # Embedding cache
├── scripts/
│   ├── gpu_vectorization.py    # GPU-optimized processing
│   ├── robust_ingest.py         # Reliable book processing
│   └── init_api.py             # API initialization
├── docker-compose.yml           # Docker orchestration
├── requirements.txt             # Python dependencies
└── DesktopConfig.md            # This file
```

## 🎯 **Vectorization Workflow**

### Phase 1: Environment Verification
```powershell
# Activate environment
venv\Scripts\activate

# Verify GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check Ollama GPU usage
ollama ps
```

### Phase 2: Start Services
```powershell
# Start vector database
docker-compose up -d qdrant redis

# Start Ollama (should auto-start with Windows)
# Verify at: http://localhost:11434

# Verify services
curl http://localhost:6333/health  # Qdrant
curl http://localhost:11434/api/version  # Ollama
```

### Phase 3: Run Vectorization
```powershell
# Process all books in Knowledge folder
python scripts/gpu_vectorization.py --folder "Knowledge" --batch-size 128

# Alternative: Use existing robust script
python robust_ingest.py "Knowledge/book_name.pdf"
```

## 📈 **Performance Optimization**

### GPU Memory Management
```python
# Monitor GPU usage during processing
nvidia-smi -l 1  # Update every second

# Optimize batch sizes based on memory usage
# GTX 1080 (4GB): Start with batch_size=64, increase if stable
```

### Expected Performance Gains
- **CPU-only (WSL)**: ~2-5 embeddings/second
- **GPU-accelerated (GTX 1080)**: ~50-100 embeddings/second
- **Total time for 18 books**: ~2-4 hours (vs 20+ hours on CPU)

## 🔍 **Monitoring and Debugging**

### GPU Monitoring Commands
```powershell
# Real-time GPU usage
nvidia-smi -l 1

# Memory usage
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

# Temperature monitoring
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits
```

### Common Issues and Solutions

#### Issue: CUDA Out of Memory
```python
# Reduce batch size
EMBEDDING_BATCH_SIZE=32  # From 128

# Clear GPU cache
torch.cuda.empty_cache()
```

#### Issue: Ollama Not Using GPU
```powershell
# Reinstall Ollama with CUDA support
# Verify CUDA installation
# Check environment variables
```

#### Issue: Slow Performance
```powershell
# Check GPU utilization
nvidia-smi

# Verify model is on GPU
python -c "
import torch
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('nomic-ai/nomic-embed-text-v1')
print(f'Model device: {next(model.parameters()).device}')
"
```

## 🎯 **Success Metrics**

### Target Performance
- **Books processed**: 18+ remaining books
- **Processing time**: 2-4 hours total
- **GPU utilization**: >80% during processing
- **Memory usage**: <3.5GB of 4GB VRAM
- **Final database**: 24+ books fully vectorized

### Verification Steps
```python
# Check final database status
python -c "
import sqlite3
conn = sqlite3.connect('data/knowledge.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM books')
books = cursor.fetchone()[0]
cursor.execute('SELECT COUNT(*) FROM chunks')
chunks = cursor.fetchone()[0]
print(f'Books: {books}, Chunks: {chunks}')
"

# Test semantic search
python -c "
from src.core.qdrant_storage import QdrantStorage
from src.ingestion.local_embeddings import LocalEmbeddingGenerator

# Test search functionality
# Should return relevant results from all processed books
"
```

## 🚀 **Next Steps After Vectorization**

1. **Verify Complete Database**: All 24+ books processed and searchable
2. **Test API Performance**: Start FastAPI server and test search endpoints
3. **Benchmark Search Speed**: Compare semantic vs. text search performance
4. **Deploy to Production**: Use Docker Compose for production deployment
5. **Monitor Performance**: Set up Grafana dashboards for system monitoring

## 📞 **Support and Troubleshooting**

### Useful Commands
```powershell
# Project status
python scripts/init_api.py --health-check-only

# Database stats
sqlite3 data/knowledge.db "SELECT title, total_chunks FROM books;"

# Vector database stats
curl http://localhost:6333/collections/tradeknowledge
```

### Log Locations
- **Ollama logs**: `%USERPROFILE%\.ollama\logs\`
- **Application logs**: `logs/` directory
- **Docker logs**: `docker logs qdrant` / `docker logs redis`

---

**🎯 Goal**: Transform your GTX 1080 into a high-performance embedding generation machine, completing vectorization of all 24+ trading books in 2-4 hours instead of 20+ hours on CPU!

**Status**: Ready for GPU-accelerated vectorization on Windows desktop.