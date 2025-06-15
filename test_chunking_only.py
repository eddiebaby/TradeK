#!/usr/bin/env python3
"""
Test chunking in isolation
"""

import sys

# Add src to path
sys.path.append('src')

from src.ingestion.text_chunker import TextChunker, ChunkingConfig

def test_chunking():
    """Test chunking with sample text"""
    
    # Create sample pages similar to DiGA content
    sample_pages = [
        {
            'page_number': 1,
            'text': """Controllable Financial Market Generation with Diffusion Guided Meta Agent

Yu-Hao Huang1, Chang Xu2, Yang Liu2, Weiqing Liu2, Wu-Jun Li1, Jiang Bian2

1National Key Laboratory for Novel Software Technology, Department of Computer Science and Technology, Nanjing University
2Microsoft Research Asia

Abstract

Financial market simulation is crucial for validating trading strategies and understanding market dynamics. However, existing methods often struggle to generate realistic and controllable market scenarios. We propose DiGA (Diffusion Guided Meta Agent), a novel approach that combines diffusion models with meta-learning to generate controllable financial market data. Our method enables fine-grained control over market conditions while maintaining realistic statistical properties. The mathematical formulation min Ex∼D(pG(x) ∥ q(x)) captures the essence of our optimization objective."""
        },
        {
            'page_number': 2,
            'text': """2. Related Work

2.1 Market Simulation

Traditional market simulation approaches have relied on agent-based models and statistical methods. The challenge lies in capturing the complex dynamics of real markets while maintaining computational efficiency. Recent advances in deep learning have opened new possibilities for market generation.

2.2 Diffusion Models

Diffusion models have shown remarkable success in various generative tasks. The forward process q(xt|xt-1) = N(xt; √(1-βt)xt-1, βtI) and reverse process pθ(xt-1|xt) form the core of the diffusion framework. Our work extends this to financial time series generation."""
        },
        {
            'page_number': 3,
            'text': """3. Methodology

3.1 Problem Formulation

Let X = {x1, x2, ..., xT} represent a financial time series where each xt ∈ ℝd corresponds to market features at time t. Our goal is to learn a generative model G that can produce realistic market scenarios while allowing control over specific market conditions.

The optimization objective can be written as:
min_θ Ex∼pdata[L(x, Gθ(z, c))]

where z is random noise, c represents control parameters, and L is the loss function measuring the quality of generated samples."""
        }
    ]
    
    print("🧪 Testing chunking with sample data...")
    print(f"📄 Sample pages: {len(sample_pages)}")
    
    total_chars = sum(len(page['text']) for page in sample_pages)
    print(f"📊 Total characters: {total_chars}")
    
    # Test chunking
    chunker = TextChunker(ChunkingConfig(
        chunk_size=1000,
        chunk_overlap=200,
        min_chunk_size=100,
        max_chunk_size=1500
    ))
    
    print("\n✂️ Creating chunks...")
    try:
        chunks = chunker.chunk_pages(sample_pages, "test_book", {})
        
        print(f"✅ Created {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i}: {len(chunk.text)} chars, page {chunk.page_start}-{chunk.page_end}")
            sample = chunk.text[:100].replace('\n', ' ') + "..." if len(chunk.text) > 100 else chunk.text
            print(f"    Text: {sample}")
        
        return True
        
    except Exception as e:
        print(f"❌ Chunking failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_chunking()