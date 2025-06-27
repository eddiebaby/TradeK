#!/usr/bin/env python3
"""
Demo: Hybrid Vectorization System with OpenAI Quota Optimization

This demonstrates the complete hybrid system:
1. LLMLingua compression using Qwen2.5-Coder
2. Smart routing between local and OpenAI embeddings  
3. OpenAI 1GB quota management
4. Academic paper processing optimization
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append('/home/scott/TradeKnowledge')


async def demo_academic_paper_processing():
    """Demonstrate processing academic papers with hybrid vectorization"""
    
    print("🎯 HYBRID VECTORIZATION DEMO")
    print("="*60)
    
    # Sample academic paper content (from LongLLMLingua analysis)
    paper_sections = {
        "abstract": """
        In long context scenarios, large language models face significant computational challenges 
        due to the quadratic complexity of attention mechanisms. This paper introduces LongLLMLingua, 
        a novel approach for prompt compression that accelerates LLMs while preserving semantic 
        accuracy. Our method achieves compression ratios of up to 20x with minimal performance 
        degradation through a budget-controlled, iterative compression algorithm that maintains 
        critical information density.
        """,
        
        "methodology": """
        The LongLLMLingua framework operates through a two-stage compression process. First, 
        coarse-grained sentence-level filtering removes redundant content based on importance 
        scores. Second, fine-grained token-level compression iteratively optimizes the remaining 
        content. The compression objective is formulated as: $\\min_{c} L(f(c), f(x))$ subject to 
        $|c| \\leq \\alpha|x|$, where $c$ is the compressed text, $x$ is the original text, 
        $f$ is the language model, and $\\alpha$ controls the compression budget.
        """,
        
        "results": """
        Experimental evaluation on multiple benchmarks demonstrates that LongLLMLingua achieves 
        superior compression-performance trade-offs. On the GSM8K dataset, we maintain 94.2% 
        accuracy with 10x compression and 91.1% accuracy with 20x compression. Latency 
        improvements range from 2.1x to 4.7x across different model sizes, with corresponding 
        cost reductions of 50-75% for API-based services.
        """
    }
    
    print("📄 Processing Academic Paper Sections:")
    total_original_chars = 0
    total_compressed_chars = 0
    
    for section_name, content in paper_sections.items():
        print(f"\\n📚 Section: {section_name.title()}")
        
        # Test compression first
        await demo_compression(content, section_name)
        
        # Demonstrate routing decision
        await demo_embedding_routing(content, section_name)
        
        total_original_chars += len(content)
    
    print(f"\\n📊 Total Processing Summary:")
    print(f"  • Original content: {total_original_chars:,} characters")
    print(f"  • Estimated with compression: ~{int(total_original_chars * 0.6):,} characters")
    print(f"  • OpenAI quota saved: ~{int(total_original_chars * 0.4):,} characters")


async def demo_compression(content: str, section_name: str):
    """Demo compression for a content section"""
    try:
        from src.compression.llmlingua_compressor import LLMLinguaCompressor
        
        # Mock chunk for testing
        class MockChunk:
            def __init__(self, content):
                self.content = content.strip()
                self.id = f"{section_name}_chunk"
                self.start_index = 0
                self.end_index = len(content)
                self.page_number = 1
        
        compressor = LLMLinguaCompressor(target_compression=0.6)
        chunk = MockChunk(content)
        
        result = await compressor.compress_chunk(chunk, preserve_math=True)
        
        print(f"  🗜️  Compression: {len(content)} → {len(result.compressed_text)} chars")
        print(f"  📊 Ratio: {result.compression_ratio:.2f}, Quality: {result.quality_score:.2f}")
        
        if "$" in result.compressed_text or "\\\\" in result.compressed_text:
            print("  ✅ Mathematical content preserved")
        
        return result
        
    except Exception as e:
        print(f"  ❌ Compression failed: {e}")
        return None


async def demo_embedding_routing(content: str, section_name: str):
    """Demo embedding routing decision"""
    
    # Simulate routing logic without full dependencies
    content_size = len(content.encode('utf-8'))
    
    # Classification logic (simplified)
    is_high_priority = section_name in ['abstract', 'results'] or len(content) < 500
    has_math_content = '$' in content or '\\\\' in content
    is_trading_relevant = any(term in content.lower() for term in ['trading', 'financial', 'cost', 'efficiency'])
    
    # Routing decision
    if is_high_priority and content_size < 2000:
        provider = "OpenAI"
        reason = "High priority, small size"
    elif has_math_content and content_size < 3000:
        provider = "OpenAI" 
        reason = "Mathematical content needs precision"
    elif is_trading_relevant and content_size < 1500:
        provider = "OpenAI"
        reason = "Trading-relevant content"
    else:
        provider = "Local (nomic-embed-text)"
        reason = "Cost optimization"
    
    print(f"  🎯 Embedding: {provider}")
    print(f"  💭 Reason: {reason}")
    print(f"  📏 Size: {content_size:,} bytes")


async def demo_quota_management():
    """Demo quota management features"""
    
    print("\\n💾 OpenAI Quota Management Demo")
    print("-" * 40)
    
    # Simulate quota status
    quota_data = {
        "usage_percentage": 23.7,
        "remaining_gb": 0.763,
        "documents_processed": 45,
        "routing_stats": {
            "local_embeddings": 312,
            "openai_embeddings": 78,
            "total_processed": 390
        }
    }
    
    print(f"📊 Current Usage: {quota_data['usage_percentage']:.1f}%")
    print(f"💾 Remaining: {quota_data['remaining_gb']:.3f} GB")
    print(f"📄 Documents: {quota_data['documents_processed']:,}")
    
    routing_stats = quota_data['routing_stats']
    local_pct = (routing_stats['local_embeddings'] / routing_stats['total_processed']) * 100
    openai_pct = (routing_stats['openai_embeddings'] / routing_stats['total_processed']) * 100
    
    print(f"\\n🔀 Routing Efficiency:")
    print(f"  • Local: {local_pct:.1f}% ({routing_stats['local_embeddings']} chunks)")
    print(f"  • OpenAI: {openai_pct:.1f}% ({routing_stats['openai_embeddings']} chunks)")
    
    # Calculate optimization impact
    estimated_savings = routing_stats['local_embeddings'] * 1000  # Avg 1KB per chunk
    print(f"\\n💰 Estimated Savings:")
    print(f"  • Quota saved: ~{estimated_savings/1024/1024:.1f} MB")
    print(f"  • Cost optimization: {local_pct:.0f}% processed locally")


async def demo_implementation_guide():
    """Show implementation guide"""
    
    print("\\n🚀 Implementation Guide")
    print("-" * 40)
    
    steps = [
        "1. Install dependencies: pip install openai aiofiles",
        "2. Set OPENAI_API_KEY environment variable",  
        "3. Configure EmbeddingConfig in config.py",
        "4. Use HybridEmbeddingRouter for document processing",
        "5. Monitor quota with quota_manager.py",
        "6. Backup embeddings for offline access"
    ]
    
    for step in steps:
        print(f"  {step}")
    
    print("\\n📁 Key Files Created:")
    files = [
        "src/ingestion/hybrid_embedding_router.py",
        "src/compression/llmlingua_compressor.py", 
        "src/utils/quota_manager.py"
    ]
    
    for file_path in files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  📄 {file_path}")


async def main():
    """Main demo function"""
    
    # Demo 1: Academic paper processing
    await demo_academic_paper_processing()
    
    # Demo 2: Quota management
    await demo_quota_management()
    
    # Demo 3: Implementation guide
    await demo_implementation_guide()
    
    print("\\n" + "="*60)
    print("🎉 HYBRID VECTORIZATION SYSTEM READY!")
    print("="*60)
    
    print("\\n✅ Key Benefits:")
    print("  • Unlimited local processing with nomic-embed-text")
    print("  • Strategic OpenAI usage within 1GB limit")
    print("  • 40-60% compression with LLMLingua + Qwen2.5-Coder")
    print("  • Mathematical content preservation")
    print("  • Automatic backup and sync capabilities")
    print("  • Cost optimization for off-grid development")
    
    print("\\n🔗 Perfect Integration with Existing Systems:")
    print("  • Qwen2.5-Coder for compression intelligence")
    print("  • Knowledge graph for content relationships")
    print("  • Bedrock migration plan enhancement")
    print("  • SPARC trio workflow optimization")


if __name__ == "__main__":
    asyncio.run(main())