#!/usr/bin/env python3
"""
Test LLMLingua compression functionality
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path  
sys.path.append('/home/scott/TradeKnowledge')

# Simple test without OpenAI dependencies
async def test_llmlingua_compression():
    """Test LLMLingua compression with Qwen2.5-Coder"""
    
    # Mock chunk class for testing
    class MockChunk:
        def __init__(self, content):
            self.content = content
            self.id = "test"
            self.start_index = 0
            self.end_index = len(content)
            self.page_number = 1
    
    # Test academic paper content with mathematical formulas
    test_content = """
    Furthermore, it is important to note that the LongLLMLingua approach represents a significant 
    advancement in the field of prompt compression for large language models. The methodology, 
    as described in the paper, involves a sophisticated two-stage compression process. In the 
    first stage, the system performs coarse-grained compression by removing entire sentences 
    that are deemed less important. Subsequently, in the second stage, the approach applies 
    fine-grained compression at the token level through an iterative process. The mathematical 
    formulation can be expressed as: $P(compression) = \\alpha \\cdot importance_{sentence} + 
    \\beta \\cdot relevance_{token}$ where $\\alpha$ and $\\beta$ are weighting parameters. 
    The experimental results demonstrate that this approach achieves compression ratios of up 
    to 20x while maintaining semantic integrity. As a result, this methodology has significant 
    implications for trading systems where computational efficiency is paramount. Moreover, 
    the application of prompt compression in financial contexts could lead to substantial 
    cost savings when processing large volumes of market data and research reports.
    """
    
    print("🗜️  LLMLingua Compression Test")
    print("="*50)
    
    try:
        # Import compression module
        from src.compression.llmlingua_compressor import LLMLinguaCompressor
        
        # Create compressor instance
        compressor = LLMLinguaCompressor(target_compression=0.6)
        
        # Create test chunk
        chunk = MockChunk(test_content.strip())
        
        print(f"📝 Original text: {len(chunk.content)} characters")
        print(f"🎯 Target compression: 60% of original")
        print("\\n🔄 Compressing with Qwen2.5-Coder...")
        
        # Test compression
        result = await compressor.compress_chunk(chunk, preserve_math=True)
        
        # Display results
        print("\\n📊 Compression Results:")
        print(f"  • Original length: {len(result.original_text)} chars")
        print(f"  • Compressed length: {len(result.compressed_text)} chars")
        print(f"  • Compression ratio: {result.compression_ratio:.2f}")
        print(f"  • Tokens saved: {result.tokens_saved}")
        print(f"  • Quality score: {result.quality_score:.2f}")
        
        # Show preserved mathematical content
        if "$" in result.compressed_text or "\\\\" in result.compressed_text:
            print("  • ✅ Mathematical formulas preserved")
        else:
            print("  • ⚠️  Mathematical content may be missing")
        
        print("\\n📄 Compressed Text Preview:")
        preview = result.compressed_text[:300] + "..." if len(result.compressed_text) > 300 else result.compressed_text
        print(preview)
        
        # Test potential OpenAI token savings
        original_tokens = len(result.original_text.split())
        compressed_tokens = len(result.compressed_text.split())
        token_savings = original_tokens - compressed_tokens
        
        print(f"\\n💰 Potential OpenAI Savings:")
        print(f"  • Original tokens: {original_tokens}")
        print(f"  • Compressed tokens: {compressed_tokens}")
        print(f"  • Token savings: {token_savings} ({(token_savings/original_tokens)*100:.1f}%)")
        
        # Estimate quota extension
        if token_savings > 0:
            quota_extension = (1 / result.compression_ratio) - 1
            print(f"  • Quota extension: ~{quota_extension*100:.0f}% more content processable")
        
        return True
        
    except Exception as e:
        print(f"❌ Compression test failed: {e}")
        return False


async def show_quota_info():
    """Show quota information without dependencies"""
    
    print("📊 OpenAI Quota Management")
    print("="*50)
    print("💡 1GB OpenAI Embedding Limit Strategy:")
    print("  • High-priority content → OpenAI embeddings")
    print("  • Mathematical formulas → OpenAI (precision)")
    print("  • Trading strategies → OpenAI (quality)")
    print("  • Bulk academic papers → Local embeddings")
    print("  • General content → Local embeddings")
    print("\\n🗜️  LLMLingua Compression Benefits:")
    print("  • 2-20x compression ratios possible")
    print("  • Preserves mathematical content")
    print("  • Extends 1GB quota significantly")
    print("  • Maintains semantic accuracy")


async def main():
    """Main test function"""
    
    print("🧪 Hybrid Vectorization System Test")
    print("="*60)
    
    # Test 1: Show quota strategy
    await show_quota_info()
    
    print("\\n")
    
    # Test 2: Test compression
    success = await test_llmlingua_compression()
    
    if success:
        print("\\n✅ Compression system working correctly!")
        print("\\n🚀 Next Steps:")
        print("  1. Install OpenAI package: pip install openai")
        print("  2. Set OPENAI_API_KEY environment variable")
        print("  3. Test full hybrid embedding system")
        print("  4. Process academic papers with quota optimization")
    else:
        print("\\n❌ Compression test failed - check Qwen2.5-Coder availability")


if __name__ == "__main__":
    asyncio.run(main())