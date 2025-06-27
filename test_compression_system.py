#!/usr/bin/env python3
"""
Test the compression and hybrid embedding system
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path  
sys.path.append('/home/scott/TradeKnowledge')

from src.compression.llmlingua_compressor import LLMLinguaCompressor, test_compression
from src.utils.quota_manager import show_quota_status


async def main():
    """Test compression and quota management"""
    
    print("🧪 Testing Hybrid Vectorization System")
    print("="*60)
    
    # Test 1: Check quota status
    print("\\n1️⃣ OpenAI Quota Status:")
    await show_quota_status()
    
    # Test 2: Test compression
    print("\\n2️⃣ LLMLingua Compression Test:")
    await test_compression()
    
    print("\\n✅ System tests completed!")


if __name__ == "__main__":
    asyncio.run(main())