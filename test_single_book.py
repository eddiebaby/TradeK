#!/usr/bin/env python3
"""
Test single book processing to debug the pipeline
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append('/home/scott/TradeKnowledge')

from comprehensive_book_processor import ComprehensiveBookProcessor

async def test_single_book():
    """Test processing a single book"""
    processor = ComprehensiveBookProcessor()
    books_directory = Path("/home/scott/TradeKnowledge/books and papers (pdf and epub)")
    
    # Test with smallest book first
    test_book = "dp439.pdf"
    book_path = books_directory / test_book
    
    if not book_path.exists():
        print(f"Test book not found: {book_path}")
        return
    
    print(f"Testing single book processing: {test_book}")
    print("=" * 60)
    
    try:
        analysis = await processor.process_single_book(book_path, "test")
        print(f"\n✅ Success! Extracted:")
        print(f"   📊 {len(analysis.trading_strategies)} strategies")
        print(f"   🧠 {len(analysis.conceptual_frameworks)} frameworks")
        print(f"   💡 {len(analysis.key_insights)} insights")
        print(f"   📈 {len(analysis.data_requirements)} data requirements")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_single_book())