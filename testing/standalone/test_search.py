#!/usr/bin/env python3
"""
Test search functionality with sample data
"""

import asyncio
import sys
from datetime import datetime

# Add src to path
sys.path.append('src')

from src.core.sqlite_storage import SQLiteStorage
from src.core.models import Book, Chunk, FileType, ChunkType

async def create_test_data():
    """Create some test data for search testing"""
    
    print("🧪 Creating test data...")
    
    storage = SQLiteStorage()
    
    # Create a test book
    test_book = Book(
        id="test_algo_trading_book",
        title="Test Algorithmic Trading Guide",
        author="Test Author",
        file_path="data/books/test.pdf",
        file_type=FileType.PDF,
        file_hash="test123hash",
        total_pages=100,
        categories=["algorithmic-trading", "python", "testing"],
        metadata={"description": "Test book for search functionality"}
    )
    
    # Save book
    success = await storage.save_book(test_book)
    if not success:
        print("❌ Failed to save test book")
        return False
    
    # Create test chunks with trading-related content
    test_chunks = [
        Chunk(
            id="chunk_001",
            book_id=test_book.id,
            chunk_index=0,
            text="Moving averages are a fundamental tool in algorithmic trading. The simple moving average (SMA) calculates the average price over a specific period. Python implementation using pandas: df['SMA_20'] = df['close'].rolling(window=20).mean()",
            chunk_type=ChunkType.TEXT
        ),
        Chunk(
            id="chunk_002", 
            book_id=test_book.id,
            chunk_index=1,
            text="Risk management is crucial in trading algorithms. Position sizing using the Kelly Criterion helps optimize bet sizes. The formula is f = (bp - q) / b where f is the fraction to bet, b is the odds, p is probability of winning, q is probability of losing.",
            chunk_type=ChunkType.TEXT
        ),
        Chunk(
            id="chunk_003",
            book_id=test_book.id, 
            chunk_index=2,
            text="Backtesting trading strategies requires historical data. Here's a Python example:\n\nimport yfinance as yf\ndata = yf.download('AAPL', start='2020-01-01')\nreturns = data['Close'].pct_change()\nsharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)",
            chunk_type=ChunkType.CODE
        ),
        Chunk(
            id="chunk_004",
            book_id=test_book.id,
            chunk_index=3, 
            text="Machine learning in trading often uses features like technical indicators. Popular indicators include RSI (Relative Strength Index), MACD (Moving Average Convergence Divergence), and Bollinger Bands. These can be computed using libraries like TA-Lib or pandas-ta.",
            chunk_type=ChunkType.TEXT
        ),
        Chunk(
            id="chunk_005",
            book_id=test_book.id,
            chunk_index=4,
            text="Portfolio optimization using Modern Portfolio Theory involves finding the efficient frontier. The Sharpe ratio measures risk-adjusted returns. Python libraries like PyPortfolioOpt and cvxpy help implement these optimization algorithms.",
            chunk_type=ChunkType.TEXT
        )
    ]
    
    # Save chunks
    success = await storage.save_chunks(test_chunks)
    if not success:
        print("❌ Failed to save test chunks")
        return False
    
    # Update book chunk count
    test_book.total_chunks = len(test_chunks)
    test_book.indexed_at = datetime.now()
    await storage.update_book(test_book)
    
    print(f"✅ Created test book: {test_book.title}")
    print(f"   📦 Chunks: {len(test_chunks)}")
    print(f"   🏷️  Categories: {', '.join(test_book.categories)}")
    
    return True

async def test_search():
    """Test search functionality"""
    
    print("\n🔍 Testing search functionality...")
    
    # Import search after ensuring test data exists
    from src.search.text_search import TextSearchEngine
    
    search_engine = TextSearchEngine()
    
    # Test queries
    test_queries = [
        "moving average",
        "Python", 
        "risk management",
        "machine learning",
        "Sharpe ratio",
        "pandas"
    ]
    
    for query in test_queries:
        print(f"\n🔎 Query: '{query}'")
        
        results = search_engine.search_exact(query, num_results=3)
        
        if results['results']:
            print(f"   ✅ Found {len(results['results'])} results in {results['search_time_ms']}ms")
            for i, result in enumerate(results['results'][:2], 1):
                text_preview = result['text'][:100] + "..." if len(result['text']) > 100 else result['text']
                print(f"   {i}. Score: {result['score']:.3f} - {text_preview}")
        else:
            print(f"   ❌ No results found")
    
    # Get database stats
    stats = search_engine.get_database_stats()
    print(f"\n📊 Database Stats:")
    print(f"   📚 Total chunks: {stats['total_chunks']}")
    print(f"   📖 Unique books: {stats['unique_books']}")
    print(f"   📏 Average chunk size: {stats['avg_chunk_size']} characters")

async def main():
    print("🚀 TradeKnowledge Search Test")
    
    # Create test data
    success = await create_test_data()
    if not success:
        print("💥 Failed to create test data")
        sys.exit(1)
    
    # Test search
    await test_search()
    
    print("\n🎉 Search test completed!")
    print("\n💡 Try these commands:")
    print("   sqlite3 data/knowledge.db \"SELECT title, total_chunks FROM books;\"")
    print("   sqlite3 data/knowledge.db \"SELECT chunk_index, substr(text, 1, 50) FROM chunks LIMIT 5;\"")

if __name__ == "__main__":
    asyncio.run(main())