#!/usr/bin/env python3
"""
Process a single book through the Trading Knowledge system
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from book_processing.orchestrator import book_orchestrator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    book_path = "/home/scottschweizer/TradeKnowledge/Knowledge/Yves Hilpisch - Python for Algorithmic Trading_ From Idea to Cloud Deployment-O'Reilly Media (2020).pdf"
    
    print(f"🚀 Processing book: {Path(book_path).name}")
    print("📖 This may take a few minutes...")
    
    try:
        # Process the book
        result = await book_orchestrator.process_book(
            file_path=book_path,
            title="Python for Algorithmic Trading",
            author="Yves Hilpisch",
            auto_vectorize=True
        )
        
        print("\n" + "="*50)
        if result.success:
            print("✅ SUCCESS! Book processed successfully")
            print(f"📚 Title: {result.title}")
            print(f"📖 Chapters: {result.chapters_processed}")
            print(f"🧠 Concepts: {result.concepts_extracted}")
            print(f"🔗 Embeddings: {result.embeddings_created}")
            print(f"⏱️  Time: {result.processing_time:.2f} seconds")
        else:
            print("❌ FAILED to process book")
            print(f"Error: {result.error_message}")
        
        # Show processing stats
        print("\n📊 System Stats:")
        stats_report = book_orchestrator.generate_processing_report()
        print(stats_report)
        
    except Exception as e:
        logger.error(f"Error processing book: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())