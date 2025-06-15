#!/usr/bin/env python3
"""
Ingest EPUB using existing enhanced book processor
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.ingestion.enhanced_book_processor import EnhancedBookProcessor

async def ingest_epub(epub_path: str):
    """Ingest EPUB using enhanced book processor"""
    
    print(f"📚 EPUB Ingestion: {Path(epub_path).name}")
    print(f"📊 File size: {Path(epub_path).stat().st_size / (1024*1024):.2f}MB")
    
    if not Path(epub_path).exists():
        print(f"❌ File not found: {epub_path}")
        return False
    
    # Use enhanced book processor
    processor = EnhancedBookProcessor()
    
    try:
        print("🚀 Starting enhanced processing...")
        result = await processor.process_book(epub_path)
        
        if result['success']:
            print("🎉 EPUB ingestion successful!")
            
            # Show results
            book_info = result.get('book_info', {})
            processing_stats = result.get('processing_stats', {})
            
            print(f"📖 Title: {book_info.get('title', 'Unknown')}")
            print(f"👤 Author: {book_info.get('author', 'Unknown')}")
            print(f"📄 Chapters: {processing_stats.get('total_pages', 0)}")
            print(f"📦 Chunks: {processing_stats.get('total_chunks', 0)}")
            print(f"🧠 Embeddings: {processing_stats.get('total_embeddings', 0)}")
            print(f"⏱️ Processing time: {processing_stats.get('total_time', 0):.2f}s")
            
            return True
        else:
            print("❌ EPUB ingestion failed!")
            print(f"Error: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Enhanced processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    if len(sys.argv) < 2:
        print("Usage: python ingest_epub.py <epub_path>")
        sys.exit(1)
    
    epub_path = sys.argv[1]
    success = await ingest_epub(epub_path)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())