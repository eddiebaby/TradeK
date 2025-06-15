#!/usr/bin/env python3
"""
Minimal test of optimized PDF processing - just first few pages
"""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.ingestion.pdf_parser import PDFParser
from src.ingestion.resource_monitor import ResourceMonitor, ResourceLimits

async def test_minimal():
    """Test just the PDF parsing with resource monitoring"""
    
    pdf_path = "data/books/Yves Hilpisch - Python for Algorithmic Trading_ From Idea to Cloud Deployment-O'Reilly Media (2020).pdf"
    
    if not Path(pdf_path).exists():
        print(f"❌ PDF not found: {pdf_path}")
        return False
    
    print(f"🧪 Testing optimized PDF parsing: {pdf_path}")
    
    # Configure resource limits
    limits = ResourceLimits(
        max_memory_percent=70.0,
        max_memory_mb=1200,
        warning_threshold=55.0,
        check_interval=2.0
    )
    
    # Initialize resource monitor
    monitor = ResourceMonitor(limits)
    await monitor.start_monitoring()
    
    # Add callback for detailed monitoring
    async def monitor_callback(check):
        usage = check['usage']
        print(f"💾 Memory: {usage['system_used_percent']:.1f}% system, {usage['process_memory_mb']:.1f}MB process")
        
        if check['memory_warning']:
            print(f"⚠️  Memory warning at {usage['system_used_percent']:.1f}%")
        
        if check['memory_critical']:
            print(f"🚨 Memory critical at {usage['system_used_percent']:.1f}%")
    
    monitor.add_callback(monitor_callback)
    
    try:
        # Initialize parser
        parser = PDFParser()
        
        print("📖 Starting PDF parsing...")
        start_time = time.time()
        
        # Parse the PDF
        result = parser.parse_file(Path(pdf_path))
        
        parsing_time = time.time() - start_time
        
        if result['errors']:
            print(f"❌ Parsing errors: {result['errors']}")
            return False
        
        pages = result['pages']
        metadata = result['metadata']
        
        print(f"✅ Parsing completed in {parsing_time:.1f} seconds")
        print(f"📄 Pages extracted: {len(pages)}")
        print(f"📚 Title: {metadata.get('title', 'Unknown')}")
        print(f"👤 Author: {metadata.get('author', 'Unknown')}")
        
        # Show sample of first page
        if pages:
            first_page = pages[0]
            sample_text = first_page['text'][:200] + "..." if len(first_page['text']) > 200 else first_page['text']
            print(f"\n📖 First page sample:\n{sample_text}")
        
        # Test chunking on just first 5 pages
        from src.ingestion.text_chunker import TextChunker, ChunkingConfig
        
        chunker = TextChunker(ChunkingConfig(
            chunk_size=500,
            chunk_overlap=60,
            min_chunk_size=50,
            max_chunk_size=1000
        ))
        
        print(f"\n📦 Testing chunking on first 5 pages...")
        test_pages = pages[:5]
        
        chunks = chunker.chunk_pages(test_pages, "test_book", {})
        
        print(f"✅ Created {len(chunks)} chunks from {len(test_pages)} pages")
        
        if chunks:
            print(f"📝 Sample chunk: {chunks[0].text[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Stop monitoring and show summary
        await monitor.stop_monitoring()
        
        try:
            summary = monitor.get_usage_summary()
            if 'error' not in summary:
                memory_stats = summary['memory_stats']
                print(f"\n📊 Resource Summary:")
                print(f"   Memory range: {memory_stats['min_percent']:.1f}% - {memory_stats['max_percent']:.1f}%")
                print(f"   Peak memory: {memory_stats['max_percent']:.1f}%")
                print(f"   Warnings: {summary['warnings_issued']}")
        except Exception as e:
            print(f"Could not generate summary: {e}")

async def main():
    print("🚀 Starting minimal PDF processing test")
    
    success = await test_minimal()
    
    if success:
        print("✅ Test completed successfully!")
    else:
        print("❌ Test failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())