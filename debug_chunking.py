#!/usr/bin/env python3
"""
Debug chunking process step by step
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.ingestion.pdf_parser import PDFParser
from src.ingestion.text_chunker import TextChunker, ChunkingConfig

def debug_chunking(pdf_path: str):
    """Debug chunking process step by step"""
    
    print(f"🔍 Debugging chunking for: {Path(pdf_path).name}")
    
    # Step 1: Parse PDF
    print("\n📖 Step 1: Parsing PDF...")
    pdf_parser = PDFParser(enable_ocr=False)
    
    try:
        parse_result = pdf_parser.parse_file(Path(pdf_path))
        print(f"✅ Found {len(parse_result['pages'])} pages")
        
        # Show page stats
        total_chars = 0
        for i, page in enumerate(parse_result['pages']):
            page_chars = len(page['text'])
            total_chars += page_chars
            print(f"  Page {i+1}: {page_chars:,} chars")
            
            # Show sample of first few pages
            if i < 3:
                sample = page['text'][:200].replace('\n', ' ') + "..." if len(page['text']) > 200 else page['text']
                print(f"    Sample: {sample}")
        
        print(f"📊 Total characters: {total_chars:,}")
        
    except Exception as e:
        print(f"❌ Parse failed: {e}")
        return False
    
    # Step 2: Test chunking
    print("\n✂️ Step 2: Testing chunking...")
    
    # Try different chunk sizes
    configs = [
        ChunkingConfig(chunk_size=800, chunk_overlap=100, min_chunk_size=50, max_chunk_size=1200),
        ChunkingConfig(chunk_size=1000, chunk_overlap=200, min_chunk_size=100, max_chunk_size=1500),
        ChunkingConfig(chunk_size=1200, chunk_overlap=200, min_chunk_size=200, max_chunk_size=2000),
    ]
    
    for i, config in enumerate(configs):
        print(f"\n🧪 Config {i+1}: size={config.chunk_size}, overlap={config.chunk_overlap}")
        
        try:
            chunker = TextChunker(config)
            chunks = chunker.chunk_pages(parse_result['pages'], f"test_book_{i}", {})
            
            print(f"  ✅ Created {len(chunks)} chunks")
            
            # Show chunk stats
            if chunks:
                chunk_sizes = [len(chunk.text) for chunk in chunks]
                avg_size = sum(chunk_sizes) / len(chunk_sizes)
                min_size = min(chunk_sizes)
                max_size = max(chunk_sizes)
                
                print(f"  📊 Chunk sizes: avg={avg_size:.0f}, min={min_size}, max={max_size}")
                
                # Show first chunk sample
                first_chunk_sample = chunks[0].text[:150].replace('\n', ' ') + "..." if len(chunks[0].text) > 150 else chunks[0].text
                print(f"  📄 First chunk: {first_chunk_sample}")
                
                # Show last chunk sample
                last_chunk_sample = chunks[-1].text[:150].replace('\n', ' ') + "..." if len(chunks[-1].text) > 150 else chunks[-1].text
                print(f"  📄 Last chunk: {last_chunk_sample}")
        
        except Exception as e:
            print(f"  ❌ Chunking failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 3: Test chunk saving simulation
    print("\n💾 Step 3: Simulating chunk saving...")
    
    # Use the best config
    chunker = TextChunker(ChunkingConfig(
        chunk_size=1000,
        chunk_overlap=200,
        min_chunk_size=100,
        max_chunk_size=1500
    ))
    
    chunks = chunker.chunk_pages(parse_result['pages'], "test_book", {})
    print(f"📦 Generated {len(chunks)} chunks for saving test")
    
    # Simulate saving process
    saved_count = 0
    for i, chunk in enumerate(chunks):
        try:
            # Basic validation
            if not chunk.text.strip():
                print(f"  ⚠️ Chunk {i} is empty")
                continue
            
            if len(chunk.text) < 10:
                print(f"  ⚠️ Chunk {i} too short: {len(chunk.text)} chars")
                continue
            
            saved_count += 1
            
            if i < 5 or i >= len(chunks) - 5:  # Show first and last 5
                print(f"  ✅ Chunk {i}: {len(chunk.text)} chars, index={chunk.chunk_index}")
        
        except Exception as e:
            print(f"  ❌ Chunk {i} validation failed: {e}")
    
    print(f"\n📊 Summary:")
    print(f"  Total pages: {len(parse_result['pages'])}")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Chunks generated: {len(chunks)}")
    print(f"  Chunks valid for saving: {saved_count}")
    print(f"  Expected tokens: ~{total_chars // 4:,}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_chunking.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    success = debug_chunking(pdf_path)
    
    if not success:
        sys.exit(1)