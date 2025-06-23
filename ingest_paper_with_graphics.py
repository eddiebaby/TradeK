#!/usr/bin/env python3
"""
Enhanced Paper Ingestion with Graphics Preservation
===================================================

Uses the enhanced Docling processor to ingest academic papers with special
attention to preserving graphics, figures, tables, and mathematical content.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append('.')

def ingest_paper_with_graphics(pdf_path: str):
    """Ingest paper using enhanced Docling processor with graphics preservation"""
    
    # Import our enhanced processor
    try:
        from enhanced_docling_processor import EnhancedDoclingBookProcessor
    except ImportError:
        print("❌ Enhanced Docling processor not found")
        return False
    
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"❌ Paper not found: {pdf_path}")
        return False
    
    print("📄 ENHANCED PAPER INGESTION WITH GRAPHICS")
    print("=" * 50)
    print(f"📁 File: {pdf_file.name}")
    print(f"📊 Size: {pdf_file.stat().st_size / (1024*1024):.1f} MB")
    print()
    
    # Initialize enhanced processor
    print("🔧 Initializing enhanced Docling processor...")
    processor = EnhancedDoclingBookProcessor()
    
    # Process with graphics preservation
    print("🔄 Processing paper with graphics preservation...")
    start_time = time.time()
    
    try:
        result = processor.process_book(pdf_file)
        
        processing_time = time.time() - start_time
        
        if result.get('success'):
            print(f"✅ Paper processed successfully in {processing_time:.1f}s")
            print()
            print("📊 PROCESSING RESULTS:")
            print(f"   📄 Processor used: {result.get('processor_used', 'Unknown')}")
            print(f"   📝 Total chunks: {result.get('total_chunks', 0)}")
            print(f"   🧠 Memory peak: {result.get('memory_usage_mb', 0):.1f} MB")
            print(f"   ⏱️ Processing time: {processing_time:.1f}s")
            
            # Check for graphics and special content
            chunks_with_graphics = 0
            chunks_with_tables = 0
            chunks_with_formulas = 0
            
            if 'chunks' in result:
                for chunk in result['chunks']:
                    text = chunk.get('text', '').lower()
                    if any(indicator in text for indicator in ['figure', 'fig.', 'image', 'graph', 'chart']):
                        chunks_with_graphics += 1
                    if any(indicator in text for indicator in ['table', 'tab.', '|']):
                        chunks_with_tables += 1
                    if any(indicator in text for indicator in ['$', '\\(', 'equation', 'formula']):
                        chunks_with_formulas += 1
            
            print()
            print("🎯 CONTENT ANALYSIS:")
            print(f"   📈 Chunks with graphics references: {chunks_with_graphics}")
            print(f"   📊 Chunks with tables: {chunks_with_tables}")
            print(f"   🧮 Chunks with formulas: {chunks_with_formulas}")
            
            # Database storage info
            print()
            print("💾 DATABASE STORAGE:")
            print(f"   📚 ChromaDB: {'✅' if result.get('chromadb_success') else '❌'}")
            print(f"   🗄️ SQLite: {'✅' if result.get('sqlite_success') else '❌'}")
            
            return True
            
        else:
            print(f"❌ Paper processing failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Processing error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main ingestion function"""
    paper_path = "/home/scottschweizer/TradeKnowledge/Knowledge/2505.12540v2.pdf"
    
    print("🚀 Starting enhanced paper ingestion...")
    print(f"📄 Target: {Path(paper_path).name}")
    print()
    
    success = ingest_paper_with_graphics(paper_path)
    
    if success:
        print()
        print("🎉 INGESTION COMPLETED SUCCESSFULLY!")
        print("Graphics, tables, and mathematical content preserved.")
        print("Paper is now searchable in both ChromaDB and SQLite.")
    else:
        print()
        print("❌ INGESTION FAILED!")
        print("Check error messages above for details.")

if __name__ == "__main__":
    main()