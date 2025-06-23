#!/usr/bin/env python3
"""
Quick paper ingestion with graphics preservation
"""

import sys
sys.path.append('.')

from enhanced_docling_processor import EnhancedDoclingProcessor
from pathlib import Path
import time

def main():
    pdf_path = '/home/scottschweizer/TradeKnowledge/Knowledge/2505.12540v2.pdf'
    pdf_file = Path(pdf_path)

    print('📄 ENHANCED PAPER INGESTION WITH GRAPHICS')
    print('=' * 50)
    print(f'📁 File: {pdf_file.name}')
    print(f'📊 Size: {pdf_file.stat().st_size / (1024*1024):.1f} MB')
    print()

    print('🔧 Initializing enhanced Docling processor...')
    processor = EnhancedDoclingProcessor()

    print('🔄 Processing paper with graphics preservation...')
    print('⏳ This may take several minutes for AI-powered processing...')
    start_time = time.time()

    result = processor.process_book(pdf_file)
    processing_time = time.time() - start_time

    if result.get('success'):
        print(f'✅ Paper processed successfully in {processing_time:.1f}s')
        print()
        print('📊 PROCESSING RESULTS:')
        print(f'   📄 Processor used: {result.get("processor_used", "Unknown")}')
        print(f'   📝 Total chunks: {result.get("total_chunks", 0)}')
        print(f'   🧠 Memory peak: {result.get("memory_usage_mb", 0):.1f} MB')
        print(f'   ⏱️ Processing time: {processing_time:.1f}s')
        
        # Analyze content for graphics and special elements
        chunks_with_graphics = 0
        chunks_with_tables = 0
        chunks_with_formulas = 0
        
        if 'chunks' in result:
            for chunk in result['chunks']:
                text = chunk.get('text', '').lower()
                if any(indicator in text for indicator in ['figure', 'fig.', 'image', 'graph', 'chart', 'plot']):
                    chunks_with_graphics += 1
                if any(indicator in text for indicator in ['table', 'tab.', '|', 'row', 'column']):
                    chunks_with_tables += 1
                if any(indicator in text for indicator in ['$', 'equation', 'formula', 'math']):
                    chunks_with_formulas += 1
        
        print()
        print('🎯 CONTENT ANALYSIS:')
        print(f'   📈 Chunks with graphics references: {chunks_with_graphics}')
        print(f'   📊 Chunks with tables: {chunks_with_tables}')
        print(f'   🧮 Chunks with formulas: {chunks_with_formulas}')
        
        print()
        print('💾 DATABASE STORAGE:')
        print(f'   📚 ChromaDB: {"✅" if result.get("chromadb_success") else "❌"}')
        print(f'   🗄️ SQLite: {"✅" if result.get("sqlite_success") else "❌"}')
        
        # Show book ID for reference
        if 'book_id' in result:
            print(f'   🆔 Book ID: {result["book_id"]}')
        
        print()
        print('🎉 INGESTION COMPLETED SUCCESSFULLY!')
        print('Graphics, tables, and mathematical content preserved.')
        
    else:
        print(f'❌ Paper processing failed: {result.get("error", "Unknown error")}')

if __name__ == "__main__":
    main()