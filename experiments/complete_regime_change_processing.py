#!/usr/bin/env python3
"""
Complete the processing of the existing regime change book
It's in the database but has 0 chunks - need to chunk and embed it
"""

import asyncio
import sys
import time
import os
from pathlib import Path
from datetime import datetime

# Set embedding dimension to match our existing data
os.environ['EMBEDDING_DIMENSION'] = '384'

# Add src to path
sys.path.append('src')

from src.core.sqlite_storage import SQLiteStorage
from src.core.qdrant_storage import QdrantStorage
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.text_chunker import TextChunker, ChunkingConfig
from src.ingestion.local_embeddings import LocalEmbeddingGenerator

async def complete_book_processing(book_id: str):
    """
    Complete processing for a book that exists but has no chunks
    """
    
    print(f"🔄 Completing processing for book: {book_id}")
    print()
    
    # Initialize components
    print("🔧 Initializing components...")
    storage = SQLiteStorage()
    qdrant_storage = QdrantStorage()
    
    # Get the existing book
    print("📖 Retrieving book from database...")
    book = await storage.get_book(book_id)
    if not book:
        print(f"❌ Book not found: {book_id}")
        return False
    
    print(f"✅ Found book: {book.title}")
    print(f"   👤 Author: {book.author}")
    print(f"   📄 Pages: {book.total_pages}")
    print(f"   📦 Current chunks: {book.total_chunks}")
    print(f"   📁 File: {book.file_path}")
    print()
    
    # Check if file still exists
    if not Path(book.file_path).exists():
        print(f"❌ File not found: {book.file_path}")
        return False
    
    # Initialize processing components
    pdf_parser = PDFParser()
    chunker = TextChunker(ChunkingConfig(
        chunk_size=1000,
        chunk_overlap=200,
        min_chunk_size=100,
        max_chunk_size=2000
    ))
    
    # Parse PDF again (we need the text content)
    print("📖 Re-parsing PDF for text content...")
    start_time = time.time()
    
    try:
        parse_result = pdf_parser.parse_file(Path(book.file_path))
    except Exception as e:
        print(f"❌ PDF parsing failed: {e}")
        print("💡 This might be a scanned PDF requiring OCR")
        return False
    
    parse_time = time.time() - start_time
    
    if parse_result['errors']:
        print(f"⚠️  Parse warnings: {parse_result['errors']}")
    
    print(f"✅ PDF re-parsed in {parse_time:.2f}s")
    print(f"📄 Extracted {len(parse_result['pages'])} pages of text")
    print()
    
    # Check if we got any text
    total_text = sum(len(page.get('text', '')) for page in parse_result['pages'])
    if total_text < 1000:
        print(f"⚠️  Very little text extracted ({total_text} characters)")
        print("💡 This might be a scanned PDF that needs OCR processing")
    
    # Create chunks
    print("✂️  Creating chunks...")
    start_time = time.time()
    chunks = chunker.chunk_pages(parse_result['pages'], book.id, {})
    chunk_time = time.time() - start_time
    
    print(f"✅ Created {len(chunks)} chunks in {chunk_time:.2f}s")
    
    if len(chunks) == 0:
        print("❌ No chunks were created - likely a scanned PDF")
        print("💡 Consider using OCR tools to extract text first")
        return False
    
    print()
    
    # Save chunks to SQLite
    print("💾 Saving chunks to SQLite...")
    success = await storage.save_chunks(chunks)
    
    if not success:
        print("❌ Failed to save chunks to SQLite")
        return False
    
    # Update book with chunk count
    book.total_chunks = len(chunks)
    book.indexed_at = datetime.now()
    await storage.update_book(book)
    
    print(f"✅ {len(chunks)} chunks saved to SQLite")
    print()
    
    # Generate embeddings and store in Qdrant
    print("🧠 Generating embeddings for Qdrant...")
    start_time = time.time()
    
    embedding_generator = LocalEmbeddingGenerator()
    
    try:
        # Fix model name issue
        embedding_generator.model_name = "nomic-embed-text:latest"
        embedding_generator.embedding_dimension = 384
        
        async with embedding_generator as gen:
            # Process in smaller batches for stability
            batch_size = 8
            total_added = 0
            
            print(f"🔄 Processing {len(chunks)} chunks in batches of {batch_size}")
            
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(chunks) + batch_size - 1) // batch_size
                
                print(f"  📈 Processing batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)")
                
                try:
                    # Generate embeddings for this batch
                    batch_embeddings = await gen.generate_embeddings(batch_chunks)
                    
                    # Prepare documents for Qdrant
                    documents = []
                    for chunk, embedding in zip(batch_chunks, batch_embeddings):
                        documents.append({
                            'id': chunk.id,
                            'vector': embedding,
                            'payload': {
                                'book_id': chunk.book_id,
                                'text': chunk.text,
                                'chunk_index': chunk.chunk_index,
                                'page_start': chunk.page_start,
                                'page_end': chunk.page_end,
                                'book_title': book.title,
                                'book_author': book.author,
                                'categories': book.categories
                            }
                        })
                    
                    # Add to Qdrant
                    success = await qdrant_storage.add_documents(documents)
                    if success:
                        total_added += len(documents)
                        print(f"    ✅ Added {len(documents)} vectors to Qdrant")
                    else:
                        print(f"    ❌ Failed to add batch {batch_num} to Qdrant")
                        
                except Exception as e:
                    print(f"    ❌ Error processing batch {batch_num}: {e}")
                    continue
    
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        print("💡 Continuing without vector embeddings - text search still available")
        total_added = 0
    
    embedding_time = time.time() - start_time
    print(f"✅ Embedding processing completed in {embedding_time:.2f}s")
    print(f"⚡ Added {total_added} vectors to Qdrant")
    print()
    
    # Final summary
    print("🎉 PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"📚 Book: {book.title}")
    print(f"👤 Author: {book.author}")
    print(f"📄 Pages: {book.total_pages}")
    print(f"📦 Chunks: {book.total_chunks}")
    print(f"🏷️  Categories: {', '.join(book.categories) if book.categories else 'None'}")
    print(f"⚡ Vectors: {total_added}")
    print()
    print("💾 FINAL STATUS:")
    print(f"  📁 SQLite: ✅ Book and {book.total_chunks} chunks")
    if total_added > 0:
        print(f"  ⚡ Qdrant: ✅ {total_added} vectors")
        print("  🔍 Search: ✅ Both text and semantic search available")
    else:
        print(f"  ⚡ Qdrant: ⚠️  No vectors (embeddings failed)")
        print("  🔍 Search: ⚠️  Only text search available")
    
    return True

async def main():
    """Main function"""
    # Hard-coded book ID from our query above
    book_id = "detecting_regime_change_in_computational_finance_data_science_machine_learning_and_algorithmic_trading_c96b7af5"
    
    print("🎯 COMPLETING REGIME CHANGE BOOK PROCESSING")
    print("=" * 60)
    print()
    
    success = await complete_book_processing(book_id)
    
    if success:
        print("\n🚀 Book processing completed!")
        print("\n🔍 Try these searches:")
        print('   python search_book.py "regime change"')
        print('   python search_book.py "machine learning finance"')
        print('   python search_book.py "data science trading"')
    else:
        print("\n💥 Processing failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())