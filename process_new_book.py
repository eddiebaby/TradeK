#!/usr/bin/env python3
"""
Process new book using existing working pipeline
Based on the successful simple_book_processor.py approach
"""

import asyncio
import sys
import hashlib
import time
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.append('src')

from src.core.sqlite_storage import SQLiteStorage
from src.core.qdrant_storage import QdrantStorage
from src.core.models import Book, Chunk, FileType, ChunkType
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.text_chunker import TextChunker, ChunkingConfig
from src.ingestion.local_embeddings import LocalEmbeddingGenerator

async def process_book(pdf_path: str, title: str, author: str, categories: list = None):
    """
    Process book using our proven working approach
    """
    
    print(f"🎯 Processing: {title}")
    print(f"👤 Author: {author}")
    print(f"📁 File: {pdf_path}")
    print(f"🏷️  Categories: {', '.join(categories) if categories else 'None'}")
    print()
    
    # Initialize components
    print("🔧 Initializing components...")
    storage = SQLiteStorage()
    qdrant_storage = QdrantStorage()
    
    pdf_parser = PDFParser()
    chunker = TextChunker(ChunkingConfig(
        chunk_size=1000,
        chunk_overlap=200,
        min_chunk_size=100,
        max_chunk_size=2000
    ))
    
    print("✅ Components initialized")
    print()
    
    # Calculate file hash
    print("🔍 Calculating file hash...")
    with open(pdf_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    # Check if already exists
    existing = await storage.get_book_by_hash(file_hash)
    if existing and existing.total_chunks > 0:
        print(f"⚠️  Book already exists: {existing.title} ({existing.total_chunks} chunks)")
        return existing
    
    # Parse PDF
    print("📖 Parsing PDF...")
    start_time = time.time()
    
    try:
        parse_result = pdf_parser.parse_file(Path(pdf_path))
    except Exception as e:
        print(f"❌ PDF parsing failed: {e}")
        print("💡 Note: Some PDFs may require OCR or different parsing approaches")
        return None
    
    parse_time = time.time() - start_time
    
    if parse_result['errors']:
        print(f"⚠️  Parse warnings: {parse_result['errors']}")
    
    print(f"✅ PDF parsed in {parse_time:.2f}s")
    print(f"📄 Found {len(parse_result['pages'])} pages")
    print()
    
    # Create book record
    print("📝 Creating book record...")
    metadata = parse_result['metadata']
    
    book_id = f"{title.lower().replace(' ', '_').replace(',', '').replace(':', '')}_{file_hash[:8]}"
    
    book = Book(
        id=book_id,
        title=title,
        author=author,
        file_path=pdf_path,
        file_type=FileType.PDF,
        file_hash=file_hash,
        total_pages=len(parse_result['pages']),
        categories=categories or [],
        metadata={
            **metadata,
            'processing_date': datetime.now().isoformat(),
            'file_size_mb': round(Path(pdf_path).stat().st_size / (1024*1024), 2)
        }
    )
    
    # Save book
    success = await storage.save_book(book)
    if not success:
        print("❌ Failed to save book")
        return None
    
    print(f"✅ Book saved: {book.title}")
    print()
    
    # Chunk the text
    print("✂️  Chunking text...")
    start_time = time.time()
    chunks = chunker.chunk_pages(parse_result['pages'], book.id, {})
    chunk_time = time.time() - start_time
    
    print(f"✅ Created {len(chunks)} chunks in {chunk_time:.2f}s")
    print()
    
    # Save chunks
    print("💾 Saving chunks...")
    success = await storage.save_chunks(chunks)
    
    if success:
        # Update book with chunk count
        book.total_chunks = len(chunks)
        book.indexed_at = datetime.now()
        await storage.update_book(book)
        
        print(f"✅ {len(chunks)} chunks saved")
        print()
    else:
        print("❌ Failed to save chunks")
        return None
    
    # Generate embeddings and store in Qdrant
    print("🧠 Generating embeddings...")
    start_time = time.time()
    
    embedding_generator = LocalEmbeddingGenerator()
    
    try:
        async with embedding_generator as gen:
            # Process in batches to avoid memory issues
            batch_size = 16
            total_added = 0
            
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                print(f"  📈 Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
                
                # Generate embeddings for this batch
                batch_texts = [chunk.text for chunk in batch_chunks]
                batch_embeddings = await gen.generate_embeddings(batch_texts)
                
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
                    print(f"    ✅ Added {len(documents)} documents to Qdrant")
                else:
                    print(f"    ❌ Failed to add batch to Qdrant")
                    break
    
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
    print("=" * 50)
    print(f"📚 Book: {book.title}")
    print(f"👤 Author: {book.author}")
    print(f"📄 Pages: {book.total_pages}")
    print(f"📦 Chunks: {book.total_chunks}")
    print(f"🏷️  Categories: {', '.join(book.categories)}")
    print(f"⚡ Vectors: {total_added}")
    print()
    print("💾 STORAGE STATUS:")
    print(f"  📁 SQLite: ✅ Book and {book.total_chunks} chunks")
    print(f"  ⚡ Qdrant: ✅ {total_added} vectors" if total_added > 0 else "  ⚡ Qdrant: ⚠️  No vectors (text search still works)")
    
    return book

async def main():
    if len(sys.argv) < 2:
        print("Usage: python process_new_book.py <pdf_path> [category1,category2,...]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    categories = []
    
    if len(sys.argv) > 2:
        categories = [cat.strip() for cat in sys.argv[2].split(',')]
    
    if not Path(pdf_path).exists():
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)
    
    # Extract title and author from filename for this specific book
    if "regime_change" in pdf_path.lower() or "detecting_regime" in pdf_path.lower():
        title = "Detecting Regime Change in Computational Finance: Data Science, Machine Learning and Algorithmic Trading"
        author = "Jun Chen, Edward Tsang"
    else:
        # Generic fallback
        filename = Path(pdf_path).stem
        title = filename.replace('_', ' ').title()
        author = "Unknown"
    
    book = await process_book(pdf_path, title, author, categories)
    
    if book:
        print(f"\n🚀 Book successfully processed!")
        print(f"📊 Total documents in system: {book.total_chunks} (from this book)")
        print("\n🔍 Try searching:")
        print(f'python search_book.py "regime change"')
        print(f'python search_book.py "machine learning finance"')
    else:
        print("\n💥 Processing failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())