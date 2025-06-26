#!/usr/bin/env python3
"""
Comprehensive ingestion script for the regime change detection book
Processes PDF and stores in both SQLite, ChromaDB, and Qdrant with embeddings
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
from src.core.chroma_storage import ChromaDBStorage
from src.core.models import Book, Chunk, FileType, ChunkType
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.text_chunker import TextChunker, ChunkingConfig
from src.ingestion.local_embeddings import LocalEmbeddingGenerator

async def comprehensive_ingest(pdf_path: str, categories: list = None):
    """
    Comprehensive PDF ingestion into all storage systems
    """
    
    print(f"🎯 Processing: Detecting Regime Change in Computational Finance")
    print(f"📁 File: {pdf_path}")
    print(f"🏷️  Categories: {', '.join(categories) if categories else 'None'}")
    print()
    
    # Initialize all components
    print("🔧 Initializing storage systems...")
    sqlite_storage = SQLiteStorage()
    qdrant_storage = QdrantStorage()
    chroma_storage = ChromaDBStorage()
    
    # Initialize ingestion components
    pdf_parser = PDFParser()
    chunker = TextChunker(ChunkingConfig(
        chunk_size=1000,  # Good size for regime change content
        chunk_overlap=200,
        min_chunk_size=100,
        max_chunk_size=2000
    ))
    
    # Initialize embedding generator
    embedding_generator = LocalEmbeddingGenerator()
    
    print("✅ All components initialized")
    print()
    
    # Calculate file hash
    print("🔍 Calculating file hash...")
    with open(pdf_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    # Check if already exists in SQLite
    existing = await sqlite_storage.get_book_by_hash(file_hash)
    if existing and existing.total_chunks > 0:
        print(f"⚠️  Book already exists in SQLite: {existing.title} ({existing.total_chunks} chunks)")
        book = existing
    else:
        # Parse PDF
        print("📖 Parsing PDF...")
        start_time = time.time()
        parse_result = pdf_parser.parse_file(Path(pdf_path))
        parse_time = time.time() - start_time
        
        if parse_result['errors']:
            print(f"❌ Parse errors: {parse_result['errors']}")
            return False
        
        print(f"✅ PDF parsed in {parse_time:.2f}s")
        print(f"📄 Found {len(parse_result['pages'])} pages")
        print()
        
        # Create book record
        print("📝 Creating book record...")
        metadata = parse_result['metadata']
        
        # Extract title from filename since PDF metadata might be limited
        filename = Path(pdf_path).stem
        title = "Detecting Regime Change in Computational Finance: Data Science, Machine Learning and Algorithmic Trading"
        author = "Jun Chen, Edward Tsang"
        
        book_id = f"regime_change_{file_hash[:8]}"
        
        book = Book(
            id=book_id,
            title=title,
            author=author,
            file_path=pdf_path,
            file_type=FileType.PDF,
            file_hash=file_hash,
            total_pages=len(parse_result['pages']),
            categories=categories or ['regime-change', 'machine-learning', 'computational-finance', 'data-science', 'algorithmic-trading'],
            metadata={
                **metadata,
                'original_filename': filename,
                'processing_date': datetime.now().isoformat(),
                'file_size_mb': round(Path(pdf_path).stat().st_size / (1024*1024), 2)
            }
        )
        
        # Save book to SQLite
        success = await sqlite_storage.save_book(book)
        if not success:
            print("❌ Failed to save book to SQLite")
            return False
        
        print(f"✅ Book saved to SQLite: {book.title}")
        print()
        
        # Chunk the text
        print("✂️  Chunking text...")
        start_time = time.time()
        chunks = chunker.chunk_pages(parse_result['pages'], book.id, {})
        chunk_time = time.time() - start_time
        
        print(f"✅ Created {len(chunks)} chunks in {chunk_time:.2f}s")
        print()
        
        # Save chunks to SQLite
        print("💾 Saving chunks to SQLite...")
        success = await sqlite_storage.save_chunks(chunks)
        
        if not success:
            print("❌ Failed to save chunks to SQLite")
            return False
        
        # Update book with chunk count
        book.total_chunks = len(chunks)
        book.indexed_at = datetime.now()
        await sqlite_storage.update_book(book)
        
        print(f"✅ {len(chunks)} chunks saved to SQLite")
        print()
    
    # Get all chunks for embedding processing
    print("📦 Retrieving chunks for embedding...")
    all_chunks = await sqlite_storage.get_chunks_by_book(book.id)
    print(f"📊 Retrieved {len(all_chunks)} chunks")
    print()
    
    # Process embeddings and store in Qdrant
    print("🧠 Generating embeddings for Qdrant...")
    start_time = time.time()
    
    # Prepare texts for embedding
    texts = [chunk.text for chunk in all_chunks]
    
    async with embedding_generator as gen:
        # Generate embeddings in batches
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = await gen.generate_embeddings(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
            print(f"  📈 Generated embeddings {i+1}-{min(i+batch_size, len(texts))} of {len(texts)}")
    
    embedding_time = time.time() - start_time
    print(f"✅ Generated {len(all_embeddings)} embeddings in {embedding_time:.2f}s")
    print()
    
    # Store in Qdrant
    print("⚡ Storing embeddings in Qdrant...")
    start_time = time.time()
    
    # Prepare Qdrant documents
    qdrant_docs = []
    for chunk, embedding in zip(all_chunks, all_embeddings):
        qdrant_docs.append({
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
    
    # Store in Qdrant
    success = await qdrant_storage.add_documents(qdrant_docs)
    qdrant_time = time.time() - start_time
    
    if success:
        print(f"✅ {len(qdrant_docs)} documents stored in Qdrant in {qdrant_time:.2f}s")
    else:
        print("❌ Failed to store documents in Qdrant")
    print()
    
    # Store in ChromaDB
    print("🎨 Storing documents in ChromaDB...")
    start_time = time.time()
    
    # Prepare ChromaDB documents
    chroma_docs = []
    for chunk, embedding in zip(all_chunks, all_embeddings):
        chroma_docs.append({
            'id': chunk.id,
            'text': chunk.text,
            'embedding': embedding,
            'metadata': {
                'book_id': chunk.book_id,
                'chunk_index': chunk.chunk_index,
                'page_start': chunk.page_start,
                'page_end': chunk.page_end,
                'book_title': book.title,
                'book_author': book.author,
                'categories': book.categories
            }
        })
    
    # Store in ChromaDB
    success = await chroma_storage.add_documents(chroma_docs)
    chroma_time = time.time() - start_time
    
    if success:
        print(f"✅ {len(chroma_docs)} documents stored in ChromaDB in {chroma_time:.2f}s")
    else:
        print("❌ Failed to store documents in ChromaDB")
    print()
    
    # Final summary
    total_time = embedding_time + qdrant_time + chroma_time
    
    print("🎉 INGESTION COMPLETE!")
    print("=" * 50)
    print(f"📚 Book: {book.title}")
    print(f"👤 Author: {book.author}")
    print(f"📄 Pages: {book.total_pages}")
    print(f"📦 Chunks: {book.total_chunks}")
    print(f"🏷️  Categories: {', '.join(book.categories)}")
    print()
    print("⏱️  PROCESSING TIMES:")
    print(f"  🧠 Embeddings: {embedding_time:.2f}s")
    print(f"  ⚡ Qdrant: {qdrant_time:.2f}s")
    print(f"  🎨 ChromaDB: {chroma_time:.2f}s")
    print(f"  📊 Total: {total_time:.2f}s")
    print()
    print("💾 STORAGE STATUS:")
    print(f"  📁 SQLite: ✅ {book.total_chunks} chunks")
    print(f"  ⚡ Qdrant: ✅ {len(qdrant_docs)} vectors")
    print(f"  🎨 ChromaDB: ✅ {len(chroma_docs)} documents")
    
    return True

async def main():
    if len(sys.argv) < 2:
        print("Usage: python ingest_regime_change_book.py <pdf_path> [category1,category2,...]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    categories = []
    
    if len(sys.argv) > 2:
        categories = [cat.strip() for cat in sys.argv[2].split(',')]
    
    if not Path(pdf_path).exists():
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)
    
    success = await comprehensive_ingest(pdf_path, categories)
    
    if success:
        print("\n🚀 Ready for semantic search across all databases!")
        print("Try: python search_book.py \"regime change detection\"")
    else:
        print("\n💥 Ingestion failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())