#!/usr/bin/env python3
"""
EPUB ingestion using existing EPUB parser
"""

import sys
import hashlib
import time
import asyncio
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append('src')

from src.core.sqlite_storage import SQLiteStorage
from src.core.qdrant_storage import QdrantStorage
from src.core.models import Book, Chunk, FileType, ChunkType
from src.ingestion.epub_parser import EPUBParser
from src.ingestion.text_chunker import TextChunker, ChunkingConfig
from src.ingestion.local_embeddings import LocalEmbeddingGenerator

async def epub_ingest(epub_path: str):
    """Ingest EPUB file with full pipeline"""
    
    print(f"📚 EPUB INGESTION")
    print(f"📁 File: {Path(epub_path).name}")
    print(f"📊 Size: {Path(epub_path).stat().st_size / (1024*1024):.2f}MB")
    print("=" * 60)
    
    if not Path(epub_path).exists():
        print(f"❌ File not found: {epub_path}")
        return False
    
    # Calculate hash
    print("🔍 Calculating file hash...")
    hash_sha256 = hashlib.sha256()
    with open(epub_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    file_hash = hash_sha256.hexdigest()
    
    # Initialize storage
    sqlite_storage = SQLiteStorage()
    qdrant_storage = QdrantStorage()
    
    # Check if already exists
    existing = await sqlite_storage.get_book_by_hash(file_hash)
    if existing and existing.total_chunks > 0:
        print(f"✅ Already exists: {existing.title} ({existing.total_chunks} chunks)")
        return True
    
    # Parse EPUB
    print("📖 Parsing EPUB...")
    epub_parser = EPUBParser()
    
    try:
        start_time = time.time()
        parse_result = epub_parser.parse_file(Path(epub_path))
        parse_time = time.time() - start_time
        
        if parse_result['errors']:
            print(f"⚠️ Parse warnings: {len(parse_result['errors'])}")
            for error in parse_result['errors'][:3]:
                print(f"    - {error}")
        
        print(f"✅ EPUB parsed in {parse_time:.2f}s")
        print(f"📄 Found {len(parse_result['pages'])} chapters")
        
        # Show metadata
        metadata = parse_result.get('metadata', {})
        print(f"📚 Title: {metadata.get('title', 'Unknown')}")
        print(f"👤 Author: {metadata.get('author', 'Unknown')}")
        print(f"📅 Publisher: {metadata.get('publisher', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ EPUB parsing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Calculate content stats
    total_chars = sum(len(page['text']) for page in parse_result['pages'])
    print(f"📊 Total characters: {total_chars:,}")
    
    # Show chapter samples
    for i, page in enumerate(parse_result['pages'][:3]):
        sample = page['text'][:200].replace('\n', ' ') + "..." if len(page['text']) > 200 else page['text']
        print(f"  📄 Chapter {i+1}: {len(page['text'])} chars - {sample}")
    
    # Create book record
    book_id = f"epub_{file_hash[:8]}"
    
    book = Book(
        id=book_id,
        title=metadata.get('title', Path(epub_path).stem),
        author=metadata.get('author', 'Unknown'),
        file_path=epub_path,
        file_type=FileType.EPUB,
        file_hash=file_hash,
        total_pages=len(parse_result['pages']),
        categories=['programming', 'statistics', 'machine-learning', 'python'],
        metadata={
            **metadata,
            'processing_date': datetime.now().isoformat(),
            'file_size_mb': round(Path(epub_path).stat().st_size / (1024*1024), 2),
            'total_characters': total_chars,
            'epub_processing': True
        }
    )
    
    print(f"\n📚 Book record: {book.title}")
    
    # Save book
    success = await sqlite_storage.save_book(book)
    if not success:
        print("❌ Failed to save book")
        return False
    
    print("✅ Book saved to database")
    
    # Chunk the content
    print("\n✂️ Chunking content...")
    chunker = TextChunker(ChunkingConfig(
        chunk_size=1200,  # Larger chunks for technical content
        chunk_overlap=300,  # More overlap for context
        min_chunk_size=200,
        max_chunk_size=2500
    ))
    
    start_time = time.time()
    chunks = chunker.chunk_pages(parse_result['pages'], book.id, {})
    chunk_time = time.time() - start_time
    
    print(f"✅ Created {len(chunks)} chunks in {chunk_time:.2f}s")
    
    # Show chunk samples
    if chunks:
        for i in range(min(3, len(chunks))):
            chunk = chunks[i]
            sample = chunk.text[:150].replace('\n', ' ') + "..." if len(chunk.text) > 150 else chunk.text
            print(f"  📦 Chunk {i}: {len(chunk.text)} chars - {sample}")
    
    # Save chunks
    print("\n💾 Saving chunks...")
    start_time = time.time()
    
    # Save in batches
    batch_size = 20
    saved_count = 0
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        success = await sqlite_storage.save_chunks(batch)
        
        if success:
            saved_count += len(batch)
            print(f"  ✅ Saved batch {i//batch_size + 1}: {len(batch)} chunks")
        else:
            print(f"  ❌ Failed to save batch {i//batch_size + 1}")
            return False
    
    save_time = time.time() - start_time
    print(f"✅ Saved {saved_count} chunks in {save_time:.2f}s")
    
    # Update book record
    book.total_chunks = saved_count
    book.indexed_at = datetime.now()
    await sqlite_storage.update_book(book)
    
    # Verify chunks
    print("\n🔍 Verifying chunks...")
    retrieved_chunks = await sqlite_storage.get_chunks_by_book(book.id)
    print(f"📦 Retrieved {len(retrieved_chunks)} chunks from database")
    
    if len(retrieved_chunks) != saved_count:
        print(f"❌ Mismatch: saved {saved_count}, retrieved {len(retrieved_chunks)}")
        return False
    
    print("✅ Chunk verification passed")
    
    # Generate embeddings
    print("\n🧠 Generating embeddings...")
    start_time = time.time()
    
    embedding_generator = LocalEmbeddingGenerator()
    
    try:
        async with embedding_generator as gen:
            batch_size = 10  # Conservative batch size
            all_embeddings = []
            
            for i in range(0, len(retrieved_chunks), batch_size):
                batch_chunks = retrieved_chunks[i:i + batch_size]
                batch_embeddings = await gen.generate_embeddings(batch_chunks)
                all_embeddings.extend(batch_embeddings)
                
                print(f"  🧠 Embedded {i+1}-{min(i+batch_size, len(retrieved_chunks))} of {len(retrieved_chunks)}")
        
        embedding_time = time.time() - start_time
        print(f"✅ Generated {len(all_embeddings)} embeddings in {embedding_time:.2f}s")
        
        # Save to Qdrant
        print("⚡ Storing embeddings in Qdrant...")
        success = await qdrant_storage.save_embeddings(retrieved_chunks, all_embeddings)
        
        if success:
            print(f"✅ {len(all_embeddings)} vectors stored in Qdrant")
        else:
            print("❌ Failed to store embeddings in Qdrant")
            return False
        
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Final summary
    print(f"\n🎉 EPUB INGESTION COMPLETE!")
    print("=" * 60)
    print(f"📚 Title: {book.title}")
    print(f"👤 Author: {book.author}")
    print(f"📄 Chapters: {book.total_pages}")
    print(f"📦 Chunks: {len(retrieved_chunks)}")
    print(f"🧠 Embeddings: {len(all_embeddings)}")
    print(f"📊 Characters: {total_chars:,}")
    print(f"🔢 Estimated tokens: ~{total_chars // 4:,}")
    print(f"⏱️ Total time: {parse_time + chunk_time + save_time + embedding_time:.2f}s")
    print(f"💾 Storage: SQLite + Qdrant")
    print(f"🔍 Ready for semantic search!")
    
    return True

async def main():
    if len(sys.argv) < 2:
        print("Usage: python epub_ingest.py <epub_path>")
        sys.exit(1)
    
    success = await epub_ingest(sys.argv[1])
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())