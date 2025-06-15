#!/usr/bin/env python3
"""
Try ingestion using PyPDF2 instead of pdfplumber
"""

import sys
import hashlib
import time
import asyncio
from pathlib import Path
from datetime import datetime

import PyPDF2

# Add src to path
sys.path.append('src')

from src.core.sqlite_storage import SQLiteStorage
from src.core.qdrant_storage import QdrantStorage
from src.core.models import Book, Chunk, FileType, ChunkType
from src.ingestion.text_chunker import TextChunker, ChunkingConfig
from src.ingestion.local_embeddings import LocalEmbeddingGenerator

def extract_with_pypdf2(pdf_path: str) -> dict:
    """Extract PDF using PyPDF2"""
    
    print(f"📖 Extracting with PyPDF2: {Path(pdf_path).name}")
    
    pages = []
    errors = []
    
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            print(f"  📄 Found {len(reader.pages)} pages")
            
            for i, page in enumerate(reader.pages):
                try:
                    text = page.extract_text() or ""
                    text = text.strip()
                    
                    pages.append({
                        'page_number': i + 1,
                        'text': text
                    })
                    
                    print(f"    Page {i+1}: {len(text)} chars")
                    
                except Exception as e:
                    error_msg = f"Page {i+1} extraction failed: {e}"
                    errors.append(error_msg)
                    print(f"    ❌ {error_msg}")
        
        return {
            'pages': pages,
            'errors': errors,
            'metadata': {
                'total_pages': len(pages),
                'extraction_method': 'pypdf2'
            }
        }
        
    except Exception as e:
        print(f"❌ PyPDF2 extraction failed: {e}")
        return {
            'pages': [],
            'errors': [f"PyPDF2 extraction failed: {e}"],
            'metadata': {}
        }

async def pypdf2_ingest(pdf_path: str):
    """Ingest using PyPDF2"""
    
    print(f"🚀 PYPDF2 INGESTION")
    print(f"📁 File: {Path(pdf_path).name}")
    
    # Extract with PyPDF2
    parse_result = extract_with_pypdf2(pdf_path)
    
    if not parse_result['pages']:
        print("❌ No pages extracted")
        return False
    
    total_chars = sum(len(page['text']) for page in parse_result['pages'])
    print(f"📊 Total characters: {total_chars:,}")
    
    # Show samples
    for i, page in enumerate(parse_result['pages'][:3]):
        sample = page['text'][:200] + "..." if len(page['text']) > 200 else page['text']
        print(f"  📄 Page {i+1}: {sample}")
    
    # Calculate hash
    hash_sha256 = hashlib.sha256()
    with open(pdf_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    file_hash = hash_sha256.hexdigest()
    
    # Create book
    book_id = f"pypdf2_{file_hash[:8]}"
    
    book = Book(
        id=book_id,
        title="DiGA for LOB (PyPDF2)",
        author="Academic Paper",
        file_path=pdf_path,
        file_type=FileType.PDF,
        file_hash=file_hash,
        total_pages=len(parse_result['pages']),
        categories=['academic', 'finance'],
        metadata={
            'processing_date': datetime.now().isoformat(),
            'total_characters': total_chars,
            'extraction_method': 'pypdf2'
        }
    )
    
    # Save to database
    sqlite_storage = SQLiteStorage()
    qdrant_storage = QdrantStorage()
    
    success = await sqlite_storage.save_book(book)
    if not success:
        print("❌ Failed to save book")
        return False
    
    # Chunk
    chunker = TextChunker(ChunkingConfig(
        chunk_size=1000,
        chunk_overlap=200,
        min_chunk_size=100,
        max_chunk_size=1500
    ))
    
    chunks = chunker.chunk_pages(parse_result['pages'], book.id, {})
    print(f"✅ Created {len(chunks)} chunks")
    
    # Save chunks
    success = await sqlite_storage.save_chunks(chunks)
    if not success:
        print("❌ Failed to save chunks")
        return False
    
    # Update book
    book.total_chunks = len(chunks)
    book.indexed_at = datetime.now()
    await sqlite_storage.update_book(book)
    
    # Verify
    retrieved = await sqlite_storage.get_chunks_by_book(book.id)
    print(f"📦 Verified: {len(retrieved)} chunks in database")
    
    # Generate embeddings
    print("🧠 Generating embeddings...")
    embedding_generator = LocalEmbeddingGenerator()
    
    async with embedding_generator as gen:
        all_embeddings = []
        batch_size = 8
        
        for i in range(0, len(retrieved), batch_size):
            batch = retrieved[i:i + batch_size]
            batch_embeddings = await gen.generate_embeddings(batch)
            all_embeddings.extend(batch_embeddings)
            print(f"  🧠 Embedded {i+1}-{min(i+batch_size, len(retrieved))} of {len(retrieved)}")
    
    # Save to Qdrant
    success = await qdrant_storage.save_embeddings(retrieved, all_embeddings)
    
    if success:
        print(f"✅ Complete! {len(retrieved)} chunks with embeddings")
        print(f"📊 Total characters: {total_chars:,}")
        print(f"🔢 Estimated tokens: ~{total_chars // 4:,}")
    else:
        print("❌ Failed to save embeddings")
        return False
    
    return True

async def main():
    if len(sys.argv) < 2:
        print("Usage: python pypdf2_ingest.py <pdf_path>")
        sys.exit(1)
    
    success = await pypdf2_ingest(sys.argv[1])
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())