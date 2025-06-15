#!/usr/bin/env python3
"""
Academic paper ingestion with enhanced LaTeX/math handling
"""

import sys
import hashlib
import time
import asyncio
import re
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append('src')

from src.core.sqlite_storage import SQLiteStorage
from src.core.qdrant_storage import QdrantStorage
from src.core.models import Book, Chunk, FileType, ChunkType
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.text_chunker import TextChunker, ChunkingConfig
from src.ingestion.local_embeddings import LocalEmbeddingGenerator

def enhance_academic_text(text: str) -> str:
    """Enhance text extraction for academic papers"""
    
    # Preserve mathematical notation
    text = re.sub(r'([a-zA-Z])\s*\(\s*([x-z])\s*\)', r'\1(\2)', text)  # f(x) spacing
    text = re.sub(r'([a-zA-Z])\s*_\s*([0-9a-zA-Z])', r'\1_\2', text)  # subscripts
    text = re.sub(r'([a-zA-Z])\s*\^\s*([0-9a-zA-Z])', r'\1^\2', text)  # superscripts
    
    # Preserve common math symbols
    text = re.sub(r'\s*∥\s*', ' ∥ ', text)  # parallel symbol
    text = re.sub(r'\s*∈\s*', ' ∈ ', text)  # element of
    text = re.sub(r'\s*≤\s*', ' ≤ ', text)  # less than or equal
    text = re.sub(r'\s*≥\s*', ' ≥ ', text)  # greater than or equal
    text = re.sub(r'\s*∞\s*', ' ∞ ', text)  # infinity
    
    # Fix broken equations split across lines
    text = re.sub(r'([a-zA-Z0-9])\s*\n\s*([+\-*/=])', r'\1 \2', text)
    text = re.sub(r'([+\-*/=])\s*\n\s*([a-zA-Z0-9])', r'\1 \2', text)
    
    # Preserve Greek letters
    greek_letters = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω']
    for letter in greek_letters:
        text = re.sub(f'\\s*{letter}\\s*', f' {letter} ', text)
    
    # Clean up excessive whitespace but preserve structure
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # max 2 newlines
    text = re.sub(r'[ \t]+', ' ', text)  # normalize spaces
    
    return text.strip()

async def academic_ingest(pdf_path: str):
    """Enhanced ingestion for academic papers"""
    
    print(f"🎓 Academic paper ingestion: {Path(pdf_path).name}")
    print(f"📊 File size: {Path(pdf_path).stat().st_size / (1024*1024):.2f}MB")
    
    if not Path(pdf_path).exists():
        print(f"❌ File not found: {pdf_path}")
        return False
    
    # Calculate hash
    print("🔍 Calculating hash...")
    hash_sha256 = hashlib.sha256()
    with open(pdf_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    file_hash = hash_sha256.hexdigest()
    
    # Initialize storage
    sqlite_storage = SQLiteStorage()
    qdrant_storage = QdrantStorage()
    
    # Check if exists
    existing = await sqlite_storage.get_book_by_hash(file_hash)
    if existing and existing.total_chunks > 0:
        print(f"✅ Already exists: {existing.title} ({existing.total_chunks} chunks)")
        return True
    
    # Parse PDF with enhanced handling
    print("📖 Parsing PDF with academic enhancements...")
    pdf_parser = PDFParser(enable_ocr=False)
    
    try:
        parse_result = pdf_parser.parse_file(Path(pdf_path))
        print(f"✅ Found {len(parse_result['pages'])} pages")
        
        if parse_result['errors']:
            print(f"⚠️ Parse warnings: {len(parse_result['errors'])}")
            
    except Exception as e:
        print(f"❌ Parse failed: {e}")
        return False
    
    # Enhance text on each page
    print("🔬 Enhancing academic text...")
    enhanced_pages = []
    total_chars = 0
    
    for i, page in enumerate(parse_result['pages']):
        enhanced_text = enhance_academic_text(page['text'])
        enhanced_pages.append({
            'page_number': page['page_number'],
            'text': enhanced_text
        })
        total_chars += len(enhanced_text)
        
        if i < 3:  # Show first 3 pages sample
            sample = enhanced_text[:200] + "..." if len(enhanced_text) > 200 else enhanced_text
            print(f"  📄 Page {i+1} sample: {sample}")
    
    print(f"📊 Total characters: {total_chars:,}")
    
    # Create book record
    filename = Path(pdf_path).stem
    title = filename.replace('_', ' ').replace('-', ' ')
    
    # Try to extract title from first page
    first_page_text = enhanced_pages[0]['text'] if enhanced_pages else ""
    lines = first_page_text.split('\n')[:10]  # First 10 lines
    for line in lines:
        if len(line.strip()) > 10 and len(line.strip()) < 100:
            if not any(word in line.lower() for word in ['abstract', 'introduction', 'page', 'figure']):
                title = line.strip()
                break
    
    book_id = f"academic_{file_hash[:8]}"
    
    book = Book(
        id=book_id,
        title=title,
        author="Academic Paper",
        file_path=pdf_path,
        file_type=FileType.PDF,
        file_hash=file_hash,
        total_pages=len(enhanced_pages),
        categories=['academic', 'research', 'mathematics', 'machine-learning'],
        metadata={
            'processing_date': datetime.now().isoformat(),
            'file_size_mb': round(Path(pdf_path).stat().st_size / (1024*1024), 2),
            'total_characters': total_chars,
            'enhanced_academic': True
        }
    )
    
    print(f"📚 Book: {book.title}")
    
    # Save book
    success = await sqlite_storage.save_book(book)
    if not success:
        print("❌ Failed to save book")
        return False
    
    # Enhanced chunking for academic content
    print("✂️ Academic chunking...")
    chunker = TextChunker(ChunkingConfig(
        chunk_size=1200,  # Larger chunks for equations
        chunk_overlap=200,  # More overlap for context
        min_chunk_size=200,  # Minimum meaningful size
        max_chunk_size=2000  # Allow larger chunks
    ))
    
    chunks = chunker.chunk_pages(enhanced_pages, book.id, {})
    print(f"✅ Created {len(chunks)} academic chunks")
    
    # Show sample chunk
    if chunks:
        sample_chunk = chunks[0].text[:300] + "..." if len(chunks[0].text) > 300 else chunks[0].text
        print(f"📄 Sample chunk: {sample_chunk}")
    
    # Save chunks
    print("💾 Saving chunks...")
    success = await sqlite_storage.save_chunks(chunks)
    if not success:
        print("❌ Failed to save chunks")
        return False
    
    # Update book
    book.total_chunks = len(chunks)
    book.indexed_at = datetime.now()
    await sqlite_storage.update_book(book)
    
    # Generate embeddings
    print("🧠 Generating embeddings...")
    start_time = time.time()
    
    embedding_generator = LocalEmbeddingGenerator()
    
    async with embedding_generator as gen:
        batch_size = 8  # Smaller batches for academic content
        all_embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = await gen.generate_embeddings(batch_chunks)
            all_embeddings.extend(batch_embeddings)
            
            print(f"  🧠 Embedded {i+1}-{min(i+batch_size, len(chunks))} of {len(chunks)}")
    
    embedding_time = time.time() - start_time
    print(f"✅ Generated {len(all_embeddings)} embeddings in {embedding_time:.2f}s")
    
    # Save to Qdrant
    print("⚡ Storing in Qdrant...")
    success = await qdrant_storage.save_embeddings(chunks, all_embeddings)
    
    if success:
        print(f"✅ {len(chunks)} vectors stored in Qdrant")
    else:
        print("❌ Failed to store in Qdrant")
        return False
    
    print()
    print("🎉 ACADEMIC INGESTION COMPLETE!")
    print("=" * 50)
    print(f"📚 Title: {book.title}")
    print(f"📄 Pages: {book.total_pages}")
    print(f"📦 Chunks: {book.total_chunks}")
    print(f"🧠 Embeddings: {len(all_embeddings)}")
    print(f"📊 Characters: {total_chars:,}")
    print(f"⏱️ Embedding time: {embedding_time:.2f}s")
    
    return True

async def main():
    if len(sys.argv) < 2:
        print("Usage: python academic_ingest.py <pdf_path>")
        sys.exit(1)
    
    success = await academic_ingest(sys.argv[1])
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())