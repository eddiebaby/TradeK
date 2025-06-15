#!/usr/bin/env python3
"""
Robust PDF ingestion that bypasses problematic PDF parser
Uses direct pdfplumber for reliable extraction
"""

import sys
import hashlib
import time
import asyncio
import re
from pathlib import Path
from datetime import datetime

import pdfplumber

# Add src to path
sys.path.append('src')

from src.core.sqlite_storage import SQLiteStorage
from src.core.qdrant_storage import QdrantStorage
from src.core.models import Book, Chunk, FileType, ChunkType
from src.ingestion.text_chunker import TextChunker, ChunkingConfig
from src.ingestion.local_embeddings import LocalEmbeddingGenerator

def extract_pdf_robust(pdf_path: str) -> dict:
    """Extract PDF content using direct pdfplumber"""
    
    print(f"📖 Extracting PDF: {Path(pdf_path).name}")
    
    pages = []
    errors = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"  📄 Found {len(pdf.pages)} pages")
            
            for i, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text() or ""
                    
                    # Basic cleanup
                    text = re.sub(r'\s+', ' ', text)  # normalize whitespace
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
                'extraction_method': 'pdfplumber_direct'
            }
        }
        
    except Exception as e:
        print(f"❌ PDF extraction failed: {e}")
        return {
            'pages': [],
            'errors': [f"PDF extraction failed: {e}"],
            'metadata': {}
        }

async def robust_ingest(pdf_path: str):
    """Robust PDF ingestion with step-by-step verification"""
    
    print(f"🚀 ROBUST INGESTION START")
    print(f"📁 File: {Path(pdf_path).name}")
    print("=" * 60)
    
    if not Path(pdf_path).exists():
        print(f"❌ File not found: {pdf_path}")
        return False
    
    file_size_mb = Path(pdf_path).stat().st_size / (1024*1024)
    print(f"📊 File size: {file_size_mb:.2f}MB")
    
    # Step 1: Extract PDF
    print(f"\n🔶 STEP 1: PDF EXTRACTION")
    start_time = time.time()
    parse_result = extract_pdf_robust(pdf_path)
    extract_time = time.time() - start_time
    
    if not parse_result['pages']:
        print(f"❌ No pages extracted")
        return False
    
    print(f"✅ Extracted {len(parse_result['pages'])} pages in {extract_time:.2f}s")
    
    # Calculate total characters
    total_chars = sum(len(page['text']) for page in parse_result['pages'])
    print(f"📊 Total characters: {total_chars:,}")
    
    # Show samples
    for i, page in enumerate(parse_result['pages'][:3]):
        sample = page['text'][:200].replace('\n', ' ') + "..." if len(page['text']) > 200 else page['text']
        print(f"  📄 Page {i+1} sample: {sample}")
    
    # Step 2: Create book record
    print(f"\n🔶 STEP 2: BOOK RECORD")
    
    # Calculate hash
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
    
    # Extract title from first page
    first_page_text = parse_result['pages'][0]['text'] if parse_result['pages'] else ""
    lines = first_page_text.split('\n' if '\n' in first_page_text else ' ')[:10]
    title = "DiGA for LOB"  # Default
    
    for line in lines:
        clean_line = line.strip()
        if 10 < len(clean_line) < 100 and not any(word in clean_line.lower() for word in ['abstract', 'page', 'figure']):
            title = clean_line
            break
    
    book_id = f"robust_{file_hash[:8]}"
    
    book = Book(
        id=book_id,
        title=title,
        author="Academic Paper",
        file_path=pdf_path,
        file_type=FileType.PDF,
        file_hash=file_hash,
        total_pages=len(parse_result['pages']),
        categories=['academic', 'finance', 'machine-learning'],
        metadata={
            'processing_date': datetime.now().isoformat(),
            'file_size_mb': file_size_mb,
            'total_characters': total_chars,
            'extraction_method': 'robust_pdfplumber'
        }
    )
    
    print(f"📚 Title: {book.title}")
    print(f"🆔 ID: {book.id}")
    
    # Save book
    success = await sqlite_storage.save_book(book)
    if not success:
        print("❌ Failed to save book")
        return False
    
    print("✅ Book record saved")
    
    # Step 3: Chunking
    print(f"\n🔶 STEP 3: CHUNKING")
    
    chunker = TextChunker(ChunkingConfig(
        chunk_size=1000,
        chunk_overlap=200,
        min_chunk_size=100,
        max_chunk_size=1500
    ))
    
    print("✂️ Creating chunks...")
    start_time = time.time()
    
    try:
        chunks = chunker.chunk_pages(parse_result['pages'], book.id, {})
        chunk_time = time.time() - start_time
        
        print(f"✅ Created {len(chunks)} chunks in {chunk_time:.2f}s")
        
        # Validate chunks
        valid_chunks = []
        for i, chunk in enumerate(chunks):
            if chunk.text.strip() and len(chunk.text) >= 50:
                valid_chunks.append(chunk)
            else:
                print(f"  ⚠️ Skipping invalid chunk {i}: {len(chunk.text)} chars")
        
        print(f"📊 Valid chunks: {len(valid_chunks)}")
        
        # Show chunk samples
        for i in range(min(3, len(valid_chunks))):
            chunk = valid_chunks[i]
            sample = chunk.text[:150].replace('\n', ' ') + "..." if len(chunk.text) > 150 else chunk.text
            print(f"  📄 Chunk {i}: {len(chunk.text)} chars - {sample}")
        
    except Exception as e:
        print(f"❌ Chunking failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Save chunks
    print(f"\n🔶 STEP 4: SAVING CHUNKS")
    
    print(f"💾 Saving {len(valid_chunks)} chunks...")
    start_time = time.time()
    
    try:
        # Save in smaller batches to avoid issues
        batch_size = 10
        saved_count = 0
        
        for i in range(0, len(valid_chunks), batch_size):
            batch = valid_chunks[i:i + batch_size]
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
        
    except Exception as e:
        print(f"❌ Saving chunks failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Verify chunks
    print(f"\n🔶 STEP 5: VERIFICATION")
    
    retrieved_chunks = await sqlite_storage.get_chunks_by_book(book.id)
    print(f"📦 Retrieved {len(retrieved_chunks)} chunks from database")
    
    if len(retrieved_chunks) != saved_count:
        print(f"❌ Mismatch: saved {saved_count}, retrieved {len(retrieved_chunks)}")
        return False
    
    print("✅ Chunk verification passed")
    
    # Step 6: Embeddings
    print(f"\n🔶 STEP 6: EMBEDDINGS")
    
    print("🧠 Generating embeddings...")
    start_time = time.time()
    
    embedding_generator = LocalEmbeddingGenerator()
    
    try:
        async with embedding_generator as gen:
            batch_size = 8
            all_embeddings = []
            
            for i in range(0, len(retrieved_chunks), batch_size):
                batch_chunks = retrieved_chunks[i:i + batch_size]
                batch_embeddings = await gen.generate_embeddings(batch_chunks)
                all_embeddings.extend(batch_embeddings)
                
                print(f"  🧠 Embedded {i+1}-{min(i+batch_size, len(retrieved_chunks))} of {len(retrieved_chunks)}")
        
        embedding_time = time.time() - start_time
        print(f"✅ Generated {len(all_embeddings)} embeddings in {embedding_time:.2f}s")
        
        # Save to Qdrant
        print("⚡ Storing in Qdrant...")
        success = await qdrant_storage.save_embeddings(retrieved_chunks, all_embeddings)
        
        if success:
            print(f"✅ {len(all_embeddings)} vectors stored in Qdrant")
        else:
            print("❌ Failed to store in Qdrant")
            return False
        
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Final summary
    print(f"\n🎉 ROBUST INGESTION COMPLETE!")
    print("=" * 60)
    print(f"📚 Title: {book.title}")
    print(f"📄 Pages: {book.total_pages}")
    print(f"📦 Chunks: {len(retrieved_chunks)}")
    print(f"🧠 Embeddings: {len(all_embeddings)}")
    print(f"📊 Characters: {total_chars:,}")
    print(f"🔢 Expected tokens: ~{total_chars // 4:,}")
    print(f"⏱️ Total time: {extract_time + chunk_time + save_time + embedding_time:.2f}s")
    
    return True

async def main():
    if len(sys.argv) < 2:
        print("Usage: python robust_ingest.py <pdf_path>")
        sys.exit(1)
    
    success = await robust_ingest(sys.argv[1])
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())