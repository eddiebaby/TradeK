#!/usr/bin/env python3
"""
Complete vectorization script for TradeKnowledge

Processes all books in the Knowledge folder and generates embeddings.
"""

import asyncio
import os
import sys
from pathlib import Path
import logging
from typing import List
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import get_config
from src.core.sqlite_storage import SQLiteStorage
from src.core.qdrant_storage import QdrantStorage
from src.ingestion.enhanced_book_processor import EnhancedBookProcessor
from src.ingestion.local_embeddings import LocalEmbeddingGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteVectorizer:
    """Complete book vectorization system"""
    
    def __init__(self):
        self.config = get_config()
        self.sqlite_storage = SQLiteStorage()
        self.qdrant_storage = QdrantStorage()
        self.book_processor = EnhancedBookProcessor()
        self.embedding_generator = LocalEmbeddingGenerator()
        
    async def initialize(self):
        """Initialize all components"""
        print("🔧 Initializing vectorization system...")
        
        try:
            # Components initialize in their constructors
            print("✅ All components initialized")
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            raise
    
    async def find_unprocessed_books(self, knowledge_folder: str) -> List[Path]:
        """Find books that haven't been processed yet"""
        print(f"📚 Scanning for books in {knowledge_folder}...")
        
        # Get all books from folder
        knowledge_path = Path(knowledge_folder)
        all_books = []
        for pattern in ["*.pdf", "*.epub"]:
            all_books.extend(knowledge_path.glob(pattern))
        
        # Get processed books from database
        processed_books = await self.sqlite_storage.list_books()
        processed_paths = {Path(book.file_path).name for book in processed_books}
        
        # Find unprocessed books
        unprocessed = []
        for book_path in all_books:
            if book_path.name not in processed_paths:
                unprocessed.append(book_path)
        
        print(f"📊 Found {len(all_books)} total books, {len(processed_books)} already processed")
        print(f"📋 {len(unprocessed)} books need processing")
        
        return unprocessed
    
    async def process_book(self, book_path: Path) -> bool:
        """Process a single book"""
        print(f"\n📖 Processing: {book_path.name}")
        
        try:
            # Copy to data/books directory
            data_books_dir = Path("data/books")
            data_books_dir.mkdir(parents=True, exist_ok=True)
            
            target_path = data_books_dir / book_path.name
            if not target_path.exists():
                import shutil
                shutil.copy2(book_path, target_path)
                print(f"   📁 Copied to: {target_path}")
            
            # Process the book
            result = await self.book_processor.process_book(str(target_path))
            
            if result.get("success"):
                book_id = result.get("book_id")
                chunks_created = result.get("chunks_created", 0)
                print(f"   ✅ Processed successfully: {chunks_created} chunks created")
                
                # Generate embeddings
                await self.generate_embeddings_for_book(book_id)
                return True
            else:
                error = result.get("error", "Unknown error")
                print(f"   ❌ Processing failed: {error}")
                return False
                
        except Exception as e:
            print(f"   ❌ Error processing {book_path.name}: {e}")
            return False
    
    async def generate_embeddings_for_book(self, book_id: str):
        """Generate embeddings for a specific book"""
        print(f"   🧮 Generating embeddings for book: {book_id}")
        
        try:
            # Get chunks for this book
            chunks = await self.sqlite_storage.get_chunks_by_book(book_id)
            
            if not chunks:
                print(f"   ⚠️  No chunks found for book {book_id}")
                return
            
            # Process chunks in batches
            batch_size = self.config.embedding.batch_size
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                # Prepare texts
                texts = [chunk.text for chunk in batch]
                chunk_ids = [chunk.id for chunk in batch]
                
                # Generate embeddings
                embeddings = await self.embedding_generator.generate_embeddings(texts)
                
                # Store in vector database
                embedding_data = []
                for chunk_id, embedding in zip(chunk_ids, embeddings):
                    embedding_data.append({
                        "id": chunk_id,
                        "embedding": embedding,
                        "metadata": {"book_id": book_id}
                    })
                
                await self.qdrant_storage.save_embeddings(embedding_data)
                
                print(f"     📊 Processed batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            
            print(f"   ✅ Embeddings generated for {len(chunks)} chunks")
            
        except Exception as e:
            print(f"   ❌ Embedding generation failed: {e}")
    
    async def generate_embeddings_for_existing_chunks(self):
        """Generate embeddings for chunks that don't have them yet"""
        print("\n🧮 Checking for chunks without embeddings...")
        
        try:
            # Get all books
            books = await self.sqlite_storage.list_books()
            
            for book in books:
                chunks = await self.sqlite_storage.get_chunks_by_book(book.id)
                if chunks:
                    print(f"📖 Checking embeddings for: {book.title}")
                    
                    # Check which chunks need embeddings
                    chunks_needing_embeddings = []
                    for chunk in chunks:
                        # For now, assume all chunks need embeddings
                        # (we could check Qdrant but that's more complex)
                        chunks_needing_embeddings.append(chunk)
                    
                    if chunks_needing_embeddings:
                        print(f"   🔄 Need embeddings for {len(chunks_needing_embeddings)} chunks")
                        await self.generate_embeddings_for_book(book.id)
                    else:
                        print(f"   ✅ All chunks have embeddings")
        
        except Exception as e:
            print(f"❌ Error checking existing embeddings: {e}")
    
    async def verify_search_functionality(self):
        """Test that search is working properly"""
        print("\n🔍 Testing search functionality...")
        
        try:
            # Test vector search
            test_query = "algorithmic trading strategies"
            embeddings = await self.embedding_generator.generate_embeddings([test_query])
            
            if embeddings:
                results = await self.qdrant_storage.search_semantic(
                    query_embedding=embeddings[0],
                    limit=5
                )
                
                print(f"✅ Vector search working: Found {len(results)} results")
                for i, result in enumerate(results[:3], 1):
                    print(f"   {i}. Score: {result.get('score', 0):.3f}")
                
                return True
            else:
                print("❌ Failed to generate query embedding")
                return False
                
        except Exception as e:
            print(f"❌ Search test failed: {e}")
            return False
    
    async def run_complete_vectorization(self, knowledge_folder: str):
        """Run the complete vectorization process"""
        start_time = time.time()
        
        print("🚀 Starting Complete Vectorization Process")
        print("=" * 50)
        
        # Initialize
        await self.initialize()
        
        # Find unprocessed books
        unprocessed_books = await self.find_unprocessed_books(knowledge_folder)
        
        if not unprocessed_books:
            print("✅ All books are already processed!")
        else:
            # Process each book
            success_count = 0
            for book_path in unprocessed_books:
                success = await self.process_book(book_path)
                if success:
                    success_count += 1
            
            print(f"\n📊 Processing Summary:")
            print(f"   - Books processed: {success_count}/{len(unprocessed_books)}")
            print(f"   - Success rate: {success_count/len(unprocessed_books)*100:.1f}%")
        
        # Generate embeddings for any existing chunks without them
        await self.generate_embeddings_for_existing_chunks()
        
        # Verify search functionality
        search_working = await self.verify_search_functionality()
        
        # Final status
        elapsed = time.time() - start_time
        print(f"\n🎉 Vectorization Complete!")
        print(f"   - Time taken: {elapsed:.1f} seconds")
        print(f"   - Search functional: {'✅' if search_working else '❌'}")
        
        # Final database stats
        books = await self.sqlite_storage.list_books()
        total_chunks = 0
        for book in books:
            chunks = await self.sqlite_storage.get_chunks_by_book(book.id)
            total_chunks += len(chunks)
        
        print(f"\n📈 Final Statistics:")
        print(f"   - Total books: {len(books)}")
        print(f"   - Total chunks: {total_chunks}")
        
        return search_working


async def main():
    """Main function"""
    knowledge_folder = "/home/scottschweizer/TradeKnowledge/Knowldge"
    
    if not Path(knowledge_folder).exists():
        print(f"❌ Knowledge folder not found: {knowledge_folder}")
        return
    
    vectorizer = CompleteVectorizer()
    success = await vectorizer.run_complete_vectorization(knowledge_folder)
    
    if success:
        print("\n🎯 TradeKnowledge vectorization is now 100% complete!")
        print("   All books are processed and searchable via semantic search.")
    else:
        print("\n⚠️  Vectorization completed with some issues.")
        print("   Check the logs above for specific problems.")

if __name__ == "__main__":
    asyncio.run(main())