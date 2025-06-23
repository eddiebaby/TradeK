#!/usr/bin/env python3
"""
Database Cleanup Script - Remove old books, keep only today's processed book
"""

import os
import sys
import sqlite3
from pathlib import Path

# Add project root to path
sys.path.append('.')

try:
    import chromadb
    from chromadb.config import Settings
except ImportError as e:
    print(f"❌ Missing ChromaDB: {e}")
    sys.exit(1)

def cleanup_chromadb():
    """Clean up old books from ChromaDB, keep only target book"""
    print("🧹 Cleaning up ChromaDB...")
    
    try:
        # Connect to ChromaDB
        client = chromadb.PersistentClient(path="./data/qdrant")
        
        # List all collections
        collections = client.list_collections()
        print(f"📊 Found {len(collections)} ChromaDB collections")
        
        if not collections:
            print("✅ No ChromaDB collections found - nothing to clean")
            return
        
        # Target book to keep (today's processed book)
        target_book_id = "machinelearningforfa_6a6d3ce0"
        
        for collection_info in collections:
            collection_name = collection_info.name
            print(f"🔍 Processing collection: {collection_name}")
            
            try:
                collection = client.get_collection(collection_name)
                
                # Get all metadata to find old books
                all_data = collection.get()
                
                if not all_data or not all_data.get('ids'):
                    print(f"   ✅ Collection {collection_name} is empty")
                    continue
                    
                # Group by book_id to count chunks per book
                book_counts = {}
                ids_to_delete = []
                
                for i, chunk_id in enumerate(all_data['ids']):
                    metadata = all_data['metadatas'][i] if all_data.get('metadatas') else {}
                    book_id = metadata.get('book_id', 'unknown')
                    
                    if book_id not in book_counts:
                        book_counts[book_id] = 0
                    book_counts[book_id] += 1
                    
                    # Mark for deletion if not target book
                    if book_id != target_book_id:
                        ids_to_delete.append(chunk_id)
                
                print(f"   📊 Found books in {collection_name}:")
                for book_id, count in book_counts.items():
                    status = "🎯 KEEP" if book_id == target_book_id else "🗑️ DELETE"
                    print(f"      {book_id}: {count} chunks - {status}")
                
                # Delete old books
                if ids_to_delete:
                    print(f"   🗑️ Deleting {len(ids_to_delete)} chunks from {len(book_counts) - 1} old books...")
                    
                    # Delete in batches to avoid memory issues
                    batch_size = 100
                    for i in range(0, len(ids_to_delete), batch_size):
                        batch = ids_to_delete[i:i + batch_size]
                        collection.delete(ids=batch)
                        print(f"      Deleted batch {i//batch_size + 1}/{(len(ids_to_delete) + batch_size - 1)//batch_size}")
                    
                    print(f"   ✅ Collection {collection_name} cleanup completed")
                else:
                    print(f"   ✅ No old books to delete from {collection_name}")
                    
            except Exception as e:
                print(f"   ❌ Error processing collection {collection_name}: {e}")
            
    except Exception as e:
        print(f"❌ ChromaDB connection error: {e}")

def cleanup_sqlite():
    """Clean up old books from SQLite, keep only target book"""
    print("\n🧹 Cleaning up SQLite...")
    
    db_path = Path("data/knowledge.db")
    if not db_path.exists():
        print("❌ SQLite database not found")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Target book to keep
        target_book_id = "machinelearningforfa_6a6d3ce0"
        
        # Check current books by analyzing chunks table
        cursor.execute("""
            SELECT book_id, COUNT(*) as chunk_count, MIN(created_at) as first_chunk
            FROM chunks 
            GROUP BY book_id 
            ORDER BY first_chunk DESC
        """)
        chunk_books = cursor.fetchall()
        
        # Check books table
        cursor.execute("SELECT id, title FROM books")
        book_metadata = {row[0]: row[1] for row in cursor.fetchall()}
        
        print("📊 Found books in SQLite:")
        books_to_delete = []
        chunks_to_delete = []
        
        for book_id, chunk_count, first_chunk in chunk_books:
            title = book_metadata.get(book_id, "Unknown Title")
            if book_id == target_book_id:
                print(f"   {book_id}: {title} - {chunk_count} chunks - 🎯 KEEP")
            else:
                print(f"   {book_id}: {title} - {chunk_count} chunks - 🗑️ DELETE")
                books_to_delete.append(book_id)
                chunks_to_delete.append(book_id)
        
        # Delete old books
        if books_to_delete:
            print(f"\n🗑️ Deleting {len(books_to_delete)} old books from SQLite...")
            
            # Delete chunks first
            for book_id in chunks_to_delete:
                cursor.execute("DELETE FROM chunks WHERE book_id = ?", (book_id,))
                deleted_chunks = cursor.rowcount
                print(f"   Deleted {deleted_chunks} chunks for book: {book_id}")
            
            # Delete book metadata
            for book_id in books_to_delete:
                cursor.execute("DELETE FROM books WHERE id = ?", (book_id,))
                deleted_books = cursor.rowcount
                print(f"   Deleted {deleted_books} book record for: {book_id}")
            
            conn.commit()
            print("✅ SQLite cleanup completed")
        else:
            print("✅ No old books to delete from SQLite")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ SQLite error: {e}")

def verify_cleanup():
    """Verify that only target book remains"""
    print("\n🔍 Verifying cleanup results...")
    
    target_book_id = "machinelearningforfa_6a6d3ce0"
    
    # Check ChromaDB
    try:
        client = chromadb.PersistentClient(path="./data/qdrant")
        collections = client.list_collections()
        
        if collections:
            print(f"📊 ChromaDB verification:")
            for collection_info in collections:
                collection = client.get_collection(collection_info.name)
                all_data = collection.get()
                
                if all_data and all_data.get('ids'):
                    book_ids = set()
                    for metadata in all_data.get('metadatas', []):
                        if metadata:
                            book_ids.add(metadata.get('book_id', 'unknown'))
                    
                    print(f"   Collection {collection_info.name}: {len(book_ids)} books")
                    for book_id in book_ids:
                        if book_id == target_book_id:
                            print(f"      ✅ {book_id} (target book)")
                        else:
                            print(f"      ⚠️ {book_id} (unexpected)")
                else:
                    print(f"   Collection {collection_info.name}: Empty")
        else:
            print("📊 ChromaDB verification: No collections found")
            
    except Exception as e:
        print(f"❌ ChromaDB verification error: {e}")
    
    # Check SQLite
    try:
        db_path = Path("data/knowledge.db")
        if db_path.exists():
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check chunks table
            cursor.execute("SELECT DISTINCT book_id FROM chunks")
            chunk_book_ids = [row[0] for row in cursor.fetchall()]
            
            # Check books table
            cursor.execute("SELECT DISTINCT id FROM books")
            book_record_ids = [row[0] for row in cursor.fetchall()]
            
            print(f"📊 SQLite verification:")
            print(f"   Chunks table - Books remaining: {len(chunk_book_ids)}")
            for book_id in chunk_book_ids:
                if book_id == target_book_id:
                    print(f"      ✅ {book_id} (target book)")
                else:
                    print(f"      ⚠️ {book_id} (unexpected)")
            
            print(f"   Books table - Records remaining: {len(book_record_ids)}")
            for book_id in book_record_ids:
                if book_id == target_book_id:
                    print(f"      ✅ {book_id} (target book)")
                else:
                    print(f"      ⚠️ {book_id} (unexpected)")
            
            conn.close()
        else:
            print("📊 SQLite verification: Database not found")
            
    except Exception as e:
        print(f"❌ SQLite verification error: {e}")

def main():
    print("🗑️ DATABASE CLEANUP - Remove Old Books")
    print("=" * 50)
    print(f"Target: Keep only 'machinelearningforfa_6a6d3ce0' (today's CRC book)")
    print()
    
    # Cleanup both databases
    cleanup_chromadb()
    cleanup_sqlite()
    
    # Verify results
    verify_cleanup()
    
    print("\n🎉 Database cleanup completed!")
    print("Only today's processed CRC book should remain in both databases.")

if __name__ == "__main__":
    main()