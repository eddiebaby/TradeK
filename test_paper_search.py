#!/usr/bin/env python3
"""
Test search functionality for the ingested paper with graphics
"""

import sys
sys.path.append('.')

import sqlite3
from pathlib import Path

def test_paper_search():
    """Test various searches on the ingested paper"""
    
    print("🔍 TESTING PAPER SEARCH FUNCTIONALITY")
    print("=" * 50)
    
    db_path = Path("data/knowledge.db")
    if not db_path.exists():
        print("❌ Database not found")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if paper exists
    cursor.execute("SELECT COUNT(*) FROM chunks WHERE book_id = 'book_ca69c36d'")
    chunk_count = cursor.fetchone()[0]
    print(f"📚 Paper chunks in database: {chunk_count}")
    
    if chunk_count == 0:
        print("❌ Paper not found in database")
        return
    
    # Test searches for graphics content
    search_terms = [
        "figure",
        "table", 
        "embedding",
        "vec2vec",
        "translation",
        "similarity",
        "universal geometry"
    ]
    
    print("\n🎯 SEARCH RESULTS:")
    print("-" * 30)
    
    for term in search_terms:
        cursor.execute("""
            SELECT COUNT(*) 
            FROM chunks 
            WHERE book_id = 'book_ca69c36d' 
            AND text LIKE ?
        """, (f'%{term}%',))
        
        count = cursor.fetchone()[0]
        print(f"'{term}': {count} chunks")
        
        if count > 0:
            # Show a sample result
            cursor.execute("""
                SELECT chunk_index, SUBSTR(text, 1, 200) 
                FROM chunks 
                WHERE book_id = 'book_ca69c36d' 
                AND text LIKE ?
                LIMIT 1
            """, (f'%{term}%',))
            
            result = cursor.fetchone()
            if result:
                chunk_idx, preview = result
                print(f"   Sample (chunk {chunk_idx}): {preview}...")
        print()
    
    # Check for specific graphics references
    print("📊 GRAPHICS AND VISUAL CONTENT:")
    print("-" * 30)
    
    graphics_queries = [
        ("Figure references", "Figure%"),
        ("Table references", "Table%"), 
        ("Mathematical notation", "%$%"),
        ("Equations", "%equation%"),
        ("Diagrams", "%diagram%"),
        ("Heatmaps", "%heatmap%"),
        ("Visualization", "%visual%")
    ]
    
    for description, pattern in graphics_queries:
        cursor.execute("""
            SELECT COUNT(*), MIN(chunk_index) as first_chunk
            FROM chunks 
            WHERE book_id = 'book_ca69c36d' 
            AND text LIKE ?
        """, (pattern,))
        
        count, first_chunk = cursor.fetchone()
        if count > 0:
            print(f"✅ {description}: {count} chunks (first in chunk {first_chunk})")
        else:
            print(f"❌ {description}: Not found")
    
    # Show paper metadata
    print("\n📄 PAPER METADATA:")
    print("-" * 20)
    cursor.execute("""
        SELECT title, total_chunks, processor_used, created_at
        FROM books 
        WHERE id = 'book_ca69c36d'
    """)
    
    result = cursor.fetchone()
    if result:
        title, total_chunks, processor_used, created_at = result
        print(f"Title: {title}")
        print(f"Total chunks: {total_chunks}")
        print(f"Processor: {processor_used}")
        print(f"Ingested: {created_at}")
    
    conn.close()
    
    print("\n✅ Search test completed!")

if __name__ == "__main__":
    test_paper_search()