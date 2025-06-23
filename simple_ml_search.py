#!/usr/bin/env python3
"""
Simple ML content search using direct database queries.
This bypasses the complex search engine to directly check what ML content exists.
"""

import sqlite3
import asyncio
import sys
from pathlib import Path
import json
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchText

# Database paths
SQLITE_DB = "data/knowledge.db"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "tradeknowledge"

def search_sqlite_ml_content():
    """Search for ML content in SQLite database"""
    print("🔍 Searching SQLite database for ML content...")
    
    if not Path(SQLITE_DB).exists():
        print(f"❌ SQLite database not found at: {SQLITE_DB}")
        return []
    
    try:
        conn = sqlite3.connect(SQLITE_DB)
        conn.row_factory = sqlite3.Row  # Access columns by name
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"📋 Available tables: {tables}")
        
        results = []
        
        if 'chunks' in tables:
            # Search for ML-related terms in chunks
            ml_terms = [
                'ML', 'machine learning', 'artificial intelligence', 'AI',
                'neural network', 'deep learning', 'regression', 'classification',
                'algorithm', 'model', 'prediction', 'feature', 'training'
            ]
            
            for term in ml_terms:
                print(f"\n🔎 Searching for: '{term}'")
                
                # Try FTS search first
                try:
                    cursor.execute("""
                        SELECT c.id, c.text, c.book_id, c.page_start, b.title as book_title, b.author
                        FROM chunks c
                        LEFT JOIN books b ON c.book_id = b.id
                        WHERE c.text MATCH ?
                        LIMIT 10
                    """, (term,))
                    
                    fts_results = cursor.fetchall()
                    if fts_results:
                        print(f"   ✅ FTS found {len(fts_results)} results")
                        for row in fts_results[:3]:  # Show top 3
                            text_preview = row['text'][:150] + "..." if len(row['text']) > 150 else row['text']
                            results.append({
                                'term': term,
                                'method': 'FTS',
                                'chunk_id': row['id'],
                                'book_title': row['book_title'],
                                'author': row['author'],
                                'page': row['page_start'],
                                'preview': text_preview
                            })
                            print(f"     📖 {row['book_title']} (Page {row['page_start']})")
                            print(f"     📝 {text_preview}")
                except Exception as e:
                    print(f"   ❌ FTS search failed: {e}")
                    
                    # Fallback to LIKE search
                    try:
                        cursor.execute("""
                            SELECT c.id, c.text, c.book_id, c.page_start, b.title as book_title, b.author
                            FROM chunks c
                            LEFT JOIN books b ON c.book_id = b.id
                            WHERE LOWER(c.text) LIKE LOWER(?)
                            LIMIT 5
                        """, (f'%{term}%',))
                        
                        like_results = cursor.fetchall()
                        if like_results:
                            print(f"   ✅ LIKE search found {len(like_results)} results")
                            for row in like_results[:2]:  # Show top 2
                                text_preview = row['text'][:150] + "..." if len(row['text']) > 150 else row['text']
                                results.append({
                                    'term': term,
                                    'method': 'LIKE',
                                    'chunk_id': row['id'],
                                    'book_title': row['book_title'],
                                    'author': row['author'],
                                    'page': row['page_start'],
                                    'preview': text_preview
                                })
                                print(f"     📖 {row['book_title']} (Page {row['page_start']})")
                                print(f"     📝 {text_preview}")
                        else:
                            print(f"   ❌ No results for '{term}'")
                    except Exception as e:
                        print(f"   ❌ LIKE search failed: {e}")
        
        # Check books table for ML-related books
        if 'books' in tables:
            print(f"\n📚 Checking books table for ML content...")
            cursor.execute("""
                SELECT id, title, author, description, page_count
                FROM books
                WHERE LOWER(title) LIKE '%machine%' 
                   OR LOWER(title) LIKE '%learning%'
                   OR LOWER(title) LIKE '%ai%'
                   OR LOWER(title) LIKE '%algorithm%'
                   OR LOWER(description) LIKE '%machine learning%'
                   OR LOWER(description) LIKE '%neural%'
            """)
            
            ml_books = cursor.fetchall()
            if ml_books:
                print(f"   ✅ Found {len(ml_books)} ML-related books:")
                for book in ml_books:
                    print(f"     📖 '{book['title']}' by {book['author']} ({book['page_count']} pages)")
                    if book['description']:
                        desc = book['description'][:200] + "..." if len(book['description']) > 200 else book['description']
                        print(f"        {desc}")
            else:
                print("   ❌ No ML-related books found in metadata")
        
        conn.close()
        return results
        
    except Exception as e:
        print(f"❌ SQLite search error: {e}")
        return []

def search_qdrant_ml_content():
    """Search for ML content in Qdrant vector database"""
    print(f"\n🔍 Searching Qdrant vector database for ML content...")
    
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        print(f"📋 Available collections: {collection_names}")
        
        if QDRANT_COLLECTION not in collection_names:
            print(f"❌ Collection '{QDRANT_COLLECTION}' not found")
            return []
        
        # Get collection info
        collection_info = client.get_collection(QDRANT_COLLECTION)
        print(f"📊 Collection '{QDRANT_COLLECTION}' has {collection_info.points_count} points")
        
        if collection_info.points_count == 0:
            print("❌ No vectors in collection")
            return []
        
        # Try to get some sample points to see the structure
        print("\n🔍 Sampling some points to check content...")
        scroll_result = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=10,
            with_payload=True,
            with_vectors=False
        )
        
        points = scroll_result[0]
        ml_related = []
        
        for point in points:
            payload = point.payload
            
            # Look for ML-related content in the text
            text = payload.get('text', '').lower()
            
            ml_keywords = ['ml', 'machine learning', 'artificial intelligence', 'ai', 
                          'neural', 'algorithm', 'model', 'prediction', 'regression', 
                          'classification', 'feature', 'training', 'deep learning']
            
            for keyword in ml_keywords:
                if keyword in text:
                    ml_related.append({
                        'point_id': point.id,
                        'keyword': keyword,
                        'book_title': payload.get('book_title', 'Unknown'),
                        'book_id': payload.get('book_id', 'Unknown'),
                        'page': payload.get('page_start', 'Unknown'),
                        'preview': payload.get('text', '')[:200] + "..." if len(payload.get('text', '')) > 200 else payload.get('text', '')
                    })
                    break  # Only count once per point
        
        if ml_related:
            print(f"✅ Found {len(ml_related)} ML-related vector points:")
            for item in ml_related[:5]:  # Show top 5
                print(f"   📖 {item['book_title']} (Page {item['page']})")
                print(f"   🔑 Keyword: '{item['keyword']}'")
                print(f"   📝 {item['preview']}")
                print()
        else:
            print("❌ No ML-related content found in sample points")
        
        return ml_related
        
    except Exception as e:
        print(f"❌ Qdrant search error: {e}")
        return []

def main():
    print("=" * 80)
    print("TradeKnowledge ML Content Discovery")
    print("=" * 80)
    
    # Search SQLite
    sqlite_results = search_sqlite_ml_content()
    
    # Search Qdrant
    qdrant_results = search_qdrant_ml_content()
    
    # Summary
    print("\n" + "=" * 80)
    print("📈 SEARCH SUMMARY")
    print("=" * 80)
    
    print(f"SQLite results: {len(sqlite_results)}")
    print(f"Qdrant results: {len(qdrant_results)}")
    
    if sqlite_results or qdrant_results:
        print("\n✅ ML content found in your TradeKnowledge system!")
        
        if sqlite_results:
            print(f"\n📊 SQLite ML content by search term:")
            term_counts = {}
            for result in sqlite_results:
                term = result['term']
                term_counts[term] = term_counts.get(term, 0) + 1
            
            for term, count in sorted(term_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"   '{term}': {count} results")
        
        if qdrant_results:
            print(f"\n📊 Qdrant ML content by keyword:")
            keyword_counts = {}
            for result in qdrant_results:
                keyword = result['keyword']
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
            
            for keyword, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"   '{keyword}': {count} results")
    else:
        print("\n❌ No ML content found in your TradeKnowledge system")
        print("   Consider ingesting ML-related trading books or papers")

if __name__ == "__main__":
    main()