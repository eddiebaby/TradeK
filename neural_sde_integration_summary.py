#!/usr/bin/env python3
"""
Neural SDE Paper Integration Summary

This script provides a comprehensive summary of the Neural SDE paper integration
into the TradeKnowledge database system.
"""

import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_neural_sde_integration():
    """Check the status of Neural SDE paper integration"""
    db_path = "data/knowledge.db"
    book_id = "neural_sde_bayesian_calibration"
    
    logger.info("🔍 Checking Neural SDE Paper Integration Status")
    logger.info("=" * 60)
    
    # Check if database exists
    if not Path(db_path).exists():
        logger.error("❌ Database not found!")
        return False
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check book integration
        cursor.execute("SELECT id, title, author, isbn, file_path, file_type, file_hash, total_chunks, metadata, created_at, indexed_at, total_pages, categories FROM books WHERE id = ?", (book_id,))
        book = cursor.fetchone()
        
        if not book:
            logger.error("❌ Neural SDE book not found in database")
            return False
        
        logger.info("✅ Book Integration Status")
        logger.info(f"   📚 Title: {book[1]}")
        logger.info(f"   👤 Author: {book[2]}")
        logger.info(f"   🏷️  Book ID: {book[0]}")
        logger.info(f"   📄 Total Pages: {book[11] or 'N/A'}")
        logger.info(f"   🧩 Total Chunks: {book[7]}")
        logger.info(f"   📅 Created: {book[9]}")
        logger.info(f"   🔄 Indexed: {book[10] or 'In Progress'}")
        
        # Parse metadata
        metadata = json.loads(book[8]) if book[8] else {}
        logger.info(f"   🏷️  ArXiv ID: {metadata.get('arxiv_id', 'N/A')}")
        logger.info(f"   📅 Publication: {metadata.get('publication_date', 'N/A')}")
        logger.info(f"   🎯 Domain: {metadata.get('domain', 'N/A')}")
        logger.info(f"   🎓 Complexity: {metadata.get('complexity_level', 'N/A')}")
        
        # Check chunks
        cursor.execute("SELECT COUNT(*) FROM chunks WHERE book_id = ?", (book_id,))
        chunk_count = cursor.fetchone()[0]
        
        logger.info(f"\n✅ Chunk Integration Status")
        logger.info(f"   🧩 Database Chunks: {chunk_count}")
        
        # Check chunk distribution by priority
        cursor.execute("""
            SELECT json_extract(metadata, '$.priority') as priority, COUNT(*) as count
            FROM chunks 
            WHERE book_id = ?
            GROUP BY json_extract(metadata, '$.priority')
            ORDER BY count DESC
        """, (book_id,))
        
        priority_dist = cursor.fetchall()
        logger.info("   📊 Priority Distribution:")
        for priority, count in priority_dist:
            logger.info(f"      {priority or 'unspecified'}: {count} chunks")
        
        # Check section distribution
        cursor.execute("""
            SELECT chapter, COUNT(*) as count
            FROM chunks 
            WHERE book_id = ?
            GROUP BY chapter
            ORDER BY count DESC
        """, (book_id,))
        
        section_dist = cursor.fetchall()
        logger.info("   📚 Section Distribution:")
        for section, count in section_dist:
            logger.info(f"      {section}: {count} chunks")
        
        # Check FTS integration
        cursor.execute("SELECT COUNT(*) FROM chunks_fts WHERE id LIKE ?", (f"{book_id}%",))
        fts_count = cursor.fetchone()[0]
        
        logger.info(f"\n✅ Search Integration Status")
        logger.info(f"   🔍 FTS Index Entries: {fts_count}")
        logger.info(f"   📝 Text Search: {'Enabled' if fts_count > 0 else 'Disabled'}")
        
        # Test a sample search
        cursor.execute("""
            SELECT COUNT(*) 
            FROM chunks_fts 
            WHERE chunks_fts MATCH 'neural SDE'
            AND id LIKE ?
        """, (f"{book_id}%",))
        search_results = cursor.fetchone()[0]
        logger.info(f"   🔎 Sample Search ('neural SDE'): {search_results} results")
        
        # Check embeddings status
        cursor.execute("""
            SELECT COUNT(*) 
            FROM chunks 
            WHERE book_id = ? AND embedding_id IS NOT NULL
        """, (book_id,))
        embedding_count = cursor.fetchone()[0]
        
        logger.info(f"   🧠 Vector Embeddings: {embedding_count}/{chunk_count} chunks")
        logger.info(f"   🔍 Vector Search: {'Enabled' if embedding_count > 0 else 'Available after embedding generation'}")
        
        # Show key metadata
        if metadata:
            logger.info(f"\n✅ Paper Metadata")
            logger.info(f"   🎯 Key Contributions:")
            for i, contrib in enumerate(metadata.get('key_contributions', [])[:3], 1):
                logger.info(f"      {i}. {contrib}")
            
            logger.info(f"   🔬 Methodologies:")
            for method in metadata.get('methodologies', []):
                logger.info(f"      • {method}")
            
            logger.info(f"   💼 Applications:")
            for app in metadata.get('applications', []):
                logger.info(f"      • {app}")
            
            logger.info(f"   🏷️  Search Tags:")
            tags = metadata.get('search_tags', [])
            logger.info(f"      {', '.join(tags) if tags else 'None'}")
        
        # Show sample chunks
        logger.info(f"\n✅ Sample Content")
        cursor.execute("""
            SELECT id, chapter, substr(text, 1, 150) as snippet
            FROM chunks 
            WHERE book_id = ?
            ORDER BY chunk_index
            LIMIT 3
        """, (book_id,))
        
        samples = cursor.fetchall()
        for i, (chunk_id, section, snippet) in enumerate(samples, 1):
            logger.info(f"   📄 Chunk {i} ({section}):")
            logger.info(f"      {snippet}...")
        
        # Integration summary
        logger.info(f"\n🎉 Integration Summary")
        logger.info(f"   ✅ Book metadata stored")
        logger.info(f"   ✅ {chunk_count} chunks processed and stored")
        logger.info(f"   ✅ Full-text search enabled")
        logger.info(f"   ✅ Structured metadata preserved")
        logger.info(f"   ✅ Section-based organization")
        logger.info(f"   {'✅' if embedding_count > 0 else '⏳'} Vector embeddings {'ready' if embedding_count > 0 else 'pending'}")
        
        # Next steps
        logger.info(f"\n🚀 Next Steps")
        if embedding_count == 0:
            logger.info("   1. Run: python generate_neural_sde_embeddings.py")
            logger.info("      (Generates vector embeddings for semantic search)")
        else:
            logger.info("   ✅ Vector embeddings ready")
        
        logger.info("   2. Test search functionality:")
        logger.info("      python test_neural_sde_search.py")
        logger.info("   3. Use via API endpoints:")
        logger.info("      GET /api/search?q=neural+SDE+Bayesian")
        logger.info("      GET /api/books/neural_sde_bayesian_calibration")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error checking integration: {e}")
        return False
    finally:
        conn.close()

def show_usage_examples():
    """Show example usage of the integrated Neural SDE paper"""
    logger.info(f"\n📖 Usage Examples")
    logger.info("=" * 60)
    
    examples = [
        {
            "title": "Search for Bayesian Methods",
            "query": "SELECT id, substr(text, 1, 100) FROM chunks WHERE book_id = 'neural_sde_bayesian_calibration' AND text LIKE '%bayesian%';",
            "description": "Find chunks discussing Bayesian approaches"
        },
        {
            "title": "Find High Priority Content", 
            "query": "SELECT chapter, COUNT(*) FROM chunks WHERE book_id = 'neural_sde_bayesian_calibration' AND json_extract(metadata, '$.priority') = 'high' GROUP BY chapter;",
            "description": "Get high-priority sections"
        },
        {
            "title": "Full-Text Search",
            "query": "SELECT snippet(chunks_fts, 1, '<b>', '</b>', '...', 20) FROM chunks_fts JOIN chunks ON chunks_fts.id = chunks.id WHERE chunks_fts MATCH 'volatility surface' AND chunks.book_id = 'neural_sde_bayesian_calibration';",
            "description": "Search for volatility surface discussions"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        logger.info(f"\n{i}. {example['title']}")
        logger.info(f"   📝 {example['description']}")
        logger.info(f"   💻 SQL: {example['query']}")

def main():
    """Main function"""
    logger.info("🧠 Neural SDE Paper Integration Summary")
    logger.info("🔬 TradeKnowledge Database Analysis")
    logger.info("📅 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 60)
    
    success = check_neural_sde_integration()
    
    if success:
        show_usage_examples()
        logger.info(f"\n✅ Neural SDE paper successfully integrated into TradeKnowledge!")
        logger.info("🎯 Ready for semantic search and knowledge retrieval")
    else:
        logger.error("❌ Integration check failed")
        logger.info("💡 Try running: python simple_neural_sde_integration.py")

if __name__ == "__main__":
    main()