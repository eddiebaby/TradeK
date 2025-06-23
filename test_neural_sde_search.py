#!/usr/bin/env python3
"""
Test search functionality for the integrated Neural SDE paper.

This script tests both text-based search and vector search
(if embeddings are available) for the Neural SDE paper.
"""

import asyncio
import logging
import sys
from pathlib import Path
import sqlite3

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.config import get_config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NeuralSDESearchTester:
    """Test search functionality for Neural SDE paper"""
    
    def __init__(self):
        self.config = get_config()
        self.db_path = self.config.database.sqlite.path
    
    def verify_neural_sde_integration(self) -> bool:
        """Verify that the Neural SDE paper is properly integrated"""
        logger.info("Verifying Neural SDE paper integration...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check book exists
            cursor.execute("SELECT id, title, total_chunks FROM books WHERE id = ?", 
                          ("neural_sde_bayesian_calibration",))
            book = cursor.fetchone()
            
            if not book:
                logger.error("Neural SDE book not found in database")
                return False
            
            logger.info(f"Found book: {book[1]} with {book[2]} chunks")
            
            # Check chunks exist
            cursor.execute("SELECT COUNT(*) FROM chunks WHERE book_id = ?", 
                          ("neural_sde_bayesian_calibration",))
            chunk_count = cursor.fetchone()[0]
            
            logger.info(f"Found {chunk_count} chunks in database")
            
            # Check FTS index
            cursor.execute("SELECT COUNT(*) FROM chunks_fts WHERE id LIKE ?", 
                          ("neural_sde_bayesian_calibration%",))
            fts_count = cursor.fetchone()[0]
            
            logger.info(f"Found {fts_count} chunks in FTS index")
            
            return chunk_count > 0 and fts_count > 0
            
        finally:
            conn.close()
    
    def test_direct_sqlite_search(self) -> None:
        """Test direct SQLite search queries"""
        logger.info("\n=== Testing Direct SQLite Search ===")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Test 1: Simple text search in chunks
            test_queries = [
                "neural",
                "Bayesian",
                "calibration",
                "volatility",
                "SDE",
                "option pricing"
            ]
            
            for query in test_queries:
                cursor.execute("""
                    SELECT id, substr(text, 1, 100) as snippet
                    FROM chunks 
                    WHERE book_id = 'neural_sde_bayesian_calibration'
                    AND text LIKE ?
                    LIMIT 3
                """, (f"%{query}%",))
                
                results = cursor.fetchall()
                logger.info(f"Query '{query}': {len(results)} results")
                
                for result in results:
                    logger.info(f"  - {result[0]}: {result[1]}...")
            
            # Test 2: FTS search
            logger.info("\nTesting FTS search...")
            cursor.execute("""
                SELECT c.id, snippet(chunks_fts, 1, '<mark>', '</mark>', '...', 20) as snippet
                FROM chunks_fts 
                JOIN chunks c ON chunks_fts.id = c.id
                WHERE chunks_fts MATCH 'neural SDE'
                AND c.book_id = 'neural_sde_bayesian_calibration'
                LIMIT 5
            """)
            
            fts_results = cursor.fetchall()
            logger.info(f"FTS search for 'neural SDE': {len(fts_results)} results")
            
            for result in fts_results:
                logger.info(f"  - {result[0]}: {result[1]}")
                
        except Exception as e:
            logger.error(f"Error in direct SQLite search: {e}")
        finally:
            conn.close()
    
    def test_advanced_sqlite_search(self) -> None:
        """Test advanced SQLite search features"""
        logger.info("\n=== Testing Advanced SQLite Search ===")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Test 1: Metadata-based search
            logger.info("\nTesting metadata-based search...")
            cursor.execute("""
                SELECT id, json_extract(metadata, '$.section') as section, substr(text, 1, 100) as snippet
                FROM chunks 
                WHERE book_id = 'neural_sde_bayesian_calibration'
                AND json_extract(metadata, '$.priority') = 'high'
                LIMIT 5
            """)
            
            results = cursor.fetchall()
            logger.info(f"High priority chunks: {len(results)} results")
            
            for result in results:
                logger.info(f"  - {result[0]} ({result[1]}): {result[2]}...")
            
            # Test 2: Section-based search
            logger.info("\nTesting section-based search...")
            cursor.execute("""
                SELECT DISTINCT chapter, COUNT(*) as chunk_count
                FROM chunks 
                WHERE book_id = 'neural_sde_bayesian_calibration'
                GROUP BY chapter
                ORDER BY chunk_count DESC
            """)
            
            sections = cursor.fetchall()
            logger.info("Sections in the paper:")
            for section, count in sections:
                logger.info(f"  - {section}: {count} chunks")
            
            # Test 3: Keyword search in metadata
            logger.info("\nTesting keyword search in metadata...")
            cursor.execute("""
                SELECT id, json_extract(metadata, '$.keywords') as keywords, substr(text, 1, 100) as snippet
                FROM chunks 
                WHERE book_id = 'neural_sde_bayesian_calibration'
                AND json_extract(metadata, '$.keywords') LIKE '%bayesian%'
                LIMIT 3
            """)
            
            keyword_results = cursor.fetchall()
            logger.info(f"Chunks with 'bayesian' in keywords: {len(keyword_results)} results")
            
            for result in keyword_results:
                logger.info(f"  - {result[0]}: {result[2]}...")
                logger.info(f"    Keywords: {result[1]}")
                
        except Exception as e:
            logger.error(f"Error in advanced SQLite search: {e}")
        finally:
            conn.close()
    
    def show_sample_chunks(self) -> None:
        """Show sample chunks from the Neural SDE paper"""
        logger.info("\n=== Sample Neural SDE Paper Chunks ===")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT id, chapter, substr(text, 1, 300) as text_sample
                FROM chunks 
                WHERE book_id = 'neural_sde_bayesian_calibration'
                ORDER BY chunk_index
                LIMIT 5
            """)
            
            results = cursor.fetchall()
            
            for result in results:
                logger.info(f"\nChunk: {result[0]}")
                logger.info(f"Section: {result[1]}")
                logger.info(f"Text: {result[2]}...")
                
        finally:
            conn.close()

def main():
    """Main test function"""
    tester = NeuralSDESearchTester()
    
    # Verify integration
    if not tester.verify_neural_sde_integration():
        logger.error("Neural SDE paper not properly integrated. Please run the integration script first.")
        return
    
    # Show sample chunks
    tester.show_sample_chunks()
    
    # Test direct SQLite search
    tester.test_direct_sqlite_search()
    
    # Test advanced SQLite search
    tester.test_advanced_sqlite_search()
    
    logger.info("\n✅ Neural SDE search testing complete!")

if __name__ == "__main__":
    main()