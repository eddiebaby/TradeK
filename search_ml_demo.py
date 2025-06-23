#!/usr/bin/env python3
"""
Demo script to search for "ML" content in TradeKnowledge system.
This demonstrates the hybrid search functionality that combines semantic and exact search.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.search.hybrid_search import HybridSearch
from src.core.config import get_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def search_ml_content():
    """Search for ML-related content in the TradeKnowledge system"""
    
    print("=" * 80)
    print("TradeKnowledge ML Content Search Demo")
    print("=" * 80)
    
    try:
        # Initialize search engine
        print("\n🔧 Initializing search engine...")
        config = get_config()
        search_engine = HybridSearch(config)
        await search_engine.initialize()
        
        print("✅ Search engine initialized successfully!")
        
        # Test different ML-related queries
        ml_queries = [
            "ML",                                    # Exact term
            "machine learning",                      # Full term
            "artificial intelligence",               # Related concept
            "neural networks",                       # Technical term
            "deep learning",                        # Advanced ML
            "algorithms",                           # General algorithms
            "predictive models",                    # ML application
            "regression analysis",                  # Statistical ML
            "classification",                       # ML task type
            "feature engineering"                   # ML preprocessing
        ]
        
        all_results = {}
        
        for query in ml_queries:
            print(f"\n{'='*60}")
            print(f"🔍 Searching for: '{query}'")
            print('='*60)
            
            # Perform hybrid search (combines semantic + exact)
            results = await search_engine.search_hybrid(
                query=query,
                num_results=5,  # Get top 5 results
                semantic_weight=0.7  # 70% semantic, 30% exact
            )
            
            all_results[query] = results
            
            print(f"📊 Found {results['total_results']} total results")
            print(f"⏱️  Search completed in {results['search_time_ms']}ms")
            print(f"🔧 Search type: {results['search_type']}")
            
            if results['results']:
                print("\n📖 Top results:")
                for i, result in enumerate(results['results'][:3], 1):
                    chunk = result['chunk']
                    print(f"\n{i}. Score: {result['score']:.3f} | Type: {result['match_type']}")
                    print(f"   Book: {result['book_title']}")
                    
                    if result.get('book_author'):
                        print(f"   Author: {result['book_author']}")
                    
                    if chunk.get('page_start'):
                        print(f"   Page: {chunk['page_start']}")
                    
                    # Show highlight/snippet
                    if result.get('highlights') and result['highlights']:
                        snippet = result['highlights'][0]
                        # Truncate if too long
                        if len(snippet) > 200:
                            snippet = snippet[:200] + "..."
                        print(f"   Preview: {snippet}")
                    else:
                        # Fallback to chunk text
                        text = chunk.get('text', '')
                        if len(text) > 200:
                            text = text[:200] + "..."
                        print(f"   Preview: {text}")
            else:
                print("   ❌ No results found")
        
        # Summary analysis
        print(f"\n{'='*80}")
        print("📈 SEARCH SUMMARY")
        print('='*80)
        
        total_unique_results = set()
        best_matches = []
        
        for query, results in all_results.items():
            if results['results']:
                result_count = len(results['results'])
                best_score = max(r['score'] for r in results['results'])
                best_matches.append((query, result_count, best_score))
                
                # Track unique chunk IDs
                for result in results['results']:
                    total_unique_results.add(result['chunk']['id'])
        
        print(f"🎯 Total unique content pieces found: {len(total_unique_results)}")
        print(f"📚 Queries with results: {len([r for r in all_results.values() if r['results']])}/{len(ml_queries)}")
        
        if best_matches:
            print("\n🏆 Best matches by query:")
            best_matches.sort(key=lambda x: x[2], reverse=True)  # Sort by score
            for query, count, score in best_matches[:5]:
                print(f"   '{query}': {count} results, best score: {score:.3f}")
        
        # Show search engine stats
        stats = search_engine.get_stats()
        print(f"\n⚡ Search engine performance:")
        print(f"   Total searches: {stats['total_searches']}")
        print(f"   Average search time: {stats['average_search_time_ms']:.1f}ms")
        print(f"   Components initialized: {all(stats['components_initialized'].values())}")
        
        # Cleanup
        await search_engine.cleanup()
        print("\n✅ Search completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Search failed: {e}")
        print(f"\n❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(search_ml_content())
    sys.exit(0 if success else 1)