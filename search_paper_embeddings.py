#!/usr/bin/env python3
"""
Search and summarize the paper using only embeddings
"""

import sys
sys.path.append('.')

try:
    from src.search.unified_search import UnifiedSearchEngine
    from src.core.config import get_config
except ImportError:
    print("❌ Search engine not available")
    sys.exit(1)

def search_and_summarize():
    """Search the paper using embeddings and create summary"""
    
    print("🔍 SEARCHING PAPER USING EMBEDDINGS")
    print("=" * 50)
    
    # Initialize search engine
    config = get_config()
    search_engine = UnifiedSearchEngine(config)
    
    # Search queries to understand the paper
    queries = [
        "what is the main contribution of this paper",
        "vec2vec embedding translation method",
        "universal geometry embeddings hypothesis", 
        "embedding space translation without paired data",
        "results experiments vec2vec performance",
        "applications information extraction embeddings"
    ]
    
    paper_summary = {}
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 40)
        
        try:
            results = search_engine.search(
                query=query,
                top_k=3,
                search_type="semantic"
            )
            
            if results and 'results' in results:
                relevant_chunks = []
                for result in results['results'][:2]:  # Top 2 results
                    if result.get('metadata', {}).get('book_id') == 'book_ca69c36d':
                        chunk_text = result.get('text', '')
                        if chunk_text:
                            relevant_chunks.append(chunk_text[:300] + "...")
                
                if relevant_chunks:
                    paper_summary[query] = relevant_chunks
                    print(f"✅ Found {len(relevant_chunks)} relevant chunks")
                    for i, chunk in enumerate(relevant_chunks):
                        print(f"   {i+1}. {chunk}")
                else:
                    print("❌ No relevant chunks found for this query")
            else:
                print("❌ No search results")
                
        except Exception as e:
            print(f"❌ Search error: {e}")
    
    # Generate final summary
    print("\n" + "="*60)
    print("📋 PAPER SUMMARY FROM EMBEDDINGS")
    print("="*60)
    
    if paper_summary:
        print("\n🎯 **Main Contribution:**")
        if "what is the main contribution of this paper" in paper_summary:
            for chunk in paper_summary["what is the main contribution of this paper"]:
                print(f"   • {chunk}")
        
        print("\n🔧 **Method (vec2vec):**")
        if "vec2vec embedding translation method" in paper_summary:
            for chunk in paper_summary["vec2vec embedding translation method"]:
                print(f"   • {chunk}")
        
        print("\n🧠 **Core Hypothesis:**")
        if "universal geometry embeddings hypothesis" in paper_summary:
            for chunk in paper_summary["universal geometry embeddings hypothesis"]:
                print(f"   • {chunk}")
        
        print("\n📊 **Results:**")
        if "results experiments vec2vec performance" in paper_summary:
            for chunk in paper_summary["results experiments vec2vec performance"]:
                print(f"   • {chunk}")
        
        print("\n💡 **Applications:**")
        if "applications information extraction embeddings" in paper_summary:
            for chunk in paper_summary["applications information extraction embeddings"]:
                print(f"   • {chunk}")
    else:
        print("❌ No summary could be generated from embeddings")

def identify_model():
    """Identify what embedding model this paper is about"""
    
    print("\n" + "="*50)
    print("🤖 IDENTIFYING THE MODEL")
    print("="*50)
    
    config = get_config()
    search_engine = UnifiedSearchEngine(config)
    
    model_queries = [
        "embedding model architecture",
        "what model does vec2vec use",
        "T5 BERT embedding models",
        "GTR GTE embedding models"
    ]
    
    for query in model_queries:
        print(f"\n🔍 Query: '{query}'")
        
        try:
            results = search_engine.search(
                query=query,
                top_k=2,
                search_type="semantic"
            )
            
            if results and 'results' in results:
                for result in results['results']:
                    if result.get('metadata', {}).get('book_id') == 'book_ca69c36d':
                        text = result.get('text', '')
                        if any(model in text.lower() for model in ['t5', 'bert', 'gtr', 'gte', 'model']):
                            print(f"   📝 {text[:400]}...")
                            break
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    search_and_summarize()
    identify_model()