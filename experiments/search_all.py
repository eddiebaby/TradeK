#!/usr/bin/env python3
"""
Search ALL occurrences of a term in the book
"""

import sys
import asyncio
from sentence_transformers import SentenceTransformer

# Add src to path
sys.path.append('src')

from src.core.qdrant_storage import QdrantStorage

async def search_all_occurrences(query, min_score=0.1):
    """Search for ALL occurrences of a term"""
    print(f"🔍 Searching for ALL occurrences of: \"{query}\"")
    print("="*60)
    
    storage = QdrantStorage()
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    
    # Generate embedding
    query_embedding = model.encode([query], convert_to_numpy=True)[0].tolist()
    
    # Search with high limit to get all results
    results = await storage.search_semantic(query_embedding, limit=372)  # All pages
    
    # Filter by minimum score and count occurrences
    relevant_results = [r for r in results if r['score'] >= min_score]
    
    print(f"📊 Found {len(relevant_results)} pages with relevance >= {min_score}")
    print()
    
    # Group by relevance levels
    high_relevance = [r for r in relevant_results if r['score'] >= 0.3]
    medium_relevance = [r for r in relevant_results if 0.2 <= r['score'] < 0.3]
    low_relevance = [r for r in relevant_results if 0.1 <= r['score'] < 0.2]
    
    print(f"🎯 High relevance (≥30%): {len(high_relevance)} pages")
    print(f"🎯 Medium relevance (20-30%): {len(medium_relevance)} pages")  
    print(f"🎯 Low relevance (10-20%): {len(low_relevance)} pages")
    print()
    
    # Show top results
    print("📄 TOP 10 MOST RELEVANT PAGES:")
    print("-" * 40)
    
    for i, result in enumerate(relevant_results[:10], 1):
        score = result['score']
        chunk_id = result['chunk_id']
        text = result['text']
        
        # Extract page number
        page_num = "Unknown"
        if "page_" in chunk_id:
            try:
                page_num = chunk_id.split("page_")[1]
                page_num = f"Page {int(page_num)}"
            except:
                pass
        
        # Show first 150 characters
        preview = text[:150]
        if len(text) > 150:
            last_space = preview.rfind(' ')
            if last_space > 100:
                preview = preview[:last_space] + "..."
            else:
                preview += "..."
        
        print(f"{i:2d}. {page_num} ({score:.1%}) - {preview}")
    
    # Show all page numbers
    if len(relevant_results) > 10:
        print(f"\n📋 ALL {len(relevant_results)} RELEVANT PAGES:")
        page_numbers = []
        for result in relevant_results:
            chunk_id = result['chunk_id']
            if "page_" in chunk_id:
                try:
                    page_num = int(chunk_id.split("page_")[1])
                    page_numbers.append(page_num)
                except:
                    pass
        
        page_numbers.sort()
        
        # Group pages for better readability
        page_ranges = []
        if page_numbers:
            start = page_numbers[0]
            end = start
            
            for page in page_numbers[1:]:
                if page == end + 1:
                    end = page
                else:
                    if start == end:
                        page_ranges.append(str(start))
                    else:
                        page_ranges.append(f"{start}-{end}")
                    start = end = page
            
            # Add the last range
            if start == end:
                page_ranges.append(str(start))
            else:
                page_ranges.append(f"{start}-{end}")
        
        print(", ".join(page_ranges))

async def main():
    if len(sys.argv) < 2:
        print("Usage: python search_all.py \"search term\"")
        print("Example: python search_all.py \"machine learning\"")
        return
    
    query = " ".join(sys.argv[1:])
    await search_all_occurrences(query)

if __name__ == "__main__":
    asyncio.run(main())