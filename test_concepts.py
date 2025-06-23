#!/usr/bin/env python3
"""
Test the Trading Knowledge Teacher with the processed book
"""

import requests
import json

API_BASE = "http://localhost:8001/api"

def test_concept(concept_name):
    """Test a specific concept"""
    print(f"\n🧠 Testing: '{concept_name}'")
    print("=" * 50)
    
    response = requests.post(f"{API_BASE}/query", json={
        "message": f"What is {concept_name}?",
        "user_id": "default"
    })
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Found concept!")
        print(f"📝 Response: {data['content'][:200]}...")
        print(f"📚 Source: {data['source_books']}")
        print(f"🔗 Related: {', '.join(data['related_concepts'][:5])}")
    else:
        print(f"❌ Error: {response.status_code}")

def list_books():
    """List all processed books"""
    print("\n📚 Processed Books:")
    print("=" * 50)
    
    response = requests.get(f"{API_BASE}/books")
    if response.status_code == 200:
        books = response.json()['books']
        for book in books:
            print(f"📖 {book['title']} by {book['author']}")
            print(f"   Status: {book['processing_status']}")
            print(f"   Chapters: {book['chapter_count']}, Words: {book['word_count']}")
    else:
        print(f"❌ Error: {response.status_code}")

def list_concepts_by_category():
    """List concepts by category"""
    print("\n🧠 Available Concepts:")
    print("=" * 50)
    
    response = requests.get(f"{API_BASE}/concepts?query=")
    if response.status_code == 200:
        concepts = response.json()['concepts']
        
        # Group by category
        categories = {}
        for concept in concepts:
            cat = concept['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(concept['name'])
        
        for category, concept_list in categories.items():
            print(f"\n📂 {category.replace('_', ' ').title()} ({len(concept_list)} concepts):")
            for concept in sorted(concept_list)[:10]:  # Show first 10
                print(f"   • {concept}")
            if len(concept_list) > 10:
                print(f"   ... and {len(concept_list) - 10} more")
    else:
        print(f"❌ Error: {response.status_code}")

def main():
    print("🎉 Trading Knowledge Teacher - Concept Test")
    print("=" * 60)
    
    # List books
    list_books()
    
    # List concepts
    list_concepts_by_category()
    
    # Test specific concepts
    test_concepts = [
        "numpy",
        "pandas", 
        "machine learning",
        "algorithmic trading",
        "time series",
        "Monte Carlo"
    ]
    
    for concept in test_concepts:
        test_concept(concept)
    
    print(f"\n🌐 Web Interface: http://localhost:8001")
    print(f"📄 Test Page: file:///home/scottschweizer/TradeKnowledge/test_web_interface.html")

if __name__ == "__main__":
    main()