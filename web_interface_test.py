#!/usr/bin/env python3
"""
Web Interface Test - Verify numpy search is working
"""

import requests
import json

API_BASE = "http://localhost:8001/api"

def test_web_interface():
    print("🌐 Testing Trading Knowledge Web Interface")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. 🏥 Health Check")
    try:
        response = requests.get(f"{API_BASE}/health")
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print(f"❌ Server error: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return
    
    # Test 2: Empty concepts search (should return all)
    print("\n2. 📚 Load All Concepts")
    try:
        response = requests.get(f"{API_BASE}/concepts?query=")
        data = response.json()
        concepts = data['concepts']
        print(f"✅ Found {len(concepts)} total concepts")
        
        # Group by category
        categories = {}
        for concept in concepts:
            cat = concept['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        for cat, count in categories.items():
            print(f"   📂 {cat}: {count} concepts")
            
    except Exception as e:
        print(f"❌ Error loading concepts: {e}")
        return
    
    # Test 3: Search for numpy specifically
    print("\n3. 🔍 Search for 'numpy'")
    try:
        response = requests.get(f"{API_BASE}/concepts?query=numpy")
        data = response.json()
        numpy_concepts = data['concepts']
        
        if len(numpy_concepts) > 0:
            concept = numpy_concepts[0]
            print("✅ Found numpy!")
            print(f"   Name: {concept['name']}")
            print(f"   Category: {concept['category']}")
            print(f"   Difficulty: {concept['difficulty_level']}/5")
        else:
            print("❌ numpy not found in search results")
            
    except Exception as e:
        print(f"❌ Error searching for numpy: {e}")
    
    # Test 4: Search in specific category
    print("\n4. 🐍 Search Python Libraries")
    try:
        response = requests.get(f"{API_BASE}/concepts?category=python_libraries")
        data = response.json()
        py_concepts = data['concepts']
        
        print(f"✅ Found {len(py_concepts)} Python libraries:")
        for concept in py_concepts[:10]:  # Show first 10
            print(f"   • {concept['name']}")
            
    except Exception as e:
        print(f"❌ Error loading Python libraries: {e}")
    
    # Test 5: Chat query about numpy
    print("\n5. 💬 Ask about numpy")
    try:
        response = requests.post(f"{API_BASE}/query", json={
            "message": "What is numpy?",
            "user_id": "test"
        })
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Chat response received:")
            print(f"   Response: {data['content'][:100]}...")
            print(f"   Concepts: {data['concepts_covered']}")
            print(f"   Sources: {len(data['source_books'])} books")
        else:
            print(f"❌ Chat error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error with chat: {e}")
    
    print(f"\n🌐 Web Interface: http://localhost:8001")
    print(f"📄 Debug Page: file:///home/scottschweizer/TradeKnowledge/debug_search.html")
    print("\n📝 Instructions:")
    print("1. Open http://localhost:8001 in your browser")
    print("2. Click on 'Concepts' in the sidebar")
    print("3. Wait for concepts to load")
    print("4. Type 'numpy' in the search box")
    print("5. Or select 'Python Libraries' from the dropdown")

if __name__ == "__main__":
    test_web_interface()