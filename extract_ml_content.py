#!/usr/bin/env python3
"""
Extract and display ML-related content from TradeKnowledge.
This shows the actual ML content found in your system.
"""

import sqlite3
import re
from pathlib import Path
from collections import defaultdict

SQLITE_DB = "data/knowledge.db"

def extract_ml_content():
    """Extract detailed ML content from the database"""
    print("=" * 80)
    print("TradeKnowledge ML Content Extraction")
    print("=" * 80)
    
    if not Path(SQLITE_DB).exists():
        print(f"❌ Database not found at: {SQLITE_DB}")
        return
    
    try:
        conn = sqlite3.connect(SQLITE_DB)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get all chunks that contain ML-related terms
        print("🔍 Extracting ML-related content...")
        
        cursor.execute("""
            SELECT c.id, c.text, c.book_id, c.page_start, c.chunk_index,
                   b.title as book_title, b.author
            FROM chunks c
            LEFT JOIN books b ON c.book_id = b.id
            WHERE LOWER(c.text) LIKE '%machine learning%'
               OR LOWER(c.text) LIKE '%neural network%'
               OR LOWER(c.text) LIKE '%deep learning%'
               OR LOWER(c.text) LIKE '%artificial intelligence%'
               OR LOWER(c.text) LIKE '%prediction%'
               OR LOWER(c.text) LIKE '%classification%'
               OR LOWER(c.text) LIKE '%regression%'
               OR LOWER(c.text) LIKE '%algorithm%'
               OR LOWER(c.text) LIKE '%model%'
            ORDER BY c.book_id, c.chunk_index
        """)
        
        results = cursor.fetchall()
        
        if not results:
            print("❌ No ML content found")
            return
        
        print(f"✅ Found {len(results)} chunks with ML content")
        
        # Group by book
        books = defaultdict(list)
        for row in results:
            books[row['book_title']].append(row)
        
        # Display content by book
        for book_title, chunks in books.items():
            print(f"\n{'='*60}")
            print(f"📖 Book: {book_title}")
            print(f"📄 {len(chunks)} ML-related chunks found")
            print('='*60)
            
            # Find ML terms in each chunk
            ml_terms = [
                'machine learning', 'neural network', 'deep learning',
                'artificial intelligence', 'prediction', 'classification',
                'regression', 'algorithm', 'model', 'feature engineering',
                'training data', 'supervised learning', 'unsupervised learning'
            ]
            
            term_counts = defaultdict(int)
            chapter_analysis = defaultdict(list)
            
            for chunk in chunks[:10]:  # Show first 10 chunks
                text_lower = chunk['text'].lower()
                
                print(f"\n📄 Chunk {chunk['chunk_index']} (Page {chunk['page_start'] or 'N/A'})")
                
                # Find which ML terms are mentioned
                found_terms = []
                for term in ml_terms:
                    if term in text_lower:
                        found_terms.append(term)
                        term_counts[term] += 1
                
                if found_terms:
                    print(f"🔑 ML terms: {', '.join(found_terms)}")
                
                # Extract relevant sentences
                sentences = chunk['text'].split('.')
                ml_sentences = []
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if any(term in sentence.lower() for term in ml_terms):
                        if len(sentence) > 20:  # Skip very short sentences
                            ml_sentences.append(sentence)
                
                if ml_sentences:
                    print("📝 Relevant content:")
                    for sentence in ml_sentences[:3]:  # Show first 3 relevant sentences
                        # Clean up the sentence
                        clean_sentence = re.sub(r'\s+', ' ', sentence).strip()
                        if len(clean_sentence) > 100:
                            clean_sentence = clean_sentence[:200] + "..."
                        print(f"   • {clean_sentence}")
                
                # Look for code examples
                if 'def ' in chunk['text'] or 'import ' in chunk['text'] or 'class ' in chunk['text']:
                    print("💻 Contains code examples")
                
                print()
            
            # Show term frequency for this book
            if term_counts:
                print(f"\n📊 ML Term Frequency in '{book_title}':")
                sorted_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)
                for term, count in sorted_terms[:10]:
                    print(f"   {term}: {count} mentions")
        
        # Overall statistics
        print(f"\n{'='*80}")
        print("📈 OVERALL ML CONTENT ANALYSIS")
        print('='*80)
        
        # Count unique books with ML content
        unique_books = len(books)
        total_chunks = len(results)
        
        print(f"📚 Books with ML content: {unique_books}")
        print(f"📄 Total ML-related chunks: {total_chunks}")
        
        # Get specific ML topics mentioned
        all_text = ' '.join([row['text'].lower() for row in results])
        
        specific_topics = {
            'Support Vector Machines': 'svm' in all_text or 'support vector' in all_text,
            'Random Forest': 'random forest' in all_text,
            'Decision Trees': 'decision tree' in all_text,
            'Linear Regression': 'linear regression' in all_text,
            'Logistic Regression': 'logistic regression' in all_text,
            'K-Means': 'k-means' in all_text or 'kmeans' in all_text,
            'Neural Networks': 'neural network' in all_text,
            'Deep Learning': 'deep learning' in all_text,
            'Reinforcement Learning': 'reinforcement learning' in all_text,
            'Feature Engineering': 'feature engineering' in all_text or 'feature selection' in all_text,
            'Cross Validation': 'cross validation' in all_text or 'cross-validation' in all_text,
            'Overfitting': 'overfitting' in all_text or 'over-fitting' in all_text,
            'Scikit-learn': 'scikit-learn' in all_text or 'sklearn' in all_text,
            'TensorFlow': 'tensorflow' in all_text,
            'Keras': 'keras' in all_text,
            'PyTorch': 'pytorch' in all_text
        }
        
        found_topics = [topic for topic, found in specific_topics.items() if found]
        
        if found_topics:
            print(f"\n🎯 Specific ML topics found:")
            for topic in found_topics:
                print(f"   ✅ {topic}")
        
        # Look for Python ML libraries
        ml_libraries = {
            'pandas': 'pandas' in all_text,
            'numpy': 'numpy' in all_text,
            'scikit-learn': 'scikit-learn' in all_text or 'sklearn' in all_text,
            'matplotlib': 'matplotlib' in all_text,
            'seaborn': 'seaborn' in all_text,
            'tensorflow': 'tensorflow' in all_text,
            'keras': 'keras' in all_text,
            'pytorch': 'pytorch' in all_text
        }
        
        found_libraries = [lib for lib, found in ml_libraries.items() if found]
        
        if found_libraries:
            print(f"\n📦 ML Libraries mentioned:")
            for lib in found_libraries:
                print(f"   ✅ {lib}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    extract_ml_content()