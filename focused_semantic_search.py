#!/usr/bin/env python3
"""
Focused semantic search queries to understand the paper better
"""

import sys
sys.path.append('.')

import chromadb
import time

def semantic_paper_analysis():
    """Use semantic search to analyze the paper comprehensively"""
    
    print("🧠 SEMANTIC ANALYSIS OF VEC2VEC PAPER")
    print("=" * 50)
    
    try:
        client = chromadb.PersistentClient(path="./data/qdrant")
        collection = client.get_collection("books")
        target_book_id = 'book_ca69c36d'
        
        # Focused queries for better understanding
        focused_queries = [
            ("Core Method", "How does vec2vec translate embeddings without paired training data?"),
            ("Main Innovation", "What is the key breakthrough or novel contribution?"),
            ("Architecture", "What is the neural network architecture design?"),
            ("Training Process", "How is the model trained using adversarial learning?"),
            ("Evaluation Results", "What performance metrics and experimental results?"),
            ("Security Impact", "How can this attack vector databases and extract private information?"),
            ("Embedding Models", "Which specific embedding models GTR GTE T5 BERT were evaluated?"),
            ("Dataset Experiments", "What datasets were used for training and evaluation?")
        ]
        
        for category, query in focused_queries:
            print(f"\n🎯 {category.upper()}")
            print(f"Query: '{query}'")
            print("-" * 60)
            
            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=2,  # Top 2 most relevant
                    where={"book_id": target_book_id}
                )
                
                if results and results.get('documents') and results['documents'][0]:
                    docs = results['documents'][0]
                    distances = results.get('distances', [None])[0] or []
                    metadatas = results.get('metadatas', [None])[0] or []
                    
                    for i, doc in enumerate(docs):
                        distance = distances[i] if i < len(distances) else None
                        metadata = metadatas[i] if i < len(metadatas) else {}
                        
                        score = 1 - distance if distance is not None else 0
                        chunk_idx = metadata.get('chunk_index', 'unknown')
                        
                        print(f"📖 Match {i+1} (Similarity: {score:.3f}, Chunk: {chunk_idx}):")
                        
                        # Show more text for better understanding
                        preview = doc[:350].replace('\n', ' ')
                        print(f"   {preview}...")
                        print()
                
                else:
                    print("❌ No semantic matches found")
                    
            except Exception as e:
                print(f"❌ Search error: {e}")
        
    except Exception as e:
        print(f"❌ ChromaDB error: {e}")

def summarize_from_embeddings():
    """Generate paper summary based on semantic search results"""
    
    print("\n📋 PAPER SUMMARY FROM SEMANTIC SEARCH")
    print("=" * 50)
    
    try:
        client = chromadb.PersistentClient(path="./data/qdrant")
        collection = client.get_collection("books")
        target_book_id = 'book_ca69c36d'
        
        # Key summary queries
        summary_queries = [
            "main contribution novel method paper",
            "vec2vec unsupervised embedding translation",
            "security implications vector database attack",
            "experimental results performance metrics"
        ]
        
        summary_points = []
        
        for query in summary_queries:
            results = collection.query(
                query_texts=[query],
                n_results=1,
                where={"book_id": target_book_id}
            )
            
            if results and results.get('documents') and results['documents'][0]:
                doc = results['documents'][0][0]
                # Extract key sentence
                sentences = doc.split('.')
                if sentences:
                    key_sentence = sentences[0].strip()
                    if len(key_sentence) > 50:  # Only meaningful sentences
                        summary_points.append(key_sentence)
        
        print("🎯 KEY FINDINGS FROM SEMANTIC SEARCH:")
        for i, point in enumerate(summary_points, 1):
            print(f"{i}. {point}...")
        
        # Get abstract/introduction for context
        print("\n📖 PAPER CONTEXT (from semantic search):")
        context_result = collection.query(
            query_texts=["abstract introduction paper contribution"],
            n_results=1,
            where={"book_id": target_book_id}
        )
        
        if context_result and context_result.get('documents') and context_result['documents'][0]:
            abstract = context_result['documents'][0][0]
            print(f"   {abstract[:400]}...")
        
    except Exception as e:
        print(f"❌ Summary generation error: {e}")

def main():
    """Main function"""
    
    print("🚀 STARTING SEMANTIC ANALYSIS...")
    
    # Run focused semantic analysis
    semantic_paper_analysis()
    
    # Generate summary from semantic search
    summarize_from_embeddings()
    
    print("\n✅ Semantic analysis completed!")
    print("This summary was generated using actual embedding-based semantic search,")
    print("not SQL text matching!")

if __name__ == "__main__":
    main()