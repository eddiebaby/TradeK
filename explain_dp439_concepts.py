#!/usr/bin/env python3
"""
Deep semantic analysis of dp439 paper concepts
"""

import sys
sys.path.append('.')

import chromadb

def semantic_concept_analysis():
    """Use semantic search to understand the paper's core concepts"""
    
    print("🧠 DEEP ANALYSIS OF DP439 CONCEPTS")
    print("=" * 50)
    
    try:
        client = chromadb.PersistentClient(path="./data/qdrant")
        collection = client.get_collection("books")
        book_id = "dp439_e1b9a54f"
        
        # Core concept queries for deep understanding
        concept_queries = [
            ("Abstract & Problem", "abstract square root time rule problem financial applications"),
            ("Square Root Rule Explained", "what is square root of time rule how does it work"),
            ("Why It's Wrong", "why square root rule incorrect problems assumptions"),
            ("Risk Underestimation", "underestimation risk degree worsens longer horizons"),
            ("IID Normal Assumption", "iid normal returns assumption derivatives pricing"),
            ("Time Scaling Methods", "time scaling risk alternative methods better approaches"),
            ("Practical Implications", "regulatory recommendations derivatives pricing implications"),
            ("Main Findings", "main results conclusions findings paper"),
            ("Mathematical Framework", "mathematical model equations formulas scaling"),
            ("Empirical Evidence", "empirical results data evidence testing")
        ]
        
        for concept, query in concept_queries:
            print(f"\n🎯 {concept.upper()}")
            print("=" * 60)
            
            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=3,
                    where={"book_id": book_id}
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
                        
                        print(f"📖 Match {i+1} (Score: {score:.3f}, Chunk: {chunk_idx}):")
                        
                        # Clean up text for better readability
                        clean_text = doc.replace('\n\n', ' ').replace('  ', ' ').strip()
                        print(f"   {clean_text[:400]}...")
                        print()
                
                else:
                    print("❌ No matches found")
                    
            except Exception as e:
                print(f"❌ Search error: {e}")
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")

def get_abstract_explanation():
    """Get a clear explanation of the abstract"""
    
    print("\n📋 ABSTRACT BREAKDOWN")
    print("=" * 30)
    
    try:
        client = chromadb.PersistentClient(path="./data/qdrant")
        collection = client.get_collection("books")
        book_id = "dp439_e1b9a54f"
        
        # Search for the abstract specifically
        results = collection.query(
            query_texts=["abstract many financial applications risk analysis derivatives"],
            n_results=2,
            where={"book_id": book_id}
        )
        
        if results and results.get('documents') and results['documents'][0]:
            abstract_text = results['documents'][0][0]
            
            print("🎯 FOUND ABSTRACT:")
            print("-" * 40)
            clean_abstract = abstract_text.replace('\n\n', ' ').replace('  ', ' ').strip()
            print(clean_abstract)
            print()
            
            # Break down the abstract sentence by sentence
            sentences = clean_abstract.split('. ')
            
            print("📝 SENTENCE-BY-SENTENCE BREAKDOWN:")
            print("-" * 40)
            
            for i, sentence in enumerate(sentences, 1):
                if sentence.strip():
                    print(f"{i}. {sentence.strip()}.")
                    print()
        
    except Exception as e:
        print(f"❌ Abstract analysis error: {e}")

def find_simple_explanation():
    """Find simpler explanations in the paper"""
    
    print("\n💡 SIMPLE EXPLANATIONS FROM THE PAPER")
    print("=" * 40)
    
    try:
        client = chromadb.PersistentClient(path="./data/qdrant")
        collection = client.get_collection("books")
        book_id = "dp439_e1b9a54f"
        
        # Search for introduction and conclusion sections
        simple_queries = [
            "introduction financial markets risk measurement",
            "conclusion main findings results",
            "example simple illustration demonstration",
            "intuition behind square root rule why used"
        ]
        
        for query in simple_queries:
            results = collection.query(
                query_texts=[query],
                n_results=1,
                where={"book_id": book_id}
            )
            
            if results and results.get('documents') and results['documents'][0]:
                doc = results['documents'][0][0]
                
                print(f"🔍 Query: '{query}'")
                clean_text = doc.replace('\n\n', ' ').replace('  ', ' ').strip()
                print(f"   {clean_text[:350]}...")
                print()
        
    except Exception as e:
        print(f"❌ Simple explanation error: {e}")

def main():
    """Main analysis function"""
    
    print("🚀 STARTING DEEP CONCEPT ANALYSIS...")
    
    # Get abstract breakdown first
    get_abstract_explanation()
    
    # Find simpler explanations
    find_simple_explanation()
    
    # Do detailed concept analysis
    semantic_concept_analysis()
    
    print("\n✅ Deep analysis completed!")
    print("This explanation was generated using semantic search on the actual paper content.")

if __name__ == "__main__":
    main()