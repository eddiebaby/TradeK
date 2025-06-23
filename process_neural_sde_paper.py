#!/usr/bin/env python3
"""
Process Neural SDE Paper for TradeKnowledge
Creates properly formatted chunks and metadata for the knowledge system
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

def create_paper_summary() -> Dict[str, Any]:
    """Create a comprehensive summary of the neural SDE paper"""
    
    return {
        "id": "neural_sde_bayesian_calibration",
        "title": "Robust financial calibration: a Bayesian approach for neural SDEs",
        "authors": ["Christa Cuchiero", "Eva Flonner", "Kevin Kurt"],
        "institutions": [
            "University of Vienna",
            "Vienna University of Economics and Business", 
            "IQAM Invest"
        ],
        "publication_date": "2024-09-13",
        "arxiv_id": "2409.06551v3",
        "subject": "Quantitative Finance",
        "keywords": [
            "neural SDE", "Bayesian calibration", "financial modeling",
            "volatility surface", "option pricing", "uncertainty quantification",
            "stochastic differential equations", "neural networks",
            "robust bounds", "posterior distribution", "mixture models"
        ],
        "domain": "quantitative_finance",
        "subdomain": "neural_sde_calibration",
        "complexity_level": "advanced",
        "target_audience": ["quantitative_analysts", "researchers", "risk_managers"],
        "abstract": """The paper presents a Bayesian framework for the calibration of financial models using neural stochastic differential equations (neural SDEs). The method is based on the specification of a prior distribution on the neural network weights and an adequately chosen likelihood function. The resulting posterior distribution can be seen as a mixture of different classical neural SDE models yielding robust bounds on the implied volatility surface. Both, historical financial time series data and option prices can be integrated into the framework in a natural way. This leads to the derivation of robust bounds also for risk measures computed under the real-world measure such as value-at-risk and expected shortfall. We demonstrate the efficiency of the approach in several numerical experiments where we observe stability across different market conditions. The implementation can also be efficiently parallelized.""",
        
        "key_contributions": [
            "Bayesian framework for neural SDE calibration",
            "Robust bounds on implied volatility surface",
            "Integration of historical data and option prices",
            "Uncertainty quantification for financial models",
            "Parallelizable implementation approach"
        ],
        
        "methodologies": [
            "Bayesian inference",
            "Neural stochastic differential equations",
            "Mixture model approach", 
            "Variational inference",
            "Monte Carlo methods"
        ],
        
        "applications": [
            "Option pricing",
            "Volatility surface modeling",
            "Risk measure computation",
            "Value-at-Risk estimation",
            "Expected shortfall calculation"
        ],
        
        "technical_details": {
            "model_type": "Neural SDE",
            "inference_method": "Bayesian",
            "likelihood_function": "Custom designed",
            "prior_specification": "Neural network weights",
            "posterior_interpretation": "Mixture of classical neural SDE models",
            "computational_approach": "Parallelizable"
        },
        
        "relevance_to_trading": {
            "high": [
                "Volatility surface modeling for options trading",
                "Risk management and VaR calculations",
                "Robust model calibration under uncertainty"
            ],
            "medium": [
                "Portfolio optimization with neural models",
                "Stress testing financial models",
                "Model validation and backtesting"
            ],
            "low": [
                "High-frequency trading strategies",
                "Market microstructure analysis"
            ]
        },
        
        "implementation_complexity": {
            "mathematical": "High - requires understanding of SDEs and Bayesian inference",
            "computational": "Medium-High - neural networks with Bayesian inference",
            "practical": "Medium - can leverage existing ML frameworks"
        }
    }

def create_structured_chunks() -> List[Dict[str, Any]]:
    """Create structured chunks for different sections of the paper"""
    
    # Load extracted text
    try:
        with open('full_paper_text.txt', 'r', encoding='utf-8') as f:
            full_text = f.read()
    except FileNotFoundError:
        print("❌ Full text file not found. Run the PDF extraction first.")
        return []
    
    try:
        with open('structured_paper_analysis.json', 'r') as f:
            analysis = json.load(f)
    except FileNotFoundError:
        print("❌ Analysis file not found. Run the PDF extraction first.")
        return []
    
    chunks = []
    
    # Abstract chunk (highest priority)
    if 'abstract' in analysis.get('sections', {}):
        chunks.append({
            "chunk_id": "neural_sde_abstract",
            "section": "abstract", 
            "priority": "critical",
            "content": analysis['sections']['abstract'],
            "metadata": {
                "section_type": "abstract",
                "importance": "critical",
                "keywords": ["neural SDE", "Bayesian calibration", "volatility surface"],
                "concepts": ["uncertainty quantification", "robust bounds", "mixture models"],
                "audience": "all_levels"
            }
        })
    
    # Introduction chunk
    if 'introduction' in analysis.get('sections', {}):
        chunks.append({
            "chunk_id": "neural_sde_introduction",
            "section": "introduction",
            "priority": "high", 
            "content": analysis['sections']['introduction'],
            "metadata": {
                "section_type": "introduction",
                "importance": "high",
                "keywords": ["financial modeling", "neural networks", "stochastic processes"],
                "concepts": ["motivation", "problem statement", "related work"],
                "audience": "intermediate_advanced"
            }
        })
    
    # Conclusion chunk
    if 'conclusion' in analysis.get('sections', {}):
        chunks.append({
            "chunk_id": "neural_sde_conclusion",
            "section": "conclusion",
            "priority": "high",
            "content": analysis['sections']['conclusion'], 
            "metadata": {
                "section_type": "conclusion",
                "importance": "high",
                "keywords": ["results", "implications", "future work"],
                "concepts": ["summary", "limitations", "extensions"],
                "audience": "all_levels"
            }
        })
    
    # Split remaining content into topical chunks
    remaining_text = full_text
    for section in analysis.get('sections', {}).values():
        remaining_text = remaining_text.replace(section, "")
    
    # Create chunks for different topics based on content analysis
    content_chunks = [
        {
            "topic": "bayesian_methodology",
            "keywords": ["bayesian", "prior", "posterior", "inference", "distribution"],
            "priority": "high"
        },
        {
            "topic": "neural_sde_framework", 
            "keywords": ["neural", "sde", "stochastic", "differential", "equation"],
            "priority": "high"
        },
        {
            "topic": "volatility_modeling",
            "keywords": ["volatility", "surface", "implied", "option", "pricing"],
            "priority": "high"
        },
        {
            "topic": "risk_measures",
            "keywords": ["risk", "var", "value-at-risk", "expected", "shortfall"],
            "priority": "medium"
        },
        {
            "topic": "numerical_experiments",
            "keywords": ["experiment", "numerical", "result", "performance", "stability"],
            "priority": "medium"
        },
        {
            "topic": "implementation",
            "keywords": ["implementation", "computational", "algorithm", "parallel"],
            "priority": "medium"
        }
    ]
    
    # Create chunks based on keywords (simplified approach)
    for i, chunk_def in enumerate(content_chunks):
        # Extract relevant portions based on keywords
        chunk_content = f"Content related to {chunk_def['topic']}: This section discusses {', '.join(chunk_def['keywords'])} in the context of neural SDE calibration."
        
        # In a real implementation, you would use NLP to extract relevant sections
        # For now, we'll create placeholder chunks with metadata
        chunks.append({
            "chunk_id": f"neural_sde_{chunk_def['topic']}",
            "section": chunk_def['topic'],
            "priority": chunk_def['priority'],
            "content": chunk_content,
            "metadata": {
                "section_type": "content",
                "topic": chunk_def['topic'],
                "importance": chunk_def['priority'],
                "keywords": chunk_def['keywords'],
                "audience": "advanced"
            }
        })
    
    return chunks

def create_tradeknowledge_entry():
    """Create a complete TradeKnowledge entry for the neural SDE paper"""
    
    print("📚 Creating TradeKnowledge entry for Neural SDE paper...")
    
    # Create paper summary
    paper_summary = create_paper_summary()
    
    # Create structured chunks
    chunks = create_structured_chunks()
    
    # Create complete entry
    entry = {
        "document_id": paper_summary["id"],
        "document_type": "academic_paper",
        "processing_date": datetime.now().isoformat(),
        "source_file": "data/books/neural_sde_paper.pdf",
        "metadata": paper_summary,
        "chunks": chunks,
        "search_tags": [
            "neural_sde", "bayesian_calibration", "financial_modeling",
            "volatility_surface", "option_pricing", "risk_management",
            "quantitative_finance", "machine_learning", "uncertainty_quantification"
        ],
        "processing_stats": {
            "total_chunks": len(chunks),
            "critical_chunks": len([c for c in chunks if c.get("priority") == "critical"]),
            "high_priority_chunks": len([c for c in chunks if c.get("priority") == "high"]),
            "medium_priority_chunks": len([c for c in chunks if c.get("priority") == "medium"])
        }
    }
    
    # Save complete entry
    output_file = Path("neural_sde_tradeknowledge_entry.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(entry, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created TradeKnowledge entry: {output_file}")
    print(f"📊 Entry statistics:")
    print(f"   - Total chunks: {entry['processing_stats']['total_chunks']}")
    print(f"   - Critical priority: {entry['processing_stats']['critical_chunks']}")
    print(f"   - High priority: {entry['processing_stats']['high_priority_chunks']}")
    print(f"   - Medium priority: {entry['processing_stats']['medium_priority_chunks']}")
    print(f"🏷️  Search tags: {len(entry['search_tags'])} tags")
    print(f"📝 Document ID: {entry['document_id']}")
    
    # Create summary for quick reference
    summary = {
        "title": paper_summary["title"],
        "authors": paper_summary["authors"],
        "key_points": [
            "Bayesian framework for neural SDE calibration in finance",
            "Provides robust bounds on implied volatility surfaces", 
            "Integrates historical data and option prices",
            "Enables uncertainty quantification for risk measures",
            "Offers parallelizable implementation approach"
        ],
        "practical_applications": [
            "Options trading and volatility modeling",
            "Risk management (VaR, Expected Shortfall)",
            "Model validation and uncertainty assessment",
            "Robust financial model calibration"
        ],
        "complexity": "Advanced (requires SDE and Bayesian knowledge)",
        "relevance": "High for quantitative finance professionals"
    }
    
    summary_file = Path("neural_sde_quick_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"📋 Quick summary: {summary_file}")
    
    return entry

if __name__ == "__main__":
    create_tradeknowledge_entry()