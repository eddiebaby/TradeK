#!/usr/bin/env python3
"""
Process Remaining Academic Papers

Efficiently process the remaining 3 academic papers with Qwen2.5-Coder analysis.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

import fitz
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def process_remaining_papers():
    """Process the remaining 3 academic papers"""
    
    papers_dir = Path("/home/scott/TradeKnowledge/books and papers (pdf and epub)")
    ollama_url = "http://localhost:11434/api/generate"
    
    # Papers to process (excluding the already processed LongLLMLingua)
    papers = [
        "2024.findings-acl.57.pdf",
        "2023.emnlp-main.825.pdf", 
        "2505.12540v2.pdf"
    ]
    
    results = {}
    
    for paper_name in papers:
        paper_path = papers_dir / paper_name
        
        if not paper_path.exists():
            logger.error(f"Paper not found: {paper_name}")
            continue
        
        logger.info(f"\\n{'='*60}")
        logger.info(f"📄 Processing: {paper_name}")
        logger.info(f"{'='*60}")
        
        try:
            # Extract paper content
            text = extract_paper_text(paper_path)
            logger.info(f"📝 Extracted {len(text):,} characters")
            
            # Get paper summary and key insights
            summary = await analyze_paper_with_qwen(text, paper_name, ollama_url)
            
            # Analyze trading relevance
            relevance = await analyze_trading_relevance(text, paper_name, ollama_url)
            
            results[paper_name] = {
                'summary': summary,
                'trading_relevance': relevance,
                'text_length': len(text)
            }
            
            # Display results
            display_paper_results(paper_name, summary, relevance)
            
        except Exception as e:
            logger.error(f"Error processing {paper_name}: {e}")
            continue
    
    # Save all results
    output_file = Path("/home/scott/TradeKnowledge/academic_papers_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\\n✅ All papers processed. Results saved to: {output_file}")
    return results


def extract_paper_text(paper_path: Path) -> str:
    """Extract text from PDF"""
    try:
        doc = fitz.open(paper_path)
        text = ""
        
        # Extract first 5 pages for analysis (enough for abstract, intro, method)
        for page_num in range(min(5, len(doc))):
            page = doc.load_page(page_num)
            text += page.get_text()
        
        doc.close()
        return text
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return ""


async def analyze_paper_with_qwen(text: str, paper_name: str, ollama_url: str) -> str:
    """Analyze paper with Qwen2.5-Coder"""
    
    prompt = f'''Analyze this academic paper and provide a concise summary covering:

1. **Research Topic**: What is the main research question or problem?
2. **Key Contributions**: What are the main contributions or innovations?
3. **Technical Approach**: What methods or algorithms are used?
4. **Results**: What are the key findings or performance improvements?
5. **Significance**: Why is this research important?

Paper: {paper_name}
Content (first 5 pages): {text[:3000]}...

Provide a clear, structured analysis focusing on the technical aspects and innovations.'''

    return await query_qwen(prompt, ollama_url)


async def analyze_trading_relevance(text: str, paper_name: str, ollama_url: str) -> str:
    """Analyze relevance to trading systems"""
    
    prompt = f'''Analyze how this research paper could be applied to financial trading and algorithmic trading systems:

Paper: {paper_name}
Content: {text[:2000]}...

Consider:
1. **Direct Applications**: How could this research directly help trading systems?
2. **Algorithmic Trading**: Benefits for automated trading strategies?
3. **Data Processing**: Improvements for financial data analysis?
4. **Performance**: Speed, accuracy, or cost improvements?
5. **Implementation Priority**: High/Medium/Low priority for a trading system?

Provide specific, actionable insights for financial technology applications.'''

    return await query_qwen(prompt, ollama_url)


async def query_qwen(prompt: str, ollama_url: str) -> str:
    """Query Qwen2.5-Coder via Ollama"""
    payload = {
        "model": "qwen2.5-coder:7b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9
        }
    }
    
    try:
        response = requests.post(ollama_url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get('response', '').strip()
    except Exception as e:
        logger.error(f"Error querying Qwen: {e}")
        return f"Analysis failed: {e}"


def display_paper_results(paper_name: str, summary: str, relevance: str):
    """Display analysis results"""
    print(f"\\n📊 ANALYSIS RESULTS: {paper_name}")
    print("="*50)
    
    print("\\n🔬 RESEARCH SUMMARY:")
    summary_preview = summary[:300].replace('\\n', ' ')
    print(f"  {summary_preview}...")
    
    print("\\n💼 TRADING RELEVANCE:")
    relevance_preview = relevance[:300].replace('\\n', ' ')
    print(f"  {relevance_preview}...")
    
    print("\\n" + "-"*50)


if __name__ == "__main__":
    asyncio.run(process_remaining_papers())