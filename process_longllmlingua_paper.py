#!/usr/bin/env python3
"""
Process LongLLMLingua Paper with Qwen2.5-Coder Integration

This script processes the LongLLMLingua paper (2024.acl-long.91.pdf) with:
- Academic content extraction
- Mathematical formula detection 
- Algorithm extraction with Qwen2.5-Coder
- Knowledge graph integration
"""

import asyncio
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import fitz  # PyMuPDF
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LLMLinguaPaperProcessor:
    """Process the LongLLMLingua academic paper with AI enhancement"""
    
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model = "qwen2.5-coder:7b"
        
    async def process_longllmlingua_paper(self):
        """Process the LongLLMLingua paper comprehensively"""
        
        paper_path = Path("/home/scott/TradeKnowledge/books and papers (pdf and epub)/2024.acl-long.91.pdf")
        
        if not paper_path.exists():
            logger.error(f"Paper not found: {paper_path}")
            return
        
        logger.info("🔬 Processing LongLLMLingua Paper (ACL 2024)")
        logger.info("=" * 60)
        
        # Extract full text
        text = self.extract_paper_text(paper_path)
        logger.info(f"📝 Extracted {len(text):,} characters")
        
        # Identify sections
        sections = self.identify_academic_sections(text)
        logger.info(f"📚 Identified {len(sections)} sections")
        
        # Extract algorithms and formulas
        algorithms = await self.extract_algorithms_with_qwen(sections)
        logger.info(f"🧮 Extracted {len(algorithms)} algorithms/formulas")
        
        # Analyze relevance to TradeKnowledge
        relevance = await self.analyze_trading_relevance(text, algorithms)
        
        # Generate implementation suggestions
        implementations = await self.generate_implementation_suggestions(algorithms)
        
        # Display results
        self.display_results(sections, algorithms, relevance, implementations)
        
        # Save to knowledge graph
        await self.save_to_knowledge_graph(paper_path.name, sections, algorithms, relevance)
        
        return {
            'sections': sections,
            'algorithms': algorithms,
            'relevance': relevance,
            'implementations': implementations
        }
    
    def extract_paper_text(self, paper_path: Path) -> str:
        """Extract full text from the PDF"""
        try:
            doc = fitz.open(paper_path)
            full_text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                full_text += f"\\n--- Page {page_num + 1} ---\\n{text}"
            
            doc.close()
            return full_text
            
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return ""
    
    def identify_academic_sections(self, text: str) -> Dict[str, str]:
        """Identify and extract academic paper sections"""
        sections = {}
        
        # Common section patterns for academic papers
        section_patterns = {
            'abstract': r'Abstract\\s*\\n(.*?)(?=\\n\\s*1\\s+|\\n\\s*Introduction|\\n\\s*\\d+\\.)',
            'introduction': r'(?:1\\s+)?Introduction\\s*\\n(.*?)(?=\\n\\s*2\\s+|\\n\\s*Related Work|\\n\\s*\\d+\\.)',
            'related_work': r'(?:2\\s+)?Related Work\\s*\\n(.*?)(?=\\n\\s*3\\s+|\\n\\s*Method|\\n\\s*\\d+\\.)',
            'method': r'(?:3\\s+)?(?:Method|Methodology|Approach)\\s*\\n(.*?)(?=\\n\\s*4\\s+|\\n\\s*Experiment|\\n\\s*\\d+\\.)',
            'experiments': r'(?:4\\s+)?(?:Experiment|Results|Evaluation)\\s*\\n(.*?)(?=\\n\\s*5\\s+|\\n\\s*Conclusion|\\n\\s*\\d+\\.)',
            'conclusion': r'(?:5\\s+)?(?:Conclusion|Summary)\\s*\\n(.*?)(?=\\n\\s*References?|\\n\\s*Acknowledge)',
        }
        
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                sections[section_name] = content[:2000]  # Limit length
                logger.info(f"✅ Found {section_name}: {len(content)} chars")
        
        return sections
    
    async def extract_algorithms_with_qwen(self, sections: Dict[str, str]) -> List[Dict]:
        """Use Qwen2.5-Coder to extract algorithms and mathematical content"""
        algorithms = []
        
        # Focus on method and experiments sections for algorithms
        relevant_sections = ['method', 'experiments']
        
        for section_name in relevant_sections:
            if section_name not in sections:
                continue
                
            logger.info(f"🔍 Analyzing {section_name} section with Qwen2.5-Coder...")
            
            prompt = f'''Analyze this academic paper section and extract:
1. Mathematical formulas and equations
2. Algorithms or computational methods
3. Key technical concepts
4. Performance metrics or benchmarks

Section: {section_name.title()}
Content: {sections[section_name]}

Please identify and format any mathematical expressions, algorithms, or computational approaches clearly.'''

            try:
                result = await self.query_qwen(prompt)
                if result:
                    algorithms.append({
                        'section': section_name,
                        'analysis': result,
                        'source_text': sections[section_name][:500]  # First 500 chars for context
                    })
            except Exception as e:
                logger.error(f"Error analyzing {section_name}: {e}")
        
        return algorithms
    
    async def analyze_trading_relevance(self, full_text: str, algorithms: List[Dict]) -> Dict:
        """Analyze relevance to trading and financial applications"""
        logger.info("💼 Analyzing relevance to trading applications...")
        
        prompt = f'''This is a paper about LongLLMLingua - a method for prompt compression to accelerate LLMs in long context scenarios.

Paper summary: {full_text[:1000]}...

Analyze how this research could be applied to financial trading systems and algorithmic trading:

1. How could prompt compression help in trading applications?
2. What are the potential benefits for financial data processing?
3. Could this improve trading algorithm performance or cost efficiency?
4. What specific trading use cases would benefit most?
5. Are there any implementation considerations for financial systems?

Provide specific, actionable insights for applying this research to trading systems.'''
        
        try:
            relevance_analysis = await self.query_qwen(prompt)
            return {
                'analysis': relevance_analysis,
                'confidence': 0.8,  # High confidence for LLMLingua relevance
                'priority': 'high'  # High priority for cost optimization
            }
        except Exception as e:
            logger.error(f"Error analyzing trading relevance: {e}")
            return {'analysis': 'Analysis failed', 'confidence': 0, 'priority': 'low'}
    
    async def generate_implementation_suggestions(self, algorithms: List[Dict]) -> List[Dict]:
        """Generate implementation suggestions using Qwen2.5-Coder"""
        implementations = []
        
        for algorithm in algorithms:
            logger.info(f"💻 Generating implementation for {algorithm['section']}...")
            
            prompt = f'''Based on this algorithm analysis from the LongLLMLingua paper:

{algorithm['analysis']}

Generate Python implementation suggestions or code snippets that could be used in a trading system. Focus on:

1. Practical Python code that implements key concepts
2. Integration with common trading libraries (pandas, numpy, etc.)
3. Performance considerations for real-time trading
4. Cost optimization techniques

Provide clean, commented Python code where applicable.'''
            
            try:
                implementation = await self.query_qwen(prompt)
                if implementation:
                    implementations.append({
                        'section': algorithm['section'],
                        'code_suggestions': implementation,
                        'complexity': 'medium'  # Most implementations are medium complexity
                    })
            except Exception as e:
                logger.error(f"Error generating implementation: {e}")
        
        return implementations
    
    async def query_qwen(self, prompt: str) -> str:
        """Query Qwen2.5-Coder model via Ollama"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for technical analysis
                "top_p": 0.9
            }
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get('response', '').strip()
        except Exception as e:
            logger.error(f"Error querying Qwen: {e}")
            return ""
    
    def display_results(self, sections: Dict, algorithms: List[Dict], relevance: Dict, implementations: List[Dict]):
        """Display comprehensive results"""
        print("\\n" + "="*80)
        print("🎯 LONGLLMLINGUA PAPER ANALYSIS RESULTS")
        print("="*80)
        
        # Paper sections
        print(f"\\n📚 SECTIONS IDENTIFIED ({len(sections)}):")
        for section_name, content in sections.items():
            print(f"  • {section_name.title()}: {len(content)} characters")
        
        # Algorithms extracted
        print(f"\\n🧮 ALGORITHMS/FORMULAS EXTRACTED ({len(algorithms)}):")
        for algo in algorithms:
            print(f"  • {algo['section'].title()}: {len(algo['analysis'])} chars analysis")
        
        # Trading relevance
        print(f"\\n💼 TRADING RELEVANCE:")
        print(f"  • Priority: {relevance.get('priority', 'unknown').upper()}")
        print(f"  • Confidence: {relevance.get('confidence', 0):.1%}")
        if relevance.get('analysis'):
            preview = relevance['analysis'][:200].replace('\\n', ' ')
            print(f"  • Summary: {preview}...")
        
        # Implementation suggestions
        print(f"\\n💻 IMPLEMENTATION SUGGESTIONS ({len(implementations)}):")
        for impl in implementations:
            print(f"  • {impl['section'].title()}: {impl['complexity']} complexity")
        
        print("\\n" + "="*80)
    
    async def save_to_knowledge_graph(self, paper_name: str, sections: Dict, algorithms: List[Dict], relevance: Dict):
        """Save extracted knowledge to the knowledge graph"""
        logger.info("💾 Saving to knowledge graph...")
        
        # This is where we would integrate with the MCP memory system
        # For now, just save to a JSON file
        results = {
            'paper': paper_name,
            'timestamp': '2025-06-26',
            'sections': sections,
            'algorithms': algorithms,
            'trading_relevance': relevance,
            'processing_notes': 'Processed with Qwen2.5-Coder for algorithm extraction'
        }
        
        output_file = Path("/home/scott/TradeKnowledge/longllmlingua_analysis.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Analysis saved to: {output_file}")


async def main():
    """Main processing function"""
    processor = LLMLinguaPaperProcessor()
    await processor.process_longllmlingua_paper()


if __name__ == "__main__":
    asyncio.run(main())