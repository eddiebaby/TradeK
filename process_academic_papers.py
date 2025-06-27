#!/usr/bin/env python3
"""
Academic Papers Processing Script

Process academic papers with LaTeX formula recognition and mathematical content extraction.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append('/home/scott/TradeKnowledge')

from src.ingestion.academic_paper_processor import AcademicPaperProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def process_papers():
    """Process all academic papers in the books and papers directory"""
    
    papers_dir = Path("/home/scott/TradeKnowledge/books and papers (pdf and epub)")
    processor = AcademicPaperProcessor()
    
    # Get all PDF files
    pdf_files = list(papers_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.error("No PDF files found in the papers directory")
        return
    
    logger.info(f"Found {len(pdf_files)} academic papers to process")
    
    # Process papers in order (most recent first)
    ordered_papers = sorted(pdf_files, key=lambda x: x.name, reverse=True)
    
    for i, paper_path in enumerate(ordered_papers, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing paper {i}/{len(ordered_papers)}: {paper_path.name}")
        logger.info(f"{'='*60}")
        
        try:
            # Skip Windows Zone.Identifier files
            if paper_path.name.endswith(':Zone.Identifier'):
                logger.info("Skipping Windows Zone.Identifier file")
                continue
            
            # Process the paper
            document = await processor.process_paper(paper_path)
            
            # Display results
            print_processing_results(document, paper_path.name)
            
            # Save results (optional - can be enhanced later)
            await save_processing_results(document, paper_path)
            
        except Exception as e:
            logger.error(f"Error processing {paper_path.name}: {e}")
            continue
    
    logger.info(f"\n{'='*60}")
    logger.info("Academic paper processing completed!")
    logger.info(f"{'='*60}")


def print_processing_results(document, filename):
    """Print processing results for a document"""
    print(f"\n📄 Paper: {filename}")
    print(f"📝 Content length: {len(document.content):,} characters")
    
    # Metadata summary
    metadata = document.metadata
    print(f"🧮 Formulas extracted: {metadata.get('formula_count', 0)}")
    print(f"📚 Sections identified: {metadata.get('section_count', 0)}")
    print(f"🔬 Complexity score: {metadata.get('complexity_score', 0):.2f}")
    
    # Show formulas if any
    if 'formulas' in metadata and metadata['formulas']:
        print(f"\n🔢 Mathematical Formulas:")
        for i, formula in enumerate(metadata['formulas'][:5], 1):  # Show first 5
            print(f"  {i}. Page {formula['page']}: {formula['description'][:100]}...")
            if formula['complexity'] > 0.5:
                print(f"     (High complexity: {formula['complexity']:.2f})")
    
    # Show sections if any
    if 'sections' in metadata and metadata['sections']:
        print(f"\n📋 Academic Sections:")
        for section in metadata['sections']:
            concepts = ", ".join(section['key_concepts'][:3])
            print(f"  • {section['type'].title()}: {section['title']}")
            print(f"    Key concepts: {concepts}")
            if section['formula_count'] > 0:
                print(f"    Contains {section['formula_count']} formulas")


async def save_processing_results(document, paper_path):
    """Save processing results for later use"""
    # For now, just log that we would save
    # Could be enhanced to save to database, JSON, etc.
    output_dir = Path("/home/scott/TradeKnowledge/processed_papers")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"{paper_path.stem}_analysis.json"
    logger.info(f"Results ready for saving to: {output_file}")
    
    # Would save document.metadata as JSON here
    # json.dump(document.metadata, open(output_file, 'w'), indent=2)


if __name__ == "__main__":
    asyncio.run(process_papers())