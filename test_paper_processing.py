#!/usr/bin/env python3
"""
Simple test for academic paper processing
"""

import asyncio
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_basic_processing():
    """Test basic paper processing without complex dependencies"""
    
    papers_dir = Path("/home/scott/TradeKnowledge/books and papers (pdf and epub)")
    
    # List available papers
    pdf_files = [f for f in papers_dir.glob("*.pdf") if not f.name.endswith(':Zone.Identifier')]
    
    if not pdf_files:
        logger.error("No PDF files found")
        return
    
    logger.info(f"Found {len(pdf_files)} academic papers:")
    for i, paper in enumerate(pdf_files, 1):
        logger.info(f"  {i}. {paper.name}")
    
    # Test with the most recent paper
    test_paper = pdf_files[0]
    logger.info(f"\nTesting with: {test_paper.name}")
    
    try:
        # Try to use PyMuPDF for basic text extraction
        import fitz
        
        doc = fitz.open(test_paper)
        logger.info(f"PDF opened successfully: {len(doc)} pages")
        
        # Extract text from first few pages
        text_sample = ""
        for page_num in range(min(3, len(doc))):
            page = doc.load_page(page_num)
            text = page.get_text()
            text_sample += text
        
        doc.close()
        
        logger.info(f"Extracted {len(text_sample)} characters from first 3 pages")
        
        # Look for mathematical content indicators
        math_indicators = ['∑', '∫', '∂', '∇', 'equation', 'formula', '\\', '$']
        found_indicators = [ind for ind in math_indicators if ind in text_sample]
        
        if found_indicators:
            logger.info(f"Mathematical content detected: {found_indicators}")
        else:
            logger.info("No obvious mathematical content found in sample")
        
        # Show a sample of the text
        sample = text_sample[:500].replace('\n', ' ')
        logger.info(f"Text sample: {sample}...")
        
        return True
        
    except ImportError:
        logger.error("PyMuPDF (fitz) not available. Install with: pip install PyMuPDF")
        return False
    except Exception as e:
        logger.error(f"Error processing paper: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_basic_processing())