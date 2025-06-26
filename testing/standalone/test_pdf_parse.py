#!/usr/bin/env python3
"""
Minimal PDF parsing test to identify the issue
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.ingestion.pdf_parser import PDFParser

def test_pdf_parse(pdf_path: str):
    """Test PDF parsing with minimal processing"""
    
    print(f"Testing PDF parsing: {pdf_path}")
    
    if not Path(pdf_path).exists():
        print(f"❌ File not found: {pdf_path}")
        return False
    
    file_size_mb = Path(pdf_path).stat().st_size / (1024*1024)
    print(f"📁 File size: {file_size_mb:.2f}MB")
    
    # Test with OCR disabled
    print("📖 Testing PDF parser (OCR disabled)...")
    pdf_parser = PDFParser(enable_ocr=False)
    
    try:
        start_time = time.time()
        parse_result = pdf_parser.parse_file(Path(pdf_path))
        parse_time = time.time() - start_time
        
        print(f"✅ Parse completed in {parse_time:.2f}s")
        print(f"📄 Pages found: {len(parse_result['pages'])}")
        print(f"❌ Errors: {len(parse_result['errors'])}")
        
        if parse_result['errors']:
            print("Error details:")
            for error in parse_result['errors'][:3]:  # Show first 3 errors
                print(f"  - {error}")
        
        # Show sample of first page
        if parse_result['pages']:
            first_page = parse_result['pages'][0]
            sample_text = first_page['text'][:200] + "..." if len(first_page['text']) > 200 else first_page['text']
            print(f"📄 First page sample: {sample_text}")
        
        return True
        
    except Exception as e:
        print(f"❌ Parse failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_pdf_parse.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    success = test_pdf_parse(pdf_path)
    
    if not success:
        sys.exit(1)