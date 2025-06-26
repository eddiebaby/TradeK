#!/usr/bin/env python3
"""
Simple PDF parsing test
"""

import sys
from pathlib import Path
import PyPDF2
import pdfplumber

def simple_pdf_test(pdf_path: str):
    """Simple PDF parsing test"""
    
    print(f"📁 Testing: {Path(pdf_path).name}")
    
    # Test 1: PyPDF2
    print("\n🔧 Test 1: PyPDF2")
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            print(f"  Pages: {len(reader.pages)}")
            
            # Get first page
            first_page = reader.pages[0]
            text = first_page.extract_text()
            print(f"  First page chars: {len(text)}")
            print(f"  Sample: {text[:200]}...")
            
    except Exception as e:
        print(f"  ❌ PyPDF2 failed: {e}")
    
    # Test 2: pdfplumber
    print("\n🔧 Test 2: pdfplumber")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"  Pages: {len(pdf.pages)}")
            
            total_chars = 0
            for i, page in enumerate(pdf.pages[:3]):  # First 3 pages only
                text = page.extract_text() or ""
                total_chars += len(text)
                print(f"  Page {i+1}: {len(text)} chars")
                
                if i == 0:  # Show first page sample
                    sample = text[:200].replace('\n', ' ') + "..." if len(text) > 200 else text
                    print(f"    Sample: {sample}")
            
            print(f"  Total chars (first 3 pages): {total_chars}")
            
    except Exception as e:
        print(f"  ❌ pdfplumber failed: {e}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_pdf_test.py <pdf_path>")
        sys.exit(1)
    
    simple_pdf_test(sys.argv[1])