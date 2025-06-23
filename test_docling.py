#!/usr/bin/env python3
"""
Quick Docling test to evaluate capabilities with CRC book
"""

import time
from pathlib import Path
from docling.document_converter import DocumentConverter

def test_docling_basic():
    """Test basic Docling functionality"""
    print("🔬 Testing Docling Basic Functionality")
    print("=" * 50)
    
    # Path to our CRC book
    pdf_path = Path("data/books/Guillaume_Coqueret_Tony_Guida_-_Machine_Learning_for_Factor_Investing__Python_Version-CRC_Press_2023.pdf")
    
    if not pdf_path.exists():
        print(f"❌ Test file not found: {pdf_path}")
        return
    
    print(f"📁 Testing file: {pdf_path.name}")
    print(f"📊 File size: {pdf_path.stat().st_size / (1024*1024):.1f} MB")
    
    try:
        # Initialize Docling converter
        print("\n🔄 Initializing Docling converter...")
        converter = DocumentConverter()
        
        # Test conversion
        print("🔄 Converting document (this may take a while)...")
        start_time = time.time()
        
        result = converter.convert(pdf_path)
        
        conversion_time = time.time() - start_time
        print(f"✅ Conversion completed in {conversion_time:.2f}s")
        
        # Analyze results
        print("\n📊 DOCLING ANALYSIS RESULTS")
        print("=" * 30)
        
        print(f"📄 Document type: {result.document.name}")
        print(f"📝 Content length: {len(result.document.export_to_markdown())} characters")
        
        # Show first 500 chars of markdown
        markdown_content = result.document.export_to_markdown()
        print(f"\n📖 First 500 characters of markdown:")
        print("-" * 40)
        print(markdown_content[:500] + "..." if len(markdown_content) > 500 else markdown_content)
        
        # Export to different formats
        print(f"\n💾 Export capabilities:")
        print(f"   • Markdown: {len(markdown_content)} chars")
        
        try:
            json_content = result.document.export_to_json()
            print(f"   • JSON: Available")
        except Exception as e:
            print(f"   • JSON: Error - {e}")
            
        try:
            html_content = result.document.export_to_html()
            print(f"   • HTML: {len(html_content)} chars")
        except Exception as e:
            print(f"   • HTML: Error - {e}")
        
        # Check for specific content types
        print(f"\n🔍 Content Analysis:")
        if "table" in markdown_content.lower():
            print("   ✅ Tables detected")
        if "$" in markdown_content or "\\(" in markdown_content:
            print("   ✅ Mathematical content detected")
        if "```" in markdown_content:
            print("   ✅ Code blocks detected")
            
        # Memory usage
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"\n🧠 Memory usage: {memory_mb:.1f} MB")
        
        return {
            'success': True,
            'conversion_time': conversion_time,
            'markdown_length': len(markdown_content),
            'memory_usage_mb': memory_mb
        }
        
    except Exception as e:
        print(f"❌ Docling test failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    test_docling_basic()