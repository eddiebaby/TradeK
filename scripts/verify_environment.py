#!/usr/bin/env python3
"""
Environment verification script - Run this FIRST!
"""

import sys
import subprocess
import importlib
from pathlib import Path

def check_python_version():
    """Ensure Python 3.11+"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} - Need 3.11+")
        return False

def check_virtual_env():
    """Ensure running in virtual environment"""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment active")
        return True
    else:
        print("❌ Not in virtual environment!")
        print("   Run: source venv/bin/activate")
        return False

def check_imports():
    """Check all required imports"""
    required_packages = [
        ('fastapi', 'FastAPI'),
        ('chromadb', 'ChromaDB'),
        ('PyPDF2', 'PyPDF2'),
        ('pdfplumber', 'PDFPlumber'),
        ('spacy', 'spaCy'),
        ('openai', 'OpenAI'),
    ]
    
    all_good = True
    for package, name in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {name} installed")
        except ImportError:
            print(f"❌ {name} missing - run: pip install {package}")
            all_good = False
    
    return all_good

def check_directories():
    """Ensure all directories exist"""
    required_dirs = [
        'src/core', 'src/ingestion', 'src/search', 
        'src/mcp', 'src/utils', 'data/books', 
        'data/chunks', 'logs', 'config'
    ]
    
    all_good = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✅ Directory: {dir_path}")
        else:
            print(f"❌ Missing: {dir_path}")
            all_good = False
    
    return all_good

def check_config_files():
    """Ensure config files exist"""
    files = ['config/config.yaml', '.env']
    all_good = True
    
    for file_path in files:
        if Path(file_path).exists():
            print(f"✅ File: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            all_good = False
    
    # Check .env has API key
    if Path('.env').exists():
        with open('.env', 'r') as f:
            content = f.read()
            if 'OPENAI_API_KEY=your_key_here' in content:
                print("⚠️  Please add your OpenAI API key to .env file!")
    
    return all_good

def main():
    """Run all checks"""
    print("=" * 50)
    print("TradeKnowledge Environment Verification")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_virtual_env),
        ("Package Imports", check_imports),
        ("Directory Structure", check_directories),
        ("Configuration Files", check_config_files),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        results.append(check_func())
    
    print("\n" + "=" * 50)
    if all(results):
        print("✅ ALL CHECKS PASSED - Ready to proceed!")
    else:
        print("❌ SOME CHECKS FAILED - Fix issues above first!")
        sys.exit(1)

if __name__ == "__main__":
    main()