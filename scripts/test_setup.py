#!/usr/bin/env python3
"""
Test script to verify setup
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_imports():
    """Test all imports"""
    logger.info("Testing imports...")
    
    try:
        import fastapi
        import chromadb
        import pdfplumber
        import spacy
        import openai
        from core.config import get_config
        
        logger.info("✅ All imports successful!")
        return True
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False

async def test_config():
    """Test configuration"""
    logger.info("Testing configuration...")
    
    try:
        from core.config import get_config
        config = get_config()
        logger.info(f"✅ Config loaded: {config.app.name} v{config.app.version}")
        return True
    except Exception as e:
        logger.error(f"❌ Config failed: {e}")
        return False

async def test_database():
    """Test database connections"""
    logger.info("Testing database connections...")
    
    try:
        import sqlite3
        from core.config import get_config
        
        config = get_config()
        conn = sqlite3.connect(config.database.sqlite.path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        conn.close()
        
        logger.info(f"✅ SQLite OK - Tables: {len(tables)}")
        
        import chromadb
        client = chromadb.PersistentClient(path=config.database.chroma.persist_directory)
        collections = client.list_collections()
        logger.info(f"✅ ChromaDB OK - Collections: {len(collections)}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Database test failed: {e}")
        return False

async def test_web_server():
    """Test web server startup"""
    logger.info("Testing web server...")
    
    try:
        from main import app
        logger.info("✅ FastAPI app created successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Web server test failed: {e}")
        return False

async def main():
    """Run all tests"""
    logger.info("Starting setup verification...")
    
    tests = [
        test_imports(),
        test_config(),
        test_database(),
        test_web_server()
    ]
    
    results = await asyncio.gather(*tests)
    
    if all(results):
        logger.info("✅ All tests passed! Setup is complete.")
        logger.info("🚀 You can now start the server with: python src/main.py")
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())