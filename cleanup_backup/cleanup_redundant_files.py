#!/usr/bin/env python3
"""
Clean up redundant scripts and organize the codebase
"""

import os
from pathlib import Path

def cleanup_redundant_files():
    """Remove redundant scripts and organize the codebase"""
    
    print("🧹 Cleaning up redundant files...")
    print("="*40)
    
    # Files to remove (redundant/temporary scripts)
    redundant_files = [
        "process_full_book.py",          # Replaced by existing ingestion system
        "simple_book_processor.py",      # Temporary workaround
        "generate_book_embeddings.py",   # Redundant with existing system
        "generate_new_embeddings.py",    # One-time use script
        "populate_persistent_qdrant.py", # One-time setup script
        "populate_chromadb.py",          # One-time fix script
        "update_vector_db.py",           # Should use existing system
        "process_new_pdf.py",            # Should use existing ingestion
    ]
    
    # Files to keep (useful/necessary)
    keep_files = [
        "search_book.py",               # Main search interface
        "search_all.py",                # Comprehensive search tool
        "generate_embeddings.py",       # Core embedding generation
        "setup_qdrant.py",              # Qdrant setup utility
        "setup_persistent_qdrant.py",   # Persistent Qdrant setup
    ]
    
    # Create cleanup directory for backup
    cleanup_dir = Path("cleanup_backup")
    cleanup_dir.mkdir(exist_ok=True)
    
    files_removed = 0
    files_backed_up = 0
    
    for filename in redundant_files:
        file_path = Path(filename)
        
        if file_path.exists():
            # Move to backup directory instead of deleting
            backup_path = cleanup_dir / filename
            
            try:
                file_path.rename(backup_path)
                print(f"  ✅ Moved {filename} → cleanup_backup/")
                files_backed_up += 1
            except Exception as e:
                print(f"  ❌ Error moving {filename}: {e}")
        else:
            print(f"  ℹ️  {filename} (not found)")
    
    print(f"\n📊 Cleanup Summary:")
    print(f"   Files backed up: {files_backed_up}")
    print(f"   Cleanup directory: {cleanup_dir.absolute()}")
    
    # Show what's left
    print(f"\n📋 Remaining utility scripts:")
    remaining_scripts = []
    for script in Path(".").glob("*.py"):
        if script.name not in ["cleanup_redundant_files.py"] and not script.name.startswith("__"):
            remaining_scripts.append(script.name)
    
    remaining_scripts.sort()
    for script in remaining_scripts:
        purpose = get_script_purpose(script)
        print(f"  📄 {script} - {purpose}")
    
    return True

def get_script_purpose(filename):
    """Get the purpose of a script"""
    purposes = {
        "search_book.py": "Main search interface",
        "search_all.py": "Comprehensive search with all results",
        "generate_embeddings.py": "Core embedding generation",
        "setup_qdrant.py": "Qdrant database setup",
        "setup_persistent_qdrant.py": "Persistent Qdrant configuration",
        "test_system.py": "System testing utilities",
        "test_search.py": "Search testing",
        "semantic_search_demo.py": "Demo script",
        "optimized_ingest.py": "Optimized ingestion pipeline",
        "fast_ingest.py": "Fast ingestion script",
        "chunk_existing_book.py": "Book chunking utility"
    }
    
    return purposes.get(filename, "Utility script")

if __name__ == "__main__":
    success = cleanup_redundant_files()
    if success:
        print("\n🏆 Cleanup completed!")
        print("📋 Redundant files moved to cleanup_backup/")
        print("📋 Core system preserved and organized")
    else:
        print("\n❌ Cleanup failed")