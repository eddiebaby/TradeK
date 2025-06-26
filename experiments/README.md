# Experiments Directory

This directory contains experimental and utility scripts used during development and testing.

## Contents

### Ingestion Experiments
- `*_ingest.py` - Various experimental ingestion scripts for different data sources
- `academic_ingest.py` - Academic paper processing
- `epub_ingest.py` - EPUB book processing
- `fast_ingest.py` - Optimized ingestion experiments
- `minimal_ingest.py` - Minimal viable ingestion script
- `robust_ingest.py` - Error-resistant ingestion script

### Database Utilities
- `check_db.py` - Database health checks
- `fix_qdrant.py` - Qdrant database repair utilities
- `reset_qdrant.py` - Qdrant database reset script
- `migrate_schema.py` - Database schema migration tools

### Processing Utilities
- `chunk_existing_book.py` - Book chunking utilities
- `generate_embeddings.py` - Embedding generation scripts
- `process_new_book.py` - New book processing pipeline

### Debug and Development
- `debug_*.py` - Various debugging utilities
- `search_*.py` - Search functionality demos
- `semantic_search_demo.py` - Semantic search demonstrations

## Usage

These scripts are primarily for development, testing, and experimentation. They are not part of the core application but may be useful for:

- Testing new ingestion approaches
- Database maintenance
- Development debugging
- Feature prototyping

Most scripts can be run independently but may require environment setup and dependencies.