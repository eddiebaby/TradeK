# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TradeKnowledge is a hybrid RAG (Retrieval-Augmented Generation) system that provides semantic and exact-match searching across algorithmic trading, machine learning, and Python books. The system uses an MCP (Model Context Protocol) server interface to enable AI assistants to efficiently search and reference book content.

## Commands

### Development Commands
- `python src/main.py serve` - Start the MCP server
- `python src/main.py add-book <path>` - Add a book to the knowledge base
- `python src/main.py search "<query>"` - Search the knowledge base
- `python src/main.py list-books` - List all indexed books
- `python src/main.py stats` - Show system statistics
- `python src/main.py interactive` - Start interactive CLI mode

### Database Management
- `python scripts/init_db.py` - Initialize SQLite and ChromaDB databases
- `python scripts/test_system.py` - Run system verification tests
- `python scripts/verify_environment.py` - Check Python environment and dependencies

### Testing
- `pytest tests/` - Run all tests
- `pytest tests/unit/` - Run unit tests only
- `pytest tests/integration/` - Run integration tests only
- `pytest --cov=src` - Run tests with coverage

### Development Setup
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
python scripts/init_db.py
```

## Core Architecture

### System Components
The application uses a 4-layer architecture:

1. **Ingestion Layer** (`src/ingestion/`)
   - `ingestion_engine.py`: Orchestrates book processing pipeline
   - `pdf_parser.py`: Extracts text from PDF files
   - `text_chunker.py`: Intelligent text chunking with overlap
   - `embeddings.py`: Generates vector embeddings

2. **Search Layer** (`src/search/`)
   - `hybrid_search.py`: Combines semantic + exact search with configurable weighting
   - `chroma_storage.py`: Vector database operations
   - `text_search.py`: SQLite FTS5 exact text search
   - `vector_search.py`: ChromaDB semantic search

3. **MCP Interface** (`src/mcp/`)
   - `server.py`: Model Context Protocol server providing tools for AI assistants

4. **Core Infrastructure** (`src/core/`)
   - `models.py`: Pydantic data models (Book, Chunk, SearchResult, etc.)
   - `interfaces.py`: Abstract base classes for storage operations
   - `config.py`: Configuration management with YAML + environment variables
   - `sqlite_storage.py`: SQLite database operations

### Data Flow
```
PDF → Parse → Chunk → Embed → Store (SQLite + ChromaDB)
Query → [Semantic Search] + [Exact Search] → Merge → Results
```

### Storage Strategy
- **SQLite**: Book metadata, relationships, FTS5 exact search, chunk context
- **ChromaDB**: Vector embeddings for semantic similarity search
- **Dual queries run concurrently** and results are merged with configurable weighting

## Important Implementation Details

### Async Architecture
- All components are fully async (`async/await`)
- Use `asyncio.to_thread()` for file I/O operations
- Parallel execution of semantic and exact search queries

### Data Models
Key models in `src/core/models.py`:
- `Book`: Core content unit with metadata and processing status
- `Chunk`: Searchable text unit with location and context
- `SearchResult`: Enriched search match with scoring
- `IngestionStatus`: Real-time processing progress

### Configuration
- Main config: `config/config.yaml`
- Environment variables: `.env` file
- Configurable embedding models, chunk sizes, search weights
- Redis caching and performance settings

### Error Handling
- Graceful degradation when MCP library unavailable
- File processing continues on individual chunk failures
- Search fallbacks if one storage backend fails

## Development Guidelines

### Adding New File Types
1. Create parser in `src/ingestion/` (follow pattern from `pdf_parser.py`)
2. Add `FileType` enum value in `src/core/models.py`
3. Register parser in `ingestion_engine.py`
4. Add tests in `tests/unit/ingestion/`

### Extending Search Capabilities
- Implement `BaseVectorStore` or `BaseFullTextStore` interfaces
- Add new search method in `hybrid_search.py`
- Update MCP server tools in `src/mcp/server.py`

### Database Schema Changes
- Modify models in `src/core/models.py`
- Update schema in `scripts/init_db.py`
- Add migration logic for existing data
- Test with `scripts/test_system.py`

### MCP Tool Development
- Tools are defined in `src/mcp/server.py`
- Follow MCP protocol specification for tool definitions
- Include proper error handling and validation
- Update tool descriptions for AI assistant consumption

## File Structure Conventions

### Module Organization
- `src/ingestion/`: All book processing and indexing
- `src/search/`: Search engines and storage backends
- `src/core/`: Shared models, interfaces, config
- `src/mcp/`: MCP server and protocol handling
- `src/utils/`: Shared utilities (logging, etc.)

### Data Directories
- `data/books/`: Source book files
- `data/chromadb/`: Vector database persistence
- `data/chunks/`: Processed text chunks (optional cache)
- `data/embeddings/`: Embedding cache files
- `logs/`: Application logs

### Important Files
- `config/config.yaml`: Main application configuration
- `.env`: Environment variables (API keys, etc.)
- `requirements.txt`: Python dependencies
- `PLAN.md`: Detailed implementation specifications

## Dependencies and Integrations

### Core Dependencies
- `fastapi`: Web framework for MCP server
- `pydantic`: Data validation and serialization
- `chromadb`: Vector database
- `sqlite-utils`: SQLite operations
- `sentence-transformers`: Local embeddings (fallback)
- `openai`: OpenAI embeddings (default)

### Optional Components
- MCP library (graceful fallback if unavailable)
- Redis (caching, can run without)
- OCR libraries (for scanned PDFs)

### System Requirements
- Python 3.11+
- SQLite 3.40+ with FTS5 support
- 4GB+ RAM for 100+ books
- SSD recommended for vector operations

## Common Debugging Tips

### Database Issues
- Check `data/knowledge.db` exists and has tables
- Verify ChromaDB at `data/chromadb/` is accessible
- Run `scripts/init_db.py` to recreate databases

### Search Problems
- Check embedding model availability (OpenAI API key or local model)
- Verify books are properly indexed (`list-books` command)
- Check search statistics (`stats` command)

### Performance Issues
- Monitor memory usage with large books
- Adjust chunk_size and batch_size in config
- Enable C++ extensions if available
- Check disk space for vector storage

### MCP Integration
- Verify MCP library installation
- Check server startup logs for tool registration
- Test with `src/main.py serve` command
- Validate tool definitions match MCP specification