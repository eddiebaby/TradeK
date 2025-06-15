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
- `python scripts/init_db.py` - Initialize SQLite and Qdrant databases
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
   - `qdrant_storage.py`: Qdrant vector database operations
   - `text_search.py`: SQLite FTS5 exact text search with security validation
   - `vector_search.py`: Qdrant semantic search

3. **MCP Interface** (`src/mcp/`)
   - `server.py`: Model Context Protocol server providing tools for AI assistants

4. **Core Infrastructure** (`src/core/`)
   - `models.py`: Pydantic data models with security validation (Book, Chunk, SearchResult, etc.)
   - `interfaces.py`: Abstract base classes for storage operations
   - `config.py`: Configuration management with YAML + environment variables
   - `sqlite_storage.py`: SQLite database operations
   - `qdrant_storage.py`: Qdrant vector database operations

### Data Flow
```
PDF → Parse → Chunk → Embed → Store (SQLite + Qdrant)
Query → [Semantic Search] + [Exact Search] → Merge → Results
```

### Storage Strategy
- **SQLite**: Book metadata, relationships, FTS5 exact search, chunk context
- **Qdrant**: Vector embeddings for semantic similarity search (768-dimensional)
- **Dual queries run concurrently** and results are merged with configurable weighting
- **Security**: File path validation, query sanitization, size limits enforced

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
- **Enhanced**: Circuit breaker patterns, retry logic with exponential backoff
- **Security**: Comprehensive input validation and sanitization
- **Resource Management**: Async context managers for proper cleanup

## Security Guidelines

### Input Validation
- **File Paths**: All file paths validated to prevent directory traversal attacks
- **Search Queries**: SQL injection prevention with comprehensive sanitization
- **File Operations**: Size limits (500MB max) and type validation enforced
- **Path Security**: Detection of suspicious patterns (`../`, `~`, command injection)

### Security Testing
- Create security test suites for all user-facing functionality
- Test path traversal prevention, query sanitization, file size limits
- Validate error messages don't expose sensitive information
- Regular security validation during development

### Resource Management
- Use async context managers for proper resource cleanup
- Implement circuit breaker patterns for external service calls
- Add retry logic with exponential backoff for resilience
- Cleanup partial state during error conditions

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
- `data/books/`: Source book files (security validated)
- `data/qdrant/`: Qdrant vector database persistence
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
- `pydantic`: Data validation and serialization with security validation
- `qdrant-client`: Qdrant vector database client
- `sqlite-utils`: SQLite operations
- `httpx`: HTTP client for local embeddings (Ollama)
- `sentence-transformers`: Local embeddings (fallback)
- `openai`: OpenAI embeddings (optional)

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
- Verify Qdrant at `data/qdrant/` is accessible
- Run `scripts/init_db.py` to recreate databases
- **Migration**: If upgrading from ChromaDB, run migration scripts

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

## Recent Improvements (2025)

### Security Enhancements
- **File Path Validation**: Comprehensive security validation in `models.py` prevents directory traversal attacks
- **Search Query Sanitization**: SQL injection prevention in `text_search.py` with pattern detection
- **File Size Limits**: 500MB maximum file size enforcement with proper error handling
- **Input Validation**: All user inputs validated and sanitized before processing

### Database Migration
- **ChromaDB → Qdrant**: Successfully migrated from ChromaDB to Qdrant for better performance
- **Embedding Dimensions**: Updated to 768-dimensional embeddings for compatibility
- **Import Cleanup**: Removed compatibility aliases and updated all imports consistently
- **Configuration Updates**: Fixed parameter passing between storage components

### Error Handling & Reliability
- **Circuit Breaker Patterns**: Added for external service calls with configurable thresholds
- **Retry Logic**: Exponential backoff for failed operations with maximum retry limits
- **Resource Management**: Async context managers for proper cleanup and resource management
- **Partial State Recovery**: Enhanced error recovery with cleanup of incomplete operations

### Testing & Validation
- **Security Test Suite**: Comprehensive tests for all security fixes (11 tests covering path validation, query sanitization, file size limits)
- **Compatibility Tests**: Qdrant migration compatibility and embedding generation validation
- **Error Recovery Tests**: Validation of cleanup logic and resource management

### Performance Optimizations
- **Local Embeddings**: Ollama integration for offline embedding generation with caching
- **Parallel Processing**: Improved concurrent execution of search operations
- **Configuration Management**: Streamlined config parameter passing and validation