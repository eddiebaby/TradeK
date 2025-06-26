# Testing Directory

This directory contains additional test scripts and utilities that supplement the main `tests/` directory.

## Structure

### `integration/`
Integration tests for complex system interactions:
- `test_agent_integration.py` - Agent system integration tests
- `test_openai_agents_integration.py` - OpenAI agents SDK integration tests
- `test_openai_integration_simple.py` - Simplified OpenAI integration validation

### `standalone/`
Standalone test scripts for specific components:
- `test_chunking_only.py` - Text chunking functionality
- `test_embeddings.py` - Embedding generation tests
- `test_minimal_pdf.py` - PDF processing tests
- `test_persistent_vector_db.py` - Vector database persistence tests
- `test_search*.py` - Various search functionality tests
- `test_semantic_search.py` - Semantic search capability tests

## Usage

### Integration Tests
```bash
cd testing/integration
python test_openai_integration_simple.py
```

### Standalone Tests
```bash
cd testing/standalone
python test_chunking_only.py
```

## Relationship to Main Tests

The main test suite is in `/tests/` and follows pytest conventions. These additional tests are:
- Standalone validation scripts
- Integration tests that may require external services
- Development and debugging test utilities

## Note

Some tests may require:
- Environment variables (OPENAI_API_KEY, etc.)
- Running services (Qdrant, Ollama)
- Specific test data in `/test_data/`