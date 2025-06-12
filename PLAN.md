# Product Requirements Document: Book Knowledge MCP Server
## For Algorithmic Trading & Machine Learning Reference System

### Document Version: 1.0
### Date: June 11, 2025
### Project Codename: "TradeKnowledge"

---

## 1. EXECUTIVE SUMMARY

### 1.1 Problem Statement
We have a collection of PDF and EPUB books covering Python, machine learning, and algorithmic trading. Currently, there's no efficient way to search, reference, and extract relevant information from these materials while developing trading algorithms. Manual searching is time-consuming and often misses relevant cross-references between concepts.

### 1.2 Solution Overview
Build a hybrid RAG (Retrieval-Augmented Generation) system with an MCP (Model Context Protocol) server interface that allows semantic and exact-match searching across all book content. The system will enable AI assistants and developers to quickly find relevant code examples, trading strategies, and ML concepts.

### 1.3 Success Metrics
- Query response time < 500ms for semantic search
- Query response time < 100ms for exact match (with C++ optimization)
- 95% accuracy in retrieving relevant content
- Support for at least 100 concurrent book searches
- Ability to handle books totaling > 10GB of content

---

## 2. DETAILED REQUIREMENTS

### 2.1 Functional Requirements

#### 2.1.1 Book Ingestion Pipeline
- **FR-001**: System MUST support PDF format (including scanned PDFs with OCR)
- **FR-002**: System MUST support EPUB format
- **FR-003**: System SHOULD support Jupyter notebook (.ipynb) format
- **FR-004**: System MUST extract and preserve:
  - Plain text content
  - Code blocks with language identification
  - Mathematical formulas (LaTeX/MathML)
  - Tables and structured data
  - Image captions and alt text
  - Chapter/section hierarchy
  - Page numbers for citation

#### 2.1.2 Text Processing
- **FR-005**: System MUST chunk text intelligently:
  - Default chunk size: 1000 tokens with 200 token overlap
  - Never split code blocks
  - Respect paragraph boundaries
  - Maintain section context
- **FR-006**: System MUST preserve metadata:
  - Source book title, author, ISBN
  - Chapter/section titles
  - Page numbers
  - Publication year
- **FR-007**: System MUST handle special content:
  - Trading formulas (preserve LaTeX)
  - Code snippets (maintain formatting)
  - Tables (convert to structured format)

#### 2.1.3 Vector Database Requirements
- **FR-008**: System MUST generate embeddings for all chunks
- **FR-009**: System MUST support multiple embedding models:
  - Default: OpenAI text-embedding-ada-002
  - Fallback: sentence-transformers/all-mpnet-base-v2
  - Future: Custom fine-tuned model for finance/trading
- **FR-010**: System MUST store embeddings in ChromaDB with:
  - Persistent storage
  - Metadata filtering
  - Collection management per book/topic

#### 2.1.4 Full-Text Search Requirements
- **FR-011**: System MUST maintain SQLite database with:
  - Full text of each chunk
  - FTS5 (Full-Text Search) enabled
  - Trigram indexing for fuzzy search
- **FR-012**: System MUST support search operators:
  - Exact phrase: "bollinger bands"
  - Wildcards: trade* (matches trader, trading, etc.)
  - Boolean: momentum AND indicator
  - Proximity: "stop NEAR/3 loss"

#### 2.1.5 MCP Server Interface
- **FR-013**: System MUST implement MCP protocol with methods:
  - `search_semantic(query, num_results, filter_books)`
  - `search_exact(query, num_results, filter_books)`
  - `search_hybrid(query, num_results, filter_books, weight_semantic)`
  - `get_chunk_context(chunk_id, context_size)`
  - `list_books()`
  - `add_book(file_path)`
- **FR-014**: System MUST return structured responses:
  ```json
  {
    "results": [
      {
        "chunk_id": "book1_ch3_p45_001",
        "text": "The moving average crossover strategy...",
        "score": 0.89,
        "metadata": {
          "book": "Algorithmic Trading with Python",
          "author": "Example Author",
          "chapter": "3. Technical Indicators",
          "page": 45,
          "type": "text|code|formula"
        },
        "context": {
          "before": "Previous paragraph...",
          "after": "Next paragraph..."
        }
      }
    ],
    "total_results": 156,
    "search_time_ms": 234
  }
  ```

#### 2.1.6 Performance Optimization
- **FR-015**: System MUST use C++ for:
  - Text preprocessing (tokenization, cleaning)
  - Exact-match search algorithms
  - Embedding similarity calculations (SIMD optimized)
- **FR-016**: System MUST implement caching:
  - LRU cache for recent queries
  - Precomputed embeddings cache
  - Frequently accessed chunks in memory

### 2.2 Non-Functional Requirements

#### 2.2.1 Performance
- **NFR-001**: Semantic search < 500ms for up to 1M chunks
- **NFR-002**: Exact search < 100ms using C++ implementation
- **NFR-003**: Book ingestion < 60 seconds per 500-page book
- **NFR-004**: Memory usage < 4GB for 100 books loaded

#### 2.2.2 Reliability
- **NFR-005**: System uptime > 99.9%
- **NFR-006**: Graceful degradation if vector DB unavailable
- **NFR-007**: Automatic recovery from crashes
- **NFR-008**: Data integrity checks on ingestion

#### 2.2.3 Scalability
- **NFR-009**: Support for 1000+ books
- **NFR-010**: Horizontal scaling for multiple MCP instances
- **NFR-011**: Incremental indexing (add books without full rebuild)

#### 2.2.4 Security
- **NFR-012**: Local-only operation (no external API calls in production)
- **NFR-013**: Read-only access to book files
- **NFR-014**: Sanitized inputs to prevent injection attacks

---

## 3. TECHNICAL ARCHITECTURE

### 3.1 System Components

```
┌─────────────────────────────────────────────────────────────┐
│                      MCP Client (AI Assistant)              │
└──────────────────────────────┬──────────────────────────────┘
                               │ MCP Protocol
┌──────────────────────────────┴──────────────────────────────┐
│                         MCP Server                           │
│  ┌────────────────────┐  ┌─────────────────────────────┐   │
│  │   Python FastAPI   │  │   C++ Extension Module      │   │
│  │  - Route handling  │  │  - Fast text search         │   │
│  │  - Query planning  │  │  - SIMD similarity calc     │   │
│  │  - Result ranking  │  │  - Tokenization             │   │
│  └────────────────────┘  └─────────────────────────────┘   │
└──────────────┬───────────────────────────┬──────────────────┘
               │                           │
┌──────────────┴───────────┐ ┌────────────┴──────────────────┐
│      ChromaDB             │ │         SQLite DB             │
│  - Vector embeddings      │ │  - Full text with FTS5       │
│  - Metadata               │ │  - Document structure        │
│  - Collections            │ │  - Citation info             │
└───────────────────────────┘ └───────────────────────────────┘
               │                           │
┌──────────────┴───────────────────────────┴──────────────────┐
│                    Ingestion Pipeline                        │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │ PDF Parser  │  │ EPUB Parser  │  │ Jupyter Parser  │   │
│  │ (PyPDF2)    │  │ (ebooklib)   │  │ (nbformat)      │   │
│  └─────────────┘  └──────────────┘  └─────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow

1. **Ingestion Flow**:
   ```
   Book File → Parser → Text Extraction → Chunking → 
   → Embedding Generation → Vector DB Storage
   → Full Text DB Storage → Metadata Indexing
   ```

2. **Query Flow**:
   ```
   User Query → MCP Server → Query Planning →
   → Parallel: [Vector Search | Text Search] →
   → Result Merging → Re-ranking → Response
   ```

### 3.3 Technology Stack

#### Core Technologies:
- **Language**: Python 3.11+ (primary), C++ 17 (performance modules)
- **MCP Framework**: Custom implementation following MCP spec
- **Web Framework**: FastAPI (async support)
- **Vector Database**: ChromaDB 0.4+
- **Text Database**: SQLite 3.40+ with FTS5
- **Message Queue**: Redis (for async operations)

#### Python Libraries:
- **PDF Processing**: PyPDF2, pdfplumber, OCRmyPDF (for scanned)
- **EPUB Processing**: ebooklib
- **Text Processing**: spaCy, nltk
- **Embeddings**: openai, sentence-transformers
- **Math Parsing**: sympy, latex2sympy
- **Code Detection**: pygments
- **C++ Binding**: pybind11

#### C++ Libraries:
- **Text Search**: RE2 (Google's regex library)
- **SIMD Operations**: xsimd
- **Serialization**: MessagePack
- **Threading**: Intel TBB

---

## 4. IMPLEMENTATION PHASES

### Phase 1: Foundation (Weeks 1-2)
1. Set up development environment
2. Create basic MCP server skeleton
3. Implement PDF parser for clean PDFs
4. Create simple chunking algorithm
5. Set up ChromaDB integration
6. Basic semantic search functionality

### Phase 2: Core Features (Weeks 3-4)
1. Add EPUB support
2. Implement intelligent chunking
3. Add SQLite FTS5 integration
4. Create hybrid search algorithm
5. Implement basic caching
6. Add metadata extraction

### Phase 3: Optimization (Weeks 5-6)
1. Develop C++ text search module
2. Implement SIMD similarity calculations
3. Add advanced caching layers
4. Optimize database queries
5. Performance testing and tuning

### Phase 4: Advanced Features (Weeks 7-8)
1. Add OCR support for scanned PDFs
2. Implement code block detection
3. Add mathematical formula parsing
4. Create Jupyter notebook support
5. Implement query suggestion engine

### Phase 5: Production Ready (Weeks 9-10)
1. Comprehensive testing suite
2. Documentation completion
3. Deployment scripts
4. Monitoring integration
5. Performance benchmarking

---

## 5. RISKS AND MITIGATION

### 5.1 Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| OCR accuracy for scanned books | High | Medium | Use multiple OCR engines, manual verification |
| Embedding model changes | Medium | Low | Abstract embedding interface, version lock |
| C++ integration complexity | Medium | Medium | Extensive testing, fallback to Python |
| Large book processing OOM | High | Medium | Streaming processing, chunked operations |

### 5.2 Business Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Copyright concerns | High | Low | Local-only processing, no redistribution |
| Scope creep | Medium | High | Strict phase gates, clear requirements |
| Team knowledge gaps | Medium | Medium | Detailed documentation, pair programming |

---

## 6. TESTING STRATEGY

### 6.1 Unit Testing
- **Coverage Target**: 90%
- **Key Areas**:
  - Text extraction accuracy
  - Chunking algorithm correctness
  - Search result relevance
  - C++ module integration

### 6.2 Integration Testing
- End-to-end ingestion pipeline
- Search accuracy across different query types
- Performance under load
- Error handling and recovery

### 6.3 Performance Testing
- Benchmark against requirements
- Load testing with 1000+ books
- Memory leak detection
- Query response time validation

### 6.4 User Acceptance Testing
- Trading strategy search scenarios
- Code example retrieval
- Cross-reference navigation
- Citation accuracy

---

## 7. MONITORING AND MAINTENANCE

### 7.1 Metrics to Track
- Query response times (p50, p95, p99)
- Search result relevance scores
- System resource usage
- Error rates and types
- Cache hit rates

### 7.2 Maintenance Tasks
- Weekly: Update embeddings model
- Monthly: Reindex for performance
- Quarterly: Architecture review
- As needed: Add new books

---

## 8. FUTURE ENHANCEMENTS

### Version 2.0 Possibilities:
1. **Multi-modal Search**: Include charts/graphs from books
2. **Fine-tuned Embeddings**: Train on trading/finance corpus
3. **Knowledge Graph**: Build concept relationships
4. **Query Understanding**: NLP for better intent detection
5. **Collaborative Filtering**: "Users who searched X also found Y useful"
6. **Real-time Updates**: Ingest new papers/articles automatically
7. **Trading Strategy Extraction**: Automatically identify and catalog strategies
8. **Backtesting Integration**: Direct connection to backtesting frameworks

---

## APPENDIX A: API SPECIFICATION

### A.1 MCP Methods

```python
# Semantic search using vector embeddings
async def search_semantic(
    query: str,
    num_results: int = 10,
    filter_books: List[str] = None,
    min_score: float = 0.7
) -> SearchResponse

# Exact text search using SQL FTS5
async def search_exact(
    query: str,
    num_results: int = 10,
    filter_books: List[str] = None,
    case_sensitive: bool = False
) -> SearchResponse

# Hybrid search combining both methods
async def search_hybrid(
    query: str,
    num_results: int = 10,
    filter_books: List[str] = None,
    semantic_weight: float = 0.7
) -> SearchResponse

# Get expanded context for a chunk
async def get_chunk_context(
    chunk_id: str,
    before_chunks: int = 1,
    after_chunks: int = 1
) -> ContextResponse

# List all indexed books
async def list_books(
    category: str = None
) -> List[BookMetadata]

# Add new book to the system
async def add_book(
    file_path: str,
    metadata: Dict[str, Any] = None
) -> IngestionResponse
```

### A.2 Response Schemas

```python
class SearchResult:
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    context: Optional[Dict[str, str]]

class SearchResponse:
    results: List[SearchResult]
    total_results: int
    search_time_ms: int
    query_interpretation: Dict[str, Any]

class BookMetadata:
    id: str
    title: str
    author: str
    isbn: Optional[str]
    pages: int
    chunks: int
    file_type: str
    ingestion_date: datetime
    categories: List[str]
```

---

## APPENDIX B: EXAMPLE QUERIES

### Trading Strategy Searches:
- "Python implementation of pairs trading with cointegration"
- "Bollinger Bands mean reversion strategy code"
- "Risk management for futures trading algorithms"

### Machine Learning Searches:
- "LSTM for time series prediction stock prices"
- "Feature engineering for high-frequency trading"
- "Reinforcement learning market making"

### Technical Searches:
- "Vectorized backtesting examples numpy"
- "Order book reconstruction from tick data"
- "C++ FIX protocol implementation"

---

## DOCUMENT APPROVAL

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Product Owner | _____________ | ___/___/___ | _____________ |
| Tech Lead | _____________ | ___/___/___ | _____________ |
| Dev Team Lead | _____________ | ___/___/___ | _____________ |

---

**END OF PRD DOCUMENT**