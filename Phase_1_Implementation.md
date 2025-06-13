# Phase 1: Foundation Implementation Guide
## Building the Core TradeKnowledge System

### Phase 1 Overview

In this phase, we're building the foundation of our book knowledge system. Think of it like constructing a house - we need solid groundwork before adding fancy features. By the end of Phase 1, you'll have a working system that can ingest PDF files, chunk them intelligently, generate embeddings, and perform basic semantic searches.

---

## Environment and Basic Infrastructure

### Complete Development Environment Setup

Let's start by ensuring everyone has an identical development environment. This prevents the classic "it works on my machine" problem.

#### Step 1.1: Create Development Branches

```bash
# In your project directory
git checkout -b dev/phase1-foundation
git push -u origin dev/phase1-foundation

# Create a personal feature branch
git checkout -b feature/your-name-phase1
```

#### Step 1.2: Verify All Prerequisites

Create a verification script that everyone must run:

```python
# Create scripts/verify_environment.py
cat > scripts/verify_environment.py << 'EOF'
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
EOF

# Make it executable and run it
chmod +x scripts/verify_environment.py
python scripts/verify_environment.py
```

### Build Core Data Models

Now we'll create the data structures that represent our domain. Think of these as blueprints for how we'll organize information about books, chunks, and search results.

#### Step 2.1: Create Base Models

```python
# Create src/core/models.py
cat > src/core/models.py << 'EOF'
"""
Core data models for TradeKnowledge

These models define the structure of our data throughout the system.
Think of them as contracts - any component that uses these models
knows exactly what data to expect.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field, validator
import hashlib
from pathlib import Path

class FileType(str, Enum):
    """Supported file types"""
    PDF = "pdf"
    EPUB = "epub"
    NOTEBOOK = "ipynb"
    
class ChunkType(str, Enum):
    """Types of content chunks"""
    TEXT = "text"
    CODE = "code"
    FORMULA = "formula"
    TABLE = "table"
    
class Book(BaseModel):
    """
    Represents a book in our system.
    
    This is our primary unit of content. Each book has metadata
    and is broken down into chunks for processing.
    """
    id: str = Field(description="Unique identifier (usually ISBN or generated)")
    title: str = Field(description="Book title")
    author: Optional[str] = Field(default=None, description="Author name(s)")
    isbn: Optional[str] = Field(default=None, description="ISBN if available")
    file_path: str = Field(description="Path to the original file")
    file_type: FileType = Field(description="Type of file")
    file_hash: str = Field(description="SHA256 hash of file for deduplication")
    total_pages: Optional[int] = Field(default=None, description="Number of pages")
    total_chunks: int = Field(default=0, description="Number of chunks created")
    categories: List[str] = Field(default_factory=list, description="Categories/tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.now)
    indexed_at: Optional[datetime] = Field(default=None)
    
    @validator('file_hash', pre=True, always=True)
    def generate_file_hash(cls, v, values):
        """Generate file hash if not provided"""
        if v:
            return v
        
        file_path = values.get('file_path')
        if file_path and Path(file_path).exists():
            # Read file in chunks to handle large files
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        return None
    
    class Config:
        """Pydantic configuration"""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class Chunk(BaseModel):
    """
    Represents a chunk of content from a book.
    
    Chunks are the atomic units we search through. Each chunk
    maintains its relationship to the source book and surrounding context.
    """
    id: str = Field(description="Unique chunk identifier")
    book_id: str = Field(description="ID of the source book")
    chunk_index: int = Field(description="Position in the book (0-based)")
    text: str = Field(description="The actual text content")
    chunk_type: ChunkType = Field(default=ChunkType.TEXT)
    embedding_id: Optional[str] = Field(default=None, description="ID in vector DB")
    
    # Location information
    chapter: Optional[str] = Field(default=None, description="Chapter title if available")
    section: Optional[str] = Field(default=None, description="Section title if available")
    page_start: Optional[int] = Field(default=None, description="Starting page number")
    page_end: Optional[int] = Field(default=None, description="Ending page number")
    
    # For maintaining context
    previous_chunk_id: Optional[str] = Field(default=None)
    next_chunk_id: Optional[str] = Field(default=None)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    
    @validator('id', pre=True, always=True)
    def generate_chunk_id(cls, v, values):
        """Generate chunk ID if not provided"""
        if v:
            return v
        
        book_id = values.get('book_id', 'unknown')
        chunk_index = values.get('chunk_index', 0)
        return f"{book_id}_chunk_{chunk_index:05d}"
    
    def get_size(self) -> int:
        """Get the size of the chunk in characters"""
        return len(self.text)
    
    def get_token_estimate(self) -> int:
        """Estimate token count (rough approximation)"""
        # Rough estimate: 1 token ≈ 4 characters for English
        return len(self.text) // 4

class SearchResult(BaseModel):
    """
    Represents a single search result.
    
    This contains not just the matching chunk, but also
    relevance scoring and context information.
    """
    chunk: Chunk = Field(description="The matching chunk")
    score: float = Field(description="Relevance score (0-1)")
    match_type: str = Field(description="Type of match: semantic, exact, or hybrid")
    
    # Highlighted snippets
    highlights: List[str] = Field(default_factory=list, description="Relevant text snippets")
    
    # Context for better understanding
    context_before: Optional[str] = Field(default=None)
    context_after: Optional[str] = Field(default=None)
    
    # Source information
    book_title: str
    book_author: Optional[str] = None
    chapter: Optional[str] = None
    page: Optional[int] = None

class SearchResponse(BaseModel):
    """
    Complete response to a search query.
    
    This includes all results plus metadata about the search itself.
    """
    query: str = Field(description="Original search query")
    results: List[SearchResult] = Field(description="List of results")
    total_results: int = Field(description="Total number of matches found")
    returned_results: int = Field(description="Number of results returned")
    search_time_ms: int = Field(description="Time taken to search in milliseconds")
    
    # Search metadata
    search_type: str = Field(description="Type of search performed")
    filters_applied: Dict[str, Any] = Field(default_factory=dict)
    
    # Query interpretation (helpful for debugging)
    interpreted_query: Optional[Dict[str, Any]] = Field(default=None)
    
    def get_top_result(self) -> Optional[SearchResult]:
        """Get the highest scoring result"""
        return self.results[0] if self.results else None
    
    def get_books_represented(self) -> List[str]:
        """Get unique list of books in results"""
        return list(set(r.book_title for r in self.results))

class IngestionStatus(BaseModel):
    """
    Tracks the status of book ingestion.
    
    This helps monitor long-running ingestion processes
    and provides feedback on progress.
    """
    book_id: str
    status: str = Field(description="current, completed, failed")
    progress_percent: float = Field(default=0.0)
    current_stage: str = Field(default="initializing")
    
    # Detailed progress
    total_pages: Optional[int] = None
    processed_pages: int = 0
    total_chunks: int = 0
    embedded_chunks: int = 0
    
    # Timing
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    # Error handling
    error_message: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    
    def update_progress(self):
        """Calculate progress percentage"""
        if self.total_pages and self.total_pages > 0:
            self.progress_percent = (self.processed_pages / self.total_pages) * 100

# Example usage demonstrating the models
if __name__ == "__main__":
    # Create a book
    book = Book(
        id="978-0-123456-78-9",
        title="Algorithmic Trading with Python",
        author="John Doe",
        file_path="/data/books/algo_trading.pdf",
        file_type=FileType.PDF,
        categories=["trading", "python", "finance"]
    )
    print(f"Book created: {book.title}")
    
    # Create a chunk
    chunk = Chunk(
        book_id=book.id,
        chunk_index=0,
        text="Moving averages are fundamental indicators in algorithmic trading...",
        chapter="Chapter 3: Technical Indicators",
        page_start=45
    )
    print(f"Chunk ID: {chunk.id}, Size: {chunk.get_size()} chars")
EOF
```

#### Step 2.2: Create Storage Interfaces

Now let's define interfaces for our storage systems. These act as contracts that our concrete implementations will follow.

```python
# Create src/core/interfaces.py
cat > src/core/interfaces.py << 'EOF'
"""
Storage interfaces for TradeKnowledge

These abstract base classes define the contracts that our storage
implementations must follow. This allows us to swap implementations
without changing the rest of the code.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime

from core.models import Book, Chunk, SearchResult, SearchResponse

class BookStorageInterface(ABC):
    """
    Interface for book metadata storage.
    
    Any class that implements this interface can be used
    to store and retrieve book information.
    """
    
    @abstractmethod
    async def save_book(self, book: Book) -> bool:
        """Save a book's metadata"""
        pass
    
    @abstractmethod
    async def get_book(self, book_id: str) -> Optional[Book]:
        """Retrieve a book by ID"""
        pass
    
    @abstractmethod
    async def get_book_by_hash(self, file_hash: str) -> Optional[Book]:
        """Retrieve a book by file hash (for deduplication)"""
        pass
    
    @abstractmethod
    async def list_books(self, 
                        category: Optional[str] = None,
                        limit: int = 100,
                        offset: int = 0) -> List[Book]:
        """List books with optional filtering"""
        pass
    
    @abstractmethod
    async def update_book(self, book: Book) -> bool:
        """Update book metadata"""
        pass
    
    @abstractmethod
    async def delete_book(self, book_id: str) -> bool:
        """Delete a book and all its chunks"""
        pass

class ChunkStorageInterface(ABC):
    """
    Interface for chunk storage.
    
    This handles both the full text storage (for exact search)
    and metadata about chunks.
    """
    
    @abstractmethod
    async def save_chunks(self, chunks: List[Chunk]) -> bool:
        """Save multiple chunks efficiently"""
        pass
    
    @abstractmethod
    async def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Retrieve a single chunk"""
        pass
    
    @abstractmethod
    async def get_chunks_by_book(self, book_id: str) -> List[Chunk]:
        """Get all chunks for a book"""
        pass
    
    @abstractmethod
    async def get_chunk_context(self, 
                               chunk_id: str,
                               before: int = 1,
                               after: int = 1) -> Dict[str, Any]:
        """Get a chunk with surrounding context"""
        pass
    
    @abstractmethod
    async def search_exact(self,
                          query: str,
                          book_ids: Optional[List[str]] = None,
                          limit: int = 10) -> List[Dict[str, Any]]:
        """Perform exact text search"""
        pass
    
    @abstractmethod
    async def delete_chunks_by_book(self, book_id: str) -> bool:
        """Delete all chunks for a book"""
        pass

class VectorStorageInterface(ABC):
    """
    Interface for vector/embedding storage.
    
    This handles semantic search capabilities using
    vector embeddings.
    """
    
    @abstractmethod
    async def save_embeddings(self, 
                             chunks: List[Chunk],
                             embeddings: List[List[float]]) -> bool:
        """Save chunk embeddings"""
        pass
    
    @abstractmethod
    async def search_semantic(self,
                             query_embedding: List[float],
                             filter_dict: Optional[Dict[str, Any]] = None,
                             limit: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic search"""
        pass
    
    @abstractmethod
    async def delete_embeddings(self, chunk_ids: List[str]) -> bool:
        """Delete embeddings by chunk IDs"""
        pass
    
    @abstractmethod
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection"""
        pass

class CacheInterface(ABC):
    """
    Interface for caching frequently accessed data.
    
    This improves performance by storing recent search results
    and frequently accessed chunks.
    """
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete from cache"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear entire cache"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass
EOF
```

### Implement Basic PDF Parser

Now let's build our first concrete functionality - a PDF parser that can extract text from clean PDF files.

#### Step 3.1: Create PDF Parser Implementation

```python
# Create src/ingestion/pdf_parser.py
cat > src/ingestion/pdf_parser.py << 'EOF'
"""
PDF Parser for TradeKnowledge

This module handles extraction of text and metadata from PDF files.
We start with simple PyPDF2 for clean PDFs, and will add OCR support later.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re
from datetime import datetime

import PyPDF2
import pdfplumber
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError

from core.models import Book, FileType

logger = logging.getLogger(__name__)

class PDFParser:
    """
    Parses PDF files and extracts text content.
    
    This class handles the complexity of PDF parsing, including:
    - Text extraction from clean PDFs
    - Metadata extraction
    - Page-by-page processing
    - Error handling for corrupted PDFs
    """
    
    def __init__(self):
        """Initialize the PDF parser"""
        self.supported_extensions = ['.pdf']
        
    def can_parse(self, file_path: Path) -> bool:
        """
        Check if this parser can handle the file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file is a PDF
        """
        return file_path.suffix.lower() in self.supported_extensions
    
    def parse_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse a PDF file and extract all content.
        
        This is the main entry point for PDF parsing. It orchestrates
        the extraction of metadata and text content.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary containing:
                - metadata: Book metadata
                - pages: List of page contents
                - errors: Any errors encountered
        """
        logger.info(f"Starting to parse PDF: {file_path}")
        
        result = {
            'metadata': {},
            'pages': [],
            'errors': []
        }
        
        # Verify file exists
        if not file_path.exists():
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            result['errors'].append(error_msg)
            return result
        
        # Try PyPDF2 first (faster for clean PDFs)
        try:
            logger.debug("Attempting PyPDF2 extraction")
            metadata, pages = self._parse_with_pypdf2(file_path)
            result['metadata'] = metadata
            result['pages'] = pages
            
            # If PyPDF2 extraction was poor, try pdfplumber
            if self._is_extraction_poor(pages):
                logger.info("PyPDF2 extraction poor, trying pdfplumber")
                metadatas.append(metadata)
            
            # Add to collection in batches
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_docs = documents[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                batch_metadata = metadatas[i:i + batch_size]
                
                # Use asyncio to avoid blocking
                await asyncio.to_thread(
                    self.collection.add,
                    ids=batch_ids,
                    documents=batch_docs,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadata
                )
                
                logger.debug(f"Added batch {i//batch_size + 1} ({len(batch_ids)} chunks)")
            
            logger.info(f"Successfully saved {len(chunks)} embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            return False
    
    async def search_semantic(self,
                             query_embedding: List[float],
                             filter_dict: Optional[Dict[str, Any]] = None,
                             limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform semantic search using vector similarity.
        
        Args:
            query_embedding: Vector embedding of the query
            filter_dict: Metadata filters (e.g., {'book_id': 'xyz'})
            limit: Maximum number of results
            
        Returns:
            List of search results with chunks and scores
        """
        try:
            # Build where clause for filtering
            where = None
            if filter_dict:
                # ChromaDB expects specific filter format
                where = {}
                if 'book_ids' in filter_dict and filter_dict['book_ids']:
                    where['book_id'] = {'$in': filter_dict['book_ids']}
                if 'chunk_type' in filter_dict:
                    where['chunk_type'] = filter_dict['chunk_type']
            
            # Perform search
            results = await asyncio.to_thread(
                self.collection.query,
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            search_results = []
            if results['ids'] and results['ids'][0]:
                for i, chunk_id in enumerate(results['ids'][0]):
                    # Convert distance to similarity score (1 - normalized_distance)
                    # ChromaDB uses L2 distance by default
                    distance = results['distances'][0][i]
                    score = 1 / (1 + distance)  # Convert distance to similarity
                    
                    search_results.append({
                        'chunk_id': chunk_id,
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'score': score,
                        'distance': distance
                    })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    async def delete_embeddings(self, chunk_ids: List[str]) -> bool:
        """Delete embeddings by chunk IDs"""
        if not chunk_ids:
            return True
        
        try:
            await asyncio.to_thread(
                self.collection.delete,
                ids=chunk_ids
            )
            logger.info(f"Deleted {len(chunk_ids)} embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting embeddings: {e}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection"""
        try:
            count = self.collection.count()
            
            # Get collection metadata
            metadata = self.collection.metadata or {}
            
            return {
                'collection_name': self.collection_name,
                'total_embeddings': count,
                'persist_directory': self.persist_directory,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {
                'collection_name': self.collection_name,
                'error': str(e)
            }

# Test ChromaDB storage
async def test_chroma_storage():
    """Test ChromaDB storage implementation"""
    storage = ChromaDBStorage()
    
    # Get stats
    stats = await storage.get_collection_stats()
    print(f"Collection stats: {stats}")
    
    # Create test data
    test_chunks = [
        Chunk(
            id=f"test_chunk_{i}",
            book_id="test_book",
            chunk_index=i,
            text=f"Test chunk {i} about trading strategies"
        )
        for i in range(3)
    ]
    
    # Create fake embeddings (normally from embedding generator)
    import random
    test_embeddings = [
        [random.random() for _ in range(384)]  # 384-dim embeddings
        for _ in test_chunks
    ]
    
    # Save embeddings
    success = await storage.save_embeddings(test_chunks, test_embeddings)
    print(f"Save embeddings: {success}")
    
    # Test search
    query_embedding = [random.random() for _ in range(384)]
    results = await storage.search_semantic(query_embedding, limit=2)
    
    print(f"\nSearch results ({len(results)} found):")
    for result in results:
        print(f"  - ID: {result['chunk_id']}")
        print(f"    Score: {result['score']:.3f}")
        print(f"    Text: {result['text'][:50]}...")

if __name__ == "__main__":
    asyncio.run(test_chroma_storage())
EOF
```

### Implement Basic Search Engine

Now let's combine everything into a working search engine.

```python
# Create src/search/hybrid_search.py
cat > src/search/hybrid_search.py << 'EOF'
"""
Hybrid search engine combining semantic and exact search

This is where the magic happens - we combine vector similarity
with traditional text search for the best results.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import time

from core.config import Config, get_config
from core.models import SearchResult, SearchResponse, Chunk
from core.sqlite_storage import SQLiteStorage
from core.chroma_storage import ChromaDBStorage
from ingestion.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)

class HybridSearch:
    """
    Hybrid search engine combining semantic and exact search.
    
    This class orchestrates:
    - Semantic search through ChromaDB
    - Exact text search through SQLite FTS5
    - Result merging and ranking
    - Context retrieval
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize search engine"""
        self.config = config or get_config()
        
        # Storage backends
        self.sqlite_storage: Optional[SQLiteStorage] = None
        self.chroma_storage: Optional[ChromaDBStorage] = None
        
        # Embedding generator
        self.embedding_generator: Optional[EmbeddingGenerator] = None
        
        # Search statistics
        self.search_count = 0
        self.total_search_time = 0
    
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing hybrid search engine...")
        
        # Initialize storage
        self.sqlite_storage = SQLiteStorage()
        self.chroma_storage = ChromaDBStorage()
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator()
        
        # Load embedding cache if available
        cache_path = "data/embeddings/cache.json"
        self.embedding_generator.load_cache(cache_path)
        
        logger.info("Search engine initialized")
    
    async def cleanup(self):
        """Cleanup resources"""
        # Save embedding cache
        if self.embedding_generator:
            self.embedding_generator.save_cache("data/embeddings/cache.json")
    
    async def search_semantic(self,
                            query: str,
                            num_results: int = 10,
                            filter_books: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform semantic search only.
        
        This searches based on meaning similarity, finding content
        that's conceptually related even if different words are used.
        """
        start_time = time.time()
        
        try:
            # Generate query embedding
            logger.debug(f"Generating embedding for query: {query[:50]}...")
            query_embedding = await self.embedding_generator.generate_query_embedding(query)
            
            # Build filter
            filter_dict = {}
            if filter_books:
                filter_dict['book_ids'] = filter_books
            
            # Search in ChromaDB
            logger.debug("Searching in vector database...")
            results = await self.chroma_storage.search_semantic(
                query_embedding=query_embedding,
                filter_dict=filter_dict,
                limit=num_results
            )
            
            # Convert to SearchResponse
            search_results = []
            for result in results:
                # Get full chunk data
                chunk = await self.sqlite_storage.get_chunk(result['chunk_id'])
                if not chunk:
                    continue
                
                # Get book info
                book = await self.sqlite_storage.get_book(chunk.book_id)
                if not book:
                    continue
                
                search_result = SearchResult(
                    chunk=chunk,
                    score=result['score'],
                    match_type='semantic',
                    highlights=[self._extract_highlight(chunk.text, query)],
                    book_title=book.title,
                    book_author=book.author,
                    chapter=result['metadata'].get('chapter'),
                    page=result['metadata'].get('page_start')
                )
                
                search_results.append(search_result)
            
            # Build response
            search_time = int((time.time() - start_time) * 1000)
            
            response = SearchResponse(
                query=query,
                results=search_results,
                total_results=len(results),
                returned_results=len(search_results),
                search_time_ms=search_time,
                search_type='semantic',
                filters_applied={'book_ids': filter_books} if filter_books else {}
            )
            
            # Update statistics
            self.search_count += 1
            self.total_search_time += search_time
            
            return response.dict()
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                returned_results=0,
                search_time_ms=int((time.time() - start_time) * 1000),
                search_type='semantic'
            ).dict()
    
    async def search_exact(self,
                          query: str,
                          num_results: int = 10,
                          filter_books: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform exact text search.
        
        This finds exact matches of words or phrases,
        useful for finding specific terms or code snippets.
        """
        start_time = time.time()
        
        try:
            # Search in SQLite FTS
            logger.debug(f"Performing exact search for: {query}")
            results = await self.sqlite_storage.search_exact(
                query=query,
                book_ids=filter_books,
                limit=num_results
            )
            
            # Convert to SearchResponse
            search_results = []
            for result in results:
                chunk = result['chunk']
                
                # Get book info
                book = await self.sqlite_storage.get_book(chunk.book_id)
                if not book:
                    continue
                
                search_result = SearchResult(
                    chunk=chunk,
                    score=result['score'],
                    match_type='exact',
                    highlights=[result.get('snippet', '')],
                    book_title=book.title,
                    book_author=book.author,
                    chapter=chunk.metadata.get('chapter'),
                    page=chunk.page_start
                )
                
                search_results.append(search_result)
            
            # Build response
            search_time = int((time.time() - start_time) * 1000)
            
            response = SearchResponse(
                query=query,
                results=search_results,
                total_results=len(results),
                returned_results=len(search_results),
                search_time_ms=search_time,
                search_type='exact',
                filters_applied={'book_ids': filter_books} if filter_books else {}
            )
            
            return response.dict()
            
        except Exception as e:
            logger.error(f"Error in exact search: {e}")
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                returned_results=0,
                search_time_ms=int((time.time() - start_time) * 1000),
                search_type='exact'
            ).dict()
    
    async def search_hybrid(self,
                           query: str,
                           num_results: int = 10,
                           filter_books: Optional[List[str]] = None,
                           semantic_weight: float = 0.7) -> Dict[str, Any]:
        """
        Perform hybrid search combining semantic and exact.
        
        This is our secret sauce - we run both searches and
        intelligently combine the results for best relevance.
        
        Args:
            query: Search query
            num_results: Number of results to return
            filter_books: Optional book IDs to search within
            semantic_weight: Weight for semantic results (0-1)
        """
        start_time = time.time()
        
        try:
            # Run both searches in parallel
            logger.debug(f"Running hybrid search for: {query}")
            
            semantic_task = self.search_semantic(query, num_results * 2, filter_books)
            exact_task = self.search_exact(query, num_results * 2, filter_books)
            
            semantic_response, exact_response = await asyncio.gather(
                semantic_task, exact_task
            )
            
            # Merge results
            merged_results = self._merge_results(
                semantic_response['results'],
                exact_response['results'],
                semantic_weight
            )
            
            # Take top N results
            final_results = merged_results[:num_results]
            
            # Build response
            search_time = int((time.time() - start_time) * 1000)
            
            response = SearchResponse(
                query=query,
                results=final_results,
                total_results=len(merged_results),
                returned_results=len(final_results),
                search_time_ms=search_time,
                search_type='hybrid',
                filters_applied={
                    'book_ids': filter_books,
                    'semantic_weight': semantic_weight
                } if filter_books else {'semantic_weight': semantic_weight}
            )
            
            return response.dict()
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                returned_results=0,
                search_time_ms=int((time.time() - start_time) * 1000),
                search_type='hybrid'
            ).dict()
    
    async def get_chunk_context(self,
                               chunk_id: str,
                               before_chunks: int = 1,
                               after_chunks: int = 1) -> Dict[str, Any]:
        """
        Get expanded context for a chunk.
        
        This is useful for showing more context around
        a search result when the user wants to see more.
        """
        try:
            context = await self.sqlite_storage.get_chunk_context(
                chunk_id=chunk_id,
                before=before_chunks,
                after=after_chunks
            )
            
            if not context:
                return {'error': 'Chunk not found'}
            
            # Format response
            response = {
                'chunk_id': chunk_id,
                'chunk': context['chunk'].dict() if context.get('chunk') else None,
                'context': {
                    'before': [c.dict() for c in context.get('before', [])],
                    'after': [c.dict() for c in context.get('after', [])]
                }
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting chunk context: {e}")
            return {'error': str(e)}
    
    def _merge_results(self,
                      semantic_results: List[Dict],
                      exact_results: List[Dict],
                      semantic_weight: float) -> List[SearchResult]:
        """
        Merge and re-rank results from both search types.
        
        This is a simple weighted combination, but could be
        made more sophisticated with learning-to-rank models.
        """
        # Create a map of chunk_id to result
        result_map = {}
        
        # Add semantic results
        for result in semantic_results:
            chunk_id = result['chunk']['id']
            result_map[chunk_id] = {
                'result': result,
                'semantic_score': result['score'],
                'exact_score': 0.0
            }
        
        # Add/update with exact results
        for result in exact_results:
            chunk_id = result['chunk']['id']
            if chunk_id in result_map:
                result_map[chunk_id]['exact_score'] = result['score']
            else:
                result_map[chunk_id] = {
                    'result': result,
                    'semantic_score': 0.0,
                    'exact_score': result['score']
                }
        
        # Calculate combined scores
        exact_weight = 1 - semantic_weight
        for chunk_id, data in result_map.items():
            # Normalize scores to 0-1 range
            semantic_score = min(data['semantic_score'], 1.0)
            exact_score = min(data['exact_score'], 1.0)
            
            # Calculate weighted score
            combined_score = (
                semantic_score * semantic_weight +
                exact_score * exact_weight
            )
            
            # Update the result
            data['result']['score'] = combined_score
            data['result']['match_type'] = 'hybrid'
        
        # Sort by combined score
        sorted_results = sorted(
            result_map.values(),
            key=lambda x: x['result']['score'],
            reverse=True
        )
        
        # Return just the result objects
        return [item['result'] for item in sorted_results]
    
    def _extract_highlight(self, text: str, query: str, context_length: int = 100) -> str:
        """
        Extract a relevant highlight from the text.
        
        This finds the most relevant snippet to show in search results.
        """
        # Simple implementation - find first occurrence
        query_lower = query.lower()
        text_lower = text.lower()
        
        pos = text_lower.find(query_lower)
        if pos == -1:
            # Query not found, return beginning
            return text[:context_length * 2] + '...' if len(text) > context_length * 2 else text
        
        # Extract context around match
        start = max(0, pos - context_length)
        end = min(len(text), pos + len(query) + context_length)
        
        highlight = text[start:end]
        
        # Add ellipsis if needed
        if start > 0:
            highlight = '...' + highlight
        if end < len(text):
            highlight = highlight + '...'
        
        return highlight

# Test the search engine
async def test_search_engine():
    """Test the hybrid search engine"""
    
    # Initialize
    search_engine = HybridSearch()
    await search_engine.initialize()
    
    # Test queries
    test_queries = [
        "moving average trading strategy",
        "def calculate_sma",
        "momentum indicators"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        # Semantic search
        print("\nSemantic Search:")
        results = await search_engine.search_semantic(query, num_results=3)
        print(f"Found {results['total_results']} results in {results['search_time_ms']}ms")
        
        # Exact search
        print("\nExact Search:")
        results = await search_engine.search_exact(query, num_results=3)
        print(f"Found {results['total_results']} results in {results['search_time_ms']}ms")
        
        # Hybrid search
        print("\nHybrid Search:")
        results = await search_engine.search_hybrid(query, num_results=3)
        print(f"Found {results['total_results']} results in {results['search_time_ms']}ms")
        
        if results['results']:
            print("\nTop result:")
            top = results['results'][0]
            print(f"Book: {top['book_title']}")
            print(f"Score: {top['score']:.3f}")
            print(f"Preview: {top['highlights'][0] if top['highlights'] else 'N/A'}")
    
    # Cleanup
    await search_engine.cleanup()

if __name__ == "__main__":
    asyncio.run(test_search_engine())
EOF
```

### Integration and Testing

Let's create a complete ingestion pipeline that ties everything together.

```python
# Create src/ingestion/book_processor.py
cat > src/ingestion/book_processor.py << 'EOF'
"""
Book processing pipeline for TradeKnowledge

This orchestrates the entire process of ingesting a book:
1. Parse PDF/EPUB
2. Chunk the text
3. Generate embeddings  
4. Store everything
"""

import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib

from core.config import Config, get_config
from core.models import Book, Chunk, FileType, IngestionStatus
from core.sqlite_storage import SQLiteStorage
from core.chroma_storage import ChromaDBStorage
from ingestion.pdf_parser import PDFParser
from ingestion.text_chunker import TextChunker, ChunkingConfig
from ingestion.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)

class BookProcessor:
    """
    Orchestrates the book ingestion pipeline.
    
    This class coordinates all the steps needed to ingest
    a book into our knowledge system.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize book processor"""
        self.config = config or get_config()
        
        # Components
        self.pdf_parser = PDFParser()
        self.text_chunker = TextChunker(
            ChunkingConfig(
                chunk_size=self.config.ingestion.chunk_size,
                chunk_overlap=self.config.ingestion.chunk_overlap,
                min_chunk_size=self.config.ingestion.min_chunk_size,
                max_chunk_size=self.config.ingestion.max_chunk_size
            )
        )
        self.embedding_generator: Optional[EmbeddingGenerator] = None
        self.sqlite_storage: Optional[SQLiteStorage] = None
        self.chroma_storage: Optional[ChromaDBStorage] = None
        
        # Processing state
        self.current_status: Optional[IngestionStatus] = None
    
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing book processor...")
        
        # Initialize storage
        self.sqlite_storage = SQLiteStorage()
        self.chroma_storage = ChromaDBStorage()
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator()
        
        logger.info("Book processor initialized")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.embedding_generator:
            self.embedding_generator.save_cache("data/embeddings/cache.json")
    
    async def add_book(self,
                      file_path: str,
                      metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add a book to the knowledge base.
        
        This is the main entry point for ingesting books.
        It handles the entire pipeline from parsing to storage.
        
        Args:
            file_path: Path to the book file
            metadata: Optional metadata about the book
            
        Returns:
            Dictionary with ingestion results
        """
        path = Path(file_path)
        
        # Validate file
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return {'success': False, 'error': 'File not found'}
        
        if not path.suffix.lower() in ['.pdf', '.epub']:
            logger.error(f"Unsupported file type: {path.suffix}")
            return {'success': False, 'error': 'Unsupported file type'}
        
        # Calculate file hash for deduplication
        file_hash = self._calculate_file_hash(path)
        
        # Check if already processed
        existing_book = await self.sqlite_storage.get_book_by_hash(file_hash)
        if existing_book:
            logger.info(f"Book already exists: {existing_book.title}")
            return {
                'success': False,
                'error': 'Book already processed',
                'book_id': existing_book.id
            }
        
        # Start processing
        logger.info(f"Starting to process: {path.name}")
        
        try:
            # Step 1: Parse the file
            logger.info("Step 1: Parsing file...")
            parse_result = await self._parse_file(path)
            
            if parse_result['errors']:
                logger.error(f"Parse errors: {parse_result['errors']}")
                return {
                    'success': False,
                    'error': 'Failed to parse file',
                    'details': parse_result['errors']
                }
            
            # Step 2: Create book record
            logger.info("Step 2: Creating book record...")
            book = await self._create_book_record(
                path, file_hash, parse_result, metadata
            )
            
            # Initialize status tracking
            self.current_status = IngestionStatus(
                book_id=book.id,
                status='processing',
                total_pages=len(parse_result['pages'])
            )
            
            # Save book to database
            await self.sqlite_storage.save_book(book)
            
            # Step 3: Chunk the text
            logger.info("Step 3: Chunking text...")
            chunks = await self._chunk_book(parse_result['pages'], book.id)
            
            self.current_status.total_chunks = len(chunks)
            self.current_status.current_stage = 'chunking'
            
            # Step 4: Generate embeddings
            logger.info("Step 4: Generating embeddings...")
            self.current_status.current_stage = 'embedding'
            
            embeddings = await self.embedding_generator.generate_embeddings(chunks)
            
            # Step 5: Store everything
            logger.info("Step 5: Storing data...")
            self.current_status.current_stage = 'storing'
            
            # Store chunks in SQLite
            await self.sqlite_storage.save_chunks(chunks)
            
            # Store embeddings in ChromaDB
            success = await self.chroma_storage.save_embeddings(chunks, embeddings)
            
            if not success:
                logger.error("Failed to save embeddings")
                return {
                    'success': False,
                    'error': 'Failed to save embeddings'
                }
            
            # Update book record
            book.total_chunks = len(chunks)
            book.indexed_at = datetime.now()
            await self.sqlite_storage.update_book(book)
            
            # Complete!
            self.current_status.status = 'completed'
            self.current_status.completed_at = datetime.now()
            self.current_status.progress_percent = 100.0
            
            logger.info(f"Successfully processed book: {book.title}")
            
            return {
                'success': True,
                'book_id': book.id,
                'title': book.title,
                'chunks_created': len(chunks),
                'processing_time': (
                    self.current_status.completed_at - self.current_status.started_at
                ).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Error processing book: {e}", exc_info=True)
            
            if self.current_status:
                self.current_status.status = 'failed'
                self.current_status.error_message = str(e)
            
            return {
                'success': False,
                'error': f'Processing failed: {str(e)}'
            }
    
    async def list_books(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all books in the system"""
        books = await self.sqlite_storage.list_books(category=category)
        
        return [
            {
                'id': book.id,
                'title': book.title,
                'author': book.author,
                'total_chunks': book.total_chunks,
                'categories': book.categories,
                'indexed_at': book.indexed_at.isoformat() if book.indexed_at else None
            }
            for book in books
        ]
    
    async def get_ingestion_status(self) -> Optional[Dict[str, Any]]:
        """Get current ingestion status"""
        if not self.current_status:
            return None
        
        return self.current_status.dict()
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    async def _parse_file(self, file_path: Path) -> Dict[str, Any]:
        """Parse file based on type"""
        if file_path.suffix.lower() == '.pdf':
            # Run in thread to avoid blocking
            return await asyncio.to_thread(
                self.pdf_parser.parse_file, file_path
            )
        else:
            # TODO: Add EPUB parser
            raise NotImplementedError(f"Parser for {file_path.suffix} not implemented")
    
    async def _create_book_record(self,
                                 file_path: Path,
                                 file_hash: str,
                                 parse_result: Dict[str, Any],
                                 metadata: Optional[Dict[str, Any]]) -> Book:
        """Create book record from parse results"""
        book_metadata = parse_result['metadata']
        
        # Generate book ID (use ISBN if available)
        book_id = book_metadata.get('isbn')
        if not book_id:
            # Generate from title and author
            title = book_metadata.get('title', file_path.stem)
            author = book_metadata.get('author', 'Unknown')
            book_id = f"{title[:20]}_{author[:20]}_{file_hash[:8]}".replace(' ', '_')
        
        # Merge metadata
        if metadata:
            book_metadata.update(metadata)
        
        # Add statistics
        book_metadata['statistics'] = parse_result.get('statistics', {})
        
        # Create book object
        book = Book(
            id=book_id,
            title=book_metadata.get('title', file_path.stem),
            author=book_metadata.get('author'),
            isbn=book_metadata.get('isbn'),
            file_path=str(file_path),
            file_type=FileType.PDF if file_path.suffix.lower() == '.pdf' else FileType.EPUB,
            file_hash=file_hash,
            total_pages=book_metadata.get('total_pages', 0),
            categories=metadata.get('categories', []) if metadata else [],
            metadata=book_metadata
        )
        
        return book
    
    async def _chunk_book(self,
                         pages: List[Dict[str, Any]],
                         book_id: str) -> List[Chunk]:
        """Chunk book pages"""
        # Update status
        if self.current_status:
            self.current_status.processed_pages = 0
        
        # Use page-aware chunking
        chunks = await asyncio.to_thread(
            self.text_chunker.chunk_pages,
            pages,
            book_id,
            {}
        )
        
        # Update chunk IDs for vector storage
        for chunk in chunks:
            chunk.embedding_id = chunk.id
        
        return chunks

# Example usage
async def process_sample_book():
    """Process a sample book"""
    
    # Initialize processor
    processor = BookProcessor()
    await processor.initialize()
    
    # Add a book
    result = await processor.add_book(
        "data/books/sample_trading_book.pdf",
        metadata={
            'categories': ['trading', 'technical-analysis'],
            'difficulty': 'intermediate'
        }
    )
    
    print(f"Processing result: {result}")
    
    # List books
    books = await processor.list_books()
    print(f"\nTotal books: {len(books)}")
    
    for book in books:
        print(f"  - {book['title']} ({book['total_chunks']} chunks)")
    
    # Cleanup
    await processor.cleanup()

if __name__ == "__main__":
    asyncio.run(process_sample_book())
EOF
```

### Phase 1 Final Verification

Create a comprehensive test to ensure everything works together:

```bash
# Create scripts/test_phase1_complete.py
cat > scripts/test_phase1_complete.py << 'EOF'
#!/usr/bin/env python3
"""
Complete end-to-end test of Phase 1 implementation
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import asyncio
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_complete_pipeline():
    """Test the complete ingestion and search pipeline"""
    
    from ingestion.book_processor import BookProcessor
    from search.hybrid_search import HybridSearch
    
    try:
        # Step 1: Create a test PDF
        logger.info("Step 1: Creating test content...")
        test_content = create_test_pdf()
        
        # Step 2: Initialize processor
        logger.info("Step 2: Initializing book processor...")
        processor = BookProcessor()
        await processor.initialize()
        
        # Step 3: Process the test book
        logger.info("Step 3: Processing test book...")
        result = await processor.add_book(
            test_content,
            metadata={'categories': ['test']}
        )
        
        if not result['success']:
            logger.error(f"Failed to process book: {result}")
            return False
        
        logger.info(f"✅ Book processed: {result['chunks_created']} chunks created")
        
        # Step 4: Initialize search engine
        logger.info("Step 4: Initializing search engine...")
        search_engine = HybridSearch()
        await search_engine.initialize()
        
        # Step 5: Test searches
        logger.info("Step 5: Testing search functionality...")
        
        test_queries = [
            "moving average",
            "calculate_sma",
            "trading strategy"
        ]
        
        for query in test_queries:
            logger.info(f"\nSearching for: '{query}'")
            
            # Test semantic search
            results = await search_engine.search_semantic(query, num_results=3)
            logger.info(f"  Semantic: {results['total_results']} results")
            
            # Test exact search
            results = await search_engine.search_exact(query, num_results=3)
            logger.info(f"  Exact: {results['total_results']} results")
            
            # Test hybrid search
            results = await search_engine.search_hybrid(query, num_results=3)
            logger.info(f"  Hybrid: {results['total_results']} results")
            
            if results['results']:
                top_result = results['results'][0]
                logger.info(f"  Top result score: {top_result['score']:.3f}")
        
        # Cleanup
        await processor.cleanup()
        await search_engine.cleanup()
        
        logger.info("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False

def create_test_pdf():
    """Create a test PDF file with sample content"""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    
    test_file = Path("data/books/test_phase1.pdf")
    test_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create PDF
    c = canvas.Canvas(str(test_file), pagesize=letter)
    
    # Page 1
    c.drawString(100, 750, "Chapter 1: Introduction to Algorithmic Trading")
    c.drawString(100, 700, "")
    c.drawString(100, 680, "Algorithmic trading uses computer programs to execute trades.")
    c.drawString(100, 660, "One common strategy is the moving average crossover.")
    c.drawString(100, 640, "")
    c.drawString(100, 620, "def calculate_sma(prices, period):")
    c.drawString(100, 600, "    return sum(prices[-period:]) / period")
    
    # Page 2
    c.showPage()
    c.drawString(100, 750, "Chapter 2: Technical Indicators")
    c.drawString(100, 700, "")
    c.drawString(100, 680, "Moving averages smooth out price action to identify trends.")
    c.drawString(100, 660, "The 50-day and 200-day moving averages are commonly used.")
    c.drawString(100, 640, "A bullish signal occurs when the 50-day crosses above the 200-day.")
    
    c.save()
    
    logger.info(f"Created test PDF: {test_file}")
    return str(test_file)

async def main():
    """Run all Phase 1 tests"""
    print("=" * 60)
    print("PHASE 1 COMPLETE TEST")
    print(f"Started at: {datetime.now()}")
    print("=" * 60)
    
    success = await test_complete_pipeline()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ PHASE 1 IMPLEMENTATION COMPLETE!")
        print("All components are working correctly.")
        print("\nYou can now proceed to Phase 2.")
    else:
        print("❌ PHASE 1 TESTS FAILED!")
        print("Please fix the issues before proceeding.")
    
    return 0 if success else 1

if __name__ == "__main__":
    # Install reportlab if needed
    try:
        import reportlab
    except ImportError:
        print("Installing reportlab for PDF creation...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
    
    sys.exit(asyncio.run(main()))
EOF

chmod +x scripts/test_phase1_complete.py
```

---

## Phase 1 Summary

### What We've Built

In Phase 1, we've created the foundation of the TradeKnowledge system:

1. **Core Data Models** - Structured representations of books, chunks, and search results
2. **PDF Parser** - Extracts text and metadata from PDF files
3. **Intelligent Text Chunker** - Breaks text into searchable pieces while preserving context
4. **Embedding Generator** - Converts text to vectors for semantic search
5. **Storage Systems** - SQLite for text/metadata, ChromaDB for vectors
6. **Basic Search Engine** - Semantic, exact, and hybrid search capabilities
7. **Book Processor** - Orchestrates the complete ingestion pipeline

### Key Achievements

- ✅ Clean architecture with interfaces and implementations
- ✅ Async/await throughout for performance
- ✅ Proper error handling and logging
- ✅ Caching for embeddings
- ✅ Support for both OpenAI and local embedding models
- ✅ Full-text search with SQLite FTS5
- ✅ Vector similarity search with ChromaDB
- ✅ Hybrid search combining both approaches

### Testing Your Implementation

Run these commands to verify Phase 1 is complete:

```bash
# 1. Verify environment
python scripts/verify_environment.py

# 2. Initialize database
python scripts/init_db.py

# 3. Run Phase 1 verification
python scripts/verify_phase1.py

# 4. Run complete pipeline test
python scripts/test_phase1_complete.py
```

### Next Steps

Once all tests pass, you're ready for Phase 2 where we'll add:
- Advanced features (OCR, EPUB support)
- Performance optimizations with C++
- Advanced caching strategies
- Query suggestion engine
- And more!

---

**END OF PHASE 1 IMPLEMENTATION GUIDE**data_plumber, pages_plumber = self._parse_with_pdfplumber(file_path)
                
                # Use pdfplumber results if better
                if self._count_words(pages_plumber) > self._count_words(pages) * 1.2:
                    result['pages'] = pages_plumber
                    # Merge metadata, preferring pdfplumber values
                    result['metadata'].update(metadata_plumber)
                    
        except Exception as e:
            error_msg = f"Error parsing PDF: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result['errors'].append(error_msg)
            
            # Try pdfplumber as fallback
            try:
                logger.info("Falling back to pdfplumber")
                metadata, pages = self._parse_with_pdfplumber(file_path)
                result['metadata'] = metadata
                result['pages'] = pages
            except Exception as e2:
                error_msg = f"Pdfplumber also failed: {str(e2)}"
                logger.error(error_msg)
                result['errors'].append(error_msg)
        
        # Post-process results
        result = self._post_process_results(result, file_path)
        
        logger.info(f"Parsed {len(result['pages'])} pages from {file_path.name}")
        return result
    
    def _parse_with_pypdf2(self, file_path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Parse PDF using PyPDF2 library.
        
        PyPDF2 is fast but sometimes struggles with complex layouts.
        We use it as our primary parser for clean PDFs.
        """
        metadata = {}
        pages = []
        
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            
            # Extract metadata
            if reader.metadata:
                metadata = {
                    'title': self._clean_text(reader.metadata.get('/Title', '')),
                    'author': self._clean_text(reader.metadata.get('/Author', '')),
                    'subject': self._clean_text(reader.metadata.get('/Subject', '')),
                    'creator': self._clean_text(reader.metadata.get('/Creator', '')),
                    'producer': self._clean_text(reader.metadata.get('/Producer', '')),
                    'creation_date': self._parse_date(reader.metadata.get('/CreationDate')),
                    'modification_date': self._parse_date(reader.metadata.get('/ModDate')),
                }
            
            # Extract text from each page
            total_pages = len(reader.pages)
            metadata['total_pages'] = total_pages
            
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    text = page.extract_text()
                    
                    # Clean up the text
                    text = self._clean_text(text)
                    
                    pages.append({
                        'page_number': page_num,
                        'text': text,
                        'word_count': len(text.split()),
                        'char_count': len(text)
                    })
                    
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num}: {e}")
                    pages.append({
                        'page_number': page_num,
                        'text': '',
                        'word_count': 0,
                        'char_count': 0,
                        'error': str(e)
                    })
        
        return metadata, pages
    
    def _parse_with_pdfplumber(self, file_path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Parse PDF using pdfplumber library.
        
        Pdfplumber is better at handling complex layouts and tables,
        but is slower than PyPDF2.
        """
        metadata = {}
        pages = []
        
        with pdfplumber.open(file_path) as pdf:
            # Extract metadata
            if pdf.metadata:
                metadata = {
                    'title': self._clean_text(pdf.metadata.get('Title', '')),
                    'author': self._clean_text(pdf.metadata.get('Author', '')),
                    'subject': self._clean_text(pdf.metadata.get('Subject', '')),
                    'creator': self._clean_text(pdf.metadata.get('Creator', '')),
                    'producer': self._clean_text(pdf.metadata.get('Producer', '')),
                }
            
            metadata['total_pages'] = len(pdf.pages)
            
            # Extract text from each page
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    # Extract text
                    text = page.extract_text() or ''
                    
                    # Also try to extract tables
                    tables = page.extract_tables()
                    if tables:
                        # Convert tables to text representation
                        for table in tables:
                            table_text = self._table_to_text(table)
                            text += f"\n\n[TABLE]\n{table_text}\n[/TABLE]\n"
                    
                    # Clean up the text
                    text = self._clean_text(text)
                    
                    pages.append({
                        'page_number': page_num,
                        'text': text,
                        'word_count': len(text.split()),
                        'char_count': len(text),
                        'has_tables': len(tables) > 0 if tables else False
                    })
                    
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num} with pdfplumber: {e}")
                    pages.append({
                        'page_number': page_num,
                        'text': '',
                        'word_count': 0,
                        'char_count': 0,
                        'error': str(e)
                    })
        
        return metadata, pages
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text.
        
        This handles common issues with PDF text extraction:
        - Excessive whitespace
        - Broken words from line breaks
        - Special characters
        - Encoding issues
        """
        if not text:
            return ''
        
        # Handle different types of input
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='ignore')
        
        # Remove null characters
        text = text.replace('\x00', '')
        
        # Fix hyphenated words at line breaks
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # Replace multiple whitespaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _is_extraction_poor(self, pages: List[Dict[str, Any]]) -> bool:
        """
        Check if text extraction quality is poor.
        
        Poor extraction indicators:
        - Very low word count
        - Many pages with no text
        - Suspicious patterns (all caps, no spaces)
        """
        if not pages:
            return True
        
        total_words = sum(p.get('word_count', 0) for p in pages)
        empty_pages = sum(1 for p in pages if p.get('word_count', 0) < 10)
        
        # Average words per page for a typical book
        avg_words = total_words / len(pages) if pages else 0
        
        # Check for poor extraction
        if avg_words < 50:  # Very low word count
            return True
        
        if empty_pages > len(pages) * 0.2:  # >20% empty pages
            return True
        
        # Check for extraction artifacts
        sample_text = ' '.join(p.get('text', '')[:100] for p in pages[:5])
        if sample_text.isupper():  # All uppercase often indicates OCR needed
            return True
        
        return False
    
    def _count_words(self, pages: List[Dict[str, Any]]) -> int:
        """Count total words across all pages"""
        return sum(p.get('word_count', 0) for p in pages)
    
    def _table_to_text(self, table: List[List[Any]]) -> str:
        """
        Convert table data to readable text format.
        
        Tables in PDFs can contain important data for trading strategies,
        so we preserve them in a readable format.
        """
        if not table:
            return ''
        
        lines = []
        for row in table:
            # Filter out None values and convert to strings
            cleaned_row = [str(cell) if cell is not None else '' for cell in row]
            lines.append(' | '.join(cleaned_row))
        
        return '\n'.join(lines)
    
    def _parse_date(self, date_str: Any) -> Optional[str]:
        """Parse PDF date format to ISO format"""
        if not date_str:
            return None
        
        try:
            # PDF dates are often in format: D:20230615120000+00'00'
            if isinstance(date_str, str) and date_str.startswith('D:'):
                date_str = date_str[2:]  # Remove 'D:' prefix
                # Extract just the date portion
                date_part = date_str[:14]
                if len(date_part) >= 8:
                    year = date_part[:4]
                    month = date_part[4:6]
                    day = date_part[6:8]
                    return f"{year}-{month}-{day}"
        except Exception as e:
            logger.debug(f"Could not parse date {date_str}: {e}")
        
        return str(date_str) if date_str else None
    
    def _post_process_results(self, result: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
        """
        Post-process extraction results.
        
        This adds additional metadata and cleans up the results.
        """
        # Add file information
        result['metadata']['file_name'] = file_path.name
        result['metadata']['file_size'] = file_path.stat().st_size
        
        # If no title found in metadata, use filename
        if not result['metadata'].get('title'):
            # Extract title from filename
            title = file_path.stem
            # Replace underscores and hyphens with spaces
            title = title.replace('_', ' ').replace('-', ' ')
            # Title case
            title = title.title()
            result['metadata']['title'] = title
        
        # Calculate total statistics
        total_words = sum(p.get('word_count', 0) for p in result['pages'])
        total_chars = sum(p.get('char_count', 0) for p in result['pages'])
        
        result['statistics'] = {
            'total_pages': len(result['pages']),
            'total_words': total_words,
            'total_characters': total_chars,
            'average_words_per_page': total_words / len(result['pages']) if result['pages'] else 0
        }
        
        return result

# Standalone function for testing
def test_parser():
    """Test the PDF parser with a sample file"""
    parser = PDFParser()
    
    # Create a test PDF path (you'll need to provide a real PDF)
    test_file = Path("data/books/sample.pdf")
    
    if test_file.exists():
        result = parser.parse_file(test_file)
        
        print(f"Title: {result['metadata'].get('title', 'Unknown')}")
        print(f"Pages: {result['statistics']['total_pages']}")
        print(f"Words: {result['statistics']['total_words']}")
        
        # Show first page sample
        if result['pages']:
            first_page = result['pages'][0]
            sample = first_page['text'][:200] + '...' if len(first_page['text']) > 200 else first_page['text']
            print(f"\nFirst page sample:\n{sample}")
    else:
        print(f"Test file not found: {test_file}")
        print("Please add a PDF file to test with")

if __name__ == "__main__":
    # Setup logging for testing
    logging.basicConfig(level=logging.DEBUG)
    test_parser()
EOF
```

### Implement Text Chunking

Text chunking is crucial - it determines how we break books into searchable pieces. Too small and we lose context, too large and search becomes imprecise.

#### Create Intelligent Text Chunker

```python
# Create src/ingestion/text_chunker.py
cat > src/ingestion/text_chunker.py << 'EOF'
"""
Intelligent Text Chunking for TradeKnowledge

This module breaks text into optimal chunks for searching and embedding.
The key challenge is maintaining context while keeping chunks at a reasonable size.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from core.models import Chunk, ChunkType

logger = logging.getLogger(__name__)

@dataclass
class ChunkingConfig:
    """Configuration for chunking behavior"""
    chunk_size: int = 1000  # Target size in characters
    chunk_overlap: int = 200  # Overlap between chunks
    min_chunk_size: int = 100  # Minimum viable chunk
    max_chunk_size: int = 2000  # Maximum chunk size
    respect_sentences: bool = True  # Try to break at sentence boundaries
    respect_paragraphs: bool = True  # Try to break at paragraph boundaries
    preserve_code_blocks: bool = True  # Don't split code blocks

class TextChunker:
    """
    Intelligently chunks text for optimal search and retrieval.
    
    This class handles the complexity of breaking text into chunks that:
    1. Maintain semantic coherence
    2. Preserve context through overlap
    3. Respect natural boundaries (sentences, paragraphs)
    4. Handle special content (code, formulas) appropriately
    """
    
    def __init__(self, config: ChunkingConfig = None):
        """Initialize chunker with configuration"""
        self.config = config or ChunkingConfig()
        
        # Compile regex patterns for efficiency
        self.sentence_end_pattern = re.compile(r'[.!?]\s+')
        self.paragraph_pattern = re.compile(r'\n\s*\n')
        self.code_block_pattern = re.compile(
            r'```[\s\S]*?```|`[^`]+`',
            re.MULTILINE
        )
        self.formula_pattern = re.compile(
            r'\$\$[\s\S]*?\$\$|\$[^\$]+\$',
            re.MULTILINE
        )
        
    def chunk_text(self, 
                   text: str, 
                   book_id: str,
                   metadata: Dict[str, Any] = None) -> List[Chunk]:
        """
        Chunk text into optimal pieces.
        
        This is the main entry point for chunking. It coordinates
        the identification of special content and the actual chunking process.
        
        Args:
            text: The text to chunk
            book_id: ID of the source book
            metadata: Additional metadata for chunks
            
        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            logger.warning(f"Empty text provided for book {book_id}")
            return []
        
        logger.info(f"Starting to chunk text for book {book_id}, length: {len(text)}")
        
        # Pre-process text to identify special regions
        special_regions = self._identify_special_regions(text)
        
        # Perform the actual chunking
        chunks = self._create_chunks(text, special_regions)
        
        # Convert to Chunk objects with proper metadata
        chunk_objects = self._create_chunk_objects(
            chunks, book_id, metadata or {}
        )
        
        # Link chunks for context
        self._link_chunks(chunk_objects)
        
        logger.info(f"Created {len(chunk_objects)} chunks for book {book_id}")
        return chunk_objects
    
    def chunk_pages(self,
                    pages: List[Dict[str, Any]],
                    book_id: str,
                    metadata: Dict[str, Any] = None) -> List[Chunk]:
        """
        Chunk a list of pages from a book.
        
        This method handles page-by-page chunking while maintaining
        continuity across page boundaries.
        
        Args:
            pages: List of page dictionaries with 'text' and 'page_number'
            book_id: ID of the source book  
            metadata: Additional metadata
            
        Returns:
            List of Chunk objects
        """
        all_chunks = []
        accumulated_text = ""
        current_page_start = 1
        
        for page in pages:
            page_num = page.get('page_number', 0)
            page_text = page.get('text', '')
            
            if not page_text.strip():
                continue
            
            # Add page text to accumulator
            if accumulated_text:
                accumulated_text += "\n"
            accumulated_text += page_text
            
            # Check if we should chunk the accumulated text
            if len(accumulated_text) >= self.config.chunk_size:
                # Chunk what we have so far
                chunks = self.chunk_text(accumulated_text, book_id, metadata)
                
                # Add page information to chunks
                for chunk in chunks:
                    chunk.page_start = current_page_start
                    chunk.page_end = page_num
                
                all_chunks.extend(chunks)
                
                # Keep overlap for next batch
                if chunks and self.config.chunk_overlap > 0:
                    last_chunk_text = chunks[-1].text
                    overlap_start = max(0, len(last_chunk_text) - self.config.chunk_overlap)
                    accumulated_text = last_chunk_text[overlap_start:]
                    current_page_start = page_num
                else:
                    accumulated_text = ""
                    current_page_start = page_num + 1
        
        # Handle remaining text
        if accumulated_text.strip():
            chunks = self.chunk_text(accumulated_text, book_id, metadata)
            for chunk in chunks:
                chunk.page_start = current_page_start
                chunk.page_end = pages[-1].get('page_number', current_page_start)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _identify_special_regions(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Identify regions that should not be split.
        
        These include:
        - Code blocks
        - Mathematical formulas  
        - Tables
        
        Returns:
            List of (start, end, type) tuples
        """
        regions = []
        
        # Find code blocks
        if self.config.preserve_code_blocks:
            for match in self.code_block_pattern.finditer(text):
                regions.append((match.start(), match.end(), 'code'))
        
        # Find formulas
        for match in self.formula_pattern.finditer(text):
            regions.append((match.start(), match.end(), 'formula'))
        
        # Sort by start position
        regions.sort(key=lambda x: x[0])
        
        # Merge overlapping regions
        merged = []
        for region in regions:
            if merged and region[0] < merged[-1][1]:
                # Overlapping - extend the previous region
                merged[-1] = (merged[-1][0], max(merged[-1][1], region[1]), 'mixed')
            else:
                merged.append(region)
        
        return merged
    
    def _create_chunks(self, 
                       text: str, 
                       special_regions: List[Tuple[int, int, str]]) -> List[str]:
        """
        Create chunks respecting special regions and boundaries.
        
        This is the core chunking algorithm that:
        1. Avoids splitting special regions
        2. Prefers natural boundaries
        3. Maintains overlap for context
        """
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            # Determine chunk end position
            chunk_end = min(current_pos + self.config.chunk_size, len(text))
            
            # Check if we're in or near a special region
            for region_start, region_end, region_type in special_regions:
                if current_pos <= region_start < chunk_end:
                    # Special region starts within our chunk
                    if region_end <= current_pos + self.config.max_chunk_size:
                        # We can include the entire special region
                        chunk_end = region_end
                    else:
                        # Special region is too large, chunk before it
                        chunk_end = region_start
                    break
            
            # If not at a special region, find a good break point
            if chunk_end < len(text):
                chunk_end = self._find_break_point(text, current_pos, chunk_end)
            
            # Extract chunk
            chunk_text = text[current_pos:chunk_end].strip()
            
            if len(chunk_text) >= self.config.min_chunk_size:
                chunks.append(chunk_text)
                
                # Move position with overlap
                if chunk_end < len(text):
                    overlap_start = max(0, chunk_end - self.config.chunk_overlap)
                    current_pos = overlap_start
                else:
                    current_pos = chunk_end
            else:
                # Chunk too small, extend it
                current_pos = chunk_end
        
        return chunks
    
    def _find_break_point(self, text: str, start: int, ideal_end: int) -> int:
        """
        Find the best position to break text.
        
        Priority:
        1. Paragraph boundary
        2. Sentence boundary  
        3. Word boundary
        4. Any position (fallback)
        """
        # Look for paragraph break
        if self.config.respect_paragraphs:
            paragraph_breaks = list(self.paragraph_pattern.finditer(
                text[start:ideal_end + 100]  # Look a bit ahead
            ))
            if paragraph_breaks:
                # Use the last paragraph break before ideal_end
                for match in reversed(paragraph_breaks):
                    if start + match.start() <= ideal_end:
                        return start + match.end()
        
        # Look for sentence break
        if self.config.respect_sentences:
            sentence_breaks = list(self.sentence_end_pattern.finditer(
                text[start:ideal_end + 50]
            ))
            if sentence_breaks:
                # Use the last sentence break
                last_break = sentence_breaks[-1]
                return start + last_break.end()
        
        # Fall back to word boundary
        space_pos = text.rfind(' ', start, ideal_end)
        if space_pos > start:
            return space_pos + 1
        
        # Last resort - break at ideal_end
        return ideal_end
    
    def _create_chunk_objects(self,
                             text_chunks: List[str],
                             book_id: str,
                             metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Convert text chunks to Chunk objects with metadata.
        """
        chunks = []
        
        for idx, text in enumerate(text_chunks):
            # Determine chunk type
            chunk_type = self._determine_chunk_type(text)
            
            chunk = Chunk(
                book_id=book_id,
                chunk_index=idx,
                text=text,
                chunk_type=chunk_type,
                metadata=metadata.copy()
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _determine_chunk_type(self, text: str) -> ChunkType:
        """
        Determine the type of content in a chunk.
        
        This helps with search relevance and display formatting.
        """
        # Check for code indicators
        code_indicators = ['def ', 'class ', 'import ', 'function', '{', '}', 
                          'return ', 'if ', 'for ', 'while ']
        code_count = sum(1 for indicator in code_indicators if indicator in text)
        if code_count >= 3 or text.strip().startswith('```'):
            return ChunkType.CODE
        
        # Check for formula indicators
        if '$' in text and any(x in text for x in ['=', '+', '-', '*', '/']):
            return ChunkType.FORMULA
        
        # Check for table indicators
        if text.count('|') > 5 and text.count('\n') > 2:
            return ChunkType.TABLE
        
        # Default to text
        return ChunkType.TEXT
    
    def _link_chunks(self, chunks: List[Chunk]) -> None:
        """
        Link chunks to maintain context.
        
        This allows us to easily retrieve surrounding context
        when displaying search results.
        """
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.previous_chunk_id = chunks[i-1].id
            if i < len(chunks) - 1:
                chunk.next_chunk_id = chunks[i+1].id

# Example usage and testing
def test_chunker():
    """Test the chunker with sample text"""
    
    # Sample text with code
    sample_text = """
    Chapter 3: Moving Averages in Trading
    
    Moving averages are one of the most popular technical indicators used in algorithmic trading.
    They help smooth out price action and identify trends.
    
    Here's a simple implementation in Python:
    
    ```python
    def calculate_sma(prices, period):
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / period
    ```
    
    The simple moving average (SMA) calculates the arithmetic mean of prices over a specified period.
    For example, a 20-day SMA sums up the closing prices of the last 20 days and divides by 20.
    
    Traders often use multiple moving averages:
    - Short-term (e.g., 10-day): Responds quickly to price changes
    - Medium-term (e.g., 50-day): Balances responsiveness and smoothness  
    - Long-term (e.g., 200-day): Shows overall trend direction
    
    The formula for exponential moving average (EMA) is:
    $EMA_t = α × Price_t + (1 - α) × EMA_{t-1}$
    
    Where α (alpha) is the smoothing factor: α = 2 / (N + 1)
    """
    
    # Create chunker with small chunks for testing
    config = ChunkingConfig(
        chunk_size=300,
        chunk_overlap=50,
        preserve_code_blocks=True
    )
    chunker = TextChunker(config)
    
    # Chunk the text
    chunks = chunker.chunk_text(sample_text, "test_book_001")
    
    # Display results
    print(f"Created {len(chunks)} chunks\n")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i} ({chunk.chunk_type.value}):")
        print(f"Length: {len(chunk.text)} characters")
        print(f"Preview: {chunk.text[:100]}...")
        print(f"Links: prev={chunk.previous_chunk_id}, next={chunk.next_chunk_id}")
        print("-" * 50)

if __name__ == "__main__":
    test_chunker()
EOF
```

### Testing & Verification

Before moving forward, we need to ensure everything works correctly.

#### Create Integration Test

```python
# Create tests/integration/test_phase1_integration.py
cat > tests/integration/test_phase1_integration.py << 'EOF'
"""
Integration test for Phase 1 components
"""

import pytest
import asyncio
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from ingestion.pdf_parser import PDFParser
from ingestion.text_chunker import TextChunker, ChunkingConfig
from core.models import Book, FileType

class TestPhase1Integration:
    """Test that all Phase 1 components work together"""
    
    @pytest.fixture
    def sample_pdf_path(self, tmp_path):
        """Create a simple test PDF"""
        # For real testing, you'd use a proper PDF
        # This is a placeholder
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("Mock PDF content")
        return pdf_path
    
    def test_pdf_parser_initialization(self):
        """Test PDF parser can be created"""
        parser = PDFParser()
        assert parser is not None
        assert parser.can_parse(Path("test.pdf"))
        assert not parser.can_parse(Path("test.txt"))
    
    def test_chunker_initialization(self):
        """Test chunker can be created"""
        config = ChunkingConfig(chunk_size=500)
        chunker = TextChunker(config)
        assert chunker is not None
        assert chunker.config.chunk_size == 500
    
    def test_chunking_basic_text(self):
        """Test basic text chunking"""
        chunker = TextChunker(ChunkingConfig(chunk_size=100, chunk_overlap=20))
        
        text = "This is a test. " * 50  # 800 characters
        chunks = chunker.chunk_text(text, "test_book")
        
        assert len(chunks) > 1
        assert all(len(c.text) <= 200 for c in chunks)  # Max size respected
        assert chunks[0].book_id == "test_book"
        assert chunks[0].next_chunk_id == chunks[1].id
    
    def test_chunking_preserves_code(self):
        """Test that code blocks are preserved"""
        chunker = TextChunker(ChunkingConfig(
            chunk_size=50,
            preserve_code_blocks=True
        ))
        
        text = """
        Some text before code.
        
        ```python
        def long_function():
            # This is a long function that should not be split
            result = 0
            for i in range(100):
                result += i
            return result
        ```
        
        Some text after code.
        """
        
        chunks = chunker.chunk_text(text, "test_book")
        
        # Find the code chunk
        code_chunks = [c for c in chunks if 'def long_function' in c.text]
        assert len(code_chunks) == 1
        assert '```' in code_chunks[0].text
        assert 'return result' in code_chunks[0].text

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
EOF
```

#### Create Verification Script

```bash
# Create scripts/verify_phase1.py
cat > scripts/verify_phase1.py << 'EOF'
#!/usr/bin/env python3
"""
Verify Phase 1 implementation is complete and working
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_file_exists(file_path: str, description: str) -> bool:
    """Check if a file exists"""
    path = Path(file_path)
    if path.exists():
        logger.info(f"✅ {description}: {file_path}")
        return True
    else:
        logger.error(f"❌ {description} missing: {file_path}")
        return False

def check_imports(module_path: str, class_name: str) -> bool:
    """Check if a module can be imported"""
    try:
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        logger.info(f"✅ Can import {class_name} from {module_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Cannot import {class_name} from {module_path}: {e}")
        return False

def run_basic_tests() -> bool:
    """Run basic functionality tests"""
    try:
        # Test PDF parser
        from ingestion.pdf_parser import PDFParser
        parser = PDFParser()
        logger.info("✅ PDFParser instantiated successfully")
        
        # Test chunker
        from ingestion.text_chunker import TextChunker
        chunker = TextChunker()
        test_chunks = chunker.chunk_text("Test text " * 100, "test_book")
        logger.info(f"✅ TextChunker created {len(test_chunks)} chunks")
        
        # Test models
        from core.models import Book, Chunk
        book = Book(
            id="test",
            title="Test Book",
            file_path="/tmp/test.pdf",
            file_type="pdf"
        )
        logger.info("✅ Models working correctly")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic tests failed: {e}")
        return False

def main():
    """Run all Phase 1 verification checks"""
    logger.info("=" * 60)
    logger.info("PHASE 1 VERIFICATION")
    logger.info(f"Started at: {datetime.now()}")
    logger.info("=" * 60)
    
    checks = []
    
    # Check core files exist
    logger.info("\nChecking core files...")
    checks.append(check_file_exists("src/core/models.py", "Core models"))
    checks.append(check_file_exists("src/core/interfaces.py", "Storage interfaces"))
    checks.append(check_file_exists("src/core/config.py", "Configuration"))
    
    # Check ingestion files
    logger.info("\nChecking ingestion files...")
    checks.append(check_file_exists("src/ingestion/pdf_parser.py", "PDF parser"))
    checks.append(check_file_exists("src/ingestion/text_chunker.py", "Text chunker"))
    
    # Check imports work
    logger.info("\nChecking imports...")
    checks.append(check_imports("core.models", "Book"))
    checks.append(check_imports("core.models", "Chunk"))
    checks.append(check_imports("ingestion.pdf_parser", "PDFParser"))
    checks.append(check_imports("ingestion.text_chunker", "TextChunker"))
    
    # Run basic tests
    logger.info("\nRunning basic functionality tests...")
    checks.append(run_basic_tests())
    
    # Summary
    logger.info("\n" + "=" * 60)
    total = len(checks)
    passed = sum(checks)
    
    if passed == total:
        logger.info(f"✅ ALL CHECKS PASSED ({passed}/{total})")
        logger.info("Phase 1 implementation is complete!")
        return 0
    else:
        logger.error(f"❌ SOME CHECKS FAILED ({passed}/{total})")
        logger.error("Please fix the issues above before proceeding to Phase 2")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

chmod +x scripts/verify_phase1.py
```

---

### Implement Embedding Generator

The embedding generator converts text into numerical vectors that capture semantic meaning.

```python
# Create src/ingestion/embeddings.py
cat > src/ingestion/embeddings.py << 'EOF'
"""
Embedding generation for semantic search

This module handles converting text chunks into vector embeddings
that can be used for semantic similarity search.
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from datetime import datetime
import hashlib
import json

import openai
from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from core.models import Chunk
from core.config import get_config

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Generates embeddings for text chunks.
    
    This class supports multiple embedding models:
    1. OpenAI embeddings (requires API key)
    2. Local sentence transformers (no API needed)
    
    The embeddings capture semantic meaning, allowing us to find
    similar content even when different words are used.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the embedding model to use
        """
        self.config = get_config()
        self.model_name = model_name or self.config.embedding.model
        
        # Initialize the appropriate model
        if self.model_name.startswith("text-embedding"):
            # OpenAI model
            self._init_openai()
        else:
            # Local sentence transformer
            self._init_sentence_transformer()
        
        # Cache for embeddings (avoid regenerating)
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def _init_openai(self):
        """Initialize OpenAI client"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_key_here":
            raise ValueError(
                "OpenAI API key not found! Please set OPENAI_API_KEY in .env file"
            )
        
        self.client = OpenAI(api_key=api_key)
        self.embedding_dimension = 1536  # for ada-002
        self.is_local = False
        logger.info(f"Initialized OpenAI embeddings with model: {self.model_name}")
        
    def _init_sentence_transformer(self):
        """Initialize local sentence transformer"""
        logger.info(f"Loading sentence transformer: {self.model_name}")
        
        # Check if CUDA is available for GPU acceleration
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        self.model = SentenceTransformer(self.model_name, device=device)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        self.is_local = True
        
        logger.info(f"Loaded model with dimension: {self.embedding_dimension}")
        
    async def generate_embeddings(self, 
                                  chunks: List[Chunk],
                                  show_progress: bool = True) -> List[List[float]]:
        """
        Generate embeddings for a list of chunks.
        
        This is the main method for generating embeddings. It handles:
        - Batching for efficiency
        - Caching to avoid regeneration
        - Progress tracking
        - Error handling and retries
        
        Args:
            chunks: List of chunks to embed
            show_progress: Whether to show progress bar
            
        Returns:
            List of embedding vectors
        """
        if not chunks:
            return []
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        # Group chunks by whether they're cached
        cached_chunks = []
        uncached_chunks = []
        
        for chunk in chunks:
            cache_key = self._get_cache_key(chunk.text)
            if cache_key in self.cache:
                cached_chunks.append((chunk, self.cache[cache_key]))
                self.cache_hits += 1
            else:
                uncached_chunks.append(chunk)
                self.cache_misses += 1
        
        logger.info(f"Cache hits: {len(cached_chunks)}, misses: {len(uncached_chunks)}")
        
        # Generate embeddings for uncached chunks
        if uncached_chunks:
            if self.is_local:
                new_embeddings = await self._generate_local_embeddings(uncached_chunks)
            else:
                new_embeddings = await self._generate_openai_embeddings(uncached_chunks)
            
            # Add to cache
            for chunk, embedding in zip(uncached_chunks, new_embeddings):
                cache_key = self._get_cache_key(chunk.text)
                self.cache[cache_key] = embedding
        else:
            new_embeddings = []
        
        # Combine cached and new embeddings in original order
        result = []
        cached_dict = {id(chunk): embedding for chunk, embedding in cached_chunks}
        new_iter = iter(new_embeddings)
        
        for chunk in chunks:
            if id(chunk) in cached_dict:
                result.append(cached_dict[id(chunk)])
            else:
                result.append(next(new_iter))
        
        return result
    
    async def _generate_openai_embeddings(self, chunks: List[Chunk]) -> List[List[float]]:
        """Generate embeddings using OpenAI API"""
        embeddings = []
        batch_size = self.config.embedding.batch_size
        
        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [chunk.text for chunk in batch]
            
            try:
                # Make API call
                response = await asyncio.to_thread(
                    self.client.embeddings.create,
                    input=texts,
                    model=self.model_name
                )
                
                # Extract embeddings
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                logger.debug(f"Generated {len(batch_embeddings)} embeddings")
                
                # Rate limiting - be nice to the API
                if i + batch_size < len(chunks):
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error generating OpenAI embeddings: {e}")
                # Return zero vectors for failed chunks
                failed_embeddings = [[0.0] * self.embedding_dimension] * len(batch)
                embeddings.extend(failed_embeddings)
        
        return embeddings
    
    async def _generate_local_embeddings(self, chunks: List[Chunk]) -> List[List[float]]:
        """Generate embeddings using local model"""
        texts = [chunk.text for chunk in chunks]
        
        try:
            # Generate embeddings
            # Run in thread to avoid blocking
            embeddings = await asyncio.to_thread(
                self.model.encode,
                texts,
                batch_size=self.config.embedding.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Convert to list format
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Error generating local embeddings: {e}")
            # Return zero vectors for failed chunks
            return [[0.0] * self.embedding_dimension] * len(chunks)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        # Include model name in cache key
        key_string = f"{self.model_name}:{text}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        
        Queries are handled separately because they might need
        different processing than document chunks.
        """
        # Check cache first
        cache_key = self._get_cache_key(query)
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        self.cache_misses += 1
        
        # Generate embedding
        if self.is_local:
            embedding = await asyncio.to_thread(
                self.model.encode,
                [query],
                convert_to_numpy=True
            )
            result = embedding[0].tolist()
        else:
            response = await asyncio.to_thread(
                self.client.embeddings.create,
                input=[query],
                model=self.model_name
            )
            result = response.data[0].embedding
        
        # Cache it
        self.cache[cache_key] = result
        
        return result
    
    def save_cache(self, file_path: str):
        """Save embedding cache to disk"""
        cache_data = {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dimension,
            'cache': self.cache,
            'stats': {
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'saved_at': datetime.now().isoformat()
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(cache_data, f)
        
        logger.info(f"Saved {len(self.cache)} cached embeddings to {file_path}")
    
    def load_cache(self, file_path: str):
        """Load embedding cache from disk"""
        if not os.path.exists(file_path):
            logger.warning(f"Cache file not found: {file_path}")
            return
        
        try:
            with open(file_path, 'r') as f:
                cache_data = json.load(f)
            
            # Verify model compatibility
            if cache_data['model_name'] != self.model_name:
                logger.warning(
                    f"Cache model mismatch: {cache_data['model_name']} != {self.model_name}"
                )
                return
            
            self.cache = cache_data['cache']
            logger.info(f"Loaded {len(self.cache)} cached embeddings")
            
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding generation statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dimension,
            'is_local': self.is_local,
            'cache_size': len(self.cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': hit_rate,
            'total_requests': total_requests
        }

# Utility functions for testing and validation

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

async def test_embedding_generator():
    """Test the embedding generator"""
    
    # Create test chunks
    test_chunks = [
        Chunk(
            book_id="test",
            chunk_index=0,
            text="Moving averages are technical indicators used in trading."
        ),
        Chunk(
            book_id="test",
            chunk_index=1,
            text="The simple moving average calculates the mean of prices."
        ),
        Chunk(
            book_id="test", 
            chunk_index=2,
            text="Python is a programming language used for data analysis."
        )
    ]
    
    # Test with local model (no API key needed)
    print("Testing with local model...")
    generator = EmbeddingGenerator("all-MiniLM-L6-v2")
    
    # Generate embeddings
    embeddings = await generator.generate_embeddings(test_chunks)
    
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding dimension: {len(embeddings[0])}")
    
    # Test similarity
    print("\nTesting semantic similarity:")
    query = "What are moving averages in trading?"
    query_embedding = await generator.generate_query_embedding(query)
    
    for i, (chunk, embedding) in enumerate(zip(test_chunks, embeddings)):
        similarity = cosine_similarity(query_embedding, embedding)
        print(f"Chunk {i}: {similarity:.3f} - {chunk.text[:50]}...")
    
    # Show stats
    print(f"\nStats: {generator.get_stats()}")

if __name__ == "__main__":
    asyncio.run(test_embedding_generator())
EOF
```

### Implement Storage Systems

Now we need concrete implementations of our storage interfaces.

#### SQLite Storage Implementation

```python
# Create src/core/sqlite_storage.py
cat > src/core/sqlite_storage.py << 'EOF'
"""
SQLite storage implementation for TradeKnowledge

This provides persistent storage for books and chunks,
with full-text search capabilities using FTS5.
"""

import sqlite3
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import asyncio
from contextlib import asynccontextmanager

from core.interfaces import BookStorageInterface, ChunkStorageInterface
from core.models import Book, Chunk, FileType, ChunkType
from core.config import get_config

logger = logging.getLogger(__name__)

class SQLiteStorage(BookStorageInterface, ChunkStorageInterface):
    """
    SQLite implementation of storage interfaces.
    
    This class provides:
    - Book metadata storage
    - Chunk text storage with FTS5 search
    - Transaction support
    - Connection pooling
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize SQLite storage"""
        config = get_config()
        self.db_path = db_path or config.database.sqlite.path
        
        # Ensure database directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Connection pool (simple implementation)
        self._connection = None
        
    def _init_database(self):
        """Initialize database schema if needed"""
        # This should already be done by init_db.py
        # but we check just in case
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if tables exist
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='books'"
        )
        if not cursor.fetchone():
            logger.warning("Database not initialized! Run scripts/init_db.py")
        
        conn.close()
    
    @asynccontextmanager
    async def _get_connection(self):
        """Get database connection (async context manager)"""
        # For simplicity, we use a single connection
        # In production, you'd want a proper connection pool
        if self._connection is None:
            self._connection = await asyncio.to_thread(
                sqlite3.connect, 
                self.db_path,
                check_same_thread=False
            )
            self._connection.row_factory = sqlite3.Row
        
        yield self._connection
    
    # Book Storage Methods
    
    async def save_book(self, book: Book) -> bool:
        """Save a book's metadata"""
        try:
            async with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Convert metadata to JSON
                metadata_json = json.dumps(book.metadata)
                
                # Insert or replace
                await asyncio.to_thread(
                    cursor.execute,
                    """
                    INSERT OR REPLACE INTO books (
                        id, title, author, isbn, file_path, file_type,
                        file_hash, total_chunks, metadata, created_at, indexed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        book.id,
                        book.title,
                        book.author,
                        book.isbn,
                        book.file_path,
                        book.file_type.value,
                        book.file_hash,
                        book.total_chunks,
                        metadata_json,
                        book.created_at.isoformat(),
                        book.indexed_at.isoformat() if book.indexed_at else None
                    )
                )
                
                await asyncio.to_thread(conn.commit)
                logger.info(f"Saved book: {book.id} - {book.title}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving book: {e}")
            return False
    
    async def get_book(self, book_id: str) -> Optional[Book]:
        """Retrieve a book by ID"""
        try:
            async with self._get_connection() as conn:
                cursor = conn.cursor()
                
                row = await asyncio.to_thread(
                    cursor.execute,
                    "SELECT * FROM books WHERE id = ?",
                    (book_id,)
                )
                
                row = cursor.fetchone()
                if row:
                    return self._row_to_book(row)
                
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving book: {e}")
            return None
    
    async def get_book_by_hash(self, file_hash: str) -> Optional[Book]:
        """Retrieve a book by file hash"""
        try:
            async with self._get_connection() as conn:
                cursor = conn.cursor()
                
                await asyncio.to_thread(
                    cursor.execute,
                    "SELECT * FROM books WHERE file_hash = ?",
                    (file_hash,)
                )
                
                row = cursor.fetchone()
                if row:
                    return self._row_to_book(row)
                
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving book by hash: {e}")
            return None
    
    async def list_books(self, 
                        category: Optional[str] = None,
                        limit: int = 100,
                        offset: int = 0) -> List[Book]:
        """List books with optional filtering"""
        try:
            async with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if category:
                    # Search in metadata JSON
                    query = """
                        SELECT * FROM books 
                        WHERE json_extract(metadata, '$.categories') LIKE ?
                        ORDER BY created_at DESC
                        LIMIT ? OFFSET ?
                    """
                    params = (f'%{category}%', limit, offset)
                else:
                    query = """
                        SELECT * FROM books
                        ORDER BY created_at DESC  
                        LIMIT ? OFFSET ?
                    """
                    params = (limit, offset)
                
                await asyncio.to_thread(cursor.execute, query, params)
                
                rows = cursor.fetchall()
                return [self._row_to_book(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error listing books: {e}")
            return []
    
    async def update_book(self, book: Book) -> bool:
        """Update book metadata"""
        # Same as save_book with INSERT OR REPLACE
        return await self.save_book(book)
    
    async def delete_book(self, book_id: str) -> bool:
        """Delete a book and all its chunks"""
        try:
            async with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Delete chunks first (foreign key constraint)
                await asyncio.to_thread(
                    cursor.execute,
                    "DELETE FROM chunks WHERE book_id = ?",
                    (book_id,)
                )
                
                # Delete book
                await asyncio.to_thread(
                    cursor.execute,
                    "DELETE FROM books WHERE id = ?",
                    (book_id,)
                )
                
                await asyncio.to_thread(conn.commit)
                logger.info(f"Deleted book and chunks: {book_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting book: {e}")
            return False
    
    # Chunk Storage Methods
    
    async def save_chunks(self, chunks: List[Chunk]) -> bool:
        """Save multiple chunks efficiently"""
        if not chunks:
            return True
        
        try:
            async with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Prepare data
                chunk_data = []
                for chunk in chunks:
                    metadata_json = json.dumps(chunk.metadata)
                    chunk_data.append((
                        chunk.id,
                        chunk.book_id,
                        chunk.chunk_index,
                        chunk.text,
                        chunk.embedding_id,
                        metadata_json,
                        chunk.created_at.isoformat()
                    ))
                
                # Batch insert
                await asyncio.to_thread(
                    cursor.executemany,
                    """
                    INSERT OR REPLACE INTO chunks (
                        id, book_id, chunk_index, text, embedding_id,
                        metadata, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    chunk_data
                )
                
                await asyncio.to_thread(conn.commit)
                logger.info(f"Saved {len(chunks)} chunks")
                return True
                
        except Exception as e:
            logger.error(f"Error saving chunks: {e}")
            return False
    
    async def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Retrieve a single chunk"""
        try:
            async with self._get_connection() as conn:
                cursor = conn.cursor()
                
                await asyncio.to_thread(
                    cursor.execute,
                    "SELECT * FROM chunks WHERE id = ?",
                    (chunk_id,)
                )
                
                row = cursor.fetchone()
                if row:
                    return self._row_to_chunk(row)
                
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving chunk: {e}")
            return None
    
    async def get_chunks_by_book(self, book_id: str) -> List[Chunk]:
        """Get all chunks for a book"""
        try:
            async with self._get_connection() as conn:
                cursor = conn.cursor()
                
                await asyncio.to_thread(
                    cursor.execute,
                    """
                    SELECT * FROM chunks 
                    WHERE book_id = ?
                    ORDER BY chunk_index
                    """,
                    (book_id,)
                )
                
                rows = cursor.fetchall()
                return [self._row_to_chunk(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error retrieving chunks by book: {e}")
            return []
    
    async def get_chunk_context(self, 
                               chunk_id: str,
                               before: int = 1,
                               after: int = 1) -> Dict[str, Any]:
        """Get a chunk with surrounding context"""
        try:
            async with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get the target chunk
                await asyncio.to_thread(
                    cursor.execute,
                    "SELECT * FROM chunks WHERE id = ?",
                    (chunk_id,)
                )
                
                target_row = cursor.fetchone()
                if not target_row:
                    return {}
                
                target_chunk = self._row_to_chunk(target_row)
                
                # Get surrounding chunks
                await asyncio.to_thread(
                    cursor.execute,
                    """
                    SELECT * FROM chunks
                    WHERE book_id = ? 
                    AND chunk_index >= ? 
                    AND chunk_index <= ?
                    ORDER BY chunk_index
                    """,
                    (
                        target_chunk.book_id,
                        target_chunk.chunk_index - before,
                        target_chunk.chunk_index + after
                    )
                )
                
                rows = cursor.fetchall()
                chunks = [self._row_to_chunk(row) for row in rows]
                
                # Build context
                context = {
                    'chunk': target_chunk,
                    'before': [],
                    'after': []
                }
                
                for chunk in chunks:
                    if chunk.chunk_index < target_chunk.chunk_index:
                        context['before'].append(chunk)
                    elif chunk.chunk_index > target_chunk.chunk_index:
                        context['after'].append(chunk)
                
                return context
                
        except Exception as e:
            logger.error(f"Error getting chunk context: {e}")
            return {}
    
    async def search_exact(self,
                          query: str,
                          book_ids: Optional[List[str]] = None,
                          limit: int = 10) -> List[Dict[str, Any]]:
        """Perform exact text search using FTS5"""
        try:
            async with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Build query
                if book_ids:
                    # Filter by book IDs
                    placeholders = ','.join('?' * len(book_ids))
                    fts_query = f"""
                        SELECT c.*, snippet(chunks_fts, 1, '<mark>', '</mark>', '...', 20) as snippet,
                               rank as score
                        FROM chunks_fts 
                        JOIN chunks c ON chunks_fts.id = c.id
                        WHERE chunks_fts MATCH ? 
                        AND c.book_id IN ({placeholders})
                        ORDER BY rank
                        LIMIT ?
                    """
                    params = [query] + book_ids + [limit]
                else:
                    fts_query = """
                        SELECT c.*, snippet(chunks_fts, 1, '<mark>', '</mark>', '...', 20) as snippet,
                               rank as score
                        FROM chunks_fts
                        JOIN chunks c ON chunks_fts.id = c.id
                        WHERE chunks_fts MATCH ?
                        ORDER BY rank
                        LIMIT ?
                    """
                    params = [query, limit]
                
                await asyncio.to_thread(cursor.execute, fts_query, params)
                
                rows = cursor.fetchall()
                results = []
                
                for row in rows:
                    chunk = self._row_to_chunk(row)
                    results.append({
                        'chunk': chunk,
                        'score': -row['score'],  # FTS5 rank is negative
                        'snippet': row['snippet']
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Error in exact search: {e}")
            return []
    
    async def delete_chunks_by_book(self, book_id: str) -> bool:
        """Delete all chunks for a book"""
        try:
            async with self._get_connection() as conn:
                cursor = conn.cursor()
                
                await asyncio.to_thread(
                    cursor.execute,
                    "DELETE FROM chunks WHERE book_id = ?",
                    (book_id,)
                )
                
                await asyncio.to_thread(conn.commit)
                return True
                
        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")
            return False
    
    # Helper methods
    
    def _row_to_book(self, row: sqlite3.Row) -> Book:
        """Convert database row to Book object"""
        return Book(
            id=row['id'],
            title=row['title'],
            author=row['author'],
            isbn=row['isbn'],
            file_path=row['file_path'],
            file_type=FileType(row['file_type']),
            file_hash=row['file_hash'],
            total_chunks=row['total_chunks'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            created_at=datetime.fromisoformat(row['created_at']),
            indexed_at=datetime.fromisoformat(row['indexed_at']) if row['indexed_at'] else None
        )
    
    def _row_to_chunk(self, row: sqlite3.Row) -> Chunk:
        """Convert database row to Chunk object"""
        metadata = json.loads(row['metadata']) if row['metadata'] else {}
        
        return Chunk(
            id=row['id'],
            book_id=row['book_id'],
            chunk_index=row['chunk_index'],
            text=row['text'],
            embedding_id=row['embedding_id'],
            metadata=metadata,
            created_at=datetime.fromisoformat(row['created_at'])
        )

# Test the storage
async def test_storage():
    """Test SQLite storage implementation"""
    storage = SQLiteStorage("data/test.db")
    
    # Test book operations
    book = Book(
        id="test-001",
        title="Test Book",
        author="Test Author",
        file_path="/tmp/test.pdf",
        file_type=FileType.PDF,
        file_hash="testhash123"
    )
    
    # Save book
    success = await storage.save_book(book)
    print(f"Save book: {success}")
    
    # Retrieve book
    retrieved = await storage.get_book("test-001")
    print(f"Retrieved: {retrieved.title if retrieved else 'Not found'}")
    
    # Test chunk operations
    chunks = [
        Chunk(
            book_id="test-001",
            chunk_index=i,
            text=f"This is test chunk {i}"
        )
        for i in range(5)
    ]
    
    # Save chunks
    success = await storage.save_chunks(chunks)
    print(f"Save chunks: {success}")
    
    # Search
    results = await storage.search_exact("test chunk", limit=3)
    print(f"Search results: {len(results)}")
    
    for result in results:
        print(f"  - {result['chunk'].text[:50]}... (score: {result['score']:.3f})")

if __name__ == "__main__":
    asyncio.run(test_storage())
EOF
```

#### ChromaDB Storage Implementation

```python
# Create src/core/chroma_storage.py
cat > src/core/chroma_storage.py << 'EOF'
"""
ChromaDB vector storage implementation

This handles semantic search using vector embeddings.
"""

import logging
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from core.interfaces import VectorStorageInterface
from core.models import Chunk
from core.config import get_config

logger = logging.getLogger(__name__)

class ChromaDBStorage(VectorStorageInterface):
    """
    ChromaDB implementation for vector storage.
    
    This provides semantic search capabilities by storing
    and searching through vector embeddings.
    """
    
    def __init__(self, persist_directory: Optional[str] = None):
        """Initialize ChromaDB storage"""
        config = get_config()
        self.persist_directory = persist_directory or config.database.chroma.persist_directory
        self.collection_name = config.database.chroma.collection_name
        
        # Ensure directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
        logger.info(f"Initialized ChromaDB with collection: {self.collection_name}")
    
    def _get_or_create_collection(self):
        """Get or create the main collection"""
        try:
            collection = self.client.get_collection(self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
        except:
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Trading and ML book embeddings",
                    "created_at": datetime.now().isoformat()
                }
            )
            logger.info(f"Created new collection: {self.collection_name}")
        
        return collection
    
    async def save_embeddings(self, 
                             chunks: List[Chunk],
                             embeddings: List[List[float]]) -> bool:
        """Save chunk embeddings to ChromaDB"""
        if not chunks or not embeddings:
            return True
        
        if len(chunks) != len(embeddings):
            logger.error(f"Mismatch: {len(chunks)} chunks, {len(embeddings)} embeddings")
            return False
        
        try:
            # Prepare data for ChromaDB
            ids = []
            documents = []
            metadatas = []
            
            for chunk in chunks:
                ids.append(chunk.id)
                documents.append(chunk.text)
                
                # Prepare metadata
                metadata = {
                    'book_id': chunk.book_id,
                    'chunk_index': chunk.chunk_index,
                    'chunk_type': chunk.chunk_type.value,
                    'created_at': chunk.created_at.isoformat()
                }
                
                # Add optional fields
                if chunk.chapter:
                    metadata['chapter'] = chunk.chapter
                if chunk.section:
                    metadata['section'] = chunk.section
                if chunk.page_start:
                    metadata['page_start'] = chunk.page_start
                if chunk.page_end:
                    metadata['page_end'] = chunk.page_end
                
                metadatas.append(metadata)
            
            # Add to collection in batches
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_docs = documents[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                batch_metadata = metadatas[i:i + batch_size]
                
                # Use asyncio to avoid blocking
                await asyncio.to_thread(
                    self.collection.add,
                    ids=batch_ids,
                    documents=batch_docs,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadata
                )
                
                logger.debug(f"Added batch {i//batch_size + 1} ({len(batch_ids)} chunks)")
            
            logger.info(f"Successfully saved {len(chunks)} embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            return False
    
    async def search_semantic(self,
                             query_embedding: List[float],
                             filter_dict: Optional[Dict[str, Any]] = None,
                             limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform semantic search using vector similarity.
        
        Args:
            query_embedding: Vector embedding of the query
            filter_dict: Metadata filters (e.g., {'book_id': 'xyz'})
            limit: Maximum number of results
            
        Returns:
            List of search results with chunks and scores
        """
        try:
            # Build where clause for filtering
            where = None
            if filter_dict:
                # ChromaDB expects specific filter format
                where = {}
                if 'book_ids' in filter_dict and filter_dict['book_ids']:
                    where['book_id'] = {'$in': filter_dict['book_ids']}
                if 'chunk_type' in filter_dict:
                    where['chunk_type'] = filter_dict['chunk_type']
            
            # Perform search
            results = await asyncio.to_thread(
                self.collection.query,
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            search_results = []
            if results['ids'] and results['ids'][0]:
                for i, chunk_id in enumerate(results['ids'][0]):
                    # Convert distance to similarity score (1 - normalized_distance)
                    # ChromaDB uses L2 distance by default
                    distance = results['distances'][0][i]
                    score = 1 / (1 + distance)  # Convert distance to similarity
                    
                    search_results.append({
                        'chunk_id': chunk_id,
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'score': score,
                        'distance': distance
                    })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    async def delete_embeddings(self, chunk_ids: List[str]) -> bool:
        """Delete embeddings by chunk IDs"""
        if not chunk_ids:
            return True
        
        try:
            await asyncio.to_thread(
                self.collection.delete,
                ids=chunk_ids
            )
            logger.info(f"Deleted {len(chunk_ids)} embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting embeddings: {e}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection"""
        try:
            count = self.collection.count()
            
            # Get collection metadata
            metadata = self.collection.metadata or {}
            
            return {
                'collection_name': self.collection_name,
                'total_embeddings': count,
                'persist_directory': self.persist_directory,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {
                'collection_name': self.collection_name,
                'error': str(e)
            }

# Test ChromaDB storage
async def test_chroma_storage():
    """Test ChromaDB storage implementation"""
    storage = ChromaDBStorage()
    
    # Get stats
    stats = await storage.get_collection_stats()
    print(f"Collection stats: {stats}")
    
    # Create test data
    test_chunks = [
        Chunk(
            id=f"test_chunk_{i}",
            book_id="test_book",
            chunk_index=i,
            text=f"Test chunk {i} about trading strategies"
        )
        for i in range(3)
    ]
    
    # Create fake embeddings (normally from embedding generator)
    import random
    test_embeddings = [
        [random.random() for _ in range(384)]  # 384-dim embeddings
        for _ in test_chunks
    ]
    
    # Save embeddings
    success = await storage.save_embeddings(test_chunks, test_embeddings)
    print(f"Save embeddings: {success}")
    
    # Test search
    query_embedding = [random.random() for _ in range(384)]
    results = await storage.search_semantic(query_embedding, limit=2)
    
    print(f"\nSearch results ({len(results)} found):")
    for result in results:
        print(f"  - ID: {result['chunk_id']}")
        print(f"    Score: {result['score']:.3f}")
        print(f"    Text: {result['text'][:50]}...")

if __name__ == "__main__":
    asyncio.run(test_chroma_storage())
EOF
```

## Phase 1 Summary

### What We've Built

In Phase 1, we've created the foundation of the TradeKnowledge system:

1. **Core Data Models** - Structured representations of books, chunks, and search results
2. **PDF Parser** - Extracts text and metadata from PDF files
3. **Intelligent Text Chunker** - Breaks text into searchable pieces while preserving context
4. **Embedding Generator** - Converts text to vectors for semantic search
5. **Storage Systems** - SQLite for text/metadata, ChromaDB for vectors
6. **Basic Search Engine** - Semantic, exact, and hybrid search capabilities
7. **Book Processor** - Orchestrates the complete ingestion pipeline

### Key Achievements

- ✅ Clean architecture with interfaces and implementations
- ✅ Async/await throughout for performance
- ✅ Proper error handling and logging
- ✅ Caching for embeddings
- ✅ Support for both OpenAI and local embedding models
- ✅ Full-text search with SQLite FTS5
- ✅ Vector similarity search with ChromaDB
- ✅ Hybrid search combining both approaches

### Testing Your Implementation

Run these commands to verify Phase 1 is complete:

```bash
# 1. Verify environment
python scripts/verify_environment.py

# 2. Initialize database
python scripts/init_db.py

# 3. Run Phase 1 verification
python scripts/verify_phase1.py

# 4. Run complete pipeline test
python scripts/test_phase1_complete.py
```

### Next Steps

Once all tests pass, you're ready for Phase 2 where we'll add:
- Advanced features (OCR, EPUB support)
- Performance optimizations with C++
- Advanced caching strategies
- Query suggestion engine
- And more!

---

**END OF PHASE 1 IMPLEMENTATION GUIDE**
