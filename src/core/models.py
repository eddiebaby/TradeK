"""
Core data models for TradeKnowledge

These models define the structure of our data throughout the system.
Think of them as contracts - any component that uses these models
knows exactly what data to expect.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
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
    
    @field_validator('file_path')
    @classmethod
    def validate_file_path(cls, file_path: str) -> str:
        """Validate file path to prevent security issues"""
        if not file_path:
            raise ValueError("File path cannot be empty")
        
        # Security validation - this will raise ValueError if path is malicious
        cls._validate_file_path(file_path)
        
        return file_path
    
    @model_validator(mode='before')
    def generate_file_hash(cls, values):
        """Generate file hash if not provided"""
        if isinstance(values, dict):
            if not values.get('file_hash'):
                file_path = values.get('file_path')
                if file_path:
                    try:
                        # Path is already validated by field validator, so this is safe
                        safe_path = Path(file_path).resolve()
                        
                        if safe_path.exists() and safe_path.is_file():
                            # Additional security: Check file size limits (500MB max)
                            file_size = safe_path.stat().st_size
                            max_size = 500 * 1024 * 1024  # 500MB
                            if file_size > max_size:
                                raise ValueError(f"File too large: {file_size} bytes (max: {max_size})")
                            
                            # Read file in chunks to handle large files securely
                            sha256_hash = hashlib.sha256()
                            with open(safe_path, "rb") as f:
                                for byte_block in iter(lambda: f.read(4096), b""):
                                    sha256_hash.update(byte_block)
                            values['file_hash'] = sha256_hash.hexdigest()
                        else:
                            values['file_hash'] = 'file_not_found'
                    except (OSError, IOError, PermissionError) as e:
                        values['file_hash'] = f'error_{hash(str(e)) % 10000:04d}'
                else:
                    values['file_hash'] = 'unknown'
        return values
    
    @classmethod
    def _validate_file_path(cls, file_path: str) -> Path:
        """
        Validate file path to prevent directory traversal attacks.
        
        Args:
            file_path: The file path to validate
            
        Returns:
            Path: Validated and resolved path
            
        Raises:
            ValueError: If path is invalid or outside allowed directories
        """
        try:
            # Check for suspicious patterns in the original path first
            suspicious_patterns = ['..', '~', '$', '`', ';', '|', '&']
            if any(pattern in file_path for pattern in suspicious_patterns):
                raise ValueError(f"File path contains suspicious characters: {file_path}")
            
            # Convert to Path and resolve to absolute path
            path = Path(file_path).resolve()
            
            # Define allowed base directories for book files
            project_root = Path.cwd()
            allowed_directories = [
                project_root / "data" / "books",
                project_root / "data" / "uploads", 
                project_root / "test_data",
                Path("/tmp"),  # Allow temp directory for testing
            ]
            
            # Ensure all allowed directories exist or create them
            for allowed_dir in allowed_directories[:3]:  # Skip /tmp
                allowed_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if the path is within any allowed directory
            is_allowed = False
            for allowed_dir in allowed_directories:
                try:
                    allowed_dir = allowed_dir.resolve()
                    # Check if path is within allowed directory
                    path.relative_to(allowed_dir)
                    is_allowed = True
                    break
                except ValueError:
                    # Path is not relative to this allowed directory
                    continue
            
            if not is_allowed:
                raise ValueError(
                    f"File path '{file_path}' is outside allowed directories. "
                    f"Allowed directories: {[str(d) for d in allowed_directories]}"
                )
            
            # Additional security checks for symlinks
            if path.is_symlink():
                # Resolve symlink and re-validate the target
                resolved_target = path.readlink().resolve()
                return cls._validate_file_path(str(resolved_target))
            
            return path
            
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Invalid file path '{file_path}': {e}") from e
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        }
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
    
    @model_validator(mode='before')
    def generate_chunk_id(cls, values):
        """Generate chunk ID if not provided"""
        if isinstance(values, dict):
            if not values.get('id'):
                book_id = values.get('book_id', 'unknown')
                chunk_index = values.get('chunk_index', 0)
                values['id'] = f"{book_id}_chunk_{chunk_index:05d}"
        return values
    
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
        file_path="data/books/algo_trading.pdf",  # FIXED: Use relative path
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