"""
Pydantic models for TradeKnowledge API

These models define the structure of requests and responses
for all API endpoints.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, validator
from pathlib import Path

# Base Models
class BaseResponse(BaseModel):
    """Base response model"""
    success: bool = True
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: int = Field(1, ge=1, description="Page number")
    size: int = Field(20, ge=1, le=100, description="Items per page")
    
    @property
    def offset(self) -> int:
        return (self.page - 1) * self.size

class PaginatedResponse(BaseResponse):
    """Paginated response wrapper"""
    page: int
    size: int
    total: int
    pages: int
    
    @classmethod
    def create(cls, items: List[Any], pagination: PaginationParams, total: int):
        pages = (total + pagination.size - 1) // pagination.size
        return cls(
            data=items,
            page=pagination.page,
            size=pagination.size,
            total=total,
            pages=pages
        )

# Health Check Models
class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Overall system status")
    timestamp: datetime
    components: Dict[str, str] = Field(..., description="Component health status")
    version: str
    error: Optional[str] = None

# Search Models
class SearchIntent(str, Enum):
    """Search intent types"""
    SEMANTIC = "semantic"
    EXACT = "exact"
    CODE = "code"
    FORMULA = "formula"
    STRATEGY = "strategy"
    CONCEPT = "concept"

class SearchRequest(BaseModel):
    """Search request model"""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    intent: Optional[SearchIntent] = Field(None, description="Search intent hint")
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")
    include_similar: bool = Field(True, description="Include similar results")
    max_results: int = Field(20, ge=1, le=100, description="Maximum results to return")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="Minimum relevance score")

class SearchResultItem(BaseModel):
    """Individual search result"""
    id: str = Field(..., description="Result ID")
    title: str = Field(..., description="Result title")
    content: str = Field(..., description="Result content excerpt")
    score: float = Field(..., description="Relevance score")
    book_id: str = Field(..., description="Source book ID")
    book_title: str = Field(..., description="Source book title")
    page_number: Optional[int] = Field(None, description="Page number in book")
    chunk_type: str = Field(..., description="Type of content chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    highlights: List[str] = Field(default_factory=list, description="Highlighted snippets")

class SearchResponse(BaseResponse):
    """Search response model"""
    query: str
    intent: Optional[str]
    results: List[SearchResultItem]
    total_found: int
    processing_time_ms: float
    suggestions: List[str] = Field(default_factory=list)
    filters_applied: Dict[str, Any] = Field(default_factory=dict)

class AutocompleteRequest(BaseModel):
    """Autocomplete request"""
    partial_query: str = Field(..., min_length=1, max_length=100)
    max_suggestions: int = Field(5, ge=1, le=20)

class AutocompleteResponse(BaseResponse):
    """Autocomplete response"""
    suggestions: List[str]
    query: str

# Book Management Models
class BookUploadRequest(BaseModel):
    """Book upload request metadata"""
    title: Optional[str] = None
    author: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    language: str = Field("en", description="Book language")
    
class BookStatus(str, Enum):
    """Book processing status"""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"

class BookInfo(BaseModel):
    """Book information"""
    id: str
    title: str
    author: Optional[str]
    file_path: str
    file_size: int
    total_pages: int
    total_chunks: int
    status: BookStatus
    upload_date: datetime
    last_updated: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)

class BookListResponse(PaginatedResponse):
    """Book list response"""
    books: List[BookInfo]

class BookUploadResponse(BaseResponse):
    """Book upload response"""
    book_id: str
    status: BookStatus
    processing_job_id: Optional[str] = None

class BookProcessingStatus(BaseModel):
    """Book processing status"""
    book_id: str
    status: BookStatus
    progress: float = Field(0.0, ge=0.0, le=1.0)
    current_step: str
    estimated_completion: Optional[datetime] = None
    error_message: Optional[str] = None
    chunks_processed: int = 0
    total_chunks: int = 0

# Analytics Models
class UsageStats(BaseModel):
    """Usage statistics"""
    total_searches: int
    total_books: int
    total_chunks: int
    active_users: int
    average_response_time: float
    cache_hit_rate: float
    storage_used_gb: float

class SearchAnalytics(BaseModel):
    """Search analytics data"""
    period: str
    total_searches: int
    unique_queries: int
    average_results_per_query: float
    top_queries: List[Dict[str, Any]]
    search_intent_distribution: Dict[str, int]
    user_satisfaction_score: Optional[float] = None

class SystemMetrics(BaseModel):
    """System performance metrics"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    database_connections: int
    active_sessions: int
    queue_depth: int

# Admin Models
class UserRole(str, Enum):
    """User roles"""
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"

class User(BaseModel):
    """User model for authentication and authorization"""
    id: str
    username: str
    email: str
    role: str  # Keep as string for compatibility
    full_name: Optional[str] = None
    is_active: bool = True
    created_at: str  # ISO format string
    last_login: Optional[str] = None

class UserInfo(BaseModel):
    """User information for API responses"""
    id: str
    username: str
    email: str
    role: UserRole
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    search_count: int = 0

class CreateUserRequest(BaseModel):
    """Create user request"""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
    password: str = Field(..., min_length=8)
    role: UserRole = UserRole.USER

class LoginRequest(BaseModel):
    """Login request"""
    username: str
    password: str

class LoginResponse(BaseResponse):
    """Login response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserInfo

class SystemConfigUpdate(BaseModel):
    """System configuration update"""
    embedding_model: Optional[str] = None
    max_file_size_mb: Optional[int] = None
    cache_ttl_hours: Optional[int] = None
    rate_limit_per_minute: Optional[int] = None
    enable_analytics: Optional[bool] = None

# Background Job Models
class JobStatus(str, Enum):
    """Background job status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobInfo(BaseModel):
    """Background job information"""
    job_id: str
    job_type: str
    status: JobStatus
    progress: float = 0.0
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

# Error Models
class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = False
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)