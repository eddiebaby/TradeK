"""
Pydantic models for TradeKnowledge API

These models define the structure of requests and responses
for all API endpoints.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# Base Models
class BaseResponse(BaseModel):
    """Base response model"""

    success: bool = True
    message: str | None = None
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
    def create(cls, items: list[Any], pagination: PaginationParams, total: int):
        pages = (total + pagination.size - 1) // pagination.size
        return cls(
            data=items,
            page=pagination.page,
            size=pagination.size,
            total=total,
            pages=pages,
        )


# Health Check Models
class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(..., description="Overall system status")
    timestamp: datetime
    components: dict[str, str] = Field(..., description="Component health status")
    version: str
    error: str | None = None


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
    intent: SearchIntent | None = Field(None, description="Search intent hint")
    filters: dict[str, Any] | None = Field(None, description="Search filters")
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
    page_number: int | None = Field(None, description="Page number in book")
    chunk_type: str = Field(..., description="Type of content chunk")
    metadata: dict[str, Any] = Field(default_factory=dict)
    highlights: list[str] = Field(
        default_factory=list, description="Highlighted snippets"
    )


class SearchResponse(BaseResponse):
    """Search response model"""

    query: str
    intent: str | None
    results: list[SearchResultItem]
    total_found: int
    processing_time_ms: float
    suggestions: list[str] = Field(default_factory=list)
    filters_applied: dict[str, Any] = Field(default_factory=dict)


class AutocompleteRequest(BaseModel):
    """Autocomplete request"""

    partial_query: str = Field(..., min_length=1, max_length=100)
    max_suggestions: int = Field(5, ge=1, le=20)


class AutocompleteResponse(BaseResponse):
    """Autocomplete response"""

    suggestions: list[str]
    query: str


# Book Management Models
class BookUploadRequest(BaseModel):
    """Book upload request metadata"""

    title: str | None = None
    author: str | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    category: str | None = None
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
    author: str | None
    file_path: str
    file_size: int
    total_pages: int
    total_chunks: int
    status: BookStatus
    upload_date: datetime
    last_updated: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class BookListResponse(PaginatedResponse):
    """Book list response"""

    books: list[BookInfo]


class BookUploadResponse(BaseResponse):
    """Book upload response"""

    book_id: str
    status: BookStatus
    processing_job_id: str | None = None


class BookProcessingStatus(BaseModel):
    """Book processing status"""

    book_id: str
    status: BookStatus
    progress: float = Field(0.0, ge=0.0, le=1.0)
    current_step: str
    estimated_completion: datetime | None = None
    error_message: str | None = None
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
    top_queries: list[dict[str, Any]]
    search_intent_distribution: dict[str, int]
    user_satisfaction_score: float | None = None


class SystemMetrics(BaseModel):
    """System performance metrics"""

    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: dict[str, float]
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
    full_name: str | None = None
    is_active: bool = True
    created_at: str  # ISO format string
    last_login: str | None = None


class UserInfo(BaseModel):
    """User information for API responses"""

    id: str
    username: str
    email: str
    role: UserRole
    is_active: bool
    created_at: datetime
    last_login: datetime | None = None
    search_count: int = 0


class CreateUserRequest(BaseModel):
    """Create user request"""

    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r"^[^@]+@[^@]+\.[^@]+$")
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

    embedding_model: str | None = None
    max_file_size_mb: int | None = None
    cache_ttl_hours: int | None = None
    rate_limit_per_minute: int | None = None
    enable_analytics: bool | None = None


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
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: dict[str, Any] | None = None
    error_message: str | None = None


# Error Models
class ErrorResponse(BaseModel):
    """Error response model"""

    success: bool = False
    error_code: str
    message: str
    details: dict[str, Any] | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
