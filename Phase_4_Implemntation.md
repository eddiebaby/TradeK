# Phase 4: Production Deployment Implementation Guide
## Building REST API, Monitoring, and Deployment Infrastructure

### Phase 4 Overview

With the core intelligence features complete in Phase 3, it's time to make TradeKnowledge production-ready. Phase 4 focuses on building robust APIs, comprehensive monitoring, automated deployment, and enterprise-grade features that allow the system to scale and operate reliably in production environments.

**Key Goals for Phase 4:**
- Build comprehensive REST API with authentication
- Implement monitoring, logging, and alerting
- Create automated deployment and scaling
- Add user management and access control
- Build administrative dashboards
- Implement backup and disaster recovery

---

## Production API Development

### FastAPI REST Service Architecture

Let's build a comprehensive API that exposes all TradeKnowledge functionality through well-designed REST endpoints.

#### Create API Foundation

```python
# Create src/api/main.py
cat > src/api/main.py << 'EOF'
"""
TradeKnowledge REST API Server

Production-ready FastAPI server with authentication, rate limiting,
monitoring, and comprehensive endpoints for all system functionality.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog

# Import our core components
from ..core.config import get_config
from ..search.unified_search import UnifiedSearchEngine
from ..ingestion.enhanced_book_processor import EnhancedBookProcessor
from ..utils.cache_manager import CacheManager
from .models import *
from .auth import AuthManager
from .middleware import RateLimitMiddleware, RequestLoggingMiddleware
from .metrics import MetricsCollector

# Setup structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Global application state
app_state = {
    "search_engine": None,
    "book_processor": None,
    "cache_manager": None,
    "auth_manager": None,
    "metrics": None,
    "config": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting TradeKnowledge API server...")
    
    try:
        # Load configuration
        config = get_config()
        app_state["config"] = config
        
        # Initialize core components
        logger.info("Initializing core components...")
        
        cache_manager = CacheManager()
        await cache_manager.initialize()
        app_state["cache_manager"] = cache_manager
        
        book_processor = EnhancedBookProcessor()
        await book_processor.initialize()
        app_state["book_processor"] = book_processor
        
        search_engine = UnifiedSearchEngine()
        await search_engine.initialize()
        app_state["search_engine"] = search_engine
        
        auth_manager = AuthManager(config.api.auth)
        app_state["auth_manager"] = auth_manager
        
        metrics = MetricsCollector()
        await metrics.initialize()
        app_state["metrics"] = metrics
        
        logger.info("✅ All components initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error("Failed to initialize application", error=str(e))
        raise
    
    finally:
        # Cleanup
        logger.info("Shutting down TradeKnowledge API server...")
        
        if app_state.get("search_engine"):
            await app_state["search_engine"].cleanup()
        
        if app_state.get("book_processor"):
            await app_state["book_processor"].cleanup()
        
        if app_state.get("cache_manager"):
            await app_state["cache_manager"].cleanup()
            
        if app_state.get("metrics"):
            await app_state["metrics"].cleanup()
        
        logger.info("✅ Cleanup completed")

# Create FastAPI app
app = FastAPI(
    title="TradeKnowledge API",
    description="Intelligent Knowledge Assistant for Algorithmic Trading",
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestLoggingMiddleware)

# Security
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Extract and validate user from JWT token"""
    auth_manager = app_state.get("auth_manager")
    if not auth_manager:
        raise HTTPException(status_code=500, detail="Authentication not initialized")
    
    try:
        user = await auth_manager.verify_token(credentials.credentials)
        return user
    except Exception as e:
        logger.warning("Authentication failed", error=str(e))
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

# Dependencies
async def get_search_engine():
    """Get search engine instance"""
    engine = app_state.get("search_engine")
    if not engine:
        raise HTTPException(status_code=500, detail="Search engine not initialized")
    return engine

async def get_book_processor():
    """Get book processor instance"""
    processor = app_state.get("book_processor")
    if not processor:
        raise HTTPException(status_code=500, detail="Book processor not initialized")
    return processor

async def get_metrics():
    """Get metrics collector"""
    metrics = app_state.get("metrics")
    if not metrics:
        raise HTTPException(status_code=500, detail="Metrics not initialized")
    return metrics

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for load balancers"""
    try:
        # Check component health
        search_healthy = app_state.get("search_engine") is not None
        processor_healthy = app_state.get("book_processor") is not None
        cache_healthy = app_state.get("cache_manager") is not None
        
        overall_status = "healthy" if all([search_healthy, processor_healthy, cache_healthy]) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.utcnow(),
            components={
                "search_engine": "healthy" if search_healthy else "unhealthy",
                "book_processor": "healthy" if processor_healthy else "unhealthy",
                "cache_manager": "healthy" if cache_healthy else "unhealthy"
            },
            version="4.0.0"
        )
    
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow(),
            components={},
            version="4.0.0",
            error=str(e)
        )

# Include API routers
from .routers import search, ingestion, admin, analytics

app.include_router(search.router, prefix="/api/v1/search", tags=["Search"])
app.include_router(ingestion.router, prefix="/api/v1/books", tags=["Book Management"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["Administration"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["Analytics"])

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with structured logging"""
    logger.error(
        "Unhandled exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error_id": str(hash(str(exc))),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        workers=1,     # Use multiple workers in production
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        }
    )
EOF
```

#### Create API Models

```python
# Create src/api/models.py
cat > src/api/models.py << 'EOF'
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

class UserInfo(BaseModel):
    """User information"""
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
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
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
EOF
```

### Search API Router

```python
# Create src/api/routers/search.py
cat > src/api/routers/search.py << 'EOF'
"""
Search API endpoints

Provides comprehensive search functionality with intent detection,
filtering, autocomplete, and analytics integration.
"""

import asyncio
import time
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
import structlog

from ..models import *
from ..main import get_search_engine, get_current_user, get_metrics
from ...search.unified_search import UnifiedSearchEngine

logger = structlog.get_logger(__name__)

router = APIRouter()

@router.post("/query", response_model=SearchResponse)
async def search_knowledge(
    request: SearchRequest,
    background_tasks: BackgroundTasks,
    search_engine: UnifiedSearchEngine = Depends(get_search_engine),
    user = Depends(get_current_user),
    metrics = Depends(get_metrics)
):
    """
    Perform intelligent search across all indexed content
    
    This endpoint provides the main search functionality with:
    - Automatic intent detection
    - Semantic and exact search
    - Result ranking and filtering
    - Query suggestions
    """
    start_time = time.time()
    
    try:
        logger.info(
            "Search request received",
            user_id=user.id,
            query=request.query,
            intent=request.intent,
            max_results=request.max_results
        )
        
        # Perform search
        search_result = await search_engine.search(
            query=request.query,
            intent=request.intent.value if request.intent else None,
            filters=request.filters or {},
            max_results=request.max_results,
            min_score=request.min_score,
            user_id=user.id
        )
        
        # Convert to API response format
        results = []
        for item in search_result.get('results', []):
            results.append(SearchResultItem(
                id=item['id'],
                title=item.get('title', ''),
                content=item.get('content', ''),
                score=item.get('score', 0.0),
                book_id=item.get('book_id', ''),
                book_title=item.get('book_title', ''),
                page_number=item.get('page_number'),
                chunk_type=item.get('chunk_type', 'text'),
                metadata=item.get('metadata', {}),
                highlights=item.get('highlights', [])
            ))
        
        processing_time = (time.time() - start_time) * 1000
        
        response = SearchResponse(
            query=request.query,
            intent=request.intent.value if request.intent else search_result.get('detected_intent'),
            results=results,
            total_found=search_result.get('total_found', len(results)),
            processing_time_ms=processing_time,
            suggestions=search_result.get('suggestions', []),
            filters_applied=search_result.get('filters_applied', {})
        )
        
        # Log analytics in background
        background_tasks.add_task(
            log_search_analytics,
            metrics,
            user.id,
            request.query,
            len(results),
            processing_time,
            request.intent
        )
        
        logger.info(
            "Search completed",
            user_id=user.id,
            query=request.query,
            results_count=len(results),
            processing_time_ms=processing_time
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "Search failed",
            user_id=user.id,
            query=request.query,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/autocomplete", response_model=AutocompleteResponse)
async def autocomplete_query(
    q: str = Query(..., min_length=1, max_length=100, description="Partial query"),
    max_suggestions: int = Query(5, ge=1, le=20, description="Maximum suggestions"),
    search_engine: UnifiedSearchEngine = Depends(get_search_engine),
    user = Depends(get_current_user)
):
    """
    Get autocomplete suggestions for partial queries
    
    Returns intelligent suggestions based on:
    - Query history
    - Popular searches
    - Trading terminology
    - Content analysis
    """
    try:
        suggestions = await search_engine.get_suggestions(
            partial_query=q,
            max_suggestions=max_suggestions,
            user_id=user.id
        )
        
        return AutocompleteResponse(
            query=q,
            suggestions=suggestions
        )
        
    except Exception as e:
        logger.error("Autocomplete failed", query=q, error=str(e))
        raise HTTPException(status_code=500, detail=f"Autocomplete failed: {str(e)}")

@router.get("/similar/{result_id}")
async def find_similar_content(
    result_id: str,
    max_results: int = Query(10, ge=1, le=50),
    search_engine: UnifiedSearchEngine = Depends(get_search_engine),
    user = Depends(get_current_user)
):
    """
    Find content similar to a specific search result
    
    Useful for discovering related concepts, implementations,
    or alternative approaches to trading strategies.
    """
    try:
        similar_results = await search_engine.find_similar(
            result_id=result_id,
            max_results=max_results,
            user_id=user.id
        )
        
        results = []
        for item in similar_results:
            results.append(SearchResultItem(
                id=item['id'],
                title=item.get('title', ''),
                content=item.get('content', ''),
                score=item.get('score', 0.0),
                book_id=item.get('book_id', ''),
                book_title=item.get('book_title', ''),
                page_number=item.get('page_number'),
                chunk_type=item.get('chunk_type', 'text'),
                metadata=item.get('metadata', {}),
                highlights=item.get('highlights', [])
            ))
        
        return SearchResponse(
            query=f"Similar to: {result_id}",
            results=results,
            total_found=len(results),
            processing_time_ms=0.0  # Not tracked for similarity
        )
        
    except Exception as e:
        logger.error("Similar content search failed", result_id=result_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Similar search failed: {str(e)}")

@router.post("/feedback")
async def submit_search_feedback(
    result_id: str,
    rating: int = Query(..., ge=1, le=5, description="Rating 1-5"),
    query: str = Query(..., description="Original search query"),
    feedback: str = Query(None, description="Optional feedback text"),
    search_engine: UnifiedSearchEngine = Depends(get_search_engine),
    user = Depends(get_current_user)
):
    """
    Submit feedback on search result relevance
    
    This helps improve the ranking algorithm through
    learning to rank techniques.
    """
    try:
        await search_engine.submit_feedback(
            user_id=user.id,
            query=query,
            result_id=result_id,
            rating=rating,
            feedback=feedback
        )
        
        logger.info(
            "Search feedback received",
            user_id=user.id,
            result_id=result_id,
            rating=rating,
            query=query
        )
        
        return {"success": True, "message": "Feedback submitted successfully"}
        
    except Exception as e:
        logger.error("Feedback submission failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")

@router.get("/trending")
async def get_trending_queries(
    period: str = Query("24h", regex="^(1h|24h|7d|30d)$"),
    limit: int = Query(20, ge=1, le=100),
    search_engine: UnifiedSearchEngine = Depends(get_search_engine),
    user = Depends(get_current_user)
):
    """
    Get trending search queries
    
    Shows what other users are searching for, helping
    discover popular topics and strategies.
    """
    try:
        trending = await search_engine.get_trending_queries(
            period=period,
            limit=limit
        )
        
        return {
            "period": period,
            "trending_queries": trending,
            "generated_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error("Trending queries failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Trending queries failed: {str(e)}")

@router.get("/export/{format}")
async def export_search_results(
    format: str = Query(..., regex="^(csv|json|xlsx)$"),
    query: str = Query(..., description="Search query to export"),
    max_results: int = Query(1000, ge=1, le=10000),
    search_engine: UnifiedSearchEngine = Depends(get_search_engine),
    user = Depends(get_current_user)
):
    """
    Export search results in various formats
    
    Useful for analysis, reporting, or integration with
    other trading systems.
    """
    try:
        # Perform search
        search_result = await search_engine.search(
            query=query,
            max_results=max_results,
            user_id=user.id
        )
        
        # Generate export
        if format == "csv":
            content = await generate_csv_export(search_result['results'])
            media_type = "text/csv"
            filename = f"search_results_{int(time.time())}.csv"
        elif format == "json":
            content = await generate_json_export(search_result['results'])
            media_type = "application/json"
            filename = f"search_results_{int(time.time())}.json"
        elif format == "xlsx":
            content = await generate_xlsx_export(search_result['results'])
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            filename = f"search_results_{int(time.time())}.xlsx"
        
        return StreamingResponse(
            iter([content]),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        logger.error("Export failed", format=format, query=query, error=str(e))
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

# Helper functions
async def log_search_analytics(metrics, user_id: str, query: str, 
                             results_count: int, processing_time: float, 
                             intent: Optional[SearchIntent]):
    """Log search analytics in background"""
    try:
        await metrics.log_search(
            user_id=user_id,
            query=query,
            results_count=results_count,
            processing_time=processing_time,
            intent=intent.value if intent else None
        )
    except Exception as e:
        logger.error("Failed to log search analytics", error=str(e))

async def generate_csv_export(results: List[Dict[str, Any]]) -> str:
    """Generate CSV export of search results"""
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        'id', 'title', 'content', 'score', 'book_title', 'page_number', 'chunk_type'
    ])
    
    writer.writeheader()
    for result in results:
        writer.writerow({
            'id': result['id'],
            'title': result.get('title', ''),
            'content': result.get('content', '')[:500],  # Truncate for CSV
            'score': result.get('score', 0.0),
            'book_title': result.get('book_title', ''),
            'page_number': result.get('page_number', ''),
            'chunk_type': result.get('chunk_type', 'text')
        })
    
    return output.getvalue()

async def generate_json_export(results: List[Dict[str, Any]]) -> str:
    """Generate JSON export of search results"""
    import json
    return json.dumps(results, indent=2, default=str)

async def generate_xlsx_export(results: List[Dict[str, Any]]) -> bytes:
    """Generate Excel export of search results"""
    import openpyxl
    from io import BytesIO
    
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Search Results"
    
    # Headers
    headers = ['ID', 'Title', 'Content', 'Score', 'Book Title', 'Page', 'Type']
    for col, header in enumerate(headers, 1):
        ws.cell(row=1, column=col, value=header)
    
    # Data
    for row, result in enumerate(results, 2):
        ws.cell(row=row, column=1, value=result['id'])
        ws.cell(row=row, column=2, value=result.get('title', ''))
        ws.cell(row=row, column=3, value=result.get('content', '')[:1000])
        ws.cell(row=row, column=4, value=result.get('score', 0.0))
        ws.cell(row=row, column=5, value=result.get('book_title', ''))
        ws.cell(row=row, column=6, value=result.get('page_number', ''))
        ws.cell(row=row, column=7, value=result.get('chunk_type', 'text'))
    
    # Save to bytes
    output = BytesIO()
    wb.save(output)
    output.seek(0)
    return output.read()
EOF
```

### Book Management API Router

```python
# Create src/api/routers/ingestion.py
cat > src/api/routers/ingestion.py << 'EOF'
"""
Book management API endpoints

Handles book upload, processing, status tracking, and metadata management.
"""

import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
import structlog

from ..models import *
from ..main import get_book_processor, get_current_user
from ...ingestion.enhanced_book_processor import EnhancedBookProcessor

logger = structlog.get_logger(__name__)

router = APIRouter()

@router.post("/upload", response_model=BookUploadResponse)
async def upload_book(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Book file to upload"),
    title: str = Form(None, description="Book title"),
    author: str = Form(None, description="Book author"),
    description: str = Form(None, description="Book description"),
    tags: str = Form("", description="Comma-separated tags"),
    category: str = Form(None, description="Book category"),
    language: str = Form("en", description="Book language"),
    book_processor: EnhancedBookProcessor = Depends(get_book_processor),
    user = Depends(get_current_user)
):
    """
    Upload a new book for processing
    
    Supported formats:
    - PDF (including scanned)
    - EPUB
    - Jupyter Notebooks (.ipynb)
    
    The file will be processed asynchronously and indexed for search.
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file size (100MB limit)
        if file.size and file.size > 100 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large (max 100MB)")
        
        # Check file type
        supported_extensions = ['.pdf', '.epub', '.ipynb']
        file_path = Path(file.filename)
        if file_path.suffix.lower() not in supported_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Supported: {', '.join(supported_extensions)}"
            )
        
        # Generate unique book ID
        book_id = str(uuid.uuid4())
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_path.suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = Path(tmp_file.name)
        
        # Parse tags
        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
        
        # Create book metadata
        book_metadata = BookUploadRequest(
            title=title or file_path.stem,
            author=author,
            description=description,
            tags=tag_list,
            category=category,
            language=language
        )
        
        logger.info(
            "Book upload started",
            user_id=user.id,
            book_id=book_id,
            filename=file.filename,
            file_size=file.size
        )
        
        # Start processing in background
        job_id = str(uuid.uuid4())
        background_tasks.add_task(
            process_book_background,
            book_processor,
            book_id,
            tmp_file_path,
            book_metadata,
            user.id,
            job_id
        )
        
        return BookUploadResponse(
            book_id=book_id,
            status=BookStatus.UPLOADED,
            processing_job_id=job_id,
            message=f"Book '{book_metadata.title}' uploaded successfully and is being processed"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Book upload failed", filename=file.filename, error=str(e))
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/list", response_model=BookListResponse)
async def list_books(
    pagination: PaginationParams = Depends(),
    category: str = None,
    author: str = None,
    tags: str = None,
    status: BookStatus = None,
    book_processor: EnhancedBookProcessor = Depends(get_book_processor),
    user = Depends(get_current_user)
):
    """
    List books with filtering and pagination
    
    Supports filtering by:
    - Category
    - Author
    - Tags (comma-separated)
    - Processing status
    """
    try:
        # Build filters
        filters = {}
        if category:
            filters['category'] = category
        if author:
            filters['author'] = author
        if tags:
            filters['tags'] = [tag.strip() for tag in tags.split(",")]
        if status:
            filters['status'] = status.value
        
        # Get books
        books_data = await book_processor.list_books(
            offset=pagination.offset,
            limit=pagination.size,
            filters=filters
        )
        
        # Convert to API format
        books = []
        for book_data in books_data['books']:
            books.append(BookInfo(
                id=book_data['id'],
                title=book_data['title'],
                author=book_data.get('author'),
                file_path=book_data['file_path'],
                file_size=book_data['file_size'],
                total_pages=book_data['total_pages'],
                total_chunks=book_data['total_chunks'],
                status=BookStatus(book_data['status']),
                upload_date=book_data['upload_date'],
                last_updated=book_data['last_updated'],
                metadata=book_data.get('metadata', {}),
                tags=book_data.get('tags', [])
            ))
        
        return BookListResponse.create(
            items=books,
            pagination=pagination,
            total=books_data['total']
        )
        
    except Exception as e:
        logger.error("List books failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list books: {str(e)}")

@router.get("/{book_id}", response_model=BookInfo)
async def get_book_details(
    book_id: str,
    book_processor: EnhancedBookProcessor = Depends(get_book_processor),
    user = Depends(get_current_user)
):
    """Get detailed information about a specific book"""
    try:
        book_data = await book_processor.get_book(book_id)
        
        if not book_data:
            raise HTTPException(status_code=404, detail="Book not found")
        
        return BookInfo(
            id=book_data['id'],
            title=book_data['title'],
            author=book_data.get('author'),
            file_path=book_data['file_path'],
            file_size=book_data['file_size'],
            total_pages=book_data['total_pages'],
            total_chunks=book_data['total_chunks'],
            status=BookStatus(book_data['status']),
            upload_date=book_data['upload_date'],
            last_updated=book_data['last_updated'],
            metadata=book_data.get('metadata', {}),
            tags=book_data.get('tags', [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Get book failed", book_id=book_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get book: {str(e)}")

@router.get("/{book_id}/status", response_model=BookProcessingStatus)
async def get_processing_status(
    book_id: str,
    book_processor: EnhancedBookProcessor = Depends(get_book_processor),
    user = Depends(get_current_user)
):
    """Get processing status for a book"""
    try:
        status_data = await book_processor.get_processing_status(book_id)
        
        if not status_data:
            raise HTTPException(status_code=404, detail="Book not found")
        
        return BookProcessingStatus(
            book_id=book_id,
            status=BookStatus(status_data['status']),
            progress=status_data.get('progress', 0.0),
            current_step=status_data.get('current_step', 'Unknown'),
            estimated_completion=status_data.get('estimated_completion'),
            error_message=status_data.get('error_message'),
            chunks_processed=status_data.get('chunks_processed', 0),
            total_chunks=status_data.get('total_chunks', 0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Get processing status failed", book_id=book_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@router.put("/{book_id}/metadata")
async def update_book_metadata(
    book_id: str,
    metadata: BookUploadRequest,
    book_processor: EnhancedBookProcessor = Depends(get_book_processor),
    user = Depends(get_current_user)
):
    """Update book metadata"""
    try:
        await book_processor.update_book_metadata(book_id, metadata.dict())
        
        logger.info("Book metadata updated", book_id=book_id, user_id=user.id)
        
        return {"success": True, "message": "Metadata updated successfully"}
        
    except Exception as e:
        logger.error("Update metadata failed", book_id=book_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to update metadata: {str(e)}")

@router.post("/{book_id}/reprocess")
async def reprocess_book(
    book_id: str,
    background_tasks: BackgroundTasks,
    force: bool = False,
    book_processor: EnhancedBookProcessor = Depends(get_book_processor),
    user = Depends(get_current_user)
):
    """Reprocess a book (useful after algorithm updates)"""
    try:
        # Check if book exists
        book_data = await book_processor.get_book(book_id)
        if not book_data:
            raise HTTPException(status_code=404, detail="Book not found")
        
        # Check if already processing
        if book_data['status'] in ['processing', 'indexing'] and not force:
            raise HTTPException(
                status_code=409, 
                detail="Book is already being processed. Use force=true to override."
            )
        
        # Start reprocessing
        job_id = str(uuid.uuid4())
        background_tasks.add_task(
            reprocess_book_background,
            book_processor,
            book_id,
            user.id,
            job_id
        )
        
        logger.info("Book reprocessing started", book_id=book_id, user_id=user.id)
        
        return {
            "success": True,
            "message": "Book reprocessing started",
            "job_id": job_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Reprocess failed", book_id=book_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to reprocess: {str(e)}")

@router.delete("/{book_id}")
async def delete_book(
    book_id: str,
    book_processor: EnhancedBookProcessor = Depends(get_book_processor),
    user = Depends(get_current_user)
):
    """Delete a book and all its data"""
    try:
        await book_processor.delete_book(book_id)
        
        logger.info("Book deleted", book_id=book_id, user_id=user.id)
        
        return {"success": True, "message": "Book deleted successfully"}
        
    except Exception as e:
        logger.error("Delete book failed", book_id=book_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to delete book: {str(e)}")

@router.get("/{book_id}/chunks")
async def get_book_chunks(
    book_id: str,
    pagination: PaginationParams = Depends(),
    chunk_type: str = None,
    book_processor: EnhancedBookProcessor = Depends(get_book_processor),
    user = Depends(get_current_user)
):
    """Get chunks for a specific book"""
    try:
        chunks_data = await book_processor.get_book_chunks(
            book_id=book_id,
            offset=pagination.offset,
            limit=pagination.size,
            chunk_type=chunk_type
        )
        
        return {
            "book_id": book_id,
            "chunks": chunks_data['chunks'],
            "total": chunks_data['total'],
            "page": pagination.page,
            "size": pagination.size
        }
        
    except Exception as e:
        logger.error("Get book chunks failed", book_id=book_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get chunks: {str(e)}")

# Background task functions
async def process_book_background(
    book_processor: EnhancedBookProcessor,
    book_id: str,
    file_path: Path,
    metadata: BookUploadRequest,
    user_id: str,
    job_id: str
):
    """Process book in background"""
    try:
        logger.info("Starting book processing", book_id=book_id, job_id=job_id)
        
        # Process the book
        result = await book_processor.add_book(
            file_path=file_path,
            book_id=book_id,
            metadata=metadata.dict(),
            user_id=user_id
        )
        
        # Clean up temporary file
        file_path.unlink(missing_ok=True)
        
        logger.info("Book processing completed", book_id=book_id, job_id=job_id)
        
    except Exception as e:
        logger.error("Book processing failed", book_id=book_id, job_id=job_id, error=str(e))
        
        # Update status to failed
        await book_processor.update_book_status(
            book_id=book_id,
            status="failed",
            error_message=str(e)
        )
        
        # Clean up temporary file
        file_path.unlink(missing_ok=True)

async def reprocess_book_background(
    book_processor: EnhancedBookProcessor,
    book_id: str,
    user_id: str,
    job_id: str
):
    """Reprocess book in background"""
    try:
        logger.info("Starting book reprocessing", book_id=book_id, job_id=job_id)
        
        await book_processor.reprocess_book(book_id)
        
        logger.info("Book reprocessing completed", book_id=book_id, job_id=job_id)
        
    except Exception as e:
        logger.error("Book reprocessing failed", book_id=book_id, job_id=job_id, error=str(e))
        
        await book_processor.update_book_status(
            book_id=book_id,
            status="failed",
            error_message=str(e)
        )
EOF
```

### Monitoring and Metrics System

```python
# Create src/api/metrics.py
cat > src/api/metrics.py << 'EOF'
"""
Metrics collection and monitoring for TradeKnowledge

Provides comprehensive monitoring of system performance,
user behavior, and operational metrics.
"""

import asyncio
import time
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
import structlog

logger = structlog.get_logger(__name__)

@dataclass
class MetricPoint:
    """Individual metric data point"""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class SearchMetric:
    """Search-specific metrics"""
    query: str
    user_id: str
    results_count: int
    processing_time: float
    intent: Optional[str]
    timestamp: datetime

class MetricsCollector:
    """
    Comprehensive metrics collection system
    
    Collects and aggregates:
    - System performance metrics
    - Search analytics
    - User behavior data
    - Error rates and types
    - Cache performance
    """
    
    def __init__(self, max_history: int = 10000):
        """Initialize metrics collector"""
        self.max_history = max_history
        
        # Metric storage
        self.system_metrics: deque = deque(maxlen=max_history)
        self.search_metrics: deque = deque(maxlen=max_history)
        self.error_metrics: deque = deque(maxlen=max_history)
        self.user_metrics: Dict[str, Any] = defaultdict(lambda: {
            'search_count': 0,
            'last_activity': None,
            'total_processing_time': 0.0,
            'avg_results_per_query': 0.0
        })
        
        # Performance counters
        self.request_count = 0
        self.error_count = 0
        self.start_time = datetime.utcnow()
        
        # Background collection task
        self._collection_task = None
        self._running = False
    
    async def initialize(self):
        """Initialize metrics collection"""
        self._running = True
        self._collection_task = asyncio.create_task(self._collect_system_metrics())
        logger.info("Metrics collector initialized")
    
    async def cleanup(self):
        """Cleanup metrics collection"""
        self._running = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        logger.info("Metrics collector stopped")
    
    async def _collect_system_metrics(self):
        """Collect system metrics periodically"""
        while self._running:
            try:
                # CPU and memory
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Network I/O
                net_io = psutil.net_io_counters()
                
                timestamp = datetime.utcnow()
                
                # Store metrics
                self.system_metrics.append(MetricPoint(
                    timestamp=timestamp,
                    value=cpu_percent,
                    tags={'metric': 'cpu_usage'}
                ))
                
                self.system_metrics.append(MetricPoint(
                    timestamp=timestamp,
                    value=memory.percent,
                    tags={'metric': 'memory_usage'}
                ))
                
                self.system_metrics.append(MetricPoint(
                    timestamp=timestamp,
                    value=disk.percent,
                    tags={'metric': 'disk_usage'}
                ))
                
                self.system_metrics.append(MetricPoint(
                    timestamp=timestamp,
                    value=net_io.bytes_sent,
                    tags={'metric': 'network_bytes_sent'}
                ))
                
                self.system_metrics.append(MetricPoint(
                    timestamp=timestamp,
                    value=net_io.bytes_recv,
                    tags={'metric': 'network_bytes_recv'}
                ))
                
            except Exception as e:
                logger.error("System metrics collection failed", error=str(e))
            
            await asyncio.sleep(30)  # Collect every 30 seconds
    
    async def log_search(self, user_id: str, query: str, results_count: int, 
                        processing_time: float, intent: Optional[str] = None):
        """Log search metrics"""
        metric = SearchMetric(
            query=query,
            user_id=user_id,
            results_count=results_count,
            processing_time=processing_time,
            intent=intent,
            timestamp=datetime.utcnow()
        )
        
        self.search_metrics.append(metric)
        
        # Update user metrics
        user_data = self.user_metrics[user_id]
        user_data['search_count'] += 1
        user_data['last_activity'] = metric.timestamp
        user_data['total_processing_time'] += processing_time
        
        # Calculate rolling average
        user_searches = [m for m in self.search_metrics if m.user_id == user_id]
        if user_searches:
            total_results = sum(m.results_count for m in user_searches)
            user_data['avg_results_per_query'] = total_results / len(user_searches)
        
        self.request_count += 1
    
    async def log_error(self, error_type: str, error_message: str, 
                       user_id: Optional[str] = None, context: Dict[str, Any] = None):
        """Log error metrics"""
        self.error_metrics.append(MetricPoint(
            timestamp=datetime.utcnow(),
            value=1,
            tags={
                'error_type': error_type,
                'user_id': user_id or 'unknown',
                'context': str(context or {})
            }
        ))
        
        self.error_count += 1
    
    async def get_system_metrics(self, period_hours: int = 24) -> Dict[str, Any]:
        """Get system performance metrics"""
        cutoff = datetime.utcnow() - timedelta(hours=period_hours)
        
        recent_metrics = [m for m in self.system_metrics if m.timestamp > cutoff]
        
        # Group by metric type
        metrics_by_type = defaultdict(list)
        for metric in recent_metrics:
            metric_type = metric.tags.get('metric', 'unknown')
            metrics_by_type[metric_type].append(metric.value)
        
        # Calculate averages
        result = {}
        for metric_type, values in metrics_by_type.items():
            if values:
                result[metric_type] = {
                    'current': values[-1] if values else 0,
                    'average': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
        
        return result
    
    async def get_search_analytics(self, period_hours: int = 24) -> Dict[str, Any]:
        """Get search analytics"""
        cutoff = datetime.utcnow() - timedelta(hours=period_hours)
        
        recent_searches = [m for m in self.search_metrics if m.timestamp > cutoff]
        
        if not recent_searches:
            return {
                'total_searches': 0,
                'unique_queries': 0,
                'unique_users': 0,
                'average_processing_time': 0.0,
                'average_results_per_query': 0.0,
                'top_queries': [],
                'intent_distribution': {}
            }
        
        # Calculate analytics
        total_searches = len(recent_searches)
        unique_queries = len(set(m.query for m in recent_searches))
        unique_users = len(set(m.user_id for m in recent_searches))
        
        avg_processing_time = sum(m.processing_time for m in recent_searches) / total_searches
        avg_results = sum(m.results_count for m in recent_searches) / total_searches
        
        # Top queries
        query_counts = defaultdict(int)
        for search in recent_searches:
            query_counts[search.query] += 1
        
        top_queries = [
            {'query': query, 'count': count}
            for query, count in sorted(query_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        # Intent distribution
        intent_counts = defaultdict(int)
        for search in recent_searches:
            intent = search.intent or 'unknown'
            intent_counts[intent] += 1
        
        return {
            'total_searches': total_searches,
            'unique_queries': unique_queries,
            'unique_users': unique_users,
            'average_processing_time': avg_processing_time,
            'average_results_per_query': avg_results,
            'top_queries': top_queries,
            'intent_distribution': dict(intent_counts)
        }
    
    async def get_user_analytics(self, period_hours: int = 24) -> Dict[str, Any]:
        """Get user behavior analytics"""
        cutoff = datetime.utcnow() - timedelta(hours=period_hours)
        
        # Active users in period
        active_users = set()
        for search in self.search_metrics:
            if search.timestamp > cutoff:
                active_users.add(search.user_id)
        
        # User activity patterns
        hourly_activity = defaultdict(int)
        for search in self.search_metrics:
            if search.timestamp > cutoff:
                hour = search.timestamp.hour
                hourly_activity[hour] += 1
        
        return {
            'active_users': len(active_users),
            'total_registered_users': len(self.user_metrics),
            'hourly_activity': dict(hourly_activity),
            'most_active_users': self._get_most_active_users(period_hours)
        }
    
    def _get_most_active_users(self, period_hours: int) -> List[Dict[str, Any]]:
        """Get most active users in period"""
        cutoff = datetime.utcnow() - timedelta(hours=period_hours)
        
        user_activity = defaultdict(int)
        for search in self.search_metrics:
            if search.timestamp > cutoff:
                user_activity[search.user_id] += 1
        
        return [
            {'user_id': user_id, 'search_count': count}
            for user_id, count in sorted(user_activity.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
    
    async def get_error_metrics(self, period_hours: int = 24) -> Dict[str, Any]:
        """Get error metrics"""
        cutoff = datetime.utcnow() - timedelta(hours=period_hours)
        
        recent_errors = [m for m in self.error_metrics if m.timestamp > cutoff]
        
        if not recent_errors:
            return {
                'total_errors': 0,
                'error_rate': 0.0,
                'error_types': {}
            }
        
        # Error types
        error_types = defaultdict(int)
        for error in recent_errors:
            error_type = error.tags.get('error_type', 'unknown')
            error_types[error_type] += 1
        
        # Error rate
        total_requests = len([m for m in self.search_metrics if m.timestamp > cutoff])
        error_rate = len(recent_errors) / max(total_requests, 1)
        
        return {
            'total_errors': len(recent_errors),
            'error_rate': error_rate,
            'error_types': dict(error_types)
        }
    
    async def get_uptime_info(self) -> Dict[str, Any]:
        """Get system uptime information"""
        uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            'uptime_seconds': uptime_seconds,
            'uptime_hours': uptime_seconds / 3600,
            'start_time': self.start_time.isoformat(),
            'total_requests': self.request_count,
            'total_errors': self.error_count,
            'requests_per_hour': self.request_count / max(uptime_seconds / 3600, 1)
        }
EOF
```

## Authentication and Authorization System

The authentication system provides enterprise-grade security for TradeKnowledge, implementing JWT-based authentication with role-based access control, session management, and comprehensive security features.

### Core Authentication Implementation

Let's complete the authentication manager with full functionality:

```python
# Continue src/api/auth.py
        # Create default admin user if none exists
        admin_exists = self.db.query(
            "SELECT COUNT(*) FROM users WHERE role = 'admin'",
            one=True
        )[0]
        
        if not admin_exists:
            self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin user"""
        import uuid
        admin_id = str(uuid.uuid4())
        admin_password = "admin123!@#"  # Change immediately in production
        
        hashed = bcrypt.hashpw(
            admin_password.encode('utf-8'),
            bcrypt.gensalt()
        )
        
        self.db.execute(
            """INSERT INTO users 
            (id, username, email, password_hash, role) 
            VALUES (?, ?, ?, ?, ?)""",
            (admin_id, "admin", "admin@tradeknowledge.local", 
             hashed.decode('utf-8'), "admin")
        )
        
        logger.warning(
            "Created default admin user",
            username="admin",
            password="admin123!@#",
            action_required="CHANGE PASSWORD IMMEDIATELY"
        )
    
    async def register_user(
        self, 
        username: str, 
        email: str, 
        password: str,
        role: str = "user"
    ) -> User:
        """Register a new user"""
        import uuid
        
        # Validate inputs
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters")
        
        if role not in ["user", "premium", "admin"]:
            raise ValueError(f"Invalid role: {role}")
        
        # Check if user exists
        existing = self.db.query(
            "SELECT id FROM users WHERE username = ? OR email = ?",
            (username, email)
        )
        
        if existing:
            raise ValueError("Username or email already exists")
        
        # Hash password
        hashed = bcrypt.hashpw(
            password.encode('utf-8'),
            bcrypt.gensalt()
        )
        
        # Create user
        user_id = str(uuid.uuid4())
        self.db.execute(
            """INSERT INTO users 
            (id, username, email, password_hash, role) 
            VALUES (?, ?, ?, ?, ?)""",
            (user_id, username, email, hashed.decode('utf-8'), role)
        )
        
        logger.info("User registered", user_id=user_id, username=username)
        
        return User(
            id=user_id,
            username=username,
            email=email,
            role=role,
            is_active=True,
            created_at=datetime.utcnow()
        )
    
    async def authenticate_user(
        self, 
        username: str, 
        password: str
    ) -> Optional[User]:
        """Authenticate user with username/password"""
        # Get user from database
        user_data = self.db.query(
            """SELECT id, username, email, password_hash, role, 
               is_active, created_at, last_login 
               FROM users WHERE username = ? OR email = ?""",
            (username, username),
            one=True
        )
        
        if not user_data:
            logger.warning("Authentication failed - user not found", 
                         username=username)
            return None
        
        # Verify password
        if not bcrypt.checkpw(
            password.encode('utf-8'),
            user_data[3].encode('utf-8')
        ):
            logger.warning("Authentication failed - invalid password",
                         username=username)
            return None
        
        # Check if user is active
        if not user_data[5]:
            logger.warning("Authentication failed - user inactive",
                         username=username)
            return None
        
        # Update last login
        self.db.execute(
            "UPDATE users SET last_login = ? WHERE id = ?",
            (datetime.utcnow(), user_data[0])
        )
        
        logger.info("User authenticated successfully",
                   user_id=user_data[0],
                   username=user_data[1])
        
        return User(
            id=user_data[0],
            username=user_data[1],
            email=user_data[2],
            role=user_data[4],
            is_active=user_data[5],
            created_at=user_data[6],
            last_login=datetime.utcnow()
        )
    
    def generate_tokens(self, user: User) -> Dict[str, Any]:
        """Generate access and refresh tokens"""
        # Access token payload
        access_payload = {
            "sub": user.id,
            "username": user.username,
            "role": user.role,
            "exp": datetime.utcnow() + timedelta(
                minutes=self.config.access_token_expire_minutes
            ),
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        # Refresh token payload
        refresh_payload = {
            "sub": user.id,
            "exp": datetime.utcnow() + timedelta(
                days=self.config.refresh_token_expire_days
            ),
            "iat": datetime.utcnow(),
            "type": "refresh"
        }
        
        # Generate tokens
        access_token = jwt.encode(
            access_payload,
            self.config.secret_key,
            algorithm=self.config.algorithm
        )
        
        refresh_token = jwt.encode(
            refresh_payload,
            self.config.secret_key,
            algorithm=self.config.algorithm
        )
        
        # Store refresh token in database
        import uuid
        session_id = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(
            days=self.config.refresh_token_expire_days
        )
        
        self.db.execute(
            """INSERT INTO user_sessions 
            (id, user_id, refresh_token, expires_at) 
            VALUES (?, ?, ?, ?)""",
            (session_id, user.id, refresh_token, expires_at)
        )
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": self.config.access_token_expire_minutes * 60
        }
    
    async def verify_token(self, token: str) -> User:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm]
            )
            
            # Verify token type
            if payload.get("type") != "access":
                raise ValueError("Invalid token type")
            
            # Get user from database
            user_data = self.db.query(
                """SELECT id, username, email, role, is_active, 
                   created_at, last_login 
                   FROM users WHERE id = ?""",
                (payload["sub"],),
                one=True
            )
            
            if not user_data:
                raise ValueError("User not found")
            
            if not user_data[4]:  # is_active
                raise ValueError("User account is inactive")
            
            return User(
                id=user_data[0],
                username=user_data[1],
                email=user_data[2],
                role=user_data[3],
                is_active=user_data[4],
                created_at=user_data[5],
                last_login=user_data[6]
            )
            
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise ValueError(f"Invalid token: {str(e)}")
    
    async def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token using refresh token"""
        try:
            payload = jwt.decode(
                refresh_token,
                self.config.secret_key,
                algorithms=[self.config.algorithm]
            )
            
            # Verify token type
            if payload.get("type") != "refresh":
                raise ValueError("Invalid token type")
            
            # Check if refresh token exists in database
            session = self.db.query(
                """SELECT user_id FROM user_sessions 
                   WHERE refresh_token = ? AND expires_at > ?""",
                (refresh_token, datetime.utcnow()),
                one=True
            )
            
            if not session:
                raise ValueError("Invalid or expired refresh token")
            
            # Get user
            user_data = self.db.query(
                """SELECT id, username, email, role, is_active, 
                   created_at, last_login 
                   FROM users WHERE id = ?""",
                (session[0],),
                one=True
            )
            
            if not user_data or not user_data[4]:
                raise ValueError("User not found or inactive")
            
            user = User(
                id=user_data[0],
                username=user_data[1],
                email=user_data[2],
                role=user_data[3],
                is_active=user_data[4],
                created_at=user_data[5],
                last_login=user_data[6]
            )
            
            # Generate new tokens
            return self.generate_tokens(user)
            
        except jwt.ExpiredSignatureError:
            raise ValueError("Refresh token has expired")
        except jwt.InvalidTokenError as e:
            raise ValueError(f"Invalid refresh token: {str(e)}")
    
    async def revoke_token(self, user_id: str, token: str = None):
        """Revoke user tokens"""
        if token:
            # Revoke specific token
            self.db.execute(
                "DELETE FROM user_sessions WHERE refresh_token = ?",
                (token,)
            )
        else:
            # Revoke all user tokens
            self.db.execute(
                "DELETE FROM user_sessions WHERE user_id = ?",
                (user_id,)
            )
        
        logger.info("Tokens revoked", user_id=user_id)
    
    async def get_user_permissions(self, user: User) -> List[str]:
        """Get user permissions based on role"""
        role_permissions = {
            "admin": [
                "system:manage",
                "users:read",
                "users:write",
                "users:delete",
                "books:read",
                "books:write",
                "books:delete",
                "search:read",
                "search:write",
                "analytics:read",
                "analytics:write",
                "config:read",
                "config:write"
            ],
            "premium": [
                "books:read",
                "books:write",
                "search:read",
                "search:write",
                "analytics:read"
            ],
            "user": [
                "books:read",
                "search:read"
            ]
        }
        
        return role_permissions.get(user.role, [])
    
    def has_permission(self, user: User, permission: str) -> bool:
        """Check if user has specific permission"""
        permissions = asyncio.run(self.get_user_permissions(user))
        return permission in permissions
EOF
```

### Authentication Middleware

Create middleware to handle authentication across all API endpoints:

```python
# Create src/api/middleware/auth_middleware.py
cat > src/api/middleware/auth_middleware.py << 'EOF'
"""
Authentication middleware for FastAPI

Handles JWT validation, user context, and permission checking.
"""

from typing import Optional, Callable
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

from ..auth import AuthManager, User

logger = structlog.get_logger(__name__)

class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Global authentication middleware
    
    Validates JWT tokens and adds user context to requests
    """
    
    def __init__(self, app, auth_manager: AuthManager):
        super().__init__(app)
        self.auth_manager = auth_manager
        self.bearer = HTTPBearer(auto_error=False)
        
        # Paths that don't require authentication
        self.public_paths = [
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/auth/login",
            "/api/auth/register",
            "/api/auth/refresh"
        ]
    
    async def dispatch(self, request: Request, call_next):
        """Process request with authentication"""
        # Skip auth for public paths
        if request.url.path in self.public_paths:
            return await call_next(request)
        
        # Skip auth for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)
        
        try:
            # Extract token from Authorization header
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                logger.warning("Missing or invalid authorization header",
                             path=request.url.path)
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Missing authentication"}
                )
            
            token = auth_header.split(" ")[1]
            
            # Verify token and get user
            user = await self.auth_manager.verify_token(token)
            
            # Add user to request state
            request.state.user = user
            
            # Log authenticated request
            logger.info("Authenticated request",
                       user_id=user.id,
                       username=user.username,
                       path=request.url.path,
                       method=request.method)
            
            # Process request
            response = await call_next(request)
            return response
            
        except Exception as e:
            logger.error("Authentication error",
                        error=str(e),
                        path=request.url.path)
            return JSONResponse(
                status_code=401,
                content={"detail": str(e)}
            )


def require_auth(permissions: Optional[List[str]] = None):
    """
    Dependency to require authentication and optionally check permissions
    
    Args:
        permissions: List of required permissions
    
    Returns:
        Dependency function that validates user permissions
    """
    async def dependency(request: Request) -> User:
        # Get user from request state (set by middleware)
        user = getattr(request.state, "user", None)
        
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Authentication required"
            )
        
        # Check permissions if specified
        if permissions:
            auth_manager = request.app.state.auth_manager
            user_permissions = await auth_manager.get_user_permissions(user)
            
            # Check if user has any of the required permissions
            has_permission = any(
                perm in user_permissions for perm in permissions
            )
            
            if not has_permission:
                logger.warning("Permission denied",
                             user_id=user.id,
                             required=permissions,
                             user_permissions=user_permissions)
                raise HTTPException(
                    status_code=403,
                    detail=f"Insufficient permissions. Required: {permissions}"
                )
        
        return user
    
    return dependency


def require_role(roles: List[str]):
    """
    Dependency to require specific user roles
    
    Args:
        roles: List of allowed roles
    
    Returns:
        Dependency function that validates user role
    """
    async def dependency(request: Request) -> User:
        user = await require_auth()(request)
        
        if user.role not in roles:
            logger.warning("Role requirement not met",
                         user_id=user.id,
                         user_role=user.role,
                         required_roles=roles)
            raise HTTPException(
                status_code=403,
                detail=f"Role required: {roles}"
            )
        
        return user
    
    return dependency


class APIKeyAuth:
    """
    API Key authentication for programmatic access
    
    Supports API keys in header or query parameter
    """
    
    def __init__(self, auth_manager: AuthManager):
        self.auth_manager = auth_manager
    
    async def __call__(
        self, 
        request: Request,
        api_key: Optional[str] = None
    ) -> User:
        """Validate API key"""
        # Check header first
        key = request.headers.get("X-API-Key")
        
        # Fall back to query parameter
        if not key:
            key = api_key
        
        if not key:
            raise HTTPException(
                status_code=401,
                detail="API key required"
            )
        
        # Validate API key (implement API key validation)
        user = await self.auth_manager.validate_api_key(key)
        
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid API key"
            )
        
        return user
EOF
```

### Authentication API Endpoints

Create the authentication router with comprehensive user management:

```python
# Create src/api/routers/auth.py
cat > src/api/routers/auth.py << 'EOF'
"""
Authentication API endpoints

Handles user registration, login, token management, and user administration.
"""

from typing import List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Body, Query
from fastapi.security import OAuth2PasswordRequestForm
import structlog

from ..models import *
from ..auth import AuthManager, User
from ..middleware.auth_middleware import require_auth, require_role

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

@router.post("/register", response_model=UserInfo)
async def register(
    request: UserRegistration,
    auth_manager: AuthManager = Depends(lambda: app_state["auth_manager"])
):
    """
    Register a new user
    
    Creates a new user account with the specified credentials.
    Default role is 'user' unless specified otherwise.
    """
    try:
        user = await auth_manager.register_user(
            username=request.username,
            email=request.email,
            password=request.password,
            role=request.role.value
        )
        
        logger.info("User registered via API",
                   user_id=user.id,
                   username=user.username)
        
        return UserInfo(
            id=user.id,
            username=user.username,
            email=user.email,
            role=UserRole(user.role),
            is_active=user.is_active,
            created_at=user.created_at,
            last_login=user.last_login
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Registration failed", error=str(e))
        raise HTTPException(status_code=500, detail="Registration failed")


@router.post("/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    auth_manager: AuthManager = Depends(lambda: app_state["auth_manager"])
):
    """
    User login
    
    Authenticates user and returns access/refresh tokens.
    """
    try:
        # Authenticate user
        user = await auth_manager.authenticate_user(
            username=request.username,
            password=request.password
        )
        
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid username or password"
            )
        
        # Generate tokens
        tokens = auth_manager.generate_tokens(user)
        
        logger.info("User logged in",
                   user_id=user.id,
                   username=user.username)
        
        return LoginResponse(
            success=True,
            message="Login successful",
            access_token=tokens["access_token"],
            expires_in=tokens["expires_in"],
            user=UserInfo(
                id=user.id,
                username=user.username,
                email=user.email,
                role=UserRole(user.role),
                is_active=user.is_active,
                created_at=user.created_at,
                last_login=user.last_login
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Login failed", error=str(e))
        raise HTTPException(status_code=500, detail="Login failed")


@router.post("/refresh", response_model=LoginResponse)
async def refresh_token(
    refresh_token: str = Body(..., embed=True),
    auth_manager: AuthManager = Depends(lambda: app_state["auth_manager"])
):
    """
    Refresh access token
    
    Uses refresh token to generate new access token.
    """
    try:
        tokens = await auth_manager.refresh_access_token(refresh_token)
        
        # Get user info from new token
        user = await auth_manager.verify_token(tokens["access_token"])
        
        return LoginResponse(
            success=True,
            message="Token refreshed successfully",
            access_token=tokens["access_token"],
            expires_in=tokens["expires_in"],
            user=UserInfo(
                id=user.id,
                username=user.username,
                email=user.email,
                role=UserRole(user.role),
                is_active=user.is_active,
                created_at=user.created_at,
                last_login=user.last_login
            )
        )
        
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.error("Token refresh failed", error=str(e))
        raise HTTPException(status_code=500, detail="Token refresh failed")


@router.post("/logout")
async def logout(
    current_user: User = Depends(require_auth()),
    auth_manager: AuthManager = Depends(lambda: app_state["auth_manager"])
):
    """
    User logout
    
    Revokes all user tokens.
    """
    try:
        await auth_manager.revoke_token(current_user.id)
        
        logger.info("User logged out",
                   user_id=current_user.id,
                   username=current_user.username)
        
        return BaseResponse(
            success=True,
            message="Logged out successfully"
        )
        
    except Exception as e:
        logger.error("Logout failed", error=str(e))
        raise HTTPException(status_code=500, detail="Logout failed")


@router.get("/me", response_model=UserInfo)
async def get_current_user(
    current_user: User = Depends(require_auth())
):
    """
    Get current user info
    
    Returns information about the authenticated user.
    """
    return UserInfo(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        role=UserRole(current_user.role),
        is_active=current_user.is_active,
        created_at=current_user.created_at,
        last_login=current_user.last_login
    )


@router.put("/me/password")
async def change_password(
    old_password: str = Body(...),
    new_password: str = Body(...),
    current_user: User = Depends(require_auth()),
    auth_manager: AuthManager = Depends(lambda: app_state["auth_manager"])
):
    """
    Change user password
    
    Allows users to change their own password.
    """
    try:
        # Verify old password
        authenticated = await auth_manager.authenticate_user(
            username=current_user.username,
            password=old_password
        )
        
        if not authenticated:
            raise HTTPException(
                status_code=401,
                detail="Current password is incorrect"
            )
        
        # Update password
        await auth_manager.update_password(
            user_id=current_user.id,
            new_password=new_password
        )
        
        # Revoke all tokens to force re-login
        await auth_manager.revoke_token(current_user.id)
        
        logger.info("Password changed",
                   user_id=current_user.id,
                   username=current_user.username)
        
        return BaseResponse(
            success=True,
            message="Password changed successfully. Please login again."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Password change failed", error=str(e))
        raise HTTPException(status_code=500, detail="Password change failed")


# Admin endpoints for user management

@router.get("/users", response_model=List[UserInfo])
async def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    role: Optional[UserRole] = None,
    is_active: Optional[bool] = None,
    current_user: User = Depends(require_role(["admin"])),
    auth_manager: AuthManager = Depends(lambda: app_state["auth_manager"])
):
    """
    List all users (Admin only)
    
    Returns paginated list of users with optional filtering.
    """
    try:
        users = await auth_manager.list_users(
            skip=skip,
            limit=limit,
            role=role.value if role else None,
            is_active=is_active
        )
        
        return [
            UserInfo(
                id=user.id,
                username=user.username,
                email=user.email,
                role=UserRole(user.role),
                is_active=user.is_active,
                created_at=user.created_at,
                last_login=user.last_login
            )
            for user in users
        ]
        
    except Exception as e:
        logger.error("Failed to list users", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list users")


@router.get("/users/{user_id}", response_model=UserInfo)
async def get_user(
    user_id: str,
    current_user: User = Depends(require_role(["admin"])),
    auth_manager: AuthManager = Depends(lambda: app_state["auth_manager"])
):
    """
    Get specific user details (Admin only)
    """
    try:
        user = await auth_manager.get_user(user_id)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return UserInfo(
            id=user.id,
            username=user.username,
            email=user.email,
            role=UserRole(user.role),
            is_active=user.is_active,
            created_at=user.created_at,
            last_login=user.last_login
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get user", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get user")


@router.put("/users/{user_id}")
async def update_user(
    user_id: str,
    role: Optional[UserRole] = Body(None),
    is_active: Optional[bool] = Body(None),
    current_user: User = Depends(require_role(["admin"])),
    auth_manager: AuthManager = Depends(lambda: app_state["auth_manager"])
):
    """
    Update user details (Admin only)
    
    Allows updating user role and active status.
    """
    try:
        updates = {}
        if role is not None:
            updates["role"] = role.value
        if is_active is not None:
            updates["is_active"] = is_active
        
        if not updates:
            raise HTTPException(status_code=400, detail="No updates provided")
        
        await auth_manager.update_user(user_id, **updates)
        
        # If deactivating user, revoke their tokens
        if is_active is False:
            await auth_manager.revoke_token(user_id)
        
        logger.info("User updated",
                   admin_id=current_user.id,
                   user_id=user_id,
                   updates=updates)
        
        return BaseResponse(
            success=True,
            message="User updated successfully"
        )
        
    except Exception as e:
        logger.error("Failed to update user", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update user")


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    current_user: User = Depends(require_role(["admin"])),
    auth_manager: AuthManager = Depends(lambda: app_state["auth_manager"])
):
    """
    Delete user (Admin only)
    
    Permanently deletes a user account.
    """
    try:
        # Prevent self-deletion
        if user_id == current_user.id:
            raise HTTPException(
                status_code=400,
                detail="Cannot delete your own account"
            )
        
        # Revoke tokens first
        await auth_manager.revoke_token(user_id)
        
        # Delete user
        await auth_manager.delete_user(user_id)
        
        logger.info("User deleted",
                   admin_id=current_user.id,
                   deleted_user_id=user_id)
        
        return BaseResponse(
            success=True,
            message="User deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete user", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to delete user")


@router.get("/permissions", response_model=List[str])
async def get_my_permissions(
    current_user: User = Depends(require_auth()),
    auth_manager: AuthManager = Depends(lambda: app_state["auth_manager"])
):
    """
    Get current user's permissions
    
    Returns list of permissions based on user role.
    """
    permissions = await auth_manager.get_user_permissions(current_user)
    return permissions


@router.post("/api-keys")
async def create_api_key(
    name: str = Body(...),
    expires_in_days: Optional[int] = Body(365),
    current_user: User = Depends(require_auth()),
    auth_manager: AuthManager = Depends(lambda: app_state["auth_manager"])
):
    """
    Create API key for programmatic access
    
    Generates a new API key for the current user.
    """
    try:
        api_key = await auth_manager.create_api_key(
            user_id=current_user.id,
            name=name,
            expires_in_days=expires_in_days
        )
        
        logger.info("API key created",
                   user_id=current_user.id,
                   key_name=name)
        
        return {
            "api_key": api_key["key"],
            "key_id": api_key["id"],
            "expires_at": api_key["expires_at"]
        }
        
    except Exception as e:
        logger.error("Failed to create API key", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create API key")


@router.get("/api-keys")
async def list_api_keys(
    current_user: User = Depends(require_auth()),
    auth_manager: AuthManager = Depends(lambda: app_state["auth_manager"])
):
    """
    List user's API keys
    
    Returns all API keys for the current user.
    """
    try:
        keys = await auth_manager.list_api_keys(current_user.id)
        
        return [
            {
                "id": key["id"],
                "name": key["name"],
                "created_at": key["created_at"],
                "expires_at": key["expires_at"],
                "last_used": key["last_used"]
            }
            for key in keys
        ]
        
    except Exception as e:
        logger.error("Failed to list API keys", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list API keys")


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    current_user: User = Depends(require_auth()),
    auth_manager: AuthManager = Depends(lambda: app_state["auth_manager"])
):
    """
    Revoke API key
    
    Revokes the specified API key.
    """
    try:
        await auth_manager.revoke_api_key(key_id, current_user.id)
        
        logger.info("API key revoked",
                   user_id=current_user.id,
                   key_id=key_id)
        
        return BaseResponse(
            success=True,
            message="API key revoked successfully"
        )
        
    except Exception as e:
        logger.error("Failed to revoke API key", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to revoke API key")
EOF
```

### Security Best Practices Implementation

Create security utilities and configurations:

```python
# Create src/api/security.py
cat > src/api/security.py << 'EOF'
"""
Security utilities and best practices

Implements OWASP security recommendations for production APIs.
"""

import secrets
import hashlib
import hmac
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import re
from functools import wraps
import structlog

from ..core.config import SecurityConfig

logger = structlog.get_logger(__name__)

class SecurityManager:
    """
    Comprehensive security management
    
    Implements:
    - Password policies
    - Rate limiting
    - CSRF protection
    - Session security
    - Input validation
    """
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.password_regex = re.compile(
            r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'
        )
    
    def validate_password(self, password: str) -> tuple[bool, str]:
        """
        Validate password against security policy
        
        Requirements:
        - Minimum 8 characters
        - At least one uppercase letter
        - At least one lowercase letter
        - At least one number
        - At least one special character
        """
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"
        
        if not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"
        
        if not re.search(r'\d', password):
            return False, "Password must contain at least one number"
        
        if not re.search(r'[@$!%*?&]', password):
            return False, "Password must contain at least one special character"
        
        # Check against common passwords
        if self._is_common_password(password):
            return False, "Password is too common. Please choose a stronger password"
        
        return True, "Password is valid"
    
    def _is_common_password(self, password: str) -> bool:
        """Check if password is in common passwords list"""
        common_passwords = [
            "password", "123456", "password123", "admin123",
            "qwerty", "letmein", "welcome", "monkey",
            "dragon", "baseball", "iloveyou", "trustno1"
        ]
        return password.lower() in common_passwords
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure random token"""
        return secrets.token_urlsafe(length)
    
    def generate_csrf_token(self, session_id: str) -> str:
        """Generate CSRF token tied to session"""
        secret = self.config.csrf_secret.encode()
        message = f"{session_id}:{datetime.utcnow().isoformat()}".encode()
        
        signature = hmac.new(secret, message, hashlib.sha256).hexdigest()
        return f"{session_id}:{signature}"
    
    def validate_csrf_token(
        self, 
        token: str, 
        session_id: str,
        max_age_minutes: int = 60
    ) -> bool:
        """Validate CSRF token"""
        try:
            token_session, signature = token.split(":")
            
            if token_session != session_id:
                return False
            
            # Regenerate signature
            secret = self.config.csrf_secret.encode()
            message = f"{session_id}:{datetime.utcnow().isoformat()}".encode()
            
            expected_signature = hmac.new(
                secret, message, hashlib.sha256
            ).hexdigest()
            
            # Constant-time comparison
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception:
            return False
    
    def sanitize_input(self, input_str: str) -> str:
        """
        Sanitize user input to prevent XSS and injection attacks
        """
        # Remove null bytes
        input_str = input_str.replace('\x00', '')
        
        # HTML entity encoding for special characters
        html_escape_table = {
            "&": "&amp;",
            '"': "&quot;",
            "'": "&#x27;",
            ">": "&gt;",
            "<": "&lt;",
            "/": "&#x2F;",
        }
        
        return "".join(
            html_escape_table.get(c, c) for c in input_str
        )
    
    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        email_regex = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        )
        return bool(email_regex.match(email))
    
    def hash_sensitive_data(self, data: str) -> str:
        """
        Hash sensitive data for storage
        
        Uses SHA-256 with salt for one-way hashing
        """
        salt = self.config.hash_salt.encode()
        return hashlib.pbkdf2_hmac(
            'sha256',
            data.encode(),
            salt,
            100000  # iterations
        ).hex()
    
    def create_session_fingerprint(
        self, 
        request_headers: Dict[str, str]
    ) -> str:
        """
        Create session fingerprint from request headers
        
        Helps detect session hijacking
        """
        components = [
            request_headers.get("User-Agent", ""),
            request_headers.get("Accept-Language", ""),
            request_headers.get("Accept-Encoding", ""),
            # Don't use IP address to avoid issues with mobile users
        ]
        
        fingerprint_str = "|".join(components)
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()


class RateLimiter:
    """
    Token bucket rate limiter implementation
    """
    
    def __init__(
        self,
        rate: int = 60,  # requests per minute
        burst: int = 10   # burst capacity
    ):
        self.rate = rate
        self.burst = burst
        self.buckets = {}
        self.window = 60  # seconds
    
    async def check_rate_limit(
        self, 
        identifier: str
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Check if request is within rate limit
        
        Returns:
            allowed: Whether request is allowed
            info: Rate limit information
        """
        now = datetime.utcnow()
        
        if identifier not in self.buckets:
            self.buckets[identifier] = {
                "tokens": self.burst,
                "last_update": now,
                "requests": []
            }
        
        bucket = self.buckets[identifier]
        
        # Remove old requests outside window
        cutoff = now - timedelta(seconds=self.window)
        bucket["requests"] = [
            req for req in bucket["requests"] if req > cutoff
        ]
        
        # Check current rate
        current_rate = len(bucket["requests"])
        
        if current_rate >= self.rate:
            # Calculate retry after
            oldest_request = min(bucket["requests"])
            retry_after = (
                oldest_request + timedelta(seconds=self.window) - now
            ).total_seconds()
            
            return False, {
                "limit": self.rate,
                "remaining": 0,
                "reset": int((now + timedelta(seconds=retry_after)).timestamp()),
                "retry_after": int(retry_after)
            }
        
        # Add current request
        bucket["requests"].append(now)
        
        return True, {
            "limit": self.rate,
            "remaining": self.rate - current_rate - 1,
            "reset": int((now + timedelta(seconds=self.window)).timestamp())
        }


def security_headers(func):
    """
    Decorator to add security headers to responses
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        response = await func(*args, **kwargs)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=()"
        )
        
        return response
    
    return wrapper


class AuditLogger:
    """
    Security audit logging
    
    Logs security-relevant events for compliance and monitoring
    """
    
    def __init__(self):
        self.logger = structlog.get_logger("security_audit")
    
    def log_authentication(
        self,
        event_type: str,
        user_id: Optional[str],
        username: Optional[str],
        ip_address: str,
        user_agent: str,
        success: bool,
        reason: Optional[str] = None
    ):
        """Log authentication events"""
        self.logger.info(
            "authentication_event",
            event_type=event_type,
            user_id=user_id,
            username=username,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            reason=reason,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def log_authorization(
        self,
        user_id: str,
        resource: str,
        action: str,
        allowed: bool,
        reason: Optional[str] = None
    ):
        """Log authorization decisions"""
        self.logger.info(
            "authorization_event",
            user_id=user_id,
            resource=resource,
            action=action,
            allowed=allowed,
            reason=reason,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def log_data_access(
        self,
        user_id: str,
        data_type: str,
        operation: str,
        record_count: int,
        filters: Optional[Dict[str, Any]] = None
    ):
        """Log data access for compliance"""
        self.logger.info(
            "data_access_event",
            user_id=user_id,
            data_type=data_type,
            operation=operation,
            record_count=record_count,
            filters=filters,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def log_security_incident(
        self,
        incident_type: str,
        severity: str,
        description: str,
        source_ip: Optional[str] = None,
        user_id: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ):
        """Log security incidents"""
        self.logger.warning(
            "security_incident",
            incident_type=incident_type,
            severity=severity,
            description=description,
            source_ip=source_ip,
            user_id=user_id,
            additional_data=additional_data,
            timestamp=datetime.utcnow().isoformat()
        )
EOF
```

### Session Management

Implement secure session handling:

```python
# Create src/api/session.py
cat > src/api/session.py << 'EOF'
"""
Session management for TradeKnowledge API

Implements secure session handling with Redis backend.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import redis.asyncio as redis
import structlog

from ..core.config import get_config

logger = structlog.get_logger(__name__)

class SessionManager:
    """
    Redis-backed session management
    
    Features:
    - Secure session tokens
    - Session expiration
    - Concurrent session limits
    - Session invalidation
    """
    
    def __init__(self):
        self.config = get_config()
        self.redis_client = None
        self.session_prefix = "session:"
        self.user_sessions_prefix = "user_sessions:"
        self.default_ttl = 3600 * 24  # 24 hours
        self.max_concurrent_sessions = 5
    
    async def initialize(self):
        """Initialize Redis connection"""
        self.redis_client = await redis.create_redis_pool(
            f"redis://{self.config.redis.host}:{self.config.redis.port}",
            password=self.config.redis.password,
            encoding="utf-8",
            db=self.config.redis.session_db
        )
        logger.info("Session manager initialized")
    
    async def create_session(
        self,
        user_id: str,
        user_data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> str:
        """
        Create new session
        
        Args:
            user_id: User identifier
            user_data: Data to store in session
            ttl: Time to live in seconds
        
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        session_key = f"{self.session_prefix}{session_id}"
        user_sessions_key = f"{self.user_sessions_prefix}{user_id}"
        
        session_data = {
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "last_accessed": datetime.utcnow().isoformat(),
            **user_data
        }
        
        # Store session
        await self.redis_client.setex(
            session_key,
            ttl or self.default_ttl,
            json.dumps(session_data)
        )
        
        # Track user sessions
        await self.redis_client.sadd(user_sessions_key, session_id)
        
        # Enforce concurrent session limit
        await self._enforce_session_limit(user_id)
        
        logger.info("Session created",
                   session_id=session_id,
                   user_id=user_id)
        
        return session_id
    
    async def get_session(
        self, 
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get session data
        
        Updates last accessed time.
        """
        session_key = f"{self.session_prefix}{session_id}"
        
        # Get session data
        session_data = await self.redis_client.get(session_key)
        
        if not session_data:
            return None
        
        data = json.loads(session_data)
        
        # Update last accessed
        data["last_accessed"] = datetime.utcnow().isoformat()
        
        # Refresh TTL
        await self.redis_client.setex(
            session_key,
            self.default_ttl,
            json.dumps(data)
        )
        
        return data
    
    async def update_session(
        self,
        session_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update session data"""
        session_data = await self.get_session(session_id)
        
        if not session_data:
            return False
        
        session_data.update(updates)
        
        session_key = f"{self.session_prefix}{session_id}"
        await self.redis_client.setex(
            session_key,
            self.default_ttl,
            json.dumps(session_data)
        )
        
        return True
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session"""
        session_key = f"{self.session_prefix}{session_id}"
        
        # Get session to find user
        session_data = await self.get_session(session_id)
        
        if not session_data:
            return False
        
        # Remove from user sessions
        user_id = session_data.get("user_id")
        if user_id:
            user_sessions_key = f"{self.user_sessions_prefix}{user_id}"
            await self.redis_client.srem(user_sessions_key, session_id)
        
        # Delete session
        await self.redis_client.delete(session_key)
        
        logger.info("Session deleted",
                   session_id=session_id,
                   user_id=user_id)
        
        return True
    
    async def delete_user_sessions(self, user_id: str) -> int:
        """Delete all sessions for a user"""
        user_sessions_key = f"{self.user_sessions_prefix}{user_id}"
        
        # Get all user sessions
        session_ids = await self.redis_client.smembers(user_sessions_key)
        
        if not session_ids:
            return 0
        
        # Delete each session
        for session_id in session_ids:
            session_key = f"{self.session_prefix}{session_id}"
            await self.redis_client.delete(session_key)
        
        # Clear user sessions set
        await self.redis_client.delete(user_sessions_key)
        
        logger.info("User sessions deleted",
                   user_id=user_id,
                   count=len(session_ids))
        
        return len(session_ids)
    
    async def _enforce_session_limit(self, user_id: str):
        """Enforce maximum concurrent sessions per user"""
        user_sessions_key = f"{self.user_sessions_prefix}{user_id}"
        
        # Get all user sessions
        session_ids = await self.redis_client.smembers(user_sessions_key)
        
        if len(session_ids) <= self.max_concurrent_sessions:
            return
        
        # Get session details to find oldest
        sessions = []
        for session_id in session_ids:
            session_key = f"{self.session_prefix}{session_id}"
            session_data = await self.redis_client.get(session_key)
            
            if session_data:
                data = json.loads(session_data)
                sessions.append({
                    "id": session_id,
                    "created_at": data.get("created_at")
                })
        
        # Sort by creation time
        sessions.sort(key=lambda x: x["created_at"])
        
        # Remove oldest sessions
        to_remove = len(sessions) - self.max_concurrent_sessions
        
        for i in range(to_remove):
            await self.delete_session(sessions[i]["id"])
        
        logger.warning("Session limit enforced",
                      user_id=user_id,
                      removed=to_remove)
    
    async def get_active_sessions_count(self) -> int:
        """Get count of active sessions"""
        pattern = f"{self.session_prefix}*"
        cursor = 0
        count = 0
        
        while True:
            cursor, keys = await self.redis_client.scan(
                cursor, 
                match=pattern,
                count=1000
            )
            count += len(keys)
            
            if cursor == 0:
                break
        
        return count
    
    async def cleanup_expired_sessions(self):
        """
        Clean up expired sessions
        
        Redis handles expiration automatically, but this ensures
        user session sets are cleaned up.
        """
        pattern = f"{self.user_sessions_prefix}*"
        cursor = 0
        cleaned = 0
        
        while True:
            cursor, keys = await self.redis_client.scan(
                cursor,
                match=pattern,
                count=100
            )
            
            for user_sessions_key in keys:
                session_ids = await self.redis_client.smembers(
                    user_sessions_key
                )
                
                for session_id in session_ids:
                    session_key = f"{self.session_prefix}{session_id}"
                    exists = await self.redis_client.exists(session_key)
                    
                    if not exists:
                        await self.redis_client.srem(
                            user_sessions_key, 
                            session_id
                        )
                        cleaned += 1
            
            if cursor == 0:
                break
        
        if cleaned > 0:
            logger.info("Cleaned expired sessions", count=cleaned)
        
        return cleaned
EOF
```

This comprehensive authentication and authorization system provides:

1. **JWT-based Authentication**: Secure token generation and validation with access/refresh token pattern
2. **Role-Based Access Control**: Flexible permission system based on user roles
3. **User Management**: Complete CRUD operations for user administration
4. **Session Management**: Redis-backed sessions with concurrent session limits
5. **Security Best Practices**: 
   - Password policies with complexity requirements
   - Rate limiting to prevent abuse
   - CSRF protection
   - Input sanitization
   - Security headers
   - Audit logging
6. **API Key Support**: For programmatic access
7. **Middleware Integration**: Seamless authentication across all endpoints

The system follows OWASP security guidelines and provides enterprise-grade authentication suitable for production deployment. It integrates smoothly with the FastAPI framework and provides comprehensive logging for security monitoring and compliance.