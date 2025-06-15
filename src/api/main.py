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