#!/usr/bin/env python3
"""
LLMLingua Service Runner - Standalone service for Claude Code
Provides HTTP API for prompt compression across all Claude Code instances
"""

import asyncio
import argparse
import logging
import signal
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.core.llmlingua_service import LLMLinguaService, CompressionConfig, CompressionResult
except ImportError as e:
    print(f"❌ Failed to import LLMLingua service: {e}")
    print("Make sure you're running from the TradeKnowledge root directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global service instance
llmlingua_service = LLMLinguaService()

# FastAPI app
app = FastAPI(
    title="LLMLingua Compression Service",
    description="Prompt compression service for Claude Code instances",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class CompressionRequest(BaseModel):
    prompt: str
    compression_ratio: Optional[float] = 0.5
    target_token: Optional[int] = None
    agent_role: Optional[str] = None  # RESEARCHER, MASTERMIND, EXECUTOR
    preserve_instructions: bool = True
    preserve_question: bool = True

class CompressionResponse(BaseModel):
    original_prompt: str
    compressed_prompt: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    processing_time_ms: float
    model_used: str
    token_savings: int
    cost_savings_estimate: float
    quality_score: Optional[float] = None
    error: Optional[str] = None
    fallback_used: bool = False
    cache_hit: bool = False

class BatchCompressionRequest(BaseModel):
    prompts: list[str]
    compression_ratio: Optional[float] = 0.5
    target_token: Optional[int] = None

class HealthResponse(BaseModel):
    status: str
    services: Dict[str, str]
    version: str = "1.0.0"

class StatsResponse(BaseModel):
    total_compressions: int
    total_tokens_saved: int
    total_cost_saved: float
    average_compression_ratio: float
    service_uptime_seconds: float

# Service state
service_start_time = None

@app.on_event("startup")
async def startup_event():
    """Initialize LLMLingua service on startup"""
    global service_start_time
    import time
    service_start_time = time.time()
    
    logger.info("🚀 Starting LLMLingua Compression Service...")
    try:
        await llmlingua_service.initialize()
        logger.info("✅ LLMLingua service initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize LLMLingua service: {e}")
        logger.warning("Service will run in fallback mode")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down LLMLingua service...")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health"""
    try:
        health_status = await llmlingua_service.health_check()
        
        overall_status = "healthy"
        for service, status in health_status.items():
            if "unhealthy" in status.lower():
                overall_status = "degraded"
                break
        
        return HealthResponse(
            status=overall_status,
            services=health_status
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# Compression endpoint
@app.post("/compress", response_model=CompressionResponse)
async def compress_prompt(request: CompressionRequest):
    """Compress a single prompt"""
    try:
        # Create compression config
        config = CompressionConfig(
            compression_ratio=request.compression_ratio,
            target_token=request.target_token,
            preserve_instructions=request.preserve_instructions,
            preserve_question=request.preserve_question
        )
        
        # Use agent-specific compression if agent_role is specified
        if request.agent_role:
            result = await llmlingua_service.compress_agent_prompt(
                agent_role=request.agent_role.upper(),
                prompt=request.prompt,
                context={"api_request": True}
            )
        else:
            result = await llmlingua_service.compress_prompt(
                prompt=request.prompt,
                config=config,
                context={"api_request": True}
            )
        
        return CompressionResponse(
            original_prompt=result.original_prompt,
            compressed_prompt=result.compressed_prompt,
            original_tokens=result.original_tokens,
            compressed_tokens=result.compressed_tokens,
            compression_ratio=result.compression_ratio,
            processing_time_ms=result.processing_time_ms,
            model_used=result.model_used,
            token_savings=result.token_savings,
            cost_savings_estimate=result.cost_savings_estimate,
            quality_score=result.quality_score,
            error=result.error,
            fallback_used=result.fallback_used,
            cache_hit=result.cache_hit
        )
        
    except Exception as e:
        logger.error(f"Compression failed: {e}")
        raise HTTPException(status_code=500, detail=f"Compression failed: {str(e)}")

# Batch compression endpoint
@app.post("/compress/batch")
async def compress_batch(request: BatchCompressionRequest):
    """Compress multiple prompts in batch"""
    try:
        config = CompressionConfig(
            compression_ratio=request.compression_ratio,
            target_token=request.target_token
        )
        
        results = await llmlingua_service.batch_compress(
            prompts=request.prompts,
            config=config
        )
        
        return {
            "results": [
                CompressionResponse(
                    original_prompt=result.original_prompt,
                    compressed_prompt=result.compressed_prompt,
                    original_tokens=result.original_tokens,
                    compressed_tokens=result.compressed_tokens,
                    compression_ratio=result.compression_ratio,
                    processing_time_ms=result.processing_time_ms,
                    model_used=result.model_used,
                    token_savings=result.token_savings,
                    cost_savings_estimate=result.cost_savings_estimate,
                    quality_score=result.quality_score,
                    error=result.error,
                    fallback_used=result.fallback_used,
                    cache_hit=result.cache_hit
                ) for result in results
            ],
            "total_prompts": len(request.prompts),
            "total_original_tokens": sum(r.original_tokens for r in results),
            "total_compressed_tokens": sum(r.compressed_tokens for r in results),
            "total_savings": sum(r.token_savings for r in results),
            "total_cost_savings": sum(r.cost_savings_estimate for r in results)
        }
        
    except Exception as e:
        logger.error(f"Batch compression failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch compression failed: {str(e)}")

# Statistics endpoint
@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get compression statistics"""
    try:
        import time
        uptime = time.time() - service_start_time if service_start_time else 0
        
        return StatsResponse(
            total_compressions=llmlingua_service.compression_stats["total_compressions"],
            total_tokens_saved=llmlingua_service.compression_stats["total_tokens_saved"],
            total_cost_saved=llmlingua_service.compression_stats["total_cost_saved"],
            average_compression_ratio=llmlingua_service.compression_stats["average_compression_ratio"],
            service_uptime_seconds=uptime
        )
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

# Reset statistics endpoint (for development)
@app.post("/stats/reset")
async def reset_stats():
    """Reset compression statistics"""
    llmlingua_service.compression_stats = {
        "total_compressions": 0,
        "total_tokens_saved": 0,
        "total_cost_saved": 0.0,
        "average_compression_ratio": 0.0
    }
    return {"message": "Statistics reset successfully"}

# Simple test endpoint
@app.get("/test")
async def test_compression():
    """Test compression with a simple prompt"""
    test_prompt = "This is a test prompt for the LLMLingua compression service. It should be compressed to verify the service is working correctly."
    
    try:
        result = await llmlingua_service.compress_prompt(
            prompt=test_prompt,
            config=CompressionConfig(target_token=10, enable_caching=False)
        )
        
        return {
            "message": "Compression test successful",
            "original_length": len(test_prompt),
            "compressed_length": len(result.compressed_prompt),
            "original_tokens": result.original_tokens,
            "compressed_tokens": result.compressed_tokens,
            "compression_ratio": result.compression_ratio
        }
    except Exception as e:
        return {"message": f"Compression test failed: {str(e)}"}

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="LLMLingua Compression Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Configure logging
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    logger.info(f"🚀 Starting LLMLingua service on {args.host}:{args.port}")
    
    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        access_log=not args.daemon
    )

if __name__ == "__main__":
    main()