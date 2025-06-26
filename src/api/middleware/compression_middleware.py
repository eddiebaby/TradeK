"""
API Compression Middleware for TradeKnowledge
Automatically compresses prompts for all Claude API calls to reduce costs
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import json

from ..core.llmlingua_service import LLMLinguaService, CompressionConfig, CompressionResult
from ..core.agent_compression import AgentCompressionService, AgentRole, AgentPromptContext

logger = logging.getLogger(__name__)


class CompressionMiddleware(BaseHTTPMiddleware):
    """Middleware to automatically compress prompts in API requests"""
    
    def __init__(
        self,
        app: ASGIApp,
        llmlingua_service: LLMLinguaService,
        agent_compression_service: AgentCompressionService,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(app)
        self.llmlingua_service = llmlingua_service
        self.agent_compression_service = agent_compression_service
        self.config = config or {}
        
        # Configuration
        self.enabled = self.config.get("enabled", True)
        self.compress_threshold = self.config.get("compress_threshold", 100)  # Min tokens
        self.max_compression_time = self.config.get("max_compression_time", 5.0)  # Seconds
        self.fallback_on_timeout = self.config.get("fallback_on_timeout", True)
        
        # Endpoints to compress
        self.compress_endpoints = self.config.get("compress_endpoints", [
            "/api/prompts",
            "/api/analysis", 
            "/api/agents",
            "/api/sparc"
        ])
        
        # Fields that contain prompts to compress
        self.prompt_fields = self.config.get("prompt_fields", [
            "prompt", "content", "message", "query", "instruction", "text"
        ])
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "compressed_requests": 0,
            "compression_errors": 0,
            "total_tokens_saved": 0,
            "total_time_saved": 0.0,
            "total_cost_saved": 0.0
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with optional prompt compression"""
        start_time = time.time()
        
        self.stats["total_requests"] += 1
        
        # Check if compression should be applied
        if not self._should_compress_request(request):
            return await call_next(request)
        
        try:
            # Extract and compress prompts from request
            modified_request = await self._compress_request_prompts(request)
            
            # Process the request
            response = await call_next(modified_request)
            
            # Add compression metadata to response headers
            response.headers["X-Compression-Applied"] = "true"
            response.headers["X-Compression-Time"] = str(time.time() - start_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Compression middleware error: {e}")
            self.stats["compression_errors"] += 1
            
            # Fall back to original request
            response = await call_next(request)
            response.headers["X-Compression-Applied"] = "false"
            response.headers["X-Compression-Error"] = str(e)
            
            return response
    
    def _should_compress_request(self, request: Request) -> bool:
        """Determine if request should have prompts compressed"""
        if not self.enabled:
            return False
        
        # Check if endpoint matches compression patterns
        path = str(request.url.path)
        for endpoint in self.compress_endpoints:
            if path.startswith(endpoint):
                return True
        
        # Check for specific compression header
        if request.headers.get("X-Enable-Compression") == "true":
            return True
        
        # Skip if explicitly disabled
        if request.headers.get("X-Disable-Compression") == "true":
            return False
        
        return False
    
    async def _compress_request_prompts(self, request: Request) -> Request:
        """Extract and compress prompts from request body"""
        # Read request body
        body = await request.body()
        if not body:
            return request
        
        try:
            # Parse JSON body
            data = json.loads(body.decode())
            
            # Find and compress prompt fields
            compression_results = []
            modified_data = await self._compress_data_prompts(data, compression_results)
            
            # Update statistics
            await self._update_compression_stats(compression_results)
            
            # Create new request with compressed data
            new_body = json.dumps(modified_data).encode()
            
            # Create new request object with modified body
            # Note: This is a simplified approach - in production, you'd want to 
            # properly handle all request attributes
            modified_request = Request(
                scope={
                    **request.scope,
                    "body": new_body
                }
            )
            
            # Add compression metadata to request state
            modified_request.state.compression_results = compression_results
            
            return modified_request
            
        except json.JSONDecodeError:
            logger.warning("Request body is not valid JSON, skipping compression")
            return request
        except Exception as e:
            logger.error(f"Error processing request for compression: {e}")
            return request
    
    async def _compress_data_prompts(
        self, 
        data: Any, 
        compression_results: List[CompressionResult],
        path: str = ""
    ) -> Any:
        """Recursively find and compress prompt fields in data"""
        if isinstance(data, dict):
            modified_dict = {}
            
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                
                # Check if this field contains a prompt
                if key.lower() in self.prompt_fields and isinstance(value, str):
                    compressed_value, result = await self._compress_prompt_field(
                        value, key, current_path, data
                    )
                    modified_dict[key] = compressed_value
                    if result:
                        compression_results.append(result)
                else:
                    # Recursively process nested structures
                    modified_dict[key] = await self._compress_data_prompts(
                        value, compression_results, current_path
                    )
            
            return modified_dict
            
        elif isinstance(data, list):
            modified_list = []
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]"
                modified_item = await self._compress_data_prompts(
                    item, compression_results, current_path
                )
                modified_list.append(modified_item)
            return modified_list
        
        else:
            return data
    
    async def _compress_prompt_field(
        self, 
        prompt: str, 
        field_name: str, 
        field_path: str,
        context_data: Dict[str, Any]
    ) -> tuple[str, Optional[CompressionResult]]:
        """Compress a specific prompt field"""
        
        # Check if prompt meets compression threshold
        estimated_tokens = len(prompt.split()) * 1.3  # Rough estimation
        if estimated_tokens < self.compress_threshold:
            logger.debug(f"Skipping compression for short prompt ({estimated_tokens} tokens)")
            return prompt, None
        
        try:
            # Determine compression strategy based on context
            compression_config, compression_context = self._determine_compression_strategy(
                prompt, field_name, field_path, context_data
            )
            
            # Apply timeout to compression
            compression_task = self.llmlingua_service.compress_prompt(
                prompt, compression_config, compression_context
            )
            
            result = await asyncio.wait_for(
                compression_task, 
                timeout=self.max_compression_time
            )
            
            if result.error and not self.fallback_on_timeout:
                logger.error(f"Compression failed for {field_path}: {result.error}")
                return prompt, result
            
            # Use compressed prompt if successful
            compressed_prompt = result.compressed_prompt
            if result.compression_ratio < 0.95:  # Only use if meaningful compression
                logger.info(
                    f"Compressed {field_path}: {result.original_tokens} -> "
                    f"{result.compressed_tokens} tokens ({result.compression_ratio:.2f} ratio)"
                )
                return compressed_prompt, result
            else:
                return prompt, result
                
        except asyncio.TimeoutError:
            logger.warning(f"Compression timeout for {field_path}")
            if self.fallback_on_timeout:
                return prompt, None
            else:
                raise
        except Exception as e:
            logger.error(f"Compression error for {field_path}: {e}")
            return prompt, None
    
    def _determine_compression_strategy(
        self, 
        prompt: str, 
        field_name: str, 
        field_path: str,
        context_data: Dict[str, Any]
    ) -> tuple[CompressionConfig, Dict[str, Any]]:
        """Determine appropriate compression strategy based on context"""
        
        # Default compression config
        config = CompressionConfig(
            compression_ratio=0.6,
            use_llmlingua2=True,
            fallback_on_error=True
        )
        
        context = {
            "field_name": field_name,
            "field_path": field_path,
            "api_endpoint": "middleware"
        }
        
        # Agent-specific compression
        if "agent" in context_data or "agent_role" in context_data:
            agent_role_str = context_data.get("agent_role", context_data.get("agent", "")).upper()
            if agent_role_str in [role.value for role in AgentRole]:
                agent_role = AgentRole(agent_role_str)
                
                # Use agent-specific compression
                config = self._get_agent_compression_config(agent_role, context_data)
                context["agent_role"] = agent_role_str
                context["compression_type"] = "agent_specific"
        
        # Analysis-specific compression
        elif "analysis" in field_path.lower() or "analysis" in context_data:
            config.compression_ratio = 0.7  # Conservative for analysis
            context["compression_type"] = "analysis"
        
        # Query-specific compression
        elif "query" in field_name.lower() or "search" in context_data:
            config.compression_ratio = 0.8  # Very conservative for queries
            context["compression_type"] = "query"
        
        # General prompt compression
        else:
            config.compression_ratio = 0.6  # Balanced default
            context["compression_type"] = "general"
        
        return config, context
    
    def _get_agent_compression_config(
        self, 
        agent_role: AgentRole, 
        context_data: Dict[str, Any]
    ) -> CompressionConfig:
        """Get compression config optimized for specific agent role"""
        
        agent_configs = {
            AgentRole.RESEARCHER: CompressionConfig(
                compression_ratio=0.7,
                force_tokens=["research", "analysis", "data", "findings"],
                preserve_instructions=True
            ),
            AgentRole.MASTERMIND: CompressionConfig(
                compression_ratio=0.5,
                force_tokens=["strategy", "architecture", "design", "plan"],
                preserve_semantic_integrity=True
            ),
            AgentRole.EXECUTOR: CompressionConfig(
                compression_ratio=0.6,
                force_tokens=["implementation", "testing", "deployment"],
                preserve_instructions=True
            )
        }
        
        return agent_configs.get(agent_role, CompressionConfig())
    
    async def _update_compression_stats(self, compression_results: List[CompressionResult]):
        """Update middleware compression statistics"""
        if not compression_results:
            return
        
        self.stats["compressed_requests"] += 1
        
        for result in compression_results:
            self.stats["total_tokens_saved"] += result.token_savings
            self.stats["total_time_saved"] += result.processing_time_ms / 1000
            self.stats["total_cost_saved"] += result.cost_savings_estimate
    
    def get_stats(self) -> Dict[str, Any]:
        """Get middleware compression statistics"""
        total_requests = self.stats["total_requests"]
        
        return {
            **self.stats,
            "compression_rate": (
                self.stats["compressed_requests"] / total_requests 
                if total_requests > 0 else 0.0
            ),
            "error_rate": (
                self.stats["compression_errors"] / total_requests 
                if total_requests > 0 else 0.0
            ),
            "average_tokens_saved_per_request": (
                self.stats["total_tokens_saved"] / self.stats["compressed_requests"]
                if self.stats["compressed_requests"] > 0 else 0.0
            ),
            "estimated_cost_savings_per_day": self.stats["total_cost_saved"] * 24,
            "config": {
                "enabled": self.enabled,
                "compress_threshold": self.compress_threshold,
                "max_compression_time": self.max_compression_time,
                "fallback_on_timeout": self.fallback_on_timeout
            }
        }
    
    async def reset_stats(self):
        """Reset compression statistics"""
        self.stats = {
            "total_requests": 0,
            "compressed_requests": 0,
            "compression_errors": 0,
            "total_tokens_saved": 0,
            "total_time_saved": 0.0,
            "total_cost_saved": 0.0
        }


class CompressionResponseMiddleware:
    """Middleware to add compression information to API responses"""
    
    def __init__(self, compression_middleware: CompressionMiddleware):
        self.compression_middleware = compression_middleware
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """Add compression metadata to response"""
        response = await call_next(request)
        
        # Add compression stats to response headers
        if hasattr(request.state, "compression_results"):
            compression_results = request.state.compression_results
            
            if compression_results:
                total_original_tokens = sum(r.original_tokens for r in compression_results)
                total_compressed_tokens = sum(r.compressed_tokens for r in compression_results)
                total_savings = sum(r.cost_savings_estimate for r in compression_results)
                
                response.headers["X-Original-Tokens"] = str(total_original_tokens)
                response.headers["X-Compressed-Tokens"] = str(total_compressed_tokens)
                response.headers["X-Token-Savings"] = str(total_original_tokens - total_compressed_tokens)
                response.headers["X-Cost-Savings"] = f"${total_savings:.4f}"
                response.headers["X-Compression-Count"] = str(len(compression_results))
        
        return response


# FastAPI middleware integration helpers
def create_compression_middleware(
    app,
    llmlingua_service: LLMLinguaService,
    agent_compression_service: AgentCompressionService,
    config: Optional[Dict[str, Any]] = None
) -> CompressionMiddleware:
    """Create and configure compression middleware for FastAPI app"""
    
    middleware = CompressionMiddleware(
        app, 
        llmlingua_service, 
        agent_compression_service, 
        config
    )
    
    return middleware


def add_compression_endpoint(app, compression_middleware: CompressionMiddleware):
    """Add compression statistics endpoint to FastAPI app"""
    
    @app.get("/api/compression/stats")
    async def get_compression_stats():
        """Get compression middleware statistics"""
        return compression_middleware.get_stats()
    
    @app.post("/api/compression/reset")
    async def reset_compression_stats():
        """Reset compression statistics"""
        await compression_middleware.reset_stats()
        return {"message": "Compression statistics reset"}
    
    @app.get("/api/compression/health")
    async def compression_health_check():
        """Health check for compression services"""
        llmlingua_health = await compression_middleware.llmlingua_service.health_check()
        agent_stats = await compression_middleware.agent_compression_service.get_agent_stats()
        
        return {
            "llmlingua_service": llmlingua_health,
            "agent_compression_service": "healthy",
            "agent_stats": agent_stats,
            "middleware_stats": compression_middleware.get_stats()
        }