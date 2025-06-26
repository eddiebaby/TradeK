"""
LLMLingua Compression Service for TradeKnowledge
Provides prompt compression capabilities across all Claude Code instances
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import time

logger = logging.getLogger(__name__)


@dataclass
class CompressionConfig:
    """Configuration for LLMLingua compression"""
    # Compression settings
    target_token: Optional[int] = None
    compression_ratio: Optional[float] = None
    
    # Model settings
    model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
    use_llmlingua2: bool = True  # Use faster LLMLingua-2 by default
    
    # Quality settings
    preserve_semantic_integrity: bool = True
    iterative_size: int = 200
    condition_compare: bool = True
    condition_in_question: str = "after"
    context_budget: str = "+100"
    dataset: str = "sharegpt"
    
    # Performance settings
    max_compression_attempts: int = 3
    fallback_on_error: bool = True
    
    # Cache settings
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    
    # Context preservation
    preserve_question: bool = True
    preserve_instructions: bool = True
    force_tokens: List[str] = field(default_factory=lambda: [
        "SPARC", "RESEARCHER", "MASTERMIND", "EXECUTOR",
        "analysis", "trading", "market", "strategy"
    ])


@dataclass
class CompressionResult:
    """Result of prompt compression"""
    original_prompt: str
    compressed_prompt: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    processing_time_ms: float
    model_used: str
    quality_score: Optional[float] = None
    error: Optional[str] = None
    fallback_used: bool = False
    cache_hit: bool = False
    
    @property
    def token_savings(self) -> int:
        """Calculate token savings"""
        return self.original_tokens - self.compressed_tokens
    
    @property
    def cost_savings_estimate(self) -> float:
        """Estimate cost savings (assuming $0.01 per 1K tokens)"""
        return (self.token_savings / 1000) * 0.01


class LLMLinguaService:
    """Core LLMLingua compression service"""
    
    def __init__(self, db_service=None, redis_service=None):
        self.db_service = db_service
        self.redis_service = redis_service
        self._compressor = None
        self._compressor_v2 = None
        self._initialized = False
        self._fallback_mode = False
        self.compression_stats = {
            "total_compressions": 0,
            "total_tokens_saved": 0,
            "total_cost_saved": 0.0,
            "average_compression_ratio": 0.0
        }
    
    async def initialize(self):
        """Initialize LLMLingua compressors lazily"""
        if self._initialized:
            return
        
        try:
            # Import here to avoid blocking if not installed
            from llmlingua import PromptCompressor
            import torch
            
            # Check if CUDA is available, if not use CPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            # Use smaller model for CPU-only environments
            model_name = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank" if device == "cpu" else "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
            
            # Initialize LLMLingua (original) with device specification
            self._compressor = PromptCompressor(
                model_name=model_name,
                use_llmlingua2=False,
                device_map=device
            )
            
            # Initialize LLMLingua-2 (faster, better) with device specification
            self._compressor_v2 = PromptCompressor(
                model_name=model_name,
                use_llmlingua2=True,
                device_map=device
            )
            
            self._initialized = True
            logger.info(f"LLMLingua compressors initialized successfully on {device}")
            
        except ImportError as e:
            logger.error(f"LLMLingua not installed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize LLMLingua: {e}")
            # Set fallback mode instead of failing completely
            logger.warning("Initializing in fallback mode - compression will return original prompts")
            self._initialized = True
            self._fallback_mode = True
    
    async def compress_prompt(
        self, 
        prompt: str, 
        config: CompressionConfig = None,
        context: Optional[Dict[str, Any]] = None
    ) -> CompressionResult:
        """
        Compress a prompt using LLMLingua
        
        Args:
            prompt: The prompt to compress
            config: Compression configuration
            context: Additional context for compression decisions
        
        Returns:
            CompressionResult with compression details
        """
        start_time = time.time()
        config = config or CompressionConfig()
        
        # Check cache first
        if config.enable_caching and self.redis_service:
            cached_result = await self._get_cached_result(prompt, config)
            if cached_result:
                cached_result.cache_hit = True
                return cached_result
        
        await self.initialize()
        
        original_tokens = self._count_tokens(prompt)
        
        try:
            # Choose compressor based on config
            compressor = self._compressor_v2 if config.use_llmlingua2 else self._compressor
            
            # Prepare compression parameters
            compression_params = self._prepare_compression_params(prompt, config, context)
            
            # Perform compression
            compression_result = compressor.compress_prompt(**compression_params)
            
            # Extract compressed prompt
            compressed_prompt = compression_result.get('compressed_prompt', prompt)
            compressed_tokens = self._count_tokens(compressed_prompt)
            
            # Calculate metrics
            compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
            processing_time_ms = (time.time() - start_time) * 1000
            
            result = CompressionResult(
                original_prompt=prompt,
                compressed_prompt=compressed_prompt,
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                compression_ratio=compression_ratio,
                processing_time_ms=processing_time_ms,
                model_used="llmlingua-2" if config.use_llmlingua2 else "llmlingua",
                quality_score=compression_result.get('quality_score'),
                fallback_used=False
            )
            
            # Cache result
            if config.enable_caching and self.redis_service:
                await self._cache_result(prompt, config, result)
            
            # Update statistics
            await self._update_stats(result)
            
            # Log to InfluxDB for monitoring
            await self._log_compression_metrics(result, context)
            
            logger.info(
                f"Compressed prompt: {original_tokens} -> {compressed_tokens} tokens "
                f"({compression_ratio:.2f} ratio) in {processing_time_ms:.1f}ms"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            
            if config.fallback_on_error:
                # Return original prompt as fallback
                result = CompressionResult(
                    original_prompt=prompt,
                    compressed_prompt=prompt,
                    original_tokens=original_tokens,
                    compressed_tokens=original_tokens,
                    compression_ratio=1.0,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    model_used="fallback",
                    error=str(e),
                    fallback_used=True
                )
                return result
            else:
                raise
    
    async def compress_agent_prompt(
        self, 
        agent_role: str, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> CompressionResult:
        """
        Compress prompt for specific SPARC agent with optimized settings
        
        Args:
            agent_role: RESEARCHER, MASTERMIND, or EXECUTOR
            prompt: The agent prompt to compress
            context: Agent-specific context
        
        Returns:
            CompressionResult optimized for the agent type
        """
        # Agent-specific compression configurations
        agent_configs = {
            "RESEARCHER": CompressionConfig(
                compression_ratio=0.7,  # Preserve more context for research
                force_tokens=["data", "analysis", "research", "findings", "sources"],
                preserve_instructions=True
            ),
            "MASTERMIND": CompressionConfig(
                compression_ratio=0.5,  # Aggressive compression for strategy
                force_tokens=["strategy", "architecture", "design", "plan", "goals"],
                preserve_semantic_integrity=True
            ),
            "EXECUTOR": CompressionConfig(
                compression_ratio=0.6,  # Balanced for implementation
                force_tokens=["implementation", "testing", "code", "deployment"],
                preserve_question=True
            )
        }
        
        config = agent_configs.get(agent_role, CompressionConfig())
        
        # Add agent context
        agent_context = {
            "agent_role": agent_role,
            "compression_type": "agent_specific",
            **(context or {})
        }
        
        return await self.compress_prompt(prompt, config, agent_context)
    
    async def batch_compress(
        self, 
        prompts: List[str], 
        config: CompressionConfig = None
    ) -> List[CompressionResult]:
        """
        Compress multiple prompts in batch for efficiency
        
        Args:
            prompts: List of prompts to compress
            config: Compression configuration
        
        Returns:
            List of CompressionResults
        """
        config = config or CompressionConfig()
        
        # Process in parallel for better performance
        tasks = [
            self.compress_prompt(prompt, config, {"batch_index": i})
            for i, prompt in enumerate(prompts)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch compression failed for prompt {i}: {result}")
                # Create fallback result
                fallback_result = CompressionResult(
                    original_prompt=prompts[i],
                    compressed_prompt=prompts[i],
                    original_tokens=self._count_tokens(prompts[i]),
                    compressed_tokens=self._count_tokens(prompts[i]),
                    compression_ratio=1.0,
                    processing_time_ms=0.0,
                    model_used="fallback",
                    error=str(result),
                    fallback_used=True
                )
                processed_results.append(fallback_result)
            else:
                processed_results.append(result)
        
        return processed_results
    
    def _prepare_compression_params(
        self, 
        prompt: str, 
        config: CompressionConfig, 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare parameters for LLMLingua compression"""
        params = {
            "prompt": prompt,
            "iterative_size": config.iterative_size,
            "condition_compare": config.condition_compare,
            "condition_in_question": config.condition_in_question,
            "context_budget": config.context_budget,
            "dataset": config.dataset
        }
        
        # Set compression target
        if config.target_token:
            params["target_token"] = config.target_token
        elif config.compression_ratio:
            # Convert ratio to target token count
            estimated_tokens = self._count_tokens(prompt)
            params["target_token"] = int(estimated_tokens * config.compression_ratio)
        
        # Add force tokens if specified
        if config.force_tokens:
            params["force_tokens"] = config.force_tokens
        
        return params
    
    def _count_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except ImportError:
            # Fallback to rough estimation
            return len(text.split()) * 1.3  # Rough approximation
    
    async def _get_cached_result(
        self, 
        prompt: str, 
        config: CompressionConfig
    ) -> Optional[CompressionResult]:
        """Get cached compression result"""
        if not self.redis_service:
            return None
        
        try:
            cache_key = self._generate_cache_key(prompt, config)
            cached_data = await self.redis_service.get_json(cache_key)
            
            if cached_data:
                return CompressionResult(**cached_data)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    async def _cache_result(
        self, 
        prompt: str, 
        config: CompressionConfig, 
        result: CompressionResult
    ):
        """Cache compression result"""
        if not self.redis_service:
            return
        
        try:
            cache_key = self._generate_cache_key(prompt, config)
            # Convert to dict for JSON serialization
            result_dict = {
                "original_prompt": result.original_prompt,
                "compressed_prompt": result.compressed_prompt,
                "original_tokens": result.original_tokens,
                "compressed_tokens": result.compressed_tokens,
                "compression_ratio": result.compression_ratio,
                "processing_time_ms": result.processing_time_ms,
                "model_used": result.model_used,
                "quality_score": result.quality_score,
                "error": result.error,
                "fallback_used": result.fallback_used
            }
            
            await self.redis_service.set_json(cache_key, result_dict, config.cache_ttl)
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    def _generate_cache_key(self, prompt: str, config: CompressionConfig) -> str:
        """Generate cache key for prompt and config"""
        import hashlib
        
        # Create hash of prompt and relevant config
        content = f"{prompt}_{config.target_token}_{config.compression_ratio}_{config.use_llmlingua2}"
        return f"llmlingua_cache:{hashlib.md5(content.encode()).hexdigest()}"
    
    async def _update_stats(self, result: CompressionResult):
        """Update compression statistics"""
        self.compression_stats["total_compressions"] += 1
        self.compression_stats["total_tokens_saved"] += result.token_savings
        self.compression_stats["total_cost_saved"] += result.cost_savings_estimate
        
        # Update average compression ratio
        current_avg = self.compression_stats["average_compression_ratio"]
        count = self.compression_stats["total_compressions"]
        new_avg = ((current_avg * (count - 1)) + result.compression_ratio) / count
        self.compression_stats["average_compression_ratio"] = new_avg
    
    async def _log_compression_metrics(
        self, 
        result: CompressionResult, 
        context: Optional[Dict[str, Any]]
    ):
        """Log compression metrics to InfluxDB"""
        if not self.db_service or not hasattr(self.db_service, 'influx'):
            return
        
        try:
            metrics = {
                "original_tokens": result.original_tokens,
                "compressed_tokens": result.compressed_tokens,
                "compression_ratio": result.compression_ratio,
                "processing_time_ms": result.processing_time_ms,
                "tokens_saved": result.token_savings,
                "cost_saved": result.cost_savings_estimate,
                "model_used": result.model_used,
                "fallback_used": result.fallback_used,
                "cache_hit": result.cache_hit,
                "timestamp": datetime.utcnow()
            }
            
            # Add context tags
            if context:
                for key, value in context.items():
                    if isinstance(value, (str, int, float, bool)):
                        metrics[f"context_{key}"] = value
            
            await self.db_service.influx.write_analysis_metrics({
                "analysis_type": "prompt_compression",
                "model_version": result.model_used,
                **metrics
            })
            
        except Exception as e:
            logger.warning(f"Failed to log compression metrics: {e}")
    
    async def get_compression_stats(self) -> Dict[str, Any]:
        """Get current compression statistics"""
        return {
            **self.compression_stats,
            "service_initialized": self._initialized,
            "models_available": {
                "llmlingua": self._compressor is not None,
                "llmlingua_v2": self._compressor_v2 is not None
            }
        }
    
    async def health_check(self) -> Dict[str, str]:
        """Check health of LLMLingua service"""
        health = {}
        
        try:
            if not self._initialized:
                await self.initialize()
            
            # Test basic compression
            test_prompt = "This is a test prompt for health check."
            result = await self.compress_prompt(
                test_prompt, 
                CompressionConfig(target_token=5, enable_caching=False)
            )
            
            if result.error:
                health["llmlingua"] = f"unhealthy: {result.error}"
            else:
                health["llmlingua"] = "healthy"
                
        except Exception as e:
            health["llmlingua"] = f"unhealthy: {str(e)}"
        
        return health


# Global LLMLingua service instance
llmlingua_service = LLMLinguaService()


# Dependency injection for FastAPI
async def get_llmlingua_service() -> LLMLinguaService:
    """Dependency to get LLMLingua service"""
    return llmlingua_service