"""
Hybrid Embedding Router

Smart routing system that optimizes between local and OpenAI embeddings
to maximize quality while staying within the 1GB OpenAI limit.
"""

import asyncio
import hashlib
import json
import logging
import os
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import aiofiles

from ..core.config import get_config
from ..core.models import Chunk
from .embeddings import EmbeddingGenerator  # OpenAI embeddings
from .local_embeddings import LocalEmbeddingGenerator  # Local embeddings

logger = logging.getLogger(__name__)


class EmbeddingProvider(Enum):
    """Embedding provider options"""
    LOCAL = "local"          # nomic-embed-text via Ollama
    OPENAI = "openai"        # OpenAI text-embedding-3-small
    HYBRID = "hybrid"        # Smart routing


@dataclass
class ContentPriority:
    """Content priority classification for embedding routing"""
    level: str              # high, medium, low
    size_bytes: int         # Content size in bytes
    content_type: str       # academic_paper, trading_book, financial_report, etc.
    importance_score: float # 0.0-1.0 importance rating
    requires_precision: bool # True for mathematical content, formulas


@dataclass
class EmbeddingUsage:
    """Track OpenAI embedding usage"""
    total_tokens: int
    total_size_bytes: int
    documents_processed: int
    last_usage: datetime
    remaining_quota: int    # Bytes remaining in 1GB limit


class OpenAIQuotaManager:
    """Manage OpenAI 1GB embedding quota"""
    
    def __init__(self, quota_file: str = "./data/openai_quota.json"):
        self.quota_file = Path(quota_file)
        self.max_quota_bytes = 1024 * 1024 * 1024  # 1GB limit
        self.usage = self._load_usage()
        
    def _load_usage(self) -> EmbeddingUsage:
        """Load usage from file"""
        if self.quota_file.exists():
            try:
                with open(self.quota_file, 'r') as f:
                    data = json.load(f)
                    return EmbeddingUsage(
                        total_tokens=data.get('total_tokens', 0),
                        total_size_bytes=data.get('total_size_bytes', 0),
                        documents_processed=data.get('documents_processed', 0),
                        last_usage=datetime.fromisoformat(data.get('last_usage', datetime.now().isoformat())),
                        remaining_quota=self.max_quota_bytes - data.get('total_size_bytes', 0)
                    )
            except Exception as e:
                logger.warning(f"Error loading quota file: {e}")
        
        # Initialize with full quota
        return EmbeddingUsage(
            total_tokens=0,
            total_size_bytes=0,
            documents_processed=0,
            last_usage=datetime.now(),
            remaining_quota=self.max_quota_bytes
        )
    
    async def _save_usage(self):
        """Save usage to file"""
        try:
            self.quota_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'total_tokens': self.usage.total_tokens,
                'total_size_bytes': self.usage.total_size_bytes,
                'documents_processed': self.usage.documents_processed,
                'last_usage': self.usage.last_usage.isoformat(),
                'quota_limit_bytes': self.max_quota_bytes
            }
            
            async with aiofiles.open(self.quota_file, 'w') as f:
                await f.write(json.dumps(data, indent=2))
                
        except Exception as e:
            logger.error(f"Error saving quota: {e}")
    
    def can_process(self, content_size_bytes: int) -> bool:
        """Check if content can be processed within quota"""
        return self.usage.remaining_quota >= content_size_bytes
    
    def get_usage_percentage(self) -> float:
        """Get current usage as percentage of quota"""
        return (self.usage.total_size_bytes / self.max_quota_bytes) * 100
    
    async def record_usage(self, content_size_bytes: int, tokens_used: int):
        """Record OpenAI embedding usage"""
        self.usage.total_size_bytes += content_size_bytes
        self.usage.total_tokens += tokens_used
        self.usage.documents_processed += 1
        self.usage.last_usage = datetime.now()
        self.usage.remaining_quota = self.max_quota_bytes - self.usage.total_size_bytes
        
        await self._save_usage()
        
        # Log quota status
        usage_pct = self.get_usage_percentage()
        logger.info(f"OpenAI usage: {usage_pct:.1f}% ({self.usage.total_size_bytes:,} bytes)")
        
        if usage_pct > 80:
            logger.warning(f"OpenAI quota usage high: {usage_pct:.1f}%")
        elif usage_pct > 95:
            logger.error(f"OpenAI quota nearly exhausted: {usage_pct:.1f}%")


class HybridEmbeddingRouter:
    """
    Smart embedding router that optimizes between local and OpenAI embeddings.
    
    Features:
    - Automatic routing based on content priority and quota
    - OpenAI 1GB quota management
    - Quality vs cost optimization
    - Backup and sync capabilities
    """
    
    def __init__(self, config=None):
        """Initialize hybrid embedding router"""
        self.config = config or get_config()
        
        # Initialize embedding generators
        self.local_generator = LocalEmbeddingGenerator(config)
        self.openai_generator = EmbeddingGenerator(config)
        
        # Initialize quota manager
        self.quota_manager = OpenAIQuotaManager()
        
        # Routing statistics
        self.stats = {
            'local_embeddings': 0,
            'openai_embeddings': 0,
            'quota_blocks': 0,
            'total_processed': 0
        }
    
    def classify_content_priority(self, chunk: Chunk, document_metadata: Dict = None) -> ContentPriority:
        """Classify content priority for embedding routing"""
        
        content = chunk.content
        size_bytes = len(content.encode('utf-8'))
        
        # Default classification
        priority = ContentPriority(
            level="medium",
            size_bytes=size_bytes,
            content_type="general",
            importance_score=0.5,
            requires_precision=False
        )
        
        # High priority indicators
        high_priority_indicators = [
            'abstract', 'conclusion', 'executive summary',
            'key findings', 'results', 'trading strategy',
            'risk management', 'portfolio optimization'
        ]
        
        # Mathematical content indicators
        math_indicators = [
            '∑', '∫', '∂', '∇', 'equation', 'formula', 
            'algorithm', '\\', '$', 'theorem', 'proof'
        ]
        
        # Analyze content
        content_lower = content.lower()
        
        # Check for high priority content
        if any(indicator in content_lower for indicator in high_priority_indicators):
            priority.level = "high"
            priority.importance_score = 0.9
        
        # Check for mathematical content (needs precision)
        if any(indicator in content for indicator in math_indicators):
            priority.requires_precision = True
            priority.importance_score = min(priority.importance_score + 0.3, 1.0)
        
        # Classify content type from metadata
        if document_metadata:
            doc_name = document_metadata.get('file_name', '').lower()
            if 'paper' in doc_name or '.pdf' in doc_name:
                priority.content_type = "academic_paper"
            elif 'trading' in doc_name or 'financial' in doc_name:
                priority.content_type = "trading_book"
                priority.importance_score = min(priority.importance_score + 0.2, 1.0)
        
        # Size-based adjustments
        if size_bytes > 10000:  # Large chunks get lower priority for OpenAI
            priority.importance_score = max(priority.importance_score - 0.2, 0.0)
        
        return priority
    
    def should_use_openai(self, priority: ContentPriority) -> bool:
        """Determine if content should use OpenAI embeddings"""
        
        # Check quota availability
        if not self.quota_manager.can_process(priority.size_bytes):
            logger.info(f"OpenAI quota insufficient for {priority.size_bytes} bytes")
            return False
        
        # Check if quota usage is too high (reserve quota for critical content)
        usage_pct = self.quota_manager.get_usage_percentage()
        if usage_pct > 80 and priority.level != "high":
            return False
        
        # Decision matrix
        if priority.level == "high" and priority.size_bytes < 5000:
            return True  # High priority, small size
        
        if priority.requires_precision and priority.importance_score > 0.8:
            return True  # Mathematical content with high importance
        
        if priority.content_type == "trading_book" and priority.importance_score > 0.7:
            return True  # Important trading content
        
        # Default to local embeddings
        return False
    
    async def generate_embeddings(self, chunks: List[Chunk], document_metadata: Dict = None) -> List[List[float]]:
        """
        Generate embeddings using optimal routing strategy.
        
        Args:
            chunks: List of text chunks to embed
            document_metadata: Document metadata for routing decisions
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for chunk in chunks:
            # Classify content priority
            priority = self.classify_content_priority(chunk, document_metadata)
            
            # Route to appropriate embedding provider
            if self.should_use_openai(priority):
                embedding = await self._generate_openai_embedding(chunk, priority)
                self.stats['openai_embeddings'] += 1
                logger.info(f"OpenAI embedding: {priority.level} priority, {priority.size_bytes} bytes")
            else:
                embedding = await self._generate_local_embedding(chunk, priority)
                self.stats['local_embeddings'] += 1
                logger.debug(f"Local embedding: {priority.level} priority, {priority.size_bytes} bytes")
            
            embeddings.append(embedding)
            self.stats['total_processed'] += 1
        
        # Log routing statistics
        if self.stats['total_processed'] % 10 == 0:
            self._log_routing_stats()
        
        return embeddings
    
    async def _generate_openai_embedding(self, chunk: Chunk, priority: ContentPriority) -> List[float]:
        """Generate OpenAI embedding and track usage"""
        try:
            # Generate embedding
            embeddings = await self.openai_generator.generate_embeddings([chunk])
            
            # Record usage
            await self.quota_manager.record_usage(
                content_size_bytes=priority.size_bytes,
                tokens_used=len(chunk.content.split()) * 2  # Rough estimate
            )
            
            return embeddings[0] if embeddings else []
            
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            # Fallback to local embedding
            return await self._generate_local_embedding(chunk, priority)
    
    async def _generate_local_embedding(self, chunk: Chunk, priority: ContentPriority) -> List[float]:
        """Generate local embedding"""
        try:
            embeddings = await self.local_generator.generate_embeddings([chunk])
            return embeddings[0] if embeddings else []
        except Exception as e:
            logger.error(f"Local embedding failed: {e}")
            return []
    
    def _log_routing_stats(self):
        """Log routing statistics"""
        total = self.stats['total_processed']
        if total == 0:
            return
        
        local_pct = (self.stats['local_embeddings'] / total) * 100
        openai_pct = (self.stats['openai_embeddings'] / total) * 100
        quota_pct = self.quota_manager.get_usage_percentage()
        
        logger.info(f"Embedding routing stats: {local_pct:.1f}% local, {openai_pct:.1f}% OpenAI")
        logger.info(f"OpenAI quota usage: {quota_pct:.1f}%")
    
    async def backup_embeddings(self, embeddings: List[List[float]], metadata: Dict) -> bool:
        """Backup embeddings for offline access"""
        try:
            backup_dir = Path("./data/embedding_backups")
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Create backup file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"embeddings_backup_{timestamp}.json"
            
            backup_data = {
                'metadata': metadata,
                'embeddings': embeddings,
                'timestamp': timestamp,
                'routing_stats': self.stats
            }
            
            async with aiofiles.open(backup_file, 'w') as f:
                await f.write(json.dumps(backup_data, indent=2))
            
            logger.info(f"Embeddings backed up to: {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"Embedding backup failed: {e}")
            return False
    
    def get_quota_status(self) -> Dict:
        """Get current quota status"""
        return {
            'usage_percentage': self.quota_manager.get_usage_percentage(),
            'remaining_bytes': self.quota_manager.usage.remaining_quota,
            'total_documents': self.quota_manager.usage.documents_processed,
            'routing_stats': self.stats
        }