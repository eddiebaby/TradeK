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