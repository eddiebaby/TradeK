"""
Compression Metrics and Monitoring Service
Tracks LLMLingua performance, cost savings, and quality metrics
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import statistics

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of compression metrics"""
    PERFORMANCE = "performance"
    COST = "cost"
    QUALITY = "quality"
    USAGE = "usage"
    ERROR = "error"


@dataclass
class CompressionMetric:
    """Individual compression metric data point"""
    timestamp: datetime
    metric_type: MetricType
    agent_role: Optional[str] = None
    endpoint: Optional[str] = None
    original_tokens: int = 0
    compressed_tokens: int = 0
    compression_ratio: float = 0.0
    processing_time_ms: float = 0.0
    cost_savings: float = 0.0
    quality_score: Optional[float] = None
    error_message: Optional[str] = None
    model_used: str = "unknown"
    fallback_used: bool = False
    cache_hit: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressionAnalytics:
    """Aggregated compression analytics"""
    time_period: str
    total_compressions: int
    total_tokens_saved: int
    total_cost_saved: float
    average_compression_ratio: float
    average_processing_time: float
    average_quality_score: Optional[float]
    error_rate: float
    cache_hit_rate: float
    fallback_rate: float
    
    # Performance breakdown
    by_agent: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_endpoint: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_model: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Trends
    compression_ratio_trend: List[float] = field(default_factory=list)
    cost_savings_trend: List[float] = field(default_factory=list)
    quality_trend: List[float] = field(default_factory=list)


class CompressionMetricsCollector:
    """Collects and stores compression metrics"""
    
    def __init__(self, db_service=None):
        self.db_service = db_service
        self.metrics_buffer: List[CompressionMetric] = []
        self.buffer_size = 1000
        self.flush_interval = 60  # seconds
        self.last_flush = datetime.utcnow()
        
        # Real-time stats
        self.realtime_stats = {
            "total_compressions": 0,
            "total_tokens_saved": 0,
            "total_cost_saved": 0.0,
            "compression_errors": 0,
            "cache_hits": 0,
            "fallback_used": 0
        }
    
    async def record_compression_metric(
        self,
        metric_type: MetricType,
        original_tokens: int = 0,
        compressed_tokens: int = 0,
        processing_time_ms: float = 0.0,
        cost_savings: float = 0.0,
        quality_score: Optional[float] = None,
        agent_role: Optional[str] = None,
        endpoint: Optional[str] = None,
        model_used: str = "unknown",
        error_message: Optional[str] = None,
        fallback_used: bool = False,
        cache_hit: bool = False,
        **metadata
    ):
        """Record a compression metric"""
        
        compression_ratio = (
            compressed_tokens / original_tokens 
            if original_tokens > 0 else 0.0
        )
        
        metric = CompressionMetric(
            timestamp=datetime.utcnow(),
            metric_type=metric_type,
            agent_role=agent_role,
            endpoint=endpoint,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            processing_time_ms=processing_time_ms,
            cost_savings=cost_savings,
            quality_score=quality_score,
            error_message=error_message,
            model_used=model_used,
            fallback_used=fallback_used,
            cache_hit=cache_hit,
            metadata=metadata
        )
        
        # Add to buffer
        self.metrics_buffer.append(metric)
        
        # Update real-time stats
        await self._update_realtime_stats(metric)
        
        # Check if buffer needs flushing
        if (len(self.metrics_buffer) >= self.buffer_size or 
            (datetime.utcnow() - self.last_flush).seconds >= self.flush_interval):
            await self._flush_metrics()
    
    async def _update_realtime_stats(self, metric: CompressionMetric):
        """Update real-time statistics"""
        if metric.metric_type == MetricType.PERFORMANCE:
            self.realtime_stats["total_compressions"] += 1
            self.realtime_stats["total_tokens_saved"] += (
                metric.original_tokens - metric.compressed_tokens
            )
            self.realtime_stats["total_cost_saved"] += metric.cost_savings
            
            if metric.cache_hit:
                self.realtime_stats["cache_hits"] += 1
            
            if metric.fallback_used:
                self.realtime_stats["fallback_used"] += 1
        
        elif metric.metric_type == MetricType.ERROR:
            self.realtime_stats["compression_errors"] += 1
    
    async def _flush_metrics(self):
        """Flush metrics buffer to persistent storage"""
        if not self.metrics_buffer:
            return
        
        try:
            # Write to InfluxDB
            await self._write_to_influxdb(self.metrics_buffer)
            
            # Write to PostgreSQL for long-term storage
            await self._write_to_postgresql(self.metrics_buffer)
            
            logger.info(f"Flushed {len(self.metrics_buffer)} compression metrics")
            
            # Clear buffer
            self.metrics_buffer.clear()
            self.last_flush = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Failed to flush compression metrics: {e}")
    
    async def _write_to_influxdb(self, metrics: List[CompressionMetric]):
        """Write metrics to InfluxDB for real-time monitoring"""
        if not self.db_service or not hasattr(self.db_service, 'influx'):
            return
        
        for metric in metrics:
            try:
                influx_data = {
                    "measurement": "compression_metrics",
                    "tags": {
                        "metric_type": metric.metric_type.value,
                        "agent_role": metric.agent_role or "unknown",
                        "endpoint": metric.endpoint or "unknown",
                        "model_used": metric.model_used,
                        "fallback_used": str(metric.fallback_used),
                        "cache_hit": str(metric.cache_hit)
                    },
                    "fields": {
                        "original_tokens": metric.original_tokens,
                        "compressed_tokens": metric.compressed_tokens,
                        "compression_ratio": metric.compression_ratio,
                        "processing_time_ms": metric.processing_time_ms,
                        "cost_savings": metric.cost_savings,
                        "quality_score": metric.quality_score or 0.0,
                        "tokens_saved": metric.original_tokens - metric.compressed_tokens
                    },
                    "timestamp": metric.timestamp
                }
                
                # Add metadata as fields
                for key, value in metric.metadata.items():
                    if isinstance(value, (int, float, bool, str)):
                        influx_data["fields"][f"meta_{key}"] = value
                
                await self.db_service.influx.write_analysis_metrics(influx_data)
                
            except Exception as e:
                logger.warning(f"Failed to write metric to InfluxDB: {e}")
    
    async def _write_to_postgresql(self, metrics: List[CompressionMetric]):
        """Write metrics summary to PostgreSQL"""
        if not self.db_service or not hasattr(self.db_service, 'postgres'):
            return
        
        try:
            # Aggregate metrics for batch insert
            summary_data = self._aggregate_metrics_for_storage(metrics)
            
            # This would be a proper table in the database
            # For now, just log the summary
            logger.info(f"PostgreSQL summary: {summary_data}")
            
        except Exception as e:
            logger.warning(f"Failed to write metrics to PostgreSQL: {e}")
    
    def _aggregate_metrics_for_storage(self, metrics: List[CompressionMetric]) -> Dict[str, Any]:
        """Aggregate metrics for efficient storage"""
        if not metrics:
            return {}
        
        performance_metrics = [m for m in metrics if m.metric_type == MetricType.PERFORMANCE]
        
        if not performance_metrics:
            return {}
        
        return {
            "time_period": f"{metrics[0].timestamp.isoformat()} - {metrics[-1].timestamp.isoformat()}",
            "total_compressions": len(performance_metrics),
            "total_tokens_saved": sum(m.original_tokens - m.compressed_tokens for m in performance_metrics),
            "total_cost_saved": sum(m.cost_savings for m in performance_metrics),
            "average_compression_ratio": statistics.mean(m.compression_ratio for m in performance_metrics),
            "average_processing_time": statistics.mean(m.processing_time_ms for m in performance_metrics),
            "error_count": len([m for m in metrics if m.metric_type == MetricType.ERROR]),
            "cache_hits": len([m for m in performance_metrics if m.cache_hit]),
            "fallback_usage": len([m for m in performance_metrics if m.fallback_used])
        }
    
    async def get_realtime_stats(self) -> Dict[str, Any]:
        """Get current real-time statistics"""
        total_compressions = self.realtime_stats["total_compressions"]
        
        return {
            **self.realtime_stats,
            "cache_hit_rate": (
                self.realtime_stats["cache_hits"] / total_compressions
                if total_compressions > 0 else 0.0
            ),
            "error_rate": (
                self.realtime_stats["compression_errors"] / total_compressions
                if total_compressions > 0 else 0.0
            ),
            "fallback_rate": (
                self.realtime_stats["fallback_used"] / total_compressions
                if total_compressions > 0 else 0.0
            ),
            "buffer_size": len(self.metrics_buffer),
            "last_flush": self.last_flush.isoformat()
        }


class CompressionAnalyticsService:
    """Service for analyzing compression metrics and generating insights"""
    
    def __init__(self, db_service=None):
        self.db_service = db_service
        self.metrics_collector = CompressionMetricsCollector(db_service)
    
    async def generate_analytics_report(
        self,
        start_time: datetime,
        end_time: datetime,
        breakdown_by: Optional[List[str]] = None
    ) -> CompressionAnalytics:
        """Generate comprehensive analytics report for time period"""
        
        # This would query InfluxDB for the time period
        # For now, we'll create a sample report structure
        analytics = CompressionAnalytics(
            time_period=f"{start_time.isoformat()} - {end_time.isoformat()}",
            total_compressions=0,
            total_tokens_saved=0,
            total_cost_saved=0.0,
            average_compression_ratio=0.0,
            average_processing_time=0.0,
            average_quality_score=None,
            error_rate=0.0,
            cache_hit_rate=0.0,
            fallback_rate=0.0
        )
        
        # Query metrics from InfluxDB
        if self.db_service and hasattr(self.db_service, 'influx'):
            try:
                await self._populate_analytics_from_influx(analytics, start_time, end_time)
            except Exception as e:
                logger.error(f"Failed to generate analytics from InfluxDB: {e}")
        
        return analytics
    
    async def _populate_analytics_from_influx(
        self,
        analytics: CompressionAnalytics,
        start_time: datetime,
        end_time: datetime
    ):
        """Populate analytics from InfluxDB data"""
        # This would contain actual InfluxDB queries
        # Sample implementation structure:
        
        query = f"""
        from(bucket: "market_data")
            |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
            |> filter(fn: (r) => r._measurement == "compression_metrics")
        """
        
        # Execute query and populate analytics
        # For now, just log the query
        logger.info(f"Analytics query: {query}")
    
    async def get_compression_trends(
        self,
        metric_type: str = "compression_ratio",
        time_window: str = "1h",
        lookback_hours: int = 24
    ) -> List[Tuple[datetime, float]]:
        """Get compression trends over time"""
        
        # This would query InfluxDB for trend data
        trends = []
        
        # Sample trend data structure
        now = datetime.utcnow()
        for i in range(lookback_hours):
            timestamp = now - timedelta(hours=i)
            value = 0.6 + (i % 5) * 0.05  # Sample trend data
            trends.append((timestamp, value))
        
        return trends
    
    async def identify_optimization_opportunities(self) -> Dict[str, Any]:
        """Identify opportunities for compression optimization"""
        
        opportunities = {
            "high_token_endpoints": [],
            "slow_compression_operations": [],
            "low_quality_compressions": [],
            "high_error_rate_agents": [],
            "cache_optimization_potential": [],
            "recommendations": []
        }
        
        # Analyze recent metrics to identify optimization opportunities
        realtime_stats = await self.metrics_collector.get_realtime_stats()
        
        # Sample optimization logic
        if realtime_stats["error_rate"] > 0.1:
            opportunities["recommendations"].append(
                "High error rate detected. Consider increasing compression timeouts or enabling fallbacks."
            )
        
        if realtime_stats["cache_hit_rate"] < 0.3:
            opportunities["recommendations"].append(
                "Low cache hit rate. Consider increasing cache TTL or optimizing cache keys."
            )
        
        if realtime_stats["fallback_rate"] > 0.2:
            opportunities["recommendations"].append(
                "High fallback rate. Review compression thresholds and timeout settings."
            )
        
        return opportunities
    
    async def generate_cost_savings_projection(
        self,
        projection_days: int = 30
    ) -> Dict[str, Any]:
        """Generate cost savings projection based on current trends"""
        
        realtime_stats = await self.metrics_collector.get_realtime_stats()
        daily_cost_savings = realtime_stats["total_cost_saved"]
        
        projection = {
            "current_daily_savings": daily_cost_savings,
            "projected_monthly_savings": daily_cost_savings * 30,
            "projected_yearly_savings": daily_cost_savings * 365,
            "token_efficiency": {
                "tokens_saved_per_day": realtime_stats["total_tokens_saved"],
                "compression_effectiveness": realtime_stats.get("cache_hit_rate", 0.0)
            },
            "optimization_potential": {
                "if_error_rate_improved": daily_cost_savings * 1.1,
                "if_cache_hit_rate_improved": daily_cost_savings * 1.2,
                "if_compression_ratio_improved": daily_cost_savings * 1.3
            }
        }
        
        return projection
    
    async def export_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
        format_type: str = "json"
    ) -> str:
        """Export compression metrics for external analysis"""
        
        analytics = await self.generate_analytics_report(start_time, end_time)
        
        if format_type.lower() == "json":
            return json.dumps({
                "analytics": analytics.__dict__,
                "export_timestamp": datetime.utcnow().isoformat(),
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                }
            }, indent=2, default=str)
        
        # Could add CSV, Excel, or other formats
        return json.dumps(analytics.__dict__, default=str)


# Global analytics service
compression_analytics_service = None

async def get_compression_analytics_service(db_service=None) -> CompressionAnalyticsService:
    """Get or create compression analytics service"""
    global compression_analytics_service
    if compression_analytics_service is None:
        compression_analytics_service = CompressionAnalyticsService(db_service)
    return compression_analytics_service