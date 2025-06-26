"""
Database Optimization Layer for TradeKnowledge
Provides query caching, connection health monitoring, and performance tracking
"""

import asyncio
import hashlib
import time
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

from ..core.caching import get_cache

logger = structlog.get_logger(__name__)


@dataclass
class QueryMetrics:
    """Query performance metrics"""

    query_type: str
    execution_count: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    avg_time: float = 0.0
    last_executed: datetime | None = None
    error_count: int = 0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))

    def add_execution(self, execution_time: float, success: bool = True):
        """Add a query execution result"""
        self.execution_count += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.avg_time = self.total_time / self.execution_count
        self.last_executed = datetime.utcnow()
        self.recent_times.append(execution_time)

        if not success:
            self.error_count += 1

    def get_percentiles(self) -> dict[str, float]:
        """Calculate execution time percentiles"""
        if not self.recent_times:
            return {}

        sorted_times = sorted(self.recent_times)
        n = len(sorted_times)

        return {
            "p50": sorted_times[int(n * 0.5)],
            "p95": sorted_times[int(n * 0.95)],
            "p99": sorted_times[int(n * 0.99)],
        }


@dataclass
class ConnectionMetrics:
    """Database connection metrics"""

    active_connections: int = 0
    peak_connections: int = 0
    total_connections_created: int = 0
    connection_errors: int = 0
    connection_timeouts: int = 0
    avg_connection_time: float = 0.0


class QueryResultCache:
    """Intelligent query result caching"""

    def __init__(self, cache_instance=None):
        self.cache = cache_instance
        self.local_cache = {}  # Fallback local cache
        self.cache_stats = defaultdict(lambda: {"hits": 0, "misses": 0})

    async def initialize(self):
        """Initialize the cache"""
        if self.cache is None:
            self.cache = await get_cache()

    def _generate_cache_key(self, query: str, params: tuple) -> str:
        """Generate cache key for query and parameters"""
        key_data = f"{query}:{params}"
        return f"query:{hashlib.md5(key_data.encode()).hexdigest()}"

    async def get(self, query: str, params: tuple) -> Any | None:
        """Get cached query result"""
        cache_key = self._generate_cache_key(query, params)

        # Try distributed cache first
        if self.cache:
            try:
                result = await self.cache.get(cache_key)
                if result is not None:
                    self.cache_stats[query]["hits"] += 1
                    return result
            except Exception as e:
                logger.warning("Cache get failed", error=str(e))

        # Try local cache
        if cache_key in self.local_cache:
            entry = self.local_cache[cache_key]
            if entry["expires"] > time.time():
                self.cache_stats[query]["hits"] += 1
                return entry["data"]
            else:
                del self.local_cache[cache_key]

        self.cache_stats[query]["misses"] += 1
        return None

    async def set(self, query: str, params: tuple, result: Any, ttl: int = 300):
        """Cache query result"""
        cache_key = self._generate_cache_key(query, params)

        # Store in distributed cache
        if self.cache:
            try:
                await self.cache.set(cache_key, result, ttl=ttl, tags=["query"])
            except Exception as e:
                logger.warning("Cache set failed", error=str(e))

        # Store in local cache as backup
        self.local_cache[cache_key] = {"data": result, "expires": time.time() + ttl}

        # Limit local cache size
        if len(self.local_cache) > 1000:
            # Remove oldest entries
            oldest_keys = sorted(
                self.local_cache.keys(), key=lambda k: self.local_cache[k]["expires"]
            )[:100]
            for key in oldest_keys:
                del self.local_cache[key]

    async def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        if self.cache:
            try:
                await self.cache.clear_by_tags(["query"])
            except Exception as e:
                logger.warning("Cache invalidation failed", error=str(e))

        # Clear local cache
        self.local_cache.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        total_hits = sum(stats["hits"] for stats in self.cache_stats.values())
        total_misses = sum(stats["misses"] for stats in self.cache_stats.values())
        total_requests = total_hits + total_misses

        return {
            "total_requests": total_requests,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "hit_rate": total_hits / total_requests if total_requests > 0 else 0,
            "local_cache_size": len(self.local_cache),
            "query_stats": dict(self.cache_stats),
        }


class DatabaseOptimizer:
    """Main database optimization manager"""

    def __init__(self):
        self.query_cache = QueryResultCache()
        self.query_metrics = defaultdict(QueryMetrics)
        self.connection_metrics = ConnectionMetrics()
        self.prepared_statements = {}
        self.slow_query_threshold = 1.0  # 1 second
        self.optimization_suggestions = []

    async def initialize(self):
        """Initialize the optimizer"""
        await self.query_cache.initialize()
        logger.info("Database optimizer initialized")

    @asynccontextmanager
    async def track_query(self, query_type: str, query: str):
        """Context manager to track query execution"""
        start_time = time.time()
        success = False

        try:
            yield
            success = True
        except Exception as e:
            logger.error("Query failed", query_type=query_type, error=str(e))
            raise
        finally:
            execution_time = time.time() - start_time

            # Update metrics
            metrics = self.query_metrics[query_type]
            metrics.add_execution(execution_time, success)

            # Log slow queries
            if execution_time > self.slow_query_threshold:
                logger.warning(
                    "Slow query detected",
                    query_type=query_type,
                    execution_time=execution_time,
                    query=query[:200],  # First 200 chars
                )

                # Add optimization suggestion
                self._suggest_optimization(query_type, query, execution_time)

    def _suggest_optimization(self, query_type: str, query: str, execution_time: float):
        """Generate optimization suggestions for slow queries"""
        suggestions = []

        # Check for missing indexes
        if "WHERE" in query.upper() and "INDEX" not in query.upper():
            suggestions.append(
                {
                    "type": "missing_index",
                    "query_type": query_type,
                    "suggestion": "Consider adding an index for WHERE clause columns",
                    "impact": "high",
                }
            )

        # Check for SELECT *
        if "SELECT *" in query.upper():
            suggestions.append(
                {
                    "type": "select_star",
                    "query_type": query_type,
                    "suggestion": "Use specific column names instead of SELECT *",
                    "impact": "medium",
                }
            )

        # Check for LIKE operations without wildcards at start
        if "LIKE '%" in query or 'LIKE "%' in query:
            suggestions.append(
                {
                    "type": "leading_wildcard",
                    "query_type": query_type,
                    "suggestion": "Leading wildcards in LIKE prevent index usage",
                    "impact": "high",
                }
            )

        # Check for large LIMIT values
        import re

        limit_match = re.search(r"LIMIT\s+(\d+)", query, re.IGNORECASE)
        if limit_match and int(limit_match.group(1)) > 1000:
            suggestions.append(
                {
                    "type": "large_limit",
                    "query_type": query_type,
                    "suggestion": "Large LIMIT values may impact performance",
                    "impact": "medium",
                }
            )

        # Add suggestions to list
        for suggestion in suggestions:
            suggestion["execution_time"] = execution_time
            suggestion["timestamp"] = datetime.utcnow()
            self.optimization_suggestions.append(suggestion)

    async def execute_cached_query(
        self,
        connection,
        query: str,
        params: tuple = (),
        cache_ttl: int = 300,
        query_type: str = "unknown",
    ) -> list[Any]:
        """Execute query with caching"""

        # Check cache first
        cached_result = await self.query_cache.get(query, params)
        if cached_result is not None:
            logger.debug("Query cache hit", query_type=query_type)
            return cached_result

        # Execute query with tracking
        async with self.track_query(query_type, query):
            if asyncio.iscoroutinefunction(connection.execute):
                cursor = await connection.execute(query, params)
                result = await cursor.fetchall()
            else:
                # For synchronous connections
                result = await asyncio.to_thread(
                    lambda: connection.execute(query, params).fetchall()
                )

        # Cache the result
        await self.query_cache.set(query, params, result, cache_ttl)

        logger.debug("Query executed and cached", query_type=query_type)
        return result

    def get_prepared_statement(self, connection, statement_id: str, query: str):
        """Get or create prepared statement"""
        if statement_id not in self.prepared_statements:
            try:
                # SQLite doesn't have true prepared statements, but we can cache parsed queries
                self.prepared_statements[statement_id] = {
                    "query": query,
                    "created_at": datetime.utcnow(),
                    "usage_count": 0,
                }
            except Exception as e:
                logger.error(
                    "Failed to prepare statement",
                    statement_id=statement_id,
                    error=str(e),
                )
                return None

        self.prepared_statements[statement_id]["usage_count"] += 1
        return self.prepared_statements[statement_id]

    async def optimize_connection_pool(self, pool_size: int = 5) -> dict[str, Any]:
        """Analyze and optimize connection pool settings"""
        metrics = self.connection_metrics

        recommendations = []

        # Check pool utilization
        if metrics.peak_connections >= pool_size * 0.9:
            recommendations.append(
                {
                    "type": "pool_size",
                    "current": pool_size,
                    "recommended": pool_size + 2,
                    "reason": "High pool utilization detected",
                }
            )

        # Check connection errors
        error_rate = metrics.connection_errors / max(
            metrics.total_connections_created, 1
        )
        if error_rate > 0.05:  # 5% error rate
            recommendations.append(
                {
                    "type": "connection_reliability",
                    "error_rate": error_rate,
                    "reason": "High connection error rate",
                }
            )

        return {
            "current_metrics": {
                "active_connections": metrics.active_connections,
                "peak_connections": metrics.peak_connections,
                "total_created": metrics.total_connections_created,
                "error_rate": error_rate,
            },
            "recommendations": recommendations,
        }

    def get_performance_report(self) -> dict[str, Any]:
        """Generate comprehensive performance report"""

        # Query performance summary
        query_summary = {}
        for query_type, metrics in self.query_metrics.items():
            query_summary[query_type] = {
                "execution_count": metrics.execution_count,
                "avg_time": metrics.avg_time,
                "max_time": metrics.max_time,
                "error_rate": metrics.error_count / max(metrics.execution_count, 1),
                "percentiles": metrics.get_percentiles(),
            }

        # Top slow queries
        slow_queries = sorted(
            [(qt, m) for qt, m in self.query_metrics.items()],
            key=lambda x: x[1].avg_time,
            reverse=True,
        )[:5]

        # Recent optimization suggestions
        recent_suggestions = sorted(
            self.optimization_suggestions, key=lambda x: x["timestamp"], reverse=True
        )[:10]

        # Cache performance
        cache_stats = self.query_cache.get_stats()

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "query_performance": query_summary,
            "slow_queries": [
                {
                    "query_type": qt,
                    "avg_time": m.avg_time,
                    "execution_count": m.execution_count,
                }
                for qt, m in slow_queries
            ],
            "optimization_suggestions": recent_suggestions,
            "cache_performance": cache_stats,
            "connection_metrics": {
                "active": self.connection_metrics.active_connections,
                "peak": self.connection_metrics.peak_connections,
                "total_created": self.connection_metrics.total_connections_created,
                "errors": self.connection_metrics.connection_errors,
            },
        }

    async def invalidate_caches(self, pattern: str | None = None):
        """Invalidate query caches"""
        await self.query_cache.invalidate_pattern(pattern or "*")
        logger.info("Query caches invalidated", pattern=pattern)

    def suggest_indexes(
        self, table_name: str, frequent_columns: list[str]
    ) -> list[str]:
        """Suggest database indexes based on usage patterns"""
        suggestions = []

        # Single column indexes
        for column in frequent_columns:
            suggestions.append(
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_{column} ON {table_name}({column})"
            )

        # Composite indexes for multiple frequently used columns
        if len(frequent_columns) >= 2:
            composite_cols = ", ".join(frequent_columns[:3])  # Max 3 columns
            suggestions.append(
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_composite ON {table_name}({composite_cols})"
            )

        return suggestions


# Global optimizer instance
database_optimizer = DatabaseOptimizer()


async def get_database_optimizer() -> DatabaseOptimizer:
    """Get the global database optimizer instance"""
    return database_optimizer


# Decorator for automatic query optimization
def optimize_query(query_type: str, cache_ttl: int = 300):
    """Decorator to automatically optimize database queries"""

    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            optimizer = await get_database_optimizer()

            # Extract query parameters
            query = kwargs.get("query") or (args[1] if len(args) > 1 else None)
            params = kwargs.get("params") or (args[2] if len(args) > 2 else ())

            if query:
                # Use cached execution
                connection = args[0] if args else kwargs.get("connection")
                if connection:
                    return await optimizer.execute_cached_query(
                        connection, query, params, cache_ttl, query_type
                    )

            # Fallback to original function with tracking
            async with optimizer.track_query(query_type, query or "unknown"):
                return await func(*args, **kwargs)

        return wrapper

    return decorator
