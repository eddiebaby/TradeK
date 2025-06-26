"""
Database Optimization Service
Coordinates all database optimization features and provides unified management
"""

import asyncio
import time
from datetime import datetime
from typing import Any

import structlog

from ..core.config import get_config
from .connection_manager import connection_manager, get_connection_manager
from .database_optimizer import database_optimizer
from .index_manager import get_index_manager, run_index_maintenance
from .query_optimizer import get_query_analyzer

logger = structlog.get_logger(__name__)


class DatabaseOptimizationService:
    """Unified database optimization service"""

    def __init__(self):
        self.config = get_config()
        self.is_initialized = False
        self.maintenance_task = None
        self.performance_monitoring_task = None
        self.databases = {}  # Track databases being managed

        # Optimization settings
        self.optimization_enabled = True
        self.auto_maintenance_enabled = True
        self.performance_monitoring_enabled = True

    async def initialize(self):
        """Initialize the optimization service"""
        try:
            logger.info("Initializing database optimization service")

            # Initialize core optimizers
            await database_optimizer.initialize()

            # Register known databases
            await self._register_databases()

            # Start background tasks
            await self._start_background_tasks()

            self.is_initialized = True
            logger.info("✅ Database optimization service initialized successfully")

        except Exception as e:
            logger.error(
                "Failed to initialize database optimization service", error=str(e)
            )
            raise

    async def _register_databases(self):
        """Register databases for optimization"""
        # SQLite databases
        sqlite_config = self.config.database.sqlite
        main_db_path = sqlite_config.path

        self.databases["main"] = {
            "type": "sqlite",
            "path": main_db_path,
            "optimization_config": sqlite_config.optimization,
        }

        # Create index managers for databases
        if sqlite_config.optimization.enable_index_management:
            self.databases["main"]["index_manager"] = await get_index_manager(
                main_db_path
            )

        logger.info(f"Registered {len(self.databases)} databases for optimization")

    async def _start_background_tasks(self):
        """Start background optimization tasks"""
        if self.auto_maintenance_enabled:
            self.maintenance_task = asyncio.create_task(self._maintenance_loop())
            logger.info("Started database maintenance task")

        if self.performance_monitoring_enabled:
            self.performance_monitoring_task = asyncio.create_task(
                self._performance_monitoring_loop()
            )
            logger.info("Started performance monitoring task")

    async def _maintenance_loop(self):
        """Background maintenance loop"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self.run_maintenance()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Maintenance loop error", error=str(e))
                await asyncio.sleep(300)  # Wait 5 minutes before retrying

    async def _performance_monitoring_loop(self):
        """Background performance monitoring loop"""
        while True:
            try:
                await asyncio.sleep(1800)  # Run every 30 minutes
                await self._collect_performance_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Performance monitoring loop error", error=str(e))
                await asyncio.sleep(300)

    async def run_maintenance(self):
        """Run comprehensive database maintenance"""
        try:
            logger.info("Running database maintenance")
            start_time = time.time()

            # Run index maintenance
            await run_index_maintenance()

            # Optimize connection pools
            connection_stats = await self._optimize_connection_pools()

            # Clean up old cache entries
            await self._cleanup_caches()

            # Generate optimization recommendations
            recommendations = await self.get_optimization_recommendations()

            # Apply high-priority recommendations
            await self._apply_critical_recommendations(recommendations)

            end_time = time.time()
            logger.info(
                "Database maintenance completed",
                duration_seconds=end_time - start_time,
                recommendations_count=len(recommendations),
            )

        except Exception as e:
            logger.error("Database maintenance failed", error=str(e))

    async def _optimize_connection_pools(self) -> dict[str, Any]:
        """Optimize database connection pools"""
        connection_manager_instance = await get_connection_manager()
        return connection_manager_instance.get_all_stats()

    async def _cleanup_caches(self):
        """Clean up old cache entries"""
        try:
            await database_optimizer.query_cache.invalidate_pattern("expired_*")
            logger.debug("Cache cleanup completed")
        except Exception as e:
            logger.warning("Cache cleanup failed", error=str(e))

    async def _collect_performance_metrics(self):
        """Collect and log performance metrics"""
        try:
            # Get performance report
            performance_report = database_optimizer.get_performance_report()

            # Log key metrics
            query_performance = performance_report.get("query_performance", {})
            slow_queries = performance_report.get("slow_queries", [])
            cache_performance = performance_report.get("cache_performance", {})

            logger.info(
                "Performance metrics collected",
                total_query_types=len(query_performance),
                slow_queries_count=len(slow_queries),
                cache_hit_rate=cache_performance.get("hit_rate", 0),
            )

            # Alert on concerning metrics
            if cache_performance.get("hit_rate", 0) < 0.3:  # Less than 30% hit rate
                logger.warning(
                    "Low cache hit rate detected",
                    hit_rate=cache_performance.get("hit_rate", 0),
                )

            if len(slow_queries) > 10:  # More than 10 slow query types
                logger.warning(
                    "High number of slow queries detected",
                    slow_query_count=len(slow_queries),
                )

        except Exception as e:
            logger.error("Performance metrics collection failed", error=str(e))

    async def get_optimization_recommendations(self) -> list[dict[str, Any]]:
        """Get comprehensive optimization recommendations"""
        recommendations = []

        try:
            # Get query optimization recommendations
            query_analyzer_instance = await get_query_analyzer()
            query_recommendations = query_analyzer_instance.get_optimization_report()

            for suggestion in query_recommendations.get("index_suggestions", []):
                recommendations.append(
                    {
                        "type": "index_optimization",
                        "priority": suggestion.get("benefit", "medium"),
                        "description": f"Create index on {suggestion['table']}.{','.join(suggestion['columns'])}",
                        "action": suggestion.get("sql", ""),
                        "estimated_impact": suggestion.get("benefit", "medium"),
                    }
                )

            # Get connection pool recommendations
            for db_name, db_info in self.databases.items():
                if "index_manager" in db_info:
                    index_manager = db_info["index_manager"]
                    index_recommendations = (
                        await index_manager.get_optimization_recommendations()
                    )

                    for rec in index_recommendations:
                        recommendations.append(
                            {
                                "type": "database_index",
                                "database": db_name,
                                "priority": rec.get("estimated_benefit", "medium"),
                                "description": rec.get("reason", "Index optimization"),
                                "action": rec.get("sql", ""),
                                "table": rec.get("table", ""),
                                "columns": rec.get("columns", []),
                            }
                        )

            # Get cache optimization recommendations
            cache_stats = database_optimizer.query_cache.get_stats()
            if cache_stats.get("hit_rate", 0) < 0.5:
                recommendations.append(
                    {
                        "type": "cache_optimization",
                        "priority": "medium",
                        "description": "Consider increasing cache TTL or size",
                        "action": "Adjust cache configuration",
                        "current_hit_rate": cache_stats.get("hit_rate", 0),
                    }
                )

            return recommendations

        except Exception as e:
            logger.error("Failed to get optimization recommendations", error=str(e))
            return []

    async def _apply_critical_recommendations(
        self, recommendations: list[dict[str, Any]]
    ):
        """Apply critical optimization recommendations automatically"""
        applied_count = 0

        for rec in recommendations:
            if rec.get("priority") == "high" and rec.get("type") == "database_index":
                try:
                    database = rec.get("database", "main")
                    if (
                        database in self.databases
                        and "index_manager" in self.databases[database]
                    ):
                        index_manager = self.databases[database]["index_manager"]

                        # Apply index recommendation
                        if rec.get("action"):
                            result = await index_manager._create_index(
                                {
                                    "name": f"auto_{rec.get('table', 'unknown')}_{int(time.time())}",
                                    "table": rec.get("table", ""),
                                    "columns": rec.get("columns", []),
                                    "sql": rec.get("action", ""),
                                }
                            )

                            if result:
                                applied_count += 1
                                logger.info(
                                    "Applied critical recommendation",
                                    type=rec.get("type"),
                                    description=rec.get("description"),
                                )

                except Exception as e:
                    logger.warning(
                        "Failed to apply recommendation",
                        recommendation=rec.get("description"),
                        error=str(e),
                    )

        if applied_count > 0:
            logger.info(
                f"Applied {applied_count} critical optimization recommendations"
            )

    async def get_comprehensive_status(self) -> dict[str, Any]:
        """Get comprehensive optimization service status"""
        try:
            # Performance metrics
            performance_report = database_optimizer.get_performance_report()

            # Connection pool stats
            connection_stats = await self._optimize_connection_pools()

            # Index statistics
            index_stats = {}
            for db_name, db_info in self.databases.items():
                if "index_manager" in db_info:
                    index_manager = db_info["index_manager"]
                    index_stats[db_name] = await index_manager.get_index_statistics()

            # Recent recommendations
            recommendations = await self.get_optimization_recommendations()

            return {
                "service_status": {
                    "initialized": self.is_initialized,
                    "optimization_enabled": self.optimization_enabled,
                    "maintenance_running": self.maintenance_task is not None
                    and not self.maintenance_task.done(),
                    "monitoring_running": self.performance_monitoring_task is not None
                    and not self.performance_monitoring_task.done(),
                },
                "performance_metrics": performance_report,
                "connection_pools": connection_stats,
                "index_statistics": index_stats,
                "recent_recommendations": recommendations[
                    :10
                ],  # Top 10 recommendations
                "databases_managed": len(self.databases),
                "generated_at": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error("Failed to get comprehensive status", error=str(e))
            return {"error": str(e)}

    async def force_optimization_run(self) -> dict[str, Any]:
        """Force an immediate optimization run"""
        logger.info("Forcing immediate optimization run")

        start_time = time.time()

        # Run maintenance
        await self.run_maintenance()

        # Collect metrics
        await self._collect_performance_metrics()

        # Get updated recommendations
        recommendations = await self.get_optimization_recommendations()

        end_time = time.time()

        return {
            "forced_run_completed": True,
            "duration_seconds": end_time - start_time,
            "recommendations_generated": len(recommendations),
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def configure_optimization(self, settings: dict[str, Any]) -> bool:
        """Configure optimization settings"""
        try:
            if "optimization_enabled" in settings:
                self.optimization_enabled = settings["optimization_enabled"]

            if "auto_maintenance_enabled" in settings:
                old_maintenance = self.auto_maintenance_enabled
                self.auto_maintenance_enabled = settings["auto_maintenance_enabled"]

                # Restart maintenance task if needed
                if old_maintenance != self.auto_maintenance_enabled:
                    if self.maintenance_task:
                        self.maintenance_task.cancel()

                    if self.auto_maintenance_enabled:
                        self.maintenance_task = asyncio.create_task(
                            self._maintenance_loop()
                        )

            if "performance_monitoring_enabled" in settings:
                old_monitoring = self.performance_monitoring_enabled
                self.performance_monitoring_enabled = settings[
                    "performance_monitoring_enabled"
                ]

                # Restart monitoring task if needed
                if old_monitoring != self.performance_monitoring_enabled:
                    if self.performance_monitoring_task:
                        self.performance_monitoring_task.cancel()

                    if self.performance_monitoring_enabled:
                        self.performance_monitoring_task = asyncio.create_task(
                            self._performance_monitoring_loop()
                        )

            logger.info("Optimization configuration updated", settings=settings)
            return True

        except Exception as e:
            logger.error("Failed to configure optimization", error=str(e))
            return False

    async def cleanup(self):
        """Cleanup optimization service"""
        logger.info("Shutting down database optimization service")

        # Cancel background tasks
        if self.maintenance_task:
            self.maintenance_task.cancel()
            try:
                await self.maintenance_task
            except asyncio.CancelledError:
                pass

        if self.performance_monitoring_task:
            self.performance_monitoring_task.cancel()
            try:
                await self.performance_monitoring_task
            except asyncio.CancelledError:
                pass

        # Close connection pools
        await connection_manager.close_all_pools()

        logger.info("✅ Database optimization service shutdown completed")


# Global optimization service instance
optimization_service = DatabaseOptimizationService()


async def get_optimization_service() -> DatabaseOptimizationService:
    """Get the global optimization service instance"""
    return optimization_service


async def initialize_database_optimization():
    """Initialize database optimization globally"""
    await optimization_service.initialize()


async def cleanup_database_optimization():
    """Cleanup database optimization globally"""
    await optimization_service.cleanup()
