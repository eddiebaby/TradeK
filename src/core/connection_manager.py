"""
Advanced Database Connection Management
Provides connection pooling, health monitoring, and automatic recovery
"""

import asyncio
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncContextManager

import structlog

from .database_optimizer import database_optimizer

logger = structlog.get_logger(__name__)


@dataclass
class ConnectionInfo:
    """Information about a database connection"""

    connection_id: str
    created_at: datetime
    last_used: datetime
    usage_count: int
    is_healthy: bool
    is_busy: bool
    database_path: str
    thread_id: int | None = None


class AdvancedConnectionPool:
    """Advanced SQLite connection pool with health monitoring"""

    def __init__(
        self,
        database_path: str,
        min_connections: int = 2,
        max_connections: int = 10,
        connection_timeout: int = 30,
        health_check_interval: int = 60,
    ):
        self.database_path = database_path
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.health_check_interval = health_check_interval

        self._pool: list[sqlite3.Connection] = []
        self._connection_info: dict[str, ConnectionInfo] = {}
        self._pool_lock = asyncio.Lock()
        self._busy_connections: set = set()

        self._health_check_task = None
        self._stats = {
            "total_created": 0,
            "total_destroyed": 0,
            "current_active": 0,
            "peak_active": 0,
            "connection_errors": 0,
            "health_check_failures": 0,
        }

    async def initialize(self):
        """Initialize the connection pool"""
        logger.info(
            "Initializing connection pool",
            database=self.database_path,
            min_connections=self.min_connections,
            max_connections=self.max_connections,
        )

        # Create minimum connections
        for _ in range(self.min_connections):
            await self._create_connection()

        # Start health check task
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        logger.info("Connection pool initialized", active_connections=len(self._pool))

    async def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection with optimized settings"""
        try:
            # Ensure database directory exists
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)

            # Create connection with optimized settings
            conn = await asyncio.to_thread(
                sqlite3.connect,
                self.database_path,
                timeout=self.connection_timeout,
                check_same_thread=False,
                isolation_level=None,  # Autocommit mode
            )

            # Apply performance optimizations
            await asyncio.to_thread(self._optimize_connection, conn)

            # Track connection info
            conn_id = f"conn_{id(conn)}"
            self._connection_info[conn_id] = ConnectionInfo(
                connection_id=conn_id,
                created_at=datetime.utcnow(),
                last_used=datetime.utcnow(),
                usage_count=0,
                is_healthy=True,
                is_busy=False,
                database_path=self.database_path,
                thread_id=threading.get_ident(),
            )

            # Update optimizer metrics
            database_optimizer.connection_metrics.total_connections_created += 1
            database_optimizer.connection_metrics.active_connections += 1
            database_optimizer.connection_metrics.peak_connections = max(
                database_optimizer.connection_metrics.peak_connections,
                database_optimizer.connection_metrics.active_connections,
            )

            self._stats["total_created"] += 1
            self._stats["current_active"] += 1
            self._stats["peak_active"] = max(
                self._stats["peak_active"], self._stats["current_active"]
            )

            logger.debug("Database connection created", connection_id=conn_id)
            return conn

        except Exception as e:
            self._stats["connection_errors"] += 1
            database_optimizer.connection_metrics.connection_errors += 1
            logger.error("Failed to create database connection", error=str(e))
            raise

    def _optimize_connection(self, conn: sqlite3.Connection):
        """Apply performance optimizations to connection"""
        optimizations = [
            "PRAGMA journal_mode=WAL",  # Write-Ahead Logging for better concurrency
            "PRAGMA synchronous=NORMAL",  # Good balance of safety and speed
            "PRAGMA cache_size=10000",  # Larger cache (10MB)
            "PRAGMA temp_store=MEMORY",  # Store temp tables in memory
            "PRAGMA mmap_size=268435456",  # 256MB memory-mapped I/O
            "PRAGMA optimize",  # Optimize query planner
        ]

        for pragma in optimizations:
            try:
                conn.execute(pragma)
            except sqlite3.Error as e:
                logger.warning(f"Failed to apply optimization: {pragma}", error=str(e))

    async def _validate_connection(self, conn: sqlite3.Connection) -> bool:
        """Validate that a connection is still healthy"""
        try:
            # Simple query to test connection
            await asyncio.to_thread(conn.execute, "SELECT 1")
            return True
        except Exception as e:
            logger.warning("Connection health check failed", error=str(e))
            return False

    async def _health_check_loop(self):
        """Periodic health check for all connections"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health check loop error", error=str(e))
                await asyncio.sleep(5)  # Brief pause before retrying

    async def _perform_health_check(self):
        """Perform health check on all connections"""
        async with self._pool_lock:
            unhealthy_connections = []

            for i, conn in enumerate(self._pool):
                conn_id = f"conn_{id(conn)}"

                # Skip busy connections
                if conn_id in self._busy_connections:
                    continue

                # Check connection health
                is_healthy = await self._validate_connection(conn)

                if conn_id in self._connection_info:
                    self._connection_info[conn_id].is_healthy = is_healthy

                if not is_healthy:
                    unhealthy_connections.append((i, conn, conn_id))
                    self._stats["health_check_failures"] += 1

            # Remove and replace unhealthy connections
            for i, conn, conn_id in reversed(
                unhealthy_connections
            ):  # Reverse to maintain indices
                try:
                    await asyncio.to_thread(conn.close)
                    self._pool.pop(i)

                    if conn_id in self._connection_info:
                        del self._connection_info[conn_id]

                    self._stats["total_destroyed"] += 1
                    self._stats["current_active"] -= 1
                    database_optimizer.connection_metrics.active_connections -= 1

                    # Create replacement connection
                    if len(self._pool) < self.min_connections:
                        new_conn = await self._create_connection()
                        self._pool.append(new_conn)

                    logger.info("Replaced unhealthy connection", connection_id=conn_id)

                except Exception as e:
                    logger.error(
                        "Failed to replace unhealthy connection",
                        connection_id=conn_id,
                        error=str(e),
                    )

    @asynccontextmanager
    async def get_connection(self) -> AsyncContextManager[sqlite3.Connection]:
        """Get a connection from the pool"""
        conn = None
        conn_id = None
        start_time = time.time()

        try:
            async with self._pool_lock:
                # Find available connection
                for connection in self._pool:
                    conn_id = f"conn_{id(connection)}"
                    if conn_id not in self._busy_connections:
                        conn = connection
                        self._busy_connections.add(conn_id)
                        break

                # Create new connection if none available and under limit
                if conn is None and len(self._pool) < self.max_connections:
                    conn = await self._create_connection()
                    self._pool.append(conn)
                    conn_id = f"conn_{id(conn)}"
                    self._busy_connections.add(conn_id)

                # Wait for available connection if at limit
                if conn is None:
                    logger.warning(
                        "Connection pool exhausted, waiting for available connection"
                    )

                    # Wait with timeout
                    timeout = 30  # 30 seconds
                    while conn is None and (time.time() - start_time) < timeout:
                        await asyncio.sleep(0.1)

                        for connection in self._pool:
                            check_conn_id = f"conn_{id(connection)}"
                            if check_conn_id not in self._busy_connections:
                                conn = connection
                                conn_id = check_conn_id
                                self._busy_connections.add(conn_id)
                                break

                    if conn is None:
                        database_optimizer.connection_metrics.connection_timeouts += 1
                        raise TimeoutError("Timeout waiting for database connection")

            # Update connection info
            if conn_id in self._connection_info:
                info = self._connection_info[conn_id]
                info.last_used = datetime.utcnow()
                info.usage_count += 1
                info.is_busy = True

            # Validate connection before use
            if not await self._validate_connection(conn):
                raise sqlite3.Error("Connection validation failed")

            yield conn

        except Exception as e:
            logger.error("Connection acquisition failed", error=str(e))
            raise
        finally:
            # Return connection to pool
            if conn_id:
                async with self._pool_lock:
                    self._busy_connections.discard(conn_id)

                    if conn_id in self._connection_info:
                        self._connection_info[conn_id].is_busy = False

    async def close_all(self):
        """Close all connections and cleanup"""
        logger.info("Closing all database connections")

        # Stop health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        async with self._pool_lock:
            for conn in self._pool:
                try:
                    await asyncio.to_thread(conn.close)
                except Exception as e:
                    logger.error("Error closing connection", error=str(e))

            self._pool.clear()
            self._connection_info.clear()
            self._busy_connections.clear()

        logger.info("All database connections closed")

    def get_stats(self) -> dict[str, Any]:
        """Get connection pool statistics"""
        active_connections = len(self._pool)
        busy_connections = len(self._busy_connections)

        return {
            "active_connections": active_connections,
            "busy_connections": busy_connections,
            "available_connections": active_connections - busy_connections,
            "total_created": self._stats["total_created"],
            "total_destroyed": self._stats["total_destroyed"],
            "peak_active": self._stats["peak_active"],
            "connection_errors": self._stats["connection_errors"],
            "health_check_failures": self._stats["health_check_failures"],
            "pool_utilization": busy_connections / max(active_connections, 1),
        }


class DatabaseConnectionManager:
    """Manager for multiple database connection pools"""

    def __init__(self):
        self.pools: dict[str, AdvancedConnectionPool] = {}
        self.default_pool_config = {
            "min_connections": 2,
            "max_connections": 10,
            "connection_timeout": 30,
            "health_check_interval": 60,
        }

    async def get_pool(
        self, database_path: str, **pool_config
    ) -> AdvancedConnectionPool:
        """Get or create a connection pool for a database"""
        if database_path not in self.pools:
            config = {**self.default_pool_config, **pool_config}
            pool = AdvancedConnectionPool(database_path, **config)
            await pool.initialize()
            self.pools[database_path] = pool

        return self.pools[database_path]

    @asynccontextmanager
    async def get_connection(self, database_path: str, **pool_config):
        """Get a connection for a specific database"""
        pool = await self.get_pool(database_path, **pool_config)
        async with pool.get_connection() as conn:
            yield conn

    async def close_all_pools(self):
        """Close all connection pools"""
        for pool in self.pools.values():
            await pool.close_all()
        self.pools.clear()

    def get_all_stats(self) -> dict[str, Any]:
        """Get statistics for all connection pools"""
        return {
            database_path: pool.get_stats()
            for database_path, pool in self.pools.items()
        }


# Global connection manager
connection_manager = DatabaseConnectionManager()


async def get_connection_manager() -> DatabaseConnectionManager:
    """Get the global connection manager"""
    return connection_manager


# Convenience function for getting database connections
async def get_database_connection(database_path: str, **pool_config):
    """Get a database connection with automatic pool management"""
    manager = await get_connection_manager()
    return manager.get_connection(database_path, **pool_config)
