"""
Database Index Management System
Automatically manages database indexes for optimal performance
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import structlog

from .connection_manager import get_database_connection
from .database_optimizer import database_optimizer
from .query_optimizer import query_analyzer

logger = structlog.get_logger(__name__)


@dataclass
class IndexInfo:
    """Information about a database index"""

    name: str
    table: str
    columns: list[str]
    unique: bool
    partial: bool
    created_at: datetime | None = None
    usage_count: int = 0
    last_used: datetime | None = None
    size_estimate: int = 0


class DatabaseIndexManager:
    """Manages database indexes for optimal performance"""

    def __init__(self, database_path: str):
        self.database_path = database_path
        self.existing_indexes: dict[str, IndexInfo] = {}
        self.recommended_indexes: list[dict[str, Any]] = []
        self.auto_create_indexes = True

    async def initialize(self):
        """Initialize index manager and scan existing indexes"""
        await self._scan_existing_indexes()
        await self._create_essential_indexes()
        logger.info(
            "Database index manager initialized",
            database=self.database_path,
            existing_indexes=len(self.existing_indexes),
        )

    async def _scan_existing_indexes(self):
        """Scan database for existing indexes"""
        try:
            async with get_database_connection(self.database_path) as conn:
                # Get all indexes
                if hasattr(conn, "execute"):
                    # aiosqlite
                    cursor = await conn.execute(
                        """
                        SELECT name, tbl_name, sql 
                        FROM sqlite_master 
                        WHERE type = 'index' AND name NOT LIKE 'sqlite_%'
                    """
                    )
                    rows = await cursor.fetchall()
                else:
                    # sqlite3
                    cursor = await asyncio.to_thread(
                        conn.execute,
                        """
                        SELECT name, tbl_name, sql 
                        FROM sqlite_master 
                        WHERE type = 'index' AND name NOT LIKE 'sqlite_%'
                    """,
                    )
                    rows = await asyncio.to_thread(cursor.fetchall)

                for row in rows:
                    index_name = row[0]
                    table_name = row[1]
                    sql = row[2] or ""

                    # Parse index information
                    columns = self._parse_index_columns(sql)
                    unique = "UNIQUE" in sql.upper()
                    partial = "WHERE" in sql.upper()

                    self.existing_indexes[index_name] = IndexInfo(
                        name=index_name,
                        table=table_name,
                        columns=columns,
                        unique=unique,
                        partial=partial,
                        created_at=datetime.utcnow(),  # Approximate
                    )

                logger.info(f"Found {len(self.existing_indexes)} existing indexes")

        except Exception as e:
            logger.error("Failed to scan existing indexes", error=str(e))

    def _parse_index_columns(self, sql: str) -> list[str]:
        """Parse column names from CREATE INDEX SQL"""
        if not sql:
            return []

        try:
            # Simple regex to extract columns between parentheses
            import re

            match = re.search(r"\((.*?)\)", sql)
            if match:
                columns_str = match.group(1)
                # Split by comma and clean up
                columns = [
                    col.strip().strip('"').strip("'") for col in columns_str.split(",")
                ]
                return [col for col in columns if col]
            return []
        except Exception:
            return []

    async def _create_essential_indexes(self):
        """Create essential indexes if they don't exist"""
        essential_indexes = [
            {
                "name": "idx_books_id",
                "table": "books",
                "columns": ["id"],
                "unique": True,
                "sql": "CREATE UNIQUE INDEX IF NOT EXISTS idx_books_id ON books(id)",
            },
            {
                "name": "idx_books_hash",
                "table": "books",
                "columns": ["file_hash"],
                "unique": True,
                "sql": "CREATE UNIQUE INDEX IF NOT EXISTS idx_books_hash ON books(file_hash)",
            },
            {
                "name": "idx_books_category",
                "table": "books",
                "columns": ["category"],
                "unique": False,
                "sql": "CREATE INDEX IF NOT EXISTS idx_books_category ON books(category)",
            },
            {
                "name": "idx_chunks_id",
                "table": "chunks",
                "columns": ["id"],
                "unique": True,
                "sql": "CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_id ON chunks(id)",
            },
            {
                "name": "idx_chunks_book_id",
                "table": "chunks",
                "columns": ["book_id"],
                "unique": False,
                "sql": "CREATE INDEX IF NOT EXISTS idx_chunks_book_id ON chunks(book_id)",
            },
            {
                "name": "idx_chunks_index",
                "table": "chunks",
                "columns": ["chunk_index"],
                "unique": False,
                "sql": "CREATE INDEX IF NOT EXISTS idx_chunks_index ON chunks(chunk_index)",
            },
            {
                "name": "idx_chunks_book_index",
                "table": "chunks",
                "columns": ["book_id", "chunk_index"],
                "unique": False,
                "sql": "CREATE INDEX IF NOT EXISTS idx_chunks_book_index ON chunks(book_id, chunk_index)",
            },
            {
                "name": "idx_chunks_created",
                "table": "chunks",
                "columns": ["created_at"],
                "unique": False,
                "sql": "CREATE INDEX IF NOT EXISTS idx_chunks_created ON chunks(created_at)",
            },
        ]

        for index_def in essential_indexes:
            if index_def["name"] not in self.existing_indexes:
                await self._create_index(index_def)

    async def _create_index(self, index_def: dict[str, Any]) -> bool:
        """Create a database index"""
        try:
            async with get_database_connection(self.database_path) as conn:
                async with database_optimizer.track_query(
                    "create_index", index_def["sql"]
                ):
                    if hasattr(conn, "execute"):
                        await conn.execute(index_def["sql"])
                        await conn.commit()
                    else:
                        await asyncio.to_thread(conn.execute, index_def["sql"])
                        await asyncio.to_thread(conn.commit)

                # Add to tracking
                self.existing_indexes[index_def["name"]] = IndexInfo(
                    name=index_def["name"],
                    table=index_def["table"],
                    columns=index_def["columns"],
                    unique=index_def.get("unique", False),
                    partial=False,
                    created_at=datetime.utcnow(),
                )

                logger.info(
                    "Created database index",
                    index_name=index_def["name"],
                    table=index_def["table"],
                    columns=index_def["columns"],
                )
                return True

        except Exception as e:
            logger.error(
                "Failed to create index", index_name=index_def["name"], error=str(e)
            )
            return False

    async def analyze_index_usage(self) -> dict[str, Any]:
        """Analyze index usage patterns"""
        try:
            async with get_database_connection(self.database_path) as conn:
                # Get index usage statistics (SQLite specific)
                usage_stats = {}

                for index_name, index_info in self.existing_indexes.items():
                    try:
                        # Query plan analysis for index usage
                        test_query = f"EXPLAIN QUERY PLAN SELECT * FROM {index_info.table} WHERE {index_info.columns[0]} = ?"

                        if hasattr(conn, "execute"):
                            cursor = await conn.execute(test_query)
                            plan_rows = await cursor.fetchall()
                        else:
                            cursor = await asyncio.to_thread(conn.execute, test_query)
                            plan_rows = await asyncio.to_thread(cursor.fetchall)

                        # Check if index is mentioned in query plan
                        plan_text = " ".join([str(row) for row in plan_rows])
                        uses_index = (
                            index_name in plan_text or "INDEX" in plan_text.upper()
                        )

                        usage_stats[index_name] = {
                            "table": index_info.table,
                            "columns": index_info.columns,
                            "appears_in_plan": uses_index,
                            "created_at": index_info.created_at,
                            "estimated_benefit": "unknown",
                        }

                    except Exception as e:
                        logger.warning(f"Failed to analyze index {index_name}: {e}")
                        usage_stats[index_name] = {
                            "table": index_info.table,
                            "columns": index_info.columns,
                            "error": str(e),
                        }

                return usage_stats

        except Exception as e:
            logger.error("Failed to analyze index usage", error=str(e))
            return {}

    async def get_optimization_recommendations(self) -> list[dict[str, Any]]:
        """Get index optimization recommendations"""
        recommendations = []

        # Get suggestions from query analyzer
        index_suggestions = query_analyzer.get_index_suggestions()

        for suggestion in index_suggestions:
            # Check if index already exists
            index_name = f"idx_{suggestion.table_name}_{'_'.join(suggestion.columns)}"

            if index_name not in self.existing_indexes:
                recommendations.append(
                    {
                        "type": "create_index",
                        "index_name": index_name,
                        "table": suggestion.table_name,
                        "columns": suggestion.columns,
                        "estimated_benefit": suggestion.estimated_benefit,
                        "reason": suggestion.reason,
                        "sql": suggestion.sql,
                        "query_patterns": suggestion.query_patterns,
                    }
                )

        # Check for unused indexes
        for index_name, index_info in self.existing_indexes.items():
            # Skip essential indexes
            if index_name.startswith("idx_") and not index_name.endswith("_id"):
                # Simple heuristic: if index was created more than 7 days ago
                # and we have no usage data, it might be unused
                if (
                    index_info.created_at
                    and datetime.utcnow() - index_info.created_at > timedelta(days=7)
                    and index_info.usage_count == 0
                ):

                    recommendations.append(
                        {
                            "type": "review_unused",
                            "index_name": index_name,
                            "table": index_info.table,
                            "columns": index_info.columns,
                            "reason": "Index may be unused - consider dropping",
                            "created_at": index_info.created_at,
                        }
                    )

        return recommendations

    async def apply_recommendations(self, max_indexes: int = 5) -> dict[str, Any]:
        """Apply optimization recommendations"""
        recommendations = await self.get_optimization_recommendations()
        results = {"created": [], "skipped": [], "errors": []}

        create_recommendations = [
            r
            for r in recommendations
            if r["type"] == "create_index"
            and r["estimated_benefit"] in ["high", "medium"]
        ]

        # Sort by estimated benefit
        create_recommendations.sort(
            key=lambda x: 0 if x["estimated_benefit"] == "high" else 1
        )

        for i, rec in enumerate(create_recommendations[:max_indexes]):
            if self.auto_create_indexes:
                success = await self._create_index(
                    {
                        "name": rec["index_name"],
                        "table": rec["table"],
                        "columns": rec["columns"],
                        "sql": rec["sql"],
                    }
                )

                if success:
                    results["created"].append(rec["index_name"])
                else:
                    results["errors"].append(rec["index_name"])
            else:
                results["skipped"].append(rec["index_name"])

        return results

    async def drop_index(self, index_name: str) -> bool:
        """Drop a database index"""
        try:
            if index_name not in self.existing_indexes:
                logger.warning(f"Index {index_name} does not exist")
                return False

            async with get_database_connection(self.database_path) as conn:
                drop_sql = f"DROP INDEX IF EXISTS {index_name}"

                async with database_optimizer.track_query("drop_index", drop_sql):
                    if hasattr(conn, "execute"):
                        await conn.execute(drop_sql)
                        await conn.commit()
                    else:
                        await asyncio.to_thread(conn.execute, drop_sql)
                        await asyncio.to_thread(conn.commit)

                # Remove from tracking
                del self.existing_indexes[index_name]

                logger.info("Dropped database index", index_name=index_name)
                return True

        except Exception as e:
            logger.error("Failed to drop index", index_name=index_name, error=str(e))
            return False

    async def get_index_statistics(self) -> dict[str, Any]:
        """Get comprehensive index statistics"""
        try:
            async with get_database_connection(self.database_path) as conn:
                # Get database file size
                if hasattr(conn, "execute"):
                    cursor = await conn.execute("PRAGMA page_count")
                    page_count_row = await cursor.fetchone()
                    cursor = await conn.execute("PRAGMA page_size")
                    page_size_row = await cursor.fetchone()
                else:
                    cursor = await asyncio.to_thread(conn.execute, "PRAGMA page_count")
                    page_count_row = await asyncio.to_thread(cursor.fetchone)
                    cursor = await asyncio.to_thread(conn.execute, "PRAGMA page_size")
                    page_size_row = await asyncio.to_thread(cursor.fetchone)

                page_count = page_count_row[0] if page_count_row else 0
                page_size = page_size_row[0] if page_size_row else 4096

                db_size_bytes = page_count * page_size

                return {
                    "total_indexes": len(self.existing_indexes),
                    "database_size_mb": db_size_bytes / (1024 * 1024),
                    "indexes_by_table": self._group_indexes_by_table(),
                    "recent_recommendations": len(
                        await self.get_optimization_recommendations()
                    ),
                    "index_details": {
                        name: {
                            "table": info.table,
                            "columns": info.columns,
                            "unique": info.unique,
                            "created_at": (
                                info.created_at.isoformat() if info.created_at else None
                            ),
                        }
                        for name, info in self.existing_indexes.items()
                    },
                }

        except Exception as e:
            logger.error("Failed to get index statistics", error=str(e))
            return {"error": str(e)}

    def _group_indexes_by_table(self) -> dict[str, int]:
        """Group indexes by table name"""
        table_counts = {}
        for index_info in self.existing_indexes.values():
            table_counts[index_info.table] = table_counts.get(index_info.table, 0) + 1
        return table_counts

    async def maintenance_task(self):
        """Periodic maintenance task for index optimization"""
        try:
            logger.info("Running index maintenance task")

            # Analyze current usage
            usage_stats = await self.analyze_index_usage()

            # Get and apply recommendations
            if self.auto_create_indexes:
                results = await self.apply_recommendations(max_indexes=3)
                logger.info("Applied index recommendations", results=results)

            # Update statistics
            stats = await self.get_index_statistics()
            logger.info(
                "Index maintenance completed",
                total_indexes=stats.get("total_indexes", 0),
                recommendations=stats.get("recent_recommendations", 0),
            )

        except Exception as e:
            logger.error("Index maintenance task failed", error=str(e))


# Global index managers for different databases
index_managers: dict[str, DatabaseIndexManager] = {}


async def get_index_manager(database_path: str) -> DatabaseIndexManager:
    """Get or create index manager for database"""
    if database_path not in index_managers:
        manager = DatabaseIndexManager(database_path)
        await manager.initialize()
        index_managers[database_path] = manager

    return index_managers[database_path]


async def run_index_maintenance():
    """Run maintenance on all managed databases"""
    for manager in index_managers.values():
        await manager.maintenance_task()
