"""
Advanced Query Optimization and Index Management
Provides optimized query patterns, index suggestions, and query analysis
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class IndexSuggestion:
    """Index suggestion with impact analysis"""

    table_name: str
    columns: list[str]
    index_type: str  # 'single', 'composite', 'covering'
    estimated_benefit: str  # 'high', 'medium', 'low'
    reason: str
    sql: str
    query_patterns: list[str]


@dataclass
class QueryAnalysis:
    """Analysis results for a SQL query"""

    query: str
    query_type: str  # SELECT, INSERT, UPDATE, DELETE
    tables_accessed: list[str]
    columns_used: list[str]
    where_columns: list[str]
    order_columns: list[str]
    join_columns: list[str]
    potential_issues: list[str]
    optimization_suggestions: list[str]
    estimated_cost: int  # 1-10 scale


class OptimizedQueries:
    """Collection of optimized query patterns"""

    @staticmethod
    def get_chunk_context_optimized() -> str:
        """Optimized query for getting chunk context in single query"""
        return """
        WITH target_chunk AS (
            SELECT book_id, chunk_index 
            FROM chunks 
            WHERE id = ?
        )
        SELECT c.* 
        FROM chunks c
        JOIN target_chunk tc ON c.book_id = tc.book_id
        WHERE c.chunk_index BETWEEN (tc.chunk_index - ?) AND (tc.chunk_index + ?)
        ORDER BY c.chunk_index
        """

    @staticmethod
    def get_book_chunks_paginated() -> str:
        """Optimized paginated query for book chunks"""
        return """
        SELECT id, text, chunk_index, metadata
        FROM chunks 
        WHERE book_id = ?
        ORDER BY chunk_index
        LIMIT ? OFFSET ?
        """

    @staticmethod
    def search_chunks_with_book_info() -> str:
        """Optimized search query with book information in single query"""
        return """
        SELECT 
            c.id, c.text, c.chunk_index, c.metadata,
            b.title as book_title, b.author, b.category
        FROM chunks c
        JOIN books b ON c.book_id = b.id
        WHERE c.text MATCH ?
        ORDER BY rank
        LIMIT ?
        """

    @staticmethod
    def get_recent_books_with_stats() -> str:
        """Get recent books with chunk statistics"""
        return """
        SELECT 
            b.*,
            COUNT(c.id) as chunk_count,
            AVG(LENGTH(c.text)) as avg_chunk_length,
            MAX(c.created_at) as last_chunk_added
        FROM books b
        LEFT JOIN chunks c ON b.id = c.book_id
        WHERE b.created_at > datetime('now', '-30 days')
        GROUP BY b.id
        ORDER BY b.created_at DESC
        LIMIT ?
        """

    @staticmethod
    def get_search_analytics() -> str:
        """Optimized query for search analytics"""
        return """
        SELECT 
            date(created_at) as search_date,
            COUNT(*) as search_count,
            COUNT(DISTINCT user_id) as unique_users,
            AVG(result_count) as avg_results
        FROM search_logs
        WHERE created_at > datetime('now', '-7 days')
        GROUP BY date(created_at)
        ORDER BY search_date DESC
        """

    @staticmethod
    def batch_insert_chunks() -> str:
        """Optimized batch insert for chunks"""
        return """
        INSERT OR REPLACE INTO chunks 
        (id, book_id, text, chunk_index, metadata, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """

    @staticmethod
    def update_book_statistics() -> str:
        """Update book statistics efficiently"""
        return """
        UPDATE books 
        SET 
            chunk_count = (SELECT COUNT(*) FROM chunks WHERE book_id = books.id),
            total_length = (SELECT SUM(LENGTH(text)) FROM chunks WHERE book_id = books.id),
            last_updated = datetime('now')
        WHERE id = ?
        """


class QueryAnalyzer:
    """Analyzes SQL queries for optimization opportunities"""

    def __init__(self):
        self.table_schemas: dict[str, list[str]] = {}
        self.query_patterns: dict[str, int] = defaultdict(int)
        self.column_usage: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def register_table_schema(self, table_name: str, columns: list[str]):
        """Register table schema for analysis"""
        self.table_schemas[table_name] = columns
        logger.debug(f"Registered schema for table {table_name}")

    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze a SQL query and provide optimization suggestions"""
        query = query.strip()
        query_upper = query.upper()

        # Determine query type
        query_type = self._get_query_type(query_upper)

        # Extract components
        tables_accessed = self._extract_tables(query_upper)
        columns_used = self._extract_columns(query, tables_accessed)
        where_columns = self._extract_where_columns(query_upper)
        order_columns = self._extract_order_columns(query_upper)
        join_columns = self._extract_join_columns(query_upper)

        # Identify potential issues
        potential_issues = self._identify_issues(query, query_upper)

        # Generate optimization suggestions
        optimization_suggestions = self._generate_suggestions(
            query, query_upper, tables_accessed, where_columns, order_columns
        )

        # Estimate query cost
        estimated_cost = self._estimate_cost(
            query_upper, tables_accessed, where_columns, join_columns
        )

        # Track patterns
        pattern = self._extract_pattern(query_upper)
        self.query_patterns[pattern] += 1

        # Track column usage
        for table in tables_accessed:
            for column in where_columns + order_columns:
                self.column_usage[table][column] += 1

        return QueryAnalysis(
            query=query,
            query_type=query_type,
            tables_accessed=tables_accessed,
            columns_used=columns_used,
            where_columns=where_columns,
            order_columns=order_columns,
            join_columns=join_columns,
            potential_issues=potential_issues,
            optimization_suggestions=optimization_suggestions,
            estimated_cost=estimated_cost,
        )

    def _get_query_type(self, query_upper: str) -> str:
        """Determine the type of SQL query"""
        if query_upper.startswith("SELECT"):
            return "SELECT"
        elif query_upper.startswith("INSERT"):
            return "INSERT"
        elif query_upper.startswith("UPDATE"):
            return "UPDATE"
        elif query_upper.startswith("DELETE"):
            return "DELETE"
        elif query_upper.startswith("CREATE"):
            return "CREATE"
        else:
            return "OTHER"

    def _extract_tables(self, query_upper: str) -> list[str]:
        """Extract table names from query"""
        tables = []

        # FROM clause
        from_match = re.search(r"FROM\s+(\w+)", query_upper)
        if from_match:
            tables.append(from_match.group(1).lower())

        # JOIN clauses
        join_matches = re.findall(r"JOIN\s+(\w+)", query_upper)
        for match in join_matches:
            tables.append(match.lower())

        # UPDATE table
        update_match = re.search(r"UPDATE\s+(\w+)", query_upper)
        if update_match:
            tables.append(update_match.group(1).lower())

        # INSERT INTO table
        insert_match = re.search(r"INSERT\s+(?:OR\s+\w+\s+)?INTO\s+(\w+)", query_upper)
        if insert_match:
            tables.append(insert_match.group(1).lower())

        return list(set(tables))

    def _extract_columns(self, query: str, tables: list[str]) -> list[str]:
        """Extract column names from query"""
        columns = []

        # Simple extraction - could be enhanced
        # This is a basic implementation
        for table in tables:
            if table in self.table_schemas:
                for column in self.table_schemas[table]:
                    if column in query:
                        columns.append(column)

        return list(set(columns))

    def _extract_where_columns(self, query_upper: str) -> list[str]:
        """Extract columns used in WHERE clauses"""
        columns = []

        # Find WHERE clause
        where_match = re.search(
            r"WHERE\s+(.*?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT|$)",
            query_upper,
            re.DOTALL,
        )
        if where_match:
            where_clause = where_match.group(1)

            # Extract column names (simplified)
            column_matches = re.findall(r"(\w+)\s*[=<>!]", where_clause)
            columns.extend([col.lower() for col in column_matches])

        return list(set(columns))

    def _extract_order_columns(self, query_upper: str) -> list[str]:
        """Extract columns used in ORDER BY"""
        columns = []

        order_match = re.search(r"ORDER\s+BY\s+(.*?)(?:\s+LIMIT|$)", query_upper)
        if order_match:
            order_clause = order_match.group(1)
            column_matches = re.findall(r"(\w+)", order_clause)
            columns.extend(
                [
                    col.lower()
                    for col in column_matches
                    if col.upper() not in ["ASC", "DESC"]
                ]
            )

        return list(set(columns))

    def _extract_join_columns(self, query_upper: str) -> list[str]:
        """Extract columns used in JOIN conditions"""
        columns = []

        join_matches = re.findall(
            r"JOIN\s+\w+\s+(?:\w+\s+)?ON\s+(.*?)(?:\s+WHERE|\s+GROUP|\s+ORDER|\s+JOIN|$)",
            query_upper,
        )
        for join_condition in join_matches:
            column_matches = re.findall(r"(\w+\.\w+|\w+)", join_condition)
            columns.extend([col.lower() for col in column_matches])

        return list(set(columns))

    def _identify_issues(self, query: str, query_upper: str) -> list[str]:
        """Identify potential performance issues"""
        issues = []

        # Check for SELECT *
        if "SELECT *" in query_upper:
            issues.append("Using SELECT * may retrieve unnecessary columns")

        # Check for leading wildcard in LIKE
        if re.search(r"LIKE\s+['\"]%", query_upper):
            issues.append("Leading wildcard in LIKE prevents index usage")

        # Check for large LIMIT
        limit_match = re.search(r"LIMIT\s+(\d+)", query_upper)
        if limit_match and int(limit_match.group(1)) > 1000:
            issues.append("Large LIMIT value may impact performance")

        # Check for missing WHERE in UPDATE/DELETE
        if query_upper.startswith(("UPDATE", "DELETE")) and "WHERE" not in query_upper:
            issues.append("UPDATE/DELETE without WHERE clause affects all rows")

        # Check for OR in WHERE clause
        if " OR " in query_upper:
            issues.append("OR conditions may prevent index usage")

        # Check for functions in WHERE clause
        if re.search(r"WHERE\s+.*?\w+\(", query_upper):
            issues.append("Functions in WHERE clause may prevent index usage")

        return issues

    def _generate_suggestions(
        self,
        query: str,
        query_upper: str,
        tables: list[str],
        where_columns: list[str],
        order_columns: list[str],
    ) -> list[str]:
        """Generate optimization suggestions"""
        suggestions = []

        # Suggest specific columns instead of SELECT *
        if "SELECT *" in query_upper:
            suggestions.append("Replace SELECT * with specific column names")

        # Suggest indexes for WHERE columns
        if where_columns:
            suggestions.append(
                f"Consider indexes on WHERE columns: {', '.join(where_columns)}"
            )

        # Suggest indexes for ORDER BY columns
        if order_columns:
            suggestions.append(
                f"Consider indexes on ORDER BY columns: {', '.join(order_columns)}"
            )

        # Suggest LIMIT for large result sets
        if query_upper.startswith("SELECT") and "LIMIT" not in query_upper:
            suggestions.append("Consider adding LIMIT clause for large result sets")

        # Suggest prepared statements for repeated queries
        if "?" not in query and any(op in query for op in ["=", "<", ">", "LIKE"]):
            suggestions.append(
                "Use parameterized queries for better performance and security"
            )

        return suggestions

    def _estimate_cost(
        self,
        query_upper: str,
        tables: list[str],
        where_columns: list[str],
        join_columns: list[str],
    ) -> int:
        """Estimate query cost on scale of 1-10"""
        cost = 1

        # Base cost by query type
        if query_upper.startswith("SELECT"):
            cost += 1
        elif query_upper.startswith(("INSERT", "UPDATE", "DELETE")):
            cost += 2

        # Cost for multiple tables
        cost += len(tables)

        # Cost for JOINs
        join_count = query_upper.count("JOIN")
        cost += join_count * 2

        # Cost for complex WHERE clauses
        if "WHERE" in query_upper:
            cost += 1
            if " OR " in query_upper:
                cost += 2
            if "LIKE" in query_upper:
                cost += 1

        # Cost for sorting
        if "ORDER BY" in query_upper:
            cost += 2

        # Cost for grouping
        if "GROUP BY" in query_upper:
            cost += 2

        # Cost for subqueries
        subquery_count = query_upper.count("SELECT") - 1
        cost += subquery_count * 3

        return min(cost, 10)  # Cap at 10

    def _extract_pattern(self, query_upper: str) -> str:
        """Extract query pattern for tracking"""
        # Simplified pattern extraction
        pattern = re.sub(r"\b\d+\b", "N", query_upper)  # Replace numbers
        pattern = re.sub(r"'[^']*'", "'X'", pattern)  # Replace string literals
        pattern = re.sub(r"\s+", " ", pattern).strip()  # Normalize whitespace
        return pattern[:100]  # Truncate for storage

    def get_index_suggestions(self) -> list[IndexSuggestion]:
        """Generate index suggestions based on query patterns"""
        suggestions = []

        for table, column_counts in self.column_usage.items():
            # Sort columns by usage frequency
            frequent_columns = sorted(
                column_counts.items(), key=lambda x: x[1], reverse=True
            )

            # Single column indexes for most used columns
            for column, count in frequent_columns[:3]:  # Top 3 columns
                if count >= 5:  # Minimum usage threshold
                    suggestions.append(
                        IndexSuggestion(
                            table_name=table,
                            columns=[column],
                            index_type="single",
                            estimated_benefit="high" if count >= 20 else "medium",
                            reason=f"Column used in {count} queries",
                            sql=f"CREATE INDEX IF NOT EXISTS idx_{table}_{column} ON {table}({column})",
                            query_patterns=[f"WHERE {column} = ?"],
                        )
                    )

            # Composite indexes for frequently used column combinations
            if len(frequent_columns) >= 2:
                top_columns = [
                    col for col, count in frequent_columns[:3] if count >= 10
                ]
                if len(top_columns) >= 2:
                    composite_columns = top_columns[:2]  # Max 2 columns for composite
                    suggestions.append(
                        IndexSuggestion(
                            table_name=table,
                            columns=composite_columns,
                            index_type="composite",
                            estimated_benefit="high",
                            reason="Columns frequently used together",
                            sql=f"CREATE INDEX IF NOT EXISTS idx_{table}_{'_'.join(composite_columns)} ON {table}({', '.join(composite_columns)})",
                            query_patterns=[
                                f"WHERE {' AND '.join([f'{col} = ?' for col in composite_columns])}"
                            ],
                        )
                    )

        return suggestions

    def get_optimization_report(self) -> dict[str, Any]:
        """Generate comprehensive optimization report"""
        total_queries = sum(self.query_patterns.values())

        # Most common query patterns
        common_patterns = sorted(
            self.query_patterns.items(), key=lambda x: x[1], reverse=True
        )[:10]

        # Most accessed columns
        all_column_usage = {}
        for table, columns in self.column_usage.items():
            for column, count in columns.items():
                key = f"{table}.{column}"
                all_column_usage[key] = count

        frequent_columns = sorted(
            all_column_usage.items(), key=lambda x: x[1], reverse=True
        )[:10]

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "total_queries_analyzed": total_queries,
            "unique_patterns": len(self.query_patterns),
            "tables_analyzed": len(self.column_usage),
            "common_patterns": [
                {"pattern": pattern, "count": count}
                for pattern, count in common_patterns
            ],
            "frequent_columns": [
                {"column": column, "usage_count": count}
                for column, count in frequent_columns
            ],
            "index_suggestions": [
                {
                    "table": suggestion.table_name,
                    "columns": suggestion.columns,
                    "type": suggestion.index_type,
                    "benefit": suggestion.estimated_benefit,
                    "sql": suggestion.sql,
                }
                for suggestion in self.get_index_suggestions()
            ],
        }


# Global query analyzer
query_analyzer = QueryAnalyzer()


# Initialize table schemas for known tables
def initialize_schemas():
    """Initialize known table schemas"""
    query_analyzer.register_table_schema(
        "books",
        [
            "id",
            "title",
            "author",
            "category",
            "file_path",
            "file_hash",
            "created_at",
            "updated_at",
            "chunk_count",
            "total_length",
        ],
    )

    query_analyzer.register_table_schema(
        "chunks", ["id", "book_id", "text", "chunk_index", "metadata", "created_at"]
    )

    query_analyzer.register_table_schema(
        "users",
        ["id", "email", "username", "password_hash", "created_at", "last_login"],
    )


# Initialize on import
initialize_schemas()


async def get_query_analyzer() -> QueryAnalyzer:
    """Get the global query analyzer"""
    return query_analyzer
