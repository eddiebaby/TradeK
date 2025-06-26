"""
Comprehensive tests for database optimization system
Tests query optimization, connection management, and index management
"""

import pytest
import asyncio
import sqlite3
import tempfile
import os
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timedelta
from pathlib import Path

from src.core.database_optimizer import (
    DatabaseOptimizer, QueryResultCache, QueryMetrics,
    database_optimizer, optimize_query
)
from src.core.connection_manager import (
    AdvancedConnectionPool, DatabaseConnectionManager, 
    connection_manager, get_database_connection
)
from src.core.query_optimizer import (
    QueryAnalyzer, OptimizedQueries, IndexSuggestion,
    query_analyzer
)
from src.core.index_manager import (
    DatabaseIndexManager, IndexInfo, get_index_manager
)


class TestQueryMetrics:
    """Test query performance metrics"""
    
    def test_metrics_initialization(self):
        """Test metrics initialization"""
        metrics = QueryMetrics("test_query")
        
        assert metrics.query_type == "test_query"
        assert metrics.execution_count == 0
        assert metrics.total_time == 0.0
        assert metrics.error_count == 0
    
    def test_add_execution_success(self):
        """Test adding successful execution"""
        metrics = QueryMetrics("test_query")
        
        metrics.add_execution(1.5, success=True)
        
        assert metrics.execution_count == 1
        assert metrics.total_time == 1.5
        assert metrics.avg_time == 1.5
        assert metrics.min_time == 1.5
        assert metrics.max_time == 1.5
        assert metrics.error_count == 0
        assert len(metrics.recent_times) == 1
    
    def test_add_execution_failure(self):
        """Test adding failed execution"""
        metrics = QueryMetrics("test_query")
        
        metrics.add_execution(2.0, success=False)
        
        assert metrics.execution_count == 1
        assert metrics.error_count == 1
        assert metrics.total_time == 2.0
    
    def test_multiple_executions(self):
        """Test multiple execution tracking"""
        metrics = QueryMetrics("test_query")
        
        execution_times = [1.0, 2.0, 1.5, 3.0]
        for time_val in execution_times:
            metrics.add_execution(time_val)
        
        assert metrics.execution_count == 4
        assert metrics.total_time == 7.5
        assert metrics.avg_time == 1.875
        assert metrics.min_time == 1.0
        assert metrics.max_time == 3.0
    
    def test_percentiles_calculation(self):
        """Test percentile calculations"""
        metrics = QueryMetrics("test_query")
        
        # Add 100 values for percentile testing
        for i in range(100):
            metrics.add_execution(i / 10.0)  # 0.0 to 9.9
        
        percentiles = metrics.get_percentiles()
        
        assert 'p50' in percentiles
        assert 'p95' in percentiles
        assert 'p99' in percentiles
        assert percentiles['p50'] < percentiles['p95'] < percentiles['p99']


class TestQueryResultCache:
    """Test query result caching"""
    
    @pytest.fixture
    async def cache(self):
        """Create test cache"""
        cache = QueryResultCache()
        await cache.initialize()
        return cache
    
    @pytest.mark.asyncio
    async def test_cache_miss_then_hit(self, cache):
        """Test cache miss followed by hit"""
        query = "SELECT * FROM test WHERE id = ?"
        params = (1,)
        
        # First access should be a miss
        result = await cache.get(query, params)
        assert result is None
        
        # Set value
        test_data = [{'id': 1, 'name': 'test'}]
        await cache.set(query, params, test_data)
        
        # Second access should be a hit
        result = await cache.get(query, params)
        assert result == test_data
    
    @pytest.mark.asyncio
    async def test_cache_key_generation(self, cache):
        """Test cache key generation"""
        query1 = "SELECT * FROM test WHERE id = ?"
        query2 = "SELECT * FROM test WHERE name = ?"
        params1 = (1,)
        params2 = (1,)
        
        key1 = cache._generate_cache_key(query1, params1)
        key2 = cache._generate_cache_key(query2, params2)
        key3 = cache._generate_cache_key(query1, params1)
        
        # Same query + params should generate same key
        assert key1 == key3
        
        # Different queries should generate different keys
        assert key1 != key2
    
    @pytest.mark.asyncio
    async def test_cache_stats(self, cache):
        """Test cache statistics tracking"""
        query = "SELECT * FROM test WHERE id = ?"
        
        # Generate some hits and misses
        await cache.get(query, (1,))  # Miss
        await cache.set(query, (1,), ['data'])
        await cache.get(query, (1,))  # Hit
        await cache.get(query, (2,))  # Miss
        
        stats = cache.get_stats()
        
        assert stats['total_requests'] == 3
        assert stats['total_hits'] == 1
        assert stats['total_misses'] == 2
        assert stats['hit_rate'] == 1/3
    
    @pytest.mark.asyncio
    async def test_cache_invalidation(self, cache):
        """Test cache invalidation"""
        query = "SELECT * FROM test WHERE id = ?"
        params = (1,)
        
        # Set value
        await cache.set(query, params, ['data'])
        
        # Verify it's cached
        result = await cache.get(query, params)
        assert result == ['data']
        
        # Invalidate
        await cache.invalidate_pattern("*")
        
        # Should be gone
        result = await cache.get(query, params)
        assert result is None


class TestDatabaseOptimizer:
    """Test database optimizer"""
    
    @pytest.fixture
    async def optimizer(self):
        """Create test optimizer"""
        optimizer = DatabaseOptimizer()
        await optimizer.initialize()
        return optimizer
    
    @pytest.mark.asyncio
    async def test_query_tracking(self, optimizer):
        """Test query execution tracking"""
        query_type = "test_query"
        query = "SELECT * FROM test"
        
        # Track a query execution
        async with optimizer.track_query(query_type, query):
            await asyncio.sleep(0.1)  # Simulate query execution
        
        # Check metrics were recorded
        metrics = optimizer.query_metrics[query_type]
        assert metrics.execution_count == 1
        assert metrics.avg_time > 0.05  # Should be around 0.1 seconds
    
    @pytest.mark.asyncio
    async def test_slow_query_detection(self, optimizer):
        """Test slow query detection and suggestions"""
        optimizer.slow_query_threshold = 0.05  # 50ms threshold
        
        query_type = "slow_query"
        slow_query = "SELECT * FROM large_table WHERE unindexed_column LIKE '%pattern%'"
        
        # Execute slow query
        async with optimizer.track_query(query_type, slow_query):
            await asyncio.sleep(0.1)  # Simulate slow execution
        
        # Check optimization suggestions were generated
        assert len(optimizer.optimization_suggestions) > 0
        
        suggestion = optimizer.optimization_suggestions[-1]
        assert suggestion['query_type'] == query_type
        assert suggestion['execution_time'] > optimizer.slow_query_threshold
    
    @pytest.mark.asyncio
    async def test_cached_query_execution(self, optimizer):
        """Test cached query execution"""
        # Mock connection
        mock_conn = Mock()
        mock_conn.execute = Mock(return_value=Mock())
        mock_conn.execute.return_value.fetchall = Mock(return_value=[{'id': 1}])
        
        query = "SELECT * FROM test WHERE id = ?"
        params = (1,)
        
        # First execution should query database
        result1 = await optimizer.execute_cached_query(
            mock_conn, query, params, cache_ttl=300, query_type="test"
        )
        
        # Second execution should use cache
        result2 = await optimizer.execute_cached_query(
            mock_conn, query, params, cache_ttl=300, query_type="test"
        )
        
        assert result1 == result2
        # Database should only be called once
        mock_conn.execute.assert_called_once()
    
    def test_optimization_suggestions(self, optimizer):
        """Test optimization suggestion generation"""
        # Test SELECT * suggestion
        optimizer._suggest_optimization("test", "SELECT * FROM table WHERE col = 1", 2.0)
        
        suggestions = [s for s in optimizer.optimization_suggestions if s['type'] == 'select_star']
        assert len(suggestions) > 0
        
        # Test leading wildcard suggestion
        optimizer._suggest_optimization("test", "SELECT col FROM table WHERE name LIKE '%pattern'", 1.5)
        
        suggestions = [s for s in optimizer.optimization_suggestions if s['type'] == 'leading_wildcard']
        assert len(suggestions) > 0
    
    def test_performance_report(self, optimizer):
        """Test performance report generation"""
        # Add some metrics
        optimizer.query_metrics['test1'].add_execution(1.0)
        optimizer.query_metrics['test1'].add_execution(2.0)
        optimizer.query_metrics['test2'].add_execution(0.5)
        
        report = optimizer.get_performance_report()
        
        assert 'generated_at' in report
        assert 'query_performance' in report
        assert 'slow_queries' in report
        assert 'cache_performance' in report
        
        assert 'test1' in report['query_performance']
        assert 'test2' in report['query_performance']


class TestAdvancedConnectionPool:
    """Test advanced connection pool"""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database"""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        os.unlink(path)
    
    @pytest.mark.asyncio
    async def test_pool_initialization(self, temp_db):
        """Test connection pool initialization"""
        pool = AdvancedConnectionPool(
            database_path=temp_db,
            min_connections=2,
            max_connections=5
        )
        
        await pool.initialize()
        
        try:
            stats = pool.get_stats()
            assert stats['active_connections'] >= 2  # Minimum connections created
            assert stats['total_created'] >= 2
        finally:
            await pool.close_all()
    
    @pytest.mark.asyncio
    async def test_connection_acquisition(self, temp_db):
        """Test getting connections from pool"""
        pool = AdvancedConnectionPool(
            database_path=temp_db,
            min_connections=1,
            max_connections=3
        )
        
        await pool.initialize()
        
        try:
            # Get connection
            async with pool.get_connection() as conn:
                assert conn is not None
                
                # Execute simple query to verify connection works
                await asyncio.to_thread(conn.execute, "SELECT 1")
            
            stats = pool.get_stats()
            assert stats['active_connections'] >= 1
        finally:
            await pool.close_all()
    
    @pytest.mark.asyncio
    async def test_concurrent_connections(self, temp_db):
        """Test concurrent connection usage"""
        pool = AdvancedConnectionPool(
            database_path=temp_db,
            min_connections=1,
            max_connections=3
        )
        
        await pool.initialize()
        
        try:
            async def use_connection(conn_id):
                async with pool.get_connection() as conn:
                    await asyncio.to_thread(conn.execute, f"CREATE TABLE IF NOT EXISTS test{conn_id} (id INTEGER)")
                    return conn_id
            
            # Use multiple connections concurrently
            tasks = [use_connection(i) for i in range(3)]
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 3
            assert results == [0, 1, 2]
            
            stats = pool.get_stats()
            assert stats['active_connections'] <= 3  # Shouldn't exceed max
        finally:
            await pool.close_all()
    
    @pytest.mark.asyncio
    async def test_connection_health_check(self, temp_db):
        """Test connection health monitoring"""
        pool = AdvancedConnectionPool(
            database_path=temp_db,
            min_connections=1,
            max_connections=2,
            health_check_interval=1  # 1 second for testing
        )
        
        await pool.initialize()
        
        try:
            # Wait for health check to run
            await asyncio.sleep(1.5)
            
            stats = pool.get_stats()
            # Health check should have run without errors
            assert stats['health_check_failures'] == 0
        finally:
            await pool.close_all()
    
    @pytest.mark.asyncio
    async def test_pool_exhaustion_handling(self, temp_db):
        """Test behavior when pool is exhausted"""
        pool = AdvancedConnectionPool(
            database_path=temp_db,
            min_connections=1,
            max_connections=1  # Very small pool
        )
        
        await pool.initialize()
        
        try:
            connections = []
            
            # Get the only connection
            conn_context = pool.get_connection()
            conn = await conn_context.__aenter__()
            connections.append((conn_context, conn))
            
            # Try to get another - should timeout quickly for testing
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(0.1):  # Very short timeout
                    async with pool.get_connection():
                        pass
            
            # Release connection
            await connections[0][0].__aexit__(None, None, None)
            
        finally:
            await pool.close_all()


class TestQueryAnalyzer:
    """Test query analyzer"""
    
    @pytest.fixture
    def analyzer(self):
        """Create test analyzer"""
        analyzer = QueryAnalyzer()
        analyzer.register_table_schema('users', ['id', 'name', 'email', 'created_at'])
        analyzer.register_table_schema('posts', ['id', 'user_id', 'title', 'content'])
        return analyzer
    
    def test_query_type_detection(self, analyzer):
        """Test query type detection"""
        select_query = "SELECT * FROM users WHERE id = 1"
        insert_query = "INSERT INTO users (name, email) VALUES ('test', 'test@example.com')"
        update_query = "UPDATE users SET name = 'updated' WHERE id = 1"
        delete_query = "DELETE FROM users WHERE id = 1"
        
        assert analyzer._get_query_type(select_query.upper()) == 'SELECT'
        assert analyzer._get_query_type(insert_query.upper()) == 'INSERT'
        assert analyzer._get_query_type(update_query.upper()) == 'UPDATE'
        assert analyzer._get_query_type(delete_query.upper()) == 'DELETE'
    
    def test_table_extraction(self, analyzer):
        """Test table name extraction"""
        queries = [
            ("SELECT * FROM users", ['users']),
            ("SELECT u.*, p.title FROM users u JOIN posts p ON u.id = p.user_id", ['users', 'posts']),
            ("UPDATE users SET name = 'test'", ['users']),
            ("INSERT INTO posts (title) VALUES ('test')", ['posts'])
        ]
        
        for query, expected_tables in queries:
            tables = analyzer._extract_tables(query.upper())
            assert set(tables) == set(expected_tables)
    
    def test_where_column_extraction(self, analyzer):
        """Test WHERE clause column extraction"""
        query = "SELECT * FROM users WHERE id = 1 AND name = 'test' AND email LIKE '%@example.com'"
        columns = analyzer._extract_where_columns(query.upper())
        
        expected_columns = ['id', 'name', 'email']
        assert set(columns) == set(expected_columns)
    
    def test_order_column_extraction(self, analyzer):
        """Test ORDER BY column extraction"""
        query = "SELECT * FROM users ORDER BY created_at DESC, name ASC"
        columns = analyzer._extract_order_columns(query.upper())
        
        expected_columns = ['created_at', 'name']
        assert set(columns) == set(expected_columns)
    
    def test_issue_identification(self, analyzer):
        """Test issue identification"""
        problematic_queries = [
            ("SELECT * FROM users", ["Using SELECT * may retrieve unnecessary columns"]),
            ("SELECT * FROM users WHERE name LIKE '%pattern'", ["Leading wildcard in LIKE prevents index usage"]),
            ("SELECT * FROM users LIMIT 5000", ["Large LIMIT value may impact performance"]),
            ("UPDATE users SET name = 'test'", ["UPDATE/DELETE without WHERE clause affects all rows"])
        ]
        
        for query, expected_issues in problematic_queries:
            issues = analyzer._identify_issues(query, query.upper())
            for expected_issue in expected_issues:
                assert any(expected_issue in issue for issue in issues)
    
    def test_cost_estimation(self, analyzer):
        """Test query cost estimation"""
        simple_query = "SELECT id FROM users WHERE id = 1"
        complex_query = """
            SELECT u.*, COUNT(p.id) as post_count
            FROM users u
            LEFT JOIN posts p ON u.id = p.user_id
            WHERE u.created_at > '2023-01-01'
            GROUP BY u.id
            ORDER BY post_count DESC
        """
        
        simple_cost = analyzer._estimate_cost(simple_query.upper(), ['users'], ['id'], [])
        complex_cost = analyzer._estimate_cost(complex_query.upper(), ['users', 'posts'], ['created_at'], ['u.id', 'p.user_id'])
        
        assert simple_cost < complex_cost
        assert 1 <= simple_cost <= 10
        assert 1 <= complex_cost <= 10
    
    def test_full_query_analysis(self, analyzer):
        """Test complete query analysis"""
        query = "SELECT u.name, COUNT(p.id) FROM users u JOIN posts p ON u.id = p.user_id WHERE u.created_at > '2023-01-01' GROUP BY u.id ORDER BY COUNT(p.id) DESC"
        
        analysis = analyzer.analyze_query(query)
        
        assert analysis.query_type == 'SELECT'
        assert 'users' in analysis.tables_accessed
        assert 'posts' in analysis.tables_accessed
        assert 'created_at' in analysis.where_columns
        assert analysis.estimated_cost > 1
        assert len(analysis.optimization_suggestions) > 0


class TestIndexManager:
    """Test database index manager"""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database with tables"""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        
        # Create tables
        conn = sqlite3.connect(path)
        conn.execute("""
            CREATE TABLE books (
                id TEXT PRIMARY KEY,
                title TEXT,
                author TEXT,
                category TEXT,
                created_at TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE chunks (
                id TEXT PRIMARY KEY,
                book_id TEXT,
                text TEXT,
                chunk_index INTEGER,
                created_at TIMESTAMP,
                FOREIGN KEY (book_id) REFERENCES books(id)
            )
        """)
        conn.commit()
        conn.close()
        
        yield path
        os.unlink(path)
    
    @pytest.mark.asyncio
    async def test_index_manager_initialization(self, temp_db):
        """Test index manager initialization"""
        manager = DatabaseIndexManager(temp_db)
        await manager.initialize()
        
        # Should have created essential indexes
        assert len(manager.existing_indexes) > 0
        
        # Check for specific essential indexes
        index_names = list(manager.existing_indexes.keys())
        assert any('books' in name for name in index_names)
        assert any('chunks' in name for name in index_names)
    
    @pytest.mark.asyncio
    async def test_index_creation(self, temp_db):
        """Test manual index creation"""
        manager = DatabaseIndexManager(temp_db)
        await manager.initialize()
        
        initial_count = len(manager.existing_indexes)
        
        # Create custom index
        index_def = {
            'name': 'idx_test_custom',
            'table': 'books',
            'columns': ['author'],
            'sql': 'CREATE INDEX IF NOT EXISTS idx_test_custom ON books(author)'
        }
        
        success = await manager._create_index(index_def)
        assert success
        assert len(manager.existing_indexes) == initial_count + 1
        assert 'idx_test_custom' in manager.existing_indexes
    
    @pytest.mark.asyncio
    async def test_index_analysis(self, temp_db):
        """Test index usage analysis"""
        manager = DatabaseIndexManager(temp_db)
        await manager.initialize()
        
        usage_stats = await manager.analyze_index_usage()
        
        assert isinstance(usage_stats, dict)
        assert len(usage_stats) > 0
        
        # Each index should have analysis data
        for index_name, stats in usage_stats.items():
            assert 'table' in stats
            assert 'columns' in stats
    
    @pytest.mark.asyncio
    async def test_optimization_recommendations(self, temp_db):
        """Test getting optimization recommendations"""
        manager = DatabaseIndexManager(temp_db)
        await manager.initialize()
        
        # Simulate some query patterns
        query_analyzer.column_usage['books']['category'] = 10
        query_analyzer.column_usage['chunks']['book_id'] = 15
        
        recommendations = await manager.get_optimization_recommendations()
        
        assert isinstance(recommendations, list)
        # Should have recommendations based on simulated usage
        create_recs = [r for r in recommendations if r['type'] == 'create_index']
        assert len(create_recs) > 0
    
    @pytest.mark.asyncio
    async def test_index_statistics(self, temp_db):
        """Test index statistics generation"""
        manager = DatabaseIndexManager(temp_db)
        await manager.initialize()
        
        stats = await manager.get_index_statistics()
        
        assert 'total_indexes' in stats
        assert 'database_size_mb' in stats
        assert 'indexes_by_table' in stats
        assert 'index_details' in stats
        assert stats['total_indexes'] > 0
    
    @pytest.mark.asyncio
    async def test_index_dropping(self, temp_db):
        """Test index removal"""
        manager = DatabaseIndexManager(temp_db)
        await manager.initialize()
        
        # Create an index to drop
        index_def = {
            'name': 'idx_test_drop',
            'table': 'books',
            'columns': ['title'],
            'sql': 'CREATE INDEX IF NOT EXISTS idx_test_drop ON books(title)'
        }
        
        await manager._create_index(index_def)
        assert 'idx_test_drop' in manager.existing_indexes
        
        # Drop the index
        success = await manager.drop_index('idx_test_drop')
        assert success
        assert 'idx_test_drop' not in manager.existing_indexes


class TestOptimizedQueries:
    """Test optimized query patterns"""
    
    def test_chunk_context_query(self):
        """Test optimized chunk context query"""
        query = OptimizedQueries.get_chunk_context_optimized()
        
        assert isinstance(query, str)
        assert 'WITH target_chunk' in query
        assert 'JOIN target_chunk' in query
        assert 'ORDER BY' in query
        
        # Should be a single query (no multiple SELECT statements at top level)
        assert query.count('SELECT') == 2  # One in CTE, one in main query
    
    def test_search_query_optimization(self):
        """Test optimized search query"""
        query = OptimizedQueries.search_chunks_with_book_info()
        
        assert isinstance(query, str)
        assert 'JOIN books' in query
        assert 'MATCH' in query
        assert 'rank' in query.lower()
        
        # Should combine chunk and book data in single query
        assert query.count('SELECT') == 1
    
    def test_batch_operations(self):
        """Test batch operation queries"""
        insert_query = OptimizedQueries.batch_insert_chunks()
        update_query = OptimizedQueries.update_book_statistics()
        
        assert 'INSERT OR REPLACE' in insert_query
        assert 'UPDATE' in update_query
        assert '?' in insert_query  # Parameterized
        assert '?' in update_query  # Parameterized


class TestIntegration:
    """Integration tests for the complete optimization system"""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database"""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        os.unlink(path)
    
    @pytest.mark.asyncio
    async def test_end_to_end_optimization(self, temp_db):
        """Test complete optimization workflow"""
        # Initialize all components
        optimizer = DatabaseOptimizer()
        await optimizer.initialize()
        
        index_manager = DatabaseIndexManager(temp_db)
        await index_manager.initialize()
        
        # Simulate some database operations
        async with get_database_connection(temp_db) as conn:
            # Create test table
            await asyncio.to_thread(conn.execute, """
                CREATE TABLE IF NOT EXISTS test_table (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    category TEXT,
                    created_at TIMESTAMP
                )
            """)
            await asyncio.to_thread(conn.commit)
            
            # Simulate queries with tracking
            async with optimizer.track_query('test_select', 'SELECT * FROM test_table WHERE category = ?'):
                await asyncio.to_thread(conn.execute, 'SELECT * FROM test_table WHERE category = ?', ('test',))
        
        # Generate performance report
        report = optimizer.get_performance_report()
        assert 'test_select' in report['query_performance']
        
        # Get index recommendations
        recommendations = await index_manager.get_optimization_recommendations()
        assert isinstance(recommendations, list)
        
        # Get statistics
        stats = await index_manager.get_index_statistics()
        assert stats['total_indexes'] >= 0
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_performance_under_load(self, temp_db):
        """Test system performance under concurrent load"""
        import time
        
        # Initialize connection pool
        pool = AdvancedConnectionPool(temp_db, min_connections=5, max_connections=10)
        await pool.initialize()
        
        try:
            async def simulate_query_load():
                """Simulate database query load"""
                for i in range(10):
                    async with pool.get_connection() as conn:
                        await asyncio.to_thread(conn.execute, 'SELECT ?', (i,))
                        await asyncio.sleep(0.01)  # Small delay
            
            # Run concurrent load
            start_time = time.time()
            tasks = [simulate_query_load() for _ in range(5)]
            await asyncio.gather(*tasks)
            end_time = time.time()
            
            # Should complete within reasonable time
            assert end_time - start_time < 5.0
            
            # Check pool stats
            stats = pool.get_stats()
            assert stats['total_created'] > 0
            assert stats['connection_errors'] == 0
            
        finally:
            await pool.close_all()
    
    @pytest.mark.asyncio
    async def test_optimization_decorator(self, temp_db):
        """Test the optimization decorator"""
        @optimize_query('decorated_query', cache_ttl=60)
        async def test_query_function(connection, query, params):
            """Test function with optimization decorator"""
            cursor = await asyncio.to_thread(connection.execute, query, params)
            return await asyncio.to_thread(cursor.fetchall)
        
        # Initialize optimizer
        optimizer = DatabaseOptimizer()
        await optimizer.initialize()
        
        async with get_database_connection(temp_db) as conn:
            # Create test table
            await asyncio.to_thread(conn.execute, """
                CREATE TABLE IF NOT EXISTS decorator_test (
                    id INTEGER PRIMARY KEY,
                    value TEXT
                )
            """)
            await asyncio.to_thread(conn.execute, 
                                  "INSERT INTO decorator_test (value) VALUES (?)", ('test',))
            await asyncio.to_thread(conn.commit)
            
            # Call decorated function
            result1 = await test_query_function(
                conn, 
                "SELECT * FROM decorator_test WHERE value = ?", 
                ('test',)
            )
            
            # Second call should use cache
            result2 = await test_query_function(
                conn,
                "SELECT * FROM decorator_test WHERE value = ?",
                ('test',)
            )
            
            assert result1 == result2
            assert len(result1) > 0
        
        # Check that metrics were recorded
        assert 'decorated_query' in optimizer.query_metrics


# Global test cleanup
@pytest.fixture(autouse=True)
async def cleanup_global_state():
    """Clean up global state between tests"""
    yield
    
    # Reset global optimizers
    database_optimizer.query_metrics.clear()
    database_optimizer.optimization_suggestions.clear()
    query_analyzer.query_patterns.clear()
    query_analyzer.column_usage.clear()
    
    # Close any open connection pools
    await connection_manager.close_all_pools()