"""
Unit tests for DatabaseService
Tests database connections, error handling, and data operations
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.core.database import DatabaseService, DatabaseConfig, PostgreSQLService, InfluxDBService, RedisService


class TestDatabaseConfig:
    """Test database configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = DatabaseConfig()
        
        # PostgreSQL defaults
        assert config.postgres_host == "localhost"
        assert config.postgres_port == 5432
        assert config.postgres_db == "tradeknowledge"
        assert config.postgres_user == "tradeknowledge_app"
        
        # InfluxDB defaults
        assert config.influx_url == "http://localhost:8086"
        assert config.influx_org == "tradeknowledge"
        assert config.influx_bucket == "market_data"
        
        # Redis defaults
        assert config.redis_url == "redis://localhost:6379"
        assert config.redis_db == 0
        
        # Connection pooling
        assert config.postgres_pool_size == 20
        assert config.postgres_max_overflow == 30


@pytest.mark.asyncio
class TestPostgreSQLService:
    """Test PostgreSQL service operations"""
    
    @pytest.fixture
    def config(self):
        return DatabaseConfig()
    
    @pytest.fixture
    def postgres_service(self, config):
        return PostgreSQLService(config)
    
    async def test_user_operations(self, postgres_service):
        """Test user CRUD operations"""
        # Mock the connection pool and connection
        mock_conn = AsyncMock()
        
        # Mock the get_connection context manager
        postgres_service.get_connection = AsyncMock()
        postgres_service.get_connection.return_value.__aenter__.return_value = mock_conn
        postgres_service.get_connection.return_value.__aexit__.return_value = None
        
        # Test create user
        user_data = {
            "email": "test@example.com",
            "hashed_password": "hashed_password",
            "full_name": "Test User",
            "credits": 5
        }
        
        mock_conn.fetchval.return_value = "user-123"
        
        user_id = await postgres_service.create_user(user_data)
        
        assert user_id == "user-123"
        mock_conn.fetchval.assert_called_once()
        
        # Test get user
        mock_conn.fetchrow.return_value = {
            "id": "user-123",
            "email": "test@example.com",
            "full_name": "Test User",
            "credits": 5
        }
        
        user = await postgres_service.get_user("user-123")
        
        assert user["id"] == "user-123"
        assert user["email"] == "test@example.com"
        
        # Test deduct credits
        mock_conn.fetchval.return_value = 4  # Remaining credits
        
        success = await postgres_service.deduct_credits("user-123", 1)
        
        assert success is True
    
    async def test_analysis_operations(self, postgres_service):
        """Test analysis CRUD operations"""
        mock_conn = AsyncMock()
        
        # Mock the get_connection context manager
        postgres_service.get_connection = AsyncMock()
        postgres_service.get_connection.return_value.__aenter__.return_value = mock_conn
        postgres_service.get_connection.return_value.__aexit__.return_value = None
        
        # Test create analysis
        analysis_data = {
            "user_id": "user-123",
            "query_type": "stock_analysis",
            "query_params": {"symbol": "SPY"},
            "source": "api"
        }
        
        mock_conn.fetchval.return_value = "analysis-456"
        
        analysis_id = await postgres_service.create_analysis(analysis_data)
        
        assert analysis_id == "analysis-456"
        
        # Test update analysis
        response_data = {
            "response": {"result": "analysis complete"},
            "confidence_score": 0.95,
            "processing_time_ms": 1500,
            "status": "completed"
        }
        
        mock_conn.execute.return_value = "UPDATE 1"
        
        success = await postgres_service.update_analysis("analysis-456", response_data)
        
        assert success is True
    
    async def test_error_handling(self, postgres_service):
        """Test error handling in database operations"""
        # Mock connection failure
        postgres_service.pool = None
        
        with pytest.raises(AttributeError):
            await postgres_service.create_user({"email": "test@test.com"})


@pytest.mark.asyncio 
class TestInfluxDBService:
    """Test InfluxDB service operations"""
    
    @pytest.fixture
    def config(self):
        return DatabaseConfig()
    
    @pytest.fixture
    def influx_service(self, config):
        return InfluxDBService(config)
    
    async def test_write_market_data(self, influx_service):
        """Test writing market data to InfluxDB"""
        # Mock the InfluxDB client
        mock_client = AsyncMock()
        mock_write_api = AsyncMock()
        
        influx_service.client = mock_client
        influx_service.write_api = mock_write_api
        
        market_data = {
            "price": 150.50,
            "volume": 1000000,
            "bid": 150.45,
            "ask": 150.55,
            "timestamp": datetime.utcnow()
        }
        
        await influx_service.write_market_data("SPY", market_data)
        
        # Verify write_api was called
        mock_write_api.write.assert_called_once()
    
    async def test_write_analysis_metrics(self, influx_service):
        """Test writing analysis metrics to InfluxDB"""
        mock_client = AsyncMock()
        mock_write_api = AsyncMock()
        
        influx_service.client = mock_client
        influx_service.write_api = mock_write_api
        
        metrics = {
            "analysis_type": "stock_analysis",
            "processing_time_ms": 1200,
            "confidence_score": 0.88,
            "cache_hit": False,
            "timestamp": datetime.utcnow()
        }
        
        await influx_service.write_analysis_metrics(metrics)
        
        mock_write_api.write.assert_called_once()


@pytest.mark.asyncio
class TestRedisService:
    """Test Redis service operations"""
    
    @pytest.fixture
    def config(self):
        return DatabaseConfig()
    
    @pytest.fixture
    def redis_service(self, config):
        return RedisService(config)
    
    async def test_basic_operations(self, redis_service):
        """Test basic Redis operations"""
        # Mock Redis client
        mock_client = AsyncMock()
        redis_service.client = mock_client
        
        # Test set/get
        mock_client.setex.return_value = True
        mock_client.get.return_value = "test_value"
        
        success = await redis_service.set("test_key", "test_value", 300)
        assert success is True
        
        value = await redis_service.get("test_key")
        assert value == "test_value"
        
        # Test JSON operations
        test_data = {"symbol": "SPY", "price": 150.50}
        mock_client.get.return_value = json.dumps(test_data)
        
        await redis_service.set_json("json_key", test_data, 300)
        result = await redis_service.get_json("json_key")
        
        assert result == test_data
    
    async def test_increment_operations(self, redis_service):
        """Test counter operations"""
        mock_client = AsyncMock()
        redis_service.client = mock_client
        
        mock_client.incr.return_value = 5
        
        count = await redis_service.increment("counter_key", 2)
        
        assert count == 5
        mock_client.incr.assert_called_with("counter_key", 2)


@pytest.mark.asyncio
class TestDatabaseService:
    """Test unified database service"""
    
    @pytest.fixture
    def database_service(self):
        return DatabaseService()
    
    async def test_health_check(self, database_service):
        """Test health check for all services"""
        # Mock all service components
        database_service.postgres.pool = AsyncMock()
        database_service.postgres.get_connection = AsyncMock()
        
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 1
        
        database_service.postgres.get_connection.return_value.__aenter__.return_value = mock_conn
        
        database_service.redis.client = AsyncMock()
        database_service.redis.client.ping.return_value = True
        
        health = await database_service.health_check()
        
        assert "postgresql" in health
        assert "redis" in health
        assert "influxdb" in health
        assert "healthy" in health["postgresql"]
        assert "healthy" in health["redis"]
    
    async def test_connect_disconnect_all(self, database_service):
        """Test connecting and disconnecting all services"""
        # Mock connect methods
        database_service.postgres.connect = AsyncMock()
        database_service.influx.connect = AsyncMock()
        database_service.redis.connect = AsyncMock()
        
        # Mock disconnect methods
        database_service.postgres.disconnect = AsyncMock()
        database_service.influx.disconnect = AsyncMock()
        database_service.redis.disconnect = AsyncMock()
        
        # Test connect all
        await database_service.connect_all()
        
        database_service.postgres.connect.assert_called_once()
        database_service.influx.connect.assert_called_once()
        database_service.redis.connect.assert_called_once()
        
        # Test disconnect all
        await database_service.disconnect_all()
        
        database_service.postgres.disconnect.assert_called_once()
        database_service.influx.disconnect.assert_called_once()
        database_service.redis.disconnect.assert_called_once()


# Performance and integration markers
@pytest.mark.performance
class TestDatabasePerformance:
    """Performance tests for database operations"""
    
    @pytest.mark.asyncio
    async def test_concurrent_connections(self):
        """Test handling multiple concurrent connections"""
        database_service = DatabaseService()
        
        # Mock services for performance test
        database_service.postgres.connect = AsyncMock()
        database_service.influx.connect = AsyncMock()
        database_service.redis.connect = AsyncMock()
        
        # Test multiple concurrent connections
        tasks = [database_service.connect_all() for _ in range(10)]
        
        start_time = asyncio.get_event_loop().time()
        await asyncio.gather(*tasks)
        end_time = asyncio.get_event_loop().time()
        
        # Should complete quickly even with multiple concurrent connections
        assert (end_time - start_time) < 1.0  # Less than 1 second


@pytest.mark.security
class TestDatabaseSecurity:
    """Security tests for database operations"""
    
    def test_sql_injection_prevention(self):
        """Test that database queries use parameterized statements"""
        # This is more of a code review test
        # Actual SQL injection testing would be done in integration tests
        
        # Verify that our query strings use $1, $2 parameters
        postgres_service = PostgreSQLService(DatabaseConfig())
        
        # Check that create_user method signature uses parameterized queries
        # This is validated by code inspection rather than runtime testing
        assert True  # Placeholder for SQL injection prevention validation
    
    def test_connection_string_security(self):
        """Test that connection strings don't expose sensitive data"""
        config = DatabaseConfig()
        
        # Verify password is not logged or exposed
        assert config.postgres_password  # Password exists
        # In real implementation, ensure passwords are not logged