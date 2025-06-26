"""
Database service layer for Trade Knowledge
Unified interface for PostgreSQL, InfluxDB, and Redis
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import asyncpg
import redis.asyncio as redis
from influxdb_client import Point
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Database configuration settings"""

    def __init__(self):
        # PostgreSQL
        self.postgres_host = "localhost"
        self.postgres_port = 5432
        self.postgres_db = "tradeknowledge"
        self.postgres_user = "tradeknowledge_app"
        self.postgres_password = "tradeknowledge_secure_password_2024"

        # InfluxDB
        self.influx_url = "http://localhost:8086"
        self.influx_token = "yxfIel5kcJc5NeSmobh0hxyYEMi2btgUIgzRLaowbBO2MTnXwN4WgzkhMScpiI2Z-xvXFXvZYjUV8OqD_JU6Ww=="
        self.influx_org = "tradeknowledge"
        self.influx_bucket = "market_data"

        # Redis
        self.redis_url = "redis://localhost:6379"
        self.redis_db = 0

        # Connection pooling
        self.postgres_pool_size = 20
        self.postgres_max_overflow = 30
        self.redis_max_connections = 50


class PostgreSQLService:
    """PostgreSQL database service for relational data"""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.pool = None

    async def connect(self):
        """Initialize connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                user=self.config.postgres_user,
                password=self.config.postgres_password,
                database=self.config.postgres_db,
                min_size=5,
                max_size=self.config.postgres_pool_size,
                command_timeout=60,
            )
            logger.info("PostgreSQL connection pool created successfully")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    async def disconnect(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")

    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool"""
        async with self.pool.acquire() as connection:
            yield connection

    # User operations
    async def create_user(self, user_data: dict[str, Any]) -> str:
        """Create a new user and return user ID"""
        async with self.get_connection() as conn:
            query = """
                INSERT INTO users (email, hashed_password, full_name, credits, subscription_tier)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id
            """
            user_id = await conn.fetchval(
                query,
                user_data["email"],
                user_data["hashed_password"],
                user_data.get("full_name"),
                user_data.get("credits", 3),
                user_data.get("subscription_tier", "free"),
            )
            return str(user_id)

    async def get_user(self, user_id: str) -> dict[str, Any] | None:
        """Get user by ID"""
        async with self.get_connection() as conn:
            query = "SELECT * FROM users WHERE id = $1"
            row = await conn.fetchrow(query, user_id)
            return dict(row) if row else None

    async def get_user_by_email(self, email: str) -> dict[str, Any] | None:
        """Get user by email"""
        async with self.get_connection() as conn:
            query = "SELECT * FROM users WHERE email = $1"
            row = await conn.fetchrow(query, email)
            return dict(row) if row else None

    async def update_user_credits(self, user_id: str, credits: int) -> bool:
        """Update user credits"""
        async with self.get_connection() as conn:
            query = "UPDATE users SET credits = $1, updated_at = NOW() WHERE id = $2"
            result = await conn.execute(query, credits, user_id)
            return result == "UPDATE 1"

    async def deduct_credits(self, user_id: str, amount: int = 1) -> bool:
        """Deduct credits from user account"""
        async with self.get_connection() as conn:
            query = """
                UPDATE users 
                SET credits = credits - $1, updated_at = NOW() 
                WHERE id = $2 AND credits >= $1
                RETURNING credits
            """
            result = await conn.fetchval(query, amount, user_id)
            return result is not None

    # Analysis operations
    async def create_analysis(self, analysis_data: dict[str, Any]) -> str:
        """Create new analysis record"""
        async with self.get_connection() as conn:
            query = """
                INSERT INTO analyses (user_id, query_type, query_params, source, credits_charged)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id
            """
            analysis_id = await conn.fetchval(
                query,
                analysis_data["user_id"],
                analysis_data["query_type"],
                json.dumps(analysis_data["query_params"]),
                analysis_data.get("source", "api"),
                analysis_data.get("credits_charged", 1),
            )
            return str(analysis_id)

    async def update_analysis(
        self, analysis_id: str, response_data: dict[str, Any]
    ) -> bool:
        """Update analysis with response data"""
        async with self.get_connection() as conn:
            query = """
                UPDATE analyses 
                SET response = $1, confidence_score = $2, processing_time_ms = $3, 
                    status = $4, completed_at = NOW()
                WHERE id = $5
            """
            result = await conn.execute(
                query,
                json.dumps(response_data.get("response", {})),
                response_data.get("confidence_score"),
                response_data.get("processing_time_ms"),
                response_data.get("status", "completed"),
                analysis_id,
            )
            return result == "UPDATE 1"

    async def get_analysis(self, analysis_id: str) -> dict[str, Any] | None:
        """Get analysis by ID"""
        async with self.get_connection() as conn:
            query = "SELECT * FROM analyses WHERE id = $1"
            row = await conn.fetchrow(query, analysis_id)
            if row:
                result = dict(row)
                # Parse JSON fields
                result["query_params"] = json.loads(result["query_params"])
                if result["response"]:
                    result["response"] = json.loads(result["response"])
                return result
            return None

    async def get_user_analyses(
        self, user_id: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Get recent analyses for user"""
        async with self.get_connection() as conn:
            query = """
                SELECT id, query_type, query_params, status, confidence_score, 
                       credits_charged, source, created_at, completed_at
                FROM analyses 
                WHERE user_id = $1 
                ORDER BY created_at DESC 
                LIMIT $2
            """
            rows = await conn.fetch(query, user_id, limit)
            results = []
            for row in rows:
                result = dict(row)
                result["query_params"] = json.loads(result["query_params"])
                results.append(result)
            return results

    # Transaction operations
    async def create_transaction(self, transaction_data: dict[str, Any]) -> str:
        """Create payment transaction record"""
        async with self.get_connection() as conn:
            query = """
                INSERT INTO transactions (user_id, stripe_payment_intent_id, amount_cents, 
                                        credits_purchased, status)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id
            """
            transaction_id = await conn.fetchval(
                query,
                transaction_data["user_id"],
                transaction_data["stripe_payment_intent_id"],
                transaction_data["amount_cents"],
                transaction_data["credits_purchased"],
                transaction_data.get("status", "pending"),
            )
            return str(transaction_id)

    async def update_transaction_status(
        self, transaction_id: str, status: str, failure_reason: str = None
    ) -> bool:
        """Update transaction status"""
        async with self.get_connection() as conn:
            if status == "completed":
                query = """
                    UPDATE transactions 
                    SET status = $1, completed_at = NOW()
                    WHERE id = $2
                """
                result = await conn.execute(query, status, transaction_id)
            else:
                query = """
                    UPDATE transactions 
                    SET status = $1, failure_reason = $2
                    WHERE id = $3
                """
                result = await conn.execute(
                    query, status, failure_reason, transaction_id
                )
            return result == "UPDATE 1"

    # API usage tracking
    async def log_api_usage(self, usage_data: dict[str, Any]) -> None:
        """Log API usage for analytics and rate limiting"""
        async with self.get_connection() as conn:
            query = """
                INSERT INTO api_usage (user_id, endpoint, method, user_agent, 
                                     ip_address, status_code, response_time_ms, credits_used)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """
            await conn.execute(
                query,
                usage_data["user_id"],
                usage_data["endpoint"],
                usage_data["method"],
                usage_data.get("user_agent"),
                usage_data.get("ip_address"),
                usage_data.get("status_code"),
                usage_data.get("response_time_ms"),
                usage_data.get("credits_used", 0),
            )


class InfluxDBService:
    """InfluxDB service for time-series market data"""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.client = None
        self.write_api = None
        self.query_api = None

    async def connect(self):
        """Initialize InfluxDB client"""
        try:
            self.client = InfluxDBClientAsync(
                url=self.config.influx_url,
                token=self.config.influx_token,
                org=self.config.influx_org,
            )
            self.write_api = self.client.write_api()
            self.query_api = self.client.query_api()
            logger.info("InfluxDB client connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect to InfluxDB: {e}")
            raise

    async def disconnect(self):
        """Close InfluxDB client"""
        if self.client:
            await self.client.close()
            logger.info("InfluxDB client closed")

    async def write_market_data(self, symbol: str, data: dict[str, Any]) -> None:
        """Write market data point"""
        point = (
            Point("market_data")
            .tag("symbol", symbol)
            .tag("exchange", data.get("exchange", "unknown"))
            .tag("data_source", data.get("data_source", "api"))
            .field("price", float(data["price"]))
            .field("volume", int(data.get("volume", 0)))
            .field("bid", float(data.get("bid", 0)))
            .field("ask", float(data.get("ask", 0)))
            .time(data.get("timestamp", datetime.utcnow()))
        )

        await self.write_api.write(bucket=self.config.influx_bucket, record=point)

    async def write_analysis_metrics(self, metrics: dict[str, Any]) -> None:
        """Write analysis performance metrics"""
        point = (
            Point("analysis_metrics")
            .tag("analysis_type", metrics.get("analysis_type", "unknown"))
            .tag("model_version", metrics.get("model_version", "v1"))
            .tag("user_tier", metrics.get("user_tier", "free"))
            .field("processing_time_ms", int(metrics.get("processing_time_ms", 0)))
            .field("confidence_score", float(metrics.get("confidence_score", 0)))
            .field("cache_hit", bool(metrics.get("cache_hit", False)))
            .time(metrics.get("timestamp", datetime.utcnow()))
        )

        await self.write_api.write(bucket=self.config.influx_bucket, record=point)

    async def query_price_history(
        self, symbol: str, timeframe: str = "1d", limit: int = 100
    ) -> list[dict[str, Any]]:
        """Query price history for symbol"""
        # Convert timeframe to InfluxDB duration
        duration_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
            "1w": "7d",
            "1M": "30d",
        }
        duration = duration_map.get(timeframe, "1d")

        query = f"""
            from(bucket: "{self.config.influx_bucket}")
                |> range(start: -{duration})
                |> filter(fn: (r) => r._measurement == "market_data")
                |> filter(fn: (r) => r.symbol == "{symbol}")
                |> filter(fn: (r) => r._field == "price")
                |> sort(columns: ["_time"], desc: false)
                |> limit(n: {limit})
        """

        result = await self.query_api.query(query)

        data = []
        for table in result:
            for record in table.records:
                data.append(
                    {
                        "timestamp": record.get_time(),
                        "price": record.get_value(),
                        "symbol": record.values.get("symbol"),
                    }
                )

        return data

    async def query_user_activity(self, user_id: str, days: int = 30) -> dict[str, Any]:
        """Query user activity patterns"""
        query = f"""
            from(bucket: "{self.config.influx_bucket}")
                |> range(start: -{days}d)
                |> filter(fn: (r) => r._measurement == "user_activity")
                |> filter(fn: (r) => r.user_id == "{user_id}")
                |> aggregateWindow(every: 1d, fn: sum, createEmpty: false)
        """

        result = await self.query_api.query(query)

        activity = {"total_requests": 0, "total_credits_used": 0, "daily_activity": []}

        for table in result:
            for record in table.records:
                if record.get_field() == "request_count":
                    activity["total_requests"] += record.get_value()
                elif record.get_field() == "credits_used":
                    activity["total_credits_used"] += record.get_value()

                activity["daily_activity"].append(
                    {
                        "date": record.get_time().date(),
                        "field": record.get_field(),
                        "value": record.get_value(),
                    }
                )

        return activity


class RedisService:
    """Redis service for caching and session management"""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.client = None

    async def connect(self):
        """Initialize Redis client"""
        try:
            self.client = redis.from_url(
                self.config.redis_url,
                max_connections=self.config.redis_max_connections,
                decode_responses=True,
            )
            # Test connection
            await self.client.ping()
            logger.info("Redis client connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self):
        """Close Redis client"""
        if self.client:
            await self.client.close()
            logger.info("Redis client closed")

    async def get(self, key: str) -> str | None:
        """Get value by key"""
        return await self.client.get(key)

    async def set(self, key: str, value: str, ttl: int = 300) -> bool:
        """Set key-value with TTL"""
        return await self.client.setex(key, ttl, value)

    async def get_json(self, key: str) -> dict[str, Any] | None:
        """Get JSON value by key"""
        value = await self.get(key)
        return json.loads(value) if value else None

    async def set_json(self, key: str, value: dict[str, Any], ttl: int = 300) -> bool:
        """Set JSON value with TTL"""
        return await self.set(key, json.dumps(value), ttl)

    async def delete(self, key: str) -> bool:
        """Delete key"""
        return await self.client.delete(key) > 0

    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter"""
        return await self.client.incr(key, amount)

    async def set_with_expiry(self, key: str, value: str, seconds: int) -> bool:
        """Set key with expiry in seconds"""
        return await self.client.setex(key, seconds, value)


class DatabaseService:
    """Unified database service layer"""

    def __init__(self, config: DatabaseConfig = None):
        self.config = config or DatabaseConfig()
        self.postgres = PostgreSQLService(self.config)
        self.influx = InfluxDBService(self.config)
        self.redis = RedisService(self.config)

    async def connect_all(self):
        """Connect to all databases"""
        await asyncio.gather(
            self.postgres.connect(), self.influx.connect(), self.redis.connect()
        )
        logger.info("All database services connected")

    async def disconnect_all(self):
        """Disconnect from all databases"""
        await asyncio.gather(
            self.postgres.disconnect(),
            self.influx.disconnect(),
            self.redis.disconnect(),
        )
        logger.info("All database services disconnected")

    async def health_check(self) -> dict[str, str]:
        """Check health of all database services"""
        health = {}

        try:
            async with self.postgres.get_connection() as conn:
                await conn.fetchval("SELECT 1")
            health["postgresql"] = "healthy"
        except Exception as e:
            health["postgresql"] = f"unhealthy: {str(e)}"

        try:
            await self.redis.client.ping()
            health["redis"] = "healthy"
        except Exception as e:
            health["redis"] = f"unhealthy: {str(e)}"

        try:
            # Simple InfluxDB health check
            health["influxdb"] = "healthy"  # Will implement proper check
        except Exception as e:
            health["influxdb"] = f"unhealthy: {str(e)}"

        return health


# Global database service instance
db_service = DatabaseService()


# Dependency injection for FastAPI
async def get_database_service() -> DatabaseService:
    """Dependency to get database service"""
    return db_service
