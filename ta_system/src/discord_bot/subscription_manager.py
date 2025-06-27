#!/usr/bin/env python3
"""
TradeKnowledge Subscription Management System

Handles user subscriptions, usage tracking, and free-to-paid conversion for the Discord bot.
Designed to convert technical users like Case from free to paid tiers.

Features:
- Usage tracking and limits enforcement
- Subscription tier management
- Free-to-paid conversion analytics
- PostgreSQL MCP integration for data persistence
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from enum import Enum

import asyncpg

logger = logging.getLogger(__name__)

class SubscriptionTier(Enum):
    """Subscription tier definitions with limits and features."""
    FREE = "free"
    ENGINEER = "engineer"
    ROCKET_SCIENTIST = "rocket_scientist" 
    MISSION_CONTROL = "mission_control"

@dataclass
class UserSubscription:
    """User subscription data model."""
    user_id: int
    discord_username: str
    tier: SubscriptionTier
    subscription_start: datetime
    subscription_end: Optional[datetime]
    daily_usage_count: int
    total_commands: int
    join_date: datetime
    last_active: datetime
    conversion_date: Optional[datetime] = None
    payment_method: Optional[str] = None

@dataclass
class UsageRecord:
    """Command usage tracking record."""
    user_id: int
    command_name: str
    timestamp: datetime
    tier: SubscriptionTier
    success: bool
    response_time_ms: int

class SubscriptionManager:
    """
    Manages user subscriptions and usage tracking for TradeKnowledge platform.
    
    Key Features:
    - Free tier with daily limits to encourage upgrades
    - Usage analytics for conversion optimization
    - PostgreSQL persistence for enterprise-grade data management
    - Automated billing and subscription lifecycle management
    """
    
    def __init__(self, database_url: str = None):
        """Initialize subscription manager with database connection."""
        self.database_url = database_url or "postgresql://localhost:5432/tradeknowledge"
        self.db_pool = None
        
        # Subscription tier limits and pricing
        self.tier_limits = {
            SubscriptionTier.FREE: {
                "daily_commands": 5,
                "neural_confidence": [90.0],
                "features": ["basic_options_scan", "community_access"],
                "price": 0
            },
            SubscriptionTier.ENGINEER: {
                "daily_commands": 100,
                "neural_confidence": [90.0, 99.0, 99.5],
                "features": ["unlimited_scans", "postgresql_access", "priority_support"],
                "price": 49
            },
            SubscriptionTier.ROCKET_SCIENTIST: {
                "daily_commands": float('inf'),
                "neural_confidence": [90.0, 99.0, 99.5],
                "features": ["blackboard_access", "custom_training", "api_access", "private_channels"],
                "price": 149
            },
            SubscriptionTier.MISSION_CONTROL: {
                "daily_commands": float('inf'),
                "neural_confidence": [90.0, 99.0, 99.5],
                "features": ["white_label", "custom_development", "enterprise_feeds", "consulting"],
                "price": 349
            }
        }

    async def initialize(self):
        """Initialize database connection and create tables."""
        try:
            # Create database connection pool
            self.db_pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            
            # Create tables if they don't exist
            await self._create_tables()
            
            logger.info("✅ Subscription manager initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize subscription manager: {e}")
            raise

    async def _create_tables(self):
        """Create subscription and usage tracking tables."""
        async with self.db_pool.acquire() as conn:
            # Users and subscriptions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_subscriptions (
                    user_id BIGINT PRIMARY KEY,
                    discord_username VARCHAR(100) NOT NULL,
                    tier VARCHAR(50) NOT NULL DEFAULT 'free',
                    subscription_start TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    subscription_end TIMESTAMP WITH TIME ZONE,
                    daily_usage_count INTEGER DEFAULT 0,
                    total_commands INTEGER DEFAULT 0,
                    join_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    last_active TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    conversion_date TIMESTAMP WITH TIME ZONE,
                    payment_method VARCHAR(50),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            
            # Command usage tracking table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS usage_records (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL REFERENCES user_subscriptions(user_id),
                    command_name VARCHAR(100) NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    tier VARCHAR(50) NOT NULL,
                    success BOOLEAN DEFAULT TRUE,
                    response_time_ms INTEGER,
                    parameters JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            
            # Daily usage summary table for analytics
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_usage_summary (
                    date DATE NOT NULL,
                    tier VARCHAR(50) NOT NULL,
                    total_users INTEGER DEFAULT 0,
                    total_commands INTEGER DEFAULT 0,
                    avg_response_time_ms FLOAT,
                    conversion_count INTEGER DEFAULT 0,
                    PRIMARY KEY (date, tier)
                )
            """)
            
            # Create indexes for performance
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_user_date ON usage_records(user_id, DATE(timestamp))")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_command ON usage_records(command_name)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_subscription_tier ON user_subscriptions(tier)")

    async def get_or_create_user(self, user_id: int, discord_username: str) -> UserSubscription:
        """Get existing user or create new free tier user."""
        async with self.db_pool.acquire() as conn:
            # Try to get existing user
            row = await conn.fetchrow(
                "SELECT * FROM user_subscriptions WHERE user_id = $1",
                user_id
            )
            
            if row:
                return UserSubscription(
                    user_id=row['user_id'],
                    discord_username=row['discord_username'],
                    tier=SubscriptionTier(row['tier']),
                    subscription_start=row['subscription_start'],
                    subscription_end=row['subscription_end'],
                    daily_usage_count=row['daily_usage_count'],
                    total_commands=row['total_commands'],
                    join_date=row['join_date'],
                    last_active=row['last_active'],
                    conversion_date=row['conversion_date'],
                    payment_method=row['payment_method']
                )
            
            # Create new free tier user
            await conn.execute("""
                INSERT INTO user_subscriptions 
                (user_id, discord_username, tier) 
                VALUES ($1, $2, $3)
            """, user_id, discord_username, SubscriptionTier.FREE.value)
            
            return UserSubscription(
                user_id=user_id,
                discord_username=discord_username,
                tier=SubscriptionTier.FREE,
                subscription_start=datetime.now(timezone.utc),
                subscription_end=None,
                daily_usage_count=0,
                total_commands=0,
                join_date=datetime.now(timezone.utc),
                last_active=datetime.now(timezone.utc)
            )

    async def get_user_tier(self, user_id: int) -> str:
        """Get user's current subscription tier."""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT tier FROM user_subscriptions WHERE user_id = $1",
                user_id
            )
            return row['tier'] if row else 'free'

    async def get_daily_usage(self, user_id: int) -> Dict[str, Any]:
        """Get user's usage for today."""
        today = datetime.now(timezone.utc).date()
        
        async with self.db_pool.acquire() as conn:
            # Get today's usage count
            usage_count = await conn.fetchval("""
                SELECT COUNT(*) FROM usage_records 
                WHERE user_id = $1 AND DATE(timestamp) = $2
            """, user_id, today)
            
            # Get user's tier and limits
            user_data = await conn.fetchrow(
                "SELECT tier, daily_usage_count FROM user_subscriptions WHERE user_id = $1",
                user_id
            )
            
            tier = user_data['tier'] if user_data else 'free'
            tier_enum = SubscriptionTier(tier)
            daily_limit = self.tier_limits[tier_enum]['daily_commands']
            
            return {
                'count': usage_count or 0,
                'limit': daily_limit,
                'tier': tier,
                'remaining': max(0, daily_limit - (usage_count or 0)) if daily_limit != float('inf') else float('inf')
            }

    async def increment_usage(self, user_id: int, command_name: str, success: bool = True, response_time_ms: int = None, parameters: Dict = None):
        """Record command usage for analytics and billing."""
        async with self.db_pool.acquire() as conn:
            # Get user tier
            tier = await self.get_user_tier(user_id)
            
            # Record usage
            await conn.execute("""
                INSERT INTO usage_records 
                (user_id, command_name, tier, success, response_time_ms, parameters)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, user_id, command_name, tier, success, response_time_ms, parameters)
            
            # Update user stats
            await conn.execute("""
                UPDATE user_subscriptions 
                SET total_commands = total_commands + 1,
                    last_active = NOW()
                WHERE user_id = $1
            """, user_id)

    async def upgrade_user(self, user_id: int, new_tier: SubscriptionTier, subscription_end: datetime = None, payment_method: str = None) -> bool:
        """Upgrade user to paid tier."""
        try:
            async with self.db_pool.acquire() as conn:
                # Check if this is a conversion from free
                current_tier = await self.get_user_tier(user_id)
                is_conversion = current_tier == 'free'
                
                # Update subscription
                await conn.execute("""
                    UPDATE user_subscriptions 
                    SET tier = $1,
                        subscription_end = $2,
                        payment_method = $3,
                        conversion_date = CASE WHEN $4 THEN NOW() ELSE conversion_date END,
                        updated_at = NOW()
                    WHERE user_id = $5
                """, new_tier.value, subscription_end, payment_method, is_conversion, user_id)
                
                logger.info(f"✅ User {user_id} upgraded to {new_tier.value}")
                
                # Track conversion analytics
                if is_conversion:
                    await self._track_conversion(user_id, new_tier)
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to upgrade user {user_id}: {e}")
            return False

    async def downgrade_user(self, user_id: int, reason: str = "subscription_expired"):
        """Downgrade user to free tier (e.g., when subscription expires)."""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE user_subscriptions 
                SET tier = 'free',
                    subscription_end = NULL,
                    updated_at = NOW()
                WHERE user_id = $1
            """, user_id)
            
            logger.info(f"📉 User {user_id} downgraded to free tier: {reason}")

    async def get_user_data(self, user_id: int) -> Dict[str, Any]:
        """Get comprehensive user data for status display."""
        async with self.db_pool.acquire() as conn:
            user_data = await conn.fetchrow(
                "SELECT * FROM user_subscriptions WHERE user_id = $1",
                user_id
            )
            
            if not user_data:
                return None
            
            # Get today's usage
            daily_usage = await self.get_daily_usage(user_id)
            
            return {
                'user_id': user_data['user_id'],
                'discord_username': user_data['discord_username'],
                'tier': user_data['tier'],
                'subscription_start': user_data['subscription_start'],
                'subscription_end': user_data['subscription_end'],
                'join_date': user_data['join_date'],
                'last_active': user_data['last_active'],
                'total_commands': user_data['total_commands'],
                'daily_usage': daily_usage['count'],
                'daily_limit': daily_usage['limit'],
                'conversion_date': user_data['conversion_date'],
                'tier_features': self.tier_limits[SubscriptionTier(user_data['tier'])]['features']
            }

    async def get_analytics_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get subscription and usage analytics for the platform."""
        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        async with self.db_pool.acquire() as conn:
            # User counts by tier
            tier_counts = await conn.fetch("""
                SELECT tier, COUNT(*) as count 
                FROM user_subscriptions 
                GROUP BY tier
            """)
            
            # Conversion metrics
            conversions = await conn.fetch("""
                SELECT DATE(conversion_date) as date, COUNT(*) as conversions
                FROM user_subscriptions 
                WHERE conversion_date >= $1
                GROUP BY DATE(conversion_date)
                ORDER BY date
            """, start_date)
            
            # Usage metrics
            usage_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(DISTINCT user_id) as active_users,
                    COUNT(*) as total_commands,
                    AVG(response_time_ms) as avg_response_time
                FROM usage_records 
                WHERE timestamp >= $1
            """, start_date)
            
            # Revenue estimation (simplified)
            revenue = await conn.fetchrow("""
                SELECT 
                    SUM(CASE tier 
                        WHEN 'engineer' THEN 49
                        WHEN 'rocket_scientist' THEN 149
                        WHEN 'mission_control' THEN 349
                        ELSE 0 
                    END) as monthly_revenue
                FROM user_subscriptions 
                WHERE tier != 'free' AND (subscription_end IS NULL OR subscription_end > NOW())
            """)
            
            return {
                'tier_distribution': {row['tier']: row['count'] for row in tier_counts},
                'daily_conversions': [(row['date'], row['conversions']) for row in conversions],
                'active_users': usage_stats['active_users'] or 0,
                'total_commands': usage_stats['total_commands'] or 0,
                'avg_response_time': usage_stats['avg_response_time'] or 0,
                'estimated_monthly_revenue': revenue['monthly_revenue'] or 0,
                'conversion_rate': await self._calculate_conversion_rate(days)
            }

    async def _track_conversion(self, user_id: int, new_tier: SubscriptionTier):
        """Track conversion event for analytics."""
        async with self.db_pool.acquire() as conn:
            # Record conversion event
            await conn.execute("""
                INSERT INTO usage_records 
                (user_id, command_name, tier, parameters)
                VALUES ($1, 'tier_conversion', $2, $3)
            """, user_id, new_tier.value, {'from_tier': 'free', 'to_tier': new_tier.value})

    async def _calculate_conversion_rate(self, days: int = 30) -> float:
        """Calculate free-to-paid conversion rate."""
        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        async with self.db_pool.acquire() as conn:
            total_free_users = await conn.fetchval("""
                SELECT COUNT(*) FROM user_subscriptions 
                WHERE join_date >= $1
            """, start_date)
            
            converted_users = await conn.fetchval("""
                SELECT COUNT(*) FROM user_subscriptions 
                WHERE join_date >= $1 AND conversion_date IS NOT NULL
            """, start_date)
            
            return (converted_users / total_free_users * 100) if total_free_users > 0 else 0.0

    async def check_subscription_expiry(self):
        """Background task to check and handle subscription expiries."""
        async with self.db_pool.acquire() as conn:
            expired_users = await conn.fetch("""
                SELECT user_id FROM user_subscriptions 
                WHERE subscription_end IS NOT NULL 
                AND subscription_end <= NOW()
                AND tier != 'free'
            """)
            
            for user in expired_users:
                await self.downgrade_user(user['user_id'], "subscription_expired")

    async def close(self):
        """Close database connections."""
        if self.db_pool:
            await self.db_pool.close()
            logger.info("🔌 Subscription manager database connections closed")