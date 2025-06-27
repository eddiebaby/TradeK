#!/usr/bin/env python3
"""
TradeKnowledge Multi-Agent Blackboard System

Centralized communication hub for coordinating multiple AI agents in systematic trading.
Enables real-time collaboration between research, strategy, and execution agents.

Features:
- Multi-agent message passing and coordination
- Real-time opportunity discovery and sharing
- Systematic signal aggregation and consensus building
- Risk assessment and position management coordination
- Knowledge graph integration for cross-agent learning
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

import asyncpg
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Types of agents in the system."""
    RESEARCHER = "researcher"
    STRATEGIST = "strategist"
    EXECUTOR = "executor"
    RISK_MANAGER = "risk_manager"
    NEURAL_NETWORK = "neural_network"

class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class OpportunityType(Enum):
    """Types of trading opportunities."""
    DIAGONAL_SPREAD = "diagonal_spread"
    IRON_CONDOR = "iron_condor"
    COVERED_CALL = "covered_call"
    PROTECTIVE_PUT = "protective_put"
    ARBITRAGE = "arbitrage"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"

@dataclass
class AgentMessage:
    """Message structure for inter-agent communication."""
    id: str
    sender_agent: AgentType
    recipient_agent: Optional[AgentType]
    message_type: str
    content: Dict[str, Any]
    priority: MessagePriority
    timestamp: datetime
    correlation_id: Optional[str] = None
    expires_at: Optional[datetime] = None
    requires_acknowledgment: bool = False

@dataclass
class TradingOpportunity:
    """Trading opportunity structure for cross-agent sharing."""
    id: str
    opportunity_type: OpportunityType
    symbol: str
    strategy: str
    entry_price: float
    target_price: float
    stop_loss: float
    max_profit: float
    max_loss: float
    probability_success: float
    confidence_level: float
    time_horizon: str
    risk_reward_ratio: float
    created_by: AgentType
    created_at: datetime
    expires_at: datetime
    validation_count: int = 0
    agent_consensus: Dict[AgentType, bool] = None
    metadata: Dict[str, Any] = None

@dataclass
class MarketSignal:
    """Market signal for systematic trading decisions."""
    id: str
    symbol: str
    signal_type: str  # buy, sell, hold, close
    strength: float  # 0-1 scale
    confidence: float  # 0-1 scale
    source_agent: AgentType
    reasoning: str
    supporting_data: Dict[str, Any]
    timestamp: datetime
    expiry: datetime

@dataclass
class RiskAlert:
    """Risk management alert for position monitoring."""
    id: str
    alert_type: str  # position_limit, volatility_spike, correlation_break
    severity: str  # low, medium, high, critical
    affected_positions: List[str]
    description: str
    recommended_action: str
    created_by: AgentType
    timestamp: datetime

class BlackboardSystem:
    """
    Multi-agent blackboard system for coordinated trading intelligence.
    
    Provides a centralized hub for agents to:
    - Share market opportunities and signals
    - Coordinate trading decisions through consensus
    - Aggregate risk assessments and alerts
    - Build collective knowledge through interaction
    """
    
    def __init__(self, database_url: str = None, redis_url: str = None):
        """Initialize blackboard system with persistence and caching."""
        self.database_url = database_url or "postgresql://localhost:5432/tradeknowledge"
        self.redis_url = redis_url or "redis://localhost:6379"
        
        self.db_pool = None
        self.redis_pool = None
        
        # Active agents registry
        self.active_agents: Set[AgentType] = set()
        self.agent_heartbeats: Dict[AgentType, datetime] = {}
        
        # Message queues for each agent
        self.message_queues: Dict[AgentType, asyncio.Queue] = {
            agent_type: asyncio.Queue() for agent_type in AgentType
        }
        
        # Opportunity tracking
        self.active_opportunities: Dict[str, TradingOpportunity] = {}
        self.opportunity_subscribers: Dict[OpportunityType, Set[AgentType]] = {}
        
        # Signal aggregation
        self.active_signals: Dict[str, List[MarketSignal]] = {}
        self.consensus_thresholds = {
            'opportunity_validation': 2,  # Minimum agents to validate opportunity
            'signal_confirmation': 3,     # Minimum agents for signal consensus
            'risk_alert_escalation': 2    # Minimum agents for risk escalation
        }

    async def initialize(self):
        """Initialize blackboard system with database and Redis connections."""
        try:
            # Initialize database connection
            self.db_pool = await asyncpg.create_pool(
                self.database_url,
                min_size=3,
                max_size=15,
                command_timeout=60
            )
            
            # Initialize Redis connection
            self.redis_pool = await redis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=20
            )
            
            # Create database tables
            await self._create_tables()
            
            # Start background tasks
            asyncio.create_task(self._heartbeat_monitor())
            asyncio.create_task(self._opportunity_expiry_monitor())
            asyncio.create_task(self._signal_aggregator())
            
            logger.info("🏛️ Blackboard system initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize blackboard system: {e}")
            raise

    async def _create_tables(self):
        """Create database tables for persistent storage."""
        async with self.db_pool.acquire() as conn:
            # Agent messages table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_messages (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    sender_agent VARCHAR(50) NOT NULL,
                    recipient_agent VARCHAR(50),
                    message_type VARCHAR(100) NOT NULL,
                    content JSONB NOT NULL,
                    priority INTEGER NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    correlation_id UUID,
                    expires_at TIMESTAMP WITH TIME ZONE,
                    requires_acknowledgment BOOLEAN DEFAULT FALSE,
                    acknowledged_at TIMESTAMP WITH TIME ZONE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            
            # Trading opportunities table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS trading_opportunities (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    opportunity_type VARCHAR(50) NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    strategy VARCHAR(100) NOT NULL,
                    entry_price DECIMAL(10,2) NOT NULL,
                    target_price DECIMAL(10,2) NOT NULL,
                    stop_loss DECIMAL(10,2) NOT NULL,
                    max_profit DECIMAL(10,2) NOT NULL,
                    max_loss DECIMAL(10,2) NOT NULL,
                    probability_success DECIMAL(5,2) NOT NULL,
                    confidence_level DECIMAL(5,2) NOT NULL,
                    time_horizon VARCHAR(20) NOT NULL,
                    risk_reward_ratio DECIMAL(5,2) NOT NULL,
                    created_by VARCHAR(50) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    validation_count INTEGER DEFAULT 0,
                    agent_consensus JSONB,
                    metadata JSONB,
                    status VARCHAR(20) DEFAULT 'active'
                )
            """)
            
            # Market signals table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS market_signals (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    symbol VARCHAR(20) NOT NULL,
                    signal_type VARCHAR(20) NOT NULL,
                    strength DECIMAL(3,2) NOT NULL,
                    confidence DECIMAL(3,2) NOT NULL,
                    source_agent VARCHAR(50) NOT NULL,
                    reasoning TEXT,
                    supporting_data JSONB,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    expiry TIMESTAMP WITH TIME ZONE NOT NULL,
                    status VARCHAR(20) DEFAULT 'active'
                )
            """)
            
            # Risk alerts table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS risk_alerts (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    alert_type VARCHAR(50) NOT NULL,
                    severity VARCHAR(20) NOT NULL,
                    affected_positions JSONB,
                    description TEXT NOT NULL,
                    recommended_action TEXT NOT NULL,
                    created_by VARCHAR(50) NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    resolved_at TIMESTAMP WITH TIME ZONE,
                    resolution_notes TEXT
                )
            """)
            
            # Create indexes for performance
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_recipient ON agent_messages(recipient_agent, timestamp)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_opportunities_symbol ON trading_opportunities(symbol, status)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol ON market_signals(symbol, status)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_severity ON risk_alerts(severity, timestamp)")

    async def register_agent(self, agent_type: AgentType, agent_id: str = None) -> str:
        """Register an agent with the blackboard system."""
        agent_id = agent_id or str(uuid.uuid4())
        
        self.active_agents.add(agent_type)
        self.agent_heartbeats[agent_type] = datetime.now(timezone.utc)
        
        # Initialize opportunity subscriptions
        if agent_type not in self.opportunity_subscribers:
            self.opportunity_subscribers[agent_type] = set()
        
        logger.info(f"🤖 Agent {agent_type.value} registered with ID: {agent_id}")
        return agent_id

    async def send_message(self, message: AgentMessage) -> bool:
        """Send message to target agent or broadcast."""
        try:
            # Store in database for persistence
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO agent_messages 
                    (sender_agent, recipient_agent, message_type, content, priority, 
                     correlation_id, expires_at, requires_acknowledgment)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, 
                message.sender_agent.value,
                message.recipient_agent.value if message.recipient_agent else None,
                message.message_type,
                json.dumps(message.content),
                message.priority.value,
                message.correlation_id,
                message.expires_at,
                message.requires_acknowledgment
                )
            
            # Send to message queues
            if message.recipient_agent:
                # Send to specific agent
                if message.recipient_agent in self.message_queues:
                    await self.message_queues[message.recipient_agent].put(message)
            else:
                # Broadcast to all active agents
                for agent_type in self.active_agents:
                    if agent_type != message.sender_agent:
                        await self.message_queues[agent_type].put(message)
            
            logger.debug(f"📨 Message sent from {message.sender_agent.value} to {message.recipient_agent}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send message: {e}")
            return False

    async def get_messages(self, agent_type: AgentType, max_messages: int = 10) -> List[AgentMessage]:
        """Get pending messages for an agent."""
        messages = []
        
        try:
            # Get messages from queue (non-blocking)
            while len(messages) < max_messages:
                try:
                    message = self.message_queues[agent_type].get_nowait()
                    messages.append(message)
                except asyncio.QueueEmpty:
                    break
            
            return messages
            
        except Exception as e:
            logger.error(f"❌ Failed to get messages for {agent_type.value}: {e}")
            return []

    async def post_opportunity(self, opportunity: TradingOpportunity) -> bool:
        """Post a trading opportunity for agent validation."""
        try:
            # Store in database
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO trading_opportunities 
                    (opportunity_type, symbol, strategy, entry_price, target_price, stop_loss,
                     max_profit, max_loss, probability_success, confidence_level, time_horizon,
                     risk_reward_ratio, created_by, expires_at, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                """,
                opportunity.opportunity_type.value,
                opportunity.symbol,
                opportunity.strategy,
                opportunity.entry_price,
                opportunity.target_price,
                opportunity.stop_loss,
                opportunity.max_profit,
                opportunity.max_loss,
                opportunity.probability_success,
                opportunity.confidence_level,
                opportunity.time_horizon,
                opportunity.risk_reward_ratio,
                opportunity.created_by.value,
                opportunity.expires_at,
                json.dumps(opportunity.metadata or {})
                )
            
            # Add to active opportunities
            self.active_opportunities[opportunity.id] = opportunity
            
            # Notify subscribed agents
            message = AgentMessage(
                id=str(uuid.uuid4()),
                sender_agent=opportunity.created_by,
                recipient_agent=None,  # Broadcast
                message_type="new_opportunity",
                content=asdict(opportunity),
                priority=MessagePriority.HIGH,
                timestamp=datetime.now(timezone.utc)
            )
            
            await self.send_message(message)
            
            logger.info(f"📈 Opportunity posted: {opportunity.strategy} on {opportunity.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to post opportunity: {e}")
            return False

    async def validate_opportunity(self, opportunity_id: str, agent_type: AgentType, approved: bool, notes: str = "") -> bool:
        """Agent validation of a trading opportunity."""
        try:
            if opportunity_id not in self.active_opportunities:
                return False
            
            opportunity = self.active_opportunities[opportunity_id]
            
            # Update agent consensus
            if opportunity.agent_consensus is None:
                opportunity.agent_consensus = {}
            
            opportunity.agent_consensus[agent_type] = approved
            opportunity.validation_count += 1
            
            # Update database
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE trading_opportunities 
                    SET validation_count = validation_count + 1,
                        agent_consensus = $1
                    WHERE id = $2
                """, json.dumps({k.value: v for k, v in opportunity.agent_consensus.items()}), opportunity_id)
            
            # Check if consensus threshold reached
            validations = list(opportunity.agent_consensus.values())
            if len(validations) >= self.consensus_thresholds['opportunity_validation']:
                consensus_reached = sum(validations) >= len(validations) * 0.6  # 60% approval
                
                if consensus_reached:
                    await self._escalate_opportunity(opportunity)
            
            logger.info(f"✅ Opportunity {opportunity_id} validated by {agent_type.value}: {approved}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to validate opportunity: {e}")
            return False

    async def post_signal(self, signal: MarketSignal) -> bool:
        """Post a market signal for aggregation."""
        try:
            # Store in database
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO market_signals 
                    (symbol, signal_type, strength, confidence, source_agent, reasoning,
                     supporting_data, expiry)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                signal.symbol,
                signal.signal_type,
                signal.strength,
                signal.confidence,
                signal.source_agent.value,
                signal.reasoning,
                json.dumps(signal.supporting_data),
                signal.expiry
                )
            
            # Add to active signals
            if signal.symbol not in self.active_signals:
                self.active_signals[signal.symbol] = []
            
            self.active_signals[signal.symbol].append(signal)
            
            logger.info(f"📊 Signal posted: {signal.signal_type} for {signal.symbol} by {signal.source_agent.value}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to post signal: {e}")
            return False

    async def get_consensus_signals(self, symbol: str) -> Dict[str, Any]:
        """Get consensus signals for a symbol."""
        if symbol not in self.active_signals:
            return {}
        
        signals = self.active_signals[symbol]
        
        # Group by signal type
        signal_groups = {}
        for signal in signals:
            if signal.signal_type not in signal_groups:
                signal_groups[signal.signal_type] = []
            signal_groups[signal.signal_type].append(signal)
        
        consensus = {}
        for signal_type, signal_list in signal_groups.items():
            if len(signal_list) >= self.consensus_thresholds['signal_confirmation']:
                avg_strength = sum(s.strength for s in signal_list) / len(signal_list)
                avg_confidence = sum(s.confidence for s in signal_list) / len(signal_list)
                
                consensus[signal_type] = {
                    'strength': avg_strength,
                    'confidence': avg_confidence,
                    'agent_count': len(signal_list),
                    'timestamp': max(s.timestamp for s in signal_list)
                }
        
        return consensus

    async def post_risk_alert(self, alert: RiskAlert) -> bool:
        """Post a risk management alert."""
        try:
            # Store in database
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO risk_alerts 
                    (alert_type, severity, affected_positions, description, 
                     recommended_action, created_by)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """,
                alert.alert_type,
                alert.severity,
                json.dumps(alert.affected_positions),
                alert.description,
                alert.recommended_action,
                alert.created_by.value
                )
            
            # Send critical alerts immediately
            if alert.severity == "critical":
                message = AgentMessage(
                    id=str(uuid.uuid4()),
                    sender_agent=alert.created_by,
                    recipient_agent=None,  # Broadcast
                    message_type="critical_risk_alert",
                    content=asdict(alert),
                    priority=MessagePriority.CRITICAL,
                    timestamp=datetime.now(timezone.utc)
                )
                
                await self.send_message(message)
            
            logger.warning(f"⚠️ Risk alert posted: {alert.alert_type} - {alert.severity}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to post risk alert: {e}")
            return False

    async def get_new_opportunities(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get new trading opportunities for alert system."""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM trading_opportunities 
                    WHERE status = 'active' 
                    AND expires_at > NOW()
                    AND validation_count >= $1
                    ORDER BY created_at DESC
                    LIMIT $2
                """, self.consensus_thresholds['opportunity_validation'], limit)
            
            opportunities = []
            for row in rows:
                opportunities.append({
                    'id': str(row['id']),
                    'symbol': row['symbol'],
                    'strategy': row['strategy'],
                    'description': f"{row['strategy']} opportunity on {row['symbol']}",
                    'confidence': float(row['confidence_level']),
                    'expected_return': float(row['max_profit']),
                    'risk_reward_ratio': float(row['risk_reward_ratio'])
                })
            
            return opportunities
            
        except Exception as e:
            logger.error(f"❌ Failed to get new opportunities: {e}")
            return []

    async def _escalate_opportunity(self, opportunity: TradingOpportunity):
        """Escalate validated opportunity to execution agents."""
        message = AgentMessage(
            id=str(uuid.uuid4()),
            sender_agent=AgentType.STRATEGIST,
            recipient_agent=AgentType.EXECUTOR,
            message_type="validated_opportunity",
            content=asdict(opportunity),
            priority=MessagePriority.HIGH,
            timestamp=datetime.now(timezone.utc)
        )
        
        await self.send_message(message)
        logger.info(f"🚀 Opportunity escalated for execution: {opportunity.strategy}")

    async def _heartbeat_monitor(self):
        """Monitor agent heartbeats and handle disconnections."""
        while True:
            try:
                current_time = datetime.now(timezone.utc)
                timeout_threshold = current_time - timedelta(minutes=5)
                
                disconnected_agents = []
                for agent_type, last_heartbeat in self.agent_heartbeats.items():
                    if last_heartbeat < timeout_threshold:
                        disconnected_agents.append(agent_type)
                
                for agent_type in disconnected_agents:
                    self.active_agents.discard(agent_type)
                    del self.agent_heartbeats[agent_type]
                    logger.warning(f"🔌 Agent {agent_type.value} disconnected")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"❌ Heartbeat monitor error: {e}")
                await asyncio.sleep(60)

    async def _opportunity_expiry_monitor(self):
        """Monitor and clean up expired opportunities."""
        while True:
            try:
                current_time = datetime.now(timezone.utc)
                
                # Clean up expired opportunities
                expired_ids = [
                    opp_id for opp_id, opp in self.active_opportunities.items()
                    if opp.expires_at < current_time
                ]
                
                for opp_id in expired_ids:
                    del self.active_opportunities[opp_id]
                
                # Update database
                if expired_ids:
                    async with self.db_pool.acquire() as conn:
                        await conn.execute("""
                            UPDATE trading_opportunities 
                            SET status = 'expired' 
                            WHERE expires_at < NOW() AND status = 'active'
                        """)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"❌ Opportunity expiry monitor error: {e}")
                await asyncio.sleep(300)

    async def _signal_aggregator(self):
        """Aggregate and process market signals."""
        while True:
            try:
                # Clean up expired signals
                current_time = datetime.now(timezone.utc)
                
                for symbol in list(self.active_signals.keys()):
                    self.active_signals[symbol] = [
                        signal for signal in self.active_signals[symbol]
                        if signal.expiry > current_time
                    ]
                    
                    if not self.active_signals[symbol]:
                        del self.active_signals[symbol]
                
                await asyncio.sleep(60)  # Process every minute
                
            except Exception as e:
                logger.error(f"❌ Signal aggregator error: {e}")
                await asyncio.sleep(60)

    async def close(self):
        """Close blackboard system connections."""
        if self.db_pool:
            await self.db_pool.close()
        
        if self.redis_pool:
            await self.redis_pool.disconnect()
        
        logger.info("🏛️ Blackboard system connections closed")