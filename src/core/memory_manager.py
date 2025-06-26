"""
Optimized Memory Manager for TradeKnowledge
Provides persistent knowledge graph capabilities with minimal overhead
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Core entity types for TradeKnowledge knowledge graph"""

    SPARC_AGENT = "sparc_agent"
    USER_PROFILE = "user_profile"
    MARKET_ANALYSIS = "market_analysis"
    TRADING_STRATEGY = "trading_strategy"
    SYSTEM_COMPONENT = "system_component"
    ANALYSIS_SESSION = "analysis_session"
    TRADE_OUTCOME = "trade_outcome"


class RelationType(Enum):
    """Core relationship types"""

    ANALYZED_BY = "analyzed_by"
    PERFORMED_FOR = "performed_for"
    USED_STRATEGY = "used_strategy"
    RESULTED_IN = "resulted_in"
    SIMILAR_TO = "similar_to"
    IMPROVES_UPON = "improves_upon"
    DEPENDS_ON = "depends_on"
    COLLABORATED_WITH = "collaborated_with"


@dataclass
class MemoryEvent:
    """Represents a significant event worth storing in memory"""

    event_type: str
    entity_id: str
    context: dict[str, Any]
    significance_score: float
    timestamp: datetime


class TradeKnowledgeMemoryManager:
    """
    Optimized memory manager with embedded schemas and smart storage
    Features:
    - Event-driven memory storage
    - Smart deduplication
    - Lazy loading
    - Batch operations
    - Progressive enhancement
    """

    def __init__(self):
        self.mcp_available = False
        self.memory_cache = {}
        self.batch_queue = []
        self.significance_threshold = 0.7

        # Memory triggers that activate storage
        self.memory_triggers = {
            "high_confidence_analysis",
            "user_asks_recommendation",
            "strategy_success",
            "strategy_failure",
            "agent_collaboration",
            "user_preference_detected",
        }

    async def initialize(self):
        """Initialize memory system and check MCP availability"""
        try:
            # Try to import and test MCP memory functions

            # Test connection
            await self._test_mcp_connection()
            self.mcp_available = True
            logger.info("MCP Memory system initialized successfully")

        except Exception as e:
            logger.warning(f"MCP Memory not available, using local cache: {e}")
            self.mcp_available = False

    async def _test_mcp_connection(self):
        """Test MCP memory connection"""
        # Try a simple search to verify connection
        try:

            # Get available MCP memory functions from globals
            available_functions = [
                name for name in globals() if name.startswith("mcp__memory__")
            ]
            if not available_functions:
                raise Exception("No MCP memory functions available")

        except Exception as e:
            raise Exception(f"MCP connection test failed: {e}")

    # Event-Driven Storage Methods

    async def store_significant_event(self, event: MemoryEvent):
        """Store event only if it meets significance threshold"""
        if event.significance_score < self.significance_threshold:
            logger.debug(
                f"Event {event.event_type} below significance threshold, skipping"
            )
            return

        # Check for duplicates using content hash
        event_hash = self._calculate_event_hash(event)
        if event_hash in self.memory_cache:
            logger.debug(f"Duplicate event detected, skipping: {event.event_type}")
            return

        self.memory_cache[event_hash] = event

        if self.mcp_available:
            await self._store_event_to_mcp(event)
        else:
            await self._store_event_locally(event)

    def _calculate_event_hash(self, event: MemoryEvent) -> str:
        """Calculate hash for deduplication"""
        content = f"{event.event_type}:{event.entity_id}:{json.dumps(event.context, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()

    async def _store_event_to_mcp(self, event: MemoryEvent):
        """Store event to MCP memory system"""
        try:
            # Create entity for the event
            entity_name = f"{event.entity_id}_{event.event_type}_{event.timestamp.strftime('%Y%m%d_%H%M%S')}"

            entities = [
                {
                    "name": entity_name,
                    "entityType": event.event_type,
                    "observations": [
                        f"Event type: {event.event_type}",
                        f"Significance score: {event.significance_score}",
                        f"Context: {json.dumps(event.context)}",
                        f"Timestamp: {event.timestamp.isoformat()}",
                    ],
                }
            ]

            # Use the create_entities function
            await self._call_mcp_function("create_entities", {"entities": entities})

            # Create relationships if context contains related entities
            if "related_entities" in event.context:
                relations = []
                for related_entity, relation_type in event.context[
                    "related_entities"
                ].items():
                    relations.append(
                        {
                            "from": entity_name,
                            "to": related_entity,
                            "relationType": relation_type,
                        }
                    )

                if relations:
                    await self._call_mcp_function(
                        "create_relations", {"relations": relations}
                    )

            logger.debug(f"Stored event to MCP: {entity_name}")

        except Exception as e:
            logger.error(f"Failed to store event to MCP: {e}")
            await self._store_event_locally(event)

    async def _store_event_locally(self, event: MemoryEvent):
        """Fallback local storage"""
        # Simple local storage implementation
        local_storage_path = "/tmp/tradeknowledge_memory.json"
        try:
            # Load existing data
            try:
                with open(local_storage_path) as f:
                    data = json.load(f)
            except FileNotFoundError:
                data = {"events": [], "entities": {}, "relations": []}

            # Add new event
            event_data = {
                "event_type": event.event_type,
                "entity_id": event.entity_id,
                "context": event.context,
                "significance_score": event.significance_score,
                "timestamp": event.timestamp.isoformat(),
            }
            data["events"].append(event_data)

            # Save back
            with open(local_storage_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Stored event locally: {event.event_type}")

        except Exception as e:
            logger.error(f"Failed to store event locally: {e}")

    async def _call_mcp_function(self, function_name: str, params: dict[str, Any]):
        """Dynamically call MCP memory functions"""
        try:
            # Import the specific function
            function_module = f"mcp__memory__{function_name}"
            if function_module in globals():
                func = globals()[function_module]
                return await func(**params)
            else:
                raise Exception(f"Function {function_name} not available")
        except Exception as e:
            logger.error(f"Failed to call MCP function {function_name}: {e}")
            raise

    # High-Level Storage Methods (Triggered by Decorators/Middleware)

    async def store_analysis_result(
        self,
        symbol: str,
        analysis_type: str,
        results: dict[str, Any],
        confidence: float,
        user_id: str | None = None,
    ):
        """Store significant analysis results"""

        # Only store high-confidence or interesting results
        significance = confidence
        if results.get("outcome") == "profitable":
            significance += 0.2
        if results.get("strategy_type") in [
            "momentum",
            "breakout",
        ]:  # High-value strategies
            significance += 0.1

        event = MemoryEvent(
            event_type="market_analysis",
            entity_id=f"{symbol}_{analysis_type}",
            context={
                "symbol": symbol,
                "analysis_type": analysis_type,
                "confidence": confidence,
                "outcome": results.get("outcome"),
                "strategy_type": results.get("strategy_type"),
                "user_id": user_id,
                "related_entities": (
                    {f"user_{user_id}": "performed_for"} if user_id else {}
                ),
            },
            significance_score=significance,
            timestamp=datetime.now(),
        )

        await self.store_significant_event(event)

    async def store_sparc_collaboration(
        self,
        agents: list[str],
        task_type: str,
        outcome_quality: float,
        duration_seconds: int,
    ):
        """Store SPARC agent collaboration patterns"""

        significance = outcome_quality
        if duration_seconds < 60:  # Fast collaboration
            significance += 0.1

        event = MemoryEvent(
            event_type="sparc_collaboration",
            entity_id=f"{'_'.join(agents)}_{task_type}",
            context={
                "agents": agents,
                "task_type": task_type,
                "outcome_quality": outcome_quality,
                "duration_seconds": duration_seconds,
                "workflow_pattern": " -> ".join(agents),
                "related_entities": dict.fromkeys(agents, "collaborated_with"),
            },
            significance_score=significance,
            timestamp=datetime.now(),
        )

        await self.store_significant_event(event)

    async def store_user_preference(
        self,
        user_id: str,
        preference_type: str,
        preference_value: Any,
        confidence: float = 0.8,
    ):
        """Store detected user preferences"""

        event = MemoryEvent(
            event_type="user_preference",
            entity_id=f"user_{user_id}_{preference_type}",
            context={
                "user_id": user_id,
                "preference_type": preference_type,
                "preference_value": preference_value,
                "detection_confidence": confidence,
            },
            significance_score=confidence,
            timestamp=datetime.now(),
        )

        await self.store_significant_event(event)

    # Smart Query Methods (Lazy Loading)

    async def get_user_context(self, user_id: str) -> dict[str, Any]:
        """Get user context with lazy loading"""

        if not self.mcp_available:
            return await self._get_user_context_local(user_id)

        try:
            # Level 1: Quick check if user has any stored preferences
            user_nodes = await self._call_mcp_function(
                "search_nodes", {"query": f"user_{user_id} preference"}
            )

            if not user_nodes.get("entities"):
                return {"user_id": user_id, "preferences": {}, "history": []}

            # Level 2: Load basic preference summary
            preferences = {}
            for node in user_nodes["entities"]:
                for obs in node.get("observations", []):
                    if "preference_type:" in obs:
                        pref_type = (
                            obs.split("preference_type:")[1].split(",")[0].strip()
                        )
                        pref_value = (
                            obs.split("preference_value:")[1].split(",")[0].strip()
                            if "preference_value:" in obs
                            else "unknown"
                        )
                        preferences[pref_type] = pref_value

            return {
                "user_id": user_id,
                "preferences": preferences,
                "has_detailed_history": len(user_nodes["entities"]) > 5,
            }

        except Exception as e:
            logger.error(f"Failed to get user context: {e}")
            return {"user_id": user_id, "preferences": {}, "history": []}

    async def get_strategy_recommendations(
        self, symbol: str, user_id: str
    ) -> list[dict[str, Any]]:
        """Get strategy recommendations with smart caching"""

        if not self.mcp_available:
            return await self._get_strategy_recommendations_local(symbol, user_id)

        try:
            # Quick search for successful strategies with this symbol
            strategy_nodes = await self._call_mcp_function(
                "search_nodes", {"query": f"market_analysis {symbol} profitable"}
            )

            recommendations = []
            for node in strategy_nodes.get("entities", []):
                strategy_info = {}
                for obs in node.get("observations", []):
                    if "strategy_type:" in obs:
                        strategy_info["strategy"] = obs.split("strategy_type:")[
                            1
                        ].strip()
                    elif "confidence:" in obs:
                        strategy_info["confidence"] = float(
                            obs.split("confidence:")[1].strip()
                        )

                if (
                    strategy_info.get("strategy")
                    and strategy_info.get("confidence", 0) > 0.7
                ):
                    recommendations.append(strategy_info)

            # Sort by confidence and return top 3
            return sorted(
                recommendations, key=lambda x: x.get("confidence", 0), reverse=True
            )[:3]

        except Exception as e:
            logger.error(f"Failed to get strategy recommendations: {e}")
            return []

    async def get_sparc_optimization_insights(self) -> dict[str, Any]:
        """Get SPARC collaboration optimization insights"""

        if not self.mcp_available:
            return {"optimal_workflows": [], "bottlenecks": []}

        try:
            # Search for collaboration patterns
            collab_nodes = await self._call_mcp_function(
                "search_nodes", {"query": "sparc_collaboration"}
            )

            workflow_performance = {}
            for node in collab_nodes.get("entities", []):
                workflow_pattern = None
                outcome_quality = 0

                for obs in node.get("observations", []):
                    if "workflow_pattern:" in obs:
                        workflow_pattern = obs.split("workflow_pattern:")[1].strip()
                    elif "outcome_quality:" in obs:
                        outcome_quality = float(
                            obs.split("outcome_quality:")[1].strip()
                        )

                if workflow_pattern and outcome_quality > 0:
                    if workflow_pattern not in workflow_performance:
                        workflow_performance[workflow_pattern] = []
                    workflow_performance[workflow_pattern].append(outcome_quality)

            # Calculate averages and identify best patterns
            optimal_workflows = []
            for pattern, qualities in workflow_performance.items():
                avg_quality = sum(qualities) / len(qualities)
                if (
                    avg_quality > 0.8 and len(qualities) >= 3
                ):  # High quality, sufficient samples
                    optimal_workflows.append(
                        {
                            "pattern": pattern,
                            "avg_quality": avg_quality,
                            "sample_count": len(qualities),
                        }
                    )

            return {
                "optimal_workflows": sorted(
                    optimal_workflows, key=lambda x: x["avg_quality"], reverse=True
                ),
                "total_collaborations": sum(
                    len(q) for q in workflow_performance.values()
                ),
            }

        except Exception as e:
            logger.error(f"Failed to get SPARC insights: {e}")
            return {"optimal_workflows": [], "bottlenecks": []}

    # Local fallback methods

    async def _get_user_context_local(self, user_id: str) -> dict[str, Any]:
        """Local fallback for user context"""
        return {
            "user_id": user_id,
            "preferences": {},
            "history": [],
            "source": "local_fallback",
        }

    async def _get_strategy_recommendations_local(
        self, symbol: str, user_id: str
    ) -> list[dict[str, Any]]:
        """Local fallback for strategy recommendations"""
        return [
            {"strategy": "momentum", "confidence": 0.75, "source": "local_fallback"}
        ]

    # Batch Operations

    async def batch_store_events(self, events: list[MemoryEvent]):
        """Store multiple events in batch for efficiency"""
        significant_events = [
            e for e in events if e.significance_score >= self.significance_threshold
        ]

        if not significant_events:
            return

        if self.mcp_available:
            await self._batch_store_to_mcp(significant_events)
        else:
            for event in significant_events:
                await self._store_event_locally(event)

    async def _batch_store_to_mcp(self, events: list[MemoryEvent]):
        """Batch store to MCP for efficiency"""
        try:
            # Prepare entities and relations
            entities = []
            relations = []

            for event in events:
                entity_name = f"{event.entity_id}_{event.event_type}_{event.timestamp.strftime('%Y%m%d_%H%M%S')}"

                entities.append(
                    {
                        "name": entity_name,
                        "entityType": event.event_type,
                        "observations": [
                            f"Event type: {event.event_type}",
                            f"Significance score: {event.significance_score}",
                            f"Context: {json.dumps(event.context)}",
                            f"Timestamp: {event.timestamp.isoformat()}",
                        ],
                    }
                )

                # Add relations from context
                if "related_entities" in event.context:
                    for related_entity, relation_type in event.context[
                        "related_entities"
                    ].items():
                        relations.append(
                            {
                                "from": entity_name,
                                "to": related_entity,
                                "relationType": relation_type,
                            }
                        )

            # Batch create
            if entities:
                await self._call_mcp_function("create_entities", {"entities": entities})
            if relations:
                await self._call_mcp_function(
                    "create_relations", {"relations": relations}
                )

            logger.info(f"Batch stored {len(events)} events to MCP")

        except Exception as e:
            logger.error(f"Batch store to MCP failed: {e}")
            # Fallback to individual local storage
            for event in events:
                await self._store_event_locally(event)


# Singleton instance
_memory_manager: TradeKnowledgeMemoryManager | None = None


async def get_memory_manager() -> TradeKnowledgeMemoryManager:
    """Get or create memory manager singleton"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = TradeKnowledgeMemoryManager()
        await _memory_manager.initialize()
    return _memory_manager


# Convenience decorator for memory-aware functions
def memory_aware(trigger_type: str = "general", significance: float = 0.8):
    """Decorator to automatically capture function results in memory"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)

            # Extract context from function arguments and results
            context = {
                "function_name": func.__name__,
                "args": str(args)[:200],  # Truncate for efficiency
                "kwargs": {
                    k: str(v)[:100] for k, v in kwargs.items()
                },  # Truncate values
                "result_type": type(result).__name__,
            }

            # Create memory event
            memory = await get_memory_manager()
            event = MemoryEvent(
                event_type=trigger_type,
                entity_id=f"{func.__name__}_{datetime.now().strftime('%Y%m%d_%H%M')}",
                context=context,
                significance_score=significance,
                timestamp=datetime.now(),
            )

            # Store asynchronously to not block main function
            asyncio.create_task(memory.store_significant_event(event))

            return result

        return wrapper

    return decorator
