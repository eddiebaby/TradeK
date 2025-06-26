#!/usr/bin/env python3
"""
Memory Initialization Script for TradeKnowledge
Minimal bootstrap to populate essential system knowledge
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.memory_manager import get_memory_manager, MemoryEvent, EntityType, RelationType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def init_system_components():
    """Initialize core system components in memory"""
    logger.info("Initializing system components...")
    
    memory = await get_memory_manager()
    
    # Core system entities
    system_events = [
        # SPARC Agents
        MemoryEvent(
            event_type="system_component",
            entity_id="agent_RESEARCHER",
            context={
                "component_type": "sparc_agent",
                "role": "Knowledge architect and intelligence synthesizer",
                "capabilities": ["market_research", "data_synthesis", "multi_source_intelligence"],
                "context_isolation": True,
                "location": "/agents/researcher/"
            },
            significance_score=1.0,
            timestamp=datetime.now()
        ),
        
        MemoryEvent(
            event_type="system_component", 
            entity_id="agent_MASTERMIND",
            context={
                "component_type": "sparc_agent",
                "role": "Strategic architect and quality orchestrator",
                "capabilities": ["strategic_analysis", "architectural_design", "quality_orchestration"],
                "context_isolation": True,
                "location": "/agents/mastermind/"
            },
            significance_score=1.0,
            timestamp=datetime.now()
        ),
        
        MemoryEvent(
            event_type="system_component",
            entity_id="agent_EXECUTOR", 
            context={
                "component_type": "sparc_agent",
                "role": "Implementation virtuoso and operational expert",
                "capabilities": ["tdd_implementation", "comprehensive_testing", "devops_automation"],
                "context_isolation": True,
                "location": "/agents/executor/"
            },
            significance_score=1.0,
            timestamp=datetime.now()
        ),
        
        # Vector Databases
        MemoryEvent(
            event_type="system_component",
            entity_id="database_qdrant",
            context={
                "component_type": "vector_database",
                "role": "Primary vector storage for semantic search",
                "capabilities": ["vector_storage", "semantic_search", "high_performance_retrieval"],
                "technology": "Qdrant",
                "related_entities": {
                    "agent_RESEARCHER": "queried_by"
                }
            },
            significance_score=0.9,
            timestamp=datetime.now()
        ),
        
        MemoryEvent(
            event_type="system_component",
            entity_id="database_chromadb",
            context={
                "component_type": "vector_database", 
                "role": "Alternative vector storage for compatibility",
                "capabilities": ["vector_storage", "document_embeddings", "compatibility_layer"],
                "technology": "ChromaDB",
                "related_entities": {
                    "database_qdrant": "alternative_to"
                }
            },
            significance_score=0.8,
            timestamp=datetime.now()
        ),
        
        # API Integrations
        MemoryEvent(
            event_type="system_component",
            entity_id="api_schwab",
            context={
                "component_type": "external_api",
                "role": "Real-time market data provider",
                "capabilities": ["market_data", "trading_execution", "account_management"],
                "provider": "Charles Schwab",
                "related_entities": {
                    "agent_RESEARCHER": "used_by"
                }
            },
            significance_score=0.9,
            timestamp=datetime.now()
        ),
        
        MemoryEvent(
            event_type="system_component",
            entity_id="api_iex_cloud",
            context={
                "component_type": "external_api",
                "role": "Market data and financial information",
                "capabilities": ["historical_data", "company_fundamentals", "market_news"],
                "provider": "IEX Cloud",
                "related_entities": {
                    "agent_RESEARCHER": "used_by"
                }
            },
            significance_score=0.8,
            timestamp=datetime.now()
        ),
        
        # Core Platform Components
        MemoryEvent(
            event_type="system_component",
            entity_id="platform_fastapi",
            context={
                "component_type": "web_framework",
                "role": "API server and endpoint management",
                "capabilities": ["rest_api", "websocket_support", "async_processing"],
                "technology": "FastAPI",
                "related_entities": {
                    "memory_middleware": "enhanced_by"
                }
            },
            significance_score=0.8,
            timestamp=datetime.now()
        ),
        
        MemoryEvent(
            event_type="system_component",
            entity_id="memory_system",
            context={
                "component_type": "knowledge_graph",
                "role": "Persistent memory and learning system",
                "capabilities": ["entity_storage", "relationship_mapping", "pattern_recognition"],
                "technology": "MCP Memory Server",
                "related_entities": {
                    "agent_RESEARCHER": "learns_from",
                    "agent_MASTERMIND": "learns_from", 
                    "agent_EXECUTOR": "learns_from"
                }
            },
            significance_score=1.0,
            timestamp=datetime.now()
        )
    ]
    
    # Store all system components
    await memory.batch_store_events(system_events)
    logger.info(f"Initialized {len(system_events)} system components")

async def init_default_user():
    """Initialize default user profile"""
    logger.info("Initializing default user...")
    
    memory = await get_memory_manager()
    
    user_event = MemoryEvent(
        event_type="user_profile",
        entity_id="user_default",
        context={
            "user_type": "default",
            "expertise_level": "intermediate",
            "preferred_analysis_types": ["technical", "fundamental"],
            "preferred_strategies": ["momentum", "value"],
            "risk_tolerance": "moderate",
            "created_at": datetime.now().isoformat(),
            "related_entities": {
                "memory_system": "tracked_by"
            }
        },
        significance_score=0.8,
        timestamp=datetime.now()
    )
    
    await memory.store_significant_event(user_event)
    logger.info("Default user profile initialized")

async def init_workflow_patterns():
    """Initialize known workflow patterns"""
    logger.info("Initializing workflow patterns...")
    
    memory = await get_memory_manager()
    
    workflow_events = [
        # Standard SPARC workflow
        MemoryEvent(
            event_type="workflow_pattern",
            entity_id="sparc_standard_workflow",
            context={
                "pattern_name": "Standard SPARC Analysis",
                "workflow_steps": ["RESEARCHER", "MASTERMIND", "EXECUTOR"],
                "description": "Sequential analysis from research to implementation",
                "use_cases": ["stock_analysis", "strategy_development", "system_implementation"],
                "estimated_duration": "300-600 seconds",
                "confidence_rating": 0.85,
                "related_entities": {
                    "agent_RESEARCHER": "starts_with",
                    "agent_MASTERMIND": "continues_with",
                    "agent_EXECUTOR": "completes_with"
                }
            },
            significance_score=0.9,
            timestamp=datetime.now()
        ),
        
        # Parallel research workflow
        MemoryEvent(
            event_type="workflow_pattern",
            entity_id="parallel_research_workflow",
            context={
                "pattern_name": "Parallel Research & Implementation",
                "workflow_steps": ["RESEARCHER+EXECUTOR", "MASTERMIND"],
                "description": "Parallel research and basic implementation, then strategic review",
                "use_cases": ["rapid_prototyping", "simple_analysis", "proof_of_concept"],
                "estimated_duration": "180-300 seconds",
                "confidence_rating": 0.75,
                "related_entities": {
                    "agent_RESEARCHER": "parallel_with",
                    "agent_EXECUTOR": "parallel_with",
                    "agent_MASTERMIND": "reviews"
                }
            },
            significance_score=0.8,
            timestamp=datetime.now()
        ),
        
        # Strategy-focused workflow
        MemoryEvent(
            event_type="workflow_pattern",
            entity_id="strategy_focused_workflow",
            context={
                "pattern_name": "Strategy-Focused Analysis", 
                "workflow_steps": ["MASTERMIND", "RESEARCHER", "EXECUTOR"],
                "description": "Strategy-first approach with targeted research and implementation",
                "use_cases": ["portfolio_optimization", "risk_management", "strategic_planning"],
                "estimated_duration": "400-800 seconds",
                "confidence_rating": 0.88,
                "related_entities": {
                    "agent_MASTERMIND": "leads",
                    "agent_RESEARCHER": "supports",
                    "agent_EXECUTOR": "implements"
                }
            },
            significance_score=0.85,
            timestamp=datetime.now()
        )
    ]
    
    await memory.batch_store_events(workflow_events)
    logger.info(f"Initialized {len(workflow_events)} workflow patterns")

async def init_financial_concepts():
    """Initialize core financial concepts and strategies"""
    logger.info("Initializing financial concepts...")
    
    memory = await get_memory_manager()
    
    concept_events = [
        # Trading Strategies
        MemoryEvent(
            event_type="trading_strategy",
            entity_id="strategy_momentum",
            context={
                "strategy_name": "Momentum Trading",
                "description": "Follow price trends and momentum indicators",
                "risk_level": "medium-high",
                "time_horizon": "short to medium term",
                "key_indicators": ["RSI", "MACD", "moving_averages"],
                "success_rate_estimate": 0.65,
                "best_markets": ["tech_stocks", "growth_stocks"],
                "related_entities": {
                    "user_default": "preferred_by"
                }
            },
            significance_score=0.8,
            timestamp=datetime.now()
        ),
        
        MemoryEvent(
            event_type="trading_strategy",
            entity_id="strategy_value",
            context={
                "strategy_name": "Value Investing",
                "description": "Identify undervalued securities based on fundamentals",
                "risk_level": "medium-low",
                "time_horizon": "long term",
                "key_indicators": ["P/E_ratio", "book_value", "debt_to_equity"],
                "success_rate_estimate": 0.72,
                "best_markets": ["dividend_stocks", "blue_chip_stocks"],
                "related_entities": {
                    "user_default": "preferred_by"
                }
            },
            significance_score=0.8,
            timestamp=datetime.now()
        ),
        
        MemoryEvent(
            event_type="trading_strategy",
            entity_id="strategy_mean_reversion",
            context={
                "strategy_name": "Mean Reversion",
                "description": "Trade based on assumption that prices return to mean",
                "risk_level": "medium",
                "time_horizon": "short term",
                "key_indicators": ["bollinger_bands", "standard_deviation", "support_resistance"],
                "success_rate_estimate": 0.58,
                "best_markets": ["range_bound_stocks", "high_volatility_stocks"],
                "related_entities": {
                    "strategy_momentum": "contrasts_with"
                }
            },
            significance_score=0.75,
            timestamp=datetime.now()
        ),
        
        # Analysis Types
        MemoryEvent(
            event_type="analysis_type",
            entity_id="analysis_technical",
            context={
                "analysis_name": "Technical Analysis",
                "description": "Price and volume pattern analysis",
                "data_requirements": ["price_history", "volume_data", "technical_indicators"],
                "time_complexity": "low",
                "accuracy_estimate": 0.68,
                "best_timeframes": ["1d", "1w", "1m"],
                "related_entities": {
                    "strategy_momentum": "supports",
                    "strategy_mean_reversion": "supports"
                }
            },
            significance_score=0.8,
            timestamp=datetime.now()
        ),
        
        MemoryEvent(
            event_type="analysis_type",
            entity_id="analysis_fundamental",
            context={
                "analysis_name": "Fundamental Analysis",
                "description": "Company financial health and valuation analysis",
                "data_requirements": ["financial_statements", "earnings_data", "industry_metrics"],
                "time_complexity": "high",
                "accuracy_estimate": 0.75,
                "best_timeframes": ["3m", "1y", "3y"],
                "related_entities": {
                    "strategy_value": "supports"
                }
            },
            significance_score=0.85,
            timestamp=datetime.now()
        )
    ]
    
    await memory.batch_store_events(concept_events)
    logger.info(f"Initialized {len(concept_events)} financial concepts")

async def verify_memory_system():
    """Verify memory system is working correctly"""
    logger.info("Verifying memory system...")
    
    try:
        memory = await get_memory_manager()
        
        # Test basic functionality
        test_event = MemoryEvent(
            event_type="system_test",
            entity_id="memory_verification_test",
            context={
                "test_type": "system_verification",
                "timestamp": datetime.now().isoformat(),
                "test_status": "running"
            },
            significance_score=0.9,
            timestamp=datetime.now()
        )
        
        await memory.store_significant_event(test_event)
        
        # Test query functionality
        if memory.mcp_available:
            # Try to search for system components
            user_context = await memory.get_user_context("default")
            sparc_insights = await memory.get_sparc_optimization_insights()
            
            logger.info(f"✅ Memory system verification successful")
            logger.info(f"   - MCP Available: {memory.mcp_available}")
            logger.info(f"   - User context loaded: {bool(user_context)}")
            logger.info(f"   - SPARC insights available: {bool(sparc_insights)}")
            
        else:
            logger.info("✅ Memory system using local fallback")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory system verification failed: {e}")
        return False

async def main():
    """Main initialization function"""
    logger.info("🚀 Starting TradeKnowledge Memory System Initialization")
    
    try:
        # Initialize components in order
        await init_system_components()
        await init_default_user()
        await init_workflow_patterns()
        await init_financial_concepts()
        
        # Verify everything is working
        verification_success = await verify_memory_system()
        
        if verification_success:
            logger.info("✅ Memory system initialization completed successfully!")
            logger.info("🧠 Knowledge graph is ready for intelligent assistance")
            
            # Print summary
            memory = await get_memory_manager()
            logger.info(f"📊 System Status:")
            logger.info(f"   - Memory Backend: {'MCP Server' if memory.mcp_available else 'Local Cache'}")
            logger.info(f"   - Significance Threshold: {memory.significance_threshold}")
            logger.info(f"   - Memory Triggers: {len(memory.memory_triggers)}")
            
        else:
            logger.error("❌ Memory system initialization completed with errors")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)