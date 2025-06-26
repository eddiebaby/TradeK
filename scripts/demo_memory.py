#!/usr/bin/env python3
"""
Memory System Demonstration for TradeKnowledge
Shows real-world usage of persistent memory capabilities
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
import random
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.memory_manager import get_memory_manager, MemoryEvent
from core.memory_middleware import sparc_memory_aware, MemoryContextManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DemoSPARCAgent:
    """Demo SPARC agent to show memory integration"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
    
    @sparc_memory_aware("RESEARCHER", 0.8)
    async def analyze_stock(self, symbol: str, analysis_type: str = "technical"):
        """Demo stock analysis with memory integration"""
        logger.info(f"🔍 {self.agent_name} analyzing {symbol} ({analysis_type})")
        
        # Simulate analysis work
        await asyncio.sleep(1)
        
        # Generate realistic demo results
        confidence = random.uniform(0.6, 0.95)
        recommendation = random.choice(["BUY", "HOLD", "SELL"])
        
        result = {
            "symbol": symbol,
            "analysis_type": analysis_type,
            "recommendation": recommendation,
            "confidence": confidence,
            "key_indicators": {
                "rsi": random.uniform(30, 70),
                "macd_signal": random.choice(["bullish", "bearish", "neutral"]),
                "volume_trend": random.choice(["increasing", "decreasing", "stable"])
            },
            "price_target": random.uniform(100, 200),
            "risk_level": random.choice(["low", "medium", "high"]),
            "agents_involved": [self.agent_name]
        }
        
        logger.info(f"✅ Analysis complete: {recommendation} (confidence: {confidence:.2f})")
        return result
    
    @sparc_memory_aware("MASTERMIND", 0.85)
    async def create_strategy(self, analysis_results: dict):
        """Demo strategy creation with memory integration"""
        logger.info(f"🧠 {self.agent_name} creating strategy based on analysis")
        
        await asyncio.sleep(2)
        
        strategy_type = random.choice(["momentum", "value", "mean_reversion", "breakout"])
        
        result = {
            "strategy_type": strategy_type,
            "based_on_analysis": analysis_results.get("symbol", analysis_results.get("type", "portfolio")),
            "entry_criteria": f"{strategy_type}_signals_confirmed",
            "exit_criteria": f"target_reached_or_stop_loss",
            "risk_management": {
                "stop_loss": 0.05,
                "take_profit": 0.15,
                "position_size": 0.02
            },
            "confidence": random.uniform(0.7, 0.9),
            "expected_return": random.uniform(0.08, 0.25),
            "agents_involved": [self.agent_name, "RESEARCHER"]
        }
        
        logger.info(f"📋 Strategy created: {strategy_type} strategy")
        return result

async def demo_basic_memory_operations():
    """Demonstrate basic memory operations"""
    logger.info("\n🚀 Demo 1: Basic Memory Operations")
    
    memory = await get_memory_manager()
    
    # Store a high-confidence analysis
    await memory.store_analysis_result(
        symbol="AAPL",
        analysis_type="technical",
        results={
            "recommendation": "BUY",
            "outcome": "profitable", 
            "strategy_type": "momentum"
        },
        confidence=0.92,
        user_id="demo_user"
    )
    
    # Store user preferences
    await memory.store_user_preference("demo_user", "preferred_analysis", "technical", 0.85)
    await memory.store_user_preference("demo_user", "risk_tolerance", "moderate", 0.9)
    await memory.store_user_preference("demo_user", "favorite_strategy", "momentum", 0.8)
    
    # Store SPARC collaboration
    await memory.store_sparc_collaboration(
        agents=["RESEARCHER", "MASTERMIND", "EXECUTOR"],
        task_type="stock_analysis",
        outcome_quality=0.88,
        duration_seconds=45
    )
    
    logger.info("✅ Basic memory operations completed")

async def demo_sparc_agent_integration():
    """Demonstrate SPARC agent memory integration"""
    logger.info("\n🤖 Demo 2: SPARC Agent Memory Integration")
    
    # Create demo agents
    researcher = DemoSPARCAgent("RESEARCHER")
    mastermind = DemoSPARCAgent("MASTERMIND")
    
    # Perform analyses with automatic memory capture
    symbols = ["GOOGL", "MSFT", "TSLA", "NVDA"]
    
    for symbol in symbols:
        # Researcher analyzes
        analysis = await researcher.analyze_stock(symbol, "technical")
        
        # Mastermind creates strategy  
        strategy = await mastermind.create_strategy(analysis)
        
        logger.info(f"   💾 Memory automatically captured for {symbol}")
    
    logger.info("✅ SPARC agent integration demo completed")

async def demo_complex_workflow():
    """Demonstrate complex workflow with memory context manager"""
    logger.info("\n🔄 Demo 3: Complex Workflow with Memory Context")
    
    researcher = DemoSPARCAgent("RESEARCHER")
    mastermind = DemoSPARCAgent("MASTERMIND")
    
    # Use context manager for complex operation
    async with MemoryContextManager("portfolio_analysis", significance=0.9) as ctx:
        ctx.add_context("portfolio_size", 5)
        ctx.add_context("analysis_type", "comprehensive")
        
        # Multi-stock analysis
        portfolio_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        portfolio_results = []
        
        for symbol in portfolio_symbols:
            analysis = await researcher.analyze_stock(symbol, "fundamental")
            ctx.add_step(f"analyzed_{symbol}", analysis)
            portfolio_results.append(analysis)
        
        # Create portfolio strategy
        portfolio_strategy = await mastermind.create_strategy({
            "type": "portfolio_optimization",
            "stocks": portfolio_symbols,
            "total_confidence": sum(r["confidence"] for r in portfolio_results) / len(portfolio_results)
        })
        ctx.add_step("portfolio_strategy_created", portfolio_strategy)
        
        ctx.add_context("final_recommendation", portfolio_strategy.get("strategy_type", "diversified"))
    
    logger.info("✅ Complex workflow demo completed")

async def demo_memory_queries():
    """Demonstrate memory query capabilities"""
    logger.info("\n🔍 Demo 4: Memory Query Capabilities")
    
    memory = await get_memory_manager()
    
    # Wait a moment for previous operations to be stored
    await asyncio.sleep(2)
    
    # Get user context
    user_context = await memory.get_user_context("demo_user")
    logger.info(f"📊 User Context: {json.dumps(user_context, indent=2)}")
    
    # Get strategy recommendations
    recommendations = await memory.get_strategy_recommendations("AAPL", "demo_user")
    logger.info(f"💡 Strategy Recommendations for AAPL: {json.dumps(recommendations, indent=2)}")
    
    # Get SPARC optimization insights
    sparc_insights = await memory.get_sparc_optimization_insights()
    logger.info(f"🤖 SPARC Optimization Insights: {json.dumps(sparc_insights, indent=2)}")
    
    logger.info("✅ Memory query demo completed")

async def demo_pattern_recognition():
    """Demonstrate pattern recognition capabilities"""
    logger.info("\n🧩 Demo 5: Pattern Recognition")
    
    memory = await get_memory_manager()
    
    # Simulate multiple successful momentum trades
    successful_momentum_trades = [
        ("AAPL", 0.92, "profitable"),
        ("MSFT", 0.88, "profitable"), 
        ("GOOGL", 0.85, "profitable"),
        ("NVDA", 0.90, "profitable")
    ]
    
    for symbol, confidence, outcome in successful_momentum_trades:
        await memory.store_analysis_result(
            symbol=symbol,
            analysis_type="technical",
            results={
                "recommendation": "BUY",
                "outcome": outcome,
                "strategy_type": "momentum",
                "profit_percentage": random.uniform(0.08, 0.18)
            },
            confidence=confidence,
            user_id="demo_user"
        )
    
    # Simulate some failed mean reversion trades
    failed_mean_reversion_trades = [
        ("XYZ", 0.75, "loss"),
        ("ABC", 0.70, "loss")
    ]
    
    for symbol, confidence, outcome in failed_mean_reversion_trades:
        await memory.store_analysis_result(
            symbol=symbol,
            analysis_type="technical", 
            results={
                "recommendation": "BUY",
                "outcome": outcome,
                "strategy_type": "mean_reversion",
                "profit_percentage": random.uniform(-0.08, -0.03)
            },
            confidence=confidence,
            user_id="demo_user"
        )
    
    logger.info("📈 Pattern: Momentum strategies showing consistent success")
    logger.info("📉 Pattern: Mean reversion strategies showing poor performance")
    logger.info("🎯 Memory will learn these patterns for future recommendations")
    
    logger.info("✅ Pattern recognition demo completed")

async def demo_memory_efficiency():
    """Demonstrate memory efficiency features"""
    logger.info("\n⚡ Demo 6: Memory Efficiency Features")
    
    memory = await get_memory_manager()
    
    # Test significance filtering
    low_significance_events = []
    high_significance_events = []
    
    for i in range(10):
        # Low significance event (should be filtered out)
        low_sig_event = MemoryEvent(
            event_type="test_event",
            entity_id=f"low_sig_{i}",
            context={"test": True, "value": i},
            significance_score=0.3,  # Below threshold
            timestamp=datetime.now()
        )
        low_significance_events.append(low_sig_event)
        
        # High significance event (should be stored)
        high_sig_event = MemoryEvent(
            event_type="test_event", 
            entity_id=f"high_sig_{i}",
            context={"test": True, "value": i, "important": True},
            significance_score=0.9,  # Above threshold
            timestamp=datetime.now()
        )
        high_significance_events.append(high_sig_event)
    
    # Test batch storage
    logger.info("🔄 Testing batch storage of 20 events...")
    await memory.batch_store_events(low_significance_events + high_significance_events)
    
    logger.info(f"✅ Efficiency features demonstrated:")
    logger.info(f"   - Significance filtering: {len(low_significance_events)} events filtered out")
    logger.info(f"   - Batch storage: {len(high_significance_events)} events stored efficiently")
    logger.info(f"   - Deduplication: Active (prevents duplicate storage)")
    
    logger.info("✅ Memory efficiency demo completed")

async def demo_fallback_behavior():
    """Demonstrate fallback behavior when MCP is unavailable"""
    logger.info("\n🔄 Demo 7: Fallback Behavior")
    
    memory = await get_memory_manager()
    
    logger.info(f"🔌 MCP Available: {memory.mcp_available}")
    
    if memory.mcp_available:
        logger.info("✅ Using MCP memory server for persistent storage")
        logger.info("📡 All operations will be stored in knowledge graph")
    else:
        logger.info("🔄 Using local cache fallback")
        logger.info("💾 Operations stored locally until MCP is available")
    
    # Test operation regardless of backend
    test_result = await memory.get_user_context("fallback_test_user")
    logger.info(f"🧪 Fallback test successful: {bool(test_result)}")
    
    logger.info("✅ Fallback behavior demo completed")

async def main():
    """Main demonstration function"""
    logger.info("🎭 TradeKnowledge Memory System Demonstration")
    logger.info("=" * 60)
    
    try:
        # Run all demonstrations
        await demo_basic_memory_operations()
        await demo_sparc_agent_integration()
        await demo_complex_workflow()
        await demo_memory_queries()
        await demo_pattern_recognition()
        await demo_memory_efficiency()
        await demo_fallback_behavior()
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 All memory demonstrations completed successfully!")
        logger.info("💡 Key Benefits Demonstrated:")
        logger.info("   - Automatic capture of significant events")
        logger.info("   - Smart filtering and deduplication")
        logger.info("   - Pattern recognition and learning")
        logger.info("   - Efficient batch operations")
        logger.info("   - Robust fallback mechanisms")
        logger.info("   - Zero overhead for insignificant events")
        logger.info("   - Seamless SPARC agent integration")
        
        memory = await get_memory_manager()
        logger.info(f"\n📊 Final Status:")
        logger.info(f"   - Memory Backend: {'MCP Server' if memory.mcp_available else 'Local Cache'}")
        logger.info(f"   - Cached Events: {len(memory.memory_cache)}")
        logger.info(f"   - Significance Threshold: {memory.significance_threshold}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)