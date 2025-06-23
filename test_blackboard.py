#!/usr/bin/env python3
"""
Test and Demonstrate the Agent Blackboard System

This script tests the blackboard communication system and shows you
how to peek into the data that agents are storing and sharing.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add the agents directory to the path
sys.path.insert(0, str(Path(__file__).parent / "agents"))

try:
    from blackboard import (
        blackboard, 
        write_task, 
        read_tasks, 
        update_status, 
        get_data, 
        log_performance, 
        get_context
    )
except ImportError as e:
    print(f"❌ Error importing blackboard: {e}")
    print("Make sure you're running from the TradeKnowledge directory")
    sys.exit(1)

async def test_blackboard_operations():
    """Test basic blackboard operations."""
    print("🧪 Testing Blackboard Operations")
    print("=" * 50)
    
    # Test writing tasks
    print("📝 Writing test tasks...")
    
    task1_id = await write_task(
        "RESEARCHER", 
        "technical_analysis", 
        {"symbol": "SPY", "timeframe": "1h", "indicators": ["RSI", "MACD"]},
        priority=1
    )
    print(f"   Created task: {task1_id}")
    
    task2_id = await write_task(
        "MASTERMIND",
        "architectural_analysis", 
        {"component": "risk_engine", "scope": "microservice"},
        priority=2
    )
    print(f"   Created task: {task2_id}")
    
    task3_id = await write_task(
        "EXECUTOR",
        "tdd_implementation",
        {"module": "trade_executor", "test_coverage": 90},
        priority=1
    )
    print(f"   Created task: {task3_id}")
    
    # Test reading tasks
    print("\n📖 Reading tasks by agent...")
    
    for agent in ["RESEARCHER", "MASTERMIND", "EXECUTOR"]:
        tasks = await read_tasks(agent)
        print(f"   {agent}: {len(tasks)} tasks")
        for task in tasks:
            print(f"      - {task.id}: {task.type} ({task.status})")
    
    # Test updating task status
    print("\n🔄 Updating task status...")
    await update_status(task1_id, "proc")
    print(f"   Updated {task1_id} to 'processing'")
    
    # Test getting task data
    print("\n💾 Getting task data...")
    data = await get_data(task1_id)
    print(f"   Task {task1_id} data: {data}")
    
    # Test logging performance
    print("\n📊 Logging performance metrics...")
    await log_performance("RESEARCHER", "technical_analysis", 150, 2.5)
    await log_performance("MASTERMIND", "architectural_analysis", 200, 3.1)
    await log_performance("EXECUTOR", "tdd_implementation", 100, 1.8)
    
    # Test getting context
    print("\n🎯 Getting agent contexts...")
    for agent in ["RESEARCHER", "MASTERMIND", "EXECUTOR"]:
        context = await get_context(agent)
        print(f"   {agent} context: {context}")
    
    return [task1_id, task2_id, task3_id]

def inspect_blackboard_files():
    """Inspect the blackboard files created on disk."""
    print("\n🔍 Inspecting Blackboard Files")
    print("=" * 50)
    
    blackboard_file = Path(__file__).parent / "agents" / "blackboard.md"
    cache_file = Path(__file__).parent / "agents" / "data_cache.json"
    
    if blackboard_file.exists():
        print(f"📄 Blackboard file: {blackboard_file}")
        with open(blackboard_file, 'r') as f:
            content = f.read()
        print(f"   Size: {len(content)} characters")
        print(f"   Preview:\n{content[:500]}...")
    else:
        print("📄 No blackboard.md file found")
    
    if cache_file.exists():
        print(f"\n💾 Cache file: {cache_file}")
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        print(f"   Entries: {len(cache_data)}")
        print(f"   Keys: {list(cache_data.keys())[:5]}...")
    else:
        print("\n💾 No data_cache.json file found")

def show_blackboard_structure():
    """Show the current blackboard structure."""
    print("\n🏗️ Blackboard Structure")
    print("=" * 50)
    
    print("📊 Current blackboard state:")
    print(f"   Entries: {len(blackboard.entries)}")
    print(f"   Metrics: {len(blackboard.metrics)}")
    print(f"   Cache size: {len(blackboard.data_cache)}")
    
    if blackboard.entries:
        print("\n📋 Current entries:")
        for entry in blackboard.entries[-5:]:  # Show last 5
            print(f"   {entry.id}: {entry.agent} -> {entry.type} ({entry.status})")
    
    if blackboard.metrics:
        print("\n📈 Recent metrics:")
        for metric in blackboard.metrics[-5:]:  # Show last 5
            success_icon = "✅" if metric.success else "❌"
            print(f"   {success_icon} {metric.agent} {metric.operation}: {metric.tokens_used}t, {metric.exec_time:.2f}s")

async def simulate_agent_workflow():
    """Simulate a typical agent workflow."""
    print("\n🎭 Simulating Agent Workflow")
    print("=" * 50)
    
    # Researcher finds market anomaly
    print("🔍 RESEARCHER: Analyzing market data...")
    anomaly_task = await write_task(
        "RESEARCHER",
        "market_intelligence",
        {
            "anomaly_type": "volume_spike",
            "symbol": "AAPL",
            "confidence": 0.85,
            "timeframe": "5min"
        },
        priority=1
    )
    await log_performance("RESEARCHER", "anomaly_detection", 75, 1.2, True)
    
    # Mastermind analyzes implications
    print("🧠 MASTERMIND: Analyzing trading implications...")
    await update_status(anomaly_task, "proc")
    
    strategy_task = await write_task(
        "MASTERMIND", 
        "strategic_analysis",
        {
            "anomaly_ref": anomaly_task,
            "strategy": "momentum_breakout",
            "risk_level": "medium",
            "position_size": 0.02
        },
        priority=1,
        dependencies=[anomaly_task]
    )
    await log_performance("MASTERMIND", "strategy_analysis", 120, 2.1, True)
    
    # Executor implements the strategy
    print("⚡ EXECUTOR: Implementing trading strategy...")
    await update_status(strategy_task, "proc")
    
    execution_task = await write_task(
        "EXECUTOR",
        "tdd_implementation", 
        {
            "strategy_ref": strategy_task,
            "order_type": "limit",
            "execution_algo": "twap",
            "tests_required": True
        },
        priority=1,
        dependencies=[strategy_task]
    )
    await log_performance("EXECUTOR", "strategy_execution", 95, 0.8, True)
    
    # Mark tasks as complete
    await update_status(anomaly_task, "done")
    await update_status(strategy_task, "done") 
    await update_status(execution_task, "done")
    
    print("✅ Workflow completed successfully!")
    
    return [anomaly_task, strategy_task, execution_task]

def show_usage_examples():
    """Show examples of how to inspect the blackboard."""
    print("\n📚 Usage Examples")
    print("=" * 50)
    
    print("To inspect the blackboard data:")
    print("1. Check files created by agents:")
    print("   ls -la agents/blackboard.md agents/data_cache.json")
    print()
    print("2. View blackboard markdown:")
    print("   cat agents/blackboard.md")
    print()
    print("3. Inspect cached data:")
    print("   python -c \"import json; print(json.dumps(json.load(open('agents/data_cache.json')), indent=2))\"")
    print()
    print("4. Use the InfluxDB inspector (if agents use InfluxDB):")
    print("   python inspect_blackboard.py --tasks")
    print("   python inspect_blackboard.py --metrics")
    print("   python inspect_blackboard.py --live")
    print()
    print("5. Monitor agent logs:")
    print("   tail -f agents/logs/*.log")

async def main():
    """Main demonstration function."""
    print("🚀 Agent Blackboard Inspector & Tester")
    print("=" * 60)
    
    try:
        # Test basic operations
        task_ids = await test_blackboard_operations()
        
        # Show current structure
        show_blackboard_structure()
        
        # Simulate workflow
        workflow_tasks = await simulate_agent_workflow()
        
        # Inspect files
        inspect_blackboard_files()
        
        # Show usage examples
        show_usage_examples()
        
        # Final summary
        print("\n🎯 Summary")
        print("=" * 50)
        print(f"✅ Created {len(task_ids + workflow_tasks)} test tasks")
        print(f"✅ Logged {len(blackboard.metrics)} performance metrics")
        print(f"✅ Generated blackboard with {len(blackboard.entries)} entries")
        
        blackboard_file = Path(__file__).parent / "agents" / "blackboard.md"
        if blackboard_file.exists():
            print(f"✅ Blackboard saved to: {blackboard_file}")
            print("📖 You can now view the blackboard markdown file to see agent communications!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())