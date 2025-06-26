#!/usr/bin/env python3
"""
Demo: Enhanced Agent Blackboard with InfluxDB 2.7
Token-Optimized Inter-Agent Communication & Self-Reflection System

This demo showcases:
- Token-first agent communication
- Self-improving agent interactions
- Real-time monitoring and optimization
- SPARC framework integration
"""

import asyncio
import time
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Import our blackboard system
from influx_blackboard import get_blackboard, write_task, read_tasks, update_status, log_performance
from enhanced_agent_base import EnhancedAgentBase, AgentCapability, TaskResult, track_tokens

class DemoResearcherAgent(EnhancedAgentBase):
    """Demo Researcher Agent with token optimization"""
    
    def __init__(self):
        capabilities = [
            AgentCapability("intelligence_gathering", "Gather market and security intelligence", 200),
            AgentCapability("security_analysis", "Analyze security threats and vulnerabilities", 250),
            AgentCapability("technical_analysis", "Perform technical market analysis", 180)
        ]
        super().__init__("Researcher", capabilities)
    
    @track_tokens("research_analysis")
    async def process_task(self, task_data: Dict[str, Any]) -> TaskResult:
        """Process research tasks with token optimization"""
        
        # Simulate research work
        await asyncio.sleep(0.5)  # Simulate processing time
        
        # Extract task type and data
        task_type = task_data.get("task_type", "general_research")
        research_topic = task_data.get("topic", "general")
        
        # Simulate different types of research
        if "technical_analysis" in task_type:
            result = await self._perform_technical_analysis(task_data)
        elif "security" in task_type:
            result = await self._perform_security_analysis(task_data)
        else:
            result = await self._perform_general_research(task_data)
        
        # Create handoff to Mastermind if needed
        if result["confidence"] > 0.8:
            await self.handoff_to_agent("Mastermind", {
                "research_findings": result["analysis"],
                "confidence": result["confidence"],
                "source_topic": research_topic
            })
        
        return TaskResult(
            success=True,
            data=result,
            tokens_used=self._estimate_tokens(str(result)),
            exec_time=0,  # Will be calculated by tracker
            confidence=result["confidence"]
        )
    
    async def _perform_technical_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate technical analysis"""
        symbol = data.get("symbol", "BTC/USD")
        timeframe = data.get("timeframe", "1h")
        
        return {
            "type": "technical_analysis",
            "symbol": symbol,
            "timeframe": timeframe,
            "analysis": f"Technical analysis for {symbol} on {timeframe} shows bullish momentum",
            "indicators": ["RSI: 65", "MACD: Bullish crossover", "Volume: Increasing"],
            "confidence": 0.85,
            "recommendation": "Consider long positions"
        }
    
    async def _perform_security_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate security analysis"""
        return {
            "type": "security_analysis",
            "threat_level": "Low",
            "analysis": "No major security threats identified in current market conditions",
            "recommendations": ["Monitor regulatory developments", "Watch for unusual trading patterns"],
            "confidence": 0.9
        }
    
    async def _perform_general_research(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate general research"""
        topic = data.get("topic", "market_conditions")
        
        return {
            "type": "general_research",
            "topic": topic,
            "analysis": f"Research on {topic} shows positive trends with moderate volatility",
            "key_findings": ["Market sentiment is positive", "Volume is above average", "No major news events"],
            "confidence": 0.75
        }

class DemoMastermindAgent(EnhancedAgentBase):
    """Demo Mastermind Agent for strategic analysis"""
    
    def __init__(self):
        capabilities = [
            AgentCapability("strategic_analysis", "Perform strategic analysis and planning", 300),
            AgentCapability("architectural_design", "Design system architecture", 350),
            AgentCapability("quality_strategy", "Develop quality assurance strategies", 250)
        ]
        super().__init__("Mastermind", capabilities)
    
    @track_tokens("strategic_planning")
    async def process_task(self, task_data: Dict[str, Any]) -> TaskResult:
        """Process strategic tasks"""
        
        await asyncio.sleep(1.0)  # Simulate more complex processing
        
        # Extract research findings if available
        research_findings = task_data.get("research_findings", "")
        confidence = task_data.get("confidence", 0.5)
        
        # Perform strategic analysis
        strategy = await self._develop_strategy(task_data)
        
        # Create implementation plan for Executor
        if strategy["priority"] == "high":
            await self.handoff_to_agent("Executor", {
                "strategy": strategy,
                "implementation_priority": strategy["priority"],
                "requirements": strategy["requirements"]
            })
        
        return TaskResult(
            success=True,
            data=strategy,
            tokens_used=self._estimate_tokens(str(strategy)),
            exec_time=0,
            confidence=confidence * 1.1  # Mastermind adds strategic confidence
        )
    
    async def _develop_strategy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Develop strategic plan"""
        research_findings = data.get("research_findings", "")
        
        return {
            "type": "strategic_plan",
            "analysis": f"Strategic analysis based on: {research_findings}",
            "strategy": "Implement gradual scaling approach with risk management",
            "priority": "high",
            "requirements": [
                "Robust error handling",
                "Real-time monitoring",
                "Scalable architecture"
            ],
            "timeline": "2-3 sprints",
            "risk_assessment": "Medium risk, high reward potential"
        }

class DemoExecutorAgent(EnhancedAgentBase):
    """Demo Executor Agent for implementation"""
    
    def __init__(self):
        capabilities = [
            AgentCapability("implementation", "Implement features and solutions", 280),
            AgentCapability("testing", "Create and run comprehensive tests", 220),
            AgentCapability("deployment", "Deploy and monitor systems", 200)
        ]
        super().__init__("Executor", capabilities)
    
    @track_tokens("implementation")
    async def process_task(self, task_data: Dict[str, Any]) -> TaskResult:
        """Process implementation tasks"""
        
        await asyncio.sleep(1.5)  # Simulate implementation time
        
        strategy = task_data.get("strategy", {})
        priority = task_data.get("implementation_priority", "medium")
        
        # Implement based on strategy
        implementation = await self._implement_solution(task_data)
        
        return TaskResult(
            success=True,
            data=implementation,
            tokens_used=self._estimate_tokens(str(implementation)),
            exec_time=0,
            confidence=0.95  # High confidence in implementation
        )
    
    async def _implement_solution(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Implement the solution"""
        strategy = data.get("strategy", {})
        
        return {
            "type": "implementation",
            "feature": "Token-optimized trading system",
            "implementation_details": f"Implemented based on strategy: {strategy.get('strategy', 'default')}",
            "tests_created": ["Unit tests", "Integration tests", "Performance tests"],
            "deployment_status": "Ready for deployment",
            "performance_metrics": {
                "response_time": "50ms",
                "throughput": "1000 TPS",
                "memory_usage": "< 100MB"
            }
        }

async def run_demo_workflow():
    """Run complete SPARC workflow demo"""
    print("🚀 Starting Enhanced Agent Blackboard Demo")
    print("=" * 60)
    
    # Initialize agents
    researcher = DemoResearcherAgent()
    mastermind = DemoMastermindAgent()
    executor = DemoExecutorAgent()
    
    print("🤖 Agents initialized:")
    print(f"   - {researcher.agent_name} ({len(researcher.capabilities)} capabilities)")
    print(f"   - {mastermind.agent_name} ({len(mastermind.capabilities)} capabilities)")
    print(f"   - {executor.agent_name} ({len(executor.capabilities)} capabilities)")
    print()
    
    try:
        # Step 1: Create initial research task
        print("📋 Step 1: Creating research task...")
        research_task_data = {
            "task_type": "technical_analysis",
            "topic": "cryptocurrency_market_analysis",
            "symbol": "BTC/USD",
            "timeframe": "1h",
            "urgency": "high"
        }
        
        research_task_id = await write_task("Researcher", "technical_analysis", research_task_data, priority=1)
        print(f"   Created research task: {research_task_id}")
        
        # Step 2: Process research task
        print("\n🔍 Step 2: Processing research task...")
        result = await researcher.execute_with_tracking("demo_research", researcher.process_task, research_task_data)
        
        if result.success:
            print(f"   ✅ Research completed - Tokens: {result.tokens_used}, Time: {result.exec_time:.2f}s")
            print(f"   📊 Confidence: {result.confidence:.1%}")
        else:
            print(f"   ❌ Research failed: {result.error_message}")
        
        # Step 3: Wait for strategic analysis (simulated handoff)
        print("\n🧠 Step 3: Strategic analysis...")
        await asyncio.sleep(1)  # Brief delay to simulate async handoff
        
        # Check for tasks created by Researcher
        mastermind_tasks = await read_tasks("Mastermind", status="new")
        if mastermind_tasks:
            print(f"   Found {len(mastermind_tasks)} task(s) for Mastermind")
            
            # Process first task
            task_data = await get_blackboard().get_task_data(mastermind_tasks[0]['id'])
            result = await mastermind.execute_with_tracking("demo_strategy", mastermind.process_task, task_data)
            
            if result.success:
                print(f"   ✅ Strategy completed - Tokens: {result.tokens_used}, Time: {result.exec_time:.2f}s")
            
        # Step 4: Implementation
        print("\n⚡ Step 4: Implementation...")
        await asyncio.sleep(1)
        
        executor_tasks = await read_tasks("Executor", status="new")
        if executor_tasks:
            print(f"   Found {len(executor_tasks)} task(s) for Executor")
            
            # Process implementation task
            task_data = await get_blackboard().get_task_data(executor_tasks[0]['id'])
            result = await executor.execute_with_tracking("demo_implementation", executor.process_task, task_data)
            
            if result.success:
                print(f"   ✅ Implementation completed - Tokens: {result.tokens_used}, Time: {result.exec_time:.2f}s")
        
        # Step 5: Display final statistics
        print("\n📊 Step 5: Final Statistics")
        print("-" * 40)
        
        for agent in [researcher, mastermind, executor]:
            status = await agent.get_agent_status()
            print(f"{agent.agent_name}:")
            print(f"   Operations: {status['operations_completed']}")
            print(f"   Success Rate: {status['success_rate']:.1%}")
            print(f"   Total Tokens: {status['total_tokens_used']}")
            print(f"   Avg Tokens/Op: {status['avg_tokens_per_operation']:.1f}")
            print()
        
        # Generate efficiency report
        print("📈 Efficiency Report")
        print("-" * 40)
        
        blackboard = get_blackboard()
        report = await blackboard.generate_efficiency_report(period_hours=1)
        
        if "error" not in report:
            print(f"Total System Tokens: {report.get('total_tokens', 0):,}")
            print(f"Total Operations: {report.get('total_operations', 0)}")
            
            if report.get('total_operations', 0) > 0:
                efficiency = report['total_tokens'] / report['total_operations']
                print(f"System Efficiency: {efficiency:.1f} tokens/operation")
        
        print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Shutdown agents
        await researcher.shutdown()
        await mastermind.shutdown()
        await executor.shutdown()

async def run_monitoring_demo():
    """Run monitoring demo"""
    print("🖥️  Starting Monitoring Demo")
    print("=" * 40)
    
    # Import monitoring
    from monitoring.blackboard_monitor import BlackboardMonitor
    
    monitor = BlackboardMonitor()
    
    print("Monitoring for 30 seconds...")
    try:
        # Run monitoring for a short period
        await asyncio.wait_for(monitor.start_monitoring(interval=5), timeout=30)
    except asyncio.TimeoutError:
        print("✅ Monitoring demo completed")
    finally:
        monitor.stop_monitoring()
        monitor.blackboard.close()

async def main():
    """Main demo function"""
    print("🏗️  Enhanced Agent Blackboard Demo")
    print("Token-Optimized Inter-Agent Communication System")
    print("=" * 80)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "monitor":
            await run_monitoring_demo()
            return
    
    # Check if InfluxDB is available
    blackboard = get_blackboard()
    if not blackboard.client:
        print("⚠️  InfluxDB not available - demo will run with limited functionality")
        print("   To enable full functionality:")
        print("   1. Run: python scripts/setup_blackboard_influxdb.py")
        print("   2. Run: ./start_blackboard.sh")
        print()
    
    await run_demo_workflow()
    
    print("\n💡 Next Steps:")
    print("   - Run monitoring: python demo_influx_blackboard.py monitor")
    print("   - Check efficiency: python monitoring/blackboard_monitor.py report")
    print("   - Start agents: python -c 'from enhanced_agent_base import *; asyncio.run(main())'")

if __name__ == "__main__":
    asyncio.run(main())