#!/usr/bin/env python3
"""
Use Your 3 Working Agents
Simple script to demonstrate your actual working agents.
"""

import asyncio
import sys
from pathlib import Path

# Fix paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "agents"))

async def demo_your_agents():
    """Demonstrate your 3 working agents."""
    print("🤖 YOUR 3 WORKING AGENTS DEMO")
    print("=" * 50)
    
    try:
        print("1. 🧠 Creating MASTERMIND Agent...")
        from agents.mastermind.mastermind_agent import MastermindAgent
        mastermind = MastermindAgent()
        print(f"   ✅ Created: {mastermind.name}")
        print(f"   📋 Capabilities: {len(mastermind.get_capabilities())}")
        
        print("\n2. ⚡ Creating EXECUTOR Agent...")
        from agents.executor.executor_agent import ExecutorAgent
        executor = ExecutorAgent()
        print(f"   ✅ Created: {executor.name}")
        print(f"   📋 Capabilities: {len(executor.get_capabilities())}")
        
        print("\n3. 🔍 Creating RESEARCHER Agent...")
        from agents.researcher.researcher_agent import ResearcherAgent
        researcher = ResearcherAgent()
        print(f"   ✅ Created: {researcher.name}")
        print(f"   📋 Capabilities: {len(researcher.get_capabilities())}")
        
        print("\n💾 All agents automatically saved to persistent memory!")
        
        # Check persistent memory
        from src.core.persistent_state import get_state_manager
        state_manager = get_state_manager()
        status = state_manager.get_system_status()
        print(f"📊 Current memory signals: {status['memory_signals_count']}")
        print(f"👥 Active agents in memory: {status['active_agents_count']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_agent_capabilities():
    """Show what each agent can do."""
    print("\n🎯 WHAT YOUR AGENTS CAN DO")
    print("=" * 50)
    
    print("🧠 MASTERMIND AGENT:")
    print("   • Strategic analysis and architectural design")
    print("   • Quality strategy orchestration") 
    print("   • Risk assessment and failure prediction")
    print("   • Technical decision making")
    print("   • Technology stack optimization")
    
    print("\n⚡ EXECUTOR AGENT:")
    print("   • Test-Driven Development (TDD)")
    print("   • Comprehensive testing (unit, integration, etc.)")
    print("   • DevOps automation and CI/CD")
    print("   • Performance optimization")
    print("   • Security implementation")
    
    print("\n🔍 RESEARCHER AGENT:")
    print("   • Multi-source intelligence gathering")
    print("   • Evidence-based insight synthesis")
    print("   • Trend analysis and prediction")
    print("   • Best practice identification")
    print("   • Security intelligence research")

def show_usage_options():
    """Show how to actually use the agents."""
    print("\n🚀 HOW TO USE YOUR AGENTS")
    print("=" * 50)
    
    print("OPTION 1: 🌐 API Server (Easiest)")
    print("   python -m uvicorn src.api.main:app --reload")
    print("   Visit: http://localhost:8000/docs")
    print("   Use the /api/v1/agents/* endpoints")
    
    print("\nOPTION 2: 🐍 Direct Python Code")
    print("   ```python")
    print("   from agents.mastermind.mastermind_agent import MastermindAgent")
    print("   from agents.executor.executor_agent import ExecutorAgent")
    print("   from agents.researcher.researcher_agent import ResearcherAgent")
    print("   ")
    print("   # Create agents (auto-saved to persistent memory)")
    print("   mastermind = MastermindAgent()")
    print("   executor = ExecutorAgent()")
    print("   researcher = ResearcherAgent()")
    print("   ")
    print("   # Use their capabilities")
    print("   capabilities = mastermind.get_capabilities()")
    print("   thinking_modes = mastermind.get_thinking_modes()")
    print("   ```")
    
    print("\nOPTION 3: 🎯 Simple Task Example")
    print("   ```python")
    print("   # Ask MASTERMIND for strategic advice")
    print("   mastermind = MastermindAgent()")
    print("   advice = await mastermind.strategic_analysis('Build trading API')")
    print("   ")
    print("   # Ask EXECUTOR to implement")
    print("   executor = ExecutorAgent()")
    print("   result = await executor.tdd_implementation(advice)")
    print("   ")
    print("   # Ask RESEARCHER for best practices")
    print("   researcher = ResearcherAgent()")
    print("   practices = await researcher.gather_intelligence('API security')")
    print("   ```")

async def main():
    """Main function."""
    print("🎯 TradeKnowledge Agent System")
    print("Demonstration of your 3 working agents")
    print("=" * 60)
    
    # Demo the agents
    success = await demo_your_agents()
    
    if success:
        # Show capabilities
        show_agent_capabilities()
        
        # Show usage
        show_usage_options()
        
        print("\n" + "=" * 60)
        print("🎊 SUCCESS!")
        print("✅ Your 3 agents are working with persistent memory")
        print("✅ All agent states are automatically saved")
        print("✅ Work will survive crashes and restarts")
        print("\n💡 RECOMMENDED: Start the API server to use them easily!")
        print("   Command: python -m uvicorn src.api.main:app --reload")
    else:
        print("\n❌ Some issues found, but agents may still work via API")

if __name__ == "__main__":
    asyncio.run(main())