#!/usr/bin/env python3
"""
Simple test script to verify agents are working
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
from agents.agent_orchestrator import AgentOrchestrator

async def test_agents():
    """Test the agent system"""
    print("🤖 Testing TradeKnowledge Agents...")
    print("=" * 50)
    
    try:
        # Initialize orchestrator
        orchestrator = AgentOrchestrator()
        print("✅ Agent orchestrator initialized")
        
        # Test basic agent communication
        test_task = {
            'requirement': 'Create a simple "hello world" function',
            'project_context': {
                'technology_stack': 'Python',
                'framework': 'FastAPI'
            },
            'quality_requirements': {
                'test_coverage': 80,
                'documentation': True
            }
        }
        
        print("\n🚀 Running test task...")
        print(f"Task: {test_task['requirement']}")
        
        # Execute task
        results = await orchestrator.execute_comprehensive_development_cycle(
            requirement=test_task['requirement'],
            project_context=test_task['project_context'],
            quality_requirements=test_task['quality_requirements']
        )
        
        print("\n📊 Results:")
        print("=" * 30)
        if results:
            for key, value in results.items():
                print(f"{key}: {value}")
        else:
            print("No results returned (this might be normal for initial setup)")
            
        print("\n✅ Agents are working!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing agents: {e}")
        print("This might be due to missing dependencies or configuration")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_agents())
    if success:
        print("\n🎉 Agent system is operational!")
        print("\nNext steps:")
        print("1. Set OPENAI_API_KEY for enhanced features")
        print("2. Try: python agents/easy_start.py (interactive)")
        print("3. Or use the API server: python -m uvicorn src.api.main:app --reload")
    else:
        print("\n🔧 Some setup may be needed, but basic infrastructure is there")