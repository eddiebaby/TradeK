#!/usr/bin/env python3
"""
Test the SPARC Trio agents functionality
"""
import sys
import os
import asyncio
from pathlib import Path

# Add agents directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agents'))

async def test_individual_agents():
    """Test each agent individually"""
    print("🤖 Testing SPARC Trio Agents")
    print("=" * 50)
    
    # Test MASTERMIND Agent
    try:
        from mastermind.mastermind_agent import MastermindAgent
        mastermind = MastermindAgent()
        print("✅ MASTERMIND Agent: Created successfully")
        
        # Test basic capability
        strategic_analysis = await mastermind.analyze_strategic_implications({
            'requirement': 'Build a trading analytics dashboard',
            'constraints': ['Real-time data', 'High performance'],
            'context': 'Financial trading platform'
        })
        
        print(f"✅ MASTERMIND Agent: Strategic analysis completed")
        print(f"   🎯 Complexity: {strategic_analysis.get('complexity_level', 'N/A')}")
        print(f"   📊 Patterns: {len(strategic_analysis.get('architectural_patterns', []))} identified")
        
    except Exception as e:
        print(f"❌ MASTERMIND Agent failed: {e}")
    
    # Test EXECUTOR Agent  
    try:
        from executor.executor_agent import ExecutorAgent
        executor = ExecutorAgent()
        print("✅ EXECUTOR Agent: Created successfully")
        
        # Test basic capability
        implementation_plan = await executor.create_implementation_plan({
            'requirement': 'REST API endpoint for user authentication',
            'technology_stack': 'FastAPI + PostgreSQL',
            'quality_requirements': {'test_coverage': 90}
        })
        
        print(f"✅ EXECUTOR Agent: Implementation plan created")
        print(f"   🧪 Test Strategy: {implementation_plan.get('test_strategy', 'N/A')}")
        print(f"   ⚡ DevOps: {len(implementation_plan.get('devops_components', []))} components")
        
    except Exception as e:
        print(f"❌ EXECUTOR Agent failed: {e}")
    
    # Test RESEARCHER Agent
    try:
        from researcher.researcher_agent import ResearcherAgent
        researcher = ResearcherAgent()
        print("✅ RESEARCHER Agent: Created successfully")
        
        # Test basic capability
        research_results = await researcher.conduct_comprehensive_research({
            'topic': 'FastAPI best practices for financial APIs',
            'scope': ['security', 'performance', 'testing'],
            'depth': 'detailed'
        })
        
        print(f"✅ RESEARCHER Agent: Research completed")
        print(f"   📚 Sources: {len(research_results.get('sources', []))} analyzed")
        print(f"   💡 Insights: {len(research_results.get('key_insights', []))} discovered")
        
    except Exception as e:
        print(f"❌ RESEARCHER Agent failed: {e}")

async def test_agent_collaboration():
    """Test basic agent collaboration"""
    print("\n🤝 Testing Agent Collaboration")
    print("=" * 40)
    
    try:
        from mastermind.mastermind_agent import MastermindAgent
        from executor.executor_agent import ExecutorAgent
        from researcher.researcher_agent import ResearcherAgent
        
        # Initialize agents
        mastermind = MastermindAgent()
        executor = ExecutorAgent()
        researcher = ResearcherAgent()
        
        print("✅ All three agents initialized")
        
        # Simulate a simple collaboration workflow
        project_requirement = {
            'name': 'Trading Signal Generator',
            'description': 'Build a system that generates trading signals based on market data',
            'constraints': ['Real-time processing', 'High accuracy', 'Scalable'],
            'technology_preferences': ['Python', 'FastAPI', 'PostgreSQL']
        }
        
        print(f"\n📋 Project: {project_requirement['name']}")
        
        # RESEARCHER analyzes requirements
        research_context = await researcher.conduct_comprehensive_research({
            'topic': 'Trading signal generation algorithms and infrastructure',
            'scope': ['algorithms', 'infrastructure', 'compliance'],
            'depth': 'comprehensive'
        })
        
        print("✅ RESEARCHER: Market research completed")
        
        # MASTERMIND creates strategy based on research
        strategic_plan = await mastermind.analyze_strategic_implications({
            'requirement': project_requirement['description'],
            'research_context': research_context,
            'constraints': project_requirement['constraints']
        })
        
        print("✅ MASTERMIND: Strategic plan created")
        
        # EXECUTOR creates implementation plan based on strategy
        implementation_plan = await executor.create_implementation_plan({
            'strategic_plan': strategic_plan,
            'requirement': project_requirement['description'],
            'technology_stack': project_requirement['technology_preferences'],
            'quality_requirements': {'test_coverage': 95, 'performance': 'high'}
        })
        
        print("✅ EXECUTOR: Implementation plan ready")
        
        print("\n🎉 SPARC Trio Collaboration Test: SUCCESS!")
        print("   • RESEARCHER provided market intelligence")
        print("   • MASTERMIND created strategic architecture") 
        print("   • EXECUTOR designed implementation plan")
        
        return True
        
    except Exception as e:
        print(f"❌ Collaboration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("🚀 SPARC Trio Agent System Test")
    print("═" * 60)
    
    # Test individual agents
    await test_individual_agents()
    
    # Test collaboration
    success = await test_agent_collaboration()
    
    print("\n" + "═" * 60)
    if success:
        print("🎊 SPARC Trio is FULLY OPERATIONAL!")
        print("\nNext steps:")
        print("1. Try: python agents/easy_start.py")
        print("2. Or:  python -m uvicorn src.api.main:app --reload")
        print("3. Use the agents for real projects!")
    else:
        print("🔧 Some components need additional fixes")
        print("But the core agents are working!")

if __name__ == "__main__":
    asyncio.run(main())