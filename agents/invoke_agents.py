#!/usr/bin/env python3
"""
Safe Agent Invocation Script
Test and use your TradeKnowledge agents with persistent memory protection.
"""

import asyncio
import sys
import traceback
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "agents"))

async def check_system_health():
    """Check if the system and agents are working properly."""
    print("🏥 System Health Check")
    print("=" * 50)
    
    try:
        # Test persistent memory
        print("1. Testing persistent memory system...")
        from src.core.persistent_state import get_state_manager
        state_manager = get_state_manager()
        status = state_manager.get_system_status()
        print(f"   ✅ Memory signals: {status['memory_signals_count']}")
        print(f"   ✅ Active agents: {status['active_agents_count']}")
        print(f"   ✅ Data integrity: {'OK' if status['data_integrity'] else 'ERROR'}")
        
        # Test agent imports
        print("\n2. Testing agent imports...")
        from agents.agent_orchestrator import AgentOrchestrator
        from agents.mastermind.mastermind_agent import MastermindAgent
        from agents.executor.executor_agent import ExecutorAgent
        from agents.researcher.researcher_agent import ResearcherAgent
        print("   ✅ AgentOrchestrator imported")
        print("   ✅ MastermindAgent imported")
        print("   ✅ ExecutorAgent imported")
        print("   ✅ ResearcherAgent imported")
        
        # Test configuration
        print("\n3. Testing configuration...")
        from src.core.config import get_config
        config = get_config()
        print(f"   ✅ Persistence enabled: {config.api.agents.persistence.enabled}")
        print(f"   ✅ Agents enabled: {config.api.agents.enable_agents}")
        
        print("\n🎉 System is healthy and ready to use!")
        return True
        
    except Exception as e:
        print(f"\n❌ System health check failed: {e}")
        print("\nFull error:")
        traceback.print_exc()
        return False

async def show_saved_agents():
    """Show any agents saved in persistent memory."""
    print("\n💾 Saved Agent States")
    print("=" * 50)
    
    try:
        from src.core.persistent_state import get_state_manager
        state_manager = get_state_manager()
        
        # Read agent state file
        agent_state_file = Path("data/persistent/.agentstate")
        if agent_state_file.exists():
            import json
            with open(agent_state_file, 'r') as f:
                agent_data = json.load(f)
            
            agents = agent_data.get("agents", {})
            if agents:
                print(f"Found {len(agents)} saved agent states:")
                for agent_name, agent_info in agents.items():
                    print(f"\n🤖 Agent: {agent_name}")
                    print(f"   Class: {agent_info.get('class_name', 'Unknown')}")
                    print(f"   Last active: {agent_info.get('last_active', 'Unknown')}")
                    if agent_info.get('current_task'):
                        task = agent_info['current_task']
                        print(f"   Current task: {task.get('task_id')} ({task.get('progress', 0)*100:.1f}% complete)")
                    print(f"   Capabilities: {len(agent_info.get('capabilities', []))}")
            else:
                print("No saved agent states found")
        else:
            print("No agent state file found")
            
    except Exception as e:
        print(f"Error reading saved agents: {e}")

async def simple_agent_test():
    """Run a simple agent test to verify functionality."""
    print("\n🧪 Simple Agent Test")
    print("=" * 50)
    
    try:
        print("Creating and testing a basic agent workflow...")
        
        # Create orchestrator with persistent memory
        from agents.agent_orchestrator import AgentOrchestrator
        orchestrator = AgentOrchestrator()
        
        # Simple test requirement
        requirement = "Create a simple FastAPI health check endpoint"
        
        project_context = {
            "project_type": "api_endpoint",
            "technology_stack": "FastAPI",
            "deployment_target": "development"
        }
        
        quality_requirements = {
            "test_coverage": 85,
            "mutation_score": 75,
            "performance": {"max_response_time": 200},
            "security": {"min_security_score": 7.0}
        }
        
        print(f"📋 Requirement: {requirement}")
        print("⚡ Starting agent workflow...")
        
        # Execute (this should save state automatically via persistent memory)
        results = await orchestrator.execute_comprehensive_development_cycle(
            requirement=requirement,
            project_context=project_context,
            quality_requirements=quality_requirements
        )
        
        # Check results
        if results and 'session_results' in results:
            metrics = results['session_results']['metrics']
            print(f"\n✅ Test completed successfully!")
            print(f"🏆 Quality Score: {metrics.quality_amplification:.1f}/10")
            print(f"🤝 Collaboration: {metrics.collaboration_effectiveness:.1f}%")
            print("💾 Agent states automatically saved to persistent memory")
            return True
        else:
            print("❌ Test completed but no results returned")
            return False
            
    except Exception as e:
        print(f"\n❌ Agent test failed: {e}")
        traceback.print_exc()
        return False

async def interactive_agent_menu():
    """Interactive menu for using agents."""
    print("\n🤖 TradeKnowledge Agent Interface")
    print("=" * 50)
    print("Choose an option:")
    print("1. 🏥 System Health Check")
    print("2. 💾 Show Saved Agent States") 
    print("3. 🧪 Run Simple Agent Test")
    print("4. 🚀 Quick Start Demo (use existing easy_start.py)")
    print("5. 🛠️  Build Custom Project (use existing use_agents.py)")
    print("6. 🔄 Check Persistent Memory Recovery")
    print("7. 📊 Start API Server")
    print("8. ❌ Exit")
    
    choice = input("\nEnter choice (1-8): ").strip()
    return choice

async def check_recovery():
    """Test persistent memory recovery."""
    print("\n🔄 Persistent Memory Recovery Test")
    print("=" * 50)
    
    try:
        # Initialize application (this triggers recovery if needed)
        from src.core.app_initialization import initialize_application
        result = await initialize_application()
        
        if result['success']:
            print("✅ Application initialization successful")
            
            if result.get('recovery', {}).get('recovery_performed'):
                recovery = result['recovery']
                print(f"🔄 Recovery performed: {recovery['shutdown_type']} shutdown detected")
                print(f"👥 Agents recovered: {recovery['agents_recovered']}")
                print(f"📋 Workflows restored: {recovery['workflows_restored']}")
                print(f"🏥 System health: {recovery['system_health_score']:.2f}")
            else:
                print("ℹ️  No recovery needed - clean shutdown detected")
                
            print(f"🏥 Final health score: {result.get('health', {}).get('health_score', 0):.2f}")
            return True
        else:
            print(f"❌ Recovery test failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Recovery test error: {e}")
        traceback.print_exc()
        return False

async def start_api_server():
    """Start the API server with agents."""
    print("\n📊 Starting TradeKnowledge API Server")
    print("=" * 50)
    print("This will start the FastAPI server with agent endpoints...")
    print("Once running, you can access:")
    print("- Health check: http://localhost:8000/health")
    print("- Agent endpoints: http://localhost:8000/api/v1/agents/")
    print("- API docs: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server")
    
    try:
        import uvicorn
        uvicorn.run(
            "src.api.main:app",
            host="127.0.0.1",
            port=8000,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")

async def main():
    """Main entry point."""
    print("🎯 TradeKnowledge Agent System")
    print("Safe invocation with persistent memory protection")
    print("=" * 60)
    
    while True:
        try:
            choice = await interactive_agent_menu()
            
            if choice == "1":
                await check_system_health()
            elif choice == "2":
                await show_saved_agents()
            elif choice == "3":
                await simple_agent_test()
            elif choice == "4":
                print("\n🚀 Running Quick Start Demo...")
                print("Executing: python agents/easy_start.py")
                import subprocess
                subprocess.run([sys.executable, "agents/easy_start.py"])
            elif choice == "5":
                print("\n🛠️  Running Custom Project Builder...")
                print("Executing: python agents/use_agents.py")
                import subprocess
                subprocess.run([sys.executable, "agents/use_agents.py"])
            elif choice == "6":
                await check_recovery()
            elif choice == "7":
                await start_api_server()
            elif choice == "8":
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-8.")
                continue
                
            input("\n⏎ Press Enter to continue...")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Full error:")
            traceback.print_exc()
            input("\n⏎ Press Enter to continue...")

if __name__ == "__main__":
    asyncio.run(main())