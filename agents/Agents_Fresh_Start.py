#!/usr/bin/env python3
"""
🤖 Agents Fresh Start
The ONE script to rule them all - your unified agent trio launcher.

This script provides a clean, simple interface to:
1. Initialize your three core agents (MASTERMIND, EXECUTOR, RESEARCHER)
2. Test their functionality
3. Launch interactive modes
4. Start the API server

Usage:
    python agents/Agents_Fresh_Start.py
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any

# Fix paths for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "agents"))

class AgentsFreshStart:
    """Unified agent management system."""
    
    def __init__(self):
        self.mastermind = None
        self.executor = None
        self.researcher = None
        self.agents_initialized = False
        
    async def initialize_agents(self) -> bool:
        """Initialize all three core agents."""
        print("🚀 INITIALIZING AGENT TRIO")
        print("=" * 50)
        
        try:
            # Initialize MASTERMIND
            print("1. 🧠 Initializing MASTERMIND Agent...")
            from mastermind.mastermind_agent import MastermindAgent
            self.mastermind = MastermindAgent()
            print(f"   ✅ {self.mastermind.name} ready")
            print(f"   📋 {len(self.mastermind.get_capabilities())} capabilities loaded")
            
            # Initialize EXECUTOR
            print("\n2. ⚡ Initializing EXECUTOR Agent...")
            from executor.executor_agent import ExecutorAgent
            self.executor = ExecutorAgent()
            print(f"   ✅ {self.executor.name} ready")
            print(f"   📋 {len(self.executor.get_capabilities())} capabilities loaded")
            
            # Initialize RESEARCHER
            print("\n3. 🔍 Initializing RESEARCHER Agent...")
            from researcher.researcher_agent import ResearcherAgent
            self.researcher = ResearcherAgent()
            print(f"   ✅ {self.researcher.name} ready")
            print(f"   📋 {len(self.researcher.get_capabilities())} capabilities loaded")
            
            # Check persistent memory
            print("\n💾 Checking persistent memory...")
            from src.core.persistent_state import get_state_manager
            state_manager = get_state_manager()
            status = state_manager.get_system_status()
            print(f"   📊 Memory signals: {status['memory_signals_count']}")
            print(f"   👥 Active agents: {status['active_agents_count']}")
            
            self.agents_initialized = True
            print("\n🎉 ALL AGENTS SUCCESSFULLY INITIALIZED!")
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def show_capabilities(self):
        """Display what each agent can do."""
        print("\n🎯 AGENT CAPABILITIES")
        print("=" * 50)
        
        print("🧠 MASTERMIND AGENT:")
        print("   • Strategic planning & architectural design")
        print("   • Risk assessment & failure prediction")
        print("   • Technology stack optimization")
        print("   • Quality orchestration")
        print("   • Technical decision making")
        
        print("\n⚡ EXECUTOR AGENT:")
        print("   • Test-Driven Development (TDD)")
        print("   • Comprehensive testing frameworks")
        print("   • DevOps automation & CI/CD")
        print("   • Performance optimization")
        print("   • Security implementation")
        
        print("\n🔍 RESEARCHER AGENT:")
        print("   • Multi-source intelligence gathering")
        print("   • Evidence-based analysis")
        print("   • Trend prediction & analysis")
        print("   • Best practice identification")
        print("   • Security research & threat intelligence")
    
    def show_menu(self):
        """Display the main menu."""
        print("\n🎛️  MAIN MENU")
        print("=" * 50)
        print("1. 🧪 Test Agent Functionality")
        print("2. 🔄 Run SPARC Methodology Demo")
        print("3. 🌐 Start API Server")
        print("4. 💬 Interactive Agent Chat")
        print("5. 📊 System Status")
        print("6. 📚 Show Capabilities")
        print("7. 🚪 Exit")
        print("\nEnter your choice (1-7): ", end="")
    
    async def test_agent_functionality(self):
        """Test basic agent functionality."""
        print("\n🧪 TESTING AGENT FUNCTIONALITY")
        print("=" * 50)
        
        if not self.agents_initialized:
            print("❌ Agents not initialized. Please restart.")
            return
        
        try:
            # Test MASTERMIND
            print("Testing MASTERMIND...")
            mastermind_test = await self.mastermind.strategic_analysis("Simple test task")
            print("   ✅ MASTERMIND responding correctly")
            
            # Test EXECUTOR
            print("Testing EXECUTOR...")
            executor_capabilities = self.executor.get_capabilities()
            print(f"   ✅ EXECUTOR has {len(executor_capabilities)} capabilities")
            
            # Test RESEARCHER
            print("Testing RESEARCHER...")
            researcher_modes = self.researcher.get_thinking_modes()
            print(f"   ✅ RESEARCHER has {len(researcher_modes)} thinking modes")
            
            print("\n🎉 ALL TESTS PASSED!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    async def run_sparc_demo(self):
        """Run the SPARC methodology demonstration."""
        print("\n🔄 RUNNING SPARC METHODOLOGY DEMO")
        print("=" * 50)
        
        try:
            from sparc.sparc_demo import run_sparc_demo
            await run_sparc_demo()
        except Exception as e:
            print(f"❌ SPARC demo failed: {e}")
            print("   💡 Try: python agents/sparc/sparc_demo.py")
    
    def start_api_server(self):
        """Start the API server."""
        print("\n🌐 STARTING API SERVER")
        print("=" * 50)
        print("Starting FastAPI server...")
        print("📍 URL: http://localhost:8000")
        print("📖 Docs: http://localhost:8000/docs")
        print("\n💡 Use Ctrl+C to stop the server")
        
        os.system("cd /home/scottschweizer/TradeKnowledge && python -m uvicorn src.api.main:app --reload")
    
    async def interactive_chat(self):
        """Start interactive chat with agents."""
        print("\n💬 INTERACTIVE AGENT CHAT")
        print("=" * 50)
        print("Type 'quit' to exit, 'help' for commands")
        
        if not self.agents_initialized:
            print("❌ Agents not initialized. Please restart.")
            return
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    print("💡 Commands:")
                    print("   • @mastermind [message] - Ask MASTERMIND")
                    print("   • @executor [message] - Ask EXECUTOR")
                    print("   • @researcher [message] - Ask RESEARCHER")
                    print("   • help - Show this help")
                    print("   • quit - Exit chat")
                    continue
                elif user_input.startswith('@mastermind'):
                    query = user_input[11:].strip()
                    response = await self.mastermind.strategic_analysis(query)
                    print(f"🧠 MASTERMIND: {response}")
                elif user_input.startswith('@executor'):
                    query = user_input[9:].strip()
                    print(f"⚡ EXECUTOR: I can help implement '{query}' using TDD methodology.")
                elif user_input.startswith('@researcher'):
                    query = user_input[11:].strip()
                    print(f"🔍 RESEARCHER: I'll research '{query}' using multi-source intelligence.")
                else:
                    print("💡 Use @mastermind, @executor, or @researcher to direct your message")
                    
            except KeyboardInterrupt:
                print("\n👋 Chat ended!")
                break
            except Exception as e:
                print(f"❌ Chat error: {e}")
    
    def show_system_status(self):
        """Show current system status."""
        print("\n📊 SYSTEM STATUS")
        print("=" * 50)
        
        try:
            from src.core.persistent_state import get_state_manager
            state_manager = get_state_manager()
            status = state_manager.get_system_status()
            
            print(f"🤖 Agents Initialized: {'✅ Yes' if self.agents_initialized else '❌ No'}")
            print(f"💾 Memory Signals: {status['memory_signals_count']}")
            print(f"👥 Active Agents: {status['active_agents_count']}")
            print(f"📂 Project Root: {project_root}")
            print(f"🐍 Python Path: {sys.executable}")
            
        except Exception as e:
            print(f"❌ Status check failed: {e}")
    
    async def run(self):
        """Main application loop."""
        print("🎯 TradeKnowledge Agent System")
        print("🚀 Fresh Start - Unified Agent Manager")
        print("=" * 60)
        
        # Initialize agents
        success = await self.initialize_agents()
        if not success:
            print("\n❌ Agent initialization failed, but you can still use the menu")
        
        # Show capabilities
        self.show_capabilities()
        
        # Main menu loop
        while True:
            try:
                self.show_menu()
                choice = input().strip()
                
                if choice == '1':
                    await self.test_agent_functionality()
                elif choice == '2':
                    await self.run_sparc_demo()
                elif choice == '3':
                    self.start_api_server()
                elif choice == '4':
                    await self.interactive_chat()
                elif choice == '5':
                    self.show_system_status()
                elif choice == '6':
                    self.show_capabilities()
                elif choice == '7':
                    print("\n👋 Goodbye! Your agents remain in persistent memory.")
                    break
                else:
                    print("❌ Invalid choice. Please enter 1-7.")
                    
            except KeyboardInterrupt:
                print("\n👋 Exiting...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

async def main():
    """Entry point."""
    app = AgentsFreshStart()
    await app.run()

if __name__ == "__main__":
    asyncio.run(main())