#!/usr/bin/env python3
"""
Simple Integration Test for OpenAI Agents SDK

This script tests the basic setup and configuration of OpenAI agents SDK integration.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_openai_agents_installation():
    """Test OpenAI agents SDK installation."""
    
    print("🔍 Testing OpenAI Agents SDK Installation...")
    
    try:
        import agents
        print("   ✅ OpenAI agents SDK imported successfully")
        print(f"   📦 Version: {getattr(agents, '__version__', 'unknown')}")
        return True
    except ImportError as e:
        print(f"   ❌ Failed to import OpenAI agents SDK: {e}")
        return False

def test_openai_tools_availability():
    """Test availability of OpenAI tools."""
    
    print("\n🛠️ Testing OpenAI Tools Availability...")
    
    try:
        from agents import WebSearchTool, CodeInterpreterTool, FileSearchTool, Agent, Runner
        print("   ✅ WebSearchTool available")
        print("   ✅ CodeInterpreterTool available") 
        print("   ✅ FileSearchTool available")
        print("   ✅ Agent class available")
        print("   ✅ Runner class available")
        return True
    except ImportError as e:
        print(f"   ❌ Failed to import OpenAI tools: {e}")
        return False

def test_mcp_support():
    """Test MCP support availability."""
    
    print("\n🔌 Testing MCP Support...")
    
    try:
        from agents.mcp import MCPServer, MCPServerStdio
        print("   ✅ MCP Server classes available")
        return True
    except ImportError as e:
        print(f"   ❌ MCP support not available: {e}")
        return False

def test_handoff_support():
    """Test handoff support availability."""
    
    print("\n🤝 Testing Handoff Support...")
    
    try:
        from agents import handoff, HandoffInputData
        print("   ✅ Handoff mechanisms available")
        return True
    except ImportError as e:
        print(f"   ❌ Handoff support not available: {e}")
        return False

def test_configuration_files():
    """Test configuration files."""
    
    print("\n📋 Testing Configuration Files...")
    
    success = True
    
    # Test config.yaml
    config_path = project_root / "config" / "config.yaml"
    if config_path.exists():
        print("   ✅ config/config.yaml found")
        
        # Read and check for OpenAI agents configuration
        with open(config_path, 'r') as f:
            content = f.read()
            if "openai_agents:" in content:
                print("   ✅ OpenAI agents configuration found in config.yaml")
            else:
                print("   ⚠️  OpenAI agents configuration not found in config.yaml")
                success = False
                
            if "mcp:" in content:
                print("   ✅ MCP configuration found in config.yaml")
            else:
                print("   ⚠️  MCP configuration not found in config.yaml")
                success = False
                
            if "coordination:" in content:
                print("   ✅ Coordination configuration found in config.yaml")
            else:
                print("   ⚠️  Coordination configuration not found in config.yaml")
                success = False
    else:
        print("   ❌ config/config.yaml not found")
        success = False
    
    # Test requirements-dev.txt
    req_path = project_root / "requirements-dev.txt"
    if req_path.exists():
        with open(req_path, 'r') as f:
            content = f.read()
            if "openai-agents" in content:
                print("   ✅ openai-agents dependency found in requirements-dev.txt")
            else:
                print("   ⚠️  openai-agents dependency not found in requirements-dev.txt")
                success = False
    else:
        print("   ❌ requirements-dev.txt not found")
        success = False
    
    return success

def test_enhanced_agent_files():
    """Test that enhanced agent files exist."""
    
    print("\n🤖 Testing Enhanced Agent Files...")
    
    success = True
    
    agent_files = [
        "agents/researcher/enhanced_researcher_agent.py",
        "agents/executor/enhanced_executor_agent.py", 
        "agents/shared/enhanced_document_processor.py",
        "agents/core/mcp_integration.py",
        "agents/core/unified_tool_interface.py",
        "agents/core/enhanced_coordination.py"
    ]
    
    for file_path in agent_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"   ✅ {file_path} exists")
        else:
            print(f"   ❌ {file_path} missing")
            success = False
    
    return success

def test_environment_variables():
    """Test environment variables."""
    
    print("\n🌍 Testing Environment Variables...")
    
    if os.getenv("OPENAI_API_KEY"):
        print("   ✅ OPENAI_API_KEY is set")
        # Don't print the actual key for security
        key_preview = os.getenv("OPENAI_API_KEY")[:8] + "..."
        print(f"   🔑 Key preview: {key_preview}")
    else:
        print("   ⚠️  OPENAI_API_KEY not set - features will run in simulation mode")
    
    if os.getenv("OPENAI_VECTOR_STORE_IDS"):
        print("   ✅ OPENAI_VECTOR_STORE_IDS is set")
    else:
        print("   ℹ️  OPENAI_VECTOR_STORE_IDS not set - file search will be limited")
    
    return True

def test_basic_agent_creation():
    """Test basic agent creation."""
    
    print("\n🚀 Testing Basic Agent Creation...")
    
    try:
        from agents import Agent
        
        # Create a simple test agent
        test_agent = Agent(
            name="TestAgent",
            instructions="You are a test agent for integration testing."
        )
        
        print("   ✅ Basic agent created successfully")
        print(f"   📝 Agent name: {test_agent.name}")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to create basic agent: {e}")
        return False

def test_tool_creation():
    """Test tool creation."""
    
    print("\n🔧 Testing Tool Creation...")
    
    try:
        from agents import WebSearchTool, CodeInterpreterTool
        
        # Test WebSearchTool creation
        web_tool = WebSearchTool(
            user_location={"type": "approximate", "city": "New York"}
        )
        print("   ✅ WebSearchTool created successfully")
        
        # Test CodeInterpreterTool creation
        code_tool = CodeInterpreterTool(
            tool_config={"type": "code_interpreter", "container": {"type": "auto"}}
        )
        print("   ✅ CodeInterpreterTool created successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to create tools: {e}")
        return False

def main():
    """Run all tests."""
    
    print("🚀 OpenAI Agents SDK Integration Test")
    print("=" * 50)
    
    tests = [
        ("OpenAI Agents Installation", test_openai_agents_installation),
        ("OpenAI Tools Availability", test_openai_tools_availability),
        ("MCP Support", test_mcp_support),
        ("Handoff Support", test_handoff_support),
        ("Configuration Files", test_configuration_files),
        ("Enhanced Agent Files", test_enhanced_agent_files),
        ("Environment Variables", test_environment_variables),
        ("Basic Agent Creation", test_basic_agent_creation),
        ("Tool Creation", test_tool_creation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"   ❌ Test '{test_name}' failed with exception: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\n📈 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! OpenAI Agents SDK integration is ready.")
        print("\n🚀 Your enhanced TradeKnowledge agents are now available with:")
        print("   • Real-time web search capabilities")
        print("   • Live code execution and validation")
        print("   • Advanced document processing")
        print("   • Enhanced coordination patterns")
        print("   • MCP integration framework")
        print("   • Unified tool interface")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Some features may be limited.")
        print("\n💡 Next steps:")
        print("   1. Install missing dependencies: pip install openai-agents")
        print("   2. Set OPENAI_API_KEY environment variable")
        print("   3. Review configuration in config/config.yaml")
    
    print(f"\n📚 Documentation:")
    print(f"   • OpenAI Agents SDK: https://github.com/openai/openai-agents-python")
    print(f"   • Configuration guide: config/config.yaml")
    print(f"   • Enhanced agents: agents/*/enhanced_*_agent.py")

if __name__ == "__main__":
    main()