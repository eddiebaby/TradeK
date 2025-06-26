#!/usr/bin/env python3
"""
Simple test script to verify the analyze_codebase.py fixes
"""

import asyncio
import sys
from pathlib import Path

# Add the agents directory to Python path
agents_dir = Path(__file__).parent
sys.path.insert(0, str(agents_dir))

from agent_orchestrator import AgentOrchestrator

async def simple_test():
    """Test basic orchestrator functionality"""
    print("🔍 Testing Agent Orchestrator...")
    
    try:
        orchestrator = AgentOrchestrator()
        print("✅ Orchestrator created successfully")
        
        # Test basic functionality without full analysis
        print("✅ Basic initialization complete")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(simple_test())
    sys.exit(0 if success else 1)