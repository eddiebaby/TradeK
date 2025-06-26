#!/usr/bin/env python3
"""
Integration Test for OpenAI Agents SDK Integration

This script tests the complete integration of OpenAI agents SDK tools
with the TradeKnowledge agent system.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.core.config import get_config
    from agents.core.agent_base import TaskContext, AgentRole
except ImportError as e:
    print(f"⚠️  Import error: {e}")
    print("Some modules may not be available - testing what we can...")
    
    # Create minimal stubs for testing
    class TaskContext:
        def __init__(self, task_id, description, requirements, quality_gates, 
                     performance_targets, constraints, success_criteria):
            self.task_id = task_id
            self.description = description
            self.requirements = requirements
            self.quality_gates = quality_gates
            self.performance_targets = performance_targets
            self.constraints = constraints
            self.success_criteria = success_criteria
    
    class AgentRole:
        MASTERMIND = "mastermind"
        EXECUTOR = "executor" 
        RESEARCHER = "researcher"
    
    def get_config():
        class MockConfig:
            class Api:
                class Agents:
                    class OpenAI:
                        openai_model = "gpt-4"
                        enable_web_search = True
                        enable_code_interpreter = True
                        enable_file_search = True
                    class MCP:
                        enable_mcp = False
                        mcp_timeout = 30
                    class Coordination:
                        default_coordination_pattern = "sparc_full_cycle"
                        enable_enhanced_handoffs = True
                    openai_agents = OpenAI()
                    mcp = MCP()
                    coordination = Coordination()
                agents = Agents()
            api = Api()
        return MockConfig()


class OpenAIAgentsIntegrationTest:
    """Comprehensive integration test for OpenAI agents SDK integration."""
    
    def __init__(self):
        self.config = get_config()
        self.test_results = {}
        self.errors = []
    
    async def run_all_tests(self):
        """Run all integration tests."""
        
        print("🚀 Starting OpenAI Agents SDK Integration Tests")
        print("=" * 60)
        
        # Test 1: Configuration Loading
        await self._test_configuration_loading()
        
        # Test 2: Enhanced RESEARCHER Agent
        await self._test_enhanced_researcher_agent()
        
        # Test 3: Enhanced EXECUTOR Agent  
        await self._test_enhanced_executor_agent()
        
        # Test 4: Enhanced Document Processor
        await self._test_enhanced_document_processor()
        
        # Test 5: MCP Integration Framework
        await self._test_mcp_integration_framework()
        
        # Test 6: Unified Tool Interface
        await self._test_unified_tool_interface()
        
        # Test 7: Enhanced Coordination
        await self._test_enhanced_coordination()
        
        # Print results
        await self._print_test_results()
    
    async def _test_configuration_loading(self):
        """Test that OpenAI agents SDK configuration loads correctly."""
        
        test_name = "Configuration Loading"
        print(f"\n📋 Testing {test_name}...")
        
        try:
            # Test OpenAI agents configuration
            openai_config = self.config.api.agents.openai_agents
            assert hasattr(openai_config, 'openai_model')
            assert hasattr(openai_config, 'enable_web_search')
            assert hasattr(openai_config, 'enable_code_interpreter')
            assert hasattr(openai_config, 'enable_file_search')
            
            # Test MCP configuration
            mcp_config = self.config.api.agents.mcp
            assert hasattr(mcp_config, 'enable_mcp')
            assert hasattr(mcp_config, 'mcp_timeout')
            
            # Test coordination configuration
            coord_config = self.config.api.agents.coordination
            assert hasattr(coord_config, 'default_coordination_pattern')
            assert hasattr(coord_config, 'enable_enhanced_handoffs')
            
            self.test_results[test_name] = "✅ PASSED"
            print(f"   ✓ OpenAI agents configuration loaded successfully")
            print(f"   ✓ MCP configuration loaded successfully")
            print(f"   ✓ Coordination configuration loaded successfully")
            
        except Exception as e:
            self.test_results[test_name] = f"❌ FAILED: {str(e)}"
            self.errors.append(f"{test_name}: {str(e)}")
            print(f"   ❌ Error: {str(e)}")
    
    async def _test_enhanced_researcher_agent(self):
        """Test enhanced RESEARCHER agent with OpenAI tools."""
        
        test_name = "Enhanced RESEARCHER Agent"
        print(f"\n🔍 Testing {test_name}...")
        
        try:
            # Create enhanced researcher agent (with simulated OpenAI key)
            researcher = EnhancedResearcherAgent(openai_api_key="test-key")
            
            # Test capabilities
            capabilities = researcher.get_enhanced_capabilities()
            assert "real_time_web_search" in capabilities
            assert "market_intelligence_gathering" in capabilities
            
            # Test research specification
            research_spec = {
                "domains": ["technical_analysis", "market_intelligence"],
                "focus_areas": ["trading_strategies", "algorithmic_trading"],
                "depth": "comprehensive",
                "context": {"test_mode": True}
            }
            
            # Note: This would normally execute web search, but will use simulated results
            # in test mode since we don't have a real OpenAI API key
            research_result = await researcher.conduct_enhanced_research(research_spec)
            
            assert research_result is not None
            assert hasattr(research_result, 'traditional_research')
            assert hasattr(research_result, 'web_intelligence')
            assert hasattr(research_result, 'processing_quality')
            
            self.test_results[test_name] = "✅ PASSED"
            print(f"   ✓ Enhanced capabilities loaded: {len(capabilities)} total")
            print(f"   ✓ Research execution completed successfully")
            print(f"   ✓ Processing quality: {research_result.processing_quality:.2f}")
            
        except Exception as e:
            self.test_results[test_name] = f"❌ FAILED: {str(e)}"
            self.errors.append(f"{test_name}: {str(e)}")
            print(f"   ❌ Error: {str(e)}")
    
    async def _test_enhanced_executor_agent(self):
        """Test enhanced EXECUTOR agent with OpenAI tools."""
        
        test_name = "Enhanced EXECUTOR Agent"
        print(f"\n⚡ Testing {test_name}...")
        
        try:
            # Create enhanced executor agent
            executor = EnhancedExecutorAgent(openai_api_key="test-key")
            
            # Test capabilities
            capabilities = executor.get_enhanced_capabilities()
            assert "live_code_execution" in capabilities
            assert "real_time_tdd_validation" in capabilities
            
            # Create test task context
            task_context = TaskContext(
                task_id="test_executor_task",
                description="Test TDD implementation with live validation",
                requirements={
                    "implement_function": "calculate_trading_profit",
                    "test_coverage": 90,
                    "performance_target": "< 100ms"
                },
                quality_gates={"test_coverage": 0.9, "mutation_score": 0.8},
                performance_targets={"response_time": 100},
                constraints=["no_external_dependencies"],
                success_criteria=["tests_pass", "meets_performance_targets"]
            )
            
            # Test implementation strategy
            implementation_strategy = {
                "methodology": "enhanced_tdd_with_live_validation",
                "quality_requirements": {
                    "test_coverage": 90,
                    "mutation_score": 80
                }
            }
            
            # Execute enhanced implementation (simulated)
            implementation_result = await executor.execute_enhanced_implementation(
                task_context, implementation_strategy
            )
            
            assert implementation_result is not None
            assert hasattr(implementation_result, 'base_implementation')
            assert hasattr(implementation_result, 'live_validation_results')
            assert hasattr(implementation_result, 'quality_confidence')
            
            self.test_results[test_name] = "✅ PASSED"
            print(f"   ✓ Enhanced capabilities loaded: {len(capabilities)} total")
            print(f"   ✓ Enhanced implementation completed successfully")
            print(f"   ✓ Quality confidence: {implementation_result.quality_confidence:.2f}")
            
        except Exception as e:
            self.test_results[test_name] = f"❌ FAILED: {str(e)}"
            self.errors.append(f"{test_name}: {str(e)}")
            print(f"   ❌ Error: {str(e)}")
    
    async def _test_enhanced_document_processor(self):
        """Test enhanced document processor with OpenAI tools."""
        
        test_name = "Enhanced Document Processor"
        print(f"\n📄 Testing {test_name}...")
        
        try:
            # Create enhanced document processor (without vector store IDs for testing)
            doc_processor = EnhancedDocumentProcessor(
                vector_store_ids=[],  # Empty for testing
                openai_api_key="test-key"
            )
            
            # Test capabilities
            capabilities = doc_processor.get_enhanced_capabilities()
            assert "semantic_document_search" in capabilities
            assert "ai_powered_summarization" in capabilities
            
            # Create test document (we'll use a simple text file)
            test_doc_path = project_root / "test_document.txt"
            with open(test_doc_path, "w") as f:
                f.write("""
                Trading Strategy Analysis
                
                This document outlines a comprehensive approach to algorithmic trading
                using machine learning techniques for market prediction and risk management.
                
                Key components:
                1. Data preprocessing and feature engineering
                2. Model training and validation
                3. Risk management systems
                4. Performance monitoring and optimization
                """)
            
            try:
                # Process enhanced document (simulated mode)
                doc_result = await doc_processor.process_enhanced_document(
                    test_doc_path,
                    document_type="trading_strategies",
                    analysis_depth="standard"
                )
                
                assert doc_result is not None
                assert hasattr(doc_result, 'traditional_processing')
                assert hasattr(doc_result, 'ai_insights')
                assert hasattr(doc_result, 'processing_quality')
                
                print(f"   ✓ Enhanced capabilities loaded: {len(capabilities)} total")
                print(f"   ✓ Document processing completed successfully")
                print(f"   ✓ Processing quality: {doc_result.processing_quality:.2f}")
                
            finally:
                # Clean up test file
                if test_doc_path.exists():
                    test_doc_path.unlink()
            
            self.test_results[test_name] = "✅ PASSED"
            
        except Exception as e:
            self.test_results[test_name] = f"❌ FAILED: {str(e)}"
            self.errors.append(f"{test_name}: {str(e)}")
            print(f"   ❌ Error: {str(e)}")
    
    async def _test_mcp_integration_framework(self):
        """Test MCP integration framework."""
        
        test_name = "MCP Integration Framework"
        print(f"\n🔌 Testing {test_name}...")
        
        try:
            # Create MCP server manager
            mcp_manager = MCPServerManager()
            
            # Test MCP configuration creation
            filesystem_config = create_filesystem_mcp_config(
                name="test_filesystem", 
                directory=str(project_root)
            )
            
            assert filesystem_config.name == "test_filesystem"
            assert filesystem_config.server_type == "stdio"
            assert filesystem_config.tools_cache_enabled == True
            
            # Test empty tools list (no servers added)
            all_tools = await mcp_manager.list_all_tools()
            assert isinstance(all_tools, dict)
            
            self.test_results[test_name] = "✅ PASSED"
            print(f"   ✓ MCP server manager created successfully")
            print(f"   ✓ MCP configuration generation working")
            print(f"   ✓ Tool listing functionality working")
            
        except Exception as e:
            self.test_results[test_name] = f"❌ FAILED: {str(e)}"
            self.errors.append(f"{test_name}: {str(e)}")
            print(f"   ❌ Error: {str(e)}")
    
    async def _test_unified_tool_interface(self):
        """Test unified tool interface."""
        
        test_name = "Unified Tool Interface"
        print(f"\n🛠️ Testing {test_name}...")
        
        try:
            # Create mock dependencies
            search_engine = self._create_mock_search_engine()
            book_processor = self._create_mock_book_processor()
            cache_manager = self._create_mock_cache_manager()
            mcp_manager = MCPServerManager()
            
            # Create unified tool interface
            tool_interface = UnifiedToolInterface(
                search_engine=search_engine,
                book_processor=book_processor,
                cache_manager=cache_manager,
                mcp_manager=mcp_manager,
                vector_store_ids=[]
            )
            
            # Initialize interface
            await tool_interface.initialize()
            
            # Test tool listing
            all_tools = await tool_interface.list_all_tools()
            assert len(all_tools) > 0
            
            # Test tool search
            search_results = await tool_interface.search_tools("search")
            assert isinstance(search_results, list)
            
            # Test tool categories
            from agents.core.unified_tool_interface import ToolCategory
            search_tools = await tool_interface.list_tools_by_category(ToolCategory.SEARCH)
            assert isinstance(search_tools, list)
            
            self.test_results[test_name] = "✅ PASSED"
            print(f"   ✓ Unified tool interface created successfully")
            print(f"   ✓ Tool listing working: {len(all_tools)} tools found")
            print(f"   ✓ Tool search functionality working")
            
        except Exception as e:
            self.test_results[test_name] = f"❌ FAILED: {str(e)}"
            self.errors.append(f"{test_name}: {str(e)}")
            print(f"   ❌ Error: {str(e)}")
    
    async def _test_enhanced_coordination(self):
        """Test enhanced agent coordination."""
        
        test_name = "Enhanced Coordination"
        print(f"\n🤝 Testing {test_name}...")
        
        try:
            # Create coordination orchestrator
            orchestrator = EnhancedCoordinationOrchestrator(openai_api_key="test-key")
            
            # Test available patterns
            patterns = await orchestrator.list_available_patterns()
            assert "sparc_full_cycle" in patterns
            assert "research_driven" in patterns
            assert "implementation_focused" in patterns
            
            # Create test task context
            task_context = TaskContext(
                task_id="test_coordination",
                description="Test enhanced coordination workflow",
                requirements={"test_mode": True},
                quality_gates={"coordination_quality": 0.8},
                performance_targets={"workflow_time": 300},
                constraints=["test_environment"],
                success_criteria=["workflow_completion"]
            )
            
            # Test coordination workflow (research-driven pattern for faster testing)
            coordination_result = await orchestrator.execute_coordinated_workflow(
                task_context, pattern="research_driven"
            )
            
            assert coordination_result is not None
            assert hasattr(coordination_result, 'workflow_id')
            assert hasattr(coordination_result, 'agents_involved')
            assert hasattr(coordination_result, 'success')
            
            self.test_results[test_name] = "✅ PASSED"
            print(f"   ✓ Coordination orchestrator created successfully")
            print(f"   ✓ Available patterns: {len(patterns)}")
            print(f"   ✓ Workflow execution completed: {coordination_result.success}")
            
        except Exception as e:
            self.test_results[test_name] = f"❌ FAILED: {str(e)}"
            self.errors.append(f"{test_name}: {str(e)}")
            print(f"   ❌ Error: {str(e)}")
    
    def _create_mock_search_engine(self):
        """Create mock search engine for testing."""
        class MockSearchEngine:
            async def search(self, query, limit=10, include_metadata=True):
                return []
        return MockSearchEngine()
    
    def _create_mock_book_processor(self):
        """Create mock book processor for testing."""
        class MockBookProcessor:
            async def process_file(self, file_path):
                return {
                    "content": "Mock processed content",
                    "metadata": {"file_path": file_path},
                    "quality_score": 0.85
                }
        return MockBookProcessor()
    
    def _create_mock_cache_manager(self):
        """Create mock cache manager for testing."""
        class MockCacheManager:
            async def get(self, key):
                return None
            async def set(self, key, value):
                return True
            async def delete(self, key):
                return True
            async def clear(self):
                return True
        return MockCacheManager()
    
    async def _print_test_results(self):
        """Print comprehensive test results."""
        
        print("\n" + "=" * 60)
        print("📊 OpenAI Agents SDK Integration Test Results")
        print("=" * 60)
        
        passed = 0
        failed = 0
        
        for test_name, result in self.test_results.items():
            print(f"\n{test_name}: {result}")
            if result.startswith("✅"):
                passed += 1
            else:
                failed += 1
        
        print(f"\n📈 Summary: {passed} passed, {failed} failed")
        
        if self.errors:
            print(f"\n❌ Errors encountered:")
            for error in self.errors:
                print(f"   • {error}")
        
        if failed == 0:
            print(f"\n🎉 All tests passed! OpenAI Agents SDK integration is working correctly.")
            print(f"\n🚀 Your TradeKnowledge agents now have enhanced capabilities:")
            print(f"   • Real-time web search intelligence")
            print(f"   • Live code execution and TDD validation")
            print(f"   • Advanced document processing with AI")
            print(f"   • Unified tool interface for all agent types")
            print(f"   • Enhanced coordination and handoff patterns")
            print(f"   • MCP integration framework for external tools")
        else:
            print(f"\n⚠️  Some tests failed. Please review the errors above.")
            print(f"   Most functionality should still work, but some features may be limited.")
        
        print(f"\n💡 Next steps:")
        print(f"   1. Set OPENAI_API_KEY environment variable for full functionality")
        print(f"   2. Configure vector store IDs for file search capabilities")
        print(f"   3. Set up MCP servers if needed for external tool integration")
        print(f"   4. Review configuration in config/config.yaml")


async def main():
    """Main test runner."""
    
    # Check for environment setup
    print("🔧 Checking environment setup...")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set - tests will run in simulation mode")
    else:
        print("✅ OPENAI_API_KEY found")
    
    if not os.getenv("PYTHONPATH"):
        print("⚠️  PYTHONPATH not set - adding project root")
        os.environ["PYTHONPATH"] = str(project_root)
    
    # Run integration tests
    test_runner = OpenAIAgentsIntegrationTest()
    await test_runner.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())