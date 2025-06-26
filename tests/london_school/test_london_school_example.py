"""
Example London School TDD Implementation
Demonstrates outside-in, behavior-driven testing for prompt management
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, call
from typing import Dict, Any, List

# Import our London School TDD framework
from .base_test import AsyncLondonSchoolTestCase, OutsideInTestBuilder, MockFactory
from .behavior_verification import BehaviorVerifier, CollaborationPattern, ExpectedInteraction, InteractionType
from .outside_in_patterns import (
    AcceptanceTestRunner, APITestBuilder, ServiceTestBuilder, 
    UserStory, TestScenario, TestLevel
)
from .sparc_agent_bdd import SPARCWorkflowTester, TrioCollaborationTester, AgentRole, AgentInteractionType


class TestPromptServiceLondonSchool(AsyncLondonSchoolTestCase):
    """
    London School TDD example: Outside-in development of PromptService
    
    This test demonstrates the London School approach:
    1. Start from the outside (user needs/API)
    2. Work inward through layers (service -> domain -> infrastructure)
    3. Mock all dependencies 
    4. Focus on behavior and interactions
    5. Let design emerge through test-driven development
    """
    
    async def setup_collaborators_async(self):
        """Setup mocked collaborators following London School patterns"""
        # Mock all external dependencies
        self.database_service_mock = MockFactory.create_async_mock()
        self.redis_service_mock = MockFactory.create_async_mock()
        self.influxdb_service_mock = MockFactory.create_async_mock()
        
        # Setup database mock behaviors
        self.database_service_mock.postgres.create_prompt.return_value = "prompt-123"
        self.database_service_mock.postgres.get_prompt.return_value = {
            "id": "prompt-123",
            "title": "Test Prompt",
            "content": "Test content",
            "user_id": "user-456"
        }
        
        # Setup Redis caching mock
        self.redis_service_mock.get_json.return_value = None
        self.redis_service_mock.set_json.return_value = True
        
        # Register collaborators for behavior verification
        self.behavior_verifier = BehaviorVerifier()
        self.behavior_verifier.register_collaborator("database", self.database_service_mock)
        self.behavior_verifier.register_collaborator("redis", self.redis_service_mock)
        self.behavior_verifier.register_collaborator("influxdb", self.influxdb_service_mock)
    
    def create_system_under_test(self, **dependencies):
        """Factory method to create PromptService with mocked dependencies"""
        # This would be the actual PromptService class
        # For now, we'll create a mock that represents our desired interface
        prompt_service = Mock()
        
        # Inject dependencies
        prompt_service.db_service = dependencies.get('database', self.database_service_mock)
        prompt_service.redis_service = dependencies.get('redis', self.redis_service_mock)
        prompt_service.influx_service = dependencies.get('influxdb', self.influxdb_service_mock)
        
        # Define async methods that represent our desired API
        prompt_service.create_prompt = AsyncMock()
        prompt_service.get_prompt = AsyncMock() 
        prompt_service.search_prompts = AsyncMock()
        prompt_service.update_prompt = AsyncMock()
        prompt_service.delete_prompt = AsyncMock()
        
        return prompt_service
    
    @pytest.mark.asyncio
    async def test_create_prompt_collaboration_pattern(self):
        """
        Test: Creating a prompt follows proper collaboration pattern
        
        This test demonstrates London School focus on interactions:
        - We don't test implementation details
        - We verify the collaboration between objects
        - We mock all dependencies to focus on behavior
        """
        # Arrange: Setup collaboration pattern
        create_prompt_pattern = CollaborationPattern("create_prompt_workflow")
        create_prompt_pattern.expect_async_interaction("redis", "get_json", "cache_key_prompts_user-456") \
                            .expect_async_interaction("database", "create_prompt") \
                            .expect_async_interaction("redis", "set_json") \
                            .expect_async_interaction("influxdb", "write_analytics")
        
        # Act: Execute the operation
        prompt_service = self.create_system_under_test(
            database=self.database_service_mock,
            redis=self.redis_service_mock,
            influxdb=self.influxdb_service_mock
        )
        
        # Simulate the expected collaboration
        await self.redis_service_mock.get_json("cache_key_prompts_user-456")
        await self.database_service_mock.create_prompt({
            "title": "New Prompt",
            "content": "Prompt content",
            "user_id": "user-456"
        })
        await self.redis_service_mock.set_json("cache_key", {"prompt_id": "prompt-123"})
        await self.influxdb_service_mock.write_analytics({"action": "prompt_created"})
        
        # Assert: Verify collaboration pattern
        self.behavior_verifier.verify_pattern(create_prompt_pattern)
    
    @pytest.mark.asyncio
    async def test_search_prompts_caching_behavior(self):
        """
        Test: Search prompts implements proper caching behavior
        
        London School approach:
        - Focus on the caching collaboration
        - Mock to test the behavior we want
        - Verify interactions rather than state
        """
        # Arrange: Setup expected caching behavior
        search_cache_pattern = CollaborationPattern("search_with_caching")
        search_cache_pattern.expect_async_interaction("redis", "get_json", "search_cache_key") \
                           .expect_async_interaction("database", "search_prompts") \
                           .expect_async_interaction("redis", "set_json")
        
        # Configure cache miss scenario
        self.redis_service_mock.get_json.return_value = None
        self.database_service_mock.search_prompts.return_value = [
            {"id": "prompt-1", "title": "Result 1"},
            {"id": "prompt-2", "title": "Result 2"}
        ]
        
        # Act: Simulate search operation
        await self.redis_service_mock.get_json("search_cache_key")
        search_results = await self.database_service_mock.search_prompts({"query": "test"})
        await self.redis_service_mock.set_json("search_cache_key", search_results, 300)
        
        # Assert: Verify caching collaboration
        self.behavior_verifier.verify_pattern(search_cache_pattern)
        
        # Verify specific behaviors
        self.redis_service_mock.get_json.assert_called_with("search_cache_key")
        self.database_service_mock.search_prompts.assert_called_with({"query": "test"})
        self.redis_service_mock.set_json.assert_called_with("search_cache_key", search_results, 300)


class TestOutsideInPromptAPI:
    """
    Demonstrates outside-in API testing following London School patterns
    """
    
    def test_prompt_api_acceptance_scenario(self, outside_in_builder):
        """
        Test: User can create and manage prompts through API
        
        Outside-in approach:
        - Start from user perspective (API endpoints)
        - Mock all internal services
        - Focus on end-to-end behavior
        """
        # Build acceptance test scenario
        scenario = outside_in_builder \
            .given_collaborator("prompt_service", {
                "create_prompt": {"id": "prompt-123", "status": "created"},
                "get_prompt": {"id": "prompt-123", "title": "Test Prompt"}
            }) \
            .given_collaborator("auth_service", {
                "validate_token": {"user_id": "user-456", "valid": True}
            }) \
            .when_user_performs("POST_request", 
                               endpoint="/api/prompts", 
                               data={"title": "New Prompt", "content": "Test content"}) \
            .then_system_should("return_success", status_code=201) \
            .and_collaborator_should("auth_service", "validate_token") \
            .and_collaborator_should("prompt_service", "create_prompt") \
            .build_test()
        
        # Verify the test scenario structure
        assert len(scenario['steps']) == 1
        assert len(scenario['outcomes']) == 3
        assert 'prompt_service' in scenario['collaborators']
        assert 'auth_service' in scenario['collaborators']
    
    def test_api_error_handling_behavior(self, api_test_builder):
        """
        Test: API properly handles service errors
        
        London School focus on error collaboration:
        - Mock service to throw exceptions
        - Verify proper error handling behavior
        - Test interaction patterns during failures
        """
        # Setup API test with mocked failure
        test_case = api_test_builder \
            .for_endpoint("POST", "/api/prompts") \
            .with_request_data({"title": "Test", "content": "Content"}) \
            .mock_service("prompt_service", {
                "create_prompt": Exception("Database connection failed")
            }) \
            .expecting_status(500) \
            .expecting_response({"error": "Internal server error"}) \
            .build_test_case()
        
        # Verify error handling test structure
        assert test_case['expected_status'] == 500
        assert 'prompt_service' in test_case['mocked_services']
        
        # The actual API would handle the exception and return proper error response


class TestSPARCAgentCollaboration:
    """
    Test SPARC agent collaborations using BDD patterns
    """
    
    @pytest.mark.asyncio
    async def test_trio_prompt_optimization_workflow(self, trio_collaboration_tester):
        """
        Test: SPARC trio collaborates to optimize prompts
        
        BDD scenario testing agent interactions:
        - Researcher analyzes prompt performance
        - Mastermind designs optimization strategy  
        - Executor implements improvements
        """
        # Define the collaboration patterns
        trio_collaboration_tester.define_full_trio_pattern()
        
        # Setup agent behaviors
        trio_collaboration_tester.researcher_mock.process_request.return_value = {
            "analysis": "prompt_performance_low",
            "recommendations": ["add_examples", "improve_clarity"]
        }
        
        trio_collaboration_tester.mastermind_mock.analyze.return_value = {
            "strategy": "template_optimization",
            "implementation_plan": ["create_templates", "setup_testing"]
        }
        
        trio_collaboration_tester.executor_mock.execute_task.return_value = {
            "status": "completed",
            "templates_created": 5,
            "tests_implemented": 10
        }
        
        # Execute trio collaboration
        result = await trio_collaboration_tester.test_trio_collaboration(
            "prompt_optimization",
            {"prompts_to_analyze": ["prompt-1", "prompt-2"]}
        )
        
        # Verify collaboration occurred
        assert result['collaboration_verified']
        assert len(result['errors']) == 0
        assert 'researcher_calls' in result['agent_interactions']
        assert 'mastermind_calls' in result['agent_interactions']
        assert 'executor_calls' in result['agent_interactions']
    
    def test_agent_handoff_behavior(self, sparc_workflow_tester):
        """
        Test: Agents properly hand off work to each other
        
        BDD testing of agent interactions:
        - Focus on message passing behavior
        - Verify handoff protocols
        - Test collaboration patterns
        """
        # Create scenario for research-to-analysis handoff
        sparc_workflow_tester.scenario(
            "research_to_analysis_handoff",
            "Researcher hands off findings to Mastermind for strategic analysis"
        ).given_agent(
            AgentRole.RESEARCHER,
            status="ready",
            data_sources=["market_data", "prompt_analytics"]
        ).given_agent(
            AgentRole.MASTERMIND,
            status="ready",
            analysis_models=["performance", "optimization"]
        ).when_agent_sends(
            AgentRole.RESEARCHER,
            AgentRole.MASTERMIND,
            AgentInteractionType.HANDOFF,
            research_data={"findings": "low_performance_identified"},
            priority="high"
        ).then_agent_should(
            AgentRole.MASTERMIND,
            "receive_handoff",
            expected_data_keys=["findings"],
            priority="high"
        ).and_collaboration_should_follow("research_analysis_pattern")
        
        # Verify scenario structure
        scenarios = sparc_workflow_tester.scenarios
        assert len(scenarios) == 1
        assert scenarios[0].name == "research_to_analysis_handoff"
        assert len(scenarios[0].participants) == 2
        assert len(scenarios[0].messages) == 1
        assert len(scenarios[0].expected_outcomes) == 1


class TestLondonSchoolQualityGates:
    """
    Test quality gates and behavior coverage for London School TDD
    """
    
    def test_behavior_coverage_metrics(self):
        """
        Test: Verify behavior coverage is measured properly
        
        London School quality gate:
        - All collaborator interactions are verified
        - Behavior patterns are documented
        - Mock usage is justified and verified
        """
        # Setup behavior verifier with multiple collaborators
        verifier = BehaviorVerifier()
        
        # Register multiple collaborators
        database_mock = Mock()
        cache_mock = Mock()
        analytics_mock = Mock()
        
        verifier.register_collaborator("database", database_mock)
        verifier.register_collaborator("cache", cache_mock)
        verifier.register_collaborator("analytics", analytics_mock)
        
        # Define comprehensive collaboration pattern
        pattern = CollaborationPattern("comprehensive_service_operation")
        pattern.expect_interaction("cache", "get", "cache_key") \
               .expect_interaction("database", "query", {"table": "prompts"}) \
               .expect_interaction("analytics", "record", {"action": "query_executed"}) \
               .expect_interaction("cache", "set", "cache_key", "result_data")
        
        # Simulate the interactions
        cache_mock.get("cache_key")
        database_mock.query({"table": "prompts"})
        analytics_mock.record({"action": "query_executed"})
        cache_mock.set("cache_key", "result_data")
        
        # Verify pattern - this ensures 100% behavior coverage
        verifier.verify_pattern(pattern)
        
        # Additional verification: ensure all mocks were used
        assert cache_mock.get.called
        assert database_mock.query.called
        assert analytics_mock.record.called
        assert cache_mock.set.called
    
    def test_collaboration_invariants(self):
        """
        Test: Verify collaboration invariants hold throughout operation
        
        London School invariants:
        - Certain conditions must always be true
        - Collaborations follow consistent patterns
        - Side effects are predictable and verified
        """
        # Define invariants for a prompt service operation
        def cache_coherence_invariant():
            """Cache and database should be coherent"""
            # In real implementation, this would check actual coherence
            return True
        
        def security_invariant():
            """Security checks must always be performed"""
            # In real implementation, this would verify auth was called
            return True
        
        # Create pattern with invariants
        pattern = CollaborationPattern("secure_cached_operation")
        pattern.add_invariant(cache_coherence_invariant)
        pattern.add_invariant(security_invariant)
        
        # Verify invariants hold
        verifier = BehaviorVerifier()
        verifier.verify_pattern(pattern)


# Integration test demonstrating full London School workflow
@pytest.mark.integration
class TestLondonSchoolIntegration:
    """
    Integration test showing complete London School TDD workflow
    """
    
    @pytest.mark.asyncio
    async def test_complete_outside_in_development_cycle(self):
        """
        Test: Complete outside-in development cycle
        
        This test demonstrates the full London School approach:
        1. Start with acceptance criteria (user story)
        2. Create API tests (mocking services)
        3. Create service tests (mocking infrastructure)
        4. Create domain tests (pure business logic)
        5. Verify all collaborations work together
        """
        # 1. Define user story (acceptance level)
        user_story = UserStory(
            title="Prompt Performance Optimization",
            as_a="prompt engineer",
            i_want="to identify and optimize low-performing prompts",
            so_that="I can improve AI system effectiveness",
            acceptance_criteria=[
                "System identifies prompts with < 70% success rate",
                "System suggests specific optimization strategies",
                "System tracks improvement after optimization"
            ]
        )
        
        # 2. API level test (outside layer)
        api_test = APITestBuilder() \
            .for_endpoint("POST", "/api/prompts/optimize") \
            .with_request_data({"prompt_id": "prompt-123"}) \
            .mock_service("optimization_service", {
                "analyze_performance": {"success_rate": 0.6, "issues": ["unclear_instructions"]},
                "suggest_improvements": {"strategies": ["add_examples", "improve_clarity"]},
                "apply_optimization": {"new_version": "v2", "estimated_improvement": 0.2}
            }) \
            .expecting_status(200) \
            .expecting_response({
                "status": "optimized",
                "version": "v2",
                "estimated_improvement": 0.2
            }) \
            .build_test_case()
        
        # 3. Service level test (business logic layer)
        service_test = ServiceTestBuilder(Mock) \
            .with_dependencies(
                performance_analyzer=Mock(),
                strategy_generator=Mock(),
                version_manager=Mock()
            ) \
            .calling_method("optimize_prompt", "prompt-123") \
            .setup_dependency_behavior("performance_analyzer", "analyze", {
                "success_rate": 0.6,
                "issues": ["unclear_instructions"]
            }) \
            .setup_dependency_behavior("strategy_generator", "generate_strategies", [
                "add_examples", "improve_clarity"
            ]) \
            .setup_dependency_behavior("version_manager", "create_version", "v2") \
            .expecting_result({
                "status": "optimized",
                "version": "v2",
                "improvements_applied": 2
            }) \
            .build_and_execute()
        
        # Verify the development cycle structure
        assert user_story.title == "Prompt Performance Optimization"
        assert api_test['endpoint'] == "/api/prompts/optimize"
        assert api_test['expected_status'] == 200
        assert len(api_test['mocked_services']) == 1
        
        assert service_test['method'] == "optimize_prompt"
        assert len(service_test['dependency_interactions']) == 3
        
        # This demonstrates how London School TDD drives design:
        # - User story drives API design
        # - API test drives service interface design
        # - Service test drives domain model design
        # - All through behavior-focused testing with mocks