"""
BDD Testing Framework for SPARC Agent Interactions
Behavior-driven testing of agent collaborations and workflows
"""

from typing import Any, Dict, List, Optional, Protocol, Callable
from dataclasses import dataclass, field
from enum import Enum
from unittest.mock import Mock, AsyncMock
import pytest
import asyncio
from abc import ABC, abstractmethod

from .behavior_verification import BehaviorVerifier, CollaborationPattern
from .outside_in_patterns import TestScenario, AcceptanceTestRunner


class AgentRole(Enum):
    """SPARC Agent roles"""
    RESEARCHER = "researcher"
    MASTERMIND = "mastermind"  
    EXECUTOR = "executor"


class AgentInteractionType(Enum):
    """Types of agent interactions"""
    QUERY = "query"
    ANALYSIS = "analysis"
    HANDOFF = "handoff"
    COLLABORATION = "collaboration"
    FEEDBACK = "feedback"


@dataclass
class AgentMessage:
    """Message passed between agents"""
    from_agent: AgentRole
    to_agent: AgentRole
    interaction_type: AgentInteractionType
    content: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None


@dataclass
class AgentScenario:
    """BDD scenario for agent interactions"""
    name: str
    description: str
    participants: List[AgentRole] = field(default_factory=list)
    messages: List[AgentMessage] = field(default_factory=list)
    expected_outcomes: List[Dict[str, Any]] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)


class SPARCWorkflowTester:
    """Testing framework for SPARC agent workflows"""
    
    def __init__(self):
        self.agent_mocks: Dict[AgentRole, Mock] = {}
        self.behavior_verifier = BehaviorVerifier()
        self.scenarios: List[AgentScenario] = []
        self.current_scenario: Optional[AgentScenario] = None
        
    def create_agent_mock(self, role: AgentRole, spec: type = None) -> Mock:
        """Create mock for specific agent role"""
        if spec:
            agent_mock = Mock(spec=spec)
        else:
            agent_mock = Mock()
        
        # Setup common agent methods
        agent_mock.process_query = AsyncMock()
        agent_mock.analyze_data = AsyncMock()
        agent_mock.collaborate = AsyncMock()
        agent_mock.provide_feedback = AsyncMock()
        
        self.agent_mocks[role] = agent_mock
        self.behavior_verifier.register_collaborator(role.value, agent_mock)
        
        return agent_mock
    
    def scenario(self, name: str, description: str = "") -> 'SPARCWorkflowTester':
        """Start defining a new agent scenario"""
        self.current_scenario = AgentScenario(
            name=name,
            description=description
        )
        self.scenarios.append(self.current_scenario)
        return self
    
    def given_agent(self, role: AgentRole, **initial_state) -> 'SPARCWorkflowTester':
        """Set up agent preconditions"""
        if not self.current_scenario:
            raise ValueError("Must call scenario() first")
        
        self.current_scenario.participants.append(role)
        self.current_scenario.preconditions.append(
            f"Agent {role.value} is initialized with state: {initial_state}"
        )
        
        # Configure agent mock with initial state
        if role not in self.agent_mocks:
            self.create_agent_mock(role)
        
        agent_mock = self.agent_mocks[role]
        for attr, value in initial_state.items():
            setattr(agent_mock, attr, value)
        
        return self
    
    def when_agent_sends(
        self, 
        from_agent: AgentRole, 
        to_agent: AgentRole,
        interaction_type: AgentInteractionType,
        **content
    ) -> 'SPARCWorkflowTester':
        """Define agent message sending"""
        if not self.current_scenario:
            raise ValueError("Must call scenario() first")
        
        message = AgentMessage(
            from_agent=from_agent,
            to_agent=to_agent,
            interaction_type=interaction_type,
            content=content
        )
        
        self.current_scenario.messages.append(message)
        return self
    
    def then_agent_should(
        self, 
        agent: AgentRole, 
        behavior: str, 
        **expectations
    ) -> 'SPARCWorkflowTester':
        """Define expected agent behavior"""
        if not self.current_scenario:
            raise ValueError("Must call scenario() first")
        
        self.current_scenario.expected_outcomes.append({
            'agent': agent,
            'behavior': behavior,
            'expectations': expectations
        })
        return self
    
    def and_collaboration_should_follow(self, pattern_name: str) -> 'SPARCWorkflowTester':
        """Define expected collaboration pattern"""
        if not self.current_scenario:
            raise ValueError("Must call scenario() first")
        
        self.current_scenario.postconditions.append(
            f"Collaboration should follow pattern: {pattern_name}"
        )
        return self


class TrioCollaborationTester:
    """Specialized tester for SPARC trio collaborations"""
    
    def __init__(self):
        self.researcher_mock = Mock()
        self.mastermind_mock = Mock()
        self.executor_mock = Mock()
        self.collaboration_patterns: Dict[str, CollaborationPattern] = {}
        
        # Setup async methods for all agents
        for agent_mock in [self.researcher_mock, self.mastermind_mock, self.executor_mock]:
            agent_mock.process_request = AsyncMock()
            agent_mock.analyze = AsyncMock()
            agent_mock.execute_task = AsyncMock()
            agent_mock.provide_feedback = AsyncMock()
            agent_mock.handoff_to = AsyncMock()
    
    def define_research_analysis_pattern(self) -> CollaborationPattern:
        """Define the research -> analysis collaboration pattern"""
        pattern = CollaborationPattern("research_analysis_handoff")
        
        pattern.expect_async_interaction("researcher", "process_request") \
               .expect_async_interaction("researcher", "handoff_to", "mastermind") \
               .expect_async_interaction("mastermind", "analyze") \
               .expect_async_interaction("mastermind", "provide_feedback", "researcher")
        
        self.collaboration_patterns["research_analysis"] = pattern
        return pattern
    
    def define_analysis_execution_pattern(self) -> CollaborationPattern:
        """Define the analysis -> execution collaboration pattern"""
        pattern = CollaborationPattern("analysis_execution_handoff")
        
        pattern.expect_async_interaction("mastermind", "analyze") \
               .expect_async_interaction("mastermind", "handoff_to", "executor") \
               .expect_async_interaction("executor", "execute_task") \
               .expect_async_interaction("executor", "provide_feedback", "mastermind")
        
        self.collaboration_patterns["analysis_execution"] = pattern
        return pattern
    
    def define_full_trio_pattern(self) -> CollaborationPattern:
        """Define complete trio collaboration pattern"""
        pattern = CollaborationPattern("full_trio_collaboration")
        
        pattern.expect_async_interaction("researcher", "process_request") \
               .expect_async_interaction("researcher", "handoff_to", "mastermind") \
               .expect_async_interaction("mastermind", "analyze") \
               .expect_async_interaction("mastermind", "handoff_to", "executor") \
               .expect_async_interaction("executor", "execute_task") \
               .expect_async_interaction("executor", "provide_feedback", "mastermind") \
               .expect_async_interaction("mastermind", "provide_feedback", "researcher")
        
        self.collaboration_patterns["full_trio"] = pattern
        return pattern
    
    async def test_trio_collaboration(self, scenario_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test complete trio collaboration scenario"""
        results = {
            'scenario': scenario_name,
            'input_data': input_data,
            'agent_interactions': {},
            'collaboration_verified': False,
            'errors': []
        }
        
        try:
            # Simulate trio workflow
            researcher_result = await self.researcher_mock.process_request(input_data)
            await self.researcher_mock.handoff_to("mastermind", researcher_result)
            
            mastermind_result = await self.mastermind_mock.analyze(researcher_result)
            await self.mastermind_mock.handoff_to("executor", mastermind_result)
            
            executor_result = await self.executor_mock.execute_task(mastermind_result)
            await self.executor_mock.provide_feedback("mastermind", executor_result)
            
            await self.mastermind_mock.provide_feedback("researcher", executor_result)
            
            # Record interactions
            results['agent_interactions'] = {
                'researcher_calls': self.researcher_mock.method_calls,
                'mastermind_calls': self.mastermind_mock.method_calls,
                'executor_calls': self.executor_mock.method_calls
            }
            
            # Verify collaboration pattern
            if "full_trio" in self.collaboration_patterns:
                behavior_verifier = BehaviorVerifier()
                behavior_verifier.register_collaborator("researcher", self.researcher_mock)
                behavior_verifier.register_collaborator("mastermind", self.mastermind_mock)
                behavior_verifier.register_collaborator("executor", self.executor_mock)
                
                behavior_verifier.verify_pattern(self.collaboration_patterns["full_trio"])
                results['collaboration_verified'] = True
            
        except Exception as e:
            results['errors'].append(str(e))
        
        return results


class PromptManagementBDD:
    """BDD testing for prompt management with SPARC agents"""
    
    def __init__(self):
        self.prompt_service_mock = Mock()
        self.database_mock = Mock()
        self.agent_mocks: Dict[AgentRole, Mock] = {}
        
        # Setup prompt service methods
        self.prompt_service_mock.create_prompt = AsyncMock()
        self.prompt_service_mock.search_prompts = AsyncMock()
        self.prompt_service_mock.analyze_prompt_performance = AsyncMock()
        self.prompt_service_mock.suggest_improvements = AsyncMock()
    
    def scenario_researcher_analyzes_prompt_performance(self) -> AgentScenario:
        """BDD scenario: Researcher analyzes prompt performance data"""
        scenario = AgentScenario(
            name="researcher_analyzes_prompt_performance",
            description="Researcher agent analyzes historical prompt performance data",
            participants=[AgentRole.RESEARCHER]
        )
        
        scenario.preconditions = [
            "Prompt usage data exists in InfluxDB",
            "Performance metrics are available",
            "Researcher agent has access to analytics tools"
        ]
        
        scenario.postconditions = [
            "Performance analysis is generated",
            "Insights are provided to other agents",
            "Recommendations are stored for future use"
        ]
        
        return scenario
    
    def scenario_mastermind_designs_prompt_optimization(self) -> AgentScenario:
        """BDD scenario: Mastermind designs prompt optimization strategy"""
        scenario = AgentScenario(
            name="mastermind_designs_optimization",
            description="Mastermind agent creates strategy for prompt optimization",
            participants=[AgentRole.MASTERMIND, AgentRole.RESEARCHER]
        )
        
        scenario.preconditions = [
            "Research analysis is available from Researcher",
            "Prompt performance data shows improvement opportunities",
            "Optimization goals are defined"
        ]
        
        scenario.postconditions = [
            "Optimization strategy is created",
            "Implementation plan is generated",
            "Success metrics are defined"
        ]
        
        return scenario
    
    def scenario_executor_implements_prompt_testing(self) -> AgentScenario:
        """BDD scenario: Executor implements prompt testing framework"""
        scenario = AgentScenario(
            name="executor_implements_testing",
            description="Executor agent implements automated prompt testing",
            participants=[AgentRole.EXECUTOR, AgentRole.MASTERMIND]
        )
        
        scenario.preconditions = [
            "Testing strategy is provided by Mastermind",
            "Test environment is available",
            "Quality gates are defined"
        ]
        
        scenario.postconditions = [
            "Automated tests are implemented",
            "Quality gates are enforced",
            "Continuous testing pipeline is active"
        ]
        
        return scenario
    
    async def test_prompt_lifecycle_collaboration(self) -> Dict[str, Any]:
        """Test complete prompt lifecycle with all three agents"""
        results = {
            'lifecycle_stages': [],
            'agent_collaborations': [],
            'quality_gates_passed': False,
            'errors': []
        }
        
        try:
            # Stage 1: Research existing prompts
            researcher_analysis = await self._simulate_researcher_analysis()
            results['lifecycle_stages'].append('research_completed')
            
            # Stage 2: Design optimization strategy
            optimization_strategy = await self._simulate_mastermind_strategy(researcher_analysis)
            results['lifecycle_stages'].append('strategy_designed')
            
            # Stage 3: Implement and test improvements
            implementation_result = await self._simulate_executor_implementation(optimization_strategy)
            results['lifecycle_stages'].append('implementation_completed')
            
            # Verify quality gates
            if self._verify_quality_gates(implementation_result):
                results['quality_gates_passed'] = True
            
        except Exception as e:
            results['errors'].append(str(e))
        
        return results
    
    async def _simulate_researcher_analysis(self) -> Dict[str, Any]:
        """Simulate researcher agent analysis"""
        # Mock researcher gathering and analyzing prompt data
        await self.prompt_service_mock.search_prompts(filters={'performance': 'low'})
        analysis_result = {
            'low_performing_prompts': 15,
            'common_issues': ['unclear instructions', 'missing context'],
            'improvement_opportunities': ['better examples', 'clearer structure']
        }
        return analysis_result
    
    async def _simulate_mastermind_strategy(self, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate mastermind agent strategy creation"""
        strategy = {
            'optimization_goals': ['improve clarity', 'add examples'],
            'implementation_plan': ['template creation', 'automated testing'],
            'success_metrics': ['performance_score > 0.8', 'user_rating > 4.0']
        }
        return strategy
    
    async def _simulate_executor_implementation(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate executor agent implementation"""
        # Mock implementation of testing framework
        await self.prompt_service_mock.create_prompt({
            'title': 'Optimized Template',
            'content': 'Improved prompt content',
            'metadata': {'optimization_applied': True}
        })
        
        implementation = {
            'tests_created': 10,
            'quality_gates_implemented': True,
            'performance_improvement': 0.25
        }
        return implementation
    
    def _verify_quality_gates(self, implementation: Dict[str, Any]) -> bool:
        """Verify quality gates are met"""
        required_gates = [
            implementation.get('tests_created', 0) > 5,
            implementation.get('quality_gates_implemented', False),
            implementation.get('performance_improvement', 0) > 0.1
        ]
        return all(required_gates)


# Pytest fixtures for SPARC agent BDD testing
@pytest.fixture
def sparc_workflow_tester():
    """Fixture providing SPARC workflow tester"""
    return SPARCWorkflowTester()


@pytest.fixture
def trio_collaboration_tester():
    """Fixture providing trio collaboration tester"""
    return TrioCollaborationTester()


@pytest.fixture
def prompt_management_bdd():
    """Fixture providing prompt management BDD tester"""
    return PromptManagementBDD()


@pytest.fixture
def researcher_agent_mock():
    """Fixture providing researcher agent mock"""
    mock = Mock()
    mock.analyze_data = AsyncMock()
    mock.gather_intelligence = AsyncMock()
    mock.provide_insights = AsyncMock()
    return mock


@pytest.fixture
def mastermind_agent_mock():
    """Fixture providing mastermind agent mock"""
    mock = Mock()
    mock.create_strategy = AsyncMock()
    mock.design_architecture = AsyncMock()
    mock.orchestrate_workflow = AsyncMock()
    return mock


@pytest.fixture
def executor_agent_mock():
    """Fixture providing executor agent mock"""
    mock = Mock()
    mock.implement_solution = AsyncMock()
    mock.run_tests = AsyncMock()
    mock.deploy_system = AsyncMock()
    return mock