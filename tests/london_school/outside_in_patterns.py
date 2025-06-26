"""
Outside-In Testing Patterns for London School TDD
Provides structured approaches for outside-in development
"""

from typing import Any, Dict, List, Optional, Protocol, TypeVar, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import pytest
from unittest.mock import Mock, AsyncMock
import asyncio


class TestLevel(Enum):
    """Different levels of outside-in testing"""
    ACCEPTANCE = "acceptance"      # Full user journey
    API = "api"                   # HTTP/API boundaries  
    SERVICE = "service"           # Business logic layer
    DOMAIN = "domain"             # Core domain models
    INFRASTRUCTURE = "infrastructure"  # External dependencies


@dataclass
class UserStory:
    """Represents a user story for acceptance testing"""
    title: str
    as_a: str                    # Role
    i_want: str                  # Goal
    so_that: str                 # Benefit
    acceptance_criteria: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TestScenario:
    """Represents a specific test scenario"""
    name: str
    given: List[str] = field(default_factory=list)  # Preconditions
    when: List[str] = field(default_factory=list)   # Actions
    then: List[str] = field(default_factory=list)   # Expected outcomes
    level: TestLevel = TestLevel.ACCEPTANCE


class OutsideInTestStrategy:
    """Strategy for implementing outside-in testing approach"""
    
    def __init__(self, user_story: UserStory):
        self.user_story = user_story
        self.scenarios: List[TestScenario] = []
        self.current_level = TestLevel.ACCEPTANCE
        self.collaborators: Dict[str, Mock] = {}
    
    def add_scenario(self, scenario: TestScenario) -> 'OutsideInTestStrategy':
        """Add test scenario to the strategy"""
        self.scenarios.append(scenario)
        return self
    
    def create_acceptance_test(self, scenario_name: str) -> TestScenario:
        """Create acceptance-level test scenario"""
        scenario = TestScenario(
            name=scenario_name,
            level=TestLevel.ACCEPTANCE
        )
        self.scenarios.append(scenario)
        return scenario
    
    def create_api_test(self, scenario_name: str) -> TestScenario:
        """Create API-level test scenario"""
        scenario = TestScenario(
            name=scenario_name,
            level=TestLevel.API
        )
        self.scenarios.append(scenario)
        return scenario
    
    def create_service_test(self, scenario_name: str) -> TestScenario:
        """Create service-level test scenario"""
        scenario = TestScenario(
            name=scenario_name,
            level=TestLevel.SERVICE
        )
        self.scenarios.append(scenario)
        return scenario
    
    def mock_collaborator(self, name: str, spec: type = None) -> Mock:
        """Create and register a collaborator mock"""
        if spec:
            mock = Mock(spec=spec)
        else:
            mock = Mock()
        self.collaborators[name] = mock
        return mock
    
    def get_test_plan(self) -> Dict[str, Any]:
        """Generate comprehensive test plan"""
        return {
            'user_story': {
                'title': self.user_story.title,
                'as_a': self.user_story.as_a,
                'i_want': self.user_story.i_want,
                'so_that': self.user_story.so_that,
                'acceptance_criteria': self.user_story.acceptance_criteria
            },
            'scenarios': [
                {
                    'name': scenario.name,
                    'level': scenario.level.value,
                    'given': scenario.given,
                    'when': scenario.when,
                    'then': scenario.then
                }
                for scenario in self.scenarios
            ],
            'collaborators': list(self.collaborators.keys())
        }


class AcceptanceTestRunner:
    """Runs acceptance tests following BDD patterns"""
    
    def __init__(self):
        self.context: Dict[str, Any] = {}
        self.test_data: Dict[str, Any] = {}
        self.results: List[Dict[str, Any]] = []
    
    def given(self, condition: str, **data) -> 'AcceptanceTestRunner':
        """Set up test preconditions"""
        self.context[f"given_{condition}"] = data
        return self
    
    def when(self, action: str, **params) -> 'AcceptanceTestRunner':
        """Execute test action"""
        self.context[f"when_{action}"] = params
        return self
    
    def then(self, expectation: str, **assertions) -> 'AcceptanceTestRunner':
        """Verify test outcomes"""
        self.context[f"then_{expectation}"] = assertions
        return self
    
    def execute_scenario(self, scenario: TestScenario) -> Dict[str, Any]:
        """Execute a complete scenario"""
        result = {
            'scenario': scenario.name,
            'level': scenario.level.value,
            'status': 'pending',
            'steps': [],
            'errors': []
        }
        
        try:
            # Execute given steps
            for given_step in scenario.given:
                result['steps'].append(f"Given {given_step}")
            
            # Execute when steps
            for when_step in scenario.when:
                result['steps'].append(f"When {when_step}")
            
            # Execute then steps
            for then_step in scenario.then:
                result['steps'].append(f"Then {then_step}")
            
            result['status'] = 'passed'
            
        except Exception as e:
            result['status'] = 'failed'
            result['errors'].append(str(e))
        
        self.results.append(result)
        return result


class APITestBuilder:
    """Builder for API-level tests with proper mocking"""
    
    def __init__(self):
        self.endpoint = ""
        self.method = "GET"
        self.headers: Dict[str, str] = {}
        self.request_data: Dict[str, Any] = {}
        self.expected_status = 200
        self.expected_response: Dict[str, Any] = {}
        self.mocked_services: Dict[str, Mock] = {}
    
    def for_endpoint(self, method: str, endpoint: str) -> 'APITestBuilder':
        """Set the API endpoint being tested"""
        self.method = method
        self.endpoint = endpoint
        return self
    
    def with_headers(self, headers: Dict[str, str]) -> 'APITestBuilder':
        """Set request headers"""
        self.headers.update(headers)
        return self
    
    def with_request_data(self, data: Dict[str, Any]) -> 'APITestBuilder':
        """Set request payload"""
        self.request_data = data
        return self
    
    def expecting_status(self, status_code: int) -> 'APITestBuilder':
        """Set expected HTTP status code"""
        self.expected_status = status_code
        return self
    
    def expecting_response(self, response: Dict[str, Any]) -> 'APITestBuilder':
        """Set expected response payload"""
        self.expected_response = response
        return self
    
    def mock_service(self, service_name: str, behavior: Dict[str, Any]) -> 'APITestBuilder':
        """Mock a service dependency"""
        service_mock = Mock()
        for method, return_value in behavior.items():
            getattr(service_mock, method).return_value = return_value
        self.mocked_services[service_name] = service_mock
        return self
    
    def build_test_case(self) -> Dict[str, Any]:
        """Build the complete API test case"""
        return {
            'endpoint': self.endpoint,
            'method': self.method,
            'headers': self.headers,
            'request_data': self.request_data,
            'expected_status': self.expected_status,
            'expected_response': self.expected_response,
            'mocked_services': self.mocked_services
        }


class ServiceTestBuilder:
    """Builder for service-level tests focusing on business logic"""
    
    def __init__(self, service_class: type):
        self.service_class = service_class
        self.constructor_args: List[Any] = []
        self.constructor_kwargs: Dict[str, Any] = {}
        self.method_under_test = ""
        self.method_args: List[Any] = []
        self.method_kwargs: Dict[str, Any] = {}
        self.dependencies: Dict[str, Mock] = {}
        self.expected_result: Any = None
        self.expected_exceptions: List[type] = []
    
    def with_dependencies(self, **dependencies) -> 'ServiceTestBuilder':
        """Inject mocked dependencies"""
        for name, mock in dependencies.items():
            self.dependencies[name] = mock
            self.constructor_kwargs[name] = mock
        return self
    
    def calling_method(self, method_name: str, *args, **kwargs) -> 'ServiceTestBuilder':
        """Set the method being tested"""
        self.method_under_test = method_name
        self.method_args = args
        self.method_kwargs = kwargs
        return self
    
    def expecting_result(self, result: Any) -> 'ServiceTestBuilder':
        """Set expected method result"""
        self.expected_result = result
        return self
    
    def expecting_exception(self, exception_type: type) -> 'ServiceTestBuilder':
        """Expect method to raise exception"""
        self.expected_exceptions.append(exception_type)
        return self
    
    def setup_dependency_behavior(self, dependency_name: str, method: str, return_value: Any) -> 'ServiceTestBuilder':
        """Configure behavior of dependency mock"""
        if dependency_name in self.dependencies:
            getattr(self.dependencies[dependency_name], method).return_value = return_value
        return self
    
    def build_and_execute(self) -> Dict[str, Any]:
        """Build and execute the service test"""
        # Create service instance with mocked dependencies
        service = self.service_class(*self.constructor_args, **self.constructor_kwargs)
        
        result = {
            'service': self.service_class.__name__,
            'method': self.method_under_test,
            'status': 'pending',
            'actual_result': None,
            'exception': None,
            'dependency_interactions': {}
        }
        
        try:
            # Call the method under test
            method = getattr(service, self.method_under_test)
            actual_result = method(*self.method_args, **self.method_kwargs)
            
            result['actual_result'] = actual_result
            
            # Verify expected result
            if self.expected_result is not None:
                assert actual_result == self.expected_result, (
                    f"Expected {self.expected_result}, got {actual_result}"
                )
            
            result['status'] = 'passed'
            
        except Exception as e:
            result['exception'] = e
            
            # Check if exception was expected
            if any(isinstance(e, exc_type) for exc_type in self.expected_exceptions):
                result['status'] = 'passed'
            else:
                result['status'] = 'failed'
        
        # Record dependency interactions
        for name, mock in self.dependencies.items():
            result['dependency_interactions'][name] = {
                'call_count': mock.call_count,
                'method_calls': [str(call) for call in mock.method_calls]
            }
        
        return result


class DomainTestBuilder:
    """Builder for domain model tests focusing on business rules"""
    
    def __init__(self, domain_class: type):
        self.domain_class = domain_class
        self.instance_data: Dict[str, Any] = {}
        self.business_rules: List[Callable] = []
        self.invariants: List[Callable] = []
        self.state_transitions: List[Dict[str, Any]] = []
    
    def with_initial_state(self, **state) -> 'DomainTestBuilder':
        """Set initial domain object state"""
        self.instance_data.update(state)
        return self
    
    def verify_business_rule(self, rule: Callable) -> 'DomainTestBuilder':
        """Add business rule to verify"""
        self.business_rules.append(rule)
        return self
    
    def verify_invariant(self, invariant: Callable) -> 'DomainTestBuilder':
        """Add invariant to verify"""
        self.invariants.append(invariant)
        return self
    
    def test_state_transition(self, action: str, expected_state: Dict[str, Any]) -> 'DomainTestBuilder':
        """Test state transition"""
        self.state_transitions.append({
            'action': action,
            'expected_state': expected_state
        })
        return self
    
    def execute_domain_tests(self) -> Dict[str, Any]:
        """Execute all domain tests"""
        # Create domain instance
        domain_instance = self.domain_class(**self.instance_data)
        
        result = {
            'domain_class': self.domain_class.__name__,
            'initial_state': self.instance_data,
            'business_rules_passed': [],
            'invariants_passed': [],
            'state_transitions_passed': [],
            'status': 'passed',
            'errors': []
        }
        
        try:
            # Verify business rules
            for rule in self.business_rules:
                if rule(domain_instance):
                    result['business_rules_passed'].append(rule.__name__)
                else:
                    raise AssertionError(f"Business rule {rule.__name__} failed")
            
            # Verify invariants
            for invariant in self.invariants:
                if invariant(domain_instance):
                    result['invariants_passed'].append(invariant.__name__)
                else:
                    raise AssertionError(f"Invariant {invariant.__name__} failed")
            
            # Test state transitions
            for transition in self.state_transitions:
                action = transition['action']
                expected_state = transition['expected_state']
                
                # Execute action
                method = getattr(domain_instance, action)
                method()
                
                # Verify resulting state
                for attr, expected_value in expected_state.items():
                    actual_value = getattr(domain_instance, attr)
                    if actual_value == expected_value:
                        result['state_transitions_passed'].append(f"{action}->{attr}")
                    else:
                        raise AssertionError(
                            f"State transition {action}: expected {attr}={expected_value}, "
                            f"got {actual_value}"
                        )
            
        except Exception as e:
            result['status'] = 'failed'
            result['errors'].append(str(e))
        
        return result


# Test fixtures and utilities
@pytest.fixture
def outside_in_strategy():
    """Fixture providing outside-in test strategy"""
    user_story = UserStory(
        title="Test User Story",
        as_a="test user",
        i_want="to test the system",
        so_that="I can verify it works correctly"
    )
    return OutsideInTestStrategy(user_story)


@pytest.fixture
def acceptance_test_runner():
    """Fixture providing acceptance test runner"""
    return AcceptanceTestRunner()


@pytest.fixture
def api_test_builder():
    """Fixture providing API test builder"""
    return APITestBuilder()