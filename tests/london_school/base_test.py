"""
London School TDD Base Test Classes
Provides foundational testing utilities following outside-in, mockist patterns
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, create_autospec
from typing import Any, Dict, List, Optional, Protocol, Type, TypeVar
from abc import ABC, abstractmethod
import inspect
from contextlib import asynccontextmanager

T = TypeVar('T')


class MockFactory:
    """Factory for creating consistent, behavior-focused mocks"""
    
    @staticmethod
    def create_async_mock(spec: Type[T] = None, **kwargs) -> AsyncMock:
        """Create async mock with proper spec"""
        if spec:
            return create_autospec(spec, spec_set=True, instance=True, **kwargs)
        return AsyncMock(**kwargs)
    
    @staticmethod
    def create_mock(spec: Type[T] = None, **kwargs) -> Mock:
        """Create synchronous mock with proper spec"""
        if spec:
            return create_autospec(spec, spec_set=True, instance=True, **kwargs)
        return Mock(**kwargs)
    
    @staticmethod
    def create_spy(real_object: Any) -> Mock:
        """Create spy that wraps real object for interaction verification"""
        return Mock(wraps=real_object)


class BehaviorAssertion:
    """Behavior-focused assertions for London School TDD"""
    
    @staticmethod
    def assert_called_with_behavior(mock: Mock, method_name: str, *args, **kwargs):
        """Assert specific method called with exact arguments"""
        method_mock = getattr(mock, method_name)
        method_mock.assert_called_with(*args, **kwargs)
    
    @staticmethod
    def assert_interaction_sequence(mock: Mock, expected_calls: List[str]):
        """Assert methods called in specific sequence"""
        actual_calls = [call[0] for call in mock.method_calls]
        assert actual_calls == expected_calls, f"Expected {expected_calls}, got {actual_calls}"
    
    @staticmethod
    def assert_collaboration_pattern(
        collaborator_mocks: Dict[str, Mock], 
        expected_pattern: Dict[str, List[str]]
    ):
        """Assert complex collaboration patterns between objects"""
        for collaborator_name, expected_calls in expected_pattern.items():
            mock = collaborator_mocks[collaborator_name]
            actual_calls = [call[0] for call in mock.method_calls]
            assert actual_calls == expected_calls, (
                f"Collaborator {collaborator_name}: expected {expected_calls}, got {actual_calls}"
            )


class TestDouble:
    """Test double creation and management"""
    
    def __init__(self):
        self.doubles: Dict[str, Mock] = {}
    
    def create_stub(self, name: str, return_value: Any = None) -> Mock:
        """Create stub that returns predetermined values"""
        stub = Mock()
        if return_value is not None:
            stub.return_value = return_value
        self.doubles[name] = stub
        return stub
    
    def create_mock(self, name: str, spec: Type = None) -> Mock:
        """Create mock for behavior verification"""
        mock = MockFactory.create_mock(spec) if spec else Mock()
        self.doubles[name] = mock
        return mock
    
    def create_fake(self, name: str, implementation: Any) -> Mock:
        """Create fake with working implementation"""
        fake = Mock(side_effect=implementation)
        self.doubles[name] = fake
        return fake
    
    def verify_all_interactions(self):
        """Verify all mocks were used as expected"""
        for name, double in self.doubles.items():
            if hasattr(double, 'assert_called'):
                try:
                    double.assert_called()
                except AssertionError:
                    pytest.fail(f"Test double '{name}' was never called")


class LondonSchoolTestCase:
    """Base class for London School TDD test cases"""
    
    def setup_method(self):
        """Setup method called before each test"""
        self.test_doubles = TestDouble()
        self.behavior_assertions = BehaviorAssertion()
        self.setup_collaborators()
    
    def teardown_method(self):
        """Teardown method called after each test"""
        self.verify_collaborations()
    
    def setup_collaborators(self):
        """Override to setup test-specific collaborators"""
        pass
    
    def verify_collaborations(self):
        """Override to add custom collaboration verification"""
        pass
    
    def create_system_under_test(self, **dependencies) -> Any:
        """Factory method to create the system being tested"""
        raise NotImplementedError("Subclasses must implement create_system_under_test")


class AsyncLondonSchoolTestCase(LondonSchoolTestCase):
    """Async version of London School test case"""
    
    async def setup_method(self):
        """Async setup method"""
        self.test_doubles = TestDouble()
        self.behavior_assertions = BehaviorAssertion()
        await self.setup_collaborators_async()
    
    async def teardown_method(self):
        """Async teardown method"""
        await self.verify_collaborations_async()
    
    async def setup_collaborators_async(self):
        """Override to setup async collaborators"""
        pass
    
    async def verify_collaborations_async(self):
        """Override to verify async collaborations"""
        pass


class OutsideInTestBuilder:
    """Builder for creating outside-in test scenarios"""
    
    def __init__(self):
        self.scenario_steps: List[Dict[str, Any]] = []
        self.expected_outcomes: List[Dict[str, Any]] = []
        self.collaborators: Dict[str, Mock] = {}
    
    def given_collaborator(self, name: str, behavior: Dict[str, Any]) -> 'OutsideInTestBuilder':
        """Setup collaborator with specific behavior"""
        mock = Mock()
        for method_name, return_value in behavior.items():
            getattr(mock, method_name).return_value = return_value
        self.collaborators[name] = mock
        return self
    
    def when_user_performs(self, action: str, **params) -> 'OutsideInTestBuilder':
        """Define user action that triggers the scenario"""
        self.scenario_steps.append({
            'type': 'user_action',
            'action': action,
            'params': params
        })
        return self
    
    def then_system_should(self, behavior: str, **expectations) -> 'OutsideInTestBuilder':
        """Define expected system behavior"""
        self.expected_outcomes.append({
            'type': 'system_behavior',
            'behavior': behavior,
            'expectations': expectations
        })
        return self
    
    def and_collaborator_should(self, collaborator: str, method: str, *args, **kwargs) -> 'OutsideInTestBuilder':
        """Define expected collaborator interaction"""
        self.expected_outcomes.append({
            'type': 'collaborator_interaction',
            'collaborator': collaborator,
            'method': method,
            'args': args,
            'kwargs': kwargs
        })
        return self
    
    def build_test(self) -> Dict[str, Any]:
        """Build the complete test scenario"""
        return {
            'steps': self.scenario_steps,
            'outcomes': self.expected_outcomes,
            'collaborators': self.collaborators
        }


class ContractTest:
    """Contract testing utilities for API boundaries"""
    
    @staticmethod
    def verify_input_contract(func, input_data: Dict[str, Any], expected_type: Type):
        """Verify function accepts expected input contract"""
        sig = inspect.signature(func)
        for param_name, param_value in input_data.items():
            if param_name in sig.parameters:
                param_type = sig.parameters[param_name].annotation
                if param_type != inspect.Parameter.empty:
                    assert isinstance(param_value, param_type), (
                        f"Parameter {param_name} expected {param_type}, got {type(param_value)}"
                    )
    
    @staticmethod
    def verify_output_contract(result: Any, expected_type: Type):
        """Verify function returns expected output contract"""
        assert isinstance(result, expected_type), (
            f"Expected return type {expected_type}, got {type(result)}"
        )


class MockistaTestHelpers:
    """Helper utilities for mockist testing patterns"""
    
    @staticmethod
    def create_database_mock() -> AsyncMock:
        """Create standardized database service mock"""
        db_mock = AsyncMock()
        
        # Common database operations
        db_mock.postgres.create_user.return_value = "user-123"
        db_mock.postgres.get_user.return_value = {"id": "user-123", "email": "test@test.com"}
        db_mock.postgres.deduct_credits.return_value = True
        
        # Mock connection context manager
        conn_mock = AsyncMock()
        db_mock.postgres.get_connection.return_value.__aenter__.return_value = conn_mock
        db_mock.postgres.get_connection.return_value.__aexit__.return_value = None
        
        return db_mock
    
    @staticmethod
    def create_redis_mock() -> AsyncMock:
        """Create standardized Redis service mock"""
        redis_mock = AsyncMock()
        redis_mock.get.return_value = None
        redis_mock.set.return_value = True
        redis_mock.get_json.return_value = None
        redis_mock.set_json.return_value = True
        return redis_mock
    
    @staticmethod
    def create_influxdb_mock() -> AsyncMock:
        """Create standardized InfluxDB service mock"""
        influx_mock = AsyncMock()
        influx_mock.write_market_data.return_value = None
        influx_mock.query_price_history.return_value = []
        return influx_mock


# Test fixtures for common mocks
@pytest.fixture
def mock_database_service():
    """Fixture providing mocked database service"""
    return MockistaTestHelpers.create_database_mock()


@pytest.fixture
def mock_redis_service():
    """Fixture providing mocked Redis service"""
    return MockistaTestHelpers.create_redis_mock()


@pytest.fixture
def mock_influxdb_service():
    """Fixture providing mocked InfluxDB service"""
    return MockistaTestHelpers.create_influxdb_mock()


@pytest.fixture
def outside_in_builder():
    """Fixture providing outside-in test builder"""
    return OutsideInTestBuilder()


@pytest.fixture
def test_doubles():
    """Fixture providing test doubles factory"""
    return TestDouble()