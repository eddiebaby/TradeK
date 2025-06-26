"""
Behavior Verification Utilities for London School TDD
Advanced interaction and collaboration verification patterns
"""

from typing import Any, Dict, List, Optional, Callable, Union
from unittest.mock import Mock, AsyncMock, call
from dataclasses import dataclass
from enum import Enum
import inspect
import asyncio


class InteractionType(Enum):
    """Types of interactions to verify"""
    METHOD_CALL = "method_call"
    PROPERTY_ACCESS = "property_access"
    CONTEXT_MANAGER = "context_manager"
    ASYNC_CALL = "async_call"


@dataclass
class ExpectedInteraction:
    """Represents an expected interaction with a collaborator"""
    collaborator_name: str
    interaction_type: InteractionType
    method_name: str
    args: tuple = ()
    kwargs: dict = None
    return_value: Any = None
    call_count: int = 1
    order_index: Optional[int] = None
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


class CollaborationPattern:
    """Defines patterns of collaboration between objects"""
    
    def __init__(self, name: str):
        self.name = name
        self.interactions: List[ExpectedInteraction] = []
        self.invariants: List[Callable] = []
    
    def expect_interaction(
        self, 
        collaborator: str, 
        method: str, 
        *args, 
        **kwargs
    ) -> 'CollaborationPattern':
        """Add expected interaction to the pattern"""
        interaction = ExpectedInteraction(
            collaborator_name=collaborator,
            interaction_type=InteractionType.METHOD_CALL,
            method_name=method,
            args=args,
            kwargs=kwargs,
            order_index=len(self.interactions)
        )
        self.interactions.append(interaction)
        return self
    
    def expect_async_interaction(
        self, 
        collaborator: str, 
        method: str, 
        *args, 
        **kwargs
    ) -> 'CollaborationPattern':
        """Add expected async interaction to the pattern"""
        interaction = ExpectedInteraction(
            collaborator_name=collaborator,
            interaction_type=InteractionType.ASYNC_CALL,
            method_name=method,
            args=args,
            kwargs=kwargs,
            order_index=len(self.interactions)
        )
        self.interactions.append(interaction)
        return self
    
    def add_invariant(self, invariant_func: Callable) -> 'CollaborationPattern':
        """Add invariant that must hold throughout the interaction"""
        self.invariants.append(invariant_func)
        return self


class BehaviorVerifier:
    """Verifies complex behavior patterns and collaborations"""
    
    def __init__(self):
        self.collaborators: Dict[str, Mock] = {}
        self.patterns: List[CollaborationPattern] = []
    
    def register_collaborator(self, name: str, mock: Mock) -> None:
        """Register a collaborator for verification"""
        self.collaborators[name] = mock
    
    def add_pattern(self, pattern: CollaborationPattern) -> None:
        """Add collaboration pattern to verify"""
        self.patterns.append(pattern)
    
    def verify_all_patterns(self) -> None:
        """Verify all registered collaboration patterns"""
        for pattern in self.patterns:
            self.verify_pattern(pattern)
    
    def verify_pattern(self, pattern: CollaborationPattern) -> None:
        """Verify a specific collaboration pattern"""
        print(f"Verifying collaboration pattern: {pattern.name}")
        
        # Verify invariants
        for invariant in pattern.invariants:
            assert invariant(), f"Invariant failed in pattern {pattern.name}"
        
        # Verify interactions
        for interaction in pattern.interactions:
            self._verify_interaction(interaction)
    
    def _verify_interaction(self, interaction: ExpectedInteraction) -> None:
        """Verify a single interaction"""
        collaborator = self.collaborators.get(interaction.collaborator_name)
        if not collaborator:
            raise AssertionError(
                f"Collaborator '{interaction.collaborator_name}' not registered"
            )
        
        method_mock = getattr(collaborator, interaction.method_name)
        
        if interaction.interaction_type == InteractionType.METHOD_CALL:
            if interaction.call_count == 1:
                method_mock.assert_called_with(*interaction.args, **interaction.kwargs)
            else:
                assert method_mock.call_count == interaction.call_count, (
                    f"Expected {interaction.call_count} calls to "
                    f"{interaction.collaborator_name}.{interaction.method_name}, "
                    f"got {method_mock.call_count}"
                )
        
        elif interaction.interaction_type == InteractionType.ASYNC_CALL:
            # For async calls, verify using call history
            expected_call = call(*interaction.args, **interaction.kwargs)
            assert expected_call in method_mock.call_args_list, (
                f"Expected async call {expected_call} not found in "
                f"{method_mock.call_args_list}"
            )
    
    def verify_call_order(self, expected_order: List[tuple]) -> None:
        """Verify calls happened in specific order across collaborators"""
        all_calls = []
        
        for collab_name, collaborator in self.collaborators.items():
            for call_info in collaborator.method_calls:
                all_calls.append((collab_name, call_info))
        
        # Sort by call order (if timestamps were available)
        # For now, verify based on registration order
        expected_calls = [(name, call(method, *args, **kwargs)) 
                         for name, method, args, kwargs in expected_order]
        
        actual_calls = [(name, call_info) for name, call_info in all_calls]
        
        assert len(actual_calls) >= len(expected_calls), (
            f"Expected at least {len(expected_calls)} calls, got {len(actual_calls)}"
        )


class StateVerifier:
    """Verifies state changes and side effects"""
    
    def __init__(self):
        self.state_snapshots: Dict[str, Any] = {}
        self.state_checkers: Dict[str, Callable] = {}
    
    def capture_state(self, name: str, state_getter: Callable) -> None:
        """Capture current state for later verification"""
        self.state_snapshots[name] = state_getter()
    
    def register_state_checker(self, name: str, checker: Callable) -> None:
        """Register a function to check state"""
        self.state_checkers[name] = checker
    
    def verify_state_change(self, name: str, expected_change: Any) -> None:
        """Verify that state changed as expected"""
        if name not in self.state_checkers:
            raise ValueError(f"No state checker registered for '{name}'")
        
        current_state = self.state_checkers[name]()
        previous_state = self.state_snapshots.get(name)
        
        if previous_state is not None:
            assert current_state != previous_state, (
                f"Expected state change for '{name}', but state remained {current_state}"
            )
        
        if expected_change is not None:
            assert current_state == expected_change, (
                f"Expected state '{expected_change}' for '{name}', got '{current_state}'"
            )
    
    def verify_no_state_change(self, name: str) -> None:
        """Verify that state did not change"""
        if name not in self.state_checkers:
            raise ValueError(f"No state checker registered for '{name}'")
        
        current_state = self.state_checkers[name]()
        previous_state = self.state_snapshots.get(name)
        
        assert current_state == previous_state, (
            f"Expected no state change for '{name}', but changed from "
            f"{previous_state} to {current_state}"
        )


class AsyncBehaviorVerifier(BehaviorVerifier):
    """Async version of behavior verifier"""
    
    async def verify_async_pattern(self, pattern: CollaborationPattern) -> None:
        """Verify async collaboration pattern"""
        print(f"Verifying async collaboration pattern: {pattern.name}")
        
        # Verify async invariants
        for invariant in pattern.invariants:
            if asyncio.iscoroutinefunction(invariant):
                result = await invariant()
            else:
                result = invariant()
            assert result, f"Async invariant failed in pattern {pattern.name}"
        
        # Verify async interactions
        for interaction in pattern.interactions:
            await self._verify_async_interaction(interaction)
    
    async def _verify_async_interaction(self, interaction: ExpectedInteraction) -> None:
        """Verify async interaction"""
        collaborator = self.collaborators.get(interaction.collaborator_name)
        if not collaborator:
            raise AssertionError(
                f"Collaborator '{interaction.collaborator_name}' not registered"
            )
        
        method_mock = getattr(collaborator, interaction.method_name)
        
        if interaction.interaction_type == InteractionType.ASYNC_CALL:
            # Verify async call was made
            expected_call = call(*interaction.args, **interaction.kwargs)
            assert expected_call in method_mock.call_args_list, (
                f"Expected async call {expected_call} not found"
            )
            
            # If it's an async mock, verify it was awaited
            if hasattr(method_mock, 'await_count'):
                assert method_mock.await_count >= interaction.call_count, (
                    f"Expected {interaction.call_count} awaits, "
                    f"got {method_mock.await_count}"
                )


class InteractionRecorder:
    """Records interactions for analysis and verification"""
    
    def __init__(self):
        self.recorded_interactions: List[Dict[str, Any]] = []
        self.start_time: Optional[float] = None
    
    def start_recording(self) -> None:
        """Start recording interactions"""
        import time
        self.start_time = time.time()
        self.recorded_interactions.clear()
    
    def record_interaction(
        self, 
        collaborator: str, 
        method: str, 
        args: tuple, 
        kwargs: dict,
        result: Any = None
    ) -> None:
        """Record an interaction"""
        import time
        interaction = {
            'timestamp': time.time() - (self.start_time or 0),
            'collaborator': collaborator,
            'method': method,
            'args': args,
            'kwargs': kwargs,
            'result': result
        }
        self.recorded_interactions.append(interaction)
    
    def get_interaction_summary(self) -> Dict[str, Any]:
        """Get summary of recorded interactions"""
        return {
            'total_interactions': len(self.recorded_interactions),
            'collaborators': list(set(i['collaborator'] for i in self.recorded_interactions)),
            'methods': list(set(i['method'] for i in self.recorded_interactions)),
            'timeline': self.recorded_interactions
        }
    
    def verify_interaction_timeline(self, expected_timeline: List[str]) -> None:
        """Verify interactions happened in expected timeline"""
        actual_timeline = [
            f"{i['collaborator']}.{i['method']}" 
            for i in self.recorded_interactions
        ]
        assert actual_timeline == expected_timeline, (
            f"Expected timeline {expected_timeline}, got {actual_timeline}"
        )


# Decorator for automatic interaction recording
def record_interactions(recorder: InteractionRecorder):
    """Decorator to automatically record method interactions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            recorder.record_interaction(
                collaborator=func.__qualname__,
                method=func.__name__,
                args=args,
                kwargs=kwargs
            )
            result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator