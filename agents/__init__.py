"""
TradeKnowledge Agents Package

This package contains the agent implementations for the TradeKnowledge system,
including persistent memory capabilities and enhanced coordination features.
"""

# Version information
__version__ = "1.0.0"
__author__ = "TradeKnowledge Team"

# Import key classes for convenience
try:
    from agents.core.persistent_mixin import PersistentAgentMixin, EventType, DocumentType
    from agents.core.agent_base import BaseAgent, AgentRole, TaskContext
    
    __all__ = [
        'PersistentAgentMixin',
        'BaseAgent', 
        'AgentRole',
        'TaskContext',
        'EventType',
        'DocumentType'
    ]
except ImportError:
    # Handle import errors gracefully during development
    __all__ = []