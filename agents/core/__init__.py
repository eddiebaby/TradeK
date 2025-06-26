"""
Agent Core Package

Core functionality for TradeKnowledge agents including base classes,
persistent memory capabilities, and communication protocols.
"""

from .persistent_mixin import PersistentAgentMixin, EventType, DocumentType
from .agent_base import BaseAgent, AgentRole, TaskContext

__all__ = [
    'PersistentAgentMixin',
    'BaseAgent',
    'AgentRole', 
    'TaskContext',
    'EventType',
    'DocumentType'
]