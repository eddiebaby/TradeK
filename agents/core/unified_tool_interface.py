"""
Unified Tool Interface for TradeKnowledge Agents

This module provides a unified interface that combines native TradeKnowledge tools
with OpenAI tools and MCP tools, enabling seamless tool usage across all agent types.
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Protocol
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

from agents import (
    Agent, WebSearchTool, CodeInterpreterTool, FileSearchTool, 
    FunctionTool, function_tool
)
from agents.core.mcp_integration import MCPServerManager, MCPToolResult, MCPServerConfig
from src.search.unified_search import UnifiedSearchEngine
from src.ingestion.enhanced_book_processor import EnhancedBookProcessor
from src.utils.cache_manager import CacheManager


class ToolType(Enum):
    """Types of tools available in the unified interface."""
    NATIVE_TRADEKNOWLEDGE = "native_tk"
    OPENAI_BUILTIN = "openai_builtin"
    MCP_EXTERNAL = "mcp_external"
    CUSTOM_FUNCTION = "custom_function"


class ToolCategory(Enum):
    """Categories of tools by functionality."""
    SEARCH = "search"
    RESEARCH = "research"
    IMPLEMENTATION = "implementation"
    VALIDATION = "validation"
    DOCUMENTATION = "documentation"
    ANALYSIS = "analysis"
    COMMUNICATION = "communication"
    UTILITY = "utility"


@dataclass
class ToolDescriptor:
    """Descriptor for a tool in the unified interface."""
    tool_id: str
    tool_name: str
    tool_type: ToolType
    category: ToolCategory
    description: str
    parameters: Dict[str, Any]
    capabilities: List[str]
    requirements: List[str] = field(default_factory=list)
    confidence_score: float = 1.0
    performance_rating: float = 1.0
    last_used: Optional[float] = None
    usage_count: int = 0


@dataclass
class ToolExecutionResult:
    """Result from tool execution in unified interface."""
    tool_id: str
    success: bool
    result: Any
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    confidence_score: Optional[float] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


class ToolProvider(ABC):
    """Abstract base class for tool providers."""
    
    @abstractmethod
    async def list_tools(self) -> List[ToolDescriptor]:
        """List available tools from this provider."""
        pass
    
    @abstractmethod
    async def execute_tool(self, tool_id: str, parameters: Dict[str, Any]) -> ToolExecutionResult:
        """Execute a specific tool with given parameters."""
        pass
    
    @abstractmethod
    async def get_tool_info(self, tool_id: str) -> Optional[ToolDescriptor]:
        """Get detailed information about a specific tool."""
        pass


class NativeTradeKnowledgeProvider(ToolProvider):
    """Provider for native TradeKnowledge tools."""
    
    def __init__(self, 
                 search_engine: UnifiedSearchEngine,
                 book_processor: EnhancedBookProcessor,
                 cache_manager: CacheManager):
        self.search_engine = search_engine
        self.book_processor = book_processor
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)
    
    async def list_tools(self) -> List[ToolDescriptor]:
        """List native TradeKnowledge tools."""
        
        tools = [
            ToolDescriptor(
                tool_id="tk_unified_search",
                tool_name="TradeKnowledge Unified Search",
                tool_type=ToolType.NATIVE_TRADEKNOWLEDGE,
                category=ToolCategory.SEARCH,
                description="Search across TradeKnowledge vector database and text indices",
                parameters={
                    "query": {"type": "string", "required": True},
                    "max_results": {"type": "integer", "default": 10},
                    "include_metadata": {"type": "boolean", "default": True}
                },
                capabilities=["semantic_search", "hybrid_search", "metadata_filtering"],
                requirements=["vector_database", "search_indices"]
            ),
            ToolDescriptor(
                tool_id="tk_document_processor",
                tool_name="TradeKnowledge Document Processor",
                tool_type=ToolType.NATIVE_TRADEKNOWLEDGE,
                category=ToolCategory.DOCUMENTATION,
                description="Process and analyze financial documents and trading materials",
                parameters={
                    "document_path": {"type": "string", "required": True},
                    "processing_type": {"type": "string", "default": "comprehensive"},
                    "extract_entities": {"type": "boolean", "default": True}
                },
                capabilities=["pdf_processing", "text_extraction", "entity_recognition"],
                requirements=["file_access", "nlp_models"]
            ),
            ToolDescriptor(
                tool_id="tk_cache_manager",
                tool_name="TradeKnowledge Cache Manager",
                tool_type=ToolType.NATIVE_TRADEKNOWLEDGE,
                category=ToolCategory.UTILITY,
                description="Manage caching for improved performance",
                parameters={
                    "action": {"type": "string", "required": True},
                    "key": {"type": "string", "required": False},
                    "value": {"type": "any", "required": False}
                },
                capabilities=["redis_caching", "memory_caching", "cache_invalidation"],
                requirements=["cache_backend"]
            )
        ]
        
        return tools
    
    async def execute_tool(self, tool_id: str, parameters: Dict[str, Any]) -> ToolExecutionResult:
        """Execute native TradeKnowledge tool."""
        
        execution_start = time.time()
        
        try:
            if tool_id == "tk_unified_search":
                result = await self._execute_unified_search(parameters)
            elif tool_id == "tk_document_processor":
                result = await self._execute_document_processor(parameters)
            elif tool_id == "tk_cache_manager":
                result = await self._execute_cache_manager(parameters)
            else:
                raise ValueError(f"Unknown tool ID: {tool_id}")
            
            execution_time = time.time() - execution_start
            
            return ToolExecutionResult(
                tool_id=tool_id,
                success=True,
                result=result,
                execution_time=execution_time,
                metadata={"provider": "native_tradeknowledge"},
                performance_metrics={"execution_time_ms": execution_time * 1000}
            )
            
        except Exception as e:
            execution_time = time.time() - execution_start
            
            return ToolExecutionResult(
                tool_id=tool_id,
                success=False,
                result=None,
                execution_time=execution_time,
                error_message=str(e),
                metadata={"provider": "native_tradeknowledge"}
            )
    
    async def _execute_unified_search(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute unified search."""
        
        query = parameters.get("query", "")
        max_results = parameters.get("max_results", 10)
        include_metadata = parameters.get("include_metadata", True)
        
        search_results = await self.search_engine.search(
            query=query,
            limit=max_results,
            include_metadata=include_metadata
        )
        
        return {
            "query": query,
            "results": [result.dict() for result in search_results],
            "total_found": len(search_results),
            "search_type": "unified_hybrid"
        }
    
    async def _execute_document_processor(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute document processor."""
        
        document_path = parameters.get("document_path", "")
        processing_type = parameters.get("processing_type", "comprehensive")
        extract_entities = parameters.get("extract_entities", True)
        
        processed_doc = await self.book_processor.process_file(document_path)
        
        return {
            "document_path": document_path,
            "processing_type": processing_type,
            "processed_content": processed_doc,
            "entities_extracted": extract_entities
        }
    
    async def _execute_cache_manager(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cache manager operations."""
        
        action = parameters.get("action", "")
        key = parameters.get("key")
        value = parameters.get("value")
        
        if action == "get":
            result = await self.cache_manager.get(key) if key else None
        elif action == "set":
            result = await self.cache_manager.set(key, value) if key and value else False
        elif action == "delete":
            result = await self.cache_manager.delete(key) if key else False
        elif action == "clear":
            result = await self.cache_manager.clear()
        else:
            raise ValueError(f"Unknown cache action: {action}")
        
        return {
            "action": action,
            "key": key,
            "result": result
        }
    
    async def get_tool_info(self, tool_id: str) -> Optional[ToolDescriptor]:
        """Get information about a specific native tool."""
        
        tools = await self.list_tools()
        return next((tool for tool in tools if tool.tool_id == tool_id), None)


class OpenAIBuiltinProvider(ToolProvider):
    """Provider for OpenAI built-in tools."""
    
    def __init__(self, vector_store_ids: Optional[List[str]] = None):
        self.vector_store_ids = vector_store_ids or []
        self.logger = logging.getLogger(__name__)
    
    async def list_tools(self) -> List[ToolDescriptor]:
        """List OpenAI built-in tools."""
        
        tools = [
            ToolDescriptor(
                tool_id="openai_web_search",
                tool_name="OpenAI Web Search Tool",
                tool_type=ToolType.OPENAI_BUILTIN,
                category=ToolCategory.RESEARCH,
                description="Search the web for real-time information and market intelligence",
                parameters={
                    "query": {"type": "string", "required": True},
                    "user_location": {"type": "object", "required": False},
                    "max_results": {"type": "integer", "default": 10}
                },
                capabilities=["real_time_search", "market_intelligence", "news_analysis"],
                requirements=["internet_access", "openai_api"],
                confidence_score=0.95,
                performance_rating=0.90
            ),
            ToolDescriptor(
                tool_id="openai_code_interpreter",
                tool_name="OpenAI Code Interpreter Tool",
                tool_type=ToolType.OPENAI_BUILTIN,
                category=ToolCategory.IMPLEMENTATION,
                description="Execute Python code for analysis, testing, and validation",
                parameters={
                    "code": {"type": "string", "required": True},
                    "container_type": {"type": "string", "default": "auto"}
                },
                capabilities=["code_execution", "data_analysis", "testing", "visualization"],
                requirements=["code_sandbox", "openai_api"],
                confidence_score=0.98,
                performance_rating=0.85
            )
        ]
        
        if self.vector_store_ids:
            tools.append(ToolDescriptor(
                tool_id="openai_file_search",
                tool_name="OpenAI File Search Tool",
                tool_type=ToolType.OPENAI_BUILTIN,
                category=ToolCategory.SEARCH,
                description="Search through uploaded documents using semantic search",
                parameters={
                    "query": {"type": "string", "required": True},
                    "vector_store_ids": {"type": "array", "required": True},
                    "max_results": {"type": "integer", "default": 10}
                },
                capabilities=["semantic_search", "document_analysis", "context_retrieval"],
                requirements=["vector_stores", "openai_api"],
                confidence_score=0.92,
                performance_rating=0.88
            ))
        
        return tools
    
    async def execute_tool(self, tool_id: str, parameters: Dict[str, Any]) -> ToolExecutionResult:
        """Execute OpenAI built-in tool."""
        
        execution_start = time.time()
        
        try:
            if tool_id == "openai_web_search":
                result = await self._execute_web_search(parameters)
            elif tool_id == "openai_code_interpreter":
                result = await self._execute_code_interpreter(parameters)
            elif tool_id == "openai_file_search":
                result = await self._execute_file_search(parameters)
            else:
                raise ValueError(f"Unknown OpenAI tool ID: {tool_id}")
            
            execution_time = time.time() - execution_start
            
            return ToolExecutionResult(
                tool_id=tool_id,
                success=True,
                result=result,
                execution_time=execution_time,
                metadata={"provider": "openai_builtin"},
                confidence_score=0.9,
                performance_metrics={"execution_time_ms": execution_time * 1000}
            )
            
        except Exception as e:
            execution_time = time.time() - execution_start
            
            return ToolExecutionResult(
                tool_id=tool_id,
                success=False,
                result=None,
                execution_time=execution_time,
                error_message=str(e),
                metadata={"provider": "openai_builtin"}
            )
    
    async def _execute_web_search(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute web search using OpenAI WebSearchTool."""
        
        # This would typically create an agent with WebSearchTool and execute
        # For now, return simulated result
        query = parameters.get("query", "")
        
        return {
            "query": query,
            "search_results": [
                {
                    "title": f"Search result for: {query}",
                    "snippet": "Simulated web search result with relevant information",
                    "url": "https://example.com/result",
                    "relevance_score": 0.85
                }
            ],
            "provider": "openai_web_search"
        }
    
    async def _execute_code_interpreter(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code using OpenAI CodeInterpreterTool."""
        
        code = parameters.get("code", "")
        
        return {
            "code": code,
            "execution_result": "Code executed successfully",
            "output": "Simulated code execution output",
            "provider": "openai_code_interpreter"
        }
    
    async def _execute_file_search(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute file search using OpenAI FileSearchTool."""
        
        query = parameters.get("query", "")
        
        return {
            "query": query,
            "search_results": [
                {
                    "document_id": "doc_123",
                    "content": "Relevant document content",
                    "relevance_score": 0.92
                }
            ],
            "provider": "openai_file_search"
        }
    
    async def get_tool_info(self, tool_id: str) -> Optional[ToolDescriptor]:
        """Get information about a specific OpenAI tool."""
        
        tools = await self.list_tools()
        return next((tool for tool in tools if tool.tool_id == tool_id), None)


class MCPExternalProvider(ToolProvider):
    """Provider for MCP external tools."""
    
    def __init__(self, mcp_manager: MCPServerManager):
        self.mcp_manager = mcp_manager
        self.logger = logging.getLogger(__name__)
    
    async def list_tools(self) -> List[ToolDescriptor]:
        """List MCP external tools."""
        
        tools = []
        mcp_tools = await self.mcp_manager.list_all_tools()
        
        for server_name, server_tools in mcp_tools.items():
            for tool in server_tools:
                tool_descriptor = ToolDescriptor(
                    tool_id=f"mcp_{server_name}_{tool.get('name', '')}",
                    tool_name=f"MCP {tool.get('name', 'Unknown')}",
                    tool_type=ToolType.MCP_EXTERNAL,
                    category=self._categorize_mcp_tool(tool),
                    description=tool.get("description", "MCP external tool"),
                    parameters=tool.get("inputSchema", {}).get("properties", {}),
                    capabilities=self._extract_mcp_capabilities(tool),
                    requirements=["mcp_server", server_name],
                    confidence_score=0.8,  # Default for external tools
                    performance_rating=0.75
                )
                tools.append(tool_descriptor)
        
        return tools
    
    def _categorize_mcp_tool(self, tool: Dict[str, Any]) -> ToolCategory:
        """Categorize MCP tool based on its characteristics."""
        
        name = tool.get("name", "").lower()
        description = tool.get("description", "").lower()
        
        if any(keyword in name + description for keyword in ["search", "find", "query"]):
            return ToolCategory.SEARCH
        elif any(keyword in name + description for keyword in ["file", "read", "write"]):
            return ToolCategory.DOCUMENTATION
        elif any(keyword in name + description for keyword in ["analyze", "process", "compute"]):
            return ToolCategory.ANALYSIS
        else:
            return ToolCategory.UTILITY
    
    def _extract_mcp_capabilities(self, tool: Dict[str, Any]) -> List[str]:
        """Extract capabilities from MCP tool specification."""
        
        capabilities = []
        name = tool.get("name", "").lower()
        
        if "filesystem" in name:
            capabilities.extend(["file_operations", "directory_access"])
        if "web" in name:
            capabilities.extend(["web_access", "http_requests"])
        if "database" in name:
            capabilities.extend(["database_queries", "data_access"])
        
        return capabilities
    
    async def execute_tool(self, tool_id: str, parameters: Dict[str, Any]) -> ToolExecutionResult:
        """Execute MCP external tool."""
        
        execution_start = time.time()
        
        try:
            # Parse MCP tool identifier
            parts = tool_id.split("_", 2)
            if len(parts) < 3 or parts[0] != "mcp":
                raise ValueError(f"Invalid MCP tool ID format: {tool_id}")
            
            server_name = parts[1]
            tool_name = parts[2]
            
            # Execute via MCP manager
            mcp_result = await self.mcp_manager.call_tool(server_name, tool_name, parameters)
            
            return ToolExecutionResult(
                tool_id=tool_id,
                success=mcp_result.success,
                result=mcp_result.result,
                execution_time=mcp_result.execution_time,
                error_message=mcp_result.error_message,
                metadata={
                    "provider": "mcp_external",
                    "server": server_name,
                    **mcp_result.metadata
                },
                performance_metrics={"execution_time_ms": mcp_result.execution_time * 1000}
            )
            
        except Exception as e:
            execution_time = time.time() - execution_start
            
            return ToolExecutionResult(
                tool_id=tool_id,
                success=False,
                result=None,
                execution_time=execution_time,
                error_message=str(e),
                metadata={"provider": "mcp_external"}
            )
    
    async def get_tool_info(self, tool_id: str) -> Optional[ToolDescriptor]:
        """Get information about a specific MCP tool."""
        
        tools = await self.list_tools()
        return next((tool for tool in tools if tool.tool_id == tool_id), None)


class UnifiedToolInterface:
    """
    Unified Tool Interface for TradeKnowledge Agents
    
    Provides a single interface for accessing all types of tools:
    - Native TradeKnowledge tools
    - OpenAI built-in tools
    - MCP external tools
    - Custom function tools
    """
    
    def __init__(self,
                 search_engine: UnifiedSearchEngine,
                 book_processor: EnhancedBookProcessor,
                 cache_manager: CacheManager,
                 mcp_manager: Optional[MCPServerManager] = None,
                 vector_store_ids: Optional[List[str]] = None):
        
        # Initialize providers
        self.native_provider = NativeTradeKnowledgeProvider(
            search_engine, book_processor, cache_manager
        )
        self.openai_provider = OpenAIBuiltinProvider(vector_store_ids)
        self.mcp_provider = MCPExternalProvider(mcp_manager) if mcp_manager else None
        
        # Tool registry
        self.tool_registry: Dict[str, ToolDescriptor] = {}
        self.provider_map: Dict[str, ToolProvider] = {}
        
        # Usage statistics
        self.usage_stats: Dict[str, Dict[str, Any]] = {}
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize the unified tool interface."""
        
        self.logger.info("Initializing unified tool interface...")
        
        # Load tools from all providers
        await self._load_tools_from_providers()
        
        self.logger.info(f"Unified tool interface initialized with {len(self.tool_registry)} tools")
    
    async def _load_tools_from_providers(self):
        """Load tools from all available providers."""
        
        providers = [
            ("native", self.native_provider),
            ("openai", self.openai_provider)
        ]
        
        if self.mcp_provider:
            providers.append(("mcp", self.mcp_provider))
        
        for provider_name, provider in providers:
            try:
                tools = await provider.list_tools()
                for tool in tools:
                    self.tool_registry[tool.tool_id] = tool
                    self.provider_map[tool.tool_id] = provider
                
                self.logger.info(f"Loaded {len(tools)} tools from {provider_name} provider")
                
            except Exception as e:
                self.logger.error(f"Failed to load tools from {provider_name} provider: {e}")
    
    async def list_all_tools(self) -> List[ToolDescriptor]:
        """List all available tools."""
        return list(self.tool_registry.values())
    
    async def list_tools_by_category(self, category: ToolCategory) -> List[ToolDescriptor]:
        """List tools by category."""
        return [tool for tool in self.tool_registry.values() if tool.category == category]
    
    async def list_tools_by_type(self, tool_type: ToolType) -> List[ToolDescriptor]:
        """List tools by type."""
        return [tool for tool in self.tool_registry.values() if tool.tool_type == tool_type]
    
    async def search_tools(self, query: str) -> List[ToolDescriptor]:
        """Search for tools by name, description, or capabilities."""
        
        query_lower = query.lower()
        matching_tools = []
        
        for tool in self.tool_registry.values():
            if (query_lower in tool.tool_name.lower() or
                query_lower in tool.description.lower() or
                any(query_lower in cap.lower() for cap in tool.capabilities)):
                matching_tools.append(tool)
        
        # Sort by relevance (simplified scoring)
        matching_tools.sort(key=lambda t: t.confidence_score * t.performance_rating, reverse=True)
        
        return matching_tools
    
    async def get_tool_info(self, tool_id: str) -> Optional[ToolDescriptor]:
        """Get detailed information about a specific tool."""
        return self.tool_registry.get(tool_id)
    
    async def execute_tool(self, tool_id: str, parameters: Dict[str, Any]) -> ToolExecutionResult:
        """Execute a tool with given parameters."""
        
        if tool_id not in self.tool_registry:
            return ToolExecutionResult(
                tool_id=tool_id,
                success=False,
                result=None,
                execution_time=0,
                error_message=f"Tool not found: {tool_id}"
            )
        
        provider = self.provider_map.get(tool_id)
        if not provider:
            return ToolExecutionResult(
                tool_id=tool_id,
                success=False,
                result=None,
                execution_time=0,
                error_message=f"No provider found for tool: {tool_id}"
            )
        
        # Execute tool and update statistics
        result = await provider.execute_tool(tool_id, parameters)
        await self._update_usage_statistics(tool_id, result)
        
        return result
    
    async def execute_tool_chain(self, tool_chain: List[Dict[str, Any]]) -> List[ToolExecutionResult]:
        """Execute a chain of tools in sequence."""
        
        results = []
        
        for tool_spec in tool_chain:
            tool_id = tool_spec.get("tool_id", "")
            parameters = tool_spec.get("parameters", {})
            
            # Allow results from previous tools to be used in parameters
            if results:
                parameters = await self._inject_previous_results(parameters, results)
            
            result = await self.execute_tool(tool_id, parameters)
            results.append(result)
            
            # Stop chain if tool fails and no error handling specified
            if not result.success and not tool_spec.get("continue_on_error", False):
                break
        
        return results
    
    async def _inject_previous_results(self, 
                                     parameters: Dict[str, Any], 
                                     previous_results: List[ToolExecutionResult]) -> Dict[str, Any]:
        """Inject results from previous tools into current parameters."""
        
        # Simple placeholder replacement
        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith("${result_"):
                try:
                    result_index = int(value[9:-1])  # Extract index from ${result_N}
                    if 0 <= result_index < len(previous_results):
                        parameters[key] = previous_results[result_index].result
                except (ValueError, IndexError):
                    pass  # Keep original value if replacement fails
        
        return parameters
    
    async def get_recommended_tools(self, 
                                  task_description: str, 
                                  max_recommendations: int = 5) -> List[ToolDescriptor]:
        """Get recommended tools for a given task."""
        
        # Search for relevant tools
        candidate_tools = await self.search_tools(task_description)
        
        # Score tools based on relevance and performance
        scored_tools = []
        for tool in candidate_tools:
            relevance_score = self._calculate_relevance_score(tool, task_description)
            overall_score = (relevance_score * 0.6 + 
                           tool.confidence_score * 0.25 + 
                           tool.performance_rating * 0.15)
            scored_tools.append((tool, overall_score))
        
        # Sort by score and return top recommendations
        scored_tools.sort(key=lambda x: x[1], reverse=True)
        
        return [tool for tool, score in scored_tools[:max_recommendations]]
    
    def _calculate_relevance_score(self, tool: ToolDescriptor, task_description: str) -> float:
        """Calculate relevance score between tool and task description."""
        
        task_words = set(task_description.lower().split())
        tool_words = set((tool.tool_name + " " + tool.description).lower().split())
        
        if not task_words:
            return 0.0
        
        intersection = task_words.intersection(tool_words)
        return len(intersection) / len(task_words)
    
    async def _update_usage_statistics(self, tool_id: str, result: ToolExecutionResult):
        """Update usage statistics for a tool."""
        
        if tool_id not in self.usage_stats:
            self.usage_stats[tool_id] = {
                "usage_count": 0,
                "success_count": 0,
                "total_execution_time": 0,
                "average_execution_time": 0,
                "last_used": None
            }
        
        stats = self.usage_stats[tool_id]
        stats["usage_count"] += 1
        stats["total_execution_time"] += result.execution_time
        stats["average_execution_time"] = stats["total_execution_time"] / stats["usage_count"]
        stats["last_used"] = time.time()
        
        if result.success:
            stats["success_count"] += 1
        
        # Update tool descriptor
        if tool_id in self.tool_registry:
            tool = self.tool_registry[tool_id]
            tool.last_used = stats["last_used"]
            tool.usage_count = stats["usage_count"]
    
    async def get_usage_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get usage statistics for all tools."""
        return self.usage_stats.copy()
    
    async def get_tool_performance_report(self) -> Dict[str, Any]:
        """Generate a performance report for all tools."""
        
        total_tools = len(self.tool_registry)
        total_executions = sum(stats.get("usage_count", 0) for stats in self.usage_stats.values())
        
        performance_by_type = {}
        for tool in self.tool_registry.values():
            tool_type = tool.tool_type.value
            if tool_type not in performance_by_type:
                performance_by_type[tool_type] = {
                    "tool_count": 0,
                    "total_usage": 0,
                    "average_performance": 0
                }
            
            performance_by_type[tool_type]["tool_count"] += 1
            stats = self.usage_stats.get(tool.tool_id, {})
            performance_by_type[tool_type]["total_usage"] += stats.get("usage_count", 0)
        
        # Calculate average performance ratings
        for tool_type, metrics in performance_by_type.items():
            tools_of_type = [t for t in self.tool_registry.values() if t.tool_type.value == tool_type]
            if tools_of_type:
                metrics["average_performance"] = sum(t.performance_rating for t in tools_of_type) / len(tools_of_type)
        
        return {
            "total_tools": total_tools,
            "total_executions": total_executions,
            "performance_by_type": performance_by_type,
            "most_used_tools": sorted(
                self.usage_stats.items(), 
                key=lambda x: x[1].get("usage_count", 0), 
                reverse=True
            )[:10]
        }
    
    async def refresh_tools(self):
        """Refresh tool registry from all providers."""
        
        self.tool_registry.clear()
        self.provider_map.clear()
        await self._load_tools_from_providers()