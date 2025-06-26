"""
MCP (Model Context Protocol) Integration Framework

This module provides MCP server integration capabilities for TradeKnowledge agents,
allowing them to use both OpenAI tools and MCP tools in a unified interface.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Protocol
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path

from agents.mcp import MCPServer, MCPServerStdio, MCPServerSse, MCPServerStreamableHttp
from agents import Agent, Runner, trace
from core.agent_base import BaseAgent, TaskContext


@dataclass
class MCPToolResult:
    """Result from MCP tool execution."""
    tool_name: str
    success: bool
    result: Any
    execution_time: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPServerConfig:
    """Configuration for MCP server connection."""
    name: str
    server_type: str  # "stdio", "sse", "streamable_http"
    connection_params: Dict[str, Any]
    tools_cache_enabled: bool = True
    timeout: int = 30
    retry_attempts: int = 3


class MCPToolInterface(Protocol):
    """Protocol for MCP tool interface."""
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from MCP server."""
        ...
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPToolResult:
        """Call a specific tool with arguments."""
        ...


class MCPServerManager:
    """
    MCP Server Manager for TradeKnowledge Agents
    
    Manages connections to multiple MCP servers and provides unified
    tool access across different server types.
    """
    
    def __init__(self):
        self.servers: Dict[str, MCPServer] = {}
        self.server_configs: Dict[str, MCPServerConfig] = {}
        self.available_tools: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
    
    async def add_server(self, config: MCPServerConfig) -> bool:
        """
        Add MCP server with specified configuration.
        
        Args:
            config: MCP server configuration
            
        Returns:
            bool: True if server added successfully
        """
        try:
            server = await self._create_server(config)
            self.servers[config.name] = server
            self.server_configs[config.name] = config
            
            # Load available tools
            await self._refresh_tools(config.name)
            
            self.logger.info(f"MCP server '{config.name}' added successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add MCP server '{config.name}': {e}")
            return False
    
    async def _create_server(self, config: MCPServerConfig) -> MCPServer:
        """Create MCP server based on configuration."""
        
        if config.server_type == "stdio":
            return MCPServerStdio(
                name=config.name,
                params=config.connection_params,
                cache_tools_list=config.tools_cache_enabled
            )
        elif config.server_type == "sse":
            return MCPServerSse(
                name=config.name,
                params=config.connection_params,
                cache_tools_list=config.tools_cache_enabled
            )
        elif config.server_type == "streamable_http":
            return MCPServerStreamableHttp(
                name=config.name,
                params=config.connection_params,
                cache_tools_list=config.tools_cache_enabled
            )
        else:
            raise ValueError(f"Unsupported server type: {config.server_type}")
    
    async def _refresh_tools(self, server_name: str):
        """Refresh available tools for a server."""
        
        if server_name not in self.servers:
            return
        
        try:
            server = self.servers[server_name]
            tools = await server.list_tools()
            self.available_tools[server_name] = {
                tool.get("name", ""): tool for tool in tools
            }
            
        except Exception as e:
            self.logger.error(f"Failed to refresh tools for server '{server_name}': {e}")
    
    async def list_all_tools(self) -> Dict[str, List[Dict[str, Any]]]:
        """List all available tools from all servers."""
        
        all_tools = {}
        for server_name, tools in self.available_tools.items():
            all_tools[server_name] = list(tools.values())
        
        return all_tools
    
    async def call_tool(self, 
                       server_name: str, 
                       tool_name: str, 
                       arguments: Dict[str, Any]) -> MCPToolResult:
        """
        Call tool on specified MCP server.
        
        Args:
            server_name: Name of MCP server
            tool_name: Name of tool to call
            arguments: Tool arguments
            
        Returns:
            MCPToolResult: Result of tool execution
        """
        start_time = asyncio.get_event_loop().time()
        
        if server_name not in self.servers:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                execution_time=0,
                error_message=f"Server '{server_name}' not found"
            )
        
        try:
            server = self.servers[server_name]
            result = await server.call_tool(tool_name, arguments)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return MCPToolResult(
                tool_name=tool_name,
                success=True,
                result=result,
                execution_time=execution_time,
                metadata={"server": server_name}
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                execution_time=execution_time,
                error_message=str(e),
                metadata={"server": server_name}
            )
    
    async def close_all_servers(self):
        """Close all MCP server connections."""
        
        for server_name, server in self.servers.items():
            try:
                await server.close()
                self.logger.info(f"Closed MCP server '{server_name}'")
            except Exception as e:
                self.logger.error(f"Error closing server '{server_name}': {e}")
        
        self.servers.clear()
        self.server_configs.clear()
        self.available_tools.clear()


class MCPIntegratedAgent(BaseAgent):
    """
    Enhanced Base Agent with MCP Integration
    
    Extends BaseAgent to support both native tools and MCP tools
    in a unified interface.
    """
    
    def __init__(self, agent_role, agent_name: str):
        super().__init__(agent_role, agent_name)
        
        # MCP integration components
        self.mcp_manager = MCPServerManager()
        self.mcp_enabled = False
        
        # Unified tool interface
        self.unified_tools = {}
        
        # MCP-specific capabilities
        self.mcp_capabilities = [
            "mcp_tool_integration",
            "unified_tool_interface",
            "multi_server_coordination",
            "tool_result_aggregation",
            "mcp_error_handling"
        ]
    
    async def initialize_mcp_integration(self, server_configs: List[MCPServerConfig]):
        """
        Initialize MCP integration with specified server configurations.
        
        Args:
            server_configs: List of MCP server configurations
        """
        self.logger.info("Initializing MCP integration...")
        
        success_count = 0
        for config in server_configs:
            if await self.mcp_manager.add_server(config):
                success_count += 1
        
        if success_count > 0:
            self.mcp_enabled = True
            await self._build_unified_tool_interface()
            self.logger.info(f"MCP integration initialized with {success_count} servers")
        else:
            self.logger.warning("MCP integration failed - no servers connected")
    
    async def _build_unified_tool_interface(self):
        """Build unified interface for native and MCP tools."""
        
        # Add native tools (would be implemented by subclasses)
        self.unified_tools["native"] = self.get_native_tools()
        
        # Add MCP tools
        mcp_tools = await self.mcp_manager.list_all_tools()
        for server_name, tools in mcp_tools.items():
            self.unified_tools[f"mcp_{server_name}"] = {
                tool["name"]: tool for tool in tools
            }
        
        total_tools = sum(len(tools) for tools in self.unified_tools.values())
        self.logger.info(f"Unified tool interface built with {total_tools} total tools")
    
    def get_native_tools(self) -> Dict[str, Any]:
        """Get native tools for this agent. Override in subclasses."""
        return {}
    
    async def execute_unified_tool(self, 
                                  tool_identifier: str, 
                                  arguments: Dict[str, Any]) -> MCPToolResult:
        """
        Execute tool using unified interface.
        
        Args:
            tool_identifier: Format "server_name:tool_name" or "native:tool_name"
            arguments: Tool arguments
            
        Returns:
            MCPToolResult: Execution result
        """
        try:
            if ":" not in tool_identifier:
                raise ValueError("Tool identifier must be in format 'server:tool_name'")
            
            server_part, tool_name = tool_identifier.split(":", 1)
            
            if server_part == "native":
                # Execute native tool
                return await self._execute_native_tool(tool_name, arguments)
            elif server_part.startswith("mcp_"):
                # Execute MCP tool
                server_name = server_part[4:]  # Remove "mcp_" prefix
                return await self.mcp_manager.call_tool(server_name, tool_name, arguments)
            else:
                raise ValueError(f"Unknown tool server: {server_part}")
                
        except Exception as e:
            return MCPToolResult(
                tool_name=tool_identifier,
                success=False,
                result=None,
                execution_time=0,
                error_message=str(e)
            )
    
    async def _execute_native_tool(self, 
                                  tool_name: str, 
                                  arguments: Dict[str, Any]) -> MCPToolResult:
        """Execute native tool. Override in subclasses."""
        
        return MCPToolResult(
            tool_name=tool_name,
            success=False,
            result=None,
            execution_time=0,
            error_message="Native tool execution not implemented"
        )
    
    async def get_available_tools(self) -> Dict[str, List[str]]:
        """Get list of all available tools."""
        
        available = {}
        
        for server_name, tools in self.unified_tools.items():
            available[server_name] = list(tools.keys())
        
        return available
    
    async def search_tools(self, query: str) -> List[Dict[str, Any]]:
        """Search for tools matching query."""
        
        matching_tools = []
        query_lower = query.lower()
        
        for server_name, tools in self.unified_tools.items():
            for tool_name, tool_info in tools.items():
                if (query_lower in tool_name.lower() or 
                    query_lower in str(tool_info).lower()):
                    matching_tools.append({
                        "server": server_name,
                        "tool_name": tool_name,
                        "tool_info": tool_info
                    })
        
        return matching_tools
    
    async def execute_tool_chain(self, tool_chain: List[Dict[str, Any]]) -> List[MCPToolResult]:
        """
        Execute a chain of tools in sequence.
        
        Args:
            tool_chain: List of tool specifications with format:
                        [{"tool": "server:tool_name", "arguments": {...}}, ...]
                        
        Returns:
            List[MCPToolResult]: Results from each tool execution
        """
        results = []
        
        for tool_spec in tool_chain:
            tool_identifier = tool_spec.get("tool", "")
            arguments = tool_spec.get("arguments", {})
            
            # Allow results from previous tools to be used in arguments
            if results:
                arguments = await self._inject_previous_results(arguments, results)
            
            result = await self.execute_unified_tool(tool_identifier, arguments)
            results.append(result)
            
            # Stop chain if tool fails and no error handling specified
            if not result.success and not tool_spec.get("continue_on_error", False):
                break
        
        return results
    
    async def _inject_previous_results(self, 
                                     arguments: Dict[str, Any], 
                                     previous_results: List[MCPToolResult]) -> Dict[str, Any]:
        """Inject results from previous tools into current arguments."""
        
        # Simple placeholder replacement
        for key, value in arguments.items():
            if isinstance(value, str) and value.startswith("${result_"):
                try:
                    result_index = int(value[9:-1])  # Extract index from ${result_N}
                    if 0 <= result_index < len(previous_results):
                        arguments[key] = previous_results[result_index].result
                except (ValueError, IndexError):
                    pass  # Keep original value if replacement fails
        
        return arguments
    
    async def process_task_with_mcp(self, task_context: TaskContext) -> Dict[str, Any]:
        """
        Process task using both native and MCP tools.
        
        Args:
            task_context: Task context with requirements
            
        Returns:
            Dict: Enhanced task processing results
        """
        if not self.mcp_enabled:
            # Fall back to native processing
            return await self.process_task(task_context)
        
        # Enhanced processing with MCP tools
        processing_start = asyncio.get_event_loop().time()
        
        # Get tool recommendations based on task
        recommended_tools = await self._recommend_tools_for_task(task_context)
        
        # Execute recommended tools
        tool_results = []
        for tool_rec in recommended_tools:
            result = await self.execute_unified_tool(
                tool_rec["tool_identifier"], 
                tool_rec["arguments"]
            )
            tool_results.append(result)
        
        # Process native task
        native_result = await self.process_task(task_context)
        
        # Combine results
        processing_time = asyncio.get_event_loop().time() - processing_start
        
        return {
            "native_result": native_result,
            "mcp_tool_results": [result.__dict__ for result in tool_results],
            "enhanced_capabilities_used": self.mcp_capabilities,
            "processing_time": processing_time,
            "tools_executed": len(tool_results),
            "mcp_integration_active": True
        }
    
    async def _recommend_tools_for_task(self, task_context: TaskContext) -> List[Dict[str, Any]]:
        """Recommend tools based on task context."""
        
        recommendations = []
        task_desc = task_context.description.lower()
        
        # Search for relevant tools
        tool_matches = await self.search_tools(task_desc)
        
        # Score and rank tools
        for match in tool_matches[:5]:  # Top 5 matches
            score = self._calculate_tool_relevance_score(match, task_context)
            if score > 0.5:  # Relevance threshold
                recommendations.append({
                    "tool_identifier": f"{match['server']}:{match['tool_name']}",
                    "arguments": self._generate_tool_arguments(match, task_context),
                    "relevance_score": score,
                    "rationale": f"Tool matches task context: {task_desc[:50]}..."
                })
        
        # Sort by relevance score
        recommendations.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return recommendations
    
    def _calculate_tool_relevance_score(self, 
                                      tool_match: Dict[str, Any], 
                                      task_context: TaskContext) -> float:
        """Calculate relevance score for tool match."""
        
        # Simple scoring based on keyword matches
        task_keywords = task_context.description.lower().split()
        tool_text = f"{tool_match['tool_name']} {str(tool_match['tool_info'])}".lower()
        
        matches = sum(1 for keyword in task_keywords if keyword in tool_text)
        score = matches / len(task_keywords) if task_keywords else 0
        
        return min(score, 1.0)
    
    def _generate_tool_arguments(self, 
                               tool_match: Dict[str, Any], 
                               task_context: TaskContext) -> Dict[str, Any]:
        """Generate arguments for tool based on task context."""
        
        # Basic argument generation - would be enhanced based on tool specifications
        return {
            "query": task_context.description,
            "context": str(task_context.requirements),
            "max_results": 10
        }
    
    def get_enhanced_capabilities(self) -> List[str]:
        """Get enhanced capabilities including MCP integration."""
        base_caps = self.capabilities if hasattr(self, 'capabilities') else []
        if self.mcp_enabled:
            return base_caps + self.mcp_capabilities
        return base_caps
    
    async def cleanup_mcp_integration(self):
        """Clean up MCP integration resources."""
        if self.mcp_enabled:
            await self.mcp_manager.close_all_servers()
            self.mcp_enabled = False
            self.logger.info("MCP integration cleaned up")


# Utility functions for MCP configuration

def create_filesystem_mcp_config(name: str = "filesystem", 
                                directory: str = ".") -> MCPServerConfig:
    """Create configuration for filesystem MCP server."""
    
    return MCPServerConfig(
        name=name,
        server_type="stdio",
        connection_params={
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", directory]
        },
        tools_cache_enabled=True
    )


def create_web_search_mcp_config(name: str = "web_search") -> MCPServerConfig:
    """Create configuration for web search MCP server."""
    
    return MCPServerConfig(
        name=name,
        server_type="stdio",
        connection_params={
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-web-search"]
        },
        tools_cache_enabled=True
    )


def create_database_mcp_config(name: str = "database", 
                              connection_string: str = "") -> MCPServerConfig:
    """Create configuration for database MCP server."""
    
    return MCPServerConfig(
        name=name,
        server_type="stdio",
        connection_params={
            "command": "database-mcp-server",
            "args": ["--connection", connection_string]
        },
        tools_cache_enabled=True
    )