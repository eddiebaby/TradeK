"""
MCP Server for TradeKnowledge

This provides an MCP (Model Context Protocol) interface to the
TradeKnowledge system, allowing AI assistants to search and
manage the book knowledge base.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
import json
from pathlib import Path

# MCP imports - these would need to be installed
try:
    from mcp import McpServer, Tool, ToolResult
    from mcp.types import TextContent
except ImportError:
    # Fallback if MCP not available
    print("MCP library not available. Install with: pip install mcp")
    
    class McpServer:
        def __init__(self, name: str, version: str):
            self.name = name
            self.version = version
            self.tools = []
        
        def tool(self, name: str, description: str):
            def decorator(func):
                self.tools.append({'name': name, 'description': description, 'func': func})
                return func
            return decorator
        
        async def run(self):
            print(f"Mock MCP server {self.name} v{self.version} running...")
            print(f"Available tools: {[t['name'] for t in self.tools]}")

from core.config import get_config
from search.hybrid_search import HybridSearch
from ingestion.ingestion_engine import IngestionEngine

logger = logging.getLogger(__name__)

class TradeKnowledgeServer:
    """MCP Server for TradeKnowledge system"""
    
    def __init__(self):
        """Initialize the server"""
        self.config = get_config()
        self.server = McpServer("TradeKnowledge", "1.0.0")
        self.search_engine: Optional[HybridSearch] = None
        self.ingestion_engine: Optional[IngestionEngine] = None
        
        # Register all tools
        self._register_tools()
    
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing TradeKnowledge MCP server...")
        
        # Initialize search engine
        self.search_engine = HybridSearch(self.config)
        await self.search_engine.initialize()
        
        # Initialize ingestion engine
        self.ingestion_engine = IngestionEngine(self.config)
        await self.ingestion_engine.initialize()
        
        logger.info("TradeKnowledge MCP server initialized")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.search_engine:
            await self.search_engine.cleanup()
        if self.ingestion_engine:
            await self.ingestion_engine.cleanup()
    
    def _register_tools(self):
        """Register all MCP tools"""
        
        @self.server.tool(
            name="search_books",
            description="Search through the book knowledge base using semantic or exact search"
        )
        async def search_books(
            query: str,
            search_type: str = "hybrid",
            num_results: int = 10,
            filter_books: Optional[List[str]] = None
        ) -> Dict[str, Any]:
            """
            Search through books in the knowledge base.
            
            Args:
                query: Search query
                search_type: Type of search ('semantic', 'exact', or 'hybrid')
                num_results: Number of results to return (max 50)
                filter_books: Optional list of book IDs to search within
            """
            try:
                if not self.search_engine:
                    return {"error": "Search engine not initialized"}
                
                # Limit results
                num_results = min(num_results, 50)
                
                # Perform search based on type
                if search_type == "semantic":
                    results = await self.search_engine.search_semantic(
                        query, num_results, filter_books
                    )
                elif search_type == "exact":
                    results = await self.search_engine.search_exact(
                        query, num_results, filter_books
                    )
                else:  # hybrid
                    results = await self.search_engine.search_hybrid(
                        query, num_results, filter_books
                    )
                
                return {
                    "success": True,
                    "query": query,
                    "search_type": search_type,
                    "total_results": results.get("total_results", 0),
                    "search_time_ms": results.get("search_time_ms", 0),
                    "results": [
                        {
                            "book_title": r.get("book_title", ""),
                            "book_author": r.get("book_author", ""),
                            "chapter": r.get("chapter"),
                            "page": r.get("page"),
                            "score": round(r.get("score", 0), 3),
                            "text_preview": r.get("highlights", [""])[0][:200] + "..." if r.get("highlights") else "",
                            "chunk_id": r.get("chunk", {}).get("id", "") if isinstance(r.get("chunk"), dict) else getattr(r.get("chunk"), "id", "")
                        }
                        for r in results.get("results", [])
                    ]
                }
                
            except Exception as e:
                logger.error(f"Error in search_books: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
        
        @self.server.tool(
            name="get_chunk_context",
            description="Get expanded context around a specific chunk"
        )
        async def get_chunk_context(
            chunk_id: str,
            before_chunks: int = 2,
            after_chunks: int = 2
        ) -> Dict[str, Any]:
            """
            Get expanded context around a chunk.
            
            Args:
                chunk_id: ID of the chunk to get context for
                before_chunks: Number of chunks before to include
                after_chunks: Number of chunks after to include
            """
            try:
                if not self.search_engine:
                    return {"error": "Search engine not initialized"}
                
                context = await self.search_engine.get_chunk_context(
                    chunk_id, before_chunks, after_chunks
                )
                
                if "error" in context:
                    return context
                
                # Format response for readability
                result = {
                    "success": True,
                    "chunk_id": chunk_id,
                    "main_chunk": {
                        "text": context.get("chunk", {}).get("text", ""),
                        "chapter": context.get("chunk", {}).get("chapter"),
                        "page": context.get("chunk", {}).get("page_start")
                    },
                    "context_before": [
                        {
                            "text": chunk.get("text", ""),
                            "chunk_index": chunk.get("chunk_index", 0)
                        }
                        for chunk in context.get("context", {}).get("before", [])
                    ],
                    "context_after": [
                        {
                            "text": chunk.get("text", ""),
                            "chunk_index": chunk.get("chunk_index", 0)
                        }
                        for chunk in context.get("context", {}).get("after", [])
                    ]
                }
                
                return result
                
            except Exception as e:
                logger.error(f"Error in get_chunk_context: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
        
        @self.server.tool(
            name="list_books",
            description="List all books in the knowledge base"
        )
        async def list_books(category: Optional[str] = None) -> Dict[str, Any]:
            """
            List all books in the knowledge base.
            
            Args:
                category: Optional category to filter by
            """
            try:
                if not self.ingestion_engine:
                    return {"error": "Ingestion engine not initialized"}
                
                books = await self.ingestion_engine.list_books(category)
                
                return {
                    "success": True,
                    "total_books": len(books),
                    "books": [
                        {
                            "id": book["id"],
                            "title": book["title"],
                            "author": book.get("author", "Unknown"),
                            "categories": book.get("categories", []),
                            "total_chunks": book.get("total_chunks", 0),
                            "indexed_at": book.get("indexed_at")
                        }
                        for book in books
                    ]
                }
                
            except Exception as e:
                logger.error(f"Error in list_books: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
        
        @self.server.tool(
            name="get_book_details",
            description="Get detailed information about a specific book"
        )
        async def get_book_details(book_id: str) -> Dict[str, Any]:
            """
            Get detailed information about a book.
            
            Args:
                book_id: ID of the book to get details for
            """
            try:
                if not self.ingestion_engine:
                    return {"error": "Ingestion engine not initialized"}
                
                details = await self.ingestion_engine.get_book_details(book_id)
                
                if not details:
                    return {
                        "success": False,
                        "error": "Book not found"
                    }
                
                return {
                    "success": True,
                    "book": {
                        "id": details["id"],
                        "title": details["title"],
                        "author": details.get("author", "Unknown"),
                        "isbn": details.get("isbn"),
                        "file_path": details["file_path"],
                        "total_pages": details.get("total_pages", 0),
                        "total_chunks": details.get("total_chunks", 0),
                        "categories": details.get("categories", []),
                        "created_at": details["created_at"],
                        "indexed_at": details.get("indexed_at"),
                        "statistics": details.get("chunk_statistics", {})
                    }
                }
                
            except Exception as e:
                logger.error(f"Error in get_book_details: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
        
        @self.server.tool(
            name="add_book",
            description="Add a new book to the knowledge base"
        )
        async def add_book(
            file_path: str,
            categories: Optional[List[str]] = None,
            description: Optional[str] = None
        ) -> Dict[str, Any]:
            """
            Add a book to the knowledge base.
            
            Args:
                file_path: Path to the PDF file
                categories: Optional list of categories
                description: Optional description
            """
            try:
                if not self.ingestion_engine:
                    return {"error": "Ingestion engine not initialized"}
                
                # Prepare metadata
                metadata = {}
                if categories:
                    metadata["categories"] = categories
                if description:
                    metadata["description"] = description
                
                result = await self.ingestion_engine.add_book(file_path, metadata)
                
                return result
                
            except Exception as e:
                logger.error(f"Error in add_book: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
        
        @self.server.tool(
            name="get_search_stats",
            description="Get statistics about the search engine"
        )
        async def get_search_stats() -> Dict[str, Any]:
            """Get search engine statistics"""
            try:
                if not self.search_engine:
                    return {"error": "Search engine not initialized"}
                
                stats = self.search_engine.get_stats()
                
                return {
                    "success": True,
                    "stats": stats
                }
                
            except Exception as e:
                logger.error(f"Error in get_search_stats: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
    
    async def run(self):
        """Run the MCP server"""
        try:
            await self.initialize()
            await self.server.run()
        finally:
            await self.cleanup()

# Create server instance
server = TradeKnowledgeServer()

# Example usage and testing
async def test_server():
    """Test the MCP server functionality"""
    print("Testing TradeKnowledge MCP Server...")
    
    # Initialize server
    await server.initialize()
    
    # Test available tools
    print(f"Available tools: {[t['name'] for t in server.server.tools]}")
    
    # Test search if data exists
    try:
        search_result = await server.server.tools[0]['func']("trading strategies", "hybrid", 3)
        print(f"Search test result: {search_result.get('total_results', 0)} results found")
    except Exception as e:
        print(f"Search test failed (this is normal if no books are indexed): {e}")
    
    # Test book listing
    try:
        books_result = await server.server.tools[2]['func']()  # list_books
        print(f"Books in system: {books_result.get('total_books', 0)}")
    except Exception as e:
        print(f"Book listing failed: {e}")
    
    # Cleanup
    await server.cleanup()
    print("Server test completed")

if __name__ == "__main__":
    # Run test
    asyncio.run(test_server())