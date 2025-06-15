"""
Main application entry point for TradeKnowledge

This provides CLI interface and coordinates all system components.
"""

import asyncio
import logging
import argparse
import sys
from pathlib import Path
from typing import Optional

# Setup logging first - FIXED: Ensure logs directory exists
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'tradeknowledge.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Import our components
from .ingestion.ingestion_engine import IngestionEngine
from .ingestion.enhanced_book_processor import EnhancedBookProcessor
from .ingestion.resource_monitor import ResourceMonitor, ResourceLimits
from .search.hybrid_search import HybridSearch
from .mcp.server import TradeKnowledgeServer

class TradeKnowledgeApp:
    """Main application class"""
    
    def __init__(self):
        """Initialize the application"""
        self.ingestion_engine: Optional[IngestionEngine] = None
        self.search_engine: Optional[HybridSearch] = None
        self.mcp_server: Optional[TradeKnowledgeServer] = None
    
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing TradeKnowledge application...")
        
        # PERFORMANCE FIX: Initialize independent components in parallel
        self.ingestion_engine = IngestionEngine()
        self.search_engine = HybridSearch()
        self.mcp_server = TradeKnowledgeServer()
        
        # Initialize components in parallel (they don't depend on each other)
        await asyncio.gather(
            self.ingestion_engine.initialize(),
            self.search_engine.initialize(),
            self.mcp_server.initialize()
        )
        
        logger.info("TradeKnowledge application initialized successfully")
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up...")
        
        # Cleanup components in parallel
        cleanup_tasks = []
        if self.mcp_server:
            cleanup_tasks.append(self.mcp_server.cleanup())
        if self.search_engine:
            cleanup_tasks.append(self.search_engine.cleanup())
        if self.ingestion_engine:
            cleanup_tasks.append(self.ingestion_engine.cleanup())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
    
    async def add_book_command(self, file_path: str, categories: Optional[str] = None):
        """Add a book via CLI"""
        if not self.ingestion_engine:
            print("Error: Application not initialized")
            return
        
        print(f"Adding book: {file_path}")
        
        # Parse categories
        category_list = []
        if categories:
            category_list = [cat.strip() for cat in categories.split(',')]
        
        # Prepare metadata
        metadata = {}
        if category_list:
            metadata['categories'] = category_list
        
        try:
            result = await self.ingestion_engine.add_book(file_path, metadata)
            
            if result['success']:
                print(f"✅ Successfully added book: {result['title']}")
                print(f"   Book ID: {result['book_id']}")
                print(f"   Chunks created: {result['chunks_created']}")
                print(f"   Processing time: {result['processing_time']:.2f}s")
            else:
                print(f"❌ Failed to add book: {result['error']}")
                
        except Exception as e:
            print(f"❌ Error adding book: {e}")
    
    async def search_command(self, query: str, search_type: str = "hybrid", num_results: int = 5):
        """Search via CLI"""
        if not self.search_engine:
            print("Error: Application not initialized")
            return
        
        print(f"Searching for: '{query}' (type: {search_type})")
        
        try:
            if search_type == "semantic":
                results = await self.search_engine.search_semantic(query, num_results)
            elif search_type == "exact":
                results = await self.search_engine.search_exact(query, num_results)
            else:
                results = await self.search_engine.search_hybrid(query, num_results)
            
            print(f"\n📚 Found {results['total_results']} results in {results['search_time_ms']}ms")
            print("=" * 80)
            
            for i, result in enumerate(results['results'], 1):
                chunk = result['chunk']
                if isinstance(chunk, dict):
                    chunk_text = chunk.get('text', '')
                else:
                    chunk_text = getattr(chunk, 'text', '')
                
                print(f"\n{i}. {result['book_title']}")
                if result.get('book_author'):
                    print(f"   Author: {result['book_author']}")
                if result.get('chapter'):
                    print(f"   Chapter: {result['chapter']}")
                if result.get('page'):
                    print(f"   Page: {result['page']}")
                print(f"   Score: {result['score']:.3f}")
                
                # Show highlight or preview
                if result.get('highlights'):
                    preview = result['highlights'][0]
                else:
                    preview = chunk_text[:300] + "..." if len(chunk_text) > 300 else chunk_text
                
                print(f"   Preview: {preview}")
                print("-" * 40)
            
        except Exception as e:
            print(f"❌ Search error: {e}")
    
    async def list_books_command(self):
        """List books via CLI"""
        if not self.ingestion_engine:
            print("Error: Application not initialized")
            return
        
        try:
            books = await self.ingestion_engine.list_books()
            
            if not books:
                print("📚 No books in the knowledge base yet.")
                print("   Use 'add-book' command to add some books!")
                return
            
            print(f"📚 Books in knowledge base ({len(books)} total):")
            print("=" * 80)
            
            for book in books:
                print(f"ID: {book['id']}")
                print(f"Title: {book['title']}")
                if book.get('author'):
                    print(f"Author: {book['author']}")
                print(f"Chunks: {book['total_chunks']}")
                if book.get('categories'):
                    print(f"Categories: {', '.join(book['categories'])}")
                if book.get('indexed_at'):
                    print(f"Indexed: {book['indexed_at']}")
                print("-" * 40)
                
        except Exception as e:
            print(f"❌ Error listing books: {e}")
    
    async def stats_command(self):
        """Show system statistics"""
        if not self.search_engine or not self.ingestion_engine:
            print("Error: Application not initialized")
            return
        
        try:
            # Get book count
            books = await self.ingestion_engine.list_books()
            
            # Get search stats
            search_stats = self.search_engine.get_stats()
            
            print("📊 TradeKnowledge System Statistics")
            print("=" * 50)
            print(f"Books indexed: {len(books)}")
            print(f"Total searches: {search_stats['total_searches']}")
            print(f"Average search time: {search_stats['average_search_time_ms']:.1f}ms")
            
            # Component status
            print("\n🔧 Component Status:")
            components = search_stats['components_initialized']
            for component, status in components.items():
                status_icon = "✅" if status else "❌"
                print(f"  {status_icon} {component}")
            
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
    
    async def test_optimized_command(self, pdf_path: str):
        """Test optimized PDF processing"""
        print(f"🧪 Testing optimized PDF processing for: {pdf_path}")
        
        path = Path(pdf_path)
        if not path.exists():
            print(f"❌ File not found: {pdf_path}")
            return
        
        # Configure resource limits for WSL2
        limits = ResourceLimits(
            max_memory_percent=75.0,  # More conservative for WSL2
            max_memory_mb=1500,       # Limit to 1.5GB
            warning_threshold=60.0,   # Warn earlier
            check_interval=3.0        # Check more frequently
        )
        
        # Initialize enhanced processor
        processor = EnhancedBookProcessor()
        processor.resource_monitor = ResourceMonitor(limits)
        
        try:
            print("🔧 Initializing enhanced book processor...")
            await processor.initialize()
            
            # Add detailed resource monitoring
            async def detailed_callback(check):
                usage = check['usage']
                print(f"💾 Memory: {usage['system_used_percent']:.1f}% system, "
                      f"{usage['process_memory_mb']:.1f}MB process")
                
                if check['memory_warning']:
                    print(f"⚠️  Memory warning: {check['recommendations']}")
                
                if check['memory_critical']:
                    print(f"🚨 Memory critical: {check['recommendations']}")
            
            processor.resource_monitor.add_callback(detailed_callback)
            
            # Process the file
            print(f"🚀 Starting optimized processing...")
            result = await processor.add_book(
                pdf_path,
                metadata={
                    'categories': ['test', 'optimized'],
                    'description': f'Test processing of {path.name}'
                },
                force_reprocess=True
            )
            
            if result['success']:
                print(f"✅ Successfully processed: {result['title']}")
                print(f"📊 Chunks created: {result['chunks_created']}")
                print(f"⏱️  Processing time: {result['processing_time']:.1f}s")
                print(f"🔍 OCR used: {result['ocr_used']}")
                
                # Show content analysis
                analysis = result['content_analysis']
                print(f"📈 Content analysis: {analysis['code_blocks']} code blocks, "
                      f"{analysis['formulas']} formulas, {analysis['tables']} tables")
                
            else:
                print(f"❌ Processing failed: {result['error']}")
                if 'details' in result:
                    print(f"Details: {result['details']}")
        
        except Exception as e:
            print(f"❌ Exception during processing: {e}")
            logger.error(f"Exception during optimized test: {e}", exc_info=True)
        
        finally:
            try:
                await processor.cleanup()
                print("🧹 Cleanup completed")
            except Exception as e:
                print(f"Error during cleanup: {e}")

    async def run_mcp_server(self):
        """Run the MCP server"""
        if not self.mcp_server:
            print("Error: MCP server not initialized")
            return
        
        print("🚀 Starting TradeKnowledge MCP server...")
        print("The server will provide the following tools:")
        print("  - search_books: Search the knowledge base")
        print("  - get_chunk_context: Get expanded context for a chunk")
        print("  - list_books: List all books")
        print("  - get_book_details: Get detailed book information")
        print("  - add_book: Add a new book")
        print("  - get_search_stats: Get search statistics")
        
        try:
            await self.mcp_server.run()
        except KeyboardInterrupt:
            print("\n🛑 MCP server stopped by user")
        except Exception as e:
            print(f"❌ MCP server error: {e}")

async def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="TradeKnowledge - AI-Powered Book Search System")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add book command
    add_parser = subparsers.add_parser('add-book', help='Add a book to the knowledge base')
    add_parser.add_argument('file_path', help='Path to the PDF file')
    add_parser.add_argument('--categories', help='Comma-separated categories')
    
    # Test optimized command
    test_parser = subparsers.add_parser('test-optimized', help='Test optimized PDF processing')
    test_parser.add_argument('file_path', help='Path to the PDF file')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search the knowledge base')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--type', choices=['semantic', 'exact', 'hybrid'], 
                              default='hybrid', help='Search type')
    search_parser.add_argument('--results', type=int, default=5, help='Number of results')
    
    # List books command
    subparsers.add_parser('list-books', help='List all books')
    
    # Stats command
    subparsers.add_parser('stats', help='Show system statistics')
    
    # MCP server command
    subparsers.add_parser('serve', help='Run MCP server')
    
    # Interactive mode
    subparsers.add_parser('interactive', help='Start interactive mode')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create and initialize app
    app = TradeKnowledgeApp()
    
    try:
        await app.initialize()
        
        # Execute command
        if args.command == 'add-book':
            await app.add_book_command(args.file_path, args.categories)
        
        elif args.command == 'test-optimized':
            await app.test_optimized_command(args.file_path)
        
        elif args.command == 'search':
            await app.search_command(args.query, args.type, args.results)
        
        elif args.command == 'list-books':
            await app.list_books_command()
        
        elif args.command == 'stats':
            await app.stats_command()
        
        elif args.command == 'serve':
            await app.run_mcp_server()
        
        elif args.command == 'interactive':
            await interactive_mode(app)
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Error: {e}")
    finally:
        await app.cleanup()

async def interactive_mode(app: TradeKnowledgeApp):
    """Interactive mode for easy exploration"""
    print("🎯 TradeKnowledge Interactive Mode")
    print("Commands: search, add-book, list-books, stats, help, quit")
    print("=" * 60)
    
    while True:
        try:
            command = input("\ntradeknowledge> ").strip()
            
            if not command:
                continue
            
            if command == 'quit' or command == 'exit':
                break
            
            elif command == 'help':
                print("Available commands:")
                print("  search <query>     - Search the knowledge base")
                print("  add-book <path>    - Add a book")
                print("  list-books         - List all books")
                print("  stats              - Show statistics")
                print("  quit               - Exit")
            
            elif command.startswith('search '):
                query = command[7:]
                await app.search_command(query)
            
            elif command.startswith('add-book '):
                file_path = command[9:]
                await app.add_book_command(file_path)
            
            elif command == 'list-books':
                await app.list_books_command()
            
            elif command == 'stats':
                await app.stats_command()
            
            else:
                print(f"Unknown command: {command}")
                print("Type 'help' for available commands")
        
        except KeyboardInterrupt:
            print("\nUse 'quit' to exit")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Run the main CLI
    asyncio.run(main())