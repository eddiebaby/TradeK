<<<<<<< HEAD
# TradeKnowledge: Book Knowledge MCP Server
## Complete Implementation Guide for Algorithmic Trading Reference System

### ⚠️ READ THIS FIRST - IMPORTANT NOTES FOR THE TEAM ⚠️

1. **Follow these instructions IN ORDER** - skipping steps will cause failures
2. **Copy-paste commands exactly** - typos will break things
3. **Check each step's "Verification" section** before moving on
4. **If something fails**, check the "Common Issues" section
5. **Work in a virtual environment** - this is mandatory
6. **Use Python 3.11+** - older versions won't work

---

## Table of Contents

1. [Quick Start (for experienced devs)](#quick-start)
2. [Detailed Setup Instructions](#detailed-setup)
3. [Project Structure](#project-structure)
4. [Step-by-Step Implementation](#implementation)
5. [Testing Guide](#testing)
6. [Deployment Instructions](#deployment)
7. [Troubleshooting](#troubleshooting)
8. [Code Examples](#code-examples)

---

## Quick Start

```bash
# For experienced developers only - everyone else use Detailed Setup
git clone https://github.com/your-org/tradeknowledge.git
cd tradeknowledge
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python setup.py build_ext --inplace  # Build C++ extensions
python scripts/init_db.py
python main.py
```

---

## Detailed Setup

### Step 1: System Requirements

#### 1.1 Check Your System

```bash
# Check Python version (MUST be 3.11 or higher)
python --version

# Check pip version
pip --version

# Check git
git --version

# Check C++ compiler
g++ --version  # Linux/Mac
# OR
cl  # Windows (in Developer Command Prompt)
```

**Expected Output:**
- Python 3.11.0 or higher
- pip 23.0 or higher
- git 2.30 or higher
- g++ 11.0 or higher (Linux/Mac) or MSVC 2019+ (Windows)

#### 1.2 Install Missing Requirements

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y python3.11 python3.11-dev python3.11-venv
sudo apt install -y build-essential cmake
sudo apt install -y libssl-dev libffi-dev
sudo apt install -y tesseract-ocr  # For OCR support
```

**macOS:**
```bash
brew install python@3.11
brew install cmake
brew install tesseract  # For OCR support
```

**Windows:**
1. Download Python 3.11 from python.org
2. Install Visual Studio 2022 Community with C++ workload
3. Install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki

### Step 2: Project Setup

#### 2.1 Create Project Directory

```bash
# Create and navigate to project directory
mkdir -p ~/projects/tradeknowledge
cd ~/projects/tradeknowledge

# Verify you're in the right place
pwd
# Should show: /home/username/projects/tradeknowledge
```

#### 2.2 Initialize Git Repository

```bash
# Initialize git
git init

# Create .gitignore
cat > .gitignore << 'EOF'
# Virtual Environment
venv/
env/
.env

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# C++ Build
*.o
*.obj
*.exe
*.dll
*.so
*.dylib
cmake-build-*/
CMakeCache.txt
CMakeFiles/

# Databases
*.db
*.sqlite
*.sqlite3
chroma_db/

# Books
books/
*.pdf
*.epub

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
logs/
*.log

# Config with secrets
config/secrets.yaml
.env.local
EOF

# Initial commit
git add .gitignore
git commit -m "Initial commit with .gitignore"
```

### Step 3: Virtual Environment Setup

#### 3.1 Create Virtual Environment

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Verify activation
which python
# Should show: /path/to/project/venv/bin/python
```

#### 3.2 Upgrade pip

```bash
# Upgrade pip to latest
python -m pip install --upgrade pip

# Verify pip version
pip --version
# Should show: pip 24.x.x or higher
```

### Step 4: Install Dependencies

#### 4.1 Create Requirements Files

```bash
# Create main requirements file
cat > requirements.txt << 'EOF'
# Core Dependencies
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3
python-multipart==0.0.6

# MCP Protocol
websockets==12.0
jsonrpc-websocket==3.1.4

# Database
chromadb==0.4.22
sqlite-utils==3.35.2
sqlalchemy==2.0.25

# Text Processing
pypdf2==3.17.4
pdfplumber==0.10.3
ebooklib==0.18
spacy==3.7.2
nltk==3.8.1
python-magic==0.4.27

# OCR Support
pytesseract==0.3.10
pdf2image==1.16.3
opencv-python==4.9.0.80

# Embeddings
openai==1.10.0
sentence-transformers==2.2.2
torch==2.1.2
transformers==4.36.2

# Math Processing
sympy==1.12
latex2sympy2==1.9.1

# Code Processing
pygments==2.17.2
black==23.12.1

# C++ Bindings
pybind11==2.11.1
cmake==3.28.1

# Utilities
python-dotenv==1.0.0
click==8.1.7
rich==13.7.0
tqdm==4.66.1
pyyaml==6.0.1

# Caching
redis==5.0.1
cachetools==5.3.2

# Testing
pytest==7.4.4
pytest-asyncio==0.21.1
pytest-cov==4.1.0

# Development
ipython==8.19.0
jupyter==1.0.0
pre-commit==3.6.0
EOF

# Create dev requirements
cat > requirements-dev.txt << 'EOF'
# Linting
flake8==7.0.0
black==23.12.1
isort==5.13.2
mypy==1.8.0

# Documentation
mkdocs==1.5.3
mkdocs-material==9.5.3
mkdocstrings[python]==0.24.0

# Performance Profiling
py-spy==0.3.14
memory-profiler==0.61.0
line-profiler==4.1.1
EOF
```

#### 4.2 Install Requirements

```bash
# Install main requirements
pip install -r requirements.txt

# Install dev requirements
pip install -r requirements-dev.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

**Verification:**
```bash
# Test imports
python -c "import fastapi, chromadb, pdfplumber, spacy, torch; print('All imports successful!')"
```

### Step 5: Project Structure Creation

#### 5.1 Create Directory Structure

```bash
# Create all necessary directories
mkdir -p src/{core,ingestion,search,mcp,utils,cpp}
mkdir -p tests/{unit,integration,performance}
mkdir -p config
mkdir -p scripts
mkdir -p data/{books,chunks,embeddings}
mkdir -p logs
mkdir -p docs
mkdir -p notebooks

# Create __init__.py files
touch src/__init__.py
touch src/core/__init__.py
touch src/ingestion/__init__.py
touch src/search/__init__.py
touch src/mcp/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py

# Verify structure
tree -d -L 3
```

**Expected Structure:**
```
.
├── config/
├── data/
│   ├── books/
│   ├── chunks/
│   └── embeddings/
├── docs/
├── logs/
├── notebooks/
├── scripts/
├── src/
│   ├── core/
│   ├── cpp/
│   ├── ingestion/
│   ├── mcp/
│   ├── search/
│   └── utils/
├── tests/
│   ├── integration/
│   ├── performance/
│   └── unit/
└── venv/
```

### Step 6: Configuration Setup

#### 6.1 Create Configuration Files

```bash
# Create main config
cat > config/config.yaml << 'EOF'
# TradeKnowledge Configuration

app:
  name: "TradeKnowledge"
  version: "1.0.0"
  debug: true
  log_level: "INFO"

server:
  host: "0.0.0.0"
  port: 8000
  workers: 4

database:
  chroma:
    persist_directory: "./data/chromadb"
    collection_name: "trading_books"
  sqlite:
    path: "./data/knowledge.db"
    fts_version: "fts5"

ingestion:
  chunk_size: 1000
  chunk_overlap: 200
  min_chunk_size: 100
  max_chunk_size: 2000
  
embedding:
  model: "text-embedding-ada-002"  # or "all-mpnet-base-v2" for local
  batch_size: 100
  cache_embeddings: true

search:
  default_results: 10
  max_results: 50
  min_score: 0.7
  hybrid_weight: 0.7  # 0.7 semantic, 0.3 exact

cache:
  redis:
    host: "localhost"
    port: 6379
    db: 0
    ttl: 3600  # 1 hour
  memory:
    max_size: 1000
    ttl: 600  # 10 minutes

performance:
  use_cpp_extensions: true
  thread_pool_size: 8
  batch_processing: true
EOF

# Create environment template
cat > .env.example << 'EOF'
# OpenAI API Key (for embeddings)
OPENAI_API_KEY=your_key_here

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/tradeknowledge.log

# Development
DEBUG=true
TESTING=false
EOF

# Copy to actual .env
cp .env.example .env
echo "⚠️  Edit .env and add your OpenAI API key!"
```

### Step 7: Core Implementation Files

#### 7.1 Create Main Application Entry Point

```python
# Create src/main.py
cat > src/main.py << 'EOF'
#!/usr/bin/env python3
"""
TradeKnowledge - Main Application Entry Point
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from core.config import load_config, Config
from mcp.server import MCPServer
from utils.logging import setup_logging

# Load configuration
config: Config = load_config()

# Setup logging
logger = setup_logging(config.app.log_level)

# Create FastAPI app
app = FastAPI(
    title=config.app.name,
    version=config.app.version,
    description="Book Knowledge MCP Server for Algorithmic Trading"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MCP Server
mcp_server = MCPServer(config)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info(f"Starting {config.app.name} v{config.app.version}")
    await mcp_server.initialize()
    logger.info("Server initialized successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down server")
    await mcp_server.cleanup()

# Mount MCP routes
app.mount("/mcp", mcp_server.app)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": config.app.name,
        "version": config.app.version,
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health = await mcp_server.health_check()
    return health

def main():
    """Main entry point"""
    uvicorn.run(
        "main:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.app.debug,
        workers=1 if config.app.debug else config.server.workers,
        log_level=config.app.log_level.lower()
    )

if __name__ == "__main__":
    main()
EOF

# Make it executable
chmod +x src/main.py
```

#### 7.2 Create Configuration Module

```python
# Create src/core/config.py
cat > src/core/config.py << 'EOF'
"""
Configuration management for TradeKnowledge
"""

from pathlib import Path
from typing import Optional
import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class AppConfig(BaseModel):
    name: str = "TradeKnowledge"
    version: str = "1.0.0"
    debug: bool = True
    log_level: str = "INFO"

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4

class ChromaConfig(BaseModel):
    persist_directory: str = "./data/chromadb"
    collection_name: str = "trading_books"

class SQLiteConfig(BaseModel):
    path: str = "./data/knowledge.db"
    fts_version: str = "fts5"

class DatabaseConfig(BaseModel):
    chroma: ChromaConfig
    sqlite: SQLiteConfig

class IngestionConfig(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000

class EmbeddingConfig(BaseModel):
    model: str = "text-embedding-ada-002"
    batch_size: int = 100
    cache_embeddings: bool = True

class SearchConfig(BaseModel):
    default_results: int = 10
    max_results: int = 50
    min_score: float = 0.7
    hybrid_weight: float = 0.7

class RedisConfig(BaseModel):
    host: str = Field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    port: int = Field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    db: int = 0
    ttl: int = 3600

class MemoryCacheConfig(BaseModel):
    max_size: int = 1000
    ttl: int = 600

class CacheConfig(BaseModel):
    redis: RedisConfig
    memory: MemoryCacheConfig

class PerformanceConfig(BaseModel):
    use_cpp_extensions: bool = True
    thread_pool_size: int = 8
    batch_processing: bool = True

class Config(BaseModel):
    app: AppConfig
    server: ServerConfig
    database: DatabaseConfig
    ingestion: IngestionConfig
    embedding: EmbeddingConfig
    search: SearchConfig
    cache: CacheConfig
    performance: PerformanceConfig

def load_config(config_path: Optional[Path] = None) -> Config:
    """Load configuration from YAML file"""
    if config_path is None:
        config_path = Path("config/config.yaml")
    
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    return Config(**config_dict)

# Singleton instance
_config: Optional[Config] = None

def get_config() -> Config:
    """Get configuration singleton"""
    global _config
    if _config is None:
        _config = load_config()
    return _config
EOF
```

#### 7.3 Create Logging Utility

```python
# Create src/utils/logging.py
cat > src/utils/logging.py << 'EOF'
"""
Logging configuration for TradeKnowledge
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from rich.logging import RichHandler
from rich.console import Console

console = Console()

def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    rich_output: bool = True
) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        level: Logging level
        log_file: Optional log file path
        rich_output: Use rich console output
        
    Returns:
        Logger instance
    """
    # Create logs directory if needed
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"tradeknowledge_{datetime.now():%Y%m%d_%H%M%S}.log"
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler with rich formatting
    if rich_output:
        console_handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_level=True,
            show_path=True
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
    
    console_handler.setLevel(getattr(logging, level.upper()))
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
    )
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file
    logger.addHandler(file_handler)
    
    # Log startup
    logger.info(f"Logging initialized - Level: {level}, File: {log_file}")
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name)
EOF
```

### Step 8: Database Initialization

#### 8.1 Create Database Schema

```python
# Create scripts/init_db.py
cat > scripts/init_db.py << 'EOF'
#!/usr/bin/env python3
"""
Initialize databases for TradeKnowledge
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import sqlite3
import logging
from datetime import datetime
import chromadb
from chromadb.config import Settings

from core.config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_sqlite():
    """Initialize SQLite database with FTS5"""
    config = get_config()
    db_path = Path(config.database.sqlite.path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Initializing SQLite database at {db_path}")
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Create main chunks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            book_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding_id TEXT,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(book_id, chunk_index)
        )
    """)
    
    # Create FTS5 virtual table
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            id UNINDEXED,
            text,
            content=chunks,
            content_rowid=rowid,
            tokenize='porter unicode61'
        )
    """)
    
    # Create triggers to keep FTS in sync
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS chunks_ai 
        AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, id, text) 
            VALUES (new.rowid, new.id, new.text);
        END
    """)
    
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS chunks_ad 
        AFTER DELETE ON chunks BEGIN
            DELETE FROM chunks_fts WHERE rowid = old.rowid;
        END
    """)
    
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS chunks_au 
        AFTER UPDATE ON chunks BEGIN
            DELETE FROM chunks_fts WHERE rowid = old.rowid;
            INSERT INTO chunks_fts(rowid, id, text) 
            VALUES (new.rowid, new.id, new.text);
        END
    """)
    
    # Create books table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS books (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            author TEXT,
            isbn TEXT,
            file_path TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            total_chunks INTEGER DEFAULT 0,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            indexed_at TIMESTAMP,
            UNIQUE(file_hash)
        )
    """)
    
    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_book_id ON chunks(book_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_embedding_id ON chunks(embedding_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_books_file_hash ON books(file_hash)")
    
    # Create search history table for analytics
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            query_type TEXT NOT NULL,
            results_count INTEGER,
            execution_time_ms INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
    
    logger.info("SQLite database initialized successfully")

def init_chromadb():
    """Initialize ChromaDB for vector storage"""
    config = get_config()
    persist_dir = Path(config.database.chroma.persist_directory)
    persist_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Initializing ChromaDB at {persist_dir}")
    
    # Create ChromaDB client
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    
    # Create or get collection
    collection = client.get_or_create_collection(
        name=config.database.chroma.collection_name,
        metadata={
            "description": "Trading and ML book embeddings",
            "created_at": datetime.now().isoformat()
        }
    )
    
    logger.info(f"ChromaDB collection '{config.database.chroma.collection_name}' ready")
    logger.info(f"Current document count: {collection.count()}")

def verify_installation():
    """Verify all components are working"""
    logger.info("Verifying installation...")
    
    # Test SQLite
    config = get_config()
    conn = sqlite3.connect(config.database.sqlite.path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    logger.info(f"SQLite tables: {[t[0] for t in tables]}")
    conn.close()
    
    # Test ChromaDB
    client = chromadb.PersistentClient(path=config.database.chroma.persist_directory)
    collections = client.list_collections()
    logger.info(f"ChromaDB collections: {[c.name for c in collections]}")
    
    logger.info("✅ All components verified successfully!")

def main():
    """Main initialization function"""
    logger.info("Starting database initialization...")
    
    try:
        init_sqlite()
        init_chromadb()
        verify_installation()
        logger.info("✅ Database initialization complete!")
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        raise

if __name__ == "__main__":
    main()
EOF

# Make executable
chmod +x scripts/init_db.py
```

### Step 9: Create Basic MCP Server

```python
# Create src/mcp/server.py
cat > src/mcp/server.py << 'EOF'
"""
MCP Server implementation for TradeKnowledge
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from core.config import Config
from search.hybrid_search import HybridSearch
from ingestion.book_processor import BookProcessor

logger = logging.getLogger(__name__)

class MCPRequest(BaseModel):
    """MCP Request structure"""
    jsonrpc: str = "2.0"
    method: str
    params: Dict[str, Any]
    id: Optional[str] = None

class MCPResponse(BaseModel):
    """MCP Response structure"""
    jsonrpc: str = "2.0"
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[str] = None

class MCPServer:
    """Main MCP Server implementation"""
    
    def __init__(self, config: Config):
        self.config = config
        self.app = FastAPI()
        self.search_engine: Optional[HybridSearch] = None
        self.book_processor: Optional[BookProcessor] = None
        self.active_connections: List[WebSocket] = []
        
        # Setup routes
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup WebSocket routes"""
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.handle_connection(websocket)
    
    async def initialize(self):
        """Initialize server components"""
        logger.info("Initializing MCP Server components...")
        
        # Initialize search engine
        self.search_engine = HybridSearch(self.config)
        await self.search_engine.initialize()
        
        # Initialize book processor
        self.book_processor = BookProcessor(self.config)
        await self.book_processor.initialize()
        
        logger.info("MCP Server initialized successfully")
    
    async def cleanup(self):
        """Cleanup server resources"""
        logger.info("Cleaning up MCP Server...")
        
        # Close all connections
        for connection in self.active_connections:
            await connection.close()
        
        # Cleanup components
        if self.search_engine:
            await self.search_engine.cleanup()
        if self.book_processor:
            await self.book_processor.cleanup()
    
    async def handle_connection(self, websocket: WebSocket):
        """Handle WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        try:
            while True:
                # Receive message
                data = await websocket.receive_text()
                request = MCPRequest.parse_raw(data)
                
                # Process request
                response = await self.process_request(request)
                
                # Send response
                await websocket.send_text(response.json())
                
        except WebSocketDisconnect:
            logger.info("Client disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
    
    async def process_request(self, request: MCPRequest) -> MCPResponse:
        """Process MCP request"""
        logger.debug(f"Processing request: {request.method}")
        
        try:
            # Route to appropriate handler
            if request.method == "search_semantic":
                result = await self.handle_search_semantic(request.params)
            elif request.method == "search_exact":
                result = await self.handle_search_exact(request.params)
            elif request.method == "search_hybrid":
                result = await self.handle_search_hybrid(request.params)
            elif request.method == "get_chunk_context":
                result = await self.handle_get_context(request.params)
            elif request.method == "list_books":
                result = await self.handle_list_books(request.params)
            elif request.method == "add_book":
                result = await self.handle_add_book(request.params)
            else:
                raise ValueError(f"Unknown method: {request.method}")
            
            return MCPResponse(
                result=result,
                id=request.id
            )
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return MCPResponse(
                error={
                    "code": -32603,
                    "message": str(e)
                },
                id=request.id
            )
    
    async def handle_search_semantic(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle semantic search request"""
        query = params.get("query", "")
        num_results = params.get("num_results", self.config.search.default_results)
        filter_books = params.get("filter_books", None)
        
        results = await self.search_engine.search_semantic(
            query=query,
            num_results=num_results,
            filter_books=filter_books
        )
        
        return results
    
    async def handle_search_exact(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle exact search request"""
        query = params.get("query", "")
        num_results = params.get("num_results", self.config.search.default_results)
        filter_books = params.get("filter_books", None)
        
        results = await self.search_engine.search_exact(
            query=query,
            num_results=num_results,
            filter_books=filter_books
        )
        
        return results
    
    async def handle_search_hybrid(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle hybrid search request"""
        query = params.get("query", "")
        num_results = params.get("num_results", self.config.search.default_results)
        filter_books = params.get("filter_books", None)
        semantic_weight = params.get("semantic_weight", self.config.search.hybrid_weight)
        
        results = await self.search_engine.search_hybrid(
            query=query,
            num_results=num_results,
            filter_books=filter_books,
            semantic_weight=semantic_weight
        )
        
        return results
    
    async def handle_get_context(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get context request"""
        chunk_id = params.get("chunk_id", "")
        before_chunks = params.get("before_chunks", 1)
        after_chunks = params.get("after_chunks", 1)
        
        context = await self.search_engine.get_chunk_context(
            chunk_id=chunk_id,
            before_chunks=before_chunks,
            after_chunks=after_chunks
        )
        
        return context
    
    async def handle_list_books(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle list books request"""
        category = params.get("category", None)
        
        books = await self.book_processor.list_books(category=category)
        
        return {"books": books}
    
    async def handle_add_book(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle add book request"""
        file_path = params.get("file_path", "")
        metadata = params.get("metadata", {})
        
        result = await self.book_processor.add_book(
            file_path=file_path,
            metadata=metadata
        )
        
        return result
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "search_engine": self.search_engine is not None,
                "book_processor": self.book_processor is not None,
                "active_connections": len(self.active_connections)
            }
        }
EOF
```

### Step 10: Run Initial Tests

#### 10.1 Create Test Script

```bash
# Create scripts/test_setup.py
cat > scripts/test_setup.py << 'EOF'
#!/usr/bin/env python3
"""
Test script to verify setup
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_imports():
    """Test all imports"""
    logger.info("Testing imports...")
    
    try:
        import fastapi
        import chromadb
        import pdfplumber
        import spacy
        import torch
        from core.config import get_config
        
        logger.info("✅ All imports successful!")
        return True
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False

async def test_config():
    """Test configuration"""
    logger.info("Testing configuration...")
    
    try:
        from core.config import get_config
        config = get_config()
        logger.info(f"✅ Config loaded: {config.app.name} v{config.app.version}")
        return True
    except Exception as e:
        logger.error(f"❌ Config failed: {e}")
        return False

async def test_database():
    """Test database connections"""
    logger.info("Testing database connections...")
    
    try:
        import sqlite3
        from core.config import get_config
        
        config = get_config()
        conn = sqlite3.connect(config.database.sqlite.path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        conn.close()
        
        logger.info(f"✅ SQLite OK - Tables: {len(tables)}")
        
        import chromadb
        client = chromadb.PersistentClient(path=config.database.chroma.persist_directory)
        collections = client.list_collections()
        logger.info(f"✅ ChromaDB OK - Collections: {len(collections)}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Database test failed: {e}")
        return False

async def main():
    """Run all tests"""
    logger.info("Starting setup verification...")
    
    tests = [
        test_imports(),
        test_config(),
        test_database()
    ]
    
    results = await asyncio.gather(*tests)
    
    if all(results):
        logger.info("✅ All tests passed! Setup is complete.")
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
EOF

# Make executable
chmod +x scripts/test_setup.py
```

#### 10.2 Run Tests

```bash
# Initialize database
python scripts/init_db.py

# Run setup tests
python scripts/test_setup.py
```

---

## Implementation

### Phase 1: Basic Functionality (Week 1-2)

Follow these steps IN ORDER:

1. **Run database initialization**:
   ```bash
   python scripts/init_db.py
   ```

2. **Implement basic PDF parser**:
   - Create `src/ingestion/pdf_parser.py`
   - Start with PyPDF2 for clean PDFs
   - Add error handling

3. **Implement text chunker**:
   - Create `src/ingestion/text_chunker.py`
   - Use simple character-based chunking first
   - Add overlap support

4. **Create embedding generator**:
   - Create `src/ingestion/embeddings.py`
   - Start with OpenAI embeddings
   - Add batching support

5. **Implement basic search**:
   - Create `src/search/vector_search.py`
   - Create `src/search/text_search.py`
   - Test with sample data

### Phase 2: Advanced Features (Week 3-4)

[Continue with remaining implementation details...]

---

## Testing

### Running Unit Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_pdf_parser.py

# Run with verbose output
pytest -v
```

### Running Integration Tests

```bash
# Run integration tests
pytest tests/integration/

# Run performance tests
pytest tests/performance/ -v
```

---

## Deployment

### Local Development

```bash
# Start server in development mode
python src/main.py

# Or use uvicorn directly
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Deployment

```bash
# Build C++ extensions
python setup.py build_ext --inplace

# Run with gunicorn
gunicorn src.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## Troubleshooting

### Common Issues

#### Issue: Import errors
**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Reinstall requirements
pip install -r requirements.txt
```

#### Issue: Database not found
**Solution**:
```bash
# Run initialization
python scripts/init_db.py

# Check permissions
ls -la data/
```

#### Issue: C++ compilation fails
**Solution**:
```bash
# Install build tools
sudo apt-get install build-essential  # Ubuntu
brew install gcc  # macOS

# Try without C++ extensions
# Edit config/config.yaml and set use_cpp_extensions: false
```

#### Issue: Out of memory during embedding
**Solution**:
```python
# Reduce batch size in config/config.yaml
embedding:
  batch_size: 50  # Reduce from 100
```

---

## Code Examples

### Example: Adding a Book

```python
import asyncio
from pathlib import Path
from src.ingestion.book_processor import BookProcessor
from src.core.config import get_config

async def add_book_example():
    config = get_config()
    processor = BookProcessor(config)
    await processor.initialize()
    
    result = await processor.add_book(
        file_path="books/algorithmic_trading.pdf",
        metadata={
            "category": "trading",
            "difficulty": "intermediate"
        }
    )
    
    print(f"Book added: {result}")

# Run
asyncio.run(add_book_example())
```

### Example: Searching for Content

```python
import asyncio
from src.search.hybrid_search import HybridSearch
from src.core.config import get_config

async def search_example():
    config = get_config()
    search = HybridSearch(config)
    await search.initialize()
    
    # Semantic search
    results = await search.search_semantic(
        query="momentum trading strategies with Python",
        num_results=5
    )
    
    for result in results["results"]:
        print(f"Score: {result['score']:.3f}")
        print(f"Book: {result['metadata']['book']}")
        print(f"Text: {result['text'][:200]}...")
        print("-" * 50)

# Run
asyncio.run(search_example())
```

---

## Next Steps

1. **Complete Phase 1** implementation
2. **Add sample books** to test with
3. **Run performance benchmarks**
4. **Create API documentation**
5. **Setup monitoring**

Remember:
- Commit code frequently
- Test each component individually
- Document any deviations from this plan
- Ask for help when stuck

---

**END OF README**
=======
# TradeK
>>>>>>> b2b9dceac139a33dbec370ff48a237d7c8f9bb2d
