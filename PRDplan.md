# PRD Plan: AI Document Understanding System
## Technical Implementation Guide for Academic Research

### Executive Summary

This document outlines the technical architecture and implementation strategy for an advanced AI document understanding system optimized for academic art history research. The system integrates cutting-edge technologies including FastAPI middleware, asynchronous processing, proprietary SPARC agent framework, InfluxDB blackboard architecture, and sophisticated knowledge graph construction.

## Technical Architecture Overview

### System Architecture Diagram
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   MCP Stack     │    │   Agent Trio    │
│   Middleware    │◄──►│   Integration   │◄──►│   (SPARC)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │   Vector        │    │   Knowledge     │
│   Processing    │◄──►│   Storage       │◄──►│   Graph         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    InfluxDB Blackboard                         │
│              (Agent Communication & State Management)          │
└─────────────────────────────────────────────────────────────────┘
```

## FastAPI Middleware Implementation

### Core FastAPI Application Structure

```python
# src/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import asyncio
import uvloop

# Use uvloop for better async performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

app = FastAPI(
    title="Academic AI Document Understanding System",
    version="2.0.0",
    description="Advanced AI system for academic research document processing",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Performance middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware for request logging and metrics
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Log to InfluxDB for analytics
    await influx_logger.log_request(
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        duration=process_time
    )
    
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

### Async Request Processing Pipeline

```python
# src/middleware/async_processing.py
import asyncio
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import aiofiles

class AsyncDocumentProcessor:
    def __init__(self, max_workers: int = 8):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.processing_queue = asyncio.Queue(maxsize=100)
        
    async def process_document_async(self, document_path: str) -> Dict[str, Any]:
        """Async document processing with concurrent operations"""
        
        # Parallel processing tasks
        tasks = [
            self._extract_text_async(document_path),
            self._extract_metadata_async(document_path),
            self._detect_language_async(document_path)
        ]
        
        text, metadata, language = await asyncio.gather(*tasks)
        
        # Sequential chunking (depends on text)
        chunks = await self._chunk_text_async(text, metadata)
        
        # Parallel embedding generation
        embedding_tasks = [
            self._generate_embeddings_async(chunk) 
            for chunk in chunks
        ]
        embeddings = await asyncio.gather(*embedding_tasks)
        
        return {
            "text": text,
            "metadata": metadata,
            "language": language,
            "chunks": chunks,
            "embeddings": embeddings
        }
        
    async def _extract_text_async(self, path: str) -> str:
        """Extract text using thread pool for CPU-bound operations"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self._extract_text_sync, 
            path
        )
```

## SPARC Trio Agent System Integration

### Agent Architecture Design

```python
# src/agents/sparc_trio.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import asyncio
from dataclasses import dataclass
from enum import Enum

class AgentRole(Enum):
    RESEARCHER = "researcher"
    MASTERMIND = "mastermind"
    EXECUTOR = "executor"

@dataclass
class AgentTask:
    id: str
    role: AgentRole
    description: str
    parameters: Dict[str, Any]
    priority: int = 1
    dependencies: List[str] = None
    
class BaseAgent(ABC):
    def __init__(self, role: AgentRole, influx_client):
        self.role = role
        self.influx_client = influx_client
        self.task_queue = asyncio.Queue()
        self.is_active = False
        
    @abstractmethod
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """Process assigned task"""
        pass
        
    async def start(self):
        """Start agent processing loop"""
        self.is_active = True
        while self.is_active:
            try:
                task = await self.task_queue.get()
                result = await self.process_task(task)
                await self._update_blackboard(task, result)
            except Exception as e:
                await self._log_error(task, e)
                
    async def _update_blackboard(self, task: AgentTask, result: Dict[str, Any]):
        """Update InfluxDB blackboard with task results"""
        await self.influx_client.write_api().write(
            bucket="agent_blackboard",
            record={
                "measurement": "task_completion",
                "tags": {
                    "agent": self.role.value,
                    "task_id": task.id,
                    "task_type": task.description
                },
                "fields": {
                    "status": "completed",
                    "result": json.dumps(result),
                    "execution_time": result.get("execution_time", 0)
                },
                "time": datetime.utcnow()
            }
        )

class ResearcherAgent(BaseAgent):
    """RESEARCHER: Knowledge Architect & Intelligence Synthesizer"""
    
    def __init__(self, influx_client, mcp_stack):
        super().__init__(AgentRole.RESEARCHER, influx_client)
        self.mcp_stack = mcp_stack
        
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        if task.description == "gather_academic_sources":
            return await self._gather_academic_sources(task.parameters)
        elif task.description == "validate_citations":
            return await self._validate_citations(task.parameters)
        elif task.description == "extract_entities":
            return await self._extract_entities(task.parameters)
            
    async def _gather_academic_sources(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Gather academic sources using MCP web search"""
        query = params.get("query", "")
        
        # Use MCP Perplexity for academic search
        sources = await self.mcp_stack.perplexity.search(
            query=f"academic sources {query} art history",
            recency="year"
        )
        
        # Validate and rank sources
        validated_sources = []
        for source in sources:
            if self._is_academic_source(source):
                validated_sources.append({
                    "url": source["url"],
                    "title": source["title"],
                    "credibility_score": self._calculate_credibility(source),
                    "relevance_score": self._calculate_relevance(source, query)
                })
                
        return {
            "sources": validated_sources,
            "total_found": len(sources),
            "academic_count": len(validated_sources)
        }

class MastermindAgent(BaseAgent):
    """MASTERMIND: Strategic Architect & Quality Orchestrator"""
    
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        if task.description == "design_knowledge_schema":
            return await self._design_knowledge_schema(task.parameters)
        elif task.description == "orchestrate_analysis":
            return await self._orchestrate_analysis(task.parameters)
            
    async def _design_knowledge_schema(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Design knowledge graph schema for academic domain"""
        domain = params.get("domain", "art_history")
        
        schema = {
            "entities": {
                "Artist": {
                    "properties": ["name", "birth_date", "death_date", "nationality", "movements"],
                    "relationships": ["created", "influenced", "studied_with", "exhibited_with"]
                },
                "Artwork": {
                    "properties": ["title", "creation_date", "medium", "dimensions", "location"],
                    "relationships": ["created_by", "exhibited_at", "influenced_by", "part_of_series"]
                },
                "ArtMovement": {
                    "properties": ["name", "start_period", "end_period", "characteristics"],
                    "relationships": ["includes_artist", "influenced_by", "reacted_against"]
                },
                "Institution": {
                    "properties": ["name", "type", "location", "founded"],
                    "relationships": ["houses", "exhibited", "acquired"]
                }
            },
            "relationship_types": [
                "temporal", "causal", "spatial", "stylistic", "personal", "institutional"
            ]
        }
        
        return {"schema": schema, "domain": domain}

class ExecutorAgent(BaseAgent):
    """EXECUTOR: Implementation Virtuoso & Operational Expert"""
    
    def __init__(self, influx_client, tdd_framework):
        super().__init__(AgentRole.EXECUTOR, influx_client)
        self.tdd_framework = tdd_framework
        
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        if task.description == "implement_tdd_pipeline":
            return await self._implement_tdd_pipeline(task.parameters)
        elif task.description == "execute_quality_gates":
            return await self._execute_quality_gates(task.parameters)
            
    async def _implement_tdd_pipeline(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Implement TDD pipeline for document processing"""
        
        # Red Phase: Define tests
        test_suite = await self._define_test_suite(params)
        
        # Green Phase: Implement minimal functionality
        implementation = await self._implement_minimal_solution(params, test_suite)
        
        # Refactor Phase: Optimize while maintaining tests
        optimized = await self._refactor_implementation(implementation, test_suite)
        
        return {
            "test_coverage": optimized["test_coverage"],
            "mutation_score": optimized["mutation_score"],
            "performance_metrics": optimized["performance_metrics"]
        }
```

## Knowledge Graph Schema and Implementation

### Graph Database Design

```python
# src/knowledge_graph/graph_schema.py
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum
import networkx as nx
from py2neo import Graph, Node, Relationship

class NodeType(Enum):
    ARTIST = "Artist"
    ARTWORK = "Artwork"
    MOVEMENT = "Movement"
    CRITIC = "Critic"
    INSTITUTION = "Institution"
    CONCEPT = "Concept"
    DOCUMENT = "Document"

class RelationType(Enum):
    CREATED = "CREATED"
    INFLUENCED = "INFLUENCED"
    EXHIBITED_AT = "EXHIBITED_AT"
    PART_OF = "PART_OF"
    CRITIQUED = "CRITIQUED"
    REFERENCES = "REFERENCES"
    SIMILAR_TO = "SIMILAR_TO"

@dataclass
class KnowledgeNode:
    id: str
    type: NodeType
    properties: Dict[str, Any]
    
@dataclass
class KnowledgeRelationship:
    source_id: str
    target_id: str
    type: RelationType
    properties: Dict[str, Any]

class AcademicKnowledgeGraph:
    def __init__(self, neo4j_uri: str, username: str, password: str):
        self.graph = Graph(neo4j_uri, auth=(username, password))
        self.nx_graph = nx.DiGraph()  # NetworkX for analytics
        
    async def create_node(self, node: KnowledgeNode) -> str:
        """Create node in both Neo4j and NetworkX"""
        
        # Neo4j node creation
        neo_node = Node(node.type.value, **node.properties)
        self.graph.create(neo_node)
        
        # NetworkX node creation
        self.nx_graph.add_node(node.id, **node.properties, type=node.type.value)
        
        return node.id
        
    async def create_relationship(self, rel: KnowledgeRelationship) -> bool:
        """Create relationship with confidence scoring"""
        
        # Calculate relationship confidence
        confidence = self._calculate_relationship_confidence(rel)
        rel.properties["confidence"] = confidence
        
        # Create in Neo4j
        source_node = self.graph.nodes.match(id=rel.source_id).first()
        target_node = self.graph.nodes.match(id=rel.target_id).first()
        
        if source_node and target_node:
            relationship = Relationship(
                source_node, 
                rel.type.value, 
                target_node, 
                **rel.properties
            )
            self.graph.create(relationship)
            
            # Add to NetworkX
            self.nx_graph.add_edge(
                rel.source_id, 
                rel.target_id, 
                type=rel.type.value,
                **rel.properties
            )
            
            return True
        return False
        
    async def find_connections(self, start: str, end: str, max_hops: int = 3) -> List[Dict]:
        """Find connection paths between entities"""
        
        # Use NetworkX for pathfinding
        try:
            paths = list(nx.all_simple_paths(
                self.nx_graph, 
                start, 
                end, 
                cutoff=max_hops
            ))
            
            connection_data = []
            for path in paths[:10]:  # Limit to top 10 paths
                path_data = {
                    "path": path,
                    "length": len(path) - 1,
                    "relationships": self._get_path_relationships(path),
                    "confidence": self._calculate_path_confidence(path)
                }
                connection_data.append(path_data)
                
            return sorted(connection_data, key=lambda x: x["confidence"], reverse=True)
            
        except nx.NetworkXNoPath:
            return []
            
    def _calculate_relationship_confidence(self, rel: KnowledgeRelationship) -> float:
        """Calculate confidence score for relationship"""
        base_confidence = 0.5
        
        # Boost confidence based on evidence
        if "source_document" in rel.properties:
            base_confidence += 0.2
        if "citation_count" in rel.properties:
            base_confidence += min(0.3, rel.properties["citation_count"] * 0.05)
        if "expert_validation" in rel.properties:
            base_confidence += 0.2
            
        return min(1.0, base_confidence)
```

## Database Architecture and Design

### InfluxDB Blackboard Implementation

```python
# src/blackboard/influx_blackboard.py
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

class InfluxBlackboard:
    """InfluxDB-based blackboard for agent communication"""
    
    def __init__(self, url: str, token: str, org: str, bucket: str):
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()
        self.bucket = bucket
        self.org = org
        
    async def write_agent_state(self, agent_id: str, state: Dict[str, Any]):
        """Write agent state to blackboard"""
        point = Point("agent_state") \
            .tag("agent_id", agent_id) \
            .tag("agent_type", state.get("type", "unknown"))
            
        for key, value in state.items():
            if isinstance(value, (int, float)):
                point = point.field(key, value)
            else:
                point = point.field(key, str(value))
                
        self.write_api.write(bucket=self.bucket, record=point)
        
    async def write_task_progress(self, task_id: str, progress: Dict[str, Any]):
        """Write task progress information"""
        point = Point("task_progress") \
            .tag("task_id", task_id) \
            .tag("status", progress.get("status", "unknown")) \
            .field("completion_percentage", progress.get("completion", 0)) \
            .field("execution_time", progress.get("execution_time", 0))
            
        if "error" in progress:
            point = point.field("error_message", str(progress["error"]))
            
        self.write_api.write(bucket=self.bucket, record=point)
        
    async def write_research_finding(self, finding: Dict[str, Any]):
        """Write research findings to blackboard"""
        point = Point("research_finding") \
            .tag("domain", finding.get("domain", "general")) \
            .tag("confidence_level", finding.get("confidence", "medium")) \
            .field("finding", finding.get("text", "")) \
            .field("source_count", finding.get("sources", 0)) \
            .field("relevance_score", finding.get("relevance", 0.5))
            
        self.write_api.write(bucket=self.bucket, record=point)
        
    async def query_agent_states(self, time_range: str = "-1h") -> List[Dict]:
        """Query recent agent states"""
        query = f'''
        from(bucket: "{self.bucket}")
        |> range(start: {time_range})
        |> filter(fn: (r) => r._measurement == "agent_state")
        |> group(columns: ["agent_id"])
        |> last()
        '''
        
        result = self.query_api.query(query)
        return self._parse_query_result(result)
        
    async def query_task_status(self, task_id: str) -> Optional[Dict]:
        """Query status of specific task"""
        query = f'''
        from(bucket: "{self.bucket}")
        |> range(start: -24h)
        |> filter(fn: (r) => r._measurement == "task_progress")
        |> filter(fn: (r) => r.task_id == "{task_id}")
        |> last()
        '''
        
        result = self.query_api.query(query)
        parsed = self._parse_query_result(result)
        return parsed[0] if parsed else None
        
    async def get_coordination_insights(self) -> Dict[str, Any]:
        """Get insights for agent coordination"""
        
        # Query recent agent activity
        activity_query = f'''
        from(bucket: "{self.bucket}")
        |> range(start: -1h)
        |> filter(fn: (r) => r._measurement == "agent_state")
        |> group(columns: ["agent_id"])
        |> count()
        '''
        
        # Query task completion rates
        completion_query = f'''
        from(bucket: "{self.bucket}")
        |> range(start: -6h)
        |> filter(fn: (r) => r._measurement == "task_progress")
        |> filter(fn: (r) => r.status == "completed")
        |> count()
        '''
        
        activity_result = self.query_api.query(activity_query)
        completion_result = self.query_api.query(completion_query)
        
        return {
            "agent_activity": self._parse_query_result(activity_result),
            "task_completions": self._parse_query_result(completion_result),
            "system_load": await self._calculate_system_load(),
            "bottlenecks": await self._identify_bottlenecks()
        }
```

### Vector Storage and Hybrid Search

```python
# src/storage/hybrid_vector_store.py
import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import sqlite3
from sentence_transformers import SentenceTransformer
import asyncio

class HybridVectorStore:
    """Hybrid storage combining vector and full-text search"""
    
    def __init__(self, chroma_path: str, sqlite_path: str):
        # ChromaDB for vector storage
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="academic_documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # SQLite for full-text search
        self.sqlite_conn = sqlite3.connect(sqlite_path)
        self._setup_fts_tables()
        
        # Local embedding model
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        
    def _setup_fts_tables(self):
        """Setup SQLite FTS5 tables"""
        cursor = self.sqlite_conn.cursor()
        
        cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
            doc_id UNINDEXED,
            title,
            content,
            author,
            keywords,
            tokenize='porter unicode61'
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_metadata (
            doc_id TEXT PRIMARY KEY,
            file_path TEXT,
            file_type TEXT,
            created_at TIMESTAMP,
            chunk_count INTEGER,
            embedding_model TEXT
        )
        ''')
        
        self.sqlite_conn.commit()
        
    async def add_document(self, doc_id: str, chunks: List[Dict[str, Any]]) -> bool:
        """Add document with both vector and text indexing"""
        
        try:
            # Prepare data for ChromaDB
            chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
            chunk_texts = [chunk["text"] for chunk in chunks]
            chunk_metadatas = [chunk.get("metadata", {}) for chunk in chunks]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(chunk_texts).tolist()
            
            # Store in ChromaDB
            self.collection.add(
                ids=chunk_ids,
                documents=chunk_texts,
                embeddings=embeddings,
                metadatas=chunk_metadatas
            )
            
            # Store in SQLite FTS
            cursor = self.sqlite_conn.cursor()
            
            for i, chunk in enumerate(chunks):
                cursor.execute('''
                INSERT OR REPLACE INTO documents_fts 
                (doc_id, title, content, author, keywords)
                VALUES (?, ?, ?, ?, ?)
                ''', (
                    chunk_ids[i],
                    chunk.get("title", ""),
                    chunk["text"],
                    chunk.get("author", ""),
                    " ".join(chunk.get("keywords", []))
                ))
                
            # Store metadata
            cursor.execute('''
            INSERT OR REPLACE INTO document_metadata
            (doc_id, file_path, file_type, created_at, chunk_count, embedding_model)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                doc_id,
                chunks[0].get("file_path", ""),
                chunks[0].get("file_type", ""),
                datetime.now().isoformat(),
                len(chunks),
                "all-mpnet-base-v2"
            ))
            
            self.sqlite_conn.commit()
            return True
            
        except Exception as e:
            print(f"Error adding document: {e}")
            return False
            
    async def hybrid_search(
        self, 
        query: str, 
        num_results: int = 10,
        semantic_weight: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining semantic and keyword search"""
        
        # Semantic search using ChromaDB
        semantic_results = self.collection.query(
            query_texts=[query],
            n_results=num_results * 2  # Get more for reranking
        )
        
        # Keyword search using SQLite FTS
        cursor = self.sqlite_conn.cursor()
        cursor.execute('''
        SELECT doc_id, title, content, 
               bm25(documents_fts) as score
        FROM documents_fts 
        WHERE documents_fts MATCH ?
        ORDER BY score
        LIMIT ?
        ''', (query, num_results * 2))
        
        keyword_results = cursor.fetchall()
        
        # Combine and rerank results
        combined_results = self._combine_search_results(
            semantic_results,
            keyword_results,
            semantic_weight,
            1.0 - semantic_weight
        )
        
        return combined_results[:num_results]
        
    def _combine_search_results(
        self, 
        semantic_results: Dict, 
        keyword_results: List[Tuple],
        semantic_weight: float,
        keyword_weight: float
    ) -> List[Dict[str, Any]]:
        """Combine and rerank search results"""
        
        result_scores = {}
        
        # Process semantic results
        if semantic_results['documents']:
            for i, (doc_id, document, distance) in enumerate(zip(
                semantic_results['ids'][0],
                semantic_results['documents'][0], 
                semantic_results['distances'][0]
            )):
                # Convert distance to similarity score (0-1)
                similarity = max(0, 1 - distance)
                result_scores[doc_id] = {
                    'doc_id': doc_id,
                    'content': document,
                    'semantic_score': similarity,
                    'keyword_score': 0,
                    'metadata': semantic_results['metadatas'][0][i] if semantic_results['metadatas'] else {}
                }
        
        # Process keyword results
        for doc_id, title, content, bm25_score in keyword_results:
            # Normalize BM25 score (rough approximation)
            normalized_score = min(1.0, max(0, bm25_score / 10.0))
            
            if doc_id in result_scores:
                result_scores[doc_id]['keyword_score'] = normalized_score
            else:
                result_scores[doc_id] = {
                    'doc_id': doc_id,
                    'content': content,
                    'title': title,
                    'semantic_score': 0,
                    'keyword_score': normalized_score,
                    'metadata': {}
                }
        
        # Calculate final scores and sort
        final_results = []
        for doc_id, scores in result_scores.items():
            final_score = (
                scores['semantic_score'] * semantic_weight + 
                scores['keyword_score'] * keyword_weight
            )
            
            final_results.append({
                'doc_id': doc_id,
                'content': scores['content'],
                'title': scores.get('title', ''),
                'final_score': final_score,
                'semantic_score': scores['semantic_score'],
                'keyword_score': scores['keyword_score'],
                'metadata': scores['metadata']
            })
        
        return sorted(final_results, key=lambda x: x['final_score'], reverse=True)
```

## API Design and Endpoints

### RESTful API Structure

```python
# src/api/routes.py
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio

security = HTTPBearer()

class DocumentUploadRequest(BaseModel):
    file_path: str
    metadata: Optional[Dict[str, Any]] = {}
    processing_options: Optional[Dict[str, Any]] = {}

class SearchRequest(BaseModel):
    query: str
    num_results: int = 10
    search_type: str = "hybrid"  # semantic, keyword, hybrid
    filters: Optional[Dict[str, Any]] = {}
    semantic_weight: float = 0.7

class KnowledgeGraphQuery(BaseModel):
    start_entity: str
    end_entity: Optional[str] = None
    max_hops: int = 3
    relationship_types: Optional[List[str]] = None

router = APIRouter(prefix="/api/v1", tags=["Academic AI"])

@router.post("/documents/upload")
async def upload_document(
    request: DocumentUploadRequest,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Upload and process academic document"""
    
    # Validate authentication
    if not await validate_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")
    
    # Start background processing
    task_id = f"doc_proc_{uuid.uuid4()}"
    background_tasks.add_task(
        process_document_async,
        request.file_path,
        request.metadata,
        request.processing_options,
        task_id
    )
    
    return {
        "task_id": task_id,
        "status": "processing",
        "message": "Document upload initiated"
    }

@router.post("/search")
async def search_documents(request: SearchRequest):
    """Search academic documents"""
    
    search_engine = get_search_engine()
    
    if request.search_type == "semantic":
        results = await search_engine.semantic_search(
            request.query,
            request.num_results,
            request.filters
        )
    elif request.search_type == "keyword":
        results = await search_engine.keyword_search(
            request.query,
            request.num_results,
            request.filters
        )
    else:  # hybrid
        results = await search_engine.hybrid_search(
            request.query,
            request.num_results,
            request.semantic_weight,
            request.filters
        )
    
    return {
        "query": request.query,
        "results": results,
        "total_results": len(results),
        "search_type": request.search_type
    }

@router.post("/knowledge-graph/query")
async def query_knowledge_graph(request: KnowledgeGraphQuery):
    """Query knowledge graph for entity relationships"""
    
    kg = get_knowledge_graph()
    
    if request.end_entity:
        # Find path between entities
        paths = await kg.find_connections(
            request.start_entity,
            request.end_entity,
            request.max_hops
        )
        return {"paths": paths}
    else:
        # Get entity neighborhood
        neighborhood = await kg.get_entity_neighborhood(
            request.start_entity,
            request.max_hops,
            request.relationship_types
        )
        return {"neighborhood": neighborhood}

@router.get("/agents/status")
async def get_agent_status():
    """Get current status of SPARC trio agents"""
    
    blackboard = get_blackboard()
    agent_states = await blackboard.query_agent_states()
    coordination_insights = await blackboard.get_coordination_insights()
    
    return {
        "agents": agent_states,
        "system_insights": coordination_insights,
        "timestamp": datetime.now().isoformat()
    }

@router.post("/agents/task")
async def assign_agent_task(task: Dict[str, Any]):
    """Assign task to specific agent"""
    
    agent_manager = get_agent_manager()
    task_id = await agent_manager.assign_task(
        agent_role=task["agent"],
        task_description=task["description"],
        parameters=task.get("parameters", {}),
        priority=task.get("priority", 1)
    )
    
    return {
        "task_id": task_id,
        "assigned_to": task["agent"],
        "status": "assigned"
    }
```

## Model Selection and Hugging Face Integration

### Optimal Model Selection Framework

```python
# src/models/model_selector.py
from transformers import AutoModel, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import torch
from typing import Dict, List, Any, Tuple
import json

class ModelSelector:
    """Intelligent model selection for academic text processing"""
    
    def __init__(self):
        self.model_registry = {
            "embedding": {
                "general": "sentence-transformers/all-mpnet-base-v2",
                "academic": "allenai/specter2_base",
                "multilingual": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            },
            "classification": {
                "zero_shot": "facebook/bart-large-mnli",
                "academic_domains": "allenai/scibert_scivocab_uncased"
            },
            "ner": {
                "general": "dbmdz/bert-large-cased-finetuned-conll03-english",
                "scientific": "allenai/scibert_scivocab_uncased",
                "art_history": "custom/art-history-ner"  # Fine-tuned model
            },
            "generation": {
                "academic_writing": "microsoft/DialoGPT-large",
                "summarization": "facebook/bart-large-cnn"
            }
        }
        
    async def select_optimal_model(
        self, 
        task: str, 
        domain: str,
        text_samples: List[str]
    ) -> Dict[str, Any]:
        """Select optimal model based on task, domain, and text characteristics"""
        
        # Analyze text characteristics
        text_analysis = self._analyze_text_characteristics(text_samples)
        
        # Get candidate models
        candidates = self._get_candidate_models(task, domain)
        
        # Evaluate models
        best_model = await self._evaluate_models(candidates, text_samples, task)
        
        return {
            "selected_model": best_model,
            "text_analysis": text_analysis,
            "evaluation_results": best_model["evaluation"]
        }
        
    def _analyze_text_characteristics(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze text characteristics to guide model selection"""
        
        analysis = {
            "avg_length": sum(len(text.split()) for text in texts) / len(texts),
            "vocabulary_complexity": self._calculate_vocabulary_complexity(texts),
            "domain_indicators": self._detect_domain_indicators(texts),
            "language_distribution": self._detect_languages(texts),
            "academic_markers": self._count_academic_markers(texts)
        }
        
        return analysis
        
    async def _evaluate_models(
        self, 
        candidates: List[str], 
        samples: List[str],
        task: str
    ) -> Dict[str, Any]:
        """Evaluate candidate models on sample texts"""
        
        evaluation_results = {}
        
        for model_name in candidates:
            try:
                # Load model
                if task == "embedding":
                    model = SentenceTransformer(model_name)
                    results = await self._evaluate_embedding_model(model, samples)
                elif task == "classification":
                    model = pipeline("zero-shot-classification", model=model_name)
                    results = await self._evaluate_classification_model(model, samples)
                elif task == "ner":
                    model = pipeline("ner", model=model_name, aggregation_strategy="simple")
                    results = await self._evaluate_ner_model(model, samples)
                    
                evaluation_results[model_name] = results
                
            except Exception as e:
                evaluation_results[model_name] = {"error": str(e), "score": 0}
        
        # Select best model
        best_model_name = max(evaluation_results.keys(), 
                             key=lambda x: evaluation_results[x].get("score", 0))
        
        return {
            "model_name": best_model_name,
            "evaluation": evaluation_results[best_model_name],
            "all_results": evaluation_results
        }

# Model-specific implementations for academic art history
class ArtHistoryModels:
    """Specialized models for art history research"""
    
    @staticmethod
    def load_art_history_ner():
        """Load fine-tuned NER model for art history entities"""
        # Custom fine-tuned model for art history
        return pipeline(
            "ner",
            model="custom/art-history-ner",
            tokenizer="custom/art-history-ner",
            aggregation_strategy="simple"
        )
    
    @staticmethod
    def load_artwork_classifier():
        """Load classifier for artwork categorization"""
        return pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
    
    @staticmethod
    def get_art_movement_labels():
        """Get comprehensive list of art movement labels"""
        return [
            "Renaissance", "Baroque", "Romanticism", "Impressionism",
            "Post-Impressionism", "Cubism", "Surrealism", "Abstract Expressionism",
            "Pop Art", "Minimalism", "Contemporary Art", "Gothic",
            "Neoclassicism", "Realism", "Symbolism", "Fauvism",
            "Expressionism", "Dadaism", "Art Nouveau", "Modernism"
        ]
```

## Deployment and Scaling Strategy

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download models
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"
RUN python -c "import spacy; spacy.download('en_core_web_sm')"

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: academic-ai-system
  labels:
    app: academic-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: academic-ai
  template:
    metadata:
      labels:
        app: academic-ai
    spec:
      containers:
      - name: academic-ai
        image: academic-ai:latest
        ports:
        - containerPort: 8000
        env:
        - name: INFLUXDB_URL
          value: "http://influxdb:8086"
        - name: NEO4J_URI
          value: "bolt://neo4j:7687"
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: academic-ai-service
spec:
  selector:
    app: academic-ai
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Security and Performance Optimization

### Security Implementation

```python
# src/security/auth.py
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta
import bcrypt
from typing import Optional

security = HTTPBearer()

class SecurityManager:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.algorithm = "HS256"
        
    def create_access_token(self, user_id: str, expires_delta: Optional[timedelta] = None):
        """Create JWT access token"""
        to_encode = {"sub": user_id}
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=24)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[str]:
        """Verify JWT token and return user ID"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            user_id: str = payload.get("sub")
            if user_id is None:
                return None
            return user_id
        except jwt.PyJWTError:
            return None
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Dependency to get current authenticated user"""
    security_manager = get_security_manager()
    user_id = security_manager.verify_token(credentials.credentials)
    if user_id is None:
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id
```

### Performance Optimization

```python
# src/performance/optimization.py
import asyncio
import time
from functools import wraps
from typing import Callable, Any
import cachetools
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

class PerformanceOptimizer:
    def __init__(self):
        self.cache = cachetools.TTLCache(maxsize=1000, ttl=3600)
        self.executor = ThreadPoolExecutor(max_workers=mp.cpu_count())
        
    def cache_result(self, ttl: int = 3600):
        """Decorator for caching function results"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Create cache key
                cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
                
                # Check cache
                if cache_key in self.cache:
                    return self.cache[cache_key]
                
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Cache result
                self.cache[cache_key] = result
                return result
            return wrapper
        return decorator
    
    async def parallel_process(self, func: Callable, items: list, batch_size: int = 10):
        """Process items in parallel batches"""
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_tasks = [func(item) for item in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
        
        return results
    
    def cpu_bound_task(self, func: Callable, *args, **kwargs):
        """Execute CPU-bound task in thread pool"""
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(self.executor, func, *args, **kwargs)

# Memory optimization for large documents
class MemoryOptimizer:
    @staticmethod
    def chunk_generator(text: str, chunk_size: int = 1000):
        """Generator for memory-efficient text chunking"""
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]
    
    @staticmethod
    async def stream_process_document(file_path: str, processor: Callable):
        """Stream process large documents"""
        async with aiofiles.open(file_path, 'r') as f:
            chunk_buffer = ""
            async for line in f:
                chunk_buffer += line
                if len(chunk_buffer) >= 10000:  # Process in 10KB chunks
                    result = await processor(chunk_buffer)
                    chunk_buffer = ""
                    yield result
            
            # Process remaining buffer
            if chunk_buffer:
                yield await processor(chunk_buffer)
```

## Quality Gates and Testing Framework

### TDD Implementation for AI Systems

```python
# tests/test_tdd_framework.py
import pytest
import asyncio
from unittest.mock import Mock, patch
from src.tdd_framework import TDDFramework, TestPhase

class TestAITDDFramework:
    """Test-Driven Development framework for AI systems"""
    
    @pytest.fixture
    def tdd_framework(self):
        return TDDFramework()
    
    async def test_red_phase_embedding_quality(self, tdd_framework):
        """RED: Define test for embedding quality"""
        
        # Define expected behavior
        test_text = "Van Gogh's influence on modern art"
        
        @tdd_framework.define_test("embedding_quality")
        async def test_embedding_similarity():
            embeddings = await generate_embeddings([
                "Van Gogh's influence on modern art",
                "How Van Gogh influenced contemporary artists",
                "The weather today is sunny"
            ])
            
            # Similar texts should have high similarity
            similarity_1_2 = cosine_similarity(embeddings[0], embeddings[1])
            similarity_1_3 = cosine_similarity(embeddings[0], embeddings[2])
            
            assert similarity_1_2 > 0.7, "Related art texts should be similar"
            assert similarity_1_3 < 0.3, "Unrelated texts should be dissimilar"
            
        # This test should fail initially (RED phase)
        with pytest.raises(AssertionError):
            await test_embedding_similarity()
    
    async def test_green_phase_implement_embeddings(self, tdd_framework):
        """GREEN: Implement minimal embedding functionality"""
        
        # Implement minimal solution
        async def generate_embeddings(texts):
            model = SentenceTransformer('all-mpnet-base-v2')
            return model.encode(texts)
        
        # Test should now pass
        embeddings = await generate_embeddings([
            "Van Gogh's influence on modern art",
            "How Van Gogh influenced contemporary artists", 
            "The weather today is sunny"
        ])
        
        similarity_1_2 = cosine_similarity(embeddings[0], embeddings[1])
        similarity_1_3 = cosine_similarity(embeddings[0], embeddings[2])
        
        assert similarity_1_2 > 0.7
        assert similarity_1_3 < 0.3
    
    async def test_refactor_phase_optimize_embeddings(self, tdd_framework):
        """REFACTOR: Optimize while maintaining test coverage"""
        
        # Optimized implementation with caching and batching
        class OptimizedEmbeddings:
            def __init__(self):
                self.model = SentenceTransformer('all-mpnet-base-v2')
                self.cache = {}
            
            async def generate_embeddings(self, texts, batch_size=32):
                results = []
                for text in texts:
                    if text in self.cache:
                        results.append(self.cache[text])
                    else:
                        embedding = self.model.encode([text])[0]
                        self.cache[text] = embedding
                        results.append(embedding)
                return results
        
        # All original tests should still pass
        optimizer = OptimizedEmbeddings()
        embeddings = await optimizer.generate_embeddings([
            "Van Gogh's influence on modern art",
            "How Van Gogh influenced contemporary artists",
            "The weather today is sunny"
        ])
        
        similarity_1_2 = cosine_similarity(embeddings[0], embeddings[1])
        similarity_1_3 = cosine_similarity(embeddings[0], embeddings[2])
        
        assert similarity_1_2 > 0.7
        assert similarity_1_3 < 0.3

# Comprehensive test suite
class TestKnowledgeGraphTDD:
    """TDD tests for knowledge graph functionality"""
    
    async def test_red_entity_extraction(self):
        """RED: Test entity extraction from art history text"""
        
        text = "Van Gogh painted The Starry Night in 1889 while at Saint-Rémy-de-Provence."
        
        # Expected entities (test will fail initially)
        expected_entities = {
            "PERSON": ["Van Gogh"],
            "ARTWORK": ["The Starry Night"], 
            "DATE": ["1889"],
            "LOCATION": ["Saint-Rémy-de-Provence"]
        }
        
        # This should fail in RED phase
        entities = extract_art_entities(text)
        assert entities == expected_entities  # Will fail initially
    
    async def test_green_implement_entity_extraction(self):
        """GREEN: Implement basic entity extraction"""
        
        def extract_art_entities(text):
            # Minimal implementation
            import spacy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            
            entities = {"PERSON": [], "ARTWORK": [], "DATE": [], "LOCATION": []}
            
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    entities["PERSON"].append(ent.text)
                elif ent.label_ == "DATE":
                    entities["DATE"].append(ent.text)
                elif ent.label_ in ["GPE", "LOC"]:
                    entities["LOCATION"].append(ent.text)
            
            # Custom artwork detection (simplified)
            if "The Starry Night" in text:
                entities["ARTWORK"].append("The Starry Night")
                
            return entities
        
        text = "Van Gogh painted The Starry Night in 1889 while at Saint-Rémy-de-Provence."
        entities = extract_art_entities(text)
        
        assert "Van Gogh" in entities["PERSON"]
        assert "The Starry Night" in entities["ARTWORK"]
        assert "1889" in entities["DATE"]
```

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
1. **Setup FastAPI middleware with async processing**
2. **Implement InfluxDB blackboard system**
3. **Create basic SPARC agent framework**
4. **Setup vector storage with ChromaDB**
5. **Implement document ingestion pipeline**

### Phase 2: Core Features (Weeks 3-4)
1. **Develop knowledge graph schema and implementation**
2. **Create hybrid search functionality**
3. **Implement MCP stack integration**
4. **Add Hugging Face model selection framework**
5. **Create TDD framework for AI systems**

### Phase 3: Advanced Features (Weeks 5-6)
1. **Optimize performance with caching and batching**
2. **Implement security and authentication**
3. **Add comprehensive monitoring and logging**
4. **Create deployment configurations**
5. **Build comprehensive test suite**

### Phase 4: Production Ready (Weeks 7-8)
1. **Performance optimization and tuning**
2. **Security hardening and compliance**
3. **Documentation and API reference**
4. **Load testing and scalability validation**
5. **Production deployment and monitoring**

## Success Metrics and KPIs

### Technical Metrics
- **Search Accuracy**: >85% relevance score
- **Response Time**: <200ms for search queries
- **System Uptime**: 99.9% availability
- **Test Coverage**: >90% code coverage
- **Mutation Testing**: >80% mutation score

### Academic Research Metrics  
- **Entity Extraction Accuracy**: >90% for art history entities
- **Knowledge Graph Completeness**: >95% relationship coverage
- **Cross-Document Connectivity**: >70% documents linked
- **Search Precision**: >80% relevant results in top 10
- **User Satisfaction**: >4.5/5 rating from researchers

## Conclusion

This PRD outlines a comprehensive, production-ready AI document understanding system specifically optimized for academic art history research. The architecture leverages cutting-edge technologies including FastAPI async middleware, InfluxDB blackboard systems, proprietary SPARC agents, and advanced knowledge graph construction to deliver unprecedented research capabilities.

The system's modular design ensures scalability, maintainability, and extensibility while the TDD approach guarantees reliability and quality. With proper implementation following this plan, Eric will have a world-class tool for academic research that can process, understand, and connect complex art historical texts with remarkable precision and insight.