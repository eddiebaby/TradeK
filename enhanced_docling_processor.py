#!/usr/bin/env python3
"""
Enhanced Docling Book Processor for TradeKnowledge
==================================================

Complete integration of Docling with hybrid fallback approach.
Features:
- Docling primary processing with AI-native capabilities
- PyPDF2/pdfplumber fallback for reliability
- Dual database storage (ChromaDB + SQLite)
- Memory management and streaming
- Comprehensive error handling
- Agent trio workflow integration

Author: Claude Code Assistant with Agent Trio
"""

import os
import sys
import json
import time
import hashlib
import psutil
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import gc
import logging

# Add project root to path
sys.path.append('.')

# Core imports
try:
    import PyPDF2
    import pdfplumber
    import chromadb
    from chromadb.config import Settings
except ImportError as e:
    print(f"❌ Missing basic dependencies: {e}")
    sys.exit(1)

# Docling imports with fallback
DOCLING_AVAILABLE = False
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    DOCLING_AVAILABLE = True
    print("✅ Docling available - enhanced processing enabled")
except ImportError as e:
    print(f"⚠️ Docling not available, using fallback mode: {e}")
    DocumentConverter = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Enhanced memory monitoring with warnings and cleanup"""
    
    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_mb = max_memory_mb
        self.process = psutil.Process(os.getpid())
        self.warning_threshold = max_memory_mb * 0.8  # 80% warning
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def is_memory_ok(self) -> bool:
        """Check if memory usage is within limits"""
        return self.get_memory_usage() < self.max_memory_mb
    
    def check_and_warn(self) -> bool:
        """Check memory and warn if approaching limits"""
        usage = self.get_memory_usage()
        if usage > self.warning_threshold:
            print(f"⚠️ Memory usage {usage:.1f}MB approaching limit {self.max_memory_mb}MB")
            gc.collect()  # Force cleanup
            return False
        return True
    
    def force_cleanup(self):
        """Force garbage collection and memory cleanup"""
        gc.collect()
        time.sleep(0.1)  # Brief pause for cleanup

class DoclingProcessor:
    """Advanced Docling document processor"""
    
    def __init__(self):
        self.converter = None
        if DOCLING_AVAILABLE:
            try:
                # Initialize with basic configuration (avoid API issues)
                self.converter = DocumentConverter()
                logger.info("Docling converter initialized with enhanced options")
            except Exception as e:
                logger.error(f"Failed to initialize Docling: {e}")
                self.converter = None
    
    def is_available(self) -> bool:
        """Check if Docling is available and initialized"""
        return self.converter is not None
    
    def process_document(self, file_path: Path) -> Dict[str, Any]:
        """Process document with Docling and return structured results"""
        if not self.is_available():
            raise RuntimeError("Docling not available")
        
        try:
            logger.info(f"Processing with Docling: {file_path.name}")
            start_time = time.time()
            
            # Convert document
            result = self.converter.convert(file_path)
            doc = result.document
            
            processing_time = time.time() - start_time
            
            # Extract content in multiple formats
            markdown_content = doc.export_to_markdown()
            
            try:
                json_content = doc.export_to_json()
                json_available = True
            except Exception:
                json_content = None
                json_available = False
            
            try:
                html_content = doc.export_to_html()
                html_available = True
            except Exception:
                html_content = None
                html_available = False
            
            # Analyze document structure
            page_count = getattr(doc, 'page_count', 0) or len(getattr(doc, 'pages', []))
            
            # Extract metadata
            metadata = {
                'processor': 'docling',
                'processing_time': processing_time,
                'has_tables': 'table' in markdown_content.lower() or '|' in markdown_content,
                'has_formulas': '$' in markdown_content or '\\(' in markdown_content,
                'has_code': '```' in markdown_content,
                'formats_available': {
                    'markdown': True,
                    'json': json_available,
                    'html': html_available
                },
                'page_count': page_count,
                'content_length': len(markdown_content)
            }
            
            return {
                'success': True,
                'content': {
                    'markdown': markdown_content,
                    'json': json_content,
                    'html': html_content
                },
                'metadata': metadata,
                'pages': self._extract_pages_from_markdown(markdown_content)
            }
            
        except Exception as e:
            logger.error(f"Docling processing failed: {e}")
            raise

    def _extract_pages_from_markdown(self, markdown: str) -> List[Dict[str, Any]]:
        """Extract page-like sections from markdown content"""
        # Simple page extraction - can be enhanced based on Docling's structure
        sections = markdown.split('\n\n')
        pages = []
        
        current_page = 1
        current_content = ""
        
        for section in sections:
            if len(current_content) > 2000:  # Approximate page break
                pages.append({
                    'page_number': current_page,
                    'text': current_content.strip(),
                    'word_count': len(current_content.split()),
                    'char_count': len(current_content)
                })
                current_page += 1
                current_content = section
            else:
                current_content += "\n\n" + section
        
        # Add final page
        if current_content.strip():
            pages.append({
                'page_number': current_page,
                'text': current_content.strip(),
                'word_count': len(current_content.split()),
                'char_count': len(current_content)
            })
        
        return pages

class FallbackProcessor:
    """Fallback processor using PyPDF2 and pdfplumber"""
    
    def __init__(self):
        self.batch_size = 50
    
    def process_document(self, file_path: Path) -> Dict[str, Any]:
        """Process document with fallback methods"""
        logger.info(f"Processing with fallback methods: {file_path.name}")
        start_time = time.time()
        
        try:
            # Try pdfplumber first (better for complex layouts)
            pages = self._process_with_pdfplumber(file_path)
            processor_used = 'pdfplumber'
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}, trying PyPDF2")
            try:
                pages = self._process_with_pypdf2(file_path)
                processor_used = 'pypdf2'
            except Exception as e2:
                logger.error(f"Both fallback methods failed: {e2}")
                raise
        
        processing_time = time.time() - start_time
        
        # Combine all text for markdown
        full_text = "\n\n".join(page['text'] for page in pages if page['text'])
        
        metadata = {
            'processor': processor_used,
            'processing_time': processing_time,
            'has_tables': False,  # Basic detection
            'has_formulas': '$' in full_text,
            'has_code': 'def ' in full_text or 'class ' in full_text,
            'formats_available': {
                'markdown': True,
                'json': False,
                'html': False
            },
            'page_count': len(pages),
            'content_length': len(full_text)
        }
        
        return {
            'success': True,
            'content': {
                'markdown': full_text,
                'json': None,
                'html': None
            },
            'metadata': metadata,
            'pages': pages
        }
    
    def _process_with_pdfplumber(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process with pdfplumber"""
        pages = []
        
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                
                # Try table extraction for sparse text
                if len(text.strip()) < 100:
                    try:
                        tables = page.extract_tables()
                        if tables:
                            for table in tables:
                                table_text = self._table_to_text(table)
                                text += f"\n\n[TABLE]\n{table_text}\n[/TABLE]\n"
                    except Exception:
                        pass
                
                pages.append({
                    'page_number': i + 1,
                    'text': text.strip(),
                    'word_count': len(text.split()),
                    'char_count': len(text)
                })
        
        return pages
    
    def _process_with_pypdf2(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process with PyPDF2"""
        pages = []
        
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                
                pages.append({
                    'page_number': i + 1,
                    'text': text.strip(),
                    'word_count': len(text.split()),
                    'char_count': len(text)
                })
        
        return pages
    
    def _table_to_text(self, table: List[List[Any]]) -> str:
        """Convert table to text representation"""
        if not table:
            return ''
        
        lines = []
        for row in table:
            cleaned_row = [str(cell) if cell is not None else '' for cell in row]
            lines.append(' | '.join(cleaned_row))
        
        return '\n'.join(lines)

class SQLiteManager:
    """Enhanced SQLite database manager"""
    
    def __init__(self, db_path: str = "./data/knowledge.db"):
        self.db_path = db_path
        self._ensure_database_exists()
    
    def _ensure_database_exists(self):
        """Ensure database and tables exist"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Check if table exists and add missing columns
            cursor = conn.execute("PRAGMA table_info(books)")
            existing_columns = [row[1] for row in cursor.fetchall()]
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS books (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    author TEXT,
                    file_path TEXT,
                    file_hash TEXT UNIQUE,
                    total_pages INTEGER,
                    total_chunks INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    indexed_at TIMESTAMP,
                    metadata TEXT
                )
            ''')
            
            # Add missing columns if they don't exist
            if 'processor_used' not in existing_columns:
                conn.execute('ALTER TABLE books ADD COLUMN processor_used TEXT')
            if 'processing_time' not in existing_columns:
                conn.execute('ALTER TABLE books ADD COLUMN processing_time REAL')
            if 'file_type' not in existing_columns:
                conn.execute('ALTER TABLE books ADD COLUMN file_type TEXT DEFAULT "PDF"')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    book_id TEXT,
                    chunk_index INTEGER,
                    text TEXT,
                    chunk_type TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (book_id) REFERENCES books (id)
                )
            ''')
            
            # Create indexes for performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_chunks_book_id ON chunks(book_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_books_hash ON books(file_hash)')
    
    def save_book(self, book_data: Dict[str, Any]) -> bool:
        """Save book metadata to SQLite"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO books 
                    (id, title, author, file_path, file_hash, total_pages, total_chunks,
                     processor_used, processing_time, indexed_at, metadata, file_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    book_data['id'],
                    book_data['title'],
                    book_data['author'],
                    book_data['file_path'],
                    book_data['file_hash'],
                    book_data['total_pages'],
                    book_data['total_chunks'],
                    book_data['processor_used'],
                    book_data['processing_time'],
                    datetime.now(),
                    json.dumps(book_data.get('metadata', {})),
                    'PDF'
                ))
            
            logger.info(f"Book saved to SQLite: {book_data['title']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save book to SQLite: {e}")
            return False
    
    def save_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """Save chunks to SQLite in batch"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                chunk_data = [
                    (
                        chunk['id'],
                        chunk['book_id'],
                        chunk['chunk_index'],
                        chunk['text'],
                        chunk.get('chunk_type', 'text'),
                        json.dumps(chunk.get('metadata', {}))
                    )
                    for chunk in chunks
                ]
                
                conn.executemany('''
                    INSERT OR REPLACE INTO chunks 
                    (id, book_id, chunk_index, text, chunk_type, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', chunk_data)
            
            logger.info(f"Saved {len(chunks)} chunks to SQLite")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save chunks to SQLite: {e}")
            return False
    
    def book_exists(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Check if book already exists by hash"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    'SELECT * FROM books WHERE file_hash = ?', 
                    (file_hash,)
                )
                row = cursor.fetchone()
                
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    return dict(zip(columns, row))
                return None
                
        except Exception as e:
            logger.error(f"Error checking book existence: {e}")
            return None

class ChromaDBManager:
    """Enhanced ChromaDB manager"""
    
    def __init__(self, db_path: str = "./data/chromadb"):
        self.db_path = db_path
        self.client = None
        self.collection = None
        self._initialize()
    
    def _initialize(self):
        """Initialize ChromaDB client and collection"""
        try:
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(anonymized_telemetry=False, allow_reset=False)
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name='trading_books')
            except:
                self.collection = self.client.create_collection(name='trading_books')
                logger.info("Created trading_books collection in ChromaDB")
            
            logger.info("ChromaDB initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def save_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """Save chunks to ChromaDB"""
        try:
            if not chunks:
                return True
            
            # Prepare data for ChromaDB
            ids = [chunk['id'] for chunk in chunks]
            documents = [chunk['text'] for chunk in chunks]
            metadatas = []
            
            for chunk in chunks:
                metadata = {
                    'book_id': chunk['book_id'],
                    'chunk_index': chunk['chunk_index'],
                    'content_type': chunk.get('chunk_type', 'text'),
                    **chunk.get('metadata', {})
                }
                # ChromaDB metadata values must be strings, numbers, or booleans
                clean_metadata = {}
                for k, v in metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        clean_metadata[k] = v
                    else:
                        clean_metadata[k] = str(v)
                
                metadatas.append(clean_metadata)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Saved {len(chunks)} chunks to ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save chunks to ChromaDB: {e}")
            return False
    
    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Search ChromaDB collection"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            return results
        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            return {'ids': [], 'documents': [], 'metadatas': [], 'distances': []}

class SmartTextChunker:
    """Enhanced text chunker with format awareness"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_content(self, content: Dict[str, Any], book_id: str) -> List[Dict[str, Any]]:
        """Chunk content based on available formats"""
        chunks = []
        
        # Use markdown as primary content
        markdown = content.get('markdown', '')
        if not markdown:
            return chunks
        
        # Smart chunking based on markdown structure
        if '# ' in markdown or '## ' in markdown:
            # Markdown has headers, chunk by sections
            chunks = self._chunk_by_sections(markdown, book_id)
        else:
            # Regular text chunking
            chunks = self._chunk_by_size(markdown, book_id)
        
        # Enhance chunks with format information
        json_available = content.get('json') is not None
        html_available = content.get('html') is not None
        
        for chunk in chunks:
            chunk['metadata'].update({
                'has_json': json_available,
                'has_html': html_available,
                'source_formats': ['markdown'] + 
                               (['json'] if json_available else []) +
                               (['html'] if html_available else [])
            })
        
        return chunks
    
    def _chunk_by_sections(self, text: str, book_id: str) -> List[Dict[str, Any]]:
        """Chunk text by markdown sections"""
        import re
        
        # Split by headers
        sections = re.split(r'\n(?=#{1,3} )', text)
        chunks = []
        chunk_index = 0
        
        for section in sections:
            if not section.strip():
                continue
            
            # If section is too large, split it further
            if len(section) > self.chunk_size * 1.5:
                sub_chunks = self._chunk_by_size(section, book_id, chunk_index)
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)
            else:
                chunk = {
                    'id': f"{book_id}_chunk_{chunk_index:04d}",
                    'book_id': book_id,
                    'chunk_index': chunk_index,
                    'text': section.strip(),
                    'chunk_type': 'text',
                    'metadata': {
                        'chunking_method': 'section',
                        'is_header_section': section.strip().startswith('#')
                    }
                }
                chunks.append(chunk)
                chunk_index += 1
        
        return chunks
    
    def _chunk_by_size(self, text: str, book_id: str, start_index: int = 0) -> List[Dict[str, Any]]:
        """Chunk text by size with overlap"""
        chunks = []
        start = 0
        chunk_index = start_index
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                for i in range(end, max(end - 100, start), -1):
                    if text[i:i+2] in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                        end = i + 1
                        break
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk = {
                    'id': f"{book_id}_chunk_{chunk_index:04d}",
                    'book_id': book_id,
                    'chunk_index': chunk_index,
                    'text': chunk_text,
                    'chunk_type': 'text',
                    'metadata': {
                        'chunking_method': 'size',
                        'start_position': start,
                        'end_position': end
                    }
                }
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start with overlap
            start = end - self.overlap
            if start >= len(text):
                break
        
        return chunks

class EnhancedDoclingProcessor:
    """Complete enhanced book processor with Docling integration"""
    
    def __init__(self, max_memory_mb: int = 1024, batch_size: int = 25):
        self.max_memory_mb = max_memory_mb
        self.batch_size = batch_size
        
        # Initialize components
        self.memory_monitor = MemoryMonitor(max_memory_mb)
        self.docling_processor = DoclingProcessor()
        self.fallback_processor = FallbackProcessor()
        self.sqlite_manager = SQLiteManager()
        self.chromadb_manager = ChromaDBManager()
        self.chunker = SmartTextChunker()
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'docling_successes': 0,
            'fallback_used': 0,
            'total_chunks': 0,
            'total_processing_time': 0
        }
    
    def process_book(self, pdf_path: str, title: str = None, author: str = None) -> Dict[str, Any]:
        """Main book processing method with enhanced capabilities"""
        start_time = time.time()
        
        print("🚀 ENHANCED DOCLING PROCESSOR")
        print("=" * 60)
        print(f"📁 File: {pdf_path}")
        print(f"💾 Memory limit: {self.max_memory_mb}MB")
        print(f"🎯 Docling available: {self.docling_processor.is_available()}")
        print(f"📦 Batch size: {self.batch_size}")
        print()
        
        try:
            file_path = Path(pdf_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {pdf_path}")
            
            # Calculate file hash
            print("🔍 Calculating file hash...")
            file_hash = self._calculate_file_hash(file_path)
            
            # Check if already processed
            existing_book = self.sqlite_manager.book_exists(file_hash)
            if existing_book and existing_book.get('total_chunks', 0) > 0:
                print(f"✅ Book already processed: {existing_book['title']} ({existing_book['total_chunks']} chunks)")
                return {
                    'success': True,
                    'already_processed': True,
                    'book_id': existing_book['id'],
                    'title': existing_book['title'],
                    'chunks': existing_book['total_chunks']
                }
            
            # Generate book ID and metadata
            book_id = self._generate_book_id(file_hash, title)
            book_title = title or self._extract_title_from_filename(file_path)
            book_author = author or "Unknown"
            
            print(f"📚 Book ID: {book_id}")
            print(f"📝 Title: {book_title}")
            print(f"✍️ Author: {book_author}")
            
            # Process document with hybrid approach
            print("\n📖 Processing document...")
            doc_result = self._process_with_hybrid_approach(file_path)
            
            if not doc_result['success']:
                raise RuntimeError(f"Document processing failed: {doc_result.get('error')}")
            
            # Create chunks
            print("✂️ Creating intelligent chunks...")
            chunks = self.chunker.chunk_content(doc_result['content'], book_id)
            print(f"✅ Created {len(chunks)} chunks")
            
            # Save to databases
            print("\n💾 Saving to databases...")
            
            # Prepare book data
            book_data = {
                'id': book_id,
                'title': book_title,
                'author': book_author,
                'file_path': str(file_path),
                'file_hash': file_hash,
                'total_pages': doc_result['metadata']['page_count'],
                'total_chunks': len(chunks),
                'processor_used': doc_result['metadata']['processor'],
                'processing_time': doc_result['metadata']['processing_time'],
                'metadata': {
                    **doc_result['metadata'],
                    'file_size_mb': round(file_path.stat().st_size / (1024*1024), 2),
                    'enhanced_processor': True,
                    'processing_date': datetime.now().isoformat()
                }
            }
            
            # Save to SQLite
            sqlite_success = self.sqlite_manager.save_book(book_data)
            if not sqlite_success:
                print("⚠️ Failed to save to SQLite")
            
            # Save chunks to both databases
            chromadb_success = self._save_chunks_to_chromadb(chunks)
            sqlite_chunks_success = self._save_chunks_to_sqlite(chunks)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats['total_processed'] += 1
            self.stats['total_chunks'] += len(chunks)
            self.stats['total_processing_time'] += processing_time
            
            if doc_result['metadata']['processor'] == 'docling':
                self.stats['docling_successes'] += 1
            else:
                self.stats['fallback_used'] += 1
            
            # Test search functionality
            print("\n🔍 Testing search functionality...")
            self._test_search(book_id, book_title)
            
            # Generate final report
            success = sqlite_success and chromadb_success
            result = {
                'success': success,
                'book_id': book_id,
                'title': book_title,
                'author': book_author,
                'pages_processed': doc_result['metadata']['page_count'],
                'chunks_created': len(chunks),
                'processor_used': doc_result['metadata']['processor'],
                'processing_time': processing_time,
                'memory_peak_mb': self.memory_monitor.get_memory_usage(),
                'databases': {
                    'sqlite': sqlite_success,
                    'chromadb': chromadb_success
                },
                'features_detected': {
                    'tables': doc_result['metadata']['has_tables'],
                    'formulas': doc_result['metadata']['has_formulas'],
                    'code': doc_result['metadata']['has_code']
                }
            }
            
            print("\n" + "=" * 60)
            print("📊 ENHANCED PROCESSING COMPLETE")
            print("=" * 60)
            print(f"✅ Success: {result['success']}")
            print(f"📚 Book: {result['title']}")
            print(f"📄 Pages: {result['pages_processed']}")
            print(f"💾 Chunks: {result['chunks_created']}")
            print(f"🔄 Processor: {result['processor_used']}")
            print(f"⏱️ Time: {result['processing_time']:.1f}s")
            print(f"🧠 Memory: {result['memory_peak_mb']:.1f}MB")
            print(f"🗄️ Databases: SQLite={result['databases']['sqlite']}, ChromaDB={result['databases']['chromadb']}")
            print(f"🎯 Features: Tables={result['features_detected']['tables']}, Formulas={result['features_detected']['formulas']}, Code={result['features_detected']['code']}")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            print(f"\n❌ CRITICAL ERROR: {error_msg}")
            
            return {
                'success': False,
                'error': error_msg,
                'processing_time': processing_time,
                'memory_usage': self.memory_monitor.get_memory_usage()
            }
    
    def _process_with_hybrid_approach(self, file_path: Path) -> Dict[str, Any]:
        """Process document with Docling first, fallback if needed"""
        
        # Try Docling first if available
        if self.docling_processor.is_available():
            try:
                print("🎯 Attempting Docling processing...")
                result = self.docling_processor.process_document(file_path)
                
                # Validate result quality
                if self._is_result_quality_good(result):
                    print("✅ Docling processing successful")
                    return result
                else:
                    print("⚠️ Docling result quality poor, falling back...")
                    
            except Exception as e:
                print(f"⚠️ Docling failed: {e}, using fallback...")
        
        # Use fallback processor
        print("🔄 Using fallback processor...")
        try:
            result = self.fallback_processor.process_document(file_path)
            print("✅ Fallback processing successful")
            return result
        except Exception as e:
            print(f"❌ Fallback also failed: {e}")
            return {
                'success': False,
                'error': f"All processors failed. Docling: {e}, Fallback: {e}"
            }
    
    def _is_result_quality_good(self, result: Dict[str, Any]) -> bool:
        """Assess if processing result quality is acceptable"""
        if not result.get('success'):
            return False
        
        content = result.get('content', {})
        markdown = content.get('markdown', '')
        
        # Basic quality checks
        if len(markdown) < 100:  # Too short
            return False
        
        # Check for reasonable word count
        word_count = len(markdown.split())
        if word_count < 50:  # Too few words
            return False
        
        # Check for extraction artifacts (too many repeated characters)
        if any(char * 10 in markdown for char in [' ', '\n', '.', '-']):
            return False
        
        return True
    
    def _save_chunks_to_chromadb(self, chunks: List[Dict[str, Any]]) -> bool:
        """Save chunks to ChromaDB in batches"""
        try:
            total_saved = 0
            for i in range(0, len(chunks), self.batch_size):
                if not self.memory_monitor.check_and_warn():
                    self.memory_monitor.force_cleanup()
                
                batch = chunks[i:i + self.batch_size]
                success = self.chromadb_manager.save_chunks(batch)
                
                if success:
                    total_saved += len(batch)
                    print(f"  📊 ChromaDB batch {i//self.batch_size + 1}: {total_saved}/{len(chunks)} chunks")
                else:
                    print(f"  ❌ ChromaDB batch {i//self.batch_size + 1} failed")
                    return False
                
                time.sleep(0.1)  # Brief pause
            
            return True
            
        except Exception as e:
            logger.error(f"ChromaDB batch save failed: {e}")
            return False
    
    def _save_chunks_to_sqlite(self, chunks: List[Dict[str, Any]]) -> bool:
        """Save chunks to SQLite in batches"""
        try:
            total_saved = 0
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i:i + self.batch_size]
                success = self.sqlite_manager.save_chunks(batch)
                
                if success:
                    total_saved += len(batch)
                    print(f"  🗄️ SQLite batch {i//self.batch_size + 1}: {total_saved}/{len(chunks)} chunks")
                else:
                    print(f"  ❌ SQLite batch {i//self.batch_size + 1} failed")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"SQLite batch save failed: {e}")
            return False
    
    def _test_search(self, book_id: str, title: str):
        """Test search functionality with the processed book"""
        try:
            # Test ChromaDB search
            query = f"machine learning {title.split()[0] if title else 'investing'}"
            results = self.chromadb_manager.search(query, n_results=3)
            
            if results['ids'] and results['ids'][0]:
                # Check if our book appears in results
                book_results = [
                    i for i, metadata in enumerate(results['metadatas'][0])
                    if metadata.get('book_id') == book_id
                ]
                
                print(f"🎯 Search test: '{query}'")
                print(f"   Found {len(book_results)} results from processed book")
                
                if book_results:
                    best_idx = book_results[0]
                    best_score = 1 - results['distances'][0][best_idx]
                    preview = results['documents'][0][best_idx][:100]
                    print(f"   Best match (score: {best_score:.3f}): {preview}...")
                else:
                    print("   ⚠️ Processed book not found in search results")
            else:
                print("   ❌ No search results returned")
                
        except Exception as e:
            print(f"   ⚠️ Search test failed: {e}")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _generate_book_id(self, file_hash: str, title: str = None) -> str:
        """Generate unique book ID"""
        if title:
            # Create ID from title
            title_part = ''.join(c for c in title.lower() if c.isalnum())[:20]
            return f"{title_part}_{file_hash[:8]}"
        else:
            return f"book_{file_hash[:8]}"
    
    def _extract_title_from_filename(self, file_path: Path) -> str:
        """Extract title from filename"""
        title = file_path.stem
        # Clean up the title
        title = title.replace('_', ' ').replace('-', ' ')
        title = ' '.join(word.capitalize() for word in title.split())
        return title
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        return {
            'processor_stats': self.stats,
            'docling_available': self.docling_processor.is_available(),
            'success_rate': (
                self.stats['docling_successes'] + self.stats['fallback_used']
            ) / max(1, self.stats['total_processed']),
            'docling_success_rate': (
                self.stats['docling_successes'] / max(1, self.stats['total_processed'])
            ),
            'avg_processing_time': (
                self.stats['total_processing_time'] / max(1, self.stats['total_processed'])
            ),
            'avg_chunks_per_book': (
                self.stats['total_chunks'] / max(1, self.stats['total_processed'])
            )
        }

def main():
    """Main entry point for enhanced processor"""
    if len(sys.argv) < 2:
        print("Usage: python enhanced_docling_processor.py <pdf_path> [title] [author]")
        print()
        print("Examples:")
        print("  python enhanced_docling_processor.py book.pdf")
        print("  python enhanced_docling_processor.py book.pdf 'My Book Title'")
        print("  python enhanced_docling_processor.py book.pdf 'My Book Title' 'Author Name'")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    title = sys.argv[2] if len(sys.argv) > 2 else None
    author = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Initialize processor
    processor = EnhancedDoclingProcessor(max_memory_mb=1024, batch_size=25)
    
    # Process book
    result = processor.process_book(pdf_path, title, author)
    
    # Show final stats
    print("\n📊 PROCESSING STATISTICS")
    print("=" * 30)
    stats = processor.get_processing_stats()
    print(f"Total processed: {stats['processor_stats']['total_processed']}")
    print(f"Docling successes: {stats['processor_stats']['docling_successes']}")
    print(f"Fallback used: {stats['processor_stats']['fallback_used']}")
    print(f"Success rate: {stats['success_rate']:.1%}")
    print(f"Docling success rate: {stats['docling_success_rate']:.1%}")
    print(f"Avg processing time: {stats['avg_processing_time']:.2f}s")
    print(f"Avg chunks per book: {stats['avg_chunks_per_book']:.0f}")
    
    # Exit with appropriate code
    sys.exit(0 if result['success'] else 1)

if __name__ == "__main__":
    main()