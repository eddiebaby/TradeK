"""
PDF Parser for TradeKnowledge
Handles PDF text extraction with metadata preservation
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import hashlib
import re

import pypdf2
import pdfplumber
from pypdf2 import PdfReader
from pdfplumber import PDF

from utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class ExtractedContent:
    """Container for extracted content from a document"""
    text: str
    metadata: Dict[str, Any]
    page_number: Optional[int] = None
    content_type: str = "text"  # text, code, formula, table
    section_info: Optional[Dict[str, Any]] = None

@dataclass
class BookMetadata:
    """Metadata extracted from a book"""
    title: str
    author: Optional[str] = None
    isbn: Optional[str] = None
    pages: int = 0
    file_type: str = "pdf"
    file_hash: str = ""
    creation_date: Optional[str] = None
    producer: Optional[str] = None
    subject: Optional[str] = None
    
class PDFParser:
    """
    Enhanced PDF parser supporting both clean and scanned PDFs
    """
    
    def __init__(self):
        self.code_patterns = [
            r'```[\s\S]*?```',  # Markdown code blocks
            r'def\s+\w+\s*\([^)]*\):',  # Python functions
            r'class\s+\w+\s*\([^)]*\):',  # Python classes
            r'import\s+\w+',  # Import statements
            r'from\s+\w+\s+import',  # From imports
            r'[\w_]+\s*=\s*.*',  # Variable assignments
        ]
        self.formula_patterns = [
            r'\$\$.*?\$\$',  # LaTeX display math
            r'\$.*?\$',      # LaTeX inline math
            r'\\begin\{equation\}.*?\\end\{equation\}',  # LaTeX equations
            r'\\begin\{align\}.*?\\end\{align\}',        # LaTeX align
        ]
        
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of the file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def extract_metadata(self, file_path: Path) -> BookMetadata:
        """Extract metadata from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                pdf_info = pdf_reader.metadata
                
                # Extract basic metadata
                title = pdf_info.get('/Title', file_path.stem) if pdf_info else file_path.stem
                author = pdf_info.get('/Author') if pdf_info else None
                creation_date = pdf_info.get('/CreationDate') if pdf_info else None
                producer = pdf_info.get('/Producer') if pdf_info else None
                subject = pdf_info.get('/Subject') if pdf_info else None
                
                # Clean up title if it's empty or just whitespace
                if not title or not title.strip():
                    title = file_path.stem
                
                # Try to extract ISBN from text if not in metadata
                isbn = self._extract_isbn_from_metadata(pdf_info) if pdf_info else None
                
                return BookMetadata(
                    title=title.strip(),
                    author=author.strip() if author else None,
                    isbn=isbn,
                    pages=len(pdf_reader.pages),
                    file_type="pdf",
                    file_hash=self.calculate_file_hash(file_path),
                    creation_date=str(creation_date) if creation_date else None,
                    producer=producer.strip() if producer else None,
                    subject=subject.strip() if subject else None
                )
                
        except Exception as e:
            logger.error(f"Error extracting metadata from {file_path}: {e}")
            return BookMetadata(
                title=file_path.stem,
                pages=0,
                file_type="pdf",
                file_hash=self.calculate_file_hash(file_path)
            )
    
    def _extract_isbn_from_metadata(self, pdf_info: Dict) -> Optional[str]:
        """Try to extract ISBN from PDF metadata"""
        isbn_pattern = r'(?:ISBN(?:-1[03])?:?\s*)?(?=[0-9X]{10}$|(?=(?:[0-9]+[- ]){3})[- 0-9X]{13}$|97[89][0-9]{10}$|(?=(?:[0-9]+[- ]){4})[- 0-9]{17}$)(?:97[89][- ]?)?[0-9]{1,5}[- ]?[0-9]+[- ]?[0-9]+[- ]?[0-9X]'
        
        for field in ['/Subject', '/Keywords', '/Title']:
            value = pdf_info.get(field, '')
            if isinstance(value, str):
                match = re.search(isbn_pattern, value)
                if match:
                    return match.group().replace('-', '').replace(' ', '')
        return None
    
    def extract_text_pdfplumber(self, file_path: Path) -> List[ExtractedContent]:
        """Extract text using pdfplumber (better for layout preservation)"""
        extracted_content = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        # Extract text with layout information
                        text = page.extract_text()
                        if not text or not text.strip():
                            continue
                            
                        # Detect content type
                        content_type = self._detect_content_type(text)
                        
                        # Extract tables if present
                        tables = page.extract_tables()
                        if tables:
                            for table in tables:
                                table_text = self._format_table(table)
                                if table_text:
                                    extracted_content.append(ExtractedContent(
                                        text=table_text,
                                        metadata={"page": page_num},
                                        page_number=page_num,
                                        content_type="table"
                                    ))
                        
                        # Add main text content
                        extracted_content.append(ExtractedContent(
                            text=text.strip(),
                            metadata={"page": page_num},
                            page_number=page_num,
                            content_type=content_type
                        ))
                        
                    except Exception as e:
                        logger.warning(f"Error extracting from page {page_num}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error processing PDF with pdfplumber: {e}")
            return []
            
        return extracted_content
    
    def extract_text_pypdf2(self, file_path: Path) -> List[ExtractedContent]:
        """Extract text using PyPDF2 (fallback method)"""
        extracted_content = []
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        text = page.extract_text()
                        if not text or not text.strip():
                            continue
                            
                        content_type = self._detect_content_type(text)
                        
                        extracted_content.append(ExtractedContent(
                            text=text.strip(),
                            metadata={"page": page_num},
                            page_number=page_num,
                            content_type=content_type
                        ))
                        
                    except Exception as e:
                        logger.warning(f"Error extracting from page {page_num}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error processing PDF with PyPDF2: {e}")
            return []
            
        return extracted_content
    
    def _detect_content_type(self, text: str) -> str:
        """Detect the type of content in text"""
        # Check for code patterns
        for pattern in self.code_patterns:
            if re.search(pattern, text, re.MULTILINE):
                return "code"
        
        # Check for mathematical formulas
        for pattern in self.formula_patterns:
            if re.search(pattern, text, re.MULTILINE):
                return "formula"
        
        # Check for table-like structure
        if self._looks_like_table(text):
            return "table"
            
        return "text"
    
    def _looks_like_table(self, text: str) -> bool:
        """Heuristic to detect table-like text"""
        lines = text.split('\n')
        if len(lines) < 3:
            return False
            
        # Look for consistent column separators
        separators = ['\t', '|', '  ']
        for sep in separators:
            if sum(line.count(sep) > 2 for line in lines) > len(lines) * 0.5:
                return True
                
        return False
    
    def _format_table(self, table: List[List[str]]) -> str:
        """Format extracted table as text"""
        if not table:
            return ""
            
        formatted_rows = []
        for row in table:
            if row:  # Skip empty rows
                cleaned_row = [cell.strip() if cell else "" for cell in row]
                formatted_rows.append(" | ".join(cleaned_row))
                
        return "\n".join(formatted_rows)
    
    def parse_pdf(self, file_path: Path, use_pdfplumber: bool = True) -> Tuple[BookMetadata, List[ExtractedContent]]:
        """
        Main method to parse PDF and extract content
        
        Args:
            file_path: Path to PDF file
            use_pdfplumber: Whether to use pdfplumber (default) or PyPDF2
            
        Returns:
            Tuple of (metadata, extracted_content_list)
        """
        logger.info(f"Parsing PDF: {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        # Extract metadata
        metadata = self.extract_metadata(file_path)
        
        # Extract content
        if use_pdfplumber:
            content = self.extract_text_pdfplumber(file_path)
            # Fallback to PyPDF2 if pdfplumber fails
            if not content:
                logger.warning("pdfplumber extraction failed, falling back to PyPDF2")
                content = self.extract_text_pypdf2(file_path)
        else:
            content = self.extract_text_pypdf2(file_path)
        
        logger.info(f"Extracted {len(content)} content blocks from {metadata.pages} pages")
        
        return metadata, content

def parse_pdf_file(file_path: str) -> Tuple[BookMetadata, List[ExtractedContent]]:
    """Convenience function to parse a PDF file"""
    parser = PDFParser()
    return parser.parse_pdf(Path(file_path))