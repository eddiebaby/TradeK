"""
EPUB Parser for TradeKnowledge
Handles EPUB text extraction with metadata preservation
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import re
from html import unescape

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

from utils.logging import get_logger
from .pdf_parser import ExtractedContent, BookMetadata  # Reuse data structures

logger = get_logger(__name__)

class EPUBParser:
    """
    EPUB parser for extracting text and metadata from EPUB files
    """
    
    def __init__(self):
        self.code_patterns = [
            r'<code>.*?</code>',
            r'<pre>.*?</pre>',
            r'```[\s\S]*?```',
            r'def\s+\w+\s*\([^)]*\):',
            r'class\s+\w+\s*\([^)]*\):',
            r'import\s+\w+',
            r'from\s+\w+\s+import',
        ]
        self.formula_patterns = [
            r'\$\$.*?\$\$',
            r'\$.*?\$',
            r'\\begin\{equation\}.*?\\end\{equation\}',
            r'\\begin\{align\}.*?\\end\{align\}',
            r'<math[^>]*>.*?</math>',  # MathML
        ]
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of the file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def extract_metadata(self, book: epub.EpubBook, file_path: Path) -> BookMetadata:
        """Extract metadata from EPUB book"""
        try:
            # Get title
            title = book.get_metadata('DC', 'title')
            title = title[0][0] if title else file_path.stem
            
            # Get author(s)
            authors = book.get_metadata('DC', 'creator')
            author = authors[0][0] if authors else None
            
            # Get ISBN
            identifiers = book.get_metadata('DC', 'identifier')
            isbn = None
            for identifier in identifiers:
                if 'isbn' in str(identifier).lower():
                    isbn = self._extract_isbn(str(identifier[0]))
                    break
            
            # Get subject/description
            subjects = book.get_metadata('DC', 'subject')
            subject = subjects[0][0] if subjects else None
            
            # Get publication date
            dates = book.get_metadata('DC', 'date')
            creation_date = dates[0][0] if dates else None
            
            # Get publisher
            publishers = book.get_metadata('DC', 'publisher')
            producer = publishers[0][0] if publishers else None
            
            # Count chapters/sections to estimate pages
            items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
            estimated_pages = max(1, len(items) * 5)  # Rough estimate
            
            return BookMetadata(
                title=title.strip() if title else file_path.stem,
                author=author.strip() if author else None,
                isbn=isbn,
                pages=estimated_pages,
                file_type="epub",
                file_hash=self.calculate_file_hash(file_path),
                creation_date=creation_date,
                producer=producer.strip() if producer else None,
                subject=subject.strip() if subject else None
            )
            
        except Exception as e:
            logger.error(f"Error extracting metadata from EPUB: {e}")
            return BookMetadata(
                title=file_path.stem,
                pages=1,
                file_type="epub",
                file_hash=self.calculate_file_hash(file_path)
            )
    
    def _extract_isbn(self, text: str) -> Optional[str]:
        """Extract ISBN from text"""
        isbn_pattern = r'(?:ISBN(?:-1[03])?:?\s*)?(?=[0-9X]{10}$|(?=(?:[0-9]+[- ]){3})[- 0-9X]{13}$|97[89][0-9]{10}$|(?=(?:[0-9]+[- ]){4})[- 0-9]{17}$)(?:97[89][- ]?)?[0-9]{1,5}[- ]?[0-9]+[- ]?[0-9]+[- ]?[0-9X]'
        match = re.search(isbn_pattern, text)
        if match:
            return match.group().replace('-', '').replace(' ', '')
        return None
    
    def extract_content(self, book: epub.EpubBook) -> List[ExtractedContent]:
        """Extract text content from EPUB chapters"""
        extracted_content = []
        
        # Get all document items (chapters, sections)
        items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
        
        for item_num, item in enumerate(items, 1):
            try:
                # Get raw HTML content
                content_html = item.get_content().decode('utf-8', errors='ignore')
                
                # Parse with BeautifulSoup
                soup = BeautifulSoup(content_html, 'html.parser')
                
                # Extract chapter title if available
                chapter_title = self._extract_chapter_title(soup, item.get_name())
                
                # Process different content types
                self._extract_from_soup(soup, extracted_content, item_num, chapter_title)
                
            except Exception as e:
                logger.warning(f"Error processing EPUB item {item.get_name()}: {e}")
                continue
        
        return extracted_content
    
    def _extract_chapter_title(self, soup: BeautifulSoup, item_name: str) -> str:
        """Extract chapter title from HTML"""
        # Try to find title in various header tags
        for tag in ['h1', 'h2', 'title']:
            title_elem = soup.find(tag)
            if title_elem and title_elem.get_text().strip():
                return title_elem.get_text().strip()
        
        # Fallback to item name
        return item_name.replace('.xhtml', '').replace('.html', '').replace('_', ' ')
    
    def _extract_from_soup(self, soup: BeautifulSoup, extracted_content: List[ExtractedContent], 
                          chapter_num: int, chapter_title: str):
        """Extract different types of content from BeautifulSoup object"""
        
        # Remove script and style tags
        for tag in soup(['script', 'style']):
            tag.decompose()
        
        # Process code blocks
        self._extract_code_blocks(soup, extracted_content, chapter_num, chapter_title)
        
        # Process tables
        self._extract_tables(soup, extracted_content, chapter_num, chapter_title)
        
        # Process mathematical content
        self._extract_math_content(soup, extracted_content, chapter_num, chapter_title)
        
        # Extract main text content
        self._extract_text_content(soup, extracted_content, chapter_num, chapter_title)
    
    def _extract_code_blocks(self, soup: BeautifulSoup, extracted_content: List[ExtractedContent],
                           chapter_num: int, chapter_title: str):
        """Extract code blocks"""
        code_tags = soup.find_all(['code', 'pre'])
        for code_tag in code_tags:
            code_text = code_tag.get_text().strip()
            if code_text and len(code_text) > 10:  # Skip very short code snippets
                extracted_content.append(ExtractedContent(
                    text=code_text,
                    metadata={
                        "chapter": chapter_num,
                        "chapter_title": chapter_title,
                        "language": code_tag.get('class', ['unknown'])[0]
                    },
                    page_number=chapter_num,
                    content_type="code",
                    section_info={"chapter_title": chapter_title}
                ))
                # Remove from soup to avoid duplication in main text
                code_tag.decompose()
    
    def _extract_tables(self, soup: BeautifulSoup, extracted_content: List[ExtractedContent],
                       chapter_num: int, chapter_title: str):
        """Extract table content"""
        tables = soup.find_all('table')
        for table in tables:
            table_text = self._format_html_table(table)
            if table_text:
                extracted_content.append(ExtractedContent(
                    text=table_text,
                    metadata={
                        "chapter": chapter_num,
                        "chapter_title": chapter_title
                    },
                    page_number=chapter_num,
                    content_type="table",
                    section_info={"chapter_title": chapter_title}
                ))
                # Remove from soup
                table.decompose()
    
    def _extract_math_content(self, soup: BeautifulSoup, extracted_content: List[ExtractedContent],
                            chapter_num: int, chapter_title: str):
        """Extract mathematical formulas"""
        math_tags = soup.find_all(['math', 'equation'])
        for math_tag in math_tags:
            math_text = str(math_tag)  # Keep original MathML/LaTeX
            if math_text:
                extracted_content.append(ExtractedContent(
                    text=math_text,
                    metadata={
                        "chapter": chapter_num,
                        "chapter_title": chapter_title
                    },
                    page_number=chapter_num,
                    content_type="formula",
                    section_info={"chapter_title": chapter_title}
                ))
                # Remove from soup
                math_tag.decompose()
    
    def _extract_text_content(self, soup: BeautifulSoup, extracted_content: List[ExtractedContent],
                            chapter_num: int, chapter_title: str):
        """Extract main text content"""
        # Get all text, preserving paragraph structure
        paragraphs = soup.find_all(['p', 'div'])
        
        chapter_text_parts = []
        for para in paragraphs:
            text = para.get_text().strip()
            if text and len(text) > 20:  # Skip very short paragraphs
                chapter_text_parts.append(text)
        
        # Also get any remaining text
        remaining_text = soup.get_text().strip()
        if remaining_text and len(remaining_text) > 50:
            chapter_text_parts.append(remaining_text)
        
        # Combine all text for this chapter
        if chapter_text_parts:
            full_text = '\n\n'.join(chapter_text_parts)
            
            # Clean up the text
            full_text = self._clean_text(full_text)
            
            if full_text:
                content_type = self._detect_content_type(full_text)
                
                extracted_content.append(ExtractedContent(
                    text=full_text,
                    metadata={
                        "chapter": chapter_num,
                        "chapter_title": chapter_title
                    },
                    page_number=chapter_num,
                    content_type=content_type,
                    section_info={"chapter_title": chapter_title}
                ))
    
    def _format_html_table(self, table_tag) -> str:
        """Format HTML table as text"""
        rows = []
        for row in table_tag.find_all('tr'):
            cells = []
            for cell in row.find_all(['td', 'th']):
                cell_text = cell.get_text().strip()
                cells.append(cell_text)
            if cells:
                rows.append(' | '.join(cells))
        return '\n'.join(rows)
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Unescape HTML entities
        text = unescape(text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text.strip()
    
    def _detect_content_type(self, text: str) -> str:
        """Detect the type of content in text"""
        # Check for code patterns
        for pattern in self.code_patterns:
            if re.search(pattern, text, re.MULTILINE | re.DOTALL):
                return "code"
        
        # Check for mathematical formulas
        for pattern in self.formula_patterns:
            if re.search(pattern, text, re.MULTILINE | re.DOTALL):
                return "formula"
        
        return "text"
    
    def parse_epub(self, file_path: Path) -> Tuple[BookMetadata, List[ExtractedContent]]:
        """
        Main method to parse EPUB and extract content
        
        Args:
            file_path: Path to EPUB file
            
        Returns:
            Tuple of (metadata, extracted_content_list)
        """
        logger.info(f"Parsing EPUB: {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"EPUB file not found: {file_path}")
        
        try:
            # Open EPUB book
            book = epub.read_epub(str(file_path))
            
            # Extract metadata
            metadata = self.extract_metadata(book, file_path)
            
            # Extract content
            content = self.extract_content(book)
            
            logger.info(f"Extracted {len(content)} content blocks from EPUB")
            
            return metadata, content
            
        except Exception as e:
            logger.error(f"Error parsing EPUB {file_path}: {e}")
            raise

def parse_epub_file(file_path: str) -> Tuple[BookMetadata, List[ExtractedContent]]:
    """Convenience function to parse an EPUB file"""
    parser = EPUBParser()
    return parser.parse_epub(Path(file_path))