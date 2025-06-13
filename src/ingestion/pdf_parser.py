"""
PDF Parser for TradeKnowledge

This module handles extraction of text and metadata from PDF files.
We start with simple PyPDF2 for clean PDFs, and will add OCR support later.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re
from datetime import datetime

import PyPDF2
import pdfplumber
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
import asyncio

from core.models import Book, FileType

logger = logging.getLogger(__name__)

class PDFParser:
    """
    Parses PDF files and extracts text content.
    
    This class handles the complexity of PDF parsing, including:
    - Text extraction from clean PDFs
    - Metadata extraction
    - Page-by-page processing
    - Error handling for corrupted PDFs
    """
    
    def __init__(self, enable_ocr: bool = True):
        """Initialize the PDF parser"""
        self.supported_extensions = ['.pdf']
        self.enable_ocr = enable_ocr
        self._ocr_processor = None
        
        # Lazy load OCR processor to avoid dependency issues
        if enable_ocr:
            try:
                from .ocr_processor import OCRProcessor
                self._ocr_processor = OCRProcessor()
                logger.info("OCR processor enabled")
            except ImportError as e:
                logger.warning(f"OCR processor not available: {e}")
                self.enable_ocr = False
        
    def can_parse(self, file_path: Path) -> bool:
        """
        Check if this parser can handle the file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file is a PDF
        """
        return file_path.suffix.lower() in self.supported_extensions
    
    def parse_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse a PDF file and extract all content.
        
        This is the main entry point for PDF parsing. It orchestrates
        the extraction of metadata and text content.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary containing:
                - metadata: Book metadata
                - pages: List of page contents
                - errors: Any errors encountered
        """
        # For synchronous usage, run in blocking mode
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            pass
        
        try:
            # Check if we're already in an async context
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context but this is a sync call
                # This is a design issue - should use parse_file_async instead
                raise RuntimeError("Cannot call parse_file() from async context. Use parse_file_async() instead.")
            else:
                return loop.run_until_complete(self.parse_file_async(file_path))
        except RuntimeError:
            # No event loop running, create one
            return asyncio.run(self.parse_file_async(file_path))
    
    async def parse_file_async(self, file_path: Path) -> Dict[str, Any]:
        """
        Async version of parse_file with OCR support.
        
        This method automatically detects if OCR is needed and
        processes the PDF accordingly.
        """
        logger.info(f"Starting to parse PDF: {file_path}")
        
        result = {
            'metadata': {},
            'pages': [],
            'errors': []
        }
        
        # Verify file exists
        if not file_path.exists():
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            result['errors'].append(error_msg)
            return result
        
        # Try PyPDF2 first (faster for clean PDFs)
        try:
            logger.debug("Attempting PyPDF2 extraction")
            metadata, pages = await asyncio.to_thread(self._parse_with_pypdf2, file_path)
            result['metadata'] = metadata
            result['pages'] = pages
            
            # If PyPDF2 extraction was poor, try pdfplumber
            if self._is_extraction_poor(pages):
                logger.info("PyPDF2 extraction poor, trying pdfplumber")
                metadata_plumber, pages_plumber = await asyncio.to_thread(self._parse_with_pdfplumber, file_path)
                
                # Use pdfplumber results if better
                if self._count_words(pages_plumber) > self._count_words(pages) * 1.2:
                    result['pages'] = pages_plumber
                    # Merge metadata, preferring pdfplumber values
                    result['metadata'].update(metadata_plumber)
                    
            # Check if we still have poor extraction and OCR is available
            if self._is_extraction_poor(result['pages']) and self.enable_ocr and self._ocr_processor:
                logger.info("Poor text extraction detected, checking if OCR is needed...")
                
                # Check if OCR would help
                needs_ocr = await self._ocr_processor.needs_ocr(file_path)
                
                if needs_ocr:
                    logger.info("Running OCR on scanned PDF...")
                    
                    # Process with OCR
                    ocr_pages = await self._ocr_processor.process_pdf(file_path)
                    
                    if ocr_pages and self._count_words(ocr_pages) > self._count_words(result['pages']):
                        # Replace pages with OCR results
                        result['pages'] = ocr_pages
                        result['metadata']['ocr_processed'] = True
                        result['metadata']['ocr_confidence'] = sum(
                            p.get('confidence', 0) for p in ocr_pages
                        ) / len(ocr_pages) if ocr_pages else 0.0
                        logger.info(f"OCR completed with average confidence: {result['metadata']['ocr_confidence']:.1f}%")
                    
        except Exception as e:
            error_msg = f"Error parsing PDF: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result['errors'].append(error_msg)
            
            # Try pdfplumber as fallback
            try:
                logger.info("Falling back to pdfplumber")
                metadata, pages = await asyncio.to_thread(self._parse_with_pdfplumber, file_path)
                result['metadata'] = metadata
                result['pages'] = pages
            except Exception as e2:
                error_msg = f"Pdfplumber also failed: {str(e2)}"
                logger.error(error_msg)
                result['errors'].append(error_msg)
                
                # Last resort: try OCR if available
                if self.enable_ocr and self._ocr_processor:
                    try:
                        logger.info("Last resort: attempting OCR processing")
                        ocr_pages = await self._ocr_processor.process_pdf(file_path)
                        if ocr_pages:
                            result['pages'] = ocr_pages
                            result['metadata']['ocr_processed'] = True
                            result['metadata']['ocr_confidence'] = sum(
                                p.get('confidence', 0) for p in ocr_pages
                            ) / len(ocr_pages) if ocr_pages else 0.0
                    except Exception as e3:
                        error_msg = f"OCR processing also failed: {str(e3)}"
                        logger.error(error_msg)
                        result['errors'].append(error_msg)
        
        # Post-process results
        result = self._post_process_results(result, file_path)
        
        logger.info(f"Parsed {len(result['pages'])} pages from {file_path.name}")
        return result
    
    def _parse_with_pypdf2(self, file_path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Parse PDF using PyPDF2 library.
        
        PyPDF2 is fast but sometimes struggles with complex layouts.
        We use it as our primary parser for clean PDFs.
        """
        metadata = {}
        pages = []
        
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            
            # Extract metadata
            if reader.metadata:
                metadata = {
                    'title': self._clean_text(reader.metadata.get('/Title', '')),
                    'author': self._clean_text(reader.metadata.get('/Author', '')),
                    'subject': self._clean_text(reader.metadata.get('/Subject', '')),
                    'creator': self._clean_text(reader.metadata.get('/Creator', '')),
                    'producer': self._clean_text(reader.metadata.get('/Producer', '')),
                    'creation_date': self._parse_date(reader.metadata.get('/CreationDate')),
                    'modification_date': self._parse_date(reader.metadata.get('/ModDate')),
                }
            
            # Extract text from each page
            total_pages = len(reader.pages)
            metadata['total_pages'] = total_pages
            
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    text = page.extract_text()
                    
                    # Clean up the text
                    text = self._clean_text(text)
                    
                    pages.append({
                        'page_number': page_num,
                        'text': text,
                        'word_count': len(text.split()),
                        'char_count': len(text)
                    })
                    
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num}: {e}")
                    pages.append({
                        'page_number': page_num,
                        'text': '',
                        'word_count': 0,
                        'char_count': 0,
                        'error': str(e)
                    })
        
        return metadata, pages
    
    def _parse_with_pdfplumber(self, file_path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Parse PDF using pdfplumber library.
        
        Pdfplumber is better at handling complex layouts and tables,
        but is slower than PyPDF2.
        """
        metadata = {}
        pages = []
        
        with pdfplumber.open(file_path) as pdf:
            # Extract metadata
            if pdf.metadata:
                metadata = {
                    'title': self._clean_text(pdf.metadata.get('Title', '')),
                    'author': self._clean_text(pdf.metadata.get('Author', '')),
                    'subject': self._clean_text(pdf.metadata.get('Subject', '')),
                    'creator': self._clean_text(pdf.metadata.get('Creator', '')),
                    'producer': self._clean_text(pdf.metadata.get('Producer', '')),
                }
            
            metadata['total_pages'] = len(pdf.pages)
            
            # Extract text from each page
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    # Extract text
                    text = page.extract_text() or ''
                    
                    # Also try to extract tables
                    tables = page.extract_tables()
                    if tables:
                        # Convert tables to text representation
                        for table in tables:
                            table_text = self._table_to_text(table)
                            text += f"\n\n[TABLE]\n{table_text}\n[/TABLE]\n"
                    
                    # Clean up the text
                    text = self._clean_text(text)
                    
                    pages.append({
                        'page_number': page_num,
                        'text': text,
                        'word_count': len(text.split()),
                        'char_count': len(text),
                        'has_tables': len(tables) > 0 if tables else False
                    })
                    
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num} with pdfplumber: {e}")
                    pages.append({
                        'page_number': page_num,
                        'text': '',
                        'word_count': 0,
                        'char_count': 0,
                        'error': str(e)
                    })
        
        return metadata, pages
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text.
        
        This handles common issues with PDF text extraction:
        - Excessive whitespace
        - Broken words from line breaks
        - Special characters
        - Encoding issues
        """
        if not text:
            return ''
        
        # Handle different types of input
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='ignore')
        
        # Remove null characters
        text = text.replace('\x00', '')
        
        # Fix hyphenated words at line breaks
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # Replace multiple whitespaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _is_extraction_poor(self, pages: List[Dict[str, Any]]) -> bool:
        """
        Check if text extraction quality is poor.
        
        Poor extraction indicators:
        - Very low word count
        - Many pages with no text
        - Suspicious patterns (all caps, no spaces)
        """
        if not pages:
            return True
        
        total_words = sum(p.get('word_count', 0) for p in pages)
        empty_pages = sum(1 for p in pages if p.get('word_count', 0) < 10)
        
        # Average words per page for a typical book
        avg_words = total_words / len(pages) if pages else 0
        
        # Check for poor extraction
        if avg_words < 50:  # Very low word count
            return True
        
        if empty_pages > len(pages) * 0.2:  # >20% empty pages
            return True
        
        # Check for extraction artifacts
        sample_text = ' '.join(p.get('text', '')[:100] for p in pages[:5])
        if sample_text.isupper():  # All uppercase often indicates OCR needed
            return True
        
        return False
    
    def _count_words(self, pages: List[Dict[str, Any]]) -> int:
        """Count total words across all pages"""
        return sum(p.get('word_count', 0) for p in pages)
    
    def _table_to_text(self, table: List[List[Any]]) -> str:
        """
        Convert table data to readable text format.
        
        Tables in PDFs can contain important data for trading strategies,
        so we preserve them in a readable format.
        """
        if not table:
            return ''
        
        lines = []
        for row in table:
            # Filter out None values and convert to strings
            cleaned_row = [str(cell) if cell is not None else '' for cell in row]
            lines.append(' | '.join(cleaned_row))
        
        return '\n'.join(lines)
    
    def _parse_date(self, date_str: Any) -> Optional[str]:
        """Parse PDF date format to ISO format"""
        if not date_str:
            return None
        
        try:
            # PDF dates are often in format: D:20230615120000+00'00'
            if isinstance(date_str, str) and date_str.startswith('D:'):
                date_str = date_str[2:]  # Remove 'D:' prefix
                # Extract just the date portion
                date_part = date_str[:14]
                if len(date_part) >= 8:
                    year = date_part[:4]
                    month = date_part[4:6]
                    day = date_part[6:8]
                    return f"{year}-{month}-{day}"
        except Exception as e:
            logger.debug(f"Could not parse date {date_str}: {e}")
        
        return str(date_str) if date_str else None
    
    def _post_process_results(self, result: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
        """
        Post-process extraction results.
        
        This adds additional metadata and cleans up the results.
        """
        # Add file information
        result['metadata']['file_name'] = file_path.name
        result['metadata']['file_size'] = file_path.stat().st_size
        
        # If no title found in metadata, use filename
        if not result['metadata'].get('title'):
            # Extract title from filename
            title = file_path.stem
            # Replace underscores and hyphens with spaces
            title = title.replace('_', ' ').replace('-', ' ')
            # Title case
            title = title.title()
            result['metadata']['title'] = title
        
        # Calculate total statistics
        total_words = sum(p.get('word_count', 0) for p in result['pages'])
        total_chars = sum(p.get('char_count', 0) for p in result['pages'])
        
        result['statistics'] = {
            'total_pages': len(result['pages']),
            'total_words': total_words,
            'total_characters': total_chars,
            'average_words_per_page': total_words / len(result['pages']) if result['pages'] else 0
        }
        
        return result

# Standalone function for testing
def test_parser():
    """Test the PDF parser with a sample file"""
    parser = PDFParser()
    
    # Create a test PDF path (you'll need to provide a real PDF)
    test_file = Path("data/books/sample.pdf")
    
    if test_file.exists():
        result = parser.parse_file(test_file)
        
        print(f"Title: {result['metadata'].get('title', 'Unknown')}")
        print(f"Pages: {result['statistics']['total_pages']}")
        print(f"Words: {result['statistics']['total_words']}")
        
        # Show first page sample
        if result['pages']:
            first_page = result['pages'][0]
            sample = first_page['text'][:200] + '...' if len(first_page['text']) > 200 else first_page['text']
            print(f"\nFirst page sample:\n{sample}")
    else:
        print(f"Test file not found: {test_file}")
        print("Please add a PDF file to test with")

if __name__ == "__main__":
    # Setup logging for testing
    logging.basicConfig(level=logging.DEBUG)
    test_parser()