"""
EPUB parser for TradeKnowledge

Handles extraction of text and metadata from EPUB files.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re
import html
import asyncio

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class EPUBParser:
    """
    Parser for EPUB format ebooks.
    
    EPUB files are essentially ZIP archives containing HTML files,
    so we need to extract and parse the HTML content.
    """
    
    def __init__(self):
        """Initialize EPUB parser"""
        self.supported_extensions = ['.epub']
    
    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the file"""
        return file_path.suffix.lower() in self.supported_extensions
    
    async def parse_file_async(self, file_path: Path) -> Dict[str, Any]:
        """Async wrapper for parse_file"""
        return await asyncio.to_thread(self.parse_file, file_path)
    
    def parse_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse an EPUB file and extract content.
        
        Args:
            file_path: Path to EPUB file
            
        Returns:
            Dictionary with metadata and pages
        """
        logger.info(f"Starting to parse EPUB: {file_path}")
        
        result = {
            'metadata': {},
            'pages': [],
            'errors': []
        }
        
        try:
            # Open EPUB file
            book = epub.read_epub(str(file_path))
            
            # Extract metadata
            result['metadata'] = self._extract_metadata(book)
            
            # Extract chapters/pages
            result['pages'] = self._extract_content(book)
            
            # Add statistics
            result['statistics'] = {
                'total_pages': len(result['pages']),
                'total_words': sum(p['word_count'] for p in result['pages']),
                'total_characters': sum(p['char_count'] for p in result['pages'])
            }
            
            logger.info(
                f"Successfully parsed EPUB: {result['statistics']['total_pages']} sections, "
                f"{result['statistics']['total_words']} words"
            )
            
        except Exception as e:
            error_msg = f"Error parsing EPUB: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result['errors'].append(error_msg)
        
        return result
    
    def _extract_metadata(self, book: epub.EpubBook) -> Dict[str, Any]:
        """Extract metadata from EPUB"""
        metadata = {}
        
        try:
            # Title
            title = book.get_metadata('DC', 'title')
            if title:
                metadata['title'] = title[0][0]
            
            # Author(s)
            creators = book.get_metadata('DC', 'creator')
            if creators:
                authors = [creator[0] for creator in creators]
                metadata['author'] = ', '.join(authors)
            
            # Language
            language = book.get_metadata('DC', 'language')
            if language:
                metadata['language'] = language[0][0]
            
            # Publisher
            publisher = book.get_metadata('DC', 'publisher')
            if publisher:
                metadata['publisher'] = publisher[0][0]
            
            # Publication date
            date = book.get_metadata('DC', 'date')
            if date:
                metadata['publication_date'] = date[0][0]
            
            # ISBN
            identifiers = book.get_metadata('DC', 'identifier')
            for identifier in identifiers:
                id_value = identifier[0]
                id_type = identifier[1].get('id', '').lower()
                if 'isbn' in id_type or self._is_isbn(id_value):
                    metadata['isbn'] = id_value
                    break
            
            # Description
            description = book.get_metadata('DC', 'description')
            if description:
                metadata['description'] = description[0][0]
            
            # Subject/Categories
            subjects = book.get_metadata('DC', 'subject')
            if subjects:
                metadata['subjects'] = [subject[0] for subject in subjects]
            
        except Exception as e:
            logger.warning(f"Error extracting metadata: {e}")
        
        return metadata
    
    def _extract_content(self, book: epub.EpubBook) -> List[Dict[str, Any]]:
        """Extract text content from EPUB"""
        pages = []
        page_number = 1
        
        # Get spine (reading order)
        spine = book.spine
        
        for spine_item in spine:
            item_id = spine_item[0]
            
            try:
                item = book.get_item_with_id(item_id)
                
                if item and isinstance(item, epub.EpubHtml):
                    # Extract text from HTML
                    content = item.get_content()
                    text, structure = self._parse_html_content(content)
                    
                    if text.strip():
                        pages.append({
                            'page_number': page_number,
                            'text': text,
                            'word_count': len(text.split()),
                            'char_count': len(text),
                            'chapter': structure.get('chapter'),
                            'section': structure.get('section'),
                            'item_id': item_id,
                            'file_name': item.file_name
                        })
                        
                        page_number += 1
                        
            except Exception as e:
                logger.warning(f"Error processing spine item {item_id}: {e}")
        
        # Also process any items not in spine (some EPUBs are weird)
        for item in book.get_items():
            if isinstance(item, epub.EpubHtml) and item.id not in [s[0] for s in spine]:
                try:
                    content = item.get_content()
                    text, structure = self._parse_html_content(content)
                    
                    if text.strip() and len(text) > 100:  # Only substantial content
                        pages.append({
                            'page_number': page_number,
                            'text': text,
                            'word_count': len(text.split()),
                            'char_count': len(text),
                            'chapter': structure.get('chapter'),
                            'section': structure.get('section'),
                            'item_id': item.id,
                            'file_name': item.file_name,
                            'not_in_spine': True
                        })
                        
                        page_number += 1
                        
                except Exception as e:
                    logger.debug(f"Error processing non-spine item: {e}")
        
        return pages
    
    def _parse_html_content(self, html_content: bytes) -> Tuple[str, Dict[str, Any]]:
        """
        Parse HTML content and extract text.
        
        Returns:
            Tuple of (text, structure_info)
        """
        try:
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract structure information
            structure = {}
            
            # Try to find chapter title
            for tag in ['h1', 'h2', 'h3']:
                heading = soup.find(tag)
                if heading:
                    structure['chapter'] = heading.get_text(strip=True)
                    break
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text(separator='\n')
            
            # Clean up text
            text = self._clean_text(text)
            
            # Look for code blocks
            code_blocks = soup.find_all(['pre', 'code'])
            if code_blocks:
                structure['has_code'] = True
                structure['code_blocks'] = []
                
                for block in code_blocks:
                    code_text = block.get_text(strip=True)
                    if code_text:
                        structure['code_blocks'].append(code_text)
            
            # Look for math formulas (MathML or LaTeX)
            math_elements = soup.find_all(['math', 'span'], 
                                        class_=re.compile(r'math|equation|formula', re.I))
            if math_elements:
                structure['has_math'] = True
            
            return text, structure
            
        except Exception as e:
            logger.error(f"Error parsing HTML content: {e}")
            return "", {}
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Remove zero-width spaces and other unicode oddities
        text = text.replace('\u200b', '')
        text = text.replace('\ufeff', '')
        
        # Normalize quotes and dashes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('–', '-').replace('—', '-')
        
        return text.strip()
    
    def _is_isbn(self, value: str) -> bool:
        """Check if a string looks like an ISBN"""
        # Remove hyphens and spaces
        clean_value = value.replace('-', '').replace(' ', '')
        
        # ISBN-10 or ISBN-13
        if len(clean_value) in [10, 13]:
            return clean_value[:-1].isdigit()
        
        return False


# Test EPUB parser
def test_epub_parser():
    """Test the EPUB parser"""
    parser = EPUBParser()
    
    # Test with a sample EPUB file
    test_file = Path("data/books/sample.epub")
    
    if test_file.exists():
        result = parser.parse_file(test_file)
        
        print(f"Title: {result['metadata'].get('title', 'Unknown')}")
        print(f"Author: {result['metadata'].get('author', 'Unknown')}")
        print(f"Sections: {result['statistics']['total_pages']}")
        print(f"Words: {result['statistics']['total_words']}")
        
        # Show first section sample
        if result['pages']:
            first_page = result['pages'][0]
            sample = first_page['text'][:200] + '...' if len(first_page['text']) > 200 else first_page['text']
            print(f"\nFirst section sample:\n{sample}")
    else:
        print(f"Test file not found: {test_file}")
        print("Please add an EPUB file to test with")


if __name__ == "__main__":
    # Setup logging for testing
    logging.basicConfig(level=logging.DEBUG)
    test_epub_parser()