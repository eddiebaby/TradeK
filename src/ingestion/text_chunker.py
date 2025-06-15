"""
Intelligent Text Chunking for TradeKnowledge

This module breaks text into optimal chunks for searching and embedding.
The key challenge is maintaining context while keeping chunks at a reasonable size.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..core.models import Chunk, ChunkType

logger = logging.getLogger(__name__)

@dataclass
class ChunkingConfig:
    """Configuration for chunking behavior"""
    chunk_size: int = 1000  # Target size in characters
    chunk_overlap: int = 200  # Overlap between chunks
    min_chunk_size: int = 100  # Minimum viable chunk
    max_chunk_size: int = 2000  # Maximum chunk size
    respect_sentences: bool = True  # Try to break at sentence boundaries
    respect_paragraphs: bool = True  # Try to break at paragraph boundaries
    preserve_code_blocks: bool = True  # Don't split code blocks

class TextChunker:
    """
    Intelligently chunks text for optimal search and retrieval.
    
    This class handles the complexity of breaking text into chunks that:
    1. Maintain semantic coherence
    2. Preserve context through overlap
    3. Respect natural boundaries (sentences, paragraphs)
    4. Handle special content (code, formulas) appropriately
    """
    
    def __init__(self, config: ChunkingConfig = None):
        """Initialize chunker with configuration"""
        self.config = config or ChunkingConfig()
        
        # Compile regex patterns for efficiency
        self.sentence_end_pattern = re.compile(r'[.!?]\s+')
        self.paragraph_pattern = re.compile(r'\n\s*\n')
        self.code_block_pattern = re.compile(
            r'```[\s\S]*?```|`[^`]+`',
            re.MULTILINE
        )
        self.formula_pattern = re.compile(
            r'\$\$[\s\S]*?\$\$|\$[^\$]+\$',
            re.MULTILINE
        )
        
    def chunk_text(self, 
                   text: str, 
                   book_id: str,
                   metadata: Dict[str, Any] = None) -> List[Chunk]:
        """
        Chunk text into optimal pieces.
        
        This is the main entry point for chunking. It coordinates
        the identification of special content and the actual chunking process.
        
        Args:
            text: The text to chunk
            book_id: ID of the source book
            metadata: Additional metadata for chunks
            
        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            logger.warning(f"Empty text provided for book {book_id}")
            return []
        
        logger.info(f"Starting to chunk text for book {book_id}, length: {len(text)}")
        
        # Pre-process text to identify special regions
        special_regions = self._identify_special_regions(text)
        
        # Perform the actual chunking
        chunks = self._create_chunks(text, special_regions)
        
        # Convert to Chunk objects with proper metadata
        chunk_objects = self._create_chunk_objects(
            chunks, book_id, metadata or {}
        )
        
        # Link chunks for context
        self._link_chunks(chunk_objects)
        
        logger.info(f"Created {len(chunk_objects)} chunks for book {book_id}")
        return chunk_objects
    
    def chunk_pages(self,
                    pages: List[Dict[str, Any]],
                    book_id: str,
                    metadata: Dict[str, Any] = None) -> List[Chunk]:
        """
        Chunk a list of pages from a book.
        
        This method handles page-by-page chunking while maintaining
        continuity across page boundaries.
        
        Args:
            pages: List of page dictionaries with 'text' and 'page_number'
            book_id: ID of the source book  
            metadata: Additional metadata
            
        Returns:
            List of Chunk objects
        """
        all_chunks = []
        accumulated_text = ""
        current_page_start = 1
        
        for page in pages:
            page_num = page.get('page_number', 0)
            page_text = page.get('text', '')
            
            if not page_text.strip():
                continue
            
            # Add page text to accumulator
            if accumulated_text:
                accumulated_text += "\n"
            accumulated_text += page_text
            
            # Check if we should chunk the accumulated text
            if len(accumulated_text) >= self.config.chunk_size:
                # Chunk what we have so far
                chunks = self.chunk_text(accumulated_text, book_id, metadata)
                
                # Add page information to chunks
                for chunk in chunks:
                    chunk.page_start = current_page_start
                    chunk.page_end = page_num
                
                all_chunks.extend(chunks)
                
                # Keep overlap for next batch
                if chunks and self.config.chunk_overlap > 0:
                    last_chunk_text = chunks[-1].text
                    overlap_start = max(0, len(last_chunk_text) - self.config.chunk_overlap)
                    accumulated_text = last_chunk_text[overlap_start:]
                    current_page_start = page_num
                else:
                    accumulated_text = ""
                    current_page_start = page_num + 1
        
        # Handle remaining text
        if accumulated_text.strip():
            chunks = self.chunk_text(accumulated_text, book_id, metadata)
            for chunk in chunks:
                chunk.page_start = current_page_start
                chunk.page_end = pages[-1].get('page_number', current_page_start)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _identify_special_regions(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Identify regions that should not be split.
        
        These include:
        - Code blocks
        - Mathematical formulas  
        - Tables
        
        Returns:
            List of (start, end, type) tuples
        """
        regions = []
        
        # Find code blocks
        if self.config.preserve_code_blocks:
            for match in self.code_block_pattern.finditer(text):
                regions.append((match.start(), match.end(), 'code'))
        
        # Find formulas
        for match in self.formula_pattern.finditer(text):
            regions.append((match.start(), match.end(), 'formula'))
        
        # Sort by start position
        regions.sort(key=lambda x: x[0])
        
        # Merge overlapping regions
        merged = []
        for region in regions:
            if merged and region[0] < merged[-1][1]:
                # Overlapping - extend the previous region
                merged[-1] = (merged[-1][0], max(merged[-1][1], region[1]), 'mixed')
            else:
                merged.append(region)
        
        return merged
    
    def _create_chunks(self, 
                       text: str, 
                       special_regions: List[Tuple[int, int, str]]) -> List[str]:
        """
        Create chunks respecting special regions and boundaries.
        
        This is the core chunking algorithm that:
        1. Avoids splitting special regions
        2. Prefers natural boundaries
        3. Maintains overlap for context
        """
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            # Determine chunk end position
            chunk_end = min(current_pos + self.config.chunk_size, len(text))
            
            # Check if we're in or near a special region
            for region_start, region_end, region_type in special_regions:
                if current_pos <= region_start < chunk_end:
                    # Special region starts within our chunk
                    if region_end <= current_pos + self.config.max_chunk_size:
                        # We can include the entire special region
                        chunk_end = region_end
                    else:
                        # Special region is too large, chunk before it
                        chunk_end = region_start
                    break
            
            # If not at a special region, find a good break point
            if chunk_end < len(text):
                chunk_end = self._find_break_point(text, current_pos, chunk_end)
            
            # Extract chunk
            chunk_text = text[current_pos:chunk_end].strip()
            
            if len(chunk_text) >= self.config.min_chunk_size:
                chunks.append(chunk_text)
                
                # Move position with overlap
                if chunk_end < len(text):
                    overlap_start = max(0, chunk_end - self.config.chunk_overlap)
                    current_pos = overlap_start
                else:
                    current_pos = chunk_end
            else:
                # Chunk too small, extend it
                current_pos = chunk_end
        
        return chunks
    
    def _find_break_point(self, text: str, start: int, ideal_end: int) -> int:
        """
        Find the best position to break text.
        
        Priority:
        1. Paragraph boundary
        2. Sentence boundary  
        3. Word boundary
        4. Any position (fallback)
        """
        # Look for paragraph break
        if self.config.respect_paragraphs:
            paragraph_breaks = list(self.paragraph_pattern.finditer(
                text[start:ideal_end + 100]  # Look a bit ahead
            ))
            if paragraph_breaks:
                # Use the last paragraph break before ideal_end
                for match in reversed(paragraph_breaks):
                    if start + match.start() <= ideal_end:
                        return start + match.end()
        
        # Look for sentence break
        if self.config.respect_sentences:
            sentence_breaks = list(self.sentence_end_pattern.finditer(
                text[start:ideal_end + 50]
            ))
            if sentence_breaks:
                # Use the last sentence break
                last_break = sentence_breaks[-1]
                return start + last_break.end()
        
        # Fall back to word boundary
        space_pos = text.rfind(' ', start, ideal_end)
        if space_pos > start:
            return space_pos + 1
        
        # Last resort - break at ideal_end
        return ideal_end
    
    def _create_chunk_objects(self,
                             text_chunks: List[str],
                             book_id: str,
                             metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Convert text chunks to Chunk objects with metadata.
        """
        chunks = []
        
        for idx, text in enumerate(text_chunks):
            # Determine chunk type
            chunk_type = self._determine_chunk_type(text)
            
            chunk = Chunk(
                book_id=book_id,
                chunk_index=idx,
                text=text,
                chunk_type=chunk_type,
                metadata=metadata.copy()
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _determine_chunk_type(self, text: str) -> ChunkType:
        """
        Determine the type of content in a chunk.
        
        This helps with search relevance and display formatting.
        """
        # Check for code indicators
        code_indicators = ['def ', 'class ', 'import ', 'function', '{', '}', 
                          'return ', 'if ', 'for ', 'while ']
        code_count = sum(1 for indicator in code_indicators if indicator in text)
        if code_count >= 3 or text.strip().startswith('```'):
            return ChunkType.CODE
        
        # Check for formula indicators
        if '$' in text and any(x in text for x in ['=', '+', '-', '*', '/']):
            return ChunkType.FORMULA
        
        # Check for table indicators
        if text.count('|') > 5 and text.count('\n') > 2:
            return ChunkType.TABLE
        
        # Default to text
        return ChunkType.TEXT
    
    def _link_chunks(self, chunks: List[Chunk]) -> None:
        """
        Link chunks to maintain context.
        
        This allows us to easily retrieve surrounding context
        when displaying search results.
        """
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.previous_chunk_id = chunks[i-1].id
            if i < len(chunks) - 1:
                chunk.next_chunk_id = chunks[i+1].id

# Example usage and testing
def test_chunker():
    """Test the chunker with sample text"""
    
    # Sample text with code
    sample_text = """
    Chapter 3: Moving Averages in Trading
    
    Moving averages are one of the most popular technical indicators used in algorithmic trading.
    They help smooth out price action and identify trends.
    
    Here's a simple implementation in Python:
    
    ```python
    def calculate_sma(prices, period):
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / period
    ```
    
    The simple moving average (SMA) calculates the arithmetic mean of prices over a specified period.
    For example, a 20-day SMA sums up the closing prices of the last 20 days and divides by 20.
    
    Traders often use multiple moving averages:
    - Short-term (e.g., 10-day): Responds quickly to price changes
    - Medium-term (e.g., 50-day): Balances responsiveness and smoothness  
    - Long-term (e.g., 200-day): Shows overall trend direction
    
    The formula for exponential moving average (EMA) is:
    $EMA_t = α × Price_t + (1 - α) × EMA_{t-1}$
    
    Where α (alpha) is the smoothing factor: α = 2 / (N + 1)
    """
    
    # Create chunker with small chunks for testing
    config = ChunkingConfig(
        chunk_size=300,
        chunk_overlap=50,
        preserve_code_blocks=True
    )
    chunker = TextChunker(config)
    
    # Chunk the text
    chunks = chunker.chunk_text(sample_text, "test_book_001")
    
    # Display results
    print(f"Created {len(chunks)} chunks\n")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i} ({chunk.chunk_type.value}):")
        print(f"Length: {len(chunk.text)} characters")
        print(f"Preview: {chunk.text[:100]}...")
        print(f"Links: prev={chunk.previous_chunk_id}, next={chunk.next_chunk_id}")
        print("-" * 50)

if __name__ == "__main__":
    test_chunker()