"""
Intelligent Text Chunker for TradeKnowledge
Handles smart text segmentation preserving semantic boundaries
"""

import logging
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from utils.logging import get_logger
from .pdf_parser import ExtractedContent

logger = get_logger(__name__)

class ChunkBoundary(Enum):
    """Types of chunk boundaries"""
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    CODE_BLOCK = "code_block"
    SECTION = "section"
    HARD_LIMIT = "hard_limit"

@dataclass
class TextChunk:
    """Container for a text chunk with metadata"""
    text: str
    chunk_id: str
    book_id: str
    chunk_index: int
    metadata: Dict[str, Any]
    content_type: str = "text"
    boundary_type: ChunkBoundary = ChunkBoundary.PARAGRAPH
    overlap_with_previous: bool = False
    
class IntelligentChunker:
    """
    Intelligent text chunker that respects semantic boundaries
    
    Features:
    - Never splits code blocks
    - Respects paragraph boundaries
    - Maintains section context
    - Configurable chunk size with overlap
    - Preserves formatting for special content
    """
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 min_chunk_size: int = 100,
                 max_chunk_size: int = 2000):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # Patterns for detecting special content
        self.code_block_patterns = [
            r'```[\s\S]*?```',  # Markdown code blocks
            r'<pre>[\s\S]*?</pre>',  # HTML pre blocks
            r'<code>[\s\S]*?</code>',  # HTML code blocks
        ]
        
        self.section_markers = [
            r'^#+\s+',  # Markdown headers
            r'^Chapter\s+\d+',  # Chapter headers
            r'^Section\s+\d+',  # Section headers
            r'^Part\s+[IVX]+',  # Part headers
        ]
        
        # Sentence boundary detection
        self.sentence_endings = re.compile(r'[.!?]+\s+')
        
        # Paragraph boundary detection
        self.paragraph_boundary = re.compile(r'\n\s*\n')
    
    def chunk_content(self, content_list: List[ExtractedContent], book_id: str) -> List[TextChunk]:
        """
        Chunk a list of extracted content into optimally sized text chunks
        
        Args:
            content_list: List of ExtractedContent objects
            book_id: Unique identifier for the book
            
        Returns:
            List of TextChunk objects
        """
        chunks = []
        chunk_index = 0
        
        for content in content_list:
            try:
                content_chunks = self._chunk_single_content(content, book_id, chunk_index)
                chunks.extend(content_chunks)
                chunk_index += len(content_chunks)
                
            except Exception as e:
                logger.warning(f"Error chunking content: {e}")
                continue
        
        logger.info(f"Created {len(chunks)} chunks from {len(content_list)} content blocks")
        return chunks
    
    def _chunk_single_content(self, content: ExtractedContent, book_id: str, 
                            start_index: int) -> List[TextChunk]:
        """Chunk a single ExtractedContent object"""
        
        if content.content_type == "code":
            return self._chunk_code_content(content, book_id, start_index)
        elif content.content_type == "table":
            return self._chunk_table_content(content, book_id, start_index)
        elif content.content_type == "formula":
            return self._chunk_formula_content(content, book_id, start_index)
        else:
            return self._chunk_text_content(content, book_id, start_index)
    
    def _chunk_code_content(self, content: ExtractedContent, book_id: str, 
                          start_index: int) -> List[TextChunk]:
        """Handle code content - never split code blocks"""
        
        text = content.text
        
        # If code is small enough, keep as single chunk
        if len(text) <= self.max_chunk_size:
            return [TextChunk(
                text=text,
                chunk_id=f"{book_id}_chunk_{start_index:06d}",
                book_id=book_id,
                chunk_index=start_index,
                metadata=content.metadata,
                content_type="code",
                boundary_type=ChunkBoundary.CODE_BLOCK
            )]
        
        # For very large code blocks, try to split on function/class boundaries
        return self._split_large_code(content, book_id, start_index)
    
    def _split_large_code(self, content: ExtractedContent, book_id: str, 
                         start_index: int) -> List[TextChunk]:
        """Split large code blocks on logical boundaries"""
        text = content.text
        chunks = []
        
        # Try to split on function/class definitions
        boundaries = []
        for pattern in [r'\ndef\s+\w+', r'\nclass\s+\w+', r'\n\n\n']:
            for match in re.finditer(pattern, text):
                boundaries.append(match.start())
        
        if not boundaries:
            # No good boundaries found, create single large chunk
            return [TextChunk(
                text=text,
                chunk_id=f"{book_id}_chunk_{start_index:06d}",
                book_id=book_id,
                chunk_index=start_index,
                metadata=content.metadata,
                content_type="code",
                boundary_type=ChunkBoundary.CODE_BLOCK
            )]
        
        # Split on boundaries
        boundaries.sort()
        boundaries = [0] + boundaries + [len(text)]
        
        for i in range(len(boundaries) - 1):
            chunk_text = text[boundaries[i]:boundaries[i + 1]].strip()
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(TextChunk(
                    text=chunk_text,
                    chunk_id=f"{book_id}_chunk_{start_index + len(chunks):06d}",
                    book_id=book_id,
                    chunk_index=start_index + len(chunks),
                    metadata=content.metadata,
                    content_type="code",
                    boundary_type=ChunkBoundary.CODE_BLOCK
                ))
        
        return chunks
    
    def _chunk_table_content(self, content: ExtractedContent, book_id: str,
                           start_index: int) -> List[TextChunk]:
        """Handle table content"""
        return [TextChunk(
            text=content.text,
            chunk_id=f"{book_id}_chunk_{start_index:06d}",
            book_id=book_id,
            chunk_index=start_index,
            metadata=content.metadata,
            content_type="table",
            boundary_type=ChunkBoundary.SECTION
        )]
    
    def _chunk_formula_content(self, content: ExtractedContent, book_id: str,
                             start_index: int) -> List[TextChunk]:
        """Handle mathematical formulas"""
        return [TextChunk(
            text=content.text,
            chunk_id=f"{book_id}_chunk_{start_index:06d}",
            book_id=book_id,
            chunk_index=start_index,
            metadata=content.metadata,
            content_type="formula",
            boundary_type=ChunkBoundary.SECTION
        )]
    
    def _chunk_text_content(self, content: ExtractedContent, book_id: str,
                          start_index: int) -> List[TextChunk]:
        """Chunk regular text content intelligently"""
        text = content.text
        
        # First, check if text contains code blocks that shouldn't be split
        protected_sections = self._identify_protected_sections(text)
        
        if not protected_sections:
            # No protected sections, use regular chunking
            return self._chunk_regular_text(text, content, book_id, start_index)
        
        # Handle text with protected sections
        return self._chunk_text_with_protected_sections(text, protected_sections, 
                                                      content, book_id, start_index)
    
    def _identify_protected_sections(self, text: str) -> List[tuple]:
        """Identify sections that shouldn't be split (code blocks, etc.)"""
        protected = []
        
        for pattern in self.code_block_patterns:
            for match in re.finditer(pattern, text, re.DOTALL):
                protected.append((match.start(), match.end(), 'code'))
        
        # Sort by start position
        protected.sort(key=lambda x: x[0])
        return protected
    
    def _chunk_regular_text(self, text: str, content: ExtractedContent, 
                          book_id: str, start_index: int) -> List[TextChunk]:
        """Chunk regular text respecting paragraph and sentence boundaries"""
        chunks = []
        
        # Split into paragraphs first
        paragraphs = self.paragraph_boundary.split(text)
        
        current_chunk = ""
        current_chunk_start = start_index
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if adding this paragraph would exceed chunk size
            if current_chunk and len(current_chunk) + len(para) + 2 > self.chunk_size:
                # Save current chunk and start new one
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(self._create_text_chunk(
                        current_chunk, content, book_id, 
                        start_index + len(chunks), ChunkBoundary.PARAGRAPH
                    ))
                
                # Start new chunk with overlap
                current_chunk = self._get_overlap_text(current_chunk) + para
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Add final chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(self._create_text_chunk(
                current_chunk, content, book_id, 
                start_index + len(chunks), ChunkBoundary.PARAGRAPH
            ))
        
        return chunks
    
    def _chunk_text_with_protected_sections(self, text: str, protected_sections: List[tuple],
                                          content: ExtractedContent, book_id: str,
                                          start_index: int) -> List[TextChunk]:
        """Chunk text while preserving protected sections"""
        chunks = []
        last_end = 0
        
        for start, end, section_type in protected_sections:
            # Chunk text before protected section
            if start > last_end:
                before_text = text[last_end:start].strip()
                if before_text:
                    text_chunks = self._chunk_regular_text(before_text, content, 
                                                         book_id, start_index + len(chunks))
                    chunks.extend(text_chunks)
            
            # Add protected section as single chunk
            protected_text = text[start:end].strip()
            if protected_text:
                chunks.append(TextChunk(
                    text=protected_text,
                    chunk_id=f"{book_id}_chunk_{start_index + len(chunks):06d}",
                    book_id=book_id,
                    chunk_index=start_index + len(chunks),
                    metadata=content.metadata,
                    content_type=section_type,
                    boundary_type=ChunkBoundary.CODE_BLOCK
                ))
            
            last_end = end
        
        # Chunk remaining text after last protected section
        if last_end < len(text):
            after_text = text[last_end:].strip()
            if after_text:
                text_chunks = self._chunk_regular_text(after_text, content, 
                                                     book_id, start_index + len(chunks))
                chunks.extend(text_chunks)
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of a chunk"""
        if len(text) <= self.chunk_overlap:
            return text + "\n\n"
        
        # Try to get overlap at sentence boundary
        overlap_start = len(text) - self.chunk_overlap
        sentences = self.sentence_endings.split(text[overlap_start:])
        
        if len(sentences) > 1:
            # Use complete sentences for overlap
            overlap = sentences[-2] + sentences[-1]
        else:
            # Fallback to character-based overlap
            overlap = text[-self.chunk_overlap:]
        
        return overlap.strip() + "\n\n"
    
    def _create_text_chunk(self, text: str, content: ExtractedContent, 
                          book_id: str, chunk_index: int, 
                          boundary_type: ChunkBoundary) -> TextChunk:
        """Create a text chunk with proper metadata"""
        return TextChunk(
            text=text.strip(),
            chunk_id=f"{book_id}_chunk_{chunk_index:06d}",
            book_id=book_id,
            chunk_index=chunk_index,
            metadata=content.metadata,
            content_type=content.content_type,
            boundary_type=boundary_type
        )
    
    def get_chunk_stats(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """Get statistics about the chunking results"""
        if not chunks:
            return {}
        
        chunk_sizes = [len(chunk.text) for chunk in chunks]
        content_types = {}
        boundary_types = {}
        
        for chunk in chunks:
            content_types[chunk.content_type] = content_types.get(chunk.content_type, 0) + 1
            boundary_types[chunk.boundary_type.value] = boundary_types.get(chunk.boundary_type.value, 0) + 1
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "content_types": content_types,
            "boundary_types": boundary_types,
            "total_characters": sum(chunk_sizes)
        }

def chunk_extracted_content(content_list: List[ExtractedContent], book_id: str,
                          chunk_size: int = 1000, chunk_overlap: int = 200) -> List[TextChunk]:
    """Convenience function to chunk extracted content"""
    chunker = IntelligentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_content(content_list, book_id)