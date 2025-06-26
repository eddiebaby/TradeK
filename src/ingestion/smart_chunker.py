"""
Smart Content-Aware Chunking Strategy

This module implements intelligent chunking that respects document structure
and content type for optimal retrieval performance.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..core.models import Chunk, ChunkType

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Content type classification"""

    ACADEMIC_PAPER = "academic_paper"
    TRADING_BOOK = "trading_book"
    TECHNICAL_MANUAL = "technical_manual"
    GENERAL_TEXT = "general_text"


@dataclass
class ChunkingStrategy:
    """Configuration for content-specific chunking"""

    content_type: ContentType
    target_chunk_size: int
    overlap_ratio: float
    preserve_structures: list[str]  # e.g., ['paragraph', 'section', 'chapter']
    split_patterns: list[str]  # regex patterns for splitting


class SmartChunker:
    """
    Intelligent chunking that adapts to content type and structure.

    Features:
    - Content type detection
    - Structure-aware splitting
    - Semantic boundary preservation
    - Optimal chunk sizing for retrieval
    """

    def __init__(self):
        """Initialize smart chunker with predefined strategies"""
        self.strategies = {
            ContentType.ACADEMIC_PAPER: ChunkingStrategy(
                content_type=ContentType.ACADEMIC_PAPER,
                target_chunk_size=800,
                overlap_ratio=0.15,
                preserve_structures=["section", "subsection", "paragraph"],
                split_patterns=[
                    r"\n\s*\d+\.\s+[A-Z]",  # Numbered sections
                    r"\n\s*[A-Z][A-Z\s]{2,}\n",  # ALL CAPS headers
                    r"\n\s*Abstract\s*\n",
                    r"\n\s*Introduction\s*\n",
                    r"\n\s*Conclusion\s*\n",
                    r"\n\s*References\s*\n",
                ],
            ),
            ContentType.TRADING_BOOK: ChunkingStrategy(
                content_type=ContentType.TRADING_BOOK,
                target_chunk_size=1000,
                overlap_ratio=0.2,
                preserve_structures=["chapter", "section", "strategy", "example"],
                split_patterns=[
                    r"\n\s*Chapter\s+\d+",
                    r"\n\s*Strategy\s+\d*:?",
                    r"\n\s*Example\s+\d*:?",
                    r"\n\s*Key\s+Points?:?",
                    r"\n\s*Summary\s*:?",
                ],
            ),
            ContentType.TECHNICAL_MANUAL: ChunkingStrategy(
                content_type=ContentType.TECHNICAL_MANUAL,
                target_chunk_size=600,
                overlap_ratio=0.1,
                preserve_structures=["procedure", "step", "warning", "note"],
                split_patterns=[
                    r"\n\s*Step\s+\d+",
                    r"\n\s*Procedure\s+\d*:?",
                    r"\n\s*WARNING:",
                    r"\n\s*NOTE:",
                    r"\n\s*CAUTION:",
                ],
            ),
            ContentType.GENERAL_TEXT: ChunkingStrategy(
                content_type=ContentType.GENERAL_TEXT,
                target_chunk_size=1000,
                overlap_ratio=0.15,
                preserve_structures=["paragraph", "section"],
                split_patterns=[
                    r"\n\s*\n\s*",  # Double newlines
                    r"\n\s*[-=]{3,}\s*\n",  # Horizontal rules
                ],
            ),
        }

    def detect_content_type(self, text: str, filename: str = "") -> ContentType:
        """
        Detect content type based on text patterns and filename.

        Args:
            text: Document text content
            filename: Original filename

        Returns:
            Detected content type
        """
        text_lower = text.lower()
        filename_lower = filename.lower()

        # Academic paper indicators
        academic_indicators = [
            "abstract",
            "introduction",
            "methodology",
            "results",
            "conclusion",
            "references",
            "bibliography",
            "et al.",
            "doi:",
            "arxiv:",
            "journal",
            "proceedings",
        ]

        # Trading book indicators
        trading_indicators = [
            "trading",
            "strategy",
            "portfolio",
            "risk management",
            "backtesting",
            "algorithmic",
            "quantitative",
            "market",
            "technical analysis",
            "fundamental analysis",
            "options",
            "futures",
            "forex",
            "cryptocurrency",
            "bitcoin",
        ]

        # Technical manual indicators
        technical_indicators = [
            "step",
            "procedure",
            "installation",
            "configuration",
            "warning:",
            "note:",
            "caution:",
            "troubleshooting",
            "api",
            "documentation",
            "tutorial",
            "guide",
        ]

        # Score each content type
        academic_score = sum(
            1 for indicator in academic_indicators if indicator in text_lower
        )
        trading_score = sum(
            1 for indicator in trading_indicators if indicator in text_lower
        )
        technical_score = sum(
            1 for indicator in technical_indicators if indicator in text_lower
        )

        # Filename-based hints
        if any(
            word in filename_lower for word in ["paper", "journal", "research", "study"]
        ):
            academic_score += 2
        if any(
            word in filename_lower
            for word in ["trading", "finance", "market", "strategy"]
        ):
            trading_score += 2
        if any(
            word in filename_lower
            for word in ["manual", "guide", "documentation", "tutorial"]
        ):
            technical_score += 2

        # Determine content type
        scores = {
            ContentType.ACADEMIC_PAPER: academic_score,
            ContentType.TRADING_BOOK: trading_score,
            ContentType.TECHNICAL_MANUAL: technical_score,
        }

        max_score = max(scores.values())
        if max_score >= 3:  # Minimum confidence threshold
            return max(scores, key=scores.get)

        return ContentType.GENERAL_TEXT

    def find_structure_boundaries(self, text: str, patterns: list[str]) -> list[int]:
        """
        Find structural boundaries in text using regex patterns.

        Args:
            text: Text to analyze
            patterns: List of regex patterns for boundaries

        Returns:
            List of character positions for boundaries
        """
        boundaries = [0]  # Start of document

        for pattern in patterns:
            try:
                matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
                for match in matches:
                    pos = match.start()
                    if pos not in boundaries:
                        boundaries.append(pos)
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")

        boundaries.append(len(text))  # End of document
        return sorted(set(boundaries))

    def find_sentence_boundaries(self, text: str) -> list[int]:
        """
        Find sentence boundaries for better chunking.

        Args:
            text: Text to analyze

        Returns:
            List of sentence boundary positions
        """
        # Simple sentence boundary detection
        sentence_endings = re.finditer(r"[.!?]+\s+", text)
        boundaries = [0]

        for match in sentence_endings:
            end_pos = match.end()
            # Avoid splitting on abbreviations and numbers
            preceding_text = text[max(0, match.start() - 10) : match.start()]
            if not re.search(
                r"\b[A-Z][a-z]?\.\s*$", preceding_text
            ):  # Not an abbreviation
                boundaries.append(end_pos)

        boundaries.append(len(text))
        return sorted(set(boundaries))

    def create_chunks_with_strategy(
        self,
        text: str,
        strategy: ChunkingStrategy,
        book_id: str,
        metadata: dict[str, Any],
    ) -> list[Chunk]:
        """
        Create chunks using the specified strategy.

        Args:
            text: Text to chunk
            strategy: Chunking strategy to use
            book_id: Book identifier
            metadata: Additional metadata

        Returns:
            List of created chunks
        """
        # Find structural boundaries
        structure_boundaries = self.find_structure_boundaries(
            text, strategy.split_patterns
        )
        sentence_boundaries = self.find_sentence_boundaries(text)

        # Combine and sort all boundaries
        all_boundaries = sorted(set(structure_boundaries + sentence_boundaries))

        chunks = []
        overlap_size = int(strategy.target_chunk_size * strategy.overlap_ratio)

        i = 0
        chunk_index = 0

        while i < len(all_boundaries) - 1:
            start_pos = max(
                0, all_boundaries[i] - (overlap_size if chunk_index > 0 else 0)
            )

            # Find end position that doesn't exceed target size
            end_pos = all_boundaries[i + 1]

            # Look ahead to find optimal end position
            for j in range(i + 1, len(all_boundaries)):
                candidate_end = all_boundaries[j]
                chunk_size = candidate_end - start_pos

                if chunk_size <= strategy.target_chunk_size * 1.2:  # Allow 20% overage
                    end_pos = candidate_end
                else:
                    break

            # Extract chunk text
            chunk_text = text[start_pos:end_pos].strip()

            if len(chunk_text) >= 50:  # Minimum chunk size
                # Detect chunk type based on content
                chunk_type = self._detect_chunk_type(chunk_text, strategy.content_type)

                # Extract structural information
                structure_info = self._extract_structure_info(
                    chunk_text, strategy.preserve_structures
                )

                chunk = Chunk(
                    id=f"{book_id}_chunk_{chunk_index:04d}",
                    book_id=book_id,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    chunk_type=chunk_type,
                    start_char=start_pos,
                    end_char=end_pos,
                    metadata={
                        **metadata,
                        "content_type": strategy.content_type.value,
                        "strategy_used": "smart_chunking",
                        "structure_info": structure_info,
                        "chunk_size": len(chunk_text),
                    },
                )

                chunks.append(chunk)
                chunk_index += 1

            # Move to next boundary, ensuring progress
            next_i = i + 1
            while (
                next_i < len(all_boundaries) - 1
                and all_boundaries[next_i]
                <= start_pos + strategy.target_chunk_size // 2
            ):
                next_i += 1
            i = max(i + 1, next_i)

        logger.info(
            f"Created {len(chunks)} smart chunks for {strategy.content_type.value}"
        )
        return chunks

    def _detect_chunk_type(self, text: str, content_type: ContentType) -> ChunkType:
        """Detect specific chunk type based on content"""
        text_lower = text.lower().strip()

        # Check for specific patterns
        if re.match(r"^(abstract|summary):", text_lower):
            return ChunkType.SUMMARY
        elif re.match(r"^(table|figure|chart):", text_lower):
            return ChunkType.TABLE
        elif any(
            word in text_lower for word in ["def ", "class ", "import ", "function"]
        ):
            return ChunkType.CODE
        elif content_type == ContentType.TRADING_BOOK and any(
            word in text_lower for word in ["strategy", "algorithm", "backtesting"]
        ):
            return ChunkType.STRATEGY
        else:
            return ChunkType.TEXT

    def _extract_structure_info(
        self, text: str, preserve_structures: list[str]
    ) -> dict[str, Any]:
        """Extract structural information from chunk"""
        info = {}

        # Check for headers
        lines = text.split("\n")
        first_line = lines[0].strip() if lines else ""

        if re.match(r"^[A-Z][A-Z\s]{5,}$", first_line):  # ALL CAPS header
            info["header"] = first_line
            info["header_type"] = "section"
        elif re.match(r"^\d+\.\s+[A-Z]", first_line):  # Numbered header
            info["header"] = first_line
            info["header_type"] = "numbered_section"
        elif re.match(r"^Chapter\s+\d+", first_line, re.IGNORECASE):
            info["header"] = first_line
            info["header_type"] = "chapter"

        # Check for special content
        if "strategy" in preserve_structures and "strategy" in text.lower():
            info["contains_strategy"] = True
        if "example" in preserve_structures and re.search(
            r"example\s*\d*:", text.lower()
        ):
            info["contains_example"] = True
        if any(word in text.lower() for word in ["warning", "caution", "note"]):
            info["contains_alert"] = True

        return info

    def chunk_text(
        self, text: str, filename: str, book_id: str, metadata: dict[str, Any] = None
    ) -> list[Chunk]:
        """
        Main chunking method that automatically detects content type and applies
        appropriate strategy.

        Args:
            text: Text to chunk
            filename: Original filename for type detection
            book_id: Book identifier
            metadata: Additional metadata

        Returns:
            List of intelligently created chunks
        """
        if not text or not text.strip():
            return []

        # Detect content type
        content_type = self.detect_content_type(text, filename)
        logger.info(f"Detected content type: {content_type.value} for {filename}")

        # Get appropriate strategy
        strategy = self.strategies[content_type]

        # Apply smart chunking
        chunks = self.create_chunks_with_strategy(
            text, strategy, book_id, metadata or {}
        )

        return chunks


# Convenience function for backward compatibility
def smart_chunk_text(
    text: str, filename: str, book_id: str, metadata: dict[str, Any] = None
) -> list[Chunk]:
    """Convenience function for smart chunking"""
    chunker = SmartChunker()
    return chunker.chunk_text(text, filename, book_id, metadata)
