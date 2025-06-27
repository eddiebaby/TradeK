"""
LongLLMLingua Compression Pipeline

Implements prompt compression using insights from the LongLLMLingua paper
to reduce token usage before embedding generation, extending OpenAI quota.
"""

import asyncio
import logging
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

import requests

from ..core.models import Chunk

logger = logging.getLogger(__name__)


@dataclass
class CompressionResult:
    """Result of LLMLingua compression"""
    original_text: str
    compressed_text: str
    compression_ratio: float
    tokens_saved: int
    quality_score: float


class LLMLinguaCompressor:
    """
    LLMLingua-inspired compression for academic papers and trading content.
    
    Based on the ACL 2024 paper findings, this implements:
    - Budget-controlled compression
    - Semantic preservation
    - Context-aware token removal
    - Mathematical content protection
    """
    
    def __init__(self, model_name: str = "qwen2.5-coder:7b", target_compression: float = 0.5):
        """
        Initialize LLMLingua compressor.
        
        Args:
            model_name: Qwen model for intelligent compression
            target_compression: Target compression ratio (0.5 = 50% of original)
        """
        self.model_name = model_name
        self.target_compression = target_compression
        self.ollama_url = "http://localhost:11434/api/generate"
        
        # Patterns to preserve during compression
        self.preserve_patterns = [
            r'\\[.*?\\]',           # LaTeX display math
            r'\\\(.*?\\\)',         # LaTeX inline math
            r'\$\$.*?\$\$',         # Display math
            r'\$.*?\$',             # Inline math
            r'\\begin\{.*?\}.*?\\end\{.*?\}',  # LaTeX environments
            r'\b\d+\.?\d*%?\b',     # Numbers and percentages
            r'\b[A-Z]{2,}\b',       # Acronyms
        ]
        
        # Stopwords and filler phrases to remove aggressively
        self.removable_patterns = [
            r'\b(furthermore|moreover|however|nevertheless|therefore|consequently)\b',
            r'\b(in addition|as a result|on the other hand|in contrast)\b',
            r'\b(it is important to note|it should be noted|as mentioned)\b',
            r'\b(the fact that|it is clear that|obviously|certainly)\b',
        ]
    
    async def compress_chunk(self, chunk: Chunk, preserve_math: bool = True) -> CompressionResult:
        """
        Compress a text chunk using LLMLingua-inspired techniques.
        
        Args:
            chunk: Text chunk to compress
            preserve_math: Whether to preserve mathematical content
            
        Returns:
            CompressionResult with compressed text and metrics
        """
        original_text = chunk.content
        original_tokens = len(original_text.split())
        
        try:
            # Stage 1: Preserve critical content
            preserved_content, text_without_preserved = self._extract_preserved_content(
                original_text, preserve_math
            )
            
            # Stage 2: Use Qwen for intelligent compression
            compressed_main = await self._intelligent_compression(
                text_without_preserved, self.target_compression
            )
            
            # Stage 3: Recombine preserved and compressed content
            final_text = self._recombine_content(preserved_content, compressed_main)
            
            # Calculate metrics
            final_tokens = len(final_text.split())
            compression_ratio = final_tokens / original_tokens if original_tokens > 0 else 1.0
            tokens_saved = original_tokens - final_tokens
            quality_score = await self._estimate_quality(original_text, final_text)
            
            return CompressionResult(
                original_text=original_text,
                compressed_text=final_text,
                compression_ratio=compression_ratio,
                tokens_saved=tokens_saved,
                quality_score=quality_score
            )
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            # Return original text if compression fails
            return CompressionResult(
                original_text=original_text,
                compressed_text=original_text,
                compression_ratio=1.0,
                tokens_saved=0,
                quality_score=1.0
            )
    
    def _extract_preserved_content(self, text: str, preserve_math: bool) -> Tuple[List[str], str]:
        """Extract content that should be preserved during compression"""
        preserved = []
        remaining_text = text
        
        if preserve_math:
            for pattern in self.preserve_patterns:
                matches = re.finditer(pattern, remaining_text, re.DOTALL)
                for match in reversed(list(matches)):  # Process in reverse to maintain indices
                    preserved.append(match.group(0))
                    # Replace with placeholder
                    placeholder = f"__PRESERVED_{len(preserved)}__"
                    remaining_text = remaining_text[:match.start()] + placeholder + remaining_text[match.end():]
        
        return preserved, remaining_text
    
    async def _intelligent_compression(self, text: str, target_ratio: float) -> str:
        """Use Qwen for intelligent text compression"""
        
        if len(text.split()) < 50:  # Don't compress very short text
            return text
        
        target_words = int(len(text.split()) * target_ratio)
        
        prompt = f"""Compress the following text to approximately {target_words} words while preserving:
1. Key technical concepts and terminology
2. Important numerical data and statistics
3. Critical findings and conclusions
4. Essential logical connections

Remove:
- Redundant phrases and filler words
- Verbose explanations that don't add value
- Repetitive content
- Unnecessary transition phrases

Original text ({len(text.split())} words):
{text}

Compressed version (~{target_words} words):"""

        try:
            response = await self._query_qwen(prompt)
            
            # Extract the compressed text (everything after the prompt)
            compressed = response.strip()
            
            # Validate compression
            if len(compressed.split()) > len(text.split()) * 0.9:
                # If compression didn't work well, fall back to simple compression
                return self._simple_compression(text, target_ratio)
            
            return compressed
            
        except Exception as e:
            logger.warning(f"Intelligent compression failed: {e}")
            return self._simple_compression(text, target_ratio)
    
    def _simple_compression(self, text: str, target_ratio: float) -> str:
        """Fallback simple compression method"""
        
        # Remove common filler patterns
        compressed = text
        for pattern in self.removable_patterns:
            compressed = re.sub(pattern, '', compressed, flags=re.IGNORECASE)
        
        # Split into sentences and take the most important ones
        sentences = [s.strip() for s in compressed.split('.') if s.strip()]
        target_sentences = max(1, int(len(sentences) * target_ratio))
        
        # Simple heuristic: keep longer sentences (often more informative)
        scored_sentences = [(len(s), s) for s in sentences]
        scored_sentences.sort(reverse=True)
        
        selected_sentences = [s for _, s in scored_sentences[:target_sentences]]
        
        return '. '.join(selected_sentences) + '.'
    
    def _recombine_content(self, preserved: List[str], compressed_main: str) -> str:
        """Recombine preserved content with compressed text"""
        result = compressed_main
        
        # Replace placeholders with preserved content
        for i, content in enumerate(preserved, 1):
            placeholder = f"__PRESERVED_{i}__"
            result = result.replace(placeholder, content)
        
        return result
    
    async def _estimate_quality(self, original: str, compressed: str) -> float:
        """Estimate compression quality (simplified)"""
        # Simple quality metrics
        original_sentences = len([s for s in original.split('.') if s.strip()])
        compressed_sentences = len([s for s in compressed.split('.') if s.strip()])
        
        # Quality based on sentence preservation ratio
        if original_sentences == 0:
            return 1.0
        
        sentence_ratio = compressed_sentences / original_sentences
        
        # Quality decreases as we lose too many sentences
        if sentence_ratio > 0.7:
            return 0.9  # High quality
        elif sentence_ratio > 0.5:
            return 0.7  # Medium quality
        elif sentence_ratio > 0.3:
            return 0.5  # Lower quality
        else:
            return 0.3  # Poor quality
    
    async def _query_qwen(self, prompt: str) -> str:
        """Query Qwen model for compression"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistent compression
                "top_p": 0.9,
                "max_tokens": 2000
            }
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get('response', '').strip()
        except Exception as e:
            logger.error(f"Qwen query failed: {e}")
            raise
    
    async def compress_chunks(self, chunks: List[Chunk], preserve_math: bool = True) -> List[CompressionResult]:
        """Compress multiple chunks"""
        results = []
        
        for chunk in chunks:
            result = await self.compress_chunk(chunk, preserve_math)
            results.append(result)
            
            # Log compression stats
            logger.info(f"Compressed chunk: {result.compression_ratio:.2f} ratio, "
                       f"{result.tokens_saved} tokens saved, "
                       f"quality: {result.quality_score:.2f}")
        
        return results


async def test_compression():
    """Test the compression system"""
    test_text = """
    Furthermore, it is important to note that the LongLLMLingua approach represents a significant advancement 
    in the field of prompt compression for large language models. The methodology, as described in the paper, 
    involves a sophisticated two-stage compression process. In the first stage, the system performs 
    coarse-grained compression by removing entire sentences that are deemed less important. Subsequently, 
    in the second stage, the approach applies fine-grained compression at the token level through an 
    iterative process. The mathematical formulation can be expressed as: $P(compression) = \\alpha \\cdot 
    importance_{sentence} + \\beta \\cdot relevance_{token}$ where $\\alpha$ and $\\beta$ are weighting 
    parameters. The experimental results demonstrate that this approach achieves compression ratios of up 
    to 20x while maintaining semantic integrity. As a result, this methodology has significant implications 
    for trading systems where computational efficiency is paramount.
    """
    
    chunk = Chunk(
        id="test",
        content=test_text,
        start_index=0,
        end_index=len(test_text),
        page_number=1
    )
    
    compressor = LLMLinguaCompressor(target_compression=0.6)
    result = await compressor.compress_chunk(chunk)
    
    print("🗜️  LLMLingua Compression Test")
    print("=" * 50)
    print(f"Original length: {len(result.original_text)} chars")
    print(f"Compressed length: {len(result.compressed_text)} chars")
    print(f"Compression ratio: {result.compression_ratio:.2f}")
    print(f"Tokens saved: {result.tokens_saved}")
    print(f"Quality score: {result.quality_score:.2f}")
    print("\\nCompressed text:")
    print(result.compressed_text)


if __name__ == "__main__":
    asyncio.run(test_compression())