"""
Content analyzer for detecting and extracting special content

This module identifies code blocks, formulas, tables, and other
structured content within text.
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ContentType(Enum):
    """Types of special content"""
    CODE = "code"
    FORMULA = "formula"
    TABLE = "table"
    DIAGRAM = "diagram"
    QUOTE = "quote"
    REFERENCE = "reference"

@dataclass
class ContentRegion:
    """Represents a region of special content"""
    content_type: ContentType
    start: int
    end: int
    text: str
    metadata: Dict[str, Any]
    confidence: float

class ContentAnalyzer:
    """
    Analyzes text to identify special content regions.
    
    This is crucial for algorithmic trading books which contain:
    - Code snippets (Python, C++, R, etc.)
    - Mathematical formulas (pricing models, statistics)
    - Data tables (performance metrics, parameters)
    - Trading strategies and rules
    """
    
    def __init__(self):
        """Initialize content analyzer"""
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
        # Programming language indicators
        self.code_indicators = {
            'python': [
                'def ', 'class ', 'import ', 'from ', 'if __name__',
                'print(', 'return ', 'for ', 'while ', 'lambda ',
                'np.', 'pd.', 'plt.', 'self.'
            ],
            'cpp': [
                '#include', 'void ', 'int main', 'std::', 'namespace',
                'template<', 'public:', 'private:', 'return 0;'
            ],
            'r': [
                '<-', '%%', 'function(', 'library(', 'data.frame',
                'ggplot', 'summary('
            ],
            'sql': [
                'SELECT', 'FROM', 'WHERE', 'JOIN', 'GROUP BY',
                'ORDER BY', 'INSERT INTO', 'CREATE TABLE'
            ]
        }
        
        # Math/formula indicators
        self.math_indicators = [
            '=', '∑', '∏', '∫', '∂', '∇', '√', '≈', '≠', '≤', '≥',
            'alpha', 'beta', 'gamma', 'sigma', 'delta', 'theta',
            'E[', 'Var(', 'Cov(', 'P(', 'N(', 'log(', 'exp(',
            'dx', 'dt', 'df'
        ]
    
    def _compile_patterns(self):
        """Compile regex patterns"""
        # Code block patterns
        self.code_block_pattern = re.compile(
            r'```(?P<lang>\w*)\n(?P<code>.*?)```|'
            r'^(?P<indent_code>(?:    |\t).*?)$',
            re.MULTILINE | re.DOTALL
        )
        
        # LaTeX formula patterns
        self.latex_pattern = re.compile(
            r'\$\$(?P<display>.*?)\$\$|'
            r'\$(?P<inline>[^\$]+)\$|'
            r'\\begin\{(?P<env>equation|align|gather)\*?\}(?P<content>.*?)\\end\{\3\*?\}',
            re.DOTALL
        )
        
        # Table patterns
        self.table_pattern = re.compile(
            r'(?P<table>(?:.*?\|.*?\n)+)',
            re.MULTILINE
        )
        
        # Trading strategy pattern (custom for finance books)
        self.strategy_pattern = re.compile(
            r'(?:Strategy|Rule|Signal|Condition):\s*\n(?P<content>(?:[-•*]\s*.*?\n)+)',
            re.MULTILINE | re.IGNORECASE
        )
    
    def analyze_text(self, text: str) -> List[ContentRegion]:
        """
        Analyze text and identify all special content regions.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of ContentRegion objects
        """
        regions = []
        
        # Find code blocks
        regions.extend(self._find_code_blocks(text))
        
        # Find formulas
        regions.extend(self._find_formulas(text))
        
        # Find tables
        regions.extend(self._find_tables(text))
        
        # Find trading strategies
        regions.extend(self._find_strategies(text))
        
        # Sort by start position and merge overlapping
        regions = self._merge_overlapping_regions(regions)
        
        return regions
    
    def _find_code_blocks(self, text: str) -> List[ContentRegion]:
        """Find code blocks in text"""
        regions = []
        
        # Look for explicit code blocks (```)
        for match in self.code_block_pattern.finditer(text):
            if match.group('code'):
                lang = match.group('lang') or self._detect_language(match.group('code'))
                regions.append(ContentRegion(
                    content_type=ContentType.CODE,
                    start=match.start(),
                    end=match.end(),
                    text=match.group('code'),
                    metadata={'language': lang},
                    confidence=0.95
                ))
        
        # Look for indented code blocks
        lines = text.split('\n')
        in_code_block = False
        code_start = 0
        code_lines = []
        
        for i, line in enumerate(lines):
            if line.startswith(('    ', '\t')) and line.strip():
                if not in_code_block:
                    in_code_block = True
                    code_start = sum(len(l) + 1 for l in lines[:i])
                code_lines.append(line[4:] if line.startswith('    ') else line[1:])
            else:
                if in_code_block and len(code_lines) > 2:
                    code_text = '\n'.join(code_lines)
                    lang = self._detect_language(code_text)
                    
                    regions.append(ContentRegion(
                        content_type=ContentType.CODE,
                        start=code_start,
                        end=code_start + len(code_text),
                        text=code_text,
                        metadata={'language': lang, 'indented': True},
                        confidence=0.8
                    ))
                
                in_code_block = False
                code_lines = []
        
        # Also look for inline code patterns
        regions.extend(self._find_inline_code(text))
        
        return regions
    
    def _find_inline_code(self, text: str) -> List[ContentRegion]:
        """Find inline code snippets"""
        regions = []
        
        # Look for function calls and code-like patterns
        patterns = [
            (r'`([^`]+)`', 0.9),  # Backtick code
            (r'\b(\w+\.\w+\([^)]*\))', 0.7),  # Method calls
            (r'\b((?:def|class|function|var|let|const)\s+\w+)', 0.8),  # Declarations
        ]
        
        for pattern, confidence in patterns:
            for match in re.finditer(pattern, text):
                code = match.group(1)
                if len(code) > 3:  # Skip very short matches
                    regions.append(ContentRegion(
                        content_type=ContentType.CODE,
                        start=match.start(),
                        end=match.end(),
                        text=code,
                        metadata={'inline': True},
                        confidence=confidence
                    ))
        
        return regions
    
    def _detect_language(self, code: str) -> str:
        """Detect programming language of code snippet"""
        code_lower = code.lower()
        
        # Count indicators for each language
        scores = {}
        for lang, indicators in self.code_indicators.items():
            score = sum(1 for ind in indicators if ind.lower() in code_lower)
            if score > 0:
                scores[lang] = score
        
        # Return language with highest score
        if scores:
            return max(scores, key=scores.get)
        
        # Check for shell/bash
        if any(code.startswith(prefix) for prefix in ['$', '>', '#!']):
            return 'bash'
        
        return 'unknown'
    
    def _find_formulas(self, text: str) -> List[ContentRegion]:
        """Find mathematical formulas"""
        regions = []
        
        # LaTeX formulas
        for match in self.latex_pattern.finditer(text):
            formula_text = (
                match.group('display') or 
                match.group('inline') or 
                match.group('content')
            )
            
            if formula_text:
                regions.append(ContentRegion(
                    content_type=ContentType.FORMULA,
                    start=match.start(),
                    end=match.end(),
                    text=formula_text,
                    metadata={
                        'format': 'latex',
                        'display': bool(match.group('display') or match.group('env'))
                    },
                    confidence=0.95
                ))
        
        # Look for non-LaTeX math expressions
        # This is more heuristic-based
        math_pattern = re.compile(
            r'(?:^|\s)([A-Za-z]+\s*=\s*[^.!?]+?)(?:[.!?]|\s*$)',
            re.MULTILINE
        )
        
        for match in math_pattern.finditer(text):
            expr = match.group(1)
            # Check if it contains math indicators
            if any(ind in expr for ind in self.math_indicators):
                regions.append(ContentRegion(
                    content_type=ContentType.FORMULA,
                    start=match.start(1),
                    end=match.end(1),
                    text=expr,
                    metadata={'format': 'plain'},
                    confidence=0.7
                ))
        
        return regions
    
    def _find_tables(self, text: str) -> List[ContentRegion]:
        """Find tables in text"""
        regions = []
        
        # Look for ASCII tables with pipes
        for match in self.table_pattern.finditer(text):
            table_text = match.group('table')
            rows = table_text.strip().split('\n')
            
            # Verify it's actually a table (multiple rows with similar structure)
            if len(rows) >= 2:
                pipe_counts = [row.count('|') for row in rows]
                if pipe_counts and all(c > 0 for c in pipe_counts):
                    # Parse table structure
                    headers = self._parse_table_row(rows[0])
                    
                    regions.append(ContentRegion(
                        content_type=ContentType.TABLE,
                        start=match.start(),
                        end=match.end(),
                        text=table_text,
                        metadata={
                            'rows': len(rows),
                            'columns': len(headers),
                            'headers': headers
                        },
                        confidence=0.85
                    ))
        
        # Also look for whitespace-aligned tables
        regions.extend(self._find_whitespace_tables(text))
        
        return regions
    
    def _find_whitespace_tables(self, text: str) -> List[ContentRegion]:
        """Find tables aligned with whitespace"""
        regions = []
        lines = text.split('\n')
        
        # Look for consecutive lines with multiple whitespace-separated columns
        potential_table = []
        table_start_line = 0
        
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) >= 3 and not line.strip().startswith(('#', '//', '--')):
                if not potential_table:
                    table_start_line = i
                potential_table.append(line)
            else:
                if len(potential_table) >= 3:
                    # Verify it's a table by checking alignment
                    if self._is_aligned_table(potential_table):
                        table_text = '\n'.join(potential_table)
                        start = sum(len(l) + 1 for l in lines[:table_start_line])
                        
                        regions.append(ContentRegion(
                            content_type=ContentType.TABLE,
                            start=start,
                            end=start + len(table_text),
                            text=table_text,
                            metadata={
                                'rows': len(potential_table),
                                'whitespace_aligned': True
                            },
                            confidence=0.7
                        ))
                
                potential_table = []
        
        return regions
    
    def _is_aligned_table(self, lines: List[str]) -> bool:
        """Check if lines form an aligned table"""
        # Simple heuristic: check if columns are roughly aligned
        if len(lines) < 3:
            return False
        
        # Get positions of whitespace for each line
        positions = []
        for line in lines:
            pos = []
            for i, char in enumerate(line):
                if char.isspace() and (i == 0 or not line[i-1].isspace()):
                    pos.append(i)
            positions.append(pos)
        
        # Check if positions are similar across lines
        if not positions:
            return False
        
        # At least 2 aligned columns
        return len(positions[0]) >= 2
    
    def _parse_table_row(self, row: str) -> List[str]:
        """Parse a table row and extract column headers"""
        # Remove leading/trailing pipes and split
        row = row.strip().strip('|')
        return [cell.strip() for cell in row.split('|')]
    
    def _find_strategies(self, text: str) -> List[ContentRegion]:
        """Find trading strategies and rules"""
        regions = []
        
        for match in self.strategy_pattern.finditer(text):
            content = match.group('content')
            
            regions.append(ContentRegion(
                content_type=ContentType.REFERENCE,
                start=match.start(),
                end=match.end(),
                text=content,
                metadata={'type': 'strategy'},
                confidence=0.8
            ))
        
        return regions
    
    def _merge_overlapping_regions(self, regions: List[ContentRegion]) -> List[ContentRegion]:
        """Merge overlapping content regions"""
        if not regions:
            return []
        
        # Sort by start position
        regions.sort(key=lambda r: r.start)
        
        merged = []
        current = regions[0]
        
        for next_region in regions[1:]:
            # Check for overlap
            if next_region.start <= current.end:
                # Merge regions, keeping higher confidence
                if next_region.confidence > current.confidence:
                    # Replace with higher confidence region
                    current = ContentRegion(
                        content_type=next_region.content_type,
                        start=min(current.start, next_region.start),
                        end=max(current.end, next_region.end),
                        text=next_region.text,
                        metadata=next_region.metadata,
                        confidence=next_region.confidence
                    )
                else:
                    # Extend current region
                    current = ContentRegion(
                        content_type=current.content_type,
                        start=current.start,
                        end=max(current.end, next_region.end),
                        text=current.text,
                        metadata=current.metadata,
                        confidence=current.confidence
                    )
            else:
                # No overlap, add current and move to next
                merged.append(current)
                current = next_region
        
        # Add the last region
        merged.append(current)
        
        return merged
    
    def extract_special_content(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract and categorize all special content.
        
        Returns:
            Dictionary with content types as keys and lists of content as values
        """
        regions = self.analyze_text(text)
        
        result = {
            'code': [],
            'formulas': [],
            'tables': [],
            'strategies': [],
            'other': []
        }
        
        for region in regions:
            content_info = {
                'text': region.text,
                'confidence': region.confidence,
                'metadata': region.metadata,
                'position': (region.start, region.end)
            }
            
            if region.content_type == ContentType.CODE:
                result['code'].append(content_info)
            elif region.content_type == ContentType.FORMULA:
                result['formulas'].append(content_info)
            elif region.content_type == ContentType.TABLE:
                result['tables'].append(content_info)
            elif region.content_type == ContentType.REFERENCE and region.metadata.get('type') == 'strategy':
                result['strategies'].append(content_info)
            else:
                result['other'].append(content_info)
        
        return result


# Test content analyzer
def test_content_analyzer():
    """Test the content analyzer"""
    analyzer = ContentAnalyzer()
    
    # Sample text with various content types
    sample_text = """
    This is a sample trading book chapter.
    
    Here's a Python code example:
    
    ```python
    def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / np.std(excess_returns)
    ```
    
    The Sharpe ratio formula is:
    
    $$S = \\frac{E[R_p] - R_f}{\\sigma_p}$$
    
    Here's a performance table:
    
    | Strategy | Return | Volatility | Sharpe |
    |----------|--------|------------|--------|
    | Long     | 12.5%  | 15.2%     | 0.82   |
    | Short    | 8.3%   | 12.1%     | 0.69   |
    
    Strategy Rules:
    - Buy when RSI < 30
    - Sell when RSI > 70
    - Stop loss at 2%
    """
    
    # Analyze content
    special_content = analyzer.extract_special_content(sample_text)
    
    print("Content Analysis Results:")
    print(f"Code blocks: {len(special_content['code'])}")
    print(f"Formulas: {len(special_content['formulas'])}")
    print(f"Tables: {len(special_content['tables'])}")
    print(f"Strategies: {len(special_content['strategies'])}")
    
    # Show details
    for content_type, items in special_content.items():
        if items:
            print(f"\n{content_type.upper()}:")
            for item in items:
                print(f"  Confidence: {item['confidence']:.2f}")
                print(f"  Text preview: {item['text'][:100]}...")
                print(f"  Metadata: {item['metadata']}")


if __name__ == "__main__":
    test_content_analyzer()