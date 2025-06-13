"""
Jupyter Notebook parser for TradeKnowledge

Extracts code, markdown, and outputs from .ipynb files.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import re
import asyncio

try:
    import nbformat
    NBFORMAT_AVAILABLE = True
except ImportError:
    NBFORMAT_AVAILABLE = False

logger = logging.getLogger(__name__)

class NotebookParser:
    """
    Parser for Jupyter Notebook files.
    
    Notebooks are valuable in trading as they often contain:
    - Strategy development and backtesting
    - Data analysis and visualization
    - Research notes and findings
    """
    
    def __init__(self):
        """Initialize notebook parser"""
        self.supported_extensions = ['.ipynb']
        
        # Patterns for identifying important cells
        self.patterns = {
            'strategy': re.compile(r'strategy|backtest|signal|entry|exit', re.I),
            'analysis': re.compile(r'analysis|performance|metrics|sharpe|returns', re.I),
            'model': re.compile(r'model|predict|forecast|machine learning|ml', re.I),
            'visualization': re.compile(r'plot|chart|visuali[sz]e|graph|figure', re.I)
        }
    
    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the file"""
        return file_path.suffix.lower() in self.supported_extensions and NBFORMAT_AVAILABLE
    
    async def parse_file_async(self, file_path: Path) -> Dict[str, Any]:
        """Async wrapper for parse_file"""
        return await asyncio.to_thread(self.parse_file, file_path)
    
    def parse_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse a Jupyter notebook and extract content.
        
        Args:
            file_path: Path to notebook file
            
        Returns:
            Dictionary with metadata and content
        """
        logger.info(f"Starting to parse notebook: {file_path}")
        
        result = {
            'metadata': {},
            'pages': [],  # We'll treat cells as pages
            'errors': []
        }
        
        if not NBFORMAT_AVAILABLE:
            result['errors'].append("nbformat package not available")
            return result
        
        try:
            # Read notebook
            with open(file_path, 'r', encoding='utf-8') as f:
                notebook = nbformat.read(f, as_version=4)
            
            # Extract metadata
            result['metadata'] = self._extract_metadata(notebook, file_path)
            
            # Extract content from cells
            result['pages'] = self._extract_cells(notebook)
            
            # Add statistics
            code_cells = sum(1 for p in result['pages'] if p['cell_type'] == 'code')
            markdown_cells = sum(1 for p in result['pages'] if p['cell_type'] == 'markdown')
            
            result['statistics'] = {
                'total_cells': len(result['pages']),
                'code_cells': code_cells,
                'markdown_cells': markdown_cells,
                'total_words': sum(p['word_count'] for p in result['pages']),
                'total_characters': sum(p['char_count'] for p in result['pages'])
            }
            
            logger.info(
                f"Successfully parsed notebook: {result['statistics']['total_cells']} cells, "
                f"{result['statistics']['total_words']} words"
            )
            
        except Exception as e:
            error_msg = f"Error parsing notebook: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result['errors'].append(error_msg)
        
        return result
    
    def _extract_metadata(self, notebook: Any, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from notebook"""
        metadata = {
            'title': file_path.stem.replace('_', ' ').replace('-', ' ').title(),
            'file_type': 'jupyter_notebook'
        }
        
        # Extract notebook metadata
        if hasattr(notebook, 'metadata') and notebook.metadata:
            nb_meta = notebook.metadata
            
            # Extract common fields
            if 'title' in nb_meta:
                metadata['title'] = nb_meta['title']
            if 'authors' in nb_meta:
                metadata['author'] = ', '.join(nb_meta['authors'])
            if 'description' in nb_meta:
                metadata['description'] = nb_meta['description']
            if 'kernelspec' in nb_meta:
                metadata['kernel'] = nb_meta['kernelspec'].get('display_name', 'Unknown')
            if 'language_info' in nb_meta:
                metadata['language'] = nb_meta['language_info'].get('name', 'python')
        
        return metadata
    
    def _extract_cells(self, notebook: Any) -> List[Dict[str, Any]]:
        """Extract content from notebook cells"""
        cells = []
        
        for i, cell in enumerate(notebook.cells):
            cell_data = {
                'page_number': i + 1,
                'cell_type': cell.cell_type,
                'text': '',
                'word_count': 0,
                'char_count': 0,
                'execution_count': getattr(cell, 'execution_count', None),
                'tags': [],
                'importance': 'normal'
            }
            
            # Extract cell content
            if hasattr(cell, 'source'):
                cell_data['text'] = cell.source
                cell_data['word_count'] = len(cell.source.split())
                cell_data['char_count'] = len(cell.source)
            
            # Extract tags from metadata
            if hasattr(cell, 'metadata') and cell.metadata:
                if 'tags' in cell.metadata:
                    cell_data['tags'] = cell.metadata['tags']
            
            # Add outputs for code cells
            if cell.cell_type == 'code' and hasattr(cell, 'outputs') and cell.outputs:
                output_text = []
                for output in cell.outputs:
                    if hasattr(output, 'text'):
                        output_text.append(output.text)
                    elif hasattr(output, 'data') and 'text/plain' in output.data:
                        output_text.append(output.data['text/plain'])
                
                if output_text:
                    cell_data['output'] = '\n'.join(output_text)
                    # Add output to main text for searching
                    cell_data['text'] += '\n\n# Output:\n' + cell_data['output']
                    cell_data['word_count'] = len(cell_data['text'].split())
                    cell_data['char_count'] = len(cell_data['text'])
            
            # Classify cell importance
            cell_data['importance'] = self._classify_cell_importance(cell_data['text'])
            
            cells.append(cell_data)
        
        return cells
    
    def _classify_cell_importance(self, text: str) -> str:
        """Classify cell importance based on content"""
        if not text:
            return 'low'
        
        # Check against patterns
        for category, pattern in self.patterns.items():
            if pattern.search(text):
                return 'high'
        
        # Long cells are usually more important
        if len(text) > 500:
            return 'medium'
        
        # Code cells with functions/classes
        if any(keyword in text for keyword in ['def ', 'class ', 'import ', 'from ']):
            return 'medium'
        
        return 'low'


# Test notebook parser
def test_notebook_parser():
    """Test the notebook parser"""
    if not NBFORMAT_AVAILABLE:
        print("nbformat not available - notebook parsing disabled")
        return
    
    parser = NotebookParser()
    
    # Test with a sample notebook file
    test_file = Path("data/books/sample.ipynb")
    
    if test_file.exists():
        result = parser.parse_file(test_file)
        
        print(f"Title: {result['metadata'].get('title', 'Unknown')}")
        print(f"Cells: {result['statistics']['total_cells']}")
        print(f"Code cells: {result['statistics']['code_cells']}")
        print(f"Words: {result['statistics']['total_words']}")
        
        # Show first cell sample
        if result['pages']:
            first_cell = result['pages'][0]
            sample = first_cell['text'][:200] + '...' if len(first_cell['text']) > 200 else first_cell['text']
            print(f"\nFirst cell ({first_cell['cell_type']}):\n{sample}")
    else:
        print(f"Test file not found: {test_file}")
        print("Please add a Jupyter notebook to test with")


if __name__ == "__main__":
    # Setup logging for testing
    logging.basicConfig(level=logging.DEBUG)
    test_notebook_parser()