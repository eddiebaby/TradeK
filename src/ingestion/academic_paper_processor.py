"""
Academic Paper Processor with LaTeX Formula Recognition

This module specializes in processing academic papers with:
- Mathematical formula extraction and preservation
- LaTeX equation recognition and conversion
- Academic structure recognition (Abstract, Methods, Results, etc.)
- Integration with Qwen2.5-Coder for formula-to-code conversion
"""

import asyncio
import io
import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import cv2
import fitz  # PyMuPDF for better PDF handling
import numpy as np
import pytesseract
from PIL import Image

from ..core.models import Document, Chunk
from .pdf_parser import PDFParser
from .ocr_processor import OCRProcessor

logger = logging.getLogger(__name__)


@dataclass
class MathematicalFormula:
    """Represents an extracted mathematical formula"""
    latex_code: str
    text_description: str
    page_number: int
    position: Tuple[float, float, float, float]  # x1, y1, x2, y2
    context: str  # Surrounding text for context
    complexity_score: float
    python_implementation: Optional[str] = None


@dataclass
class AcademicSection:
    """Represents a section of an academic paper"""
    title: str
    content: str
    section_type: str  # abstract, introduction, methods, results, conclusion
    page_range: Tuple[int, int]
    formulas: List[MathematicalFormula]
    key_concepts: List[str]


class AcademicPaperProcessor:
    """
    Enhanced processor for academic papers with mathematical content.
    
    Features:
    - LaTeX formula detection and extraction
    - Academic structure recognition
    - Mathematical symbol OCR
    - Formula-to-Python conversion via Qwen2.5-Coder
    """
    
    def __init__(self):
        """Initialize the academic paper processor"""
        self.pdf_parser = PDFParser(enable_ocr=True)
        self.ocr_processor = OCRProcessor()
        
        # LaTeX patterns for formula detection
        self.latex_patterns = [
            r'\$\$.*?\$\$',  # Display math
            r'\$.*?\$',      # Inline math
            r'\\begin\{equation\}.*?\\end\{equation\}',
            r'\\begin\{align\}.*?\\end\{align\}',
            r'\\begin\{eqnarray\}.*?\\end\{eqnarray\}',
            r'\\[.*?\\]',    # LaTeX display math
            r'\\\(.*?\\\)',  # LaTeX inline math
        ]
        
        # Academic section patterns
        self.section_patterns = {
            'abstract': r'(?i)\babstract\b',
            'introduction': r'(?i)\b(introduction|intro)\b',
            'methods': r'(?i)\b(methods?|methodology|approach)\b',
            'results': r'(?i)\b(results?|findings|experiments?)\b',
            'conclusion': r'(?i)\b(conclusion|summary|discussion)\b',
            'references': r'(?i)\b(references?|bibliography)\b'
        }
        
        # Mathematical symbols for enhanced OCR
        self.math_symbols = [
            'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ',
            'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω',
            '∑', '∏', '∫', '∂', '∇', '∆', '∞', '±', '≤', '≥', '≈', '≡',
            '∈', '∉', '⊂', '⊃', '∪', '∩', '→', '←', '↔', '⇒', '⇔'
        ]
    
    async def process_paper(self, file_path: Path) -> Document:
        """
        Process an academic paper with formula extraction.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Document with enhanced academic content
        """
        logger.info(f"Processing academic paper: {file_path.name}")
        
        try:
            # Extract basic document info
            doc = await self._extract_basic_content(file_path)
            
            # Extract mathematical formulas
            formulas = await self._extract_formulas(file_path)
            
            # Identify academic sections
            sections = await self._identify_sections(doc.content, formulas)
            
            # Enhance with mathematical analysis
            doc = await self._enhance_with_math_analysis(doc, sections, formulas)
            
            logger.info(f"Successfully processed paper: {len(formulas)} formulas, {len(sections)} sections")
            return doc
            
        except Exception as e:
            logger.error(f"Error processing academic paper {file_path}: {e}")
            raise
    
    async def _extract_basic_content(self, file_path: Path) -> Document:
        """Extract basic content using existing PDF parser"""
        return await self.pdf_parser.parse(file_path)
    
    async def _extract_formulas(self, file_path: Path) -> List[MathematicalFormula]:
        """
        Extract mathematical formulas from the PDF.
        
        Uses multiple approaches:
        1. Text-based LaTeX pattern matching
        2. Image-based formula detection
        3. OCR with mathematical symbol recognition
        """
        formulas = []
        
        # Open PDF with PyMuPDF for better handling
        doc = fitz.open(file_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Extract text and search for LaTeX patterns
            text = page.get_text()
            text_formulas = self._extract_latex_from_text(text, page_num)
            formulas.extend(text_formulas)
            
            # Extract images and detect formulas
            image_formulas = await self._extract_formulas_from_images(page, page_num)
            formulas.extend(image_formulas)
        
        doc.close()
        return formulas
    
    def _extract_latex_from_text(self, text: str, page_num: int) -> List[MathematicalFormula]:
        """Extract LaTeX formulas from text using pattern matching"""
        formulas = []
        
        for pattern in self.latex_patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            
            for match in matches:
                latex_code = match.group(0)
                
                # Get surrounding context
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end]
                
                # Calculate complexity score based on LaTeX complexity
                complexity = self._calculate_formula_complexity(latex_code)
                
                formula = MathematicalFormula(
                    latex_code=latex_code,
                    text_description=self._latex_to_description(latex_code),
                    page_number=page_num,
                    position=(0, 0, 0, 0),  # Will be refined later
                    context=context,
                    complexity_score=complexity
                )
                
                formulas.append(formula)
        
        return formulas
    
    async def _extract_formulas_from_images(self, page, page_num: int) -> List[MathematicalFormula]:
        """Extract formulas from image regions in the PDF"""
        formulas = []
        
        try:
            # Get page images
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                # Extract image
                xref = img[0]
                pix = fitz.Pixmap(page.parent, xref)
                
                if pix.n < 5:  # Skip if not suitable for OCR
                    # Convert to PIL Image for processing
                    img_data = pix.tobytes("ppm")
                    pil_image = Image.open(io.BytesIO(img_data))
                    
                    # Check if image contains mathematical content
                    if await self._contains_mathematical_content(pil_image):
                        # Enhanced OCR for mathematical symbols
                        formula_text = await self._ocr_mathematical_content(pil_image)
                        
                        if formula_text.strip():
                            formula = MathematicalFormula(
                                latex_code=f"% Extracted from image {img_index}",
                                text_description=formula_text,
                                page_number=page_num,
                                position=(0, 0, 0, 0),
                                context="Image-based formula",
                                complexity_score=0.5
                            )
                            formulas.append(formula)
                
                pix = None  # Free memory
                
        except Exception as e:
            logger.warning(f"Error extracting formulas from images on page {page_num}: {e}")
        
        return formulas
    
    async def _contains_mathematical_content(self, image: Image.Image) -> bool:
        """Detect if an image likely contains mathematical formulas"""
        # Simple heuristic: check image dimensions and content
        width, height = image.size
        
        # Mathematical formulas are often wide and short, or contain specific patterns
        aspect_ratio = width / height
        
        # Convert to grayscale for analysis
        gray = image.convert('L')
        pixels = np.array(gray)
        
        # Check for high contrast (typical of mathematical notation)
        contrast = np.std(pixels)
        
        # Heuristic: likely mathematical if wide aspect ratio and high contrast
        return aspect_ratio > 2.0 and contrast > 50
    
    async def _ocr_mathematical_content(self, image: Image.Image) -> str:
        """Perform OCR specifically tuned for mathematical content"""
        try:
            # Configure Tesseract for mathematical symbols
            config = '--psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-*/=()[]{}αβγδεζηθικλμνξοπρστυφχψω∑∏∫∂∇∆∞±≤≥≈≡∈∉⊂⊃∪∩→←↔⇒⇔'
            
            # Convert PIL to cv2 format
            cv_image = cv2.cvtarray(np.array(image))
            
            # Preprocess for better OCR
            processed_image = self._preprocess_for_math_ocr(cv_image)
            
            # Perform OCR
            text = pytesseract.image_to_string(processed_image, config=config)
            
            return text.strip()
            
        except Exception as e:
            logger.warning(f"Error in mathematical OCR: {e}")
            return ""
    
    def _preprocess_for_math_ocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better mathematical OCR"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.medianBlur(enhanced, 3)
        
        # Threshold
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    async def _identify_sections(self, content: str, formulas: List[MathematicalFormula]) -> List[AcademicSection]:
        """Identify and extract academic paper sections"""
        sections = []
        
        # Split content by common section markers
        lines = content.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a section header
            section_type = self._identify_section_type(line)
            
            if section_type:
                # Save previous section
                if current_section:
                    section_formulas = self._get_formulas_in_range(
                        formulas, current_section['start_line'], len(current_content)
                    )
                    
                    section = AcademicSection(
                        title=current_section['title'],
                        content='\n'.join(current_content),
                        section_type=current_section['type'],
                        page_range=(0, 0),  # Will be calculated later
                        formulas=section_formulas,
                        key_concepts=self._extract_key_concepts('\n'.join(current_content))
                    )
                    sections.append(section)
                
                # Start new section
                current_section = {
                    'title': line,
                    'type': section_type,
                    'start_line': len(current_content)
                }
                current_content = []
            else:
                current_content.append(line)
        
        # Add final section
        if current_section:
            section_formulas = self._get_formulas_in_range(
                formulas, current_section['start_line'], len(current_content)
            )
            
            section = AcademicSection(
                title=current_section['title'],
                content='\n'.join(current_content),
                section_type=current_section['type'],
                page_range=(0, 0),
                formulas=section_formulas,
                key_concepts=self._extract_key_concepts('\n'.join(current_content))
            )
            sections.append(section)
        
        return sections
    
    def _identify_section_type(self, line: str) -> Optional[str]:
        """Identify the type of academic section from a line"""
        line_lower = line.lower()
        
        for section_type, pattern in self.section_patterns.items():
            if re.search(pattern, line_lower):
                return section_type
        
        return None
    
    def _get_formulas_in_range(self, formulas: List[MathematicalFormula], start: int, end: int) -> List[MathematicalFormula]:
        """Get formulas that appear within a text range"""
        # Simplified implementation - in practice, would need position mapping
        return formulas  # Return all for now
    
    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from academic text"""
        # Simple keyword extraction - could be enhanced with NLP
        concepts = []
        
        # Look for capitalized terms and technical phrases
        concept_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Capitalized phrases
            r'\b\w+(?:_\w+)+\b',  # Underscore terms
            r'\b\w*(?:algorithm|method|approach|technique|model)\b'  # Technical terms
        ]
        
        for pattern in concept_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts.extend(matches)
        
        # Remove duplicates and filter
        concepts = list(set([c for c in concepts if len(c) > 3]))
        return concepts[:20]  # Limit to top 20
    
    async def _enhance_with_math_analysis(self, doc: Document, sections: List[AcademicSection], formulas: List[MathematicalFormula]) -> Document:
        """Enhance document with mathematical analysis"""
        # Add formula metadata to document
        doc.metadata['formula_count'] = len(formulas)
        doc.metadata['section_count'] = len(sections)
        doc.metadata['complexity_score'] = np.mean([f.complexity_score for f in formulas]) if formulas else 0
        
        # Store formulas and sections as additional metadata
        doc.metadata['formulas'] = [
            {
                'latex': f.latex_code,
                'description': f.text_description,
                'page': f.page_number,
                'complexity': f.complexity_score
            }
            for f in formulas
        ]
        
        doc.metadata['sections'] = [
            {
                'title': s.title,
                'type': s.section_type,
                'key_concepts': s.key_concepts,
                'formula_count': len(s.formulas)
            }
            for s in sections
        ]
        
        return doc
    
    def _calculate_formula_complexity(self, latex_code: str) -> float:
        """Calculate complexity score for a LaTeX formula"""
        complexity = 0.0
        
        # Count complexity indicators
        complexity += latex_code.count('\\') * 0.1  # LaTeX commands
        complexity += latex_code.count('{') * 0.05  # Nested structures
        complexity += latex_code.count('_') * 0.02  # Subscripts
        complexity += latex_code.count('^') * 0.02  # Superscripts
        complexity += len(re.findall(r'\\[a-zA-Z]+', latex_code)) * 0.1  # Named commands
        
        # Special complexity markers
        if '\\int' in latex_code: complexity += 0.3  # Integrals
        if '\\sum' in latex_code: complexity += 0.2  # Summations
        if '\\prod' in latex_code: complexity += 0.2  # Products
        if '\\frac' in latex_code: complexity += 0.1  # Fractions
        
        return min(complexity, 1.0)  # Cap at 1.0
    
    def _latex_to_description(self, latex_code: str) -> str:
        """Convert LaTeX code to human-readable description"""
        # Simple conversion - could be enhanced with proper LaTeX parser
        description = latex_code
        
        # Remove LaTeX delimiters
        description = re.sub(r'\$+', '', description)
        description = re.sub(r'\\begin\{.*?\}', '', description)
        description = re.sub(r'\\end\{.*?\}', '', description)
        
        # Convert common LaTeX commands
        replacements = {
            r'\\alpha': 'alpha',
            r'\\beta': 'beta',
            r'\\gamma': 'gamma',
            r'\\delta': 'delta',
            r'\\sum': 'sum',
            r'\\int': 'integral',
            r'\\frac\{([^}]+)\}\{([^}]+)\}': r'(\1)/(\2)',
            r'_\{([^}]+)\}': r'_(\1)',
            r'\^\{([^}]+)\}': r'^(\1)',
        }
        
        for pattern, replacement in replacements.items():
            description = re.sub(pattern, replacement, description)
        
        return description.strip()