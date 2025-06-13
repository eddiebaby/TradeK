"""
OCR processor for scanned PDFs

This module handles optical character recognition for PDFs
that contain scanned images instead of text.
"""

import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os

import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class OCRProcessor:
    """
    Handles OCR processing for scanned PDFs.
    
    This class:
    1. Detects if a PDF needs OCR
    2. Converts PDF pages to images
    3. Applies image preprocessing for better OCR
    4. Extracts text using Tesseract
    """
    
    def __init__(self, 
                 language: str = 'eng',
                 dpi: int = 300,
                 thread_workers: int = 4):
        """
        Initialize OCR processor.
        
        Args:
            language: Tesseract language code
            dpi: DPI for PDF to image conversion
            thread_workers: Number of parallel OCR workers
        """
        self.language = language
        self.dpi = dpi
        self.thread_workers = thread_workers
        
        # Verify Tesseract installation
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            raise RuntimeError(f"Tesseract not found: {e}")
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=thread_workers)
        
        logger.info(f"OCR processor initialized with {thread_workers} workers")
    
    async def needs_ocr(self, pdf_path: Path, sample_pages: int = 3) -> bool:
        """
        Detect if a PDF needs OCR.
        
        This checks a sample of pages to see if they contain
        extractable text or are scanned images.
        
        Args:
            pdf_path: Path to PDF file
            sample_pages: Number of pages to sample
            
        Returns:
            True if OCR is needed
        """
        try:
            import PyPDF2
            
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = min(len(reader.pages), sample_pages)
                
                text_found = False
                for i in range(total_pages):
                    page_text = reader.pages[i].extract_text()
                    # Check if meaningful text exists
                    if page_text and len(page_text.strip()) > 50:
                        words = page_text.split()
                        # Check for actual words, not just garbage characters
                        if len(words) > 10 and any(len(w) > 3 for w in words):
                            text_found = True
                            break
                
                logger.debug(f"OCR needed for {pdf_path.name}: {not text_found}")
                return not text_found
                
        except Exception as e:
            logger.warning(f"Error checking PDF for OCR: {e}")
            # If we can't determine, assume OCR is needed
            return True
    
    async def process_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Process a scanned PDF using OCR.
        
        This is the main entry point for OCR processing.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of page dictionaries with extracted text
        """
        logger.info(f"Starting OCR processing for: {pdf_path.name}")
        
        # Convert PDF to images
        logger.debug("Converting PDF to images...")
        images = await self._pdf_to_images(pdf_path)
        
        if not images:
            logger.error("Failed to convert PDF to images")
            return []
        
        logger.info(f"Converted {len(images)} pages to images")
        
        # Process images in parallel
        logger.debug("Running OCR on images...")
        pages = await self._process_images_parallel(images)
        
        # Cleanup temporary images
        for img_path in images:
            try:
                os.remove(img_path)
            except OSError:
                pass
        
        logger.info(f"OCR completed: extracted text from {len(pages)} pages")
        return pages
    
    async def _pdf_to_images(self, pdf_path: Path) -> List[str]:
        """Convert PDF pages to images."""
        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Convert PDF to images
                images = await asyncio.to_thread(
                    convert_from_path,
                    pdf_path,
                    dpi=self.dpi,
                    output_folder=temp_dir,
                    fmt='png',
                    thread_count=self.thread_workers
                )
                
                # Save images and return paths
                image_paths = []
                for i, image in enumerate(images):
                    img_path = temp_path / f"page_{i+1:04d}.png"
                    image.save(img_path, 'PNG')
                    image_paths.append(str(img_path))
                
                # Move to persistent temp location
                persistent_paths = []
                for img_path in image_paths:
                    new_path = Path(tempfile.gettempdir()) / Path(img_path).name
                    Path(img_path).rename(new_path)
                    persistent_paths.append(str(new_path))
                
                return persistent_paths
                
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            return []
    
    async def _process_images_parallel(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple images in parallel."""
        # Create tasks for parallel processing
        tasks = []
        for i, img_path in enumerate(image_paths):
            task = asyncio.create_task(self._process_single_image(img_path, i + 1))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out errors
        pages = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing page {i+1}: {result}")
                pages.append({
                    'page_number': i + 1,
                    'text': '',
                    'word_count': 0,
                    'confidence': 0.0,
                    'error': str(result)
                })
            else:
                pages.append(result)
        
        return pages
    
    async def _process_single_image(self, image_path: str, page_number: int) -> Dict[str, Any]:
        """Process a single image with OCR."""
        try:
            # Load and preprocess image
            preprocessed = await asyncio.to_thread(
                self._preprocess_image, image_path
            )
            
            # Run OCR
            result = await asyncio.to_thread(
                self._run_tesseract, preprocessed
            )
            
            # Clean up text
            text = self._clean_ocr_text(result['text'])
            
            return {
                'page_number': page_number,
                'text': text,
                'word_count': len(text.split()),
                'char_count': len(text),
                'confidence': result['confidence'],
                'preprocessing': result['preprocessing']
            }
            
        except Exception as e:
            logger.error(f"Error in OCR for page {page_number}: {e}")
            raise
    
    def _preprocess_image(self, image_path: str) -> Dict[str, Any]:
        """
        Preprocess image for better OCR results.
        
        This applies various image processing techniques to improve
        OCR accuracy on scanned documents.
        """
        # Load image
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply different preprocessing based on image characteristics
        preprocessing_applied = []
        
        # Check if image is too dark or too light
        mean_brightness = np.mean(gray)
        if mean_brightness < 100:
            # Image is dark, apply histogram equalization
            gray = cv2.equalizeHist(gray)
            preprocessing_applied.append('histogram_equalization')
        elif mean_brightness > 200:
            # Image is too bright, adjust gamma
            gray = self._adjust_gamma(gray, 0.7)
            preprocessing_applied.append('gamma_correction')
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        preprocessing_applied.append('denoising')
        
        # Threshold to get binary image
        # Try adaptive thresholding for better results on uneven lighting
        binary = cv2.adaptiveThreshold(
            denoised, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        preprocessing_applied.append('adaptive_threshold')
        
        # Deskew if needed
        angle = self._detect_skew(binary)
        if abs(angle) > 0.5:
            binary = self._rotate_image(binary, angle)
            preprocessing_applied.append(f'deskew_{angle:.1f}')
        
        # Remove noise with morphological operations
        kernel = np.ones((1, 1), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        preprocessing_applied.append('morphological_ops')
        
        return {
            'image': binary,
            'preprocessing': preprocessing_applied
        }
    
    def _adjust_gamma(self, image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """Adjust image gamma for brightness correction."""
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(0, 256)
        ]).astype("uint8")
        
        return cv2.LUT(image, table)
    
    def _detect_skew(self, image: np.ndarray) -> float:
        """Detect skew angle of scanned page."""
        # Find all white pixels
        coords = np.column_stack(np.where(image > 0))
        
        # Find minimum area rectangle
        if len(coords) > 100:
            angle = cv2.minAreaRect(coords)[-1]
            
            # Adjust angle
            if angle < -45:
                angle = 90 + angle
            elif angle > 45:
                angle = angle - 90
                
            return angle
        
        return 0.0
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image to correct skew."""
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotate image
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
    
    def _run_tesseract(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """Run Tesseract OCR on preprocessed image."""
        image = preprocessed['image']
        
        # Configure Tesseract
        config = r'--oem 3 --psm 3'  # Use best OCR engine mode and automatic page segmentation
        
        # Run OCR with confidence scores
        data = pytesseract.image_to_data(
            image,
            lang=self.language,
            config=config,
            output_type=pytesseract.Output.DICT
        )
        
        # Extract text and calculate average confidence
        words = []
        confidences = []
        
        for i, word in enumerate(data['text']):
            if word.strip():
                words.append(word)
                conf = int(data['conf'][i])
                if conf > 0:  # -1 means no confidence available
                    confidences.append(conf)
        
        text = ' '.join(words)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            'text': text,
            'confidence': avg_confidence,
            'preprocessing': preprocessed['preprocessing']
        }
    
    def _clean_ocr_text(self, text: str) -> str:
        """
        Clean OCR text to fix common errors.
        
        OCR often produces artifacts that need cleaning.
        """
        if not text:
            return ''
        
        # Fix common OCR errors
        replacements = {
            ' ,': ',',
            ' .': '.',
            ' ;': ';',
            ' :': ':',
            ' !': '!',
            ' ?': '?',
            '  ': ' ',  # Multiple spaces
            '\n\n\n': '\n\n',  # Multiple newlines
            '|': 'I',  # Common I/| confusion
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char.isspace())
        
        # Fix quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()


async def test_ocr_processor():
    """Test the OCR processor"""
    processor = OCRProcessor()
    
    # Create a simple test image with text
    test_image = Path("data/test_ocr.png")
    
    if test_image.exists():
        # Process single image
        result = await processor._process_single_image(str(test_image), 1)
        
        print(f"OCR Result:")
        print(f"Confidence: {result['confidence']:.2f}%")
        print(f"Word count: {result['word_count']}")
        print(f"Text preview: {result['text'][:200]}...")
    else:
        print("Please create a test image with text to test OCR")


if __name__ == "__main__":
    asyncio.run(test_ocr_processor())