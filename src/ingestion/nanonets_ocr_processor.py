"""
Enhanced OCR processor using Nanonets-OCR-s model for intelligent document processing.
Converts PDF pages to structured markdown with semantic understanding.
"""

import io
import logging
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

logger = logging.getLogger(__name__)


class NanonetsOCRProcessor:
    """Advanced OCR processor using Nanonets-OCR-s model."""

    def __init__(self, model_name: str = "nanonets/Nanonets-OCR-s"):
        """Initialize the Nanonets OCR processor."""
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the Nanonets OCR model and processors."""
        try:
            logger.info(f"Loading Nanonets OCR model: {self.model_name}")

            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
                device_map="auto" if torch.cuda.is_available() else None,
            )

            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            logger.info("Nanonets OCR model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Nanonets OCR model: {e}")
            raise

    def extract_text_from_pdf(self, pdf_path: Path) -> list[dict[str, Any]]:
        """
        Extract structured text from PDF using advanced OCR.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of dictionaries containing page content and metadata
        """
        results = []

        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)

            logger.info(f"Processing {total_pages} pages from {pdf_path.name}")

            for page_num in range(total_pages):
                page = doc[page_num]

                # Convert page to image
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(2.0, 2.0)
                )  # 2x scaling for better quality
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))

                # Process with Nanonets OCR
                markdown_content = self._process_image_to_markdown(image)

                page_result = {
                    "page_number": page_num + 1,
                    "content": markdown_content,
                    "metadata": {
                        "pdf_path": str(pdf_path),
                        "page_count": total_pages,
                        "ocr_model": self.model_name,
                        "image_size": image.size,
                    },
                }

                results.append(page_result)

                if page_num % 10 == 0:
                    logger.info(f"Processed {page_num + 1}/{total_pages} pages")

            doc.close()
            logger.info(f"Successfully processed all {total_pages} pages")

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise

        return results

    def _process_image_to_markdown(self, image: Image.Image) -> str:
        """
        Convert image to structured markdown using Nanonets OCR.

        Args:
            image: PIL Image object

        Returns:
            Structured markdown content
        """
        try:
            # Prepare inputs for the model
            inputs = self.processor(images=image, return_tensors="pt")

            # Move to appropriate device
            if torch.cuda.is_available():
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate markdown content
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode the generated content
            generated_text = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            return generated_text.strip()

        except Exception as e:
            logger.error(f"Error processing image to markdown: {e}")
            return f"Error processing image: {str(e)}"

    def extract_text_from_images(self, image_paths: list[Path]) -> list[dict[str, Any]]:
        """
        Extract structured text from multiple images.

        Args:
            image_paths: List of paths to image files

        Returns:
            List of dictionaries containing image content and metadata
        """
        results = []

        for idx, image_path in enumerate(image_paths):
            try:
                image = Image.open(image_path)
                markdown_content = self._process_image_to_markdown(image)

                result = {
                    "image_number": idx + 1,
                    "content": markdown_content,
                    "metadata": {
                        "image_path": str(image_path),
                        "ocr_model": self.model_name,
                        "image_size": image.size,
                    },
                }

                results.append(result)

            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
                continue

        return results

    def process_single_page(
        self, pdf_path: Path, page_number: int
    ) -> dict[str, Any] | None:
        """
        Process a single page from a PDF.

        Args:
            pdf_path: Path to the PDF file
            page_number: Page number to process (1-based)

        Returns:
            Dictionary containing page content and metadata, or None if error
        """
        try:
            doc = fitz.open(pdf_path)

            if page_number < 1 or page_number > len(doc):
                logger.error(
                    f"Page number {page_number} out of range for {pdf_path.name}"
                )
                return None

            page = doc[page_number - 1]  # Convert to 0-based

            # Convert page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))

            # Process with Nanonets OCR
            markdown_content = self._process_image_to_markdown(image)

            result = {
                "page_number": page_number,
                "content": markdown_content,
                "metadata": {
                    "pdf_path": str(pdf_path),
                    "page_count": len(doc),
                    "ocr_model": self.model_name,
                    "image_size": image.size,
                },
            }

            doc.close()
            return result

        except Exception as e:
            logger.error(f"Error processing page {page_number} from {pdf_path}: {e}")
            return None

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": str(self.model.device) if self.model else "Not loaded",
            "dtype": str(self.model.dtype) if self.model else "Not loaded",
            "capabilities": [
                "LaTeX equation recognition",
                "Intelligent image description",
                "Signature and watermark detection",
                "Checkbox handling",
                "Complex table extraction",
                "Structured markdown output",
            ],
        }
