"""
PDF OCR backend — library-based text extraction.

Tries pdfplumber first (best for text-layer PDFs), falls back to
pytesseract + pdf2image for scanned PDFs.  Both are optional imports.
"""

import logging
from typing import Any, Dict, Optional

from .base import OCRBackend

logger = logging.getLogger(__name__)

try:
    import pdfplumber as _pdfplumber
    _PDFPLUMBER_AVAILABLE = True
except ImportError:
    _PDFPLUMBER_AVAILABLE = False

try:
    from pdf2image import convert_from_path as _convert_from_path
    _PDF2IMAGE_AVAILABLE = True
except ImportError:
    _PDF2IMAGE_AVAILABLE = False

try:
    import pytesseract as _pytesseract
    _PYTESSERACT_AVAILABLE = True
except ImportError:
    _PYTESSERACT_AVAILABLE = False

_PAGE_SEP = "\n\n---\n\n"


class PDFBackend(OCRBackend):
    """Extract text from PDF files using pdfplumber (text) or pytesseract (scan)."""

    NAME = "pdf"

    def is_available(self) -> bool:
        return _PDFPLUMBER_AVAILABLE or (_PDF2IMAGE_AVAILABLE and _PYTESSERACT_AVAILABLE)

    def extract(self, file_path: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.is_available():
            return self._result(
                file_path,
                error="PDF backend requires 'pdfplumber' or 'pdf2image' + 'pytesseract'",
            )
        opts = options or {}
        try:
            if _PDFPLUMBER_AVAILABLE:
                return self._extract_text_layer(file_path, opts)
            return self._extract_ocr(file_path, opts)
        except Exception as exc:
            logger.warning("PDFBackend failed for %s: %s", file_path, exc)
            return self._result(file_path, error=str(exc))

    def _extract_text_layer(self, file_path: str, opts: dict) -> Dict[str, Any]:
        pages_md = []
        with _pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages_md.append(text.strip())
        markdown = _PAGE_SEP.join(p for p in pages_md if p)
        return self._result(file_path, markdown=markdown, pages=len(pages_md))

    def _extract_ocr(self, file_path: str, opts: dict) -> Dict[str, Any]:
        dpi = opts.get("dpi", 200)
        images = _convert_from_path(file_path, dpi=dpi)
        pages_md = [_pytesseract.image_to_string(img).strip() for img in images]
        markdown = _PAGE_SEP.join(p for p in pages_md if p)
        return self._result(file_path, markdown=markdown, pages=len(images))
