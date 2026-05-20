"""
Image OCR backend — pytesseract based extraction.

Extracts text from PNG, JPG, TIFF, BMP, and similar raster images.
Optional dependencies: pytesseract, Pillow.
"""

import logging
from typing import Any, Dict, Optional

from .base import OCRBackend

logger = logging.getLogger(__name__)

try:
    import pytesseract as _pytesseract
    _PYTESSERACT_AVAILABLE = True
except ImportError:
    _PYTESSERACT_AVAILABLE = False

try:
    from PIL import Image as _Image
    _PILLOW_AVAILABLE = True
except ImportError:
    _PILLOW_AVAILABLE = False


class ImageBackend(OCRBackend):
    """Extract text from raster images using pytesseract."""

    NAME = "image"

    def is_available(self) -> bool:
        return _PYTESSERACT_AVAILABLE and _PILLOW_AVAILABLE

    def extract(self, file_path: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.is_available():
            return self._result(
                file_path, error="Image backend requires 'pytesseract' and 'Pillow'"
            )
        opts = options or {}
        lang = opts.get("lang", "eng")
        try:
            img = _Image.open(file_path)
            text = _pytesseract.image_to_string(img, lang=lang).strip()
            return self._result(file_path, markdown=text, pages=1)
        except Exception as exc:
            logger.warning("ImageBackend failed for %s: %s", file_path, exc)
            return self._result(file_path, error=str(exc))
