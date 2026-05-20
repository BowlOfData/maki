"""
LLM OCR backend — delegates extraction to a vision-capable Ollama model.

Preferred backend. Defaults to glm-ocr (available via Ollama).

Supported inputs:
  - Images (PNG, JPG, TIFF, BMP, GIF) — encoded directly as base64.
  - PDFs — rasterised page-by-page via pdf2image (optional).
  - DOCX / XLSX — converted to PDF first via libreoffice --headless (optional),
    then processed as PDF.

Dependencies:
  - Maki LLM instance with vision support (required).
  - pdf2image + poppler (optional, for PDF input).
  - libreoffice (optional system tool, for DOCX/XLSX input).
"""

import base64
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from .base import OCRBackend

logger = logging.getLogger(__name__)

try:
    from pdf2image import convert_from_path as _convert_from_path
    _PDF2IMAGE_AVAILABLE = True
except ImportError:
    _PDF2IMAGE_AVAILABLE = False

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp"}
_PDF_EXTENSIONS = {".pdf"}
_OFFICE_EXTENSIONS = {".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt", ".odt", ".ods"}

_PAGE_SEP = "\n\n---\n\n"

_DEFAULT_SYSTEM_PROMPT = (
    "You are an OCR assistant. Extract all text from the provided document image "
    "and format it as clean Markdown. Preserve headings, lists, tables, and text "
    "layout where possible. Return only the extracted Markdown content, no commentary."
)

_DEFAULT_USER_PROMPT = (
    "Extract all text from this image as Markdown."
)


class LLMBackend(OCRBackend):
    """OCR via a vision-capable Ollama model (default: glm-ocr)."""

    NAME = "llm"

    def __init__(
        self,
        maki_instance=None,
        model: str = "glm-ocr",
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
    ) -> None:
        self._maki = maki_instance
        self.model = model
        self._system = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self._user_prompt = user_prompt or _DEFAULT_USER_PROMPT

    def is_available(self) -> bool:
        return self._maki is not None

    def extract(self, file_path: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.is_available():
            return self._result(file_path, error="LLM backend requires a maki_instance")
        opts = options or {}
        system = opts.get("system_prompt", self._system)
        user_prompt = opts.get("user_prompt", self._user_prompt)
        dpi = opts.get("dpi", 200)

        ext = Path(file_path).suffix.lower()
        try:
            if ext in _IMAGE_EXTENSIONS:
                return self._process_images([file_path], file_path, system, user_prompt)
            if ext in _PDF_EXTENSIONS:
                return self._process_pdf(file_path, system, user_prompt, dpi)
            if ext in _OFFICE_EXTENSIONS:
                return self._process_office(file_path, system, user_prompt, dpi)
            return self._result(
                file_path,
                error=f"LLM backend: unsupported file type '{ext}' — convert manually to an image or PDF first",
            )
        except Exception as exc:
            logger.warning("LLMBackend failed for %s: %s", file_path, exc)
            return self._result(file_path, error=str(exc))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, image_path: str) -> str:
        return base64.standard_b64encode(Path(image_path).read_bytes()).decode("utf-8")

    def _ocr_image(self, image_path: str, system: str, user_prompt: str) -> str:
        """Send a single image to the LLM and return the extracted Markdown."""
        b64 = self._encode(image_path)
        response = self._maki.chat_with_image(user_prompt, image_b64=b64, system=system)
        return response.content.strip()

    def _process_images(
        self, image_paths: list, source_path: str, system: str, user_prompt: str
    ) -> Dict[str, Any]:
        pages = []
        for p in image_paths:
            pages.append(self._ocr_image(p, system, user_prompt))
        markdown = _PAGE_SEP.join(p for p in pages if p)
        return self._result(source_path, markdown=markdown, pages=len(image_paths))

    def _process_pdf(
        self, file_path: str, system: str, user_prompt: str, dpi: int
    ) -> Dict[str, Any]:
        if not _PDF2IMAGE_AVAILABLE:
            return self._result(
                file_path,
                error="LLM backend PDF support requires 'pdf2image' (pip install pdf2image)",
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            images = _convert_from_path(file_path, dpi=dpi, output_folder=tmpdir, fmt="png")
            # Save PIL images to disk so _encode can read them.
            image_paths = []
            for i, img in enumerate(images):
                p = os.path.join(tmpdir, f"page_{i:04d}.png")
                img.save(p, "PNG")
                image_paths.append(p)
            return self._process_images(image_paths, file_path, system, user_prompt)

    def _process_office(
        self, file_path: str, system: str, user_prompt: str, dpi: int
    ) -> Dict[str, Any]:
        if not shutil.which("libreoffice"):
            return self._result(
                file_path,
                error="LLM backend DOCX/XLSX support requires 'libreoffice' to be installed",
            )
        if not _PDF2IMAGE_AVAILABLE:
            return self._result(
                file_path,
                error="LLM backend DOCX/XLSX support requires 'pdf2image' (pip install pdf2image)",
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                ["libreoffice", "--headless", "--convert-to", "pdf",
                 "--outdir", tmpdir, file_path],
                check=True,
                capture_output=True,
            )
            pdf_name = Path(file_path).stem + ".pdf"
            pdf_path = os.path.join(tmpdir, pdf_name)
            if not os.path.exists(pdf_path):
                return self._result(
                    file_path, error="libreoffice conversion produced no PDF output"
                )
            return self._process_pdf(pdf_path, system, user_prompt, dpi)
