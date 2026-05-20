"""
OCR Plugin for Maki Framework

Extracts text from documents (PDF, DOCX, XLSX, images, …) and writes
the result as Markdown to a configurable output directory.

Backends (all optional except llm):
  - llm    : vision-capable Ollama model, default glm-ocr  [preferred]
  - pdf    : pdfplumber / pytesseract
  - docx   : python-docx
  - xlsx   : openpyxl
  - image  : pytesseract + Pillow

Auto-selection when no backend is specified:
  maki_instance present  → llm
  no maki_instance       → dispatch by file extension
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from maki.config import DEFAULT_OCR_OUTPUT_DIR, DEFAULT_OCR_MODEL
from maki.plugins.file_writer.file_writer import FileWriter

from .backends import DocxBackend, ImageBackend, LLMBackend, PDFBackend, XlsxBackend
from .backends.base import OCRBackend

logger = logging.getLogger(__name__)

# Extension → library-based backend name (used only when maki_instance is absent).
_EXT_BACKEND_MAP: Dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "docx",
    ".xlsx": "xlsx",
    ".xls": "xlsx",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".tiff": "image",
    ".tif": "image",
    ".bmp": "image",
    ".gif": "image",
    ".webp": "image",
}


class OCR:
    """
    OCR plugin for Maki agents.

    Extracts text from document files and writes the result as a Markdown
    file under output_dir.

    Args:
        maki_instance:   Maki LLM backend (required for the llm backend).
        output_dir:      Directory where .md output files are written.
                         Defaults to MAKI_OCR_OUTPUT_DIR env var or ~/maki_ocr_output.
        base_dir:        Root directory for safe input path resolution.
                         Defaults to the current working directory.
        default_backend: Backend to use when extract() is called without
                         specifying one.  None → auto-select.
        backend_options: Dict of kwargs passed to backend constructors
                         (e.g. {"llm": {"model": "glm-ocr"}}).
    """

    ALLOWED_METHODS = ["extract", "extract_to_file", "list_backends"]

    def __init__(
        self,
        maki_instance=None,
        output_dir: Optional[str] = None,
        base_dir: Optional[str] = None,
        default_backend: Optional[str] = None,
        backend_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.maki = maki_instance
        self.base_dir = os.path.realpath(base_dir or os.getcwd())
        self.output_dir = os.path.expanduser(output_dir or DEFAULT_OCR_OUTPUT_DIR)
        self.default_backend = default_backend
        self._backend_options: Dict[str, Any] = backend_options or {}

        os.makedirs(self.output_dir, exist_ok=True)
        self._writer = FileWriter(base_dir=self.output_dir)

        # Build backend registry.
        llm_opts = self._backend_options.get("llm", {})
        model = llm_opts.get("model", DEFAULT_OCR_MODEL)
        self._backends: Dict[str, OCRBackend] = {
            "llm": LLMBackend(
                maki_instance=maki_instance,
                model=model,
                system_prompt=llm_opts.get("system_prompt"),
                user_prompt=llm_opts.get("user_prompt"),
            ),
            "pdf": PDFBackend(),
            "docx": DocxBackend(),
            "xlsx": XlsxBackend(),
            "image": ImageBackend(),
        }
        logger.info(
            "OCR plugin initialised (output_dir='%s', default_backend=%s)",
            self.output_dir,
            default_backend or "auto",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        file_path: str,
        backend: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extract text from file_path and return it as Markdown.

        Args:
            file_path: Path to the source document (relative to base_dir or absolute).
            backend:   Backend name override ('llm', 'pdf', 'docx', 'xlsx', 'image').
            options:   Backend-specific options (dpi, lang, system_prompt, …).

        Returns:
            Dict with keys: success, file_path, backend, markdown, pages, error.
        """
        if not isinstance(file_path, str) or not file_path.strip():
            return _error_result(file_path, "file_path must be a non-empty string")

        try:
            safe_path = self._safe_path(file_path)
        except ValueError as exc:
            return _error_result(file_path, str(exc))

        if not os.path.isfile(safe_path):
            return _error_result(file_path, f"File not found: {file_path}")

        chosen = self._resolve_backend(backend, safe_path)
        engine = self._backends.get(chosen)
        if engine is None:
            return _error_result(file_path, f"Unknown backend: '{chosen}'")
        if not engine.is_available():
            return _error_result(
                file_path,
                f"Backend '{chosen}' is not available — missing dependencies. "
                "Run list_backends() for details.",
            )

        logger.info("OCR: extracting '%s' with backend '%s'", file_path, chosen)
        return engine.extract(safe_path, options)

    def extract_to_file(
        self,
        file_path: str,
        output_name: Optional[str] = None,
        backend: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extract text from file_path and persist the result as a Markdown file.

        Args:
            file_path:   Source document path.
            output_name: Output filename (without path).  Defaults to <source_stem>.md.
            backend:     Backend override.
            options:     Backend-specific options.

        Returns:
            The extract() dict plus 'output_path' (str or None on failure).
        """
        result = self.extract(file_path, backend=backend, options=options)
        result["output_path"] = None
        if not result["success"]:
            return result

        stem = Path(file_path).stem
        out_name = output_name or f"{stem}.md"
        write_result = self._writer.write_file(
            out_name,
            result["markdown"],
            mode="x",
        )
        if write_result["success"]:
            abs_output = os.path.join(self.output_dir, out_name)
            result["output_path"] = abs_output
            logger.info("OCR: wrote output to '%s'", abs_output)
        else:
            result["success"] = False
            result["error"] = write_result.get("error", "Unknown write error")
        return result

    def list_backends(self) -> Dict[str, Any]:
        """Report all registered backends and their availability."""
        return {
            name: {
                "available": backend.is_available(),
            }
            for name, backend in self._backends.items()
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _safe_path(self, path: str) -> str:
        resolved = os.path.realpath(os.path.join(self.base_dir, path))
        if resolved != self.base_dir and not resolved.startswith(self.base_dir + os.sep):
            raise ValueError(
                f"Path '{path}' resolves outside the allowed directory '{self.base_dir}'"
            )
        return resolved

    def _resolve_backend(self, backend: Optional[str], file_path: str) -> str:
        """Determine which backend to use for file_path."""
        if backend:
            return backend
        if self.default_backend:
            return self.default_backend
        # Auto: prefer LLM when maki_instance is available.
        if self.maki is not None:
            return "llm"
        ext = Path(file_path).suffix.lower()
        return _EXT_BACKEND_MAP.get(ext, "llm")


def _error_result(file_path: str, error: str) -> Dict[str, Any]:
    return {
        "success": False,
        "file_path": file_path,
        "backend": None,
        "markdown": "",
        "pages": 0,
        "error": error,
    }


def register_plugin(maki_instance=None, **kwargs) -> OCR:
    """Register the OCR plugin with the Maki framework."""
    return OCR(maki_instance=maki_instance, **kwargs)
