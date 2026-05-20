"""
Abstract base class for OCR backends.

Every backend must implement is_available() and extract(). The _result()
helper builds the standard return dict so all backends stay consistent.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class OCRBackend(ABC):
    """Contract that every OCR extraction engine must satisfy."""

    NAME: str = ""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if all required dependencies are installed."""

    @abstractmethod
    def extract(self, file_path: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract text from file_path and return it as Markdown.

        Args:
            file_path: Absolute path to the source document.
            options:   Backend-specific overrides (model, prompt, dpi, …).

        Returns:
            Dict produced by _result().
        """

    def _result(
        self,
        file_path: str,
        markdown: str = "",
        pages: int = 0,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "success": error is None,
            "file_path": file_path,
            "backend": self.NAME,
            "markdown": markdown,
            "pages": pages,
            "error": error,
        }
