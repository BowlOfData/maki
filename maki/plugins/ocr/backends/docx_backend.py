"""
DOCX OCR backend — python-docx based text extraction.

Converts headings, paragraphs, and tables to Markdown.
Optional dependency: python-docx.
"""

import logging
from typing import Any, Dict, Optional

from .base import OCRBackend

logger = logging.getLogger(__name__)

try:
    import docx as _docx
    _DOCX_AVAILABLE = True
except ImportError:
    _DOCX_AVAILABLE = False


class DocxBackend(OCRBackend):
    """Extract text from DOCX/DOC files using python-docx."""

    NAME = "docx"

    # Map python-docx heading style names to Markdown heading levels.
    _HEADING_LEVELS = {f"Heading {i}": "#" * i for i in range(1, 7)}
    _HEADING_LEVELS["Title"] = "#"

    def is_available(self) -> bool:
        return _DOCX_AVAILABLE

    def extract(self, file_path: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not _DOCX_AVAILABLE:
            return self._result(file_path, error="DOCX backend requires 'python-docx'")
        try:
            doc = _docx.Document(file_path)
            blocks = []
            for block in doc.element.body:
                tag = block.tag.split("}")[-1] if "}" in block.tag else block.tag
                if tag == "p":
                    para = _docx.text.paragraph.Paragraph(block, doc)
                    blocks.append(self._para_to_md(para))
                elif tag == "tbl":
                    table = _docx.table.Table(block, doc)
                    blocks.append(self._table_to_md(table))
            markdown = "\n\n".join(b for b in blocks if b)
            return self._result(file_path, markdown=markdown, pages=1)
        except Exception as exc:
            logger.warning("DocxBackend failed for %s: %s", file_path, exc)
            return self._result(file_path, error=str(exc))

    def _para_to_md(self, para) -> str:
        text = para.text.strip()
        if not text:
            return ""
        style = para.style.name if para.style else ""
        prefix = self._HEADING_LEVELS.get(style, "")
        return f"{prefix} {text}" if prefix else text

    def _table_to_md(self, table) -> str:
        rows = []
        for i, row in enumerate(table.rows):
            cells = [c.text.strip().replace("|", "\\|") for c in row.cells]
            rows.append("| " + " | ".join(cells) + " |")
            if i == 0:
                rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
        return "\n".join(rows)
