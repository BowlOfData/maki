"""
OCR Plugin for Maki Framework

Extracts text from documents (PDF, DOCX, XLSX, images) and writes
Markdown output to a configurable directory.
"""

from .ocr import OCR, register_plugin

__all__ = ["OCR", "register_plugin"]
