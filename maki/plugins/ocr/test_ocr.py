"""
Unit tests for the OCR plugin.
"""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from maki.plugins.ocr.ocr import OCR, register_plugin
from maki.plugins.ocr.backends.llm_backend import LLMBackend
from maki.plugins.ocr.backends.pdf_backend import PDFBackend
from maki.plugins.ocr.backends.docx_backend import DocxBackend
from maki.plugins.ocr.backends.xlsx_backend import XlsxBackend
from maki.plugins.ocr.backends.image_backend import ImageBackend


def _fake_maki(response_text: str = "# Extracted\n\nSome text."):
    mock = MagicMock()
    mock.chat_with_image.return_value = SimpleNamespace(content=response_text)
    return mock


class TestRegisterPlugin(unittest.TestCase):

    def test_returns_ocr_instance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ocr = register_plugin(output_dir=tmpdir)
            self.assertIsInstance(ocr, OCR)

    def test_with_maki_instance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_maki = _fake_maki()
            ocr = register_plugin(maki_instance=mock_maki, output_dir=tmpdir)
            self.assertIs(ocr.maki, mock_maki)

    def test_allowed_methods(self):
        self.assertIn("extract", OCR.ALLOWED_METHODS)
        self.assertIn("extract_to_file", OCR.ALLOWED_METHODS)
        self.assertIn("list_backends", OCR.ALLOWED_METHODS)


class TestListBackends(unittest.TestCase):

    def test_returns_all_backends(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ocr = OCR(output_dir=tmpdir)
            report = ocr.list_backends()
            self.assertIn("llm", report)
            self.assertIn("pdf", report)
            self.assertIn("docx", report)
            self.assertIn("xlsx", report)
            self.assertIn("image", report)

    def test_llm_unavailable_without_maki(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ocr = OCR(maki_instance=None, output_dir=tmpdir)
            self.assertFalse(ocr.list_backends()["llm"]["available"])

    def test_llm_available_with_maki(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ocr = OCR(maki_instance=_fake_maki(), output_dir=tmpdir)
            self.assertTrue(ocr.list_backends()["llm"]["available"])


class TestExtractValidation(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._ocr = OCR(maki_instance=_fake_maki(), output_dir=self._tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_empty_file_path(self):
        result = self._ocr.extract("")
        self.assertFalse(result["success"])
        self.assertIn("non-empty", result["error"])

    def test_file_not_found(self):
        result = self._ocr.extract("does_not_exist.pdf")
        self.assertFalse(result["success"])
        self.assertIn("not found", result["error"])

    def test_path_traversal_blocked(self):
        result = self._ocr.extract("../../etc/passwd")
        self.assertFalse(result["success"])
        self.assertIn("outside the allowed directory", result["error"])

    def test_unknown_backend(self):
        # Create a real file so validation passes before the backend check.
        f = os.path.join(self._tmpdir, "sample.txt")
        with open(f, "w") as fh:
            fh.write("hello")
        ocr = OCR(maki_instance=_fake_maki(), base_dir=self._tmpdir, output_dir=self._tmpdir)
        result = ocr.extract("sample.txt", backend="nonexistent")
        self.assertFalse(result["success"])
        self.assertIn("Unknown backend", result["error"])


class TestBackendAutoSelection(unittest.TestCase):

    def _make_ocr(self, maki=None):
        tmpdir = tempfile.mkdtemp()
        return OCR(maki_instance=maki, base_dir=tmpdir, output_dir=tempfile.mkdtemp()), tmpdir

    def test_auto_selects_llm_when_maki_present(self):
        ocr, tmpdir = self._make_ocr(maki=_fake_maki())
        self.assertEqual(ocr._resolve_backend(None, "doc.pdf"), "llm")

    def test_auto_selects_pdf_without_maki(self):
        ocr, _ = self._make_ocr()
        self.assertEqual(ocr._resolve_backend(None, "doc.pdf"), "pdf")

    def test_auto_selects_docx_without_maki(self):
        ocr, _ = self._make_ocr()
        self.assertEqual(ocr._resolve_backend(None, "doc.docx"), "docx")

    def test_auto_selects_xlsx_without_maki(self):
        ocr, _ = self._make_ocr()
        self.assertEqual(ocr._resolve_backend(None, "sheet.xlsx"), "xlsx")

    def test_auto_selects_image_without_maki(self):
        ocr, _ = self._make_ocr()
        self.assertEqual(ocr._resolve_backend(None, "scan.png"), "image")

    def test_explicit_backend_overrides_auto(self):
        ocr, _ = self._make_ocr(maki=_fake_maki())
        self.assertEqual(ocr._resolve_backend("pdf", "doc.pdf"), "pdf")

    def test_default_backend_overrides_auto(self):
        tmpdir = tempfile.mkdtemp()
        ocr = OCR(maki_instance=_fake_maki(), output_dir=tmpdir, default_backend="pdf")
        self.assertEqual(ocr._resolve_backend(None, "doc.docx"), "pdf")


class TestLLMBackend(unittest.TestCase):

    def test_unavailable_without_maki(self):
        backend = LLMBackend(maki_instance=None)
        self.assertFalse(backend.is_available())

    def test_available_with_maki(self):
        backend = LLMBackend(maki_instance=_fake_maki())
        self.assertTrue(backend.is_available())

    def test_returns_error_without_maki(self):
        backend = LLMBackend(maki_instance=None)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
            path = f.name
        try:
            result = backend.extract(path)
            self.assertFalse(result["success"])
            self.assertIn("maki_instance", result["error"])
        finally:
            os.unlink(path)

    def test_extract_image(self):
        mock_maki = _fake_maki("# Document\n\nHello world")
        backend = LLMBackend(maki_instance=mock_maki)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
            path = f.name
        try:
            result = backend.extract(path)
            self.assertTrue(result["success"])
            self.assertIn("Hello world", result["markdown"])
            self.assertEqual(result["pages"], 1)
            self.assertEqual(result["backend"], "llm")
        finally:
            os.unlink(path)

    def test_unsupported_extension(self):
        backend = LLMBackend(maki_instance=_fake_maki())
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"data")
            path = f.name
        try:
            result = backend.extract(path)
            self.assertFalse(result["success"])
            self.assertIn("unsupported file type", result["error"])
        finally:
            os.unlink(path)

    def test_custom_model_passed_through(self):
        backend = LLMBackend(maki_instance=_fake_maki(), model="custom-ocr")
        self.assertEqual(backend.model, "custom-ocr")


class TestExtractToFile(unittest.TestCase):

    def test_writes_md_file(self):
        with tempfile.TemporaryDirectory() as src_dir, \
             tempfile.TemporaryDirectory() as out_dir:
            img = os.path.join(src_dir, "scan.png")
            with open(img, "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)

            mock_maki = _fake_maki("# Result\n\nExtracted text here.")
            ocr = OCR(maki_instance=mock_maki, base_dir=src_dir, output_dir=out_dir)
            result = ocr.extract_to_file("scan.png")

            self.assertTrue(result["success"], result.get("error"))
            self.assertIsNotNone(result["output_path"])
            self.assertTrue(os.path.isfile(result["output_path"]))
            with open(result["output_path"]) as fh:
                content = fh.read()
            self.assertIn("Extracted text here", content)

    def test_custom_output_name(self):
        with tempfile.TemporaryDirectory() as src_dir, \
             tempfile.TemporaryDirectory() as out_dir:
            img = os.path.join(src_dir, "scan.png")
            with open(img, "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)

            ocr = OCR(maki_instance=_fake_maki(), base_dir=src_dir, output_dir=out_dir)
            result = ocr.extract_to_file("scan.png", output_name="my_output.md")
            self.assertTrue(result["success"], result.get("error"))
            self.assertTrue(result["output_path"].endswith("my_output.md"))

    def test_failed_extract_propagates_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ocr = OCR(output_dir=tmpdir)
            result = ocr.extract_to_file("nonexistent.pdf")
            self.assertFalse(result["success"])
            self.assertIsNone(result["output_path"])


class TestLibraryBackendAvailability(unittest.TestCase):
    """Verify that missing optional deps surface as clean errors rather than exceptions."""

    def _run_unavailable(self, backend, tmp_path):
        result = backend.extract(tmp_path)
        self.assertFalse(result["success"])
        self.assertIsNotNone(result["error"])
        self.assertEqual(result["markdown"], "")

    def test_pdf_backend_unavailable(self):
        with patch("maki.plugins.ocr.backends.pdf_backend._PDFPLUMBER_AVAILABLE", False), \
             patch("maki.plugins.ocr.backends.pdf_backend._PDF2IMAGE_AVAILABLE", False):
            backend = PDFBackend()
            self.assertFalse(backend.is_available())
            with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
                self._run_unavailable(backend, f.name)

    def test_docx_backend_unavailable(self):
        with patch("maki.plugins.ocr.backends.docx_backend._DOCX_AVAILABLE", False):
            backend = DocxBackend()
            self.assertFalse(backend.is_available())
            with tempfile.NamedTemporaryFile(suffix=".docx") as f:
                self._run_unavailable(backend, f.name)

    def test_xlsx_backend_unavailable(self):
        with patch("maki.plugins.ocr.backends.xlsx_backend._OPENPYXL_AVAILABLE", False):
            backend = XlsxBackend()
            self.assertFalse(backend.is_available())
            with tempfile.NamedTemporaryFile(suffix=".xlsx") as f:
                self._run_unavailable(backend, f.name)

    def test_image_backend_unavailable(self):
        with patch("maki.plugins.ocr.backends.image_backend._PYTESSERACT_AVAILABLE", False):
            backend = ImageBackend()
            self.assertFalse(backend.is_available())
            with tempfile.NamedTemporaryFile(suffix=".png") as f:
                self._run_unavailable(backend, f.name)


if __name__ == "__main__":
    unittest.main()
