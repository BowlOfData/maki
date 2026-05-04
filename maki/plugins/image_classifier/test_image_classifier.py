"""
Unit tests for the ImageClassifier plugin.
"""

import asyncio
import base64
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from maki.plugins.image_classifier.image_classifier import ImageClassifier, register_plugin


def _fake_maki(response_text: str):
    """Return a minimal mock Maki instance whose chat methods echo response_text."""
    mock = MagicMock()
    mock.chat_with_image.return_value = SimpleNamespace(content=response_text)

    async def _async_chat(*args, **kwargs):
        return SimpleNamespace(content=response_text)

    mock.async_chat = _async_chat
    return mock


class TestImageClassifierInit(unittest.TestCase):

    def test_register_plugin_returns_instance(self):
        ic = register_plugin()
        self.assertIsInstance(ic, ImageClassifier)

    def test_register_plugin_with_maki(self):
        mock_maki = MagicMock()
        ic = register_plugin(maki_instance=mock_maki)
        self.assertEqual(ic._maki, mock_maki)

    def test_allowed_methods_on_class(self):
        self.assertIn("classify_image", ImageClassifier.ALLOWED_METHODS)
        self.assertIn("classify_image_async", ImageClassifier.ALLOWED_METHODS)
        self.assertIn("classify_image_async_coro", ImageClassifier.ALLOWED_METHODS)


class TestClassifyImage(unittest.TestCase):

    def setUp(self):
        self._image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # minimal fake PNG bytes

    def _patch_encode(self, classifier):
        """Patch _encode so no real file access is needed."""
        classifier._encode = lambda path: base64.standard_b64encode(self._image_bytes).decode()

    def test_returns_error_without_maki(self):
        ic = ImageClassifier(maki_instance=None)
        self._patch_encode(ic)
        result = ic.classify_image("image.jpg", "What is this?")
        self.assertFalse(result["success"])
        self.assertIn("label", result)
        self.assertEqual(result["label"], "unknown")
        self.assertIn("error", result)
        self.assertEqual(result["image_path"], "image.jpg")

    def test_successful_classification(self):
        ic = ImageClassifier(maki_instance=_fake_maki("outdoors"))
        self._patch_encode(ic)
        result = ic.classify_image("photo.jpg", "Indoors or outdoors?")
        self.assertTrue(result["success"])
        self.assertEqual(result["label"], "outdoors")
        self.assertIsNone(result["error"])

    def test_valid_labels_accepted(self):
        ic = ImageClassifier(maki_instance=_fake_maki("safe"))
        self._patch_encode(ic)
        result = ic.classify_image(
            "img.jpg", "Safe or unsafe?", valid_labels=["safe", "unsafe"]
        )
        self.assertEqual(result["label"], "safe")

    def test_invalid_label_falls_back(self):
        ic = ImageClassifier(maki_instance=_fake_maki("maybe"))
        self._patch_encode(ic)
        result = ic.classify_image(
            "img.jpg",
            "Safe or unsafe?",
            valid_labels=["safe", "unsafe"],
            fallback_label="unknown",
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["label"], "unknown")

    def test_model_response_stripped_and_lowercased(self):
        ic = ImageClassifier(maki_instance=_fake_maki("  INDOORS  "))
        self._patch_encode(ic)
        result = ic.classify_image(
            "img.jpg", "Indoors or outdoors?", valid_labels=["indoors", "outdoors"]
        )
        self.assertEqual(result["label"], "indoors")

    def test_encode_failure_returns_error(self):
        ic = ImageClassifier(maki_instance=_fake_maki("ok"))
        ic._encode = lambda path: (_ for _ in ()).throw(FileNotFoundError("not found"))
        result = ic.classify_image("missing.jpg", "What is this?")
        self.assertFalse(result["success"])
        self.assertIn("not found", result["error"])
        self.assertEqual(result["label"], "unknown")

    def test_fallback_label_default_is_unknown(self):
        ic = ImageClassifier(maki_instance=_fake_maki("gibberish"))
        self._patch_encode(ic)
        result = ic.classify_image(
            "img.jpg", "Safe or unsafe?", valid_labels=["safe", "unsafe"]
        )
        self.assertEqual(result["label"], "unknown")

    def test_image_path_in_result(self):
        ic = ImageClassifier(maki_instance=_fake_maki("yes"))
        self._patch_encode(ic)
        result = ic.classify_image("path/to/image.png", "Has a person?")
        self.assertEqual(result["image_path"], "path/to/image.png")


class TestClassifyImageAsync(unittest.TestCase):

    def setUp(self):
        self._image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

    def _patch_encode(self, classifier):
        classifier._encode = lambda path: base64.standard_b64encode(self._image_bytes).decode()

    def test_sync_wrapper_returns_result(self):
        ic = ImageClassifier(maki_instance=_fake_maki("indoors"))
        self._patch_encode(ic)
        result = ic.classify_image_async("photo.jpg", "Indoors or outdoors?")
        self.assertTrue(result["success"])
        self.assertEqual(result["label"], "indoors")

    def test_async_wrapper_without_maki(self):
        ic = ImageClassifier(maki_instance=None)
        self._patch_encode(ic)
        result = ic.classify_image_async("photo.jpg", "What is this?")
        self.assertFalse(result["success"])
        self.assertIn("label", result)
        self.assertEqual(result["label"], "unknown")

    def test_async_failure_returns_error_dict(self):
        async def _bad_chat(*args, **kwargs):
            raise RuntimeError("model unavailable")

        mock_maki = MagicMock()
        mock_maki.async_chat = _bad_chat
        ic = ImageClassifier(maki_instance=mock_maki)
        self._patch_encode(ic)
        result = asyncio.run(
            ic.classify_image_async_coro("img.jpg", "What is this?", fallback_label="unknown")
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["label"], "unknown")
        self.assertIn("model unavailable", result["error"])

    def test_native_coroutine(self):
        ic = ImageClassifier(maki_instance=_fake_maki("safe"))
        self._patch_encode(ic)
        result = asyncio.run(
            ic.classify_image_async_coro(
                "img.jpg", "Safe or unsafe?", valid_labels=["safe", "unsafe"]
            )
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["label"], "safe")

    def test_native_coroutine_without_maki(self):
        ic = ImageClassifier(maki_instance=None)
        self._patch_encode(ic)
        result = asyncio.run(
            ic.classify_image_async_coro("img.jpg", "What is this?")
        )
        self.assertFalse(result["success"])
        self.assertIn("label", result)
        self.assertEqual(result["label"], "unknown")


if __name__ == "__main__":
    unittest.main()
