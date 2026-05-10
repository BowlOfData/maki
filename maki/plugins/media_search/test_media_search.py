"""
Tests for the MediaSearch plugin.
"""

import unittest
from unittest.mock import MagicMock, patch

from maki.plugins.media_search.media_search import ALLOWED_METHODS, MediaSearch, register_plugin


class TestMediaSearch(unittest.TestCase):
    def setUp(self):
        self.plugin = MediaSearch()

    def test_register_plugin_returns_instance(self):
        self.assertIsInstance(register_plugin(), MediaSearch)

    def test_allowed_methods(self):
        self.assertEqual(ALLOWED_METHODS, ["fetch_pexels_image"])

    def test_returns_none_when_api_key_missing(self):
        self.assertIsNone(self.plugin.fetch_pexels_image("ai", ""))

    def test_returns_none_when_no_results(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"photos": []}
        with patch("maki.plugins.media_search.media_search.requests.get", return_value=mock_resp):
            self.assertIsNone(self.plugin.fetch_pexels_image("ai", "key"))

    def test_returns_first_large_image_url(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "photos": [{"src": {"large": "https://images.example.com/pic.jpg"}}]
        }
        with patch("maki.plugins.media_search.media_search.requests.get", return_value=mock_resp):
            result = self.plugin.fetch_pexels_image("ai", "key")
        self.assertEqual(result, "https://images.example.com/pic.jpg")
