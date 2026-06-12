"""
Tests for the ProviderUpdates plugin.
"""

import unittest
from unittest.mock import MagicMock, patch

from maki.plugins.provider_updates.provider_updates import (
    ALLOWED_METHODS,
    ProviderUpdates,
    register_plugin,
)


class TestProviderUpdates(unittest.TestCase):
    def setUp(self):
        self.plugin = ProviderUpdates()

    def test_register_plugin_returns_instance(self):
        self.assertIsInstance(register_plugin(), ProviderUpdates)

    def test_allowed_methods(self):
        self.assertEqual(ALLOWED_METHODS, ["fetch_model_releases"])

    def test_skips_non_200_responses(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        with patch("maki.plugins.provider_updates.provider_updates._http_get", return_value=mock_resp), \
             patch("maki.plugins.provider_updates.provider_updates.time.sleep"):
            result = self.plugin.fetch_model_releases({"OpenAI": "https://example.com"})
        self.assertEqual(result, [])

    def test_returns_cleaned_content(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><h1>Release</h1><p>New model shipped.</p></body></html>"
        with patch("maki.plugins.provider_updates.provider_updates._http_get", return_value=mock_resp), \
             patch("maki.plugins.provider_updates.provider_updates.time.sleep"):
            result = self.plugin.fetch_model_releases({"OpenAI": "https://example.com"})
        self.assertEqual(result[0]["provider"], "OpenAI")
        self.assertIn("Release", result[0]["content"])

    def test_trims_loading_noise_when_date_is_present(self):
        noisy_text = ("Loading " * 8) + "May 10, 2026 New release notes here"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = noisy_text
        with patch("maki.plugins.provider_updates.provider_updates._http_get", return_value=mock_resp), \
             patch("maki.plugins.provider_updates.provider_updates.time.sleep"):
            result = self.plugin.fetch_model_releases({"OpenAI": "https://example.com"})
        self.assertTrue(result[0]["content"].startswith("May 10, 2026"))
