"""
Tests for the WebToMd plugin
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from maki.plugins.web_to_md.web_to_md import WebToMd

class TestWebToMd(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock Maki instance
        mock_maki = MagicMock()
        self.web_to_md = WebToMd(mock_maki)

    @patch('requests.get')
    def test_fetch_and_convert_to_md_success(self, mock_get):
        """Test successful fetching and conversion."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><head><title>Test</title></head><body><h1>Hello World</h1></body></html>"
        mock_get.return_value = mock_response

        # Mock the file writer
        with patch.object(self.web_to_md.file_writer, 'write_file') as mock_write:
            mock_write.return_value = {'success': True, 'file_path': 'test.md', 'bytes_written': 100}

            result = self.web_to_md.fetch_and_convert_to_md("https://example.com")

            self.assertTrue(result['success'])
            self.assertEqual(result['url'], "https://example.com")
            self.assertIsNotNone(result['output_file'])
            self.assertIsNotNone(result['content'])

    @patch('requests.get')
    def test_fetch_and_convert_to_md_failure(self, mock_get):
        """Test failed fetching."""
        # Mock the response with an error
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = self.web_to_md.fetch_and_convert_to_md("https://example.com")

        self.assertFalse(result['success'])
        self.assertIn('HTTP 404', result['error'])

    def test_fetch_and_convert_to_md_invalid_url(self):
        """Test with invalid URL."""
        result = self.web_to_md.fetch_and_convert_to_md("")

        self.assertFalse(result['success'])
        self.assertIn('URL must be a non-empty string', result['error'])

    def test_html_to_markdown_conversion(self):
        """Test HTML to markdown conversion."""
        html_content = "<h1>Test Title</h1><p>This is a <strong>test</strong> paragraph.</p>"
        markdown_content = self.web_to_md._html_to_markdown(html_content)

        self.assertIn('# Test Title', markdown_content)
        self.assertIn('**test**', markdown_content)

if __name__ == '__main__':
    unittest.main()