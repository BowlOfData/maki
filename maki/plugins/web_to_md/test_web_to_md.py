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

    def test_fetch_and_convert_to_md_success(self):
        """Test successful fetching and conversion."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/html; charset=utf-8"}
        mock_response.text = "<html><head><title>Test</title></head><body><h1>Hello World</h1></body></html>"

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response

        with patch('requests.Session', return_value=mock_session), \
             patch.object(self.web_to_md.file_writer, 'write_file') as mock_write:
            mock_write.return_value = {'success': True, 'file_path': 'test.md', 'bytes_written': 100}
            result = self.web_to_md.fetch_and_convert_to_md("https://example.com")

        self.assertTrue(result['success'])
        self.assertEqual(result['url'], "https://example.com")
        self.assertIsNotNone(result['output_file'])
        self.assertIsNotNone(result['content'])

    def test_fetch_and_convert_to_md_failure(self):
        """Test failed fetching (404 response)."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response

        with patch('requests.Session', return_value=mock_session):
            result = self.web_to_md.fetch_and_convert_to_md("https://example.com")

        self.assertFalse(result['success'])
        self.assertIn('HTTP 404', result['error'])

    def test_fetch_and_convert_to_md_invalid_url(self):
        """Test with invalid URL."""
        result = self.web_to_md.fetch_and_convert_to_md("")

        self.assertFalse(result['success'])
        self.assertIn('URL must be a non-empty string', result['error'])

    def test_html_to_markdown_conversion(self):
        """Test HTML to markdown conversion via the regex fallback converter."""
        html_content = "<h1>Test Title</h1><p>This is a <strong>test</strong> paragraph.</p>"
        markdown_content = self.web_to_md._regex_to_markdown(html_content)

        self.assertIn('# Test Title', markdown_content)
        self.assertIn('**test**', markdown_content)

def _addrinfo(ip, port):
    import socket
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]


class TestWebToMdSSRFProtection(unittest.TestCase):
    """The URL comes from page/feed content, so the fetch must not be able to
    reach internal addresses, whether directly or via a redirect."""

    def setUp(self):
        self.web_to_md = WebToMd(MagicMock())

    def test_hostname_resolving_to_metadata_address_is_blocked(self):
        # A public-looking name whose DNS points at the cloud metadata IP
        # passes any static hostname check; only resolution-time validation
        # catches it.
        with patch("maki.connector.socket.getaddrinfo",
                   return_value=_addrinfo("169.254.169.254", 80)), \
             patch("maki.plugins.web_to_md.web_to_md.time.sleep"):
            result = self.web_to_md.fetch_and_convert_to_md(
                "http://rebind.attacker.example/latest/meta-data/")

        self.assertFalse(result["success"])
        self.assertIn("169.254.169.254", result["error"])

    def test_redirect_to_internal_host_is_blocked(self):
        import http.server
        import threading

        hits = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                hits.append(self.headers.get("Host"))
                if self.headers.get("Host", "").startswith("public.example"):
                    self.send_response(302)
                    self.send_header(
                        "Location", f"http://internal.example:{self.server.server_port}/secret")
                    self.end_headers()
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(b"<html><body>internal secret</body></html>")

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_port
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        table = {
            "public.example": "127.0.0.1",    # the loopback test server (allowed)
            "internal.example": "10.0.0.5",   # a private address (must be refused)
        }

        def fake_resolve(host, *args, **kwargs):
            # IP literals (the already-pinned address) resolve to themselves
            return _addrinfo(table.get(host, host), port)

        try:
            with patch("maki.connector.socket.getaddrinfo", side_effect=fake_resolve), \
                 patch("maki.plugins.web_to_md.web_to_md.time.sleep"):
                result = self.web_to_md.fetch_and_convert_to_md(
                    f"http://public.example:{port}/article")
        finally:
            server.shutdown()
            server.server_close()

        self.assertFalse(result["success"])
        self.assertIn("10.0.0.5", result["error"])
        # The first hop was served; the internal hop was never requested.
        self.assertEqual(len(hits), 1)

    def test_connector_runs_with_ssrf_protection_enabled(self):
        with patch("maki.plugins.web_to_md.web_to_md.Connector") as mock_connector:
            mock_connector.return_value.get.return_value = MagicMock(status_code=404)
            self.web_to_md.fetch_and_convert_to_md("https://example.com")

        _, kwargs = mock_connector.call_args
        self.assertTrue(kwargs.get("ssrf_protect", True))
        self.assertFalse(kwargs.get("allow_private", False))


if __name__ == '__main__':
    unittest.main()