"""
Tests for the hardened HTTP layer (maki/connector.py): URL validation,
DNS resolution + pinning, error classification, and config-driven timeouts.
"""
import asyncio
import socket
import unittest
from unittest.mock import MagicMock, patch

import requests

from maki.config import DEFAULT_REQUEST_TIMEOUT
from maki.connector import (
    AsyncConnector,
    Connector,
    _PinnedHTTPConnection,
    _PinnedHTTPSConnection,
    _resolve_and_validate,
    _SSRFAdapter,
    validate_url,
)
from maki.exceptions import (
    MakiAPIError,
    MakiNetworkError,
    MakiTimeoutError,
    MakiValidationError,
)


def _addrinfo(*ips):
    """Build a getaddrinfo-style result for the given IPs."""
    return [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 80)) for ip in ips
    ]


# ---------------------------------------------------------------------------
# Static URL validation
# ---------------------------------------------------------------------------

class TestValidateUrl(unittest.TestCase):

    def test_rejects_non_http_scheme(self):
        for url in ("ftp://example.com/x", "file:///etc/passwd", "gopher://x"):
            with self.assertRaises(MakiValidationError):
                validate_url(url)

    def test_rejects_empty_and_hostless(self):
        with self.assertRaises(MakiValidationError):
            validate_url("")
        with self.assertRaises(MakiValidationError):
            validate_url("http://")

    def test_rejects_private_ip_literal(self):
        with self.assertRaises(MakiValidationError):
            validate_url("http://192.168.1.1/admin")

    def test_allows_loopback(self):
        validate_url("http://127.0.0.1:11434/api/chat")
        validate_url("http://localhost:11434/api/chat")

    def test_allows_public_hosts(self):
        validate_url("https://example.com/page")
        validate_url("https://8.8.8.8/x")

    def test_allow_private_permits_lan_addresses(self):
        validate_url("http://192.168.1.50:8100/execute", allow_private=True)
        validate_url("http://10.1.2.3:11434", allow_private=True)


# ---------------------------------------------------------------------------
# DNS resolution + validation (rebinding defence)
# ---------------------------------------------------------------------------

class TestResolveAndValidate(unittest.TestCase):

    def test_public_resolution_returns_addresses(self):
        with patch("maki.connector.socket.getaddrinfo",
                   return_value=_addrinfo("93.184.216.34")):
            self.assertEqual(
                _resolve_and_validate("example.com", 80), ["93.184.216.34"]
            )

    def test_resolution_deduplicates_addresses(self):
        with patch("maki.connector.socket.getaddrinfo",
                   return_value=_addrinfo("1.2.3.4", "1.2.3.4", "5.6.7.8")):
            self.assertEqual(
                _resolve_and_validate("example.com", 80),
                ["1.2.3.4", "5.6.7.8"],
            )

    def test_private_resolution_blocked(self):
        """A hostname resolving to a private IP (DNS rebinding) is rejected."""
        with patch("maki.connector.socket.getaddrinfo",
                   return_value=_addrinfo("192.168.1.1")):
            with self.assertRaises(MakiValidationError):
                _resolve_and_validate("rebind.attacker.com", 80)

    def test_mixed_public_private_resolution_blocked(self):
        """One private record among public ones poisons the whole host."""
        with patch("maki.connector.socket.getaddrinfo",
                   return_value=_addrinfo("93.184.216.34", "10.0.0.5")):
            with self.assertRaises(MakiValidationError):
                _resolve_and_validate("mixed.attacker.com", 80)

    def test_resolution_failure_raises_network_error(self):
        with patch("maki.connector.socket.getaddrinfo",
                   side_effect=socket.gaierror("NXDOMAIN")):
            with self.assertRaises(MakiNetworkError):
                _resolve_and_validate("no-such-host.invalid", 80)

    def test_ip_literal_validated_without_resolution(self):
        with patch("maki.connector.socket.getaddrinfo") as mock_resolve:
            self.assertEqual(_resolve_and_validate("8.8.8.8", 80), ["8.8.8.8"])
            with self.assertRaises(MakiValidationError):
                _resolve_and_validate("10.0.0.1", 80)
            mock_resolve.assert_not_called()

    def test_loopback_literal_allowed(self):
        self.assertEqual(_resolve_and_validate("127.0.0.1", 80), ["127.0.0.1"])


# ---------------------------------------------------------------------------
# Connect-time pinning
# ---------------------------------------------------------------------------

class TestConnectionPinning(unittest.TestCase):

    def test_http_connection_pins_resolved_ip(self):
        conn = _PinnedHTTPConnection("example.com", port=80)
        with patch("maki.connector.socket.getaddrinfo",
                   return_value=_addrinfo("93.184.216.34")), \
             patch("urllib3.connection.HTTPConnection._new_conn",
                   return_value=MagicMock()):
            conn._new_conn()
        self.assertEqual(conn._dns_host, "93.184.216.34")

    def test_https_connection_preserves_sni_hostname(self):
        conn = _PinnedHTTPSConnection("example.com", port=443)
        with patch("maki.connector.socket.getaddrinfo",
                   return_value=_addrinfo("93.184.216.34")), \
             patch("urllib3.connection.HTTPSConnection._new_conn",
                   return_value=MagicMock()):
            conn._new_conn()
        self.assertEqual(conn._dns_host, "93.184.216.34")
        # TLS verification must still target the original hostname
        self.assertEqual(conn.server_hostname, "example.com")

    def test_connection_falls_back_through_validated_addresses(self):
        """Dual-stack: if the first resolved address refuses the connection
        (e.g. service listens on IPv4 only), the next validated address is
        tried instead of failing outright."""
        conn = _PinnedHTTPConnection("example.com", port=80)
        attempted = []

        def fake_new_conn(self_conn):
            attempted.append(self_conn._dns_host)
            if self_conn._dns_host == "1.1.1.1":
                raise ConnectionRefusedError("refused")
            return MagicMock()

        with patch("maki.connector.socket.getaddrinfo",
                   return_value=_addrinfo("1.1.1.1", "2.2.2.2")), \
             patch("urllib3.connection.HTTPConnection._new_conn",
                   fake_new_conn):
            conn._new_conn()
        self.assertEqual(attempted, ["1.1.1.1", "2.2.2.2"])

    def test_http_connection_blocks_private_resolution(self):
        conn = _PinnedHTTPConnection("rebind.attacker.com", port=80)
        with patch("maki.connector.socket.getaddrinfo",
                   return_value=_addrinfo("169.254.169.254")):
            with self.assertRaises(MakiValidationError):
                conn._new_conn()

    def test_ssrf_connector_mounts_pinning_adapter(self):
        conn = Connector()
        self.assertIsInstance(conn._session.get_adapter("http://x"), _SSRFAdapter)
        self.assertIsInstance(conn._session.get_adapter("https://x"), _SSRFAdapter)

    def test_allow_private_connector_skips_pinning_adapter(self):
        conn = Connector(allow_private=True)
        self.assertNotIsInstance(conn._session.get_adapter("http://x"), _SSRFAdapter)


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

class TestErrorClassification(unittest.TestCase):

    def _connector_with_session_error(self, exc):
        conn = Connector(allow_private=True)
        conn._session.get = MagicMock(side_effect=exc)
        return conn

    def test_transport_timeout(self):
        conn = self._connector_with_session_error(
            requests.exceptions.Timeout("slow"))
        with self.assertRaises(MakiTimeoutError):
            conn.get("http://localhost:1/x")

    def test_transport_connection_error(self):
        conn = self._connector_with_session_error(
            requests.exceptions.ConnectionError("refused"))
        with self.assertRaises(MakiNetworkError):
            conn.get("http://localhost:1/x")

    def test_other_request_exception(self):
        conn = self._connector_with_session_error(
            requests.exceptions.ChunkedEncodingError("cut off"))
        with self.assertRaises(MakiNetworkError):
            conn.get("http://localhost:1/x")

    def _response(self, status, text="detail"):
        return MagicMock(status_code=status, text=text)

    def test_status_mapping(self):
        Connector.raise_for_response(self._response(200))  # no raise
        Connector.raise_for_response(self._response(302))  # no raise
        with self.assertRaises(MakiTimeoutError):
            Connector.raise_for_response(self._response(408))
        with self.assertRaises(MakiTimeoutError):
            Connector.raise_for_response(self._response(504))
        with self.assertRaises(MakiNetworkError):
            Connector.raise_for_response(self._response(500))
        with self.assertRaises(MakiAPIError):
            Connector.raise_for_response(self._response(404))
        with self.assertRaises(MakiAPIError):
            Connector.raise_for_response(self._response(400))

    def test_raise_on_status_false_returns_response(self):
        conn = Connector(allow_private=True)
        conn._session.get = MagicMock(return_value=self._response(500))
        resp = conn.get("http://localhost:1/x", raise_on_status=False)
        self.assertEqual(resp.status_code, 500)

    def test_json_or_raise_invalid_json(self):
        bad = MagicMock()
        bad.json.side_effect = ValueError("not json")
        with self.assertRaises(MakiAPIError):
            Connector.json_or_raise(bad)

    def test_iter_lines_maps_midstream_errors(self):
        response = MagicMock()
        response.iter_lines.side_effect = \
            requests.exceptions.ConnectionError("reset")
        with self.assertRaises(MakiNetworkError):
            list(Connector.iter_lines(response))


# ---------------------------------------------------------------------------
# Timeouts come from config, not hardcoded values
# ---------------------------------------------------------------------------

class TestTimeouts(unittest.TestCase):

    def test_default_timeout_from_config(self):
        self.assertEqual(Connector().timeout, DEFAULT_REQUEST_TIMEOUT)

    def test_default_timeout_passed_to_session(self):
        conn = Connector(allow_private=True)
        conn._session.get = MagicMock(
            return_value=MagicMock(status_code=200))
        conn.get("http://localhost:1/x")
        _, kwargs = conn._session.get.call_args
        self.assertEqual(kwargs["timeout"], DEFAULT_REQUEST_TIMEOUT)

    def test_per_request_timeout_override(self):
        conn = Connector(allow_private=True, timeout=30)
        conn._session.get = MagicMock(
            return_value=MagicMock(status_code=200))
        conn.get("http://localhost:1/x", timeout=5)
        _, kwargs = conn._session.get.call_args
        self.assertEqual(kwargs["timeout"], 5)


# ---------------------------------------------------------------------------
# SSRF enforcement on the request path
# ---------------------------------------------------------------------------

class TestRequestValidation(unittest.TestCase):

    def test_private_url_blocked_before_any_io(self):
        conn = Connector()
        conn._session.get = MagicMock()
        with self.assertRaises(MakiValidationError):
            conn.get("http://192.168.1.1/internal")
        conn._session.get.assert_not_called()

    def test_bad_scheme_blocked(self):
        conn = Connector()
        with self.assertRaises(MakiValidationError):
            conn.get("file:///etc/passwd")

    def test_ssrf_protect_false_skips_validation(self):
        conn = Connector(ssrf_protect=False)
        conn._session.get = MagicMock(return_value=MagicMock(status_code=200))
        conn.get("http://192.168.1.1/internal")  # no raise
        conn._session.get.assert_called_once()


# ---------------------------------------------------------------------------
# AsyncConnector
# ---------------------------------------------------------------------------

class TestAsyncConnector(unittest.TestCase):

    def test_private_url_blocked(self):
        conn = AsyncConnector()
        with self.assertRaises(MakiValidationError):
            asyncio.run(conn.get("http://192.168.1.1/internal"))

    def test_hostname_resolution_validated(self):
        conn = AsyncConnector()
        with patch("maki.connector.socket.getaddrinfo",
                   return_value=_addrinfo("10.0.0.5")):
            with self.assertRaises(MakiValidationError):
                asyncio.run(conn.get("http://rebind.attacker.com/x"))

    def test_allow_private_skips_resolution(self):
        conn = AsyncConnector(allow_private=True)

        class _FakeClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def request(self, method, url, **kwargs):
                return MagicMock(status_code=200)

        import httpx
        with patch.object(httpx, "AsyncClient", _FakeClient), \
             patch("maki.connector.socket.getaddrinfo") as mock_resolve:
            resp = asyncio.run(conn.post("http://192.168.1.50:11434/api/chat"))
        self.assertEqual(resp.status_code, 200)
        mock_resolve.assert_not_called()

    def test_timeout_classification(self):
        import httpx

        class _TimeoutClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def request(self, method, url, **kwargs):
                raise httpx.TimeoutException("slow")

        conn = AsyncConnector(allow_private=True)
        with patch.object(httpx, "AsyncClient", _TimeoutClient):
            with self.assertRaises(MakiTimeoutError):
                asyncio.run(conn.post("http://localhost:11434/api/chat"))


if __name__ == "__main__":
    unittest.main()
