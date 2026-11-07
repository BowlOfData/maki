"""Edge behaviour of the connector and the address validators that the mutation check showed untested:
URL and domain format limits, DNS answers, TLS settings, request-body replay, mid-stream failures, async errors."""
import asyncio
import io
import unittest
from unittest.mock import MagicMock, patch

import httpx
import requests

from maki.connector import (AsyncConnector, Connector, _replayable, _resolve_addresses, _resolve_and_validate,
                            rate_limit_wait, validate_url)
from maki.exceptions import (MakiAPIError, MakiNetworkError, MakiTimeoutError, MakiValidationError)
from maki.utils import Utils


def response(status=200, headers=None, text="{}"):
    r = requests.Response()
    r.status_code = status
    r._content = text.encode()
    r._content_consumed = True
    r.headers.update(headers or {})
    return r


class TestDomainValidation(unittest.TestCase):
    def test_empty_and_non_string_domains_are_refused(self):
        for bad in ("", "   ", None, 5, [], b"host"):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "non-empty string"):
                Utils._validate_domain(bad)

    def test_bracketed_ipv6_is_unwrapped_before_it_is_judged(self):
        Utils._validate_domain("[2606:4700:4700::1111]")                      # public: fine
        for bad in ("[::1]", "[fe80::1]", "[2002:7f00:1::1]", "[::ffff:10.0.0.1]"):
            with self.subTest(host=bad), self.assertRaises(ValueError):
                Utils._validate_domain(bad)
        with self.assertRaises(ValueError):                                    # one bracket is not a bracketed address
            Utils._validate_domain("[::1")
        with self.assertRaises(ValueError):
            Utils._validate_domain("::1]")

    def test_hostname_format_rules(self):
        for good in ("example.com", "a.b.c.example.org", "xn--bcher-kva.example", "EXAMPLE.COM", "a-b.example",
                     "1.example.com", "x"):
            with self.subTest(host=good):
                Utils._validate_domain(good)
        for bad in ("exa mple.com", "-bad.com", "bad-.com", "a..b", ".leading.com", "ex$ample.com", "ex_ample.com",
                    "exa\nmple.com", "a/b.com", "a@b.com", "例え.jp", "a.b.", "‮evil.com"):
            with self.subTest(host=bad), self.assertRaises(ValueError):
                Utils._validate_domain(bad)

    def test_length_limits_sit_exactly_where_dns_puts_them(self):
        label63 = "a" * 63
        Utils._validate_domain(label63 + ".com")                               # a 63-character label is allowed
        with self.assertRaises(ValueError):
            Utils._validate_domain("a" * 64 + ".com")
        ok253 = ".".join(["a" * 61] * 4 + ["b" * 7])[:253]
        while len(ok253) < 253:
            ok253 += "c"
        self.assertEqual(len(ok253), 253)
        Utils._validate_domain(ok253)
        with self.assertRaises(ValueError):
            Utils._validate_domain(ok253 + "d")                                # 254 characters

    def test_blacklisted_names_in_any_case(self):
        for host in ("0.0.0.0", "255.255.255.255", " 0.0.0.0 "):
            with self.subTest(host=host), self.assertRaises(ValueError):
                Utils._validate_domain(host)

    def test_allow_private_skips_only_the_address_check_not_the_format_check(self):
        Utils._validate_domain("10.0.0.1", allow_private=True)
        Utils._validate_domain("[fe80::1]", allow_private=True)
        with self.assertRaises(ValueError):
            Utils._validate_domain("bad host", allow_private=True)
        with self.assertRaises(ValueError):
            Utils._validate_domain("0.0.0.0", allow_private=True)              # the blacklist still applies

    def test_every_special_range_in_the_private_list_is_refused_at_both_ends(self):
        import ipaddress
        for cidr in Utils.PRIVATE_IP_RANGES:
            net = ipaddress.ip_network(cidr)
            for ip in (net[0], net[-1]):
                with self.subTest(cidr=cidr, ip=str(ip)), self.assertRaises(ValueError):
                    Utils._validate_ip(ip, str(ip))

    def test_the_address_just_outside_a_private_range_follows_the_global_rule(self):
        import ipaddress
        for outside in ("9.255.255.255", "11.0.0.0", "172.15.255.255", "172.32.0.0", "192.167.255.255", "192.169.0.0",
                        "100.63.255.255", "100.128.0.0", "169.253.255.255", "169.255.0.0"):
            ip = ipaddress.ip_address(outside)
            with self.subTest(ip=outside):
                if ip.is_global:
                    Utils._validate_ip(ip, outside)
                else:
                    with self.assertRaises(ValueError):
                        Utils._validate_ip(ip, outside)


class TestUrlAndResolution(unittest.TestCase):
    def test_non_string_or_empty_urls(self):
        for bad in ("", "  ", None, 7, b"http://x"):
            with self.subTest(bad=bad), self.assertRaisesRegex(MakiValidationError, "non-empty string"):
                validate_url(bad)

    def test_a_url_without_a_hostname(self):
        for url in ("http://", "http:///path", "https://:443/", "http://user@/x"):
            with self.subTest(url=url), self.assertRaisesRegex(MakiValidationError, "hostname"):
                validate_url(url)

    def test_malformed_urls_are_a_validation_error_not_a_bare_valueerror(self):
        for url in ("http://[::1", "http://[::1]:99999999999999999999/", "http://[zz]/"):
            with self.subTest(url=url), self.assertRaises(MakiValidationError):
                validate_url(url)

    def test_dns_answers(self):
        with patch("maki.connector.socket.getaddrinfo", return_value=[]):
            with self.assertRaisesRegex(MakiNetworkError, "no addresses"):
                _resolve_addresses("x.example", "x.example", 80)
        import socket
        with patch("maki.connector.socket.getaddrinfo", side_effect=socket.gaierror(-2, "unknown")):
            with self.assertRaisesRegex(MakiNetworkError, "DNS resolution failed for 'x.example'"):
                _resolve_addresses("x.example", "x.example", 80)
        infos = [(2, 1, 6, "", ("8.8.8.8", 80)), (2, 1, 6, "", ("8.8.8.8", 80)), (10, 1, 6, "", ("2606:4700::1111", 80, 0, 0))]
        with patch("maki.connector.socket.getaddrinfo", return_value=infos):
            self.assertEqual(_resolve_addresses("x.example", "x.example", 80), ["8.8.8.8", "8.8.8.8", "2606:4700::1111"])
            self.assertEqual(_resolve_and_validate("x.example", 80), ["8.8.8.8", "2606:4700::1111"])  # de-duplicated


class TestTlsAndSessionSettings(unittest.TestCase):
    def test_verification_is_on_by_default_and_the_overrides_reach_the_session(self):
        default = Connector(ssrf_protect=False)
        self.assertIs(default._session.verify, True)
        self.assertIsNone(default._session.cert)
        off = Connector(ssrf_protect=False, verify=False)
        self.assertIs(off._session.verify, False)
        bundle = Connector(ssrf_protect=False, verify="/etc/ssl/ca.pem", cert=("/c.pem", "/k.pem"))
        self.assertEqual(bundle._session.verify, "/etc/ssl/ca.pem")
        self.assertEqual(bundle._session.cert, ("/c.pem", "/k.pem"))

    def test_headers_and_timeouts(self):
        c = Connector(ssrf_protect=False, headers={"X-A": "1"}, timeout=7)
        self.assertEqual(c._session.headers["X-A"], "1")
        self.assertEqual(c.timeout, 7)
        c._session.get = MagicMock(return_value=response())
        c.get("http://x.test/")
        self.assertEqual(c._session.get.call_args.kwargs["timeout"], 7)
        c.get("http://x.test/", timeout=3)
        self.assertEqual(c._session.get.call_args.kwargs["timeout"], 3)

    def test_the_pinning_adapter_is_mounted_only_when_it_applies(self):
        self.assertIn("https://", {k for k in Connector(ssrf_protect=True)._session.adapters})
        pinned = Connector(ssrf_protect=True)._session.get_adapter("http://x.test/")
        plain = Connector(ssrf_protect=True, allow_private=True)._session.get_adapter("http://x.test/")
        unprotected = Connector(ssrf_protect=False)._session.get_adapter("http://x.test/")
        self.assertEqual(type(pinned).__name__, "_SSRFAdapter")
        self.assertNotEqual(type(plain).__name__, "_SSRFAdapter")
        self.assertNotEqual(type(unprotected).__name__, "_SSRFAdapter")

    def test_context_manager_returns_itself_and_closes(self):
        with Connector(ssrf_protect=False) as c:
            self.assertIsInstance(c, Connector)
            c._session.close = MagicMock()
        c._session.close.assert_called_once()

    def test_every_verb_goes_through_request(self):
        c = Connector(ssrf_protect=False)
        for verb in ("get", "post", "delete"):
            setattr(c._session, verb, MagicMock(return_value=response()))
            self.assertEqual(getattr(c, verb)("http://x.test/").status_code, 200)
            getattr(c._session, verb).assert_called_once()
        c._session.request = MagicMock(return_value=response(204))
        self.assertEqual(c.request("PURGE", "http://x.test/").status_code, 204)  # a verb requests has no shortcut for
        c._session.request.assert_called_once()


class TestReplayableBodies(unittest.TestCase):
    def test_streams_and_files_are_not_replayable_one_attribute_at_a_time(self):
        class OnlyRead:
            def read(self):
                return b""

        class OnlyNext:
            def __next__(self):
                raise StopIteration

        class OnlyAiter:
            def __aiter__(self):
                return self
        for key in ("data", "content"):
            for body in (OnlyRead(), OnlyNext(), OnlyAiter(), io.BytesIO(b"x"), iter([b"x"])):
                with self.subTest(key=key, body=type(body).__name__):
                    self.assertFalse(_replayable({key: body}))

    def test_plain_bodies_are_replayable(self):
        for kwargs in ({}, {"data": None}, {"data": b"bytes"}, {"data": "text"}, {"data": {"a": "1"}}, {"data": [("a", "1")]},
                       {"content": b"x"}, {"json": {"a": 1}}, {"params": {"q": "x"}}, {"files": None}, {"files": {}}):
            with self.subTest(kwargs=kwargs):
                self.assertTrue(_replayable(kwargs))

    def test_files_with_a_stream_are_not_replayable_in_every_shape(self):
        for files in ({"f": io.BytesIO(b"x")}, {"f": ("name", io.BytesIO(b"x"))}, {"f": ("n", io.BytesIO(b"x"), "text/plain")},
                      [("f", io.BytesIO(b"x"))], [("f", ("n", io.BytesIO(b"x")))]):
            with self.subTest(files=type(files).__name__):
                self.assertFalse(_replayable({"files": files}))
        self.assertTrue(_replayable({"files": {"f": ("n", b"bytes")}}))
        self.assertTrue(_replayable({"files": [("f", ("n", "text"))]}))


class TestRateLimitWait(unittest.TestCase):
    def test_boundaries(self):
        r = lambda h=None: response(429, h)  # noqa: E731
        self.assertEqual(rate_limit_wait(r({"Retry-After": "0"}), 0, 30), 0.0)
        self.assertEqual(rate_limit_wait(r({"Retry-After": "30"}), 0, 30), 30.0)
        self.assertIsNone(rate_limit_wait(r({"Retry-After": "30.001"}), 0, 30))
        self.assertEqual([rate_limit_wait(r(), n, 30) for n in range(7)], [1.0, 2.0, 4.0, 8.0, 16.0, 30, 30])
        self.assertEqual(rate_limit_wait(r(), 0, 0.5), 0.5)


class TestStreamFailures(unittest.TestCase):
    def _stream(self, error):
        def lines(**kw):
            yield b"first"
            raise error
        resp = MagicMock()
        resp.iter_lines = lines
        return resp

    def test_a_timeout_mid_body(self):
        got = []
        with self.assertRaisesRegex(MakiTimeoutError, "timed out mid-body"):
            for line in Connector.iter_lines(self._stream(requests.exceptions.ReadTimeout("slow"))):
                got.append(line)
        self.assertEqual(got, [b"first"])

    def test_any_other_transport_failure_mid_body(self):
        for error in (requests.exceptions.ChunkedEncodingError("cut"), requests.exceptions.ConnectionError("reset")):
            with self.subTest(error=type(error).__name__), self.assertRaisesRegex(MakiNetworkError, "failed mid-body"):
                list(Connector.iter_lines(self._stream(error)))

    def test_http_error_bodies_and_json(self):
        for status, fragment in ((408, "timeout"), (504, "timeout"), (500, "server error"), (503, "server error"),
                                 (400, "client error"), (404, "client error"), (429, "client error")):
            with self.subTest(status=status), self.assertRaisesRegex(Exception, fragment):
                Connector.raise_for_response(response(status, text="boom"))
        Connector.raise_for_response(response(399))
        Connector.raise_for_response(response(200))
        long_body = response(500, text="x" * 5000)
        with self.assertRaises(Exception) as caught:
            Connector.raise_for_response(long_body)
        self.assertLess(len(str(caught.exception)), 700)                       # the body is clipped
        unreadable = MagicMock(status_code=500)
        type(unreadable).text = property(lambda self: (_ for _ in ()).throw(RuntimeError("gone")))
        with self.assertRaisesRegex(Exception, "unreadable body"):
            Connector.raise_for_response(unreadable)
        with self.assertRaisesRegex(MakiAPIError, "Invalid JSON"):
            Connector.json_or_raise(response(200, text="not json"))
        self.assertEqual(Connector.json_or_raise(response(200, text='{"a": 1}')), {"a": 1})


class TestAsyncErrors(unittest.TestCase):
    def _call(self, error, **kw):
        def handler(request):
            raise error
        real = httpx.AsyncClient
        with patch("httpx.AsyncClient", lambda **k: real(transport=httpx.MockTransport(handler), **k)):
            return asyncio.run(AsyncConnector(ssrf_protect=False, **kw).get("http://x.test/"))

    def test_timeouts_connect_failures_and_other_request_errors_are_classified(self):
        for error, expected, text in ((httpx.ReadTimeout("slow"), MakiTimeoutError, "timed out"),
                                      (httpx.ConnectTimeout("slow"), MakiTimeoutError, "timed out"),
                                      (httpx.ConnectError("refused"), MakiNetworkError, "connection failed"),
                                      (httpx.ReadError("reset"), MakiNetworkError, "request failed"),
                                      (httpx.RemoteProtocolError("bad"), MakiNetworkError, "request failed")):
            with self.subTest(error=type(error).__name__), self.assertRaisesRegex(expected, text):
                self._call(error)

    def test_a_private_target_is_refused_before_any_request(self):
        sent = []

        def handler(request):
            sent.append(request)
            return httpx.Response(200)
        real = httpx.AsyncClient
        with patch("httpx.AsyncClient", lambda **k: real(transport=httpx.MockTransport(handler), **k)):
            for url in ("http://10.0.0.1/", "http://127.0.0.1/", "http://[::1]/", "http://169.254.169.254/"):
                with self.subTest(url=url), self.assertRaises(MakiValidationError):
                    asyncio.run(AsyncConnector().get(url))
            asyncio.run(AsyncConnector(allow_private=True).get("http://10.0.0.1/"))  # operator endpoints are exempt
        self.assertEqual(len(sent), 1)

    def test_the_default_timeout_and_header_handling(self):
        from maki.config import DEFAULT_REQUEST_TIMEOUT
        self.assertEqual(AsyncConnector(ssrf_protect=False).timeout, DEFAULT_REQUEST_TIMEOUT)
        self.assertEqual(AsyncConnector(ssrf_protect=False, timeout=3).timeout, 3)
        self.assertIsNone(AsyncConnector(ssrf_protect=False)._headers)
        self.assertEqual(AsyncConnector(ssrf_protect=False, headers={"X": "1"})._headers, {"X": "1"})
        self.assertEqual(AsyncConnector(rate_limit_retries=-4).rate_limit_retries, 0)


if __name__ == "__main__":
    unittest.main()
