"""429 handling: Retry-After parsing, Connector/AsyncConnector retries, error metadata, fallback cooldown."""
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import requests

from maki.connector import AsyncConnector, Connector, rate_limit_wait
from maki.exceptions import MakiAPIError, error_for_status, parse_retry_after, retry_after_of


def response(status=200, headers=None, text="{}"):
    r = requests.Response()
    r.status_code = status
    r._content = text.encode()
    r._content_consumed = True
    r.headers.update(headers or {})
    return r


class TestParse(unittest.TestCase):
    def test_seconds_dates_and_junk(self):
        self.assertEqual(parse_retry_after("5"), 5.0)
        self.assertEqual(parse_retry_after(" 1.5 "), 1.5)
        self.assertEqual(parse_retry_after("-3"), 0.0)
        for bad in (None, "", "soon", "nan", "inf", True):
            with self.subTest(bad=bad):
                self.assertIsNone(parse_retry_after(bad))
        with patch("maki.exceptions.time.time", return_value=1_000_000_000):
            self.assertAlmostEqual(parse_retry_after("Sun, 09 Sep 2001 01:46:50 GMT"), 10.0, places=0)
        self.assertEqual(parse_retry_after("Mon, 01 Jan 1990 00:00:00 GMT"), 0.0)

    def test_a_zone_less_date_is_read_as_utc(self):
        with patch("maki.exceptions.time.time", return_value=1_000_000_000):
            self.assertAlmostEqual(parse_retry_after("Sun, 09 Sep 2001 01:46:50 -0000"), 10.0, places=0)

    def test_plain_dict_headers_are_matched_case_insensitively(self):
        self.assertEqual(retry_after_of(SimpleNamespace(headers={"Retry-After": "6"})), 6.0)
        self.assertEqual(retry_after_of(SimpleNamespace(headers={"RETRY-AFTER": "6"})), 6.0)

    def test_from_responses_and_sdk_errors(self):
        self.assertEqual(retry_after_of(response(429, {"Retry-After": "7"})), 7.0)
        self.assertEqual(retry_after_of(httpx.Response(429, headers={"retry-after": "2"})), 2.0)
        sdk_error = MagicMock(response=response(429, {"retry-after": "4"}))
        self.assertEqual(retry_after_of(sdk_error), 4.0)
        self.assertIsNone(retry_after_of(object()))
        self.assertIsNone(retry_after_of(MagicMock(response=None)))

    def test_error_carries_it(self):
        err = error_for_status(429, "slow down", retry_after=3.0)
        self.assertIsInstance(err, MakiAPIError)
        self.assertEqual((err.status_code, err.retry_after), (429, 3.0))
        self.assertIsNone(error_for_status(400, "bad").retry_after)


class TestWait(unittest.TestCase):
    def test_server_value_wins_up_to_the_cap(self):
        self.assertEqual(rate_limit_wait(response(429, {"Retry-After": "7"}), 0, 30), 7.0)
        self.assertIsNone(rate_limit_wait(response(429, {"Retry-After": "31"}), 0, 30))
        self.assertEqual(rate_limit_wait(response(429, {"Retry-After": "30"}), 1, 30), 30.0)

    def test_backoff_without_a_header(self):
        self.assertEqual([rate_limit_wait(response(429), n, 30) for n in range(3)], [1.0, 2.0, 4.0])
        self.assertEqual(rate_limit_wait(response(429), 10, 5), 5)


class TestSyncConnector(unittest.TestCase):
    def _connector(self, replies, **kw):
        c = Connector(ssrf_protect=False, **kw)
        c._session.post = MagicMock(side_effect=list(replies))
        return c

    def test_retries_then_succeeds_waiting_the_server_value(self):
        c = self._connector([response(429, {"Retry-After": "3"}), response(429, {"Retry-After": "1"}), response(200)])
        with patch("maki.connector.time.sleep") as sleep:
            self.assertEqual(c.post("http://x/y").status_code, 200)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [3.0, 1.0])
        self.assertEqual(c._session.post.call_count, 3)

    def test_gives_up_after_the_configured_retries_with_retry_after_on_the_error(self):
        c = self._connector([response(429, {"Retry-After": "2"})] * 3)
        with patch("maki.connector.time.sleep") as sleep, self.assertRaises(MakiAPIError) as caught:
            c.post("http://x/y")
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual((caught.exception.status_code, caught.exception.retry_after), (429, 2.0))

    def test_a_wait_above_the_cap_is_not_slept_and_raises_at_once(self):
        c = self._connector([response(429, {"Retry-After": "120"})])
        with patch("maki.connector.time.sleep") as sleep, self.assertRaises(MakiAPIError) as caught:
            c.post("http://x/y")
        sleep.assert_not_called()
        self.assertEqual(c._session.post.call_count, 1)
        self.assertEqual(caught.exception.retry_after, 120.0)

    def test_disabled_and_other_statuses(self):
        c = self._connector([response(429)], rate_limit_retries=0)
        with patch("maki.connector.time.sleep") as sleep, self.assertRaises(MakiAPIError):
            c.post("http://x/y")
        sleep.assert_not_called()
        for status in (400, 401, 404, 500):
            c = self._connector([response(status)])
            with patch("maki.connector.time.sleep") as sleep, self.assertRaises(Exception):
                c.post("http://x/y")
            sleep.assert_not_called()
            self.assertEqual(c._session.post.call_count, 1)

    def test_raise_on_status_false_still_retries_but_returns_the_final_response(self):
        c = self._connector([response(429), response(429), response(429)])
        with patch("maki.connector.time.sleep"):
            self.assertEqual(c.post("http://x/y", raise_on_status=False).status_code, 429)

    def test_a_consumed_body_is_not_replayed(self):
        import io
        for kwargs in ({"data": io.BytesIO(b"x")}, {"data": iter([b"x"])}, {"files": {"f": io.BytesIO(b"x")}},
                       {"files": {"f": ("n", io.BytesIO(b"x"))}}):
            with self.subTest(kwargs=list(kwargs)):
                c = self._connector([response(429)])
                with patch("maki.connector.time.sleep") as sleep, self.assertRaises(MakiAPIError):
                    c.post("http://x/y", **kwargs)
                sleep.assert_not_called()
                self.assertEqual(c._session.post.call_count, 1)
        c = self._connector([response(429), response(200)])  # json and bytes bodies replay fine
        with patch("maki.connector.time.sleep"):
            self.assertEqual(c.post("http://x/y", json={"a": 1}, data=None).status_code, 200)

    def test_streaming_responses_are_retried_and_closed(self):
        first = response(429)
        first.close = MagicMock()
        c = self._connector([first, response(200)])
        with patch("maki.connector.time.sleep"):
            c.post("http://x/y", stream=True)
        first.close.assert_called_once()


class TestAsyncConnector(unittest.TestCase):
    def _run(self, handler, **kw):
        """Run one POST through AsyncConnector against *handler*; returns (sleep mock, result or error)."""
        real = httpx.AsyncClient
        connector = AsyncConnector(ssrf_protect=False, **kw)

        async def no_sleep(seconds):
            return None
        with patch("httpx.AsyncClient", lambda **k: real(transport=httpx.MockTransport(handler), **k)), \
                patch("maki.connector.asyncio.sleep", side_effect=no_sleep) as sleep:
            try:
                return sleep, asyncio.run(connector.post("http://x/y", json={}))
            except MakiAPIError as exc:
                return sleep, exc

    def test_retries_then_succeeds(self):
        calls = []

        def handler(request):
            calls.append(1)
            return httpx.Response(429, headers={"Retry-After": "2"}) if len(calls) < 3 else httpx.Response(200, json={})
        sleep, result = self._run(handler)
        self.assertEqual(result.status_code, 200)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [2.0, 2.0])

    def test_gives_up_and_over_the_cap(self):
        sleep, result = self._run(lambda r: httpx.Response(429, headers={"Retry-After": "9"}))
        self.assertIsInstance(result, MakiAPIError)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(result.retry_after, 9.0)
        sleep, result = self._run(lambda r: httpx.Response(429, headers={"Retry-After": "99"}))
        self.assertIsInstance(result, MakiAPIError)
        sleep.assert_not_called()


class TestFallbackUsesRetryAfter(unittest.TestCase):
    def test_a_rate_limited_backend_stays_out_for_the_asked_time(self):
        from maki import FallbackBackend
        from maki.test.test_fallback import Fake
        limited = Fake("a", error=MakiAPIError("slow", status_code=429, retry_after=120.0), fail_times=1)
        other = Fake("b")
        fb = FallbackBackend([limited, other], cooldown=30)
        with patch("maki.fallback.time.monotonic", return_value=0.0):
            fb.request("1")
        with patch("maki.fallback.time.monotonic", return_value=60.0):  # past the 30s cooldown, inside 120s
            self.assertEqual(fb.request("2").content, "b")
        with patch("maki.fallback.time.monotonic", return_value=121.0):
            self.assertEqual(fb.request("3").content, "a")

    def test_without_retry_after_the_plain_cooldown_applies(self):
        from maki import FallbackBackend
        from maki.test.test_fallback import Fake
        fb = FallbackBackend([Fake("a", error=MakiAPIError("slow", status_code=429), fail_times=1), Fake("b")],
                             cooldown=30)
        with patch("maki.fallback.time.monotonic", return_value=0.0):
            fb.request("1")
        with patch("maki.fallback.time.monotonic", return_value=31.0):
            self.assertEqual(fb.request("2").content, "a")


if __name__ == "__main__":
    unittest.main()


class TestRedaction(unittest.TestCase):
    def test_credential_parameters_are_masked_and_the_rest_is_kept(self):
        from maki.connector import redact
        cases = {
            "GET https://x.test/v1?api_key=SECRET123&q=cats failed": "GET https://x.test/v1?api_key=***&q=cats failed",
            "url: /p?q=1&token=abc.def-1_2&z=3": "url: /p?q=1&token=***&z=3",
            "https://x.test/?KEY=AbC": "https://x.test/?KEY=***",
            "https://x.test/?access_token=zzz#frag": "https://x.test/?access_token=***#frag",
            "https://x.test/?password=p%40ss&user=bob": "https://x.test/?password=***&user=bob",
            "('Connection aborted.', url: /a?signature=ff00&b=2)": "('Connection aborted.', url: /a?signature=***&b=2)",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(redact(text), expected)
        self.assertEqual(redact("no secrets here: key words"), "no secrets here: key words")
        self.assertEqual(redact(12345), "12345")

    def test_errors_from_the_connector_do_not_carry_the_key(self):
        c = Connector(ssrf_protect=False, rate_limit_retries=0)
        c._session.get = MagicMock(side_effect=requests.exceptions.ConnectionError(
            "Max retries exceeded with url: /v1?api_key=TOPSECRET&q=1"))
        with self.assertRaises(Exception) as caught:
            c.get("http://x.test/v1?api_key=TOPSECRET&q=1")
        self.assertNotIn("TOPSECRET", str(caught.exception))
        self.assertIn("api_key=***", str(caught.exception))

    def test_the_retry_log_line_does_not_carry_the_key(self):
        c = Connector(ssrf_protect=False)
        c._session.get = MagicMock(side_effect=[response(429, {"Retry-After": "1"}), response(200)])
        with patch("maki.connector.time.sleep"), self.assertLogs("maki.connector", "WARNING") as logs:
            c.get("http://x.test/v1?token=TOPSECRET")
        self.assertNotIn("TOPSECRET", "\n".join(logs.output))
        self.assertIn("token=***", "\n".join(logs.output))

    def test_async_errors_are_masked_too(self):
        async def run():
            def handler(request):
                raise httpx.ConnectError("boom for https://x.test/?api_key=TOPSECRET")
            real = httpx.AsyncClient
            with patch("httpx.AsyncClient", lambda **k: real(transport=httpx.MockTransport(handler), **k)):
                return await AsyncConnector(ssrf_protect=False).get("http://x.test/?api_key=TOPSECRET")
        with self.assertRaises(Exception) as caught:
            asyncio.run(run())
        self.assertNotIn("TOPSECRET", str(caught.exception))
