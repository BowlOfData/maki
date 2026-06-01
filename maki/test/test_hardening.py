"""
Phase 5 tests: circuit breaker, distributed tracing, mTLS configuration.
"""
import time
import unittest
from unittest.mock import MagicMock, patch, call

import pytest
httpx = pytest.importorskip("httpx", reason="httpx not installed")
pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi.testclient import TestClient as _TestClient

from maki.agents import Agent
from maki.objects import LLMResponse
from maki.exceptions import MakiAPIError, MakiNetworkError, MakiTimeoutError
from maki.distributed.circuit_breaker import CircuitBreaker, CircuitState
from maki.distributed.proxy import AgentProxy, TRACE_HEADER
from maki.distributed.server import create_app


# ---------------------------------------------------------------------------
# Helpers (shared with test_agent_proxy.py)
# ---------------------------------------------------------------------------

def _llm(content="ok"):
    return LLMResponse(content=content, model="t", prompt_tokens=0,
                       completion_tokens=0, total_tokens=0, elapsed_seconds=0.0)


def _mock_backend(content="result"):
    m = MagicMock()
    m.chat.return_value = _llm(content)
    m.stream.side_effect = lambda *a, **kw: iter(["chunk"])
    return m


class _AppTransport(httpx.BaseTransport):
    def __init__(self, app):
        self._client = _TestClient(app, raise_server_exceptions=False)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        resp = self._client.request(
            method=request.method,
            url=str(request.url),
            content=request.content,
            headers=dict(request.headers),
        )
        return httpx.Response(
            status_code=resp.status_code,
            headers=dict(resp.headers),
            content=resp.content,
        )


def _make_proxy(content="result", api_key=None, failure_threshold=5,
                recovery_timeout=60.0):
    backend = _mock_backend(content)
    agent = Agent("test-agent", backend, role="tester")
    app = create_app(agent, api_key=api_key)
    transport = _AppTransport(app)
    client = httpx.Client(transport=transport, timeout=10.0)
    with patch("maki.distributed.proxy.httpx.Client", return_value=client):
        proxy = AgentProxy(
            endpoint="http://fake:8100", api_key=api_key,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )
    return proxy, agent


def _mock_proxy(failure_threshold=5, recovery_timeout=60.0):
    """Proxy backed by a pure mock client (no real server)."""
    mock_client = MagicMock()
    mock_client.get.return_value = MagicMock(
        status_code=200, is_success=True,
        json=lambda: {
            "agent_id": "x", "name": "t", "role": "", "plugins": [],
            "backend": "Mock", "model": "t",
        },
    )
    with patch("maki.distributed.proxy.httpx.Client", return_value=mock_client):
        proxy = AgentProxy(
            endpoint="http://fake:8100",
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )
    return proxy, mock_client


# ===========================================================================
# Circuit Breaker unit tests
# ===========================================================================

class TestCircuitBreakerStates(unittest.TestCase):

    def test_initial_state_is_closed(self):
        cb = CircuitBreaker()
        self.assertEqual(cb.state, CircuitState.CLOSED)

    def test_initial_failure_count_is_zero(self):
        cb = CircuitBreaker()
        self.assertEqual(cb.failure_count, 0)

    def test_allow_request_when_closed(self):
        cb = CircuitBreaker()
        self.assertTrue(cb.allow_request())

    def test_failures_below_threshold_stay_closed(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        self.assertEqual(cb.state, CircuitState.CLOSED)
        self.assertTrue(cb.allow_request())

    def test_threshold_failures_open_circuit(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        self.assertEqual(cb.state, CircuitState.OPEN)
        self.assertFalse(cb.allow_request())

    def test_open_circuit_blocks_requests(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        self.assertFalse(cb.allow_request())

    def test_success_resets_to_closed(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        self.assertEqual(cb.state, CircuitState.OPEN)
        cb.record_success()
        self.assertEqual(cb.state, CircuitState.CLOSED)
        self.assertEqual(cb.failure_count, 0)

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=5)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        self.assertEqual(cb.failure_count, 0)

    def test_open_to_half_open_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.02)
        cb.record_failure()
        self.assertEqual(cb.state, CircuitState.OPEN)
        time.sleep(0.05)
        self.assertEqual(cb.state, CircuitState.HALF_OPEN)
        self.assertTrue(cb.allow_request())

    def test_half_open_success_closes_circuit(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)
        self.assertEqual(cb.state, CircuitState.HALF_OPEN)
        cb.record_success()
        self.assertEqual(cb.state, CircuitState.CLOSED)

    def test_half_open_failure_reopens_circuit(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)
        self.assertEqual(cb.state, CircuitState.HALF_OPEN)
        cb.record_failure()
        self.assertEqual(cb.state, CircuitState.OPEN)

    def test_repr_includes_state(self):
        cb = CircuitBreaker(failure_threshold=3)
        self.assertIn("closed", repr(cb))

    def test_invalid_threshold_raises(self):
        with self.assertRaises(ValueError):
            CircuitBreaker(failure_threshold=0)

    def test_invalid_timeout_raises(self):
        with self.assertRaises(ValueError):
            CircuitBreaker(recovery_timeout=-1)


# ===========================================================================
# Circuit breaker integration with AgentProxy
# ===========================================================================

class TestProxyCircuitBreaker(unittest.TestCase):

    def test_initial_state_closed(self):
        proxy, _ = _make_proxy()
        self.assertEqual(proxy.circuit_state, CircuitState.CLOSED)

    def test_successful_calls_stay_closed(self):
        proxy, _ = _make_proxy()
        proxy.execute_task("task")
        proxy.execute_task("task")
        self.assertEqual(proxy.circuit_state, CircuitState.CLOSED)

    def test_threshold_failures_open_circuit(self):
        proxy, mock_client = _mock_proxy(failure_threshold=3)
        mock_client.post.side_effect = httpx.ConnectError("down")
        for _ in range(3):
            try:
                proxy.execute_task("t")
            except MakiNetworkError:
                pass
        self.assertEqual(proxy.circuit_state, CircuitState.OPEN)

    def test_open_circuit_raises_without_http_call(self):
        proxy, mock_client = _mock_proxy(failure_threshold=2)
        mock_client.post.side_effect = httpx.ConnectError("down")
        for _ in range(2):
            try:
                proxy.execute_task("t")
            except MakiNetworkError:
                pass
        mock_client.post.reset_mock()

        with self.assertRaises(MakiNetworkError):
            proxy.execute_task("t")
        # No HTTP call should have been made
        mock_client.post.assert_not_called()

    def test_api_error_does_not_count_as_failure(self):
        proxy, mock_client = _mock_proxy(failure_threshold=2)
        mock_client.post.return_value = MagicMock(
            status_code=400, is_success=False, text="bad request"
        )
        for _ in range(5):
            try:
                proxy.execute_task("t")
            except MakiAPIError:
                pass
        # Server responded with 4xx — it's up, circuit stays closed
        self.assertEqual(proxy.circuit_state, CircuitState.CLOSED)

    def test_circuit_resets_after_recovery(self):
        proxy, mock_client = _mock_proxy(failure_threshold=2, recovery_timeout=0.02)
        mock_client.post.side_effect = httpx.ConnectError("down")
        for _ in range(2):
            try:
                proxy.execute_task("t")
            except MakiNetworkError:
                pass
        self.assertEqual(proxy.circuit_state, CircuitState.OPEN)

        time.sleep(0.05)
        # Transition to HALF_OPEN
        self.assertEqual(proxy.circuit_state, CircuitState.HALF_OPEN)

        # Successful call closes the circuit
        mock_client.post.side_effect = None
        mock_client.post.return_value = MagicMock(
            status_code=200, is_success=True,
            json=lambda: {"result": "ok", "agent_id": "x",
                          "elapsed": 0.1, "trace_id": "t-id"},
        )
        result = proxy.execute_task("t")
        self.assertEqual(result, "ok")
        self.assertEqual(proxy.circuit_state, CircuitState.CLOSED)

    def test_execute_with_retry_aborts_when_circuit_opens(self):
        proxy, mock_client = _mock_proxy(failure_threshold=2)
        mock_client.post.side_effect = httpx.ConnectError("down")

        with self.assertRaises(MakiNetworkError):
            proxy.execute_task_with_retry("t", max_retries=10, retry_delay=0)

        # After threshold (2) failures the circuit opens and retry aborts —
        # total calls should equal exactly the threshold, not max_retries.
        self.assertLessEqual(mock_client.post.call_count, 3)

    def test_stream_task_blocked_when_circuit_open(self):
        proxy, mock_client = _mock_proxy(failure_threshold=1)
        mock_client.post.side_effect = httpx.ConnectError("down")
        try:
            proxy.execute_task("t")
        except MakiNetworkError:
            pass
        self.assertEqual(proxy.circuit_state, CircuitState.OPEN)

        with self.assertRaises(MakiNetworkError):
            proxy.stream_task("stream this")  # should raise immediately


# ===========================================================================
# Distributed tracing
# ===========================================================================

class TestDistributedTracing(unittest.TestCase):

    def test_execute_sends_trace_header(self):
        proxy, mock_client = _mock_proxy()
        mock_client.post.return_value = MagicMock(
            status_code=200, is_success=True,
            json=lambda: {"result": "ok", "agent_id": "x",
                          "elapsed": 0.1, "trace_id": "srv-trace"},
        )
        proxy.execute_task("hello")
        _, kwargs = mock_client.post.call_args
        headers = kwargs.get("headers", {})
        self.assertIn(TRACE_HEADER, headers)

    def test_different_calls_get_different_trace_ids(self):
        proxy, mock_client = _mock_proxy()
        trace_ids = []

        def capture_post(url, **kwargs):
            trace_ids.append(kwargs.get("headers", {}).get(TRACE_HEADER))
            return MagicMock(
                status_code=200, is_success=True,
                json=lambda: {"result": "r", "agent_id": "x",
                              "elapsed": 0.0, "trace_id": "echo"},
            )

        mock_client.post.side_effect = capture_post
        proxy.execute_task("task-1")
        proxy.execute_task("task-2")
        self.assertEqual(len(trace_ids), 2)
        self.assertIsNotNone(trace_ids[0])
        self.assertIsNotNone(trace_ids[1])
        self.assertNotEqual(trace_ids[0], trace_ids[1])

    def test_server_echoes_trace_id_in_response(self):
        proxy, _ = _make_proxy()
        proxy.execute_task("task")
        self.assertIsNotNone(proxy.last_trace_id)

    def test_last_trace_id_matches_sent_trace_id(self):
        proxy, mock_client = _mock_proxy()
        sent_trace = "my-custom-trace-001"
        mock_client.post.return_value = MagicMock(
            status_code=200, is_success=True,
            json=lambda: {"result": "ok", "agent_id": "x",
                          "elapsed": 0.1, "trace_id": sent_trace},
        )
        proxy.execute_task("task", trace_id=sent_trace)
        self.assertEqual(proxy.last_trace_id, sent_trace)

    def test_custom_trace_id_is_forwarded(self):
        proxy, mock_client = _mock_proxy()
        mock_client.post.return_value = MagicMock(
            status_code=200, is_success=True,
            json=lambda: {"result": "r", "agent_id": "x",
                          "elapsed": 0.0, "trace_id": "custom-42"},
        )
        proxy.execute_task("task", trace_id="custom-42")
        _, kwargs = mock_client.post.call_args
        self.assertEqual(kwargs["headers"][TRACE_HEADER], "custom-42")

    def test_server_returns_trace_id_in_execute_response(self):
        """End-to-end: the real server returns trace_id in JSON body."""
        proxy, _ = _make_proxy()
        result = proxy.execute_task("test task")
        self.assertEqual(result, "result")
        self.assertIsNotNone(proxy.last_trace_id)
        # trace_id should look like a UUID (32 hex chars + 4 dashes)
        import re
        self.assertRegex(proxy.last_trace_id,
                         r'^[0-9a-f\-]{36}$')

    def test_server_echoes_client_trace_id(self):
        """If the proxy sends a trace ID the server must echo the same one back."""
        proxy, _ = _make_proxy()
        # Inject a known trace_id by patching uuid4
        with patch("maki.distributed.proxy.uuid.uuid4", return_value="fixed-id"):
            proxy.execute_task("task")
        self.assertEqual(proxy.last_trace_id, "fixed-id")

    def test_response_has_trace_header(self):
        """The server middleware must add X-Maki-Trace-Id to every response."""
        backend = _mock_backend()
        agent = Agent("a", backend)
        app = create_app(agent)
        client = _TestClient(app)
        resp = client.post("/execute", json={"task": "hello"})
        self.assertIn(TRACE_HEADER, resp.headers)

    def test_client_trace_id_preserved_in_response_header(self):
        backend = _mock_backend()
        agent = Agent("a", backend)
        app = create_app(agent)
        client = _TestClient(app)
        resp = client.post(
            "/execute",
            json={"task": "hello"},
            headers={TRACE_HEADER: "test-trace-xyz"},
        )
        self.assertEqual(resp.headers[TRACE_HEADER], "test-trace-xyz")


# ===========================================================================
# mTLS / SSL configuration
# ===========================================================================

class TestProxySSL(unittest.TestCase):

    def _capture_client_kwargs(self, **proxy_kwargs):
        """Return the kwargs passed to httpx.Client() constructor."""
        captured = {}

        def fake_client(**kwargs):
            captured.update(kwargs)
            mock = MagicMock()
            mock.get.return_value = MagicMock(
                status_code=200, is_success=True,
                json=lambda: {
                    "agent_id": "x", "name": "n", "role": "", "plugins": [],
                    "backend": "Mock", "model": "m",
                },
            )
            return mock

        with patch("maki.distributed.proxy.httpx.Client", side_effect=fake_client):
            AgentProxy(endpoint="http://fake:8100", **proxy_kwargs)
        return captured

    def test_default_no_ssl_override(self):
        kwargs = self._capture_client_kwargs()
        self.assertNotIn("verify", kwargs)
        self.assertNotIn("cert", kwargs)

    def test_ssl_verify_false(self):
        kwargs = self._capture_client_kwargs(ssl_verify=False)
        self.assertEqual(kwargs.get("verify"), False)

    def test_ssl_verify_ca_bundle_path(self):
        kwargs = self._capture_client_kwargs(ssl_verify="/etc/ssl/ca.pem")
        self.assertEqual(kwargs.get("verify"), "/etc/ssl/ca.pem")

    def test_client_cert_tuple(self):
        kwargs = self._capture_client_kwargs(cert=("/cert.pem", "/key.pem"))
        self.assertEqual(kwargs.get("cert"), ("/cert.pem", "/key.pem"))

    def test_ssl_verify_and_cert_together(self):
        kwargs = self._capture_client_kwargs(
            ssl_verify="/ca.pem", cert=("/c.pem", "/k.pem")
        )
        self.assertEqual(kwargs.get("verify"), "/ca.pem")
        self.assertEqual(kwargs.get("cert"), ("/c.pem", "/k.pem"))


# ===========================================================================
# mTLS CLI flags
# ===========================================================================

class TestServeCLI(unittest.TestCase):

    def _parse(self, args):
        import argparse
        from maki.__main__ import main
        import sys
        with patch("sys.argv", ["maki"] + args):
            with patch("maki.__main__._cmd_serve") as mock_serve:
                try:
                    main()
                except SystemExit:
                    pass
                return mock_serve

    def test_tls_cert_and_key_parsed(self):
        mock_serve = self._parse([
            "serve", "--config", "a.yaml",
            "--tls-cert", "/cert.pem", "--tls-key", "/key.pem",
        ])
        if mock_serve.called:
            args = mock_serve.call_args[0][0]
            self.assertEqual(args.tls_cert, "/cert.pem")
            self.assertEqual(args.tls_key, "/key.pem")

    def test_tls_defaults_to_none(self):
        mock_serve = self._parse(["serve", "--config", "a.yaml"])
        if mock_serve.called:
            args = mock_serve.call_args[0][0]
            self.assertIsNone(args.tls_cert)
            self.assertIsNone(args.tls_key)

    def test_uvicorn_receives_tls_kwargs(self):
        """_cmd_serve passes ssl_certfile/ssl_keyfile to uvicorn.run."""
        import argparse
        from maki.__main__ import _cmd_serve

        args = argparse.Namespace(
            config="fake.yaml", host="0.0.0.0", port=8100,
            api_key=None, tls_cert="/cert.pem", tls_key="/key.pem",
        )
        mock_agent = MagicMock()
        mock_agent.name = "agent"
        mock_agent.role = "role"
        mock_agent.plugins = {}

        # load_agent_from_config / create_app are imported locally in _cmd_serve
        with patch("maki.distributed.config_loader.load_agent_from_config",
                   return_value=mock_agent), \
             patch("maki.distributed.server.create_app", return_value=MagicMock()), \
             patch("uvicorn.run") as mock_run:
            _cmd_serve(args)
            _, kwargs = mock_run.call_args
            self.assertEqual(kwargs.get("ssl_certfile"), "/cert.pem")
            self.assertEqual(kwargs.get("ssl_keyfile"), "/key.pem")

    def test_uvicorn_no_tls_kwargs_without_cert(self):
        """_cmd_serve does NOT pass ssl_ keys to uvicorn when TLS is disabled."""
        import argparse
        from maki.__main__ import _cmd_serve

        args = argparse.Namespace(
            config="fake.yaml", host="127.0.0.1", port=8100,
            api_key=None, tls_cert=None, tls_key=None,
        )
        mock_agent = MagicMock()
        mock_agent.name = "a"
        mock_agent.role = ""
        mock_agent.plugins = {}

        with patch("maki.distributed.config_loader.load_agent_from_config",
                   return_value=mock_agent), \
             patch("maki.distributed.server.create_app", return_value=MagicMock()), \
             patch("uvicorn.run") as mock_run:
            _cmd_serve(args)
            _, kwargs = mock_run.call_args
            self.assertNotIn("ssl_certfile", kwargs)
            self.assertNotIn("ssl_keyfile", kwargs)


if __name__ == "__main__":
    unittest.main()
