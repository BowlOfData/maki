"""Production hardening of ``maki serve``: request-size limits, docs exposure, probe responsiveness, key handling."""
import argparse
import inspect
import io
import os
import unittest
from contextlib import redirect_stderr
from unittest.mock import MagicMock, patch

pytest = __import__("pytest")
fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")
TestClient = __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient

from maki import __version__  # noqa: E402
from maki.__main__ import _cmd_serve, _resolve_api_key, _warn_if_exposed  # noqa: E402
from maki.agents import Agent  # noqa: E402
from maki.distributed import server  # noqa: E402
from maki.distributed.server import DEFAULT_BODY_LIMIT, create_app  # noqa: E402
from maki.objects import LLMResponse  # noqa: E402


def _agent():
    backend = MagicMock()
    backend.chat.return_value = LLMResponse(content="ok", model="t", prompt_tokens=1, completion_tokens=1,
                                            total_tokens=2, elapsed_seconds=0.0)
    return Agent("a", backend, role="r")


def _client(**kw):
    return TestClient(create_app(_agent(), **kw), raise_server_exceptions=False)


class TestBodyLimit(unittest.TestCase):
    def test_the_default_limit_covers_every_endpoint_not_only_loops(self):
        client = _client(max_body_bytes=2000)
        small = client.post("/execute", json={"task": "hello"})
        self.assertEqual(small.status_code, 200)
        for path in ("/execute", "/stream", "/memory/key"):
            with self.subTest(path=path):
                big = client.post(path, content=b'{"task": "' + b"x" * 5000 + b'"}',
                                  headers={"content-type": "application/json"})
                self.assertEqual(big.status_code, 413)
                self.assertIn("larger than 2000", big.json()["detail"])

    def test_a_body_without_content_length_is_counted_as_it_arrives(self):
        client = _client(max_body_bytes=2000)

        def chunks():
            for _ in range(10):
                yield b"x" * 500
        response = client.post("/execute", content=chunks(), headers={"content-type": "application/json"})
        self.assertEqual(response.status_code, 413)

    def test_the_agent_never_ran_for_a_refused_request(self):
        agent = _agent()
        client = TestClient(create_app(agent, max_body_bytes=500), raise_server_exceptions=False)
        client.post("/execute", content=b'{"task": "' + b"x" * 5000 + b'"}', headers={"content-type": "application/json"})
        agent.maki.chat.assert_not_called()

    def test_loops_keep_their_own_smaller_limit(self):
        client = _client()  # the 4 MiB default
        big = client.post("/loop", content=b"x" * (300 * 1024), headers={"content-type": "application/json"})
        self.assertEqual(big.status_code, 413)  # refused before anything looks at the (missing) judge
        self.assertIn("262144", big.json()["detail"])

    def test_a_small_general_limit_also_caps_loops(self):
        client = _client(max_body_bytes=1000)
        big = client.post("/loop", content=b"x" * 2000, headers={"content-type": "application/json"})
        self.assertIn("1000", big.json()["detail"])

    def test_get_requests_are_unaffected(self):
        self.assertEqual(_client(max_body_bytes=10).get("/health").status_code, 200)

    def test_default_and_validation(self):
        self.assertEqual(DEFAULT_BODY_LIMIT, 4 * 1024 * 1024)
        for bad in (0, -1, 1.5, "big", True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                create_app(_agent(), max_body_bytes=bad)


class TestDocsExposure(unittest.TestCase):
    def test_interactive_docs_are_off_when_an_api_key_protects_the_server(self):
        client = _client(api_key="secret")
        for path in ("/docs", "/redoc", "/openapi.json"):
            with self.subTest(path=path):
                self.assertEqual(client.get(path).status_code, 404)

    def test_they_stay_available_for_development_without_a_key(self):
        client = _client()
        self.assertEqual(client.get("/docs").status_code, 200)
        self.assertEqual(client.get("/openapi.json").json()["info"]["version"], __version__)

    def test_authenticated_calls_still_work(self):
        client = _client(api_key="secret")
        self.assertEqual(client.post("/execute", json={"task": "t"}).status_code, 401)
        ok = client.post("/execute", json={"task": "t"}, headers={"Authorization": "Bearer secret"})
        self.assertEqual(ok.status_code, 200)


class TestHealthStaysResponsive(unittest.TestCase):
    def test_health_is_a_coroutine_so_it_needs_no_worker_thread(self):
        self.assertTrue(inspect.iscoroutinefunction(server.health))
        # blocking endpoints keep running in the thread pool
        self.assertFalse(inspect.iscoroutinefunction(server.execute))

    def test_health_needs_no_credentials_and_leaks_no_backend_details(self):
        body = _client(api_key="secret").get("/health").json()
        self.assertEqual(set(body), {"status", "agent_id", "name", "role"})


class Env:
    """Run a block with MAKI_API_KEY set or cleared."""

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        self.saved = os.environ.get("MAKI_API_KEY")
        if self.value is None:
            os.environ.pop("MAKI_API_KEY", None)
        else:
            os.environ["MAKI_API_KEY"] = self.value

    def __exit__(self, *exc):
        if self.saved is None:
            os.environ.pop("MAKI_API_KEY", None)
        else:
            os.environ["MAKI_API_KEY"] = self.saved


def _args(**kw):
    base = dict(config="x.yaml", host="127.0.0.1", port=8100, api_key=None, tls_cert=None, tls_key=None)
    base.update(kw)
    return argparse.Namespace(**base)


class TestMemoryIsBounded(unittest.TestCase):
    def test_new_keys_stop_at_the_limit_but_existing_ones_can_change(self):
        agent = _agent()
        agent.MAX_MEMORY_ENTRIES = 3
        for i in range(3):
            agent.remember(f"k{i}", i)
        with self.assertRaisesRegex(ValueError, "memory is full"):
            agent.remember("one-too-many", 1)
        agent.remember("k1", "updated")          # an existing key is not a new entry
        self.assertEqual(agent.recall("k1"), "updated")
        agent.memory.pop("k0")
        agent.remember("fits-now", 1)

    def test_keys_must_be_short_non_empty_strings(self):
        agent = _agent()
        for bad in ("", "k" * 513, None, 5):
            with self.subTest(key=bad), self.assertRaises(ValueError):
                agent.remember(bad, 1)
        agent.remember("k" * 512, 1)

    def test_the_http_endpoint_refuses_beyond_the_limit_without_a_crash(self):
        agent = _agent()
        agent.MAX_MEMORY_ENTRIES = 2
        client = TestClient(create_app(agent), raise_server_exceptions=False)
        codes = [client.post("/memory/set", json={"key": f"k{i}", "value": "v"}).status_code for i in range(4)]
        self.assertEqual(codes, [200, 200, 409, 409])
        self.assertEqual(client.post("/memory/set", json={"key": "k0", "value": "new"}).status_code, 200)


class TestApiKeyHandling(unittest.TestCase):
    def test_the_environment_supplies_the_key_without_a_warning(self):
        args = _args()
        err = io.StringIO()
        with Env("from-env"), redirect_stderr(err):
            _resolve_api_key(args)
        self.assertEqual(args.api_key, "from-env")
        self.assertEqual(err.getvalue(), "")

    def test_the_flag_wins_and_comes_with_a_warning_about_the_process_list(self):
        args = _args(api_key="from-flag")
        err = io.StringIO()
        with Env("from-env"), redirect_stderr(err):
            _resolve_api_key(args)
        self.assertEqual(args.api_key, "from-flag")
        self.assertIn("process list", err.getvalue())
        self.assertIn("MAKI_API_KEY", err.getvalue())

    def test_an_empty_variable_is_no_key(self):
        args = _args()
        with Env(""):
            _resolve_api_key(args)
        self.assertIsNone(args.api_key)


class TestExposureWarnings(unittest.TestCase):
    def _warnings(self, **kw):
        err = io.StringIO()
        with redirect_stderr(err):
            _warn_if_exposed(_args(**kw))
        return err.getvalue()

    def test_an_open_server_on_a_network_address_warns_about_auth_and_tls(self):
        text = self._warnings(host="0.0.0.0")
        self.assertIn("no API key", text)
        self.assertIn("no TLS", text)

    def test_with_a_key_but_no_tls_it_stays_quiet(self):
        self.assertEqual(self._warnings(host="0.0.0.0", api_key="k"), "")

    def test_loopback_never_warns(self):
        for host in ("127.0.0.1", "localhost", "::1"):
            self.assertEqual(self._warnings(host=host), "")

    def test_tls_without_a_key_still_warns_about_the_key_only(self):
        text = self._warnings(host="0.0.0.0", tls_cert="c.pem")
        self.assertIn("no API key", text)
        self.assertNotIn("no TLS", text)


class TestServeWiring(unittest.TestCase):
    def _serve(self, **kw):
        agent = MagicMock()
        agent.name, agent.role, agent.plugins = "a", "", {}
        with patch("maki.distributed.config_loader.load_server_config",
                   return_value=MagicMock(agent=agent, judge=None, loop_caps=None)), \
                patch("maki.distributed.server.create_app", return_value=MagicMock()) as make, \
                patch("uvicorn.run") as run, Env(None), redirect_stderr(io.StringIO()):
            _cmd_serve(_args(**kw))
        return make, run

    def test_limits_reach_the_app_and_uvicorn(self):
        make, run = self._serve(max_body_mb=1.5, max_concurrency=8)
        self.assertEqual(make.call_args.kwargs["max_body_bytes"], int(1.5 * 1024 * 1024))
        self.assertEqual(run.call_args.kwargs["limit_concurrency"], 8)

    def test_shutdown_is_bounded_by_default_and_configurable(self):
        self.assertEqual(self._serve()[1].call_args.kwargs["timeout_graceful_shutdown"], 30.0)
        self.assertEqual(self._serve(graceful_timeout=5)[1].call_args.kwargs["timeout_graceful_shutdown"], 5)

    def test_defaults_add_no_uvicorn_limit(self):
        make, run = self._serve()
        self.assertEqual(make.call_args.kwargs["max_body_bytes"], DEFAULT_BODY_LIMIT)
        self.assertNotIn("limit_concurrency", run.call_args.kwargs)

    def test_non_positive_limits_are_refused(self):
        for kw in (dict(max_body_mb=0), dict(max_body_mb=-1), dict(max_concurrency=0), dict(graceful_timeout=0)):
            with self.subTest(kw=kw), self.assertRaises(SystemExit):
                self._serve(**kw)

    def test_the_cli_parses_the_new_options(self):
        from maki.__main__ import main
        with patch("sys.argv", ["maki", "serve", "--config", "a.yaml", "--max-body-mb", "2", "--max-concurrency", "5"]), \
                patch("maki.__main__._cmd_serve") as serve:
            main()
        args = serve.call_args[0][0]
        self.assertEqual((args.max_body_mb, args.max_concurrency), (2.0, 5))


if __name__ == "__main__":
    unittest.main()
