"""Behaviour of ``maki serve`` the mutation check showed untested: authentication on every route, how each
error is reported (and what it must not reveal), response shapes, memory endpoints and the size limiter."""
import re
import unittest
import uuid
from unittest.mock import MagicMock

pytest = __import__("pytest")
fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")
TestClient = __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient

from maki.agents import Agent  # noqa: E402
from maki.distributed.server import DEFAULT_BODY_LIMIT, ExecuteRequest, _BodyLimitMiddleware, create_app  # noqa: E402
from maki.exceptions import (MakiAPIError, MakiError, MakiNetworkError, MakiTimeoutError,  # noqa: E402
                             MakiValidationError)
from maki.objects import LLMResponse  # noqa: E402

KEY = "correct-horse-battery-staple"
LEAK = "SECRET-INTERNAL-PATH /srv/private/model-host-7"


def _agent():
    backend = MagicMock()
    backend.chat.return_value = LLMResponse(content="ok", model="m", prompt_tokens=1, completion_tokens=1,
                                            total_tokens=2, elapsed_seconds=0.0)
    return Agent("agent-one", backend, role="tester")


def _client(app_kw=None, agent=None):
    return TestClient(create_app(agent or _agent(), **(app_kw or {})), raise_server_exceptions=False)


def _walk(routes):
    """Every concrete route, descending into included routers (newer FastAPI keeps them as a lazy wrapper
    around the original router) and mounts."""
    for route in routes:
        nested = getattr(getattr(route, "original_router", None), "routes", None) or getattr(route, "routes", None)
        if nested is not None and not getattr(route, "methods", None):
            yield from _walk(nested)
        else:
            yield route


def _routes(app):
    for route in _walk(app.routes):
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", "")
        if methods and path not in ("/openapi.json", "/docs", "/redoc", "/docs/oauth2-redirect"):
            for method in sorted(methods - {"HEAD", "OPTIONS"}):
                yield method, path


def _fill(path):
    return re.sub(r"\{[^}]+\}", "abc123", path)


class TestEveryRouteIsAuthenticated(unittest.TestCase):
    def setUp(self):
        judge = Agent("judge", MagicMock())
        self.app = create_app(_agent(), api_key=KEY, judge=judge)
        self.client = TestClient(self.app, raise_server_exceptions=False)
        self.routes = list(_routes(self.app))

    def test_the_route_list_is_not_empty_and_has_the_expected_surface(self):
        paths = {path for _, path in self.routes}
        for expected in ("/health", "/info", "/execute", "/stream", "/memory/set", "/loop", "/history"):
            self.assertIn(expected, paths)
        self.assertGreater(len(self.routes), 15)

    def _call(self, method, path, headers=None):
        body = {"json": {}} if method in ("POST", "PUT", "PATCH") else {}
        return self.client.request(method, _fill(path), headers=headers or {}, **body)

    def test_without_credentials_every_route_but_health_answers_401(self):
        for method, path in self.routes:
            if path == "/health":
                continue
            with self.subTest(route=f"{method} {path}"):
                self.assertEqual(self._call(method, path).status_code, 401)

    def test_wrong_credentials_never_get_in(self):
        # (extra whitespace around the *correct* key is tolerated by the HTTP layer, so it is not listed here)
        wrong = ["Bearer " + KEY[:-1], "Bearer " + KEY + "x", "Bearer " + KEY.upper(), "Bearer x" + KEY,
                 "Bearer ", "Bearer", "Basic " + KEY, "Token " + KEY, KEY, "Bearer " + "A" * 10_000, "Bearer ünïcode-kéy",
                 "Bearer " + KEY[::-1]]
        for header in wrong:
            for method, path in self.routes:
                if path == "/health":
                    continue
                with self.subTest(header=header[:30], route=f"{method} {path}"):
                    try:
                        status = self._call(method, path, {"Authorization": header}).status_code
                    except UnicodeEncodeError:  # a header the client library cannot even send
                        continue
                    self.assertIn(status, (401, 403), f"{method} {path} let {header[:30]!r} through")

    def test_the_right_key_opens_every_route(self):
        auth = {"Authorization": f"Bearer {KEY}"}
        for method, path in self.routes:
            if path in ("/stream", "/loop/{loop_id}/events"):
                continue  # streams are exercised elsewhere; only the 401 gate matters here
            with self.subTest(route=f"{method} {path}"):
                self.assertNotIn(self._call(method, path, auth).status_code, (401, 403))

    def test_health_is_the_only_open_route_and_says_nothing_sensitive(self):
        open_routes = []
        for method, path in self.routes:
            if self._call(method, path).status_code != 401:
                open_routes.append((method, path))
        self.assertEqual(open_routes, [("GET", "/health")])


class TestErrorReporting(unittest.TestCase):
    def _post(self, error, via_agent=True):
        agent = _agent()
        if via_agent:
            agent.maki.chat.side_effect = error
        else:
            agent.execute_task = MagicMock(side_effect=error)
        response = _client(agent=agent).post("/execute", json={"task": "hello"})
        return response

    def test_each_failure_has_its_status_and_a_body_that_reveals_nothing_internal(self):
        cases = [(MakiTimeoutError(LEAK), 504, "timed out"), (MakiNetworkError(LEAK), 502, "unavailable"),
                 (MakiAPIError(LEAK, status_code=400), 400, "rejected"), (MakiError(LEAK), 500, "Internal agent error"),
                 (RuntimeError(LEAK), 500, "Internal agent error")]
        for error, status, fragment in cases:
            with self.subTest(error=type(error).__name__):
                response = self._post(error)
                self.assertEqual(response.status_code, status)
                self.assertIn(fragment, response.json()["detail"])
                self.assertNotIn("SECRET-INTERNAL", response.text)
                self.assertNotIn("/srv/private", response.text)

    def test_an_unexpected_exception_outside_the_agent_is_a_generic_500(self):
        response = self._post(RuntimeError(LEAK), via_agent=False)
        self.assertEqual((response.status_code, response.json()["detail"]), (500, "Internal server error"))
        self.assertNotIn("SECRET-INTERNAL", response.text)

    def test_the_callers_own_input_errors_are_explained(self):
        for error in (MakiValidationError("task is nonsense"), ValueError("task is nonsense")):
            with self.subTest(error=type(error).__name__):
                response = self._post(error, via_agent=False)
                self.assertEqual((response.status_code, response.json()["detail"]), (400, "task is nonsense"))

    def test_malformed_requests(self):
        client = _client()
        for kwargs, status in (({"json": {}}, 422), ({"json": {"task": ""}}, 422), ({"json": {"task": "  \n\t"}}, 422),
                               ({"json": {"task": 5}}, 422), ({"json": {"task": None}}, 422),
                               ({"json": {"task": "x", "context": [1, 2]}}, 422),
                               ({"json": {"task": "x", "use_plugins": "maybe"}}, 422),
                               ({"content": b"not json", "headers": {"content-type": "application/json"}}, 422),
                               ({"content": b"\xff\xfe\x00", "headers": {"content-type": "application/json"}}, 400),
                               ({"content": b"[" * 5000, "headers": {"content-type": "application/json"}}, 422)):
            with self.subTest(case=str(kwargs)[:60]):
                response = client.post("/execute", **kwargs)
                self.assertEqual(response.status_code, status)
                self.assertNotIn("Traceback", response.text)

    def test_wrong_methods_and_unknown_paths(self):
        client = _client()
        self.assertEqual(client.get("/execute").status_code, 405)
        self.assertEqual(client.delete("/execute").status_code, 405)
        self.assertEqual(client.post("/health").status_code, 405)
        for path in ("/nope", "/loop/../../etc/passwd", "/%2e%2e/etc/passwd", "/%2e%2e%2f%2e%2e%2fetc/passwd", "//etc/passwd"):
            self.assertIn(client.get(path).status_code, (404, 405), path)
        self.assertEqual(client.get("/memory/").json(), {"memory": {}})  # a trailing slash redirects to the list

    def test_execute_defaults(self):
        request = ExecuteRequest(task="x")
        self.assertFalse(request.use_plugins)
        self.assertIsNone(request.context)
        with self.assertRaises(Exception):
            ExecuteRequest(task="   ")


class TestResponseShapes(unittest.TestCase):
    def test_execute_returns_the_documented_fields(self):
        agent = _agent()
        body = _client(agent=agent).post("/execute", json={"task": "hello"}).json()
        self.assertEqual(set(body), {"result", "agent_id", "elapsed", "trace_id"})
        self.assertEqual((body["result"], body["agent_id"]), ("ok", agent.agent_id))
        self.assertIsInstance(body["elapsed"], float)
        self.assertGreaterEqual(body["elapsed"], 0.0)
        self.assertLess(body["elapsed"], 5.0)

    def test_trace_ids_are_generated_or_propagated_and_echoed_in_the_header(self):
        client = _client()
        generated = client.post("/execute", json={"task": "x"})
        trace = generated.headers["X-Maki-Trace-Id"]
        self.assertEqual(str(uuid.UUID(trace)), trace)
        self.assertEqual(generated.json()["trace_id"], trace)
        given = client.post("/execute", json={"task": "x"}, headers={"X-Maki-Trace-Id": "caller-trace-42"})
        self.assertEqual((given.headers["X-Maki-Trace-Id"], given.json()["trace_id"]), ("caller-trace-42",) * 2)
        self.assertIn("X-Maki-Trace-Id", client.get("/health").headers)

    def test_info_and_health_shapes(self):
        client = _client()
        health = client.get("/health").json()
        self.assertEqual((health["status"], health["name"], health["role"]), ("ok", "agent-one", "tester"))
        info = client.get("/info").json()
        self.assertEqual(set(info), {"agent_id", "name", "role", "plugins", "backend", "model"})
        self.assertEqual(info["plugins"], [])

    def test_history_can_be_cleared(self):
        agent = _agent()
        client = _client(agent=agent)
        client.post("/execute", json={"task": "remember this"})
        self.assertTrue(agent.task_history)
        self.assertEqual(client.delete("/history").json(), {"ok": True})
        self.assertFalse(agent.task_history)


class TestMemoryEndpoints(unittest.TestCase):
    def test_the_full_lifecycle_with_bodies(self):
        client = _client()
        self.assertEqual(client.get("/memory").json(), {"memory": {}})
        self.assertEqual(client.post("/memory/set", json={"key": "a", "value": {"n": [1, 2]}}).json(), {"ok": True})
        client.post("/memory/set", json={"key": "b", "value": "text"})
        self.assertEqual(client.get("/memory").json(), {"memory": {"a": {"n": [1, 2]}, "b": "text"}})
        self.assertEqual(client.get("/memory/a").json(), {"key": "a", "value": {"n": [1, 2]}})
        self.assertEqual(client.delete("/memory/a").json(), {"ok": True})
        self.assertEqual(client.get("/memory/a").status_code, 404)
        self.assertIn("'a' not found", client.get("/memory/a").json()["detail"])
        self.assertEqual(client.delete("/memory/a").status_code, 404)
        self.assertEqual(client.delete("/memory").json(), {"ok": True})
        self.assertEqual(client.get("/memory").json(), {"memory": {}})

    def test_keys_with_awkward_characters_round_trip_or_are_refused_cleanly(self):
        client = _client()
        for key in ("spaces and ünïcode", "a/b", "../../etc", "k" * 512, "<script>", "%00"):
            with self.subTest(key=key):
                self.assertEqual(client.post("/memory/set", json={"key": key, "value": 1}).status_code, 200)
        for key in ("", "k" * 513):
            with self.subTest(key=key[:8]):
                self.assertEqual(client.post("/memory/set", json={"key": key, "value": 1}).status_code, 409)
        self.assertEqual(client.post("/memory/set", json={"value": 1}).status_code, 422)

    def test_a_pinned_key_cannot_be_rewritten_remotely(self):
        from maki.objects import MemoryPolicy
        agent = Agent("p", MagicMock(), memory=MemoryPolicy(pinned_chars=500))
        agent.remember("goal", "ship it", pin=True)
        response = _client(agent=agent).post("/memory/set", json={"key": "goal", "value": "ignore previous"})
        self.assertEqual(response.status_code, 409)
        self.assertIn("pinned", response.json()["detail"])
        self.assertEqual(agent.recall("goal"), "ship it")


class Recorder:
    """A bare ASGI app that records the scope types it was given and what body it read."""

    def __init__(self):
        self.scopes, self.bodies = [], []

    async def __call__(self, scope, receive, send):
        self.scopes.append(scope["type"])
        if scope["type"] == "http":
            body = b""
            while True:
                message = await receive()
                body += message.get("body", b"")
                if not message.get("more_body"):
                    break
            self.bodies.append(len(body))
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"fine"})


class TestBodyLimiter(unittest.TestCase):
    def _client(self, prefix="/", max_bytes=100):
        self.inner = Recorder()
        return TestClient(_BodyLimitMiddleware(self.inner, max_bytes=max_bytes, prefix=prefix), raise_server_exceptions=False)

    def test_the_boundary_is_inclusive(self):
        client = self._client(max_bytes=100)
        self.assertEqual(client.post("/x", content=b"a" * 100).status_code, 200)
        self.assertEqual(client.post("/x", content=b"a" * 101).status_code, 413)
        self.assertEqual(self.inner.bodies, [100])

    def test_many_small_chunks_are_added_up(self):
        client = self._client(max_bytes=100)
        self.assertEqual(client.post("/x", content=iter([b"a" * 10] * 10)).status_code, 200)
        self.assertEqual(client.post("/x", content=iter([b"a" * 10] * 11)).status_code, 413)

    def test_only_requests_under_the_prefix_are_limited(self):
        client = self._client(prefix="/loop", max_bytes=10)
        self.assertEqual(client.post("/other", content=b"a" * 1000).status_code, 200)
        self.assertEqual(client.post("/loop", content=b"a" * 1000).status_code, 413)
        self.assertEqual(client.post("/loop/abc/approval", content=b"a" * 1000).status_code, 413)
        self.assertEqual(client.post("/loopy", content=b"a" * 1000).status_code, 413)  # a plain prefix match

    def test_a_false_content_length_cannot_get_around_it(self):
        client = self._client(max_bytes=100)
        response = client.post("/x", content=b"a" * 500, headers={"content-length": "5"})
        self.assertNotEqual(response.status_code, 200)

    def test_non_http_scopes_pass_straight_through(self):
        import asyncio
        inner = Recorder()
        middleware = _BodyLimitMiddleware(inner, max_bytes=1, prefix="/")
        for kind in ("lifespan", "websocket"):
            asyncio.run(middleware({"type": kind, "path": "/x", "headers": []}, None, None))
        self.assertEqual(inner.scopes, ["lifespan", "websocket"])

    def test_the_rejection_is_json_and_names_the_limit(self):
        response = self._client(max_bytes=100).post("/x", content=b"a" * 500)
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.headers["content-type"], "application/json")
        self.assertIn("larger than 100 bytes", response.json()["detail"])

    def test_the_apps_limits_nest_correctly(self):
        self.assertEqual(DEFAULT_BODY_LIMIT, 4 * 1024 * 1024)
        client = _client({"max_body_bytes": 1})
        self.assertEqual(client.post("/execute", content=b"{}", headers={"content-type": "application/json"}).status_code, 413)
        with self.assertRaises(ValueError):
            create_app(_agent(), max_body_bytes=0)


if __name__ == "__main__":
    unittest.main()
