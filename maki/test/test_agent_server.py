"""
Phase 2 tests: AgentServer (FastAPI) endpoints.

Uses FastAPI's TestClient (no real network, no real LLM backend).
All tests are skipped automatically if fastapi is not installed.
"""
import json
import unittest
from unittest.mock import MagicMock

pytest = __import__("pytest")
fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")
TestClient = __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient

from maki.agents import Agent
from maki.objects import LLMResponse
from maki.distributed.server import create_app


def _llm_response(content: str = "mock result") -> LLMResponse:
    return LLMResponse(
        content=content, model="test", prompt_tokens=1,
        completion_tokens=1, total_tokens=2, elapsed_seconds=0.01,
    )


def _mock_backend(content: str = "mock result"):
    m = MagicMock()
    m.chat.return_value = _llm_response(content)
    m.stream.side_effect = lambda *a, **kw: iter(["chunk1", " chunk2"])
    return m


def _make_client(api_key=None, content="mock result"):
    backend = _mock_backend(content)
    agent = Agent("test-agent", backend, role="tester", instructions="be helpful")
    app = create_app(agent, api_key=api_key)
    return TestClient(app, raise_server_exceptions=True), agent


class TestHealth(unittest.TestCase):

    def test_health_ok(self):
        client, agent = _make_client()
        r = client.get("/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["name"], "test-agent")
        self.assertEqual(body["role"], "tester")
        self.assertEqual(body["agent_id"], agent.agent_id)

    def test_info_ok(self):
        client, agent = _make_client()
        r = client.get("/info")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["name"], "test-agent")
        self.assertIn("plugins", body)
        self.assertIn("backend", body)
        self.assertIn("model", body)
        self.assertEqual(body["agent_id"], agent.agent_id)


class TestExecute(unittest.TestCase):

    def test_execute_returns_result(self):
        client, agent = _make_client(content="the answer is 42")
        r = client.post("/execute", json={"task": "what is the answer?"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["result"], "the answer is 42")
        self.assertEqual(body["agent_id"], agent.agent_id)
        self.assertIn("elapsed", body)
        self.assertIsInstance(body["elapsed"], float)

    def test_execute_with_context(self):
        client, _ = _make_client()
        r = client.post("/execute", json={"task": "summarise", "context": {"text": "hello world"}})
        self.assertEqual(r.status_code, 200)

    def test_execute_empty_task_is_422(self):
        client, _ = _make_client()
        r = client.post("/execute", json={"task": ""})
        self.assertEqual(r.status_code, 422)

    def test_execute_missing_task_is_422(self):
        client, _ = _make_client()
        r = client.post("/execute", json={})
        self.assertEqual(r.status_code, 422)

    @staticmethod
    def _client_with_error(exc):
        backend = _mock_backend()
        backend.chat.side_effect = exc
        agent = Agent("err-agent", backend)
        app = create_app(agent)
        return TestClient(app, raise_server_exceptions=False)

    def test_execute_network_error_is_502(self):
        from maki.exceptions import MakiNetworkError
        client = self._client_with_error(MakiNetworkError("connection refused to http://10.0.0.5:11434"))
        r = client.post("/execute", json={"task": "do something"})
        self.assertEqual(r.status_code, 502)
        # §2.4: internal details (URLs/paths) must not reach remote callers.
        self.assertNotIn("10.0.0.5", r.json()["detail"])

    def test_execute_timeout_is_504(self):
        from maki.exceptions import MakiTimeoutError
        client = self._client_with_error(MakiTimeoutError("timed out after 180s"))
        r = client.post("/execute", json={"task": "do something"})
        self.assertEqual(r.status_code, 504)

    def test_execute_api_error_is_400_with_generic_body(self):
        from maki.exceptions import MakiAPIError
        client = self._client_with_error(MakiAPIError("upstream said no: http://internal:11434/api"))
        r = client.post("/execute", json={"task": "do something"})
        self.assertEqual(r.status_code, 400)
        self.assertNotIn("internal", r.json()["detail"])

    def test_execute_validation_error_is_400_with_message(self):
        from maki.exceptions import MakiValidationError
        client = self._client_with_error(MakiValidationError("task too long"))
        r = client.post("/execute", json={"task": "do something"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("task too long", r.json()["detail"])

    def test_execute_value_error_is_400(self):
        # §2.4 regression: a raw ValueError from the agent used to escape as
        # an unhandled 500.
        client = self._client_with_error(ValueError("bad task input"))
        r = client.post("/execute", json={"task": "do something"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("bad task input", r.json()["detail"])

    def test_execute_unexpected_error_is_500_generic(self):
        client = self._client_with_error(RuntimeError("secret /etc/internal/path"))
        r = client.post("/execute", json={"task": "do something"})
        self.assertEqual(r.status_code, 500)
        self.assertNotIn("/etc/internal/path", r.json()["detail"])

    def test_execute_updates_task_history(self):
        client, agent = _make_client()
        client.post("/execute", json={"task": "first task"})
        self.assertEqual(len(agent.task_history), 1)
        self.assertEqual(agent.task_history[0]["task"], "first task")


class TestStream(unittest.TestCase):

    def test_stream_returns_sse(self):
        client, _ = _make_client()
        with client.stream("GET", "/stream", params={"task": "tell me a story"}) as r:
            self.assertEqual(r.status_code, 200)
            self.assertIn("text/event-stream", r.headers["content-type"])
            body = r.read().decode()
        self.assertIn("chunk1", body)
        self.assertIn("chunk2", body)
        self.assertIn("[DONE]", body)

    def test_stream_sse_format(self):
        client, _ = _make_client()
        with client.stream("GET", "/stream", params={"task": "go"}) as r:
            lines = r.read().decode().strip().splitlines()
        data_lines = [l for l in lines if l.startswith("data: ")]
        self.assertTrue(len(data_lines) >= 2)
        first_event = json.loads(data_lines[0][len("data: "):])
        self.assertIn("chunk", first_event)


class TestMemory(unittest.TestCase):

    def test_set_and_get(self):
        client, _ = _make_client()
        r = client.post("/memory/set", json={"key": "color", "value": "blue"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])

        r = client.get("/memory/color")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["value"], "blue")

    def test_get_missing_key_is_404(self):
        client, _ = _make_client()
        r = client.get("/memory/nonexistent")
        self.assertEqual(r.status_code, 404)

    def test_delete_key(self):
        client, _ = _make_client()
        client.post("/memory/set", json={"key": "x", "value": 1})
        r = client.delete("/memory/x")
        self.assertEqual(r.status_code, 200)
        r = client.get("/memory/x")
        self.assertEqual(r.status_code, 404)

    def test_delete_missing_key_is_404(self):
        client, _ = _make_client()
        r = client.delete("/memory/ghost")
        self.assertEqual(r.status_code, 404)

    def test_set_complex_value(self):
        client, _ = _make_client()
        val = {"nested": [1, 2, {"deep": True}]}
        client.post("/memory/set", json={"key": "obj", "value": val})
        r = client.get("/memory/obj")
        self.assertEqual(r.json()["value"], val)


class TestHistory(unittest.TestCase):

    def test_history_empty_initially(self):
        client, _ = _make_client()
        r = client.get("/history")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["history"], [])

    def test_history_after_execute(self):
        client, _ = _make_client()
        client.post("/execute", json={"task": "task one"})
        client.post("/execute", json={"task": "task two"})
        r = client.get("/history")
        history = r.json()["history"]
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["task"], "task one")
        self.assertEqual(history[1]["task"], "task two")

    def test_history_clear(self):
        client, _ = _make_client()
        client.post("/execute", json={"task": "something"})
        r = client.delete("/history")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])
        r = client.get("/history")
        self.assertEqual(r.json()["history"], [])


class TestAuth(unittest.TestCase):

    def test_no_key_open_access(self):
        client, _ = _make_client(api_key=None)
        r = client.get("/health")
        self.assertEqual(r.status_code, 200)

    def test_correct_key_accepted(self):
        client, _ = _make_client(api_key="secret123")
        r = client.get("/info", headers={"Authorization": "Bearer secret123"})
        self.assertEqual(r.status_code, 200)

    def test_wrong_key_rejected(self):
        client, _ = _make_client(api_key="secret123")
        r = client.get("/info", headers={"Authorization": "Bearer wrongkey"})
        self.assertEqual(r.status_code, 401)

    def test_missing_key_rejected(self):
        client, _ = _make_client(api_key="secret123")
        r = client.get("/info")
        self.assertEqual(r.status_code, 401)

    def test_health_is_unauthenticated(self):
        # §5: /health must stay open for load-balancer/k8s probes even
        # when an API key is configured.
        client, _ = _make_client(api_key="secret123")
        r = client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_auth_applies_to_execute(self):
        client, _ = _make_client(api_key="tok")
        r = client.post("/execute", json={"task": "go"})
        self.assertEqual(r.status_code, 401)

    def test_auth_applies_to_memory(self):
        client, _ = _make_client(api_key="tok")
        r = client.get("/memory/x")
        self.assertEqual(r.status_code, 401)

    def test_auth_applies_to_history(self):
        client, _ = _make_client(api_key="tok")
        r = client.get("/history")
        self.assertEqual(r.status_code, 401)

    def test_multiple_apps_independent_keys(self):
        backend = _mock_backend()
        agent_a = Agent("a", backend)
        agent_b = Agent("b", _mock_backend())
        app_a = create_app(agent_a, api_key="key-a")
        app_b = create_app(agent_b, api_key="key-b")
        ca = TestClient(app_a, raise_server_exceptions=False)
        cb = TestClient(app_b, raise_server_exceptions=False)
        self.assertEqual(ca.get("/info", headers={"Authorization": "Bearer key-a"}).status_code, 200)
        self.assertEqual(ca.get("/info", headers={"Authorization": "Bearer key-b"}).status_code, 401)
        self.assertEqual(cb.get("/info", headers={"Authorization": "Bearer key-b"}).status_code, 200)
        self.assertEqual(cb.get("/info", headers={"Authorization": "Bearer key-a"}).status_code, 401)


if __name__ == "__main__":
    unittest.main()
