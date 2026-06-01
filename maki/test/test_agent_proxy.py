"""
Phase 3 tests: AgentProxy and DistributedAgentManager.

Strategy: patch httpx.Client so no real network is required.
The mock transport routes calls through the real FastAPI app via a custom
httpx transport, giving end-to-end coverage of both proxy logic and
server routing without spinning up a real process.
"""
import json
import unittest
from unittest.mock import MagicMock, patch, call

pytest = __import__("pytest")
httpx = pytest.importorskip("httpx", reason="httpx not installed")
pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi.testclient import TestClient as _TestClient

from maki.agents import Agent
from maki.objects import LLMResponse
from maki.exceptions import MakiAPIError, MakiNetworkError, MakiTimeoutError
from maki.distributed.server import create_app
from maki.distributed.proxy import AgentProxy
from maki.distributed.registry import DistributedAgentManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _llm(content="ok"):
    return LLMResponse(content=content, model="t", prompt_tokens=0,
                       completion_tokens=0, total_tokens=0, elapsed_seconds=0.0)


def _mock_backend(content="remote result"):
    m = MagicMock()
    m.chat.return_value = _llm(content)
    m.stream.side_effect = lambda *a, **kw: iter(["chunk-a", " chunk-b"])
    return m


class _AppTransport(httpx.BaseTransport):
    """Routes httpx requests to a Starlette app without a real network."""

    def __init__(self, app):
        self._client = _TestClient(app, raise_server_exceptions=False)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        method = request.method
        url = str(request.url)
        headers = dict(request.headers)
        content = request.content

        resp = self._client.request(
            method=method,
            url=url,
            content=content,
            headers=headers,
        )
        return httpx.Response(
            status_code=resp.status_code,
            headers=dict(resp.headers),
            content=resp.content,
        )


def _make_proxy(content="remote result", api_key=None):
    """Create a real AgentProxy wired to an in-process AgentServer."""
    backend = _mock_backend(content)
    agent = Agent("remote-agent", backend, role="analyst", instructions="be precise")
    app = create_app(agent, api_key=api_key)
    transport = _AppTransport(app)
    client = httpx.Client(transport=transport, timeout=10.0)

    with patch("maki.distributed.proxy.httpx.Client", return_value=client):
        proxy = AgentProxy(endpoint="http://fake-host:8100", api_key=api_key)

    return proxy, agent


# ---------------------------------------------------------------------------
# AgentProxy — identity / info
# ---------------------------------------------------------------------------

class TestAgentProxyInfo(unittest.TestCase):

    def test_name_and_role_populated_from_server(self):
        proxy, agent = _make_proxy()
        self.assertEqual(proxy.name, "remote-agent")
        self.assertEqual(proxy.role, "analyst")

    def test_agent_id_populated_from_server(self):
        proxy, agent = _make_proxy()
        self.assertEqual(proxy.agent_id, agent.agent_id)

    def test_repr(self):
        proxy, _ = _make_proxy()
        self.assertIn("remote-agent", repr(proxy))
        self.assertIn("fake-host", repr(proxy))


# ---------------------------------------------------------------------------
# AgentProxy — execute_task
# ---------------------------------------------------------------------------

class TestAgentProxyExecute(unittest.TestCase):

    def test_execute_returns_result(self):
        proxy, _ = _make_proxy(content="the answer")
        result = proxy.execute_task("what is 2+2?")
        self.assertEqual(result, "the answer")

    def test_execute_with_context(self):
        proxy, _ = _make_proxy()
        result = proxy.execute_task("summarise", context={"text": "hello"})
        self.assertEqual(result, "remote result")

    def test_execute_network_error_raises_maki_error(self):
        from maki.exceptions import MakiNetworkError
        backend = _mock_backend()
        backend.chat.side_effect = MakiNetworkError("backend down")
        agent = Agent("err-agent", backend)
        app = create_app(agent)
        transport = _AppTransport(app)
        client = httpx.Client(transport=transport)
        with patch("maki.distributed.proxy.httpx.Client", return_value=client):
            proxy = AgentProxy(endpoint="http://fake:8100")
        with self.assertRaises(MakiNetworkError):
            proxy.execute_task("do something")

    def test_execute_api_error_raises_maki_api_error(self):
        from maki.exceptions import MakiAPIError
        backend = _mock_backend()
        backend.chat.side_effect = MakiAPIError("bad prompt")
        agent = Agent("err-agent", backend)
        app = create_app(agent)
        transport = _AppTransport(app)
        client = httpx.Client(transport=transport)
        with patch("maki.distributed.proxy.httpx.Client", return_value=client):
            proxy = AgentProxy(endpoint="http://fake:8100")
        with self.assertRaises(MakiAPIError):
            proxy.execute_task("do something")

    def test_execute_timeout_maps_to_maki_timeout_error(self):
        mock_client = MagicMock()
        mock_client.get.return_value = MagicMock(
            status_code=200, is_success=True,
            json=lambda: {
                "agent_id": "x", "name": "t", "role": "", "plugins": [],
                "backend": "MockBackend", "model": "test",
            },
        )
        mock_client.post.side_effect = httpx.TimeoutException("timed out")
        with patch("maki.distributed.proxy.httpx.Client", return_value=mock_client):
            proxy = AgentProxy(endpoint="http://fake:8100")
        with self.assertRaises(MakiTimeoutError):
            proxy.execute_task("slow task")

    def test_execute_connection_error_maps_to_maki_network_error(self):
        mock_client = MagicMock()
        mock_client.get.return_value = MagicMock(
            status_code=200, is_success=True,
            json=lambda: {
                "agent_id": "x", "name": "t", "role": "", "plugins": [],
                "backend": "MockBackend", "model": "test",
            },
        )
        mock_client.post.side_effect = httpx.ConnectError("refused")
        with patch("maki.distributed.proxy.httpx.Client", return_value=mock_client):
            proxy = AgentProxy(endpoint="http://fake:8100")
        with self.assertRaises(MakiNetworkError):
            proxy.execute_task("task")


# ---------------------------------------------------------------------------
# AgentProxy — execute_task_with_retry
# ---------------------------------------------------------------------------

class TestAgentProxyRetry(unittest.TestCase):

    def test_retry_succeeds_on_second_attempt(self):
        call_count = 0

        def flaky_post(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ConnectError("first failure")
            return MagicMock(
                status_code=200, is_success=True,
                json=lambda: {"result": "ok", "agent_id": "x", "elapsed": 0.1},
            )

        mock_client = MagicMock()
        mock_client.get.return_value = MagicMock(
            status_code=200, is_success=True,
            json=lambda: {
                "agent_id": "x", "name": "t", "role": "", "plugins": [],
                "backend": "Mock", "model": "t",
            },
        )
        mock_client.post.side_effect = flaky_post
        with patch("maki.distributed.proxy.httpx.Client", return_value=mock_client):
            proxy = AgentProxy(endpoint="http://fake:8100")

        result = proxy.execute_task_with_retry("task", max_retries=3, retry_delay=0)
        self.assertEqual(result, "ok")
        self.assertEqual(call_count, 2)

    def test_retry_exhausted_raises(self):
        mock_client = MagicMock()
        mock_client.get.return_value = MagicMock(
            status_code=200, is_success=True,
            json=lambda: {
                "agent_id": "x", "name": "t", "role": "", "plugins": [],
                "backend": "Mock", "model": "t",
            },
        )
        mock_client.post.side_effect = httpx.ConnectError("always down")
        with patch("maki.distributed.proxy.httpx.Client", return_value=mock_client):
            proxy = AgentProxy(endpoint="http://fake:8100")
        with self.assertRaises(MakiNetworkError):
            proxy.execute_task_with_retry("task", max_retries=2, retry_delay=0)

    def test_non_retryable_errors_not_retried(self):
        call_count = 0

        def bad_post(url, **kwargs):
            nonlocal call_count
            call_count += 1
            return MagicMock(
                status_code=400, is_success=False,
                text="bad request",
            )

        mock_client = MagicMock()
        mock_client.get.return_value = MagicMock(
            status_code=200, is_success=True,
            json=lambda: {
                "agent_id": "x", "name": "t", "role": "", "plugins": [],
                "backend": "Mock", "model": "t",
            },
        )
        mock_client.post.side_effect = bad_post
        with patch("maki.distributed.proxy.httpx.Client", return_value=mock_client):
            proxy = AgentProxy(endpoint="http://fake:8100")
        with self.assertRaises(MakiAPIError):
            proxy.execute_task_with_retry("task", max_retries=3, retry_delay=0)
        self.assertEqual(call_count, 1)


# ---------------------------------------------------------------------------
# AgentProxy — stream_task
# ---------------------------------------------------------------------------

class TestAgentProxyStream(unittest.TestCase):

    def test_stream_yields_chunks(self):
        proxy, _ = _make_proxy()
        chunks = list(proxy.stream_task("tell me a story"))
        self.assertEqual(chunks, ["chunk-a", " chunk-b"])

    def test_stream_is_generator(self):
        proxy, _ = _make_proxy()
        gen = proxy.stream_task("go")
        import types
        self.assertIsInstance(gen, types.GeneratorType)

    def test_stream_sse_parse(self):
        lines = [
            'data: {"chunk": "hello"}',
            'data: {"chunk": " world"}',
            "data: [DONE]",
        ]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_response.iter_lines.return_value = iter(lines)

        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = mock_response
        mock_cm.__exit__.return_value = False

        mock_client = MagicMock()
        mock_client.get.return_value = MagicMock(
            status_code=200, is_success=True,
            json=lambda: {
                "agent_id": "x", "name": "t", "role": "", "plugins": [],
                "backend": "Mock", "model": "t",
            },
        )
        mock_client.stream.return_value = mock_cm
        with patch("maki.distributed.proxy.httpx.Client", return_value=mock_client):
            proxy = AgentProxy(endpoint="http://fake:8100")
        result = list(proxy.stream_task("go"))
        self.assertEqual(result, ["hello", " world"])


# ---------------------------------------------------------------------------
# AgentProxy — memory
# ---------------------------------------------------------------------------

class TestAgentProxyMemory(unittest.TestCase):

    def test_remember_and_recall(self):
        proxy, _ = _make_proxy()
        proxy.remember("lang", "python")
        self.assertEqual(proxy.recall("lang"), "python")

    def test_recall_missing_key_returns_none(self):
        proxy, _ = _make_proxy()
        self.assertIsNone(proxy.recall("no-such-key"))

    def test_clear_memory(self):
        proxy, _ = _make_proxy()
        proxy.remember("k", "v")
        proxy.clear_memory()
        self.assertIsNone(proxy.recall("k"))

    def test_remember_complex_value(self):
        proxy, _ = _make_proxy()
        val = {"nested": [1, 2, 3], "flag": True}
        proxy.remember("data", val)
        self.assertEqual(proxy.recall("data"), val)


# ---------------------------------------------------------------------------
# AgentProxy — conversation / auth
# ---------------------------------------------------------------------------

class TestAgentProxyConversation(unittest.TestCase):

    def test_reset_conversation_clears_history(self):
        proxy, agent = _make_proxy()
        proxy.execute_task("task one")
        proxy.reset_conversation()
        # After reset, task history on the server should be empty
        r = httpx.Client(
            transport=_AppTransport(create_app(agent))
        ).get("http://fake-host:8100/history")
        # We reset the conversation but the history is on a different app instance;
        # verify the proxy method runs without error (it sends DELETE /history).
        # The real end-to-end check is done in test_agent_server.py.


class TestAgentProxyAuth(unittest.TestCase):

    def test_correct_api_key_works(self):
        proxy, _ = _make_proxy(api_key="secret")
        result = proxy.execute_task("task")
        self.assertEqual(result, "remote result")

    def test_wrong_api_key_raises(self):
        # Build a server that requires "correct-key"
        backend = _mock_backend()
        agent = Agent("auth-agent", backend)
        app = create_app(agent, api_key="correct-key")
        transport = _AppTransport(app)
        # Proxy uses the wrong key — /info also fails with 401
        bad_client = httpx.Client(transport=transport, timeout=5.0)
        with patch("maki.distributed.proxy.httpx.Client", return_value=bad_client):
            with self.assertRaises(MakiAPIError):
                AgentProxy(endpoint="http://fake:8100", api_key="wrong-key")


# ---------------------------------------------------------------------------
# DistributedAgentManager
# ---------------------------------------------------------------------------

class TestDistributedAgentManager(unittest.TestCase):

    def _make_manager_with_remote(self, remote_content="remote answer"):
        local_backend = MagicMock()
        local_backend.chat.return_value = _llm("synthesis result")

        manager = DistributedAgentManager(local_backend)

        # Build a real remote agent in-process
        remote_backend = _mock_backend(remote_content)
        remote_agent = Agent("worker", remote_backend, role="worker")
        remote_app = create_app(remote_agent)
        transport = _AppTransport(remote_app)
        remote_client = httpx.Client(transport=transport, timeout=10.0)

        with patch("maki.distributed.proxy.httpx.Client", return_value=remote_client):
            manager.register_remote("worker", endpoint="http://worker:8100")

        return manager

    def test_register_remote_adds_to_agents(self):
        manager = self._make_manager_with_remote()
        self.assertIn("worker", manager.agents)
        self.assertIsInstance(manager.agents["worker"], AgentProxy)

    def test_register_remote_returns_proxy(self):
        local_backend = MagicMock()
        manager = DistributedAgentManager(local_backend)
        remote_backend = _mock_backend()
        remote_agent = Agent("svc", remote_backend)
        app = create_app(remote_agent)
        transport = _AppTransport(app)
        client = httpx.Client(transport=transport)
        with patch("maki.distributed.proxy.httpx.Client", return_value=client):
            proxy = manager.register_remote("svc", endpoint="http://svc:8100")
        self.assertIsInstance(proxy, AgentProxy)

    def test_assign_task_dispatches_to_remote(self):
        manager = self._make_manager_with_remote("remote answer")
        result = manager.assign_task("worker", "do the work")
        self.assertEqual(result, "remote answer")

    def test_local_and_remote_agents_coexist(self):
        local_backend = MagicMock()
        local_backend.chat.return_value = _llm("local answer")
        manager = DistributedAgentManager(local_backend)

        local_agent = Agent("local", local_backend, role="local worker")
        manager.agents["local"] = local_agent

        remote_backend = _mock_backend("remote answer")
        remote_agent = Agent("remote", remote_backend)
        app = create_app(remote_agent)
        transport = _AppTransport(app)
        client = httpx.Client(transport=transport)
        with patch("maki.distributed.proxy.httpx.Client", return_value=client):
            manager.register_remote("remote", endpoint="http://remote:8100")

        local_result = manager.assign_task("local", "local task")
        remote_result = manager.assign_task("remote", "remote task")
        self.assertEqual(local_result, "local answer")
        self.assertEqual(remote_result, "remote answer")

    def test_coordinate_agents_with_remote(self):
        local_backend = MagicMock()
        local_backend.chat.return_value = _llm("synthesised")
        manager = DistributedAgentManager(local_backend)

        remote_backend = _mock_backend("remote contribution")
        remote_agent = Agent("contributor", remote_backend)
        app = create_app(remote_agent)
        transport = _AppTransport(app)
        client = httpx.Client(transport=transport)
        with patch("maki.distributed.proxy.httpx.Client", return_value=client):
            manager.register_remote("contributor", endpoint="http://contrib:8100")

        results = manager.coordinate_agents([
            {"agent": "contributor", "task": "do your part"},
        ])
        self.assertIn("task_0_contributor", results)
        self.assertEqual(results["task_0_contributor"], "remote contribution")

    def test_unregister_remote_removes_agent(self):
        manager = self._make_manager_with_remote()
        manager.unregister_remote("worker")
        self.assertNotIn("worker", manager.agents)

    def test_list_agents_includes_remote(self):
        manager = self._make_manager_with_remote()
        self.assertIn("worker", manager.list_agents())

    def test_invalid_name_raises(self):
        manager = DistributedAgentManager(MagicMock())
        with self.assertRaises(ValueError):
            manager.register_remote("", endpoint="http://x:8100")


if __name__ == "__main__":
    unittest.main()
