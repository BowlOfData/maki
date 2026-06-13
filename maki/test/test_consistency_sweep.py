"""
Regression tests for Phase 3.4 — consistency sweep.

Covers:
- MakiAPIError.status_code attribute and connector population
- proxy recall() uses status_code instead of str(e)
- run_workflow homogeneity validation
- execute_task_with_retry forwards use_plugins
- AgentManager has no task_queue attribute
- WorkflowTask.timestamp set on execution start
- print_history does not emit Rich markup through logging
- MakiLLama.stream() accepts images parameter
"""

import inspect
import logging
import time
from typing import Optional
from unittest.mock import MagicMock, patch, call

import pytest

from maki.exceptions import MakiAPIError, MakiNetworkError
from maki.agents.agent import Agent
from maki.agents.agent_manager import AgentManager
from maki.agents.workflow import WorkflowTask, TaskStatus


# ---------------------------------------------------------------------------
# MakiAPIError.status_code
# ---------------------------------------------------------------------------

class TestMakiAPIErrorStatusCode:
    def test_default_status_code_is_zero(self):
        err = MakiAPIError("some error")
        assert err.status_code == 0

    def test_status_code_is_set(self):
        err = MakiAPIError("HTTP client error 404: not found", status_code=404)
        assert err.status_code == 404

    def test_str_is_the_message(self):
        err = MakiAPIError("bad request", status_code=400)
        assert "bad request" in str(err)

    def test_connector_raise_for_response_sets_status_code(self):
        from maki.connector import Connector

        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"

        with pytest.raises(MakiAPIError) as exc_info:
            Connector.raise_for_response(mock_resp)
        assert exc_info.value.status_code == 403

    def test_connector_404_sets_status_code(self):
        from maki.connector import Connector

        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = "Not Found"

        with pytest.raises(MakiAPIError) as exc_info:
            Connector.raise_for_response(mock_resp)
        assert exc_info.value.status_code == 404

    def test_connector_5xx_does_not_raise_api_error(self):
        from maki.connector import Connector

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        with pytest.raises(MakiNetworkError):
            Connector.raise_for_response(mock_resp)


# ---------------------------------------------------------------------------
# AgentProxy.recall() uses status_code, not str(e)
# ---------------------------------------------------------------------------

class TestProxyRecall:
    def _make_proxy(self):
        """Return a proxy instance without triggering a real HTTP connection."""
        try:
            from maki.distributed.proxy import AgentProxy
        except ImportError:
            pytest.skip("distributed extra not installed")

        with patch.object(AgentProxy, "_refresh_info", return_value=None):
            proxy = AgentProxy.__new__(AgentProxy)
            proxy._connected = False
            proxy._circuit_breaker = MagicMock()
            proxy._circuit_breaker.state.name = "CLOSED"
            proxy.name = "test"
            return proxy

    def test_recall_returns_none_on_status_code_404(self):
        proxy = self._make_proxy()
        err = MakiAPIError("HTTP client error 404: not found", status_code=404)

        with patch.object(proxy.__class__, "_get", side_effect=err):
            result = proxy.recall("missing_key")
        assert result is None

    def test_recall_reraises_on_other_status_codes(self):
        proxy = self._make_proxy()
        err = MakiAPIError("HTTP client error 403: forbidden", status_code=403)

        with patch.object(proxy.__class__, "_get", side_effect=err):
            with pytest.raises(MakiAPIError) as exc_info:
                proxy.recall("some_key")
        assert exc_info.value.status_code == 403

    def test_recall_does_not_match_on_string_404_in_message(self):
        """An error whose *message* contains '404' but status_code is 0 must re-raise."""
        proxy = self._make_proxy()
        # Old code: "404" in str(e) would match; new code: checks .status_code
        err = MakiAPIError("Something about 404 in the body", status_code=0)

        with patch.object(proxy.__class__, "_get", side_effect=err):
            with pytest.raises(MakiAPIError):
                proxy.recall("key")


# ---------------------------------------------------------------------------
# run_workflow homogeneity validation
# ---------------------------------------------------------------------------

class TestRunWorkflowHomogeneity:
    def _manager(self):
        backend = MagicMock()
        return AgentManager(backend)

    def test_mixed_list_raises_value_error(self):
        mgr = self._manager()
        wt = WorkflowTask("t1", "agent1", "do something")
        mixed = [wt, {"step": "1", "agent": "a", "task": "t"}]
        with pytest.raises(ValueError, match="homogeneous"):
            mgr.run_workflow(mixed)

    def test_all_dicts_is_accepted(self):
        mgr = self._manager()
        agent = MagicMock()
        agent.execute_task.return_value = "done"
        mgr.agents["a"] = agent
        # dict-based steps use "name" key; falls back to step_N if absent
        workflow = [{"name": "step1", "agent": "a", "task": "t"}]
        with patch.object(mgr, "assign_task", return_value="done") as mock_assign:
            result = mgr.run_workflow(workflow)
        assert "step1" in result

    def test_all_workflow_tasks_is_accepted(self):
        mgr = self._manager()
        agent = MagicMock()
        agent.execute_task_with_retry.return_value = "done"
        mgr.agents["agent1"] = agent

        wt = WorkflowTask("t1", "agent1", "do something")
        result = mgr.run_workflow([wt])
        assert "t1" in result

    def test_empty_list_returns_empty_dict(self):
        mgr = self._manager()
        assert mgr.run_workflow([]) == {}


# ---------------------------------------------------------------------------
# execute_task_with_retry forwards use_plugins
# ---------------------------------------------------------------------------

class TestExecuteTaskWithRetryPlugins:
    def _make_agent(self):
        backend = MagicMock()
        backend.chat = MagicMock(return_value=MagicMock(content="result"))
        backend.supports_native_tools = False
        return Agent("test_agent", backend, role="tester")

    def test_use_plugins_forwarded_to_execute_task(self):
        agent = self._make_agent()
        with patch.object(agent, "execute_task", return_value="ok") as mock_exec:
            agent.execute_task_with_retry("do it", use_plugins=True)
        mock_exec.assert_called_once_with("do it", None, use_plugins=True)

    def test_use_plugins_false_by_default(self):
        agent = self._make_agent()
        with patch.object(agent, "execute_task", return_value="ok") as mock_exec:
            agent.execute_task_with_retry("do it")
        mock_exec.assert_called_once_with("do it", None, use_plugins=False)

    def test_signature_has_use_plugins(self):
        sig = inspect.signature(Agent.execute_task_with_retry)
        assert "use_plugins" in sig.parameters


# ---------------------------------------------------------------------------
# AgentManager has no task_queue attribute
# ---------------------------------------------------------------------------

class TestNoTaskQueue:
    def test_task_queue_is_removed(self):
        backend = MagicMock()
        mgr = AgentManager(backend)
        assert not hasattr(mgr, "task_queue"), (
            "task_queue was removed in 3.4 (§4.7) — do not re-add it"
        )


# ---------------------------------------------------------------------------
# WorkflowTask.timestamp is set on execution start
# ---------------------------------------------------------------------------

class TestWorkflowTaskTimestamp:
    def test_timestamp_set_when_task_executes(self):
        backend = MagicMock()
        mgr = AgentManager(backend)

        agent = MagicMock()
        agent.execute_task_with_retry.return_value = "done"
        mgr.agents["agent1"] = agent

        wt = WorkflowTask("t1", "agent1", "task text")
        assert wt.timestamp is None

        mgr.run_workflow([wt])

        assert wt.timestamp is not None
        assert isinstance(wt.timestamp, float)

    def test_timestamp_is_before_execution_end(self):
        backend = MagicMock()
        mgr = AgentManager(backend)

        start_times = []

        def slow_task(*args, **kwargs):
            start_times.append(time.time())
            return "done"

        agent = MagicMock()
        agent.execute_task_with_retry.side_effect = slow_task
        mgr.agents["agent1"] = agent

        wt = WorkflowTask("t1", "agent1", "task text")
        before = time.time()
        mgr.run_workflow([wt])
        after = time.time()

        assert before <= wt.timestamp <= after


# ---------------------------------------------------------------------------
# print_history does not emit Rich markup through logging
# ---------------------------------------------------------------------------

class TestPrintHistoryNoRichMarkup:
    def test_print_history_uses_print_not_logging(self, capsys):
        from maki.session import ChatSession

        backend = MagicMock()
        backend.chat = MagicMock(return_value=MagicMock(content="hello"))
        session = ChatSession(backend)

        # Manually add messages to the memory
        from maki.objects import Message
        session._memory._messages.extend([
            Message("user", "Hi"),
            Message("assistant", "Hello there"),
        ])

        # Capture log output too
        with patch("maki.session.log") as mock_log:
            session.print_history()
            # log.info should NOT have been called with Rich markup
            for c in mock_log.info.call_args_list:
                text = str(c)
                assert "[bold" not in text, f"Rich markup leaked into logging: {text}"

        # The output should appear on stdout via print()
        captured = capsys.readouterr()
        assert "USER" in captured.out
        assert "ASSISTANT" in captured.out
        assert "[bold" not in captured.out

    def test_print_history_content_appears(self, capsys):
        from maki.session import ChatSession
        from maki.objects import Message

        backend = MagicMock()
        session = ChatSession(backend)
        session._memory._messages.extend([
            Message("user", "my question"),
            Message("assistant", "my answer"),
        ])

        session.print_history()
        out = capsys.readouterr().out
        assert "my question" in out
        assert "my answer" in out


# ---------------------------------------------------------------------------
# MakiLLama.stream() accepts images
# ---------------------------------------------------------------------------

class TestMakiLLamaStreamImages:
    def test_stream_signature_has_images(self):
        from maki.makiLLama import MakiLLama
        sig = inspect.signature(MakiLLama.stream)
        assert "images" in sig.parameters

    def test_stream_passes_images_to_payload(self):
        from maki.makiLLama import MakiLLama

        with patch.object(MakiLLama, "_verify_connection", return_value=None), \
             patch("maki.makiLLama.Connector") as mock_connector_cls:

            mock_conn = MagicMock()
            mock_connector_cls.return_value = mock_conn

            llm = MakiLLama.__new__(MakiLLama)
            llm.base_url = "http://localhost:11434"
            llm.model = "llama3"
            llm.timeout = 30
            llm._rate_limiter = None
            llm._http = mock_conn

            # Make post() return an iterable of lines
            mock_response = MagicMock()
            mock_conn.post.return_value = mock_response

            with patch("maki.makiLLama.Connector.iter_lines", return_value=[
                b'{"message": {"content": "chunk"}, "done": false}',
                b'{"message": {"content": ""}, "done": true}',
            ]):
                with patch.object(llm, "_build_payload", wraps=llm._build_payload if hasattr(llm, "_build_payload") else None) as mock_build:
                    mock_build.return_value = {"model": "llama3", "messages": []}
                    list(llm.stream("hello", images=["base64data"]))

                mock_build.assert_called_once()
                _, kwargs = mock_build.call_args
                assert kwargs.get("images") == ["base64data"]

    def test_stream_images_none_by_default(self):
        from maki.makiLLama import MakiLLama
        sig = inspect.signature(MakiLLama.stream)
        default = sig.parameters["images"].default
        assert default is None
