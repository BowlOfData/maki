"""
Phase 1 serialization tests: Agent, WorkflowTask, WorkflowState, and data-class
round-trips must survive to_dict() → JSON → from_dict() without data loss.
"""
import json
import time
import unittest
from collections import deque
from unittest.mock import MagicMock

from maki.agents import Agent
from maki.agents.workflow import WorkflowTask, WorkflowState, TaskStatus
from maki.objects import Message, GenerationConfig, LLMResponse, BackendType


def _mock_backend():
    m = MagicMock()
    m.chat.return_value = LLMResponse(
        content="ok", model="test", prompt_tokens=1,
        completion_tokens=1, total_tokens=2, elapsed_seconds=0.1,
    )
    return m


def _roundtrip(obj):
    """Serialize to JSON string and back to a plain dict."""
    return json.loads(json.dumps(obj.to_dict()))


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------

class TestMessageSerialization(unittest.TestCase):

    def test_roundtrip_plain(self):
        msg = Message(role="user", content="hello")
        restored = Message.from_dict(_roundtrip(msg))
        self.assertEqual(restored.role, msg.role)
        self.assertEqual(restored.content, msg.content)
        self.assertIsNone(restored.images)

    def test_roundtrip_with_images(self):
        msg = Message(role="user", content="look", images=["base64data"])
        restored = Message.from_dict(_roundtrip(msg))
        self.assertEqual(restored.images, ["base64data"])


# ---------------------------------------------------------------------------
# GenerationConfig
# ---------------------------------------------------------------------------

class TestGenerationConfigSerialization(unittest.TestCase):

    def test_roundtrip_defaults(self):
        cfg = GenerationConfig()
        restored = GenerationConfig.from_dict(_roundtrip(cfg))
        self.assertEqual(restored.temperature, cfg.temperature)
        self.assertEqual(restored.max_tokens, cfg.max_tokens)
        self.assertEqual(restored.stop, cfg.stop)
        self.assertIsNone(restored.num_ctx)

    def test_roundtrip_custom(self):
        cfg = GenerationConfig(temperature=0.2, max_tokens=512, seed=42,
                               stop=["</s>"], num_ctx=4096)
        restored = GenerationConfig.from_dict(_roundtrip(cfg))
        self.assertEqual(restored.temperature, 0.2)
        self.assertEqual(restored.max_tokens, 512)
        self.assertEqual(restored.seed, 42)
        self.assertEqual(restored.stop, ["</s>"])
        self.assertEqual(restored.num_ctx, 4096)


# ---------------------------------------------------------------------------
# LLMResponse
# ---------------------------------------------------------------------------

class TestLLMResponseSerialization(unittest.TestCase):

    def test_roundtrip(self):
        resp = LLMResponse(
            content="test output", model="llama3", prompt_tokens=10,
            completion_tokens=20, total_tokens=30, elapsed_seconds=1.5,
            done=True, backend=BackendType.OLLAMA,
        )
        restored = LLMResponse.from_dict(_roundtrip(resp))
        self.assertEqual(restored.content, resp.content)
        self.assertEqual(restored.model, resp.model)
        self.assertEqual(restored.prompt_tokens, resp.prompt_tokens)
        self.assertEqual(restored.completion_tokens, resp.completion_tokens)
        self.assertEqual(restored.total_tokens, resp.total_tokens)
        self.assertAlmostEqual(restored.elapsed_seconds, resp.elapsed_seconds)
        self.assertEqual(restored.done, resp.done)
        self.assertEqual(restored.backend, BackendType.OLLAMA)

    def test_all_backends(self):
        for bt in BackendType:
            resp = LLMResponse(content="x", model="m", prompt_tokens=0,
                               completion_tokens=0, total_tokens=0,
                               elapsed_seconds=0.0, backend=bt)
            restored = LLMResponse.from_dict(_roundtrip(resp))
            self.assertEqual(restored.backend, bt)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TestAgentSerialization(unittest.TestCase):

    def test_agent_id_is_uuid(self):
        agent = Agent("tester", _mock_backend())
        import re
        self.assertRegex(agent.agent_id,
                         r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$')

    def test_two_agents_have_distinct_ids(self):
        a1 = Agent("a1", _mock_backend())
        a2 = Agent("a2", _mock_backend())
        self.assertNotEqual(a1.agent_id, a2.agent_id)

    def test_roundtrip_empty(self):
        backend = _mock_backend()
        agent = Agent("tester", backend, role="analyst",
                      instructions="be precise", stateful=True)
        data = _roundtrip(agent)
        restored = Agent.from_dict(data, backend)

        self.assertEqual(restored.agent_id, agent.agent_id)
        self.assertEqual(restored.name, "tester")
        self.assertEqual(restored.role, "analyst")
        self.assertEqual(restored.instructions, "be precise")
        self.assertTrue(restored.stateful)
        self.assertEqual(restored.memory, {})
        self.assertEqual(len(restored.task_history), 0)
        self.assertEqual(len(restored._conversation_history), 0)

    def test_roundtrip_preserves_memory(self):
        backend = _mock_backend()
        agent = Agent("mem-test", backend)
        agent.remember("key1", "value1")
        agent.remember("nested", {"a": 1})
        restored = Agent.from_dict(_roundtrip(agent), backend)
        self.assertEqual(restored.recall("key1"), "value1")
        self.assertEqual(restored.recall("nested"), {"a": 1})

    def test_roundtrip_preserves_task_history(self):
        backend = _mock_backend()
        agent = Agent("hist-test", backend, stateful=True)
        agent.execute_task("do something")
        agent.execute_task("do another thing")
        data = _roundtrip(agent)
        restored = Agent.from_dict(data, backend)
        self.assertEqual(len(restored.task_history), 2)
        self.assertEqual(restored.task_history[0]['task'], "do something")
        self.assertEqual(restored.task_history[1]['task'], "do another thing")

    def test_roundtrip_preserves_conversation_history(self):
        backend = _mock_backend()
        agent = Agent("conv-test", backend, stateful=True)
        agent.execute_task("first task")
        data = _roundtrip(agent)
        restored = Agent.from_dict(data, backend)
        self.assertEqual(len(restored._conversation_history), 1)
        self.assertEqual(restored._conversation_history[0]['task'], "first task")

    def test_roundtrip_max_history_entries(self):
        backend = _mock_backend()
        agent = Agent("limit-test", backend)
        agent.set_max_history_entries(50)
        data = _roundtrip(agent)
        restored = Agent.from_dict(data, backend)
        self.assertEqual(restored._max_history_entries, 50)
        self.assertEqual(restored.task_history.maxlen, 50)

    def test_roundtrip_stateful_context_window(self):
        backend = _mock_backend()
        agent = Agent("ctx-test", backend)
        agent._stateful_context_window = 5
        data = _roundtrip(agent)
        restored = Agent.from_dict(data, backend)
        self.assertEqual(restored._stateful_context_window, 5)

    def test_to_dict_excludes_backend(self):
        agent = Agent("no-backend-test", _mock_backend())
        data = agent.to_dict()
        self.assertNotIn('maki', data)

    def test_restored_agent_can_execute_tasks(self):
        backend = _mock_backend()
        agent = Agent("exec-test", backend)
        data = _roundtrip(agent)
        restored = Agent.from_dict(data, backend)
        result = restored.execute_task("hello")
        self.assertEqual(result, "ok")
        self.assertEqual(len(restored.task_history), 1)


# ---------------------------------------------------------------------------
# WorkflowTask
# ---------------------------------------------------------------------------

class TestWorkflowTaskSerialization(unittest.TestCase):

    def test_roundtrip_minimal(self):
        wt = WorkflowTask(name="step1", agent="writer", task="write a poem")
        data = _roundtrip(wt)
        restored = WorkflowTask.from_dict(data)
        self.assertEqual(restored.name, "step1")
        self.assertEqual(restored.agent, "writer")
        self.assertEqual(restored.task, "write a poem")
        self.assertEqual(restored.dependencies, [])
        self.assertEqual(restored.status, TaskStatus.PENDING)

    def test_roundtrip_full(self):
        wt = WorkflowTask(
            name="step2", agent="analyst", task="analyse data",
            dependencies=["step1"], max_retries=5, retry_delay=2.0,
            parallelizable=True,
        )
        wt.status = TaskStatus.COMPLETED
        wt.result = "analysis done"
        wt.data = {"score": 0.9}
        wt.attempts = 2
        wt.execution_time = 3.14
        wt.resources_used = {"cpu": "50%"}
        data = _roundtrip(wt)
        restored = WorkflowTask.from_dict(data)

        self.assertEqual(restored.dependencies, ["step1"])
        self.assertEqual(restored.max_retries, 5)
        self.assertTrue(restored.parallelizable)
        self.assertEqual(restored.status, TaskStatus.COMPLETED)
        self.assertEqual(restored.result, "analysis done")
        self.assertEqual(restored.data, {"score": 0.9})
        self.assertEqual(restored.attempts, 2)
        self.assertAlmostEqual(restored.execution_time, 3.14)
        self.assertEqual(restored.resources_used, {"cpu": "50%"})

    def test_conditions_excluded(self):
        wt = WorkflowTask(name="cond", agent="a", task="t",
                          conditions=[lambda ctx: True])
        data = wt.to_dict()
        self.assertNotIn('conditions', data)
        restored = WorkflowTask.from_dict(data)
        self.assertEqual(restored.conditions, [])

    def test_all_task_statuses(self):
        for status in TaskStatus:
            wt = WorkflowTask(name="s", agent="a", task="t")
            wt.status = status
            restored = WorkflowTask.from_dict(_roundtrip(wt))
            self.assertEqual(restored.status, status)


# ---------------------------------------------------------------------------
# WorkflowState
# ---------------------------------------------------------------------------

class TestWorkflowStateSerialization(unittest.TestCase):

    def test_roundtrip_empty(self):
        state = WorkflowState(workflow_id="wf-001")
        data = _roundtrip(state)
        restored = WorkflowState.from_dict(data)
        self.assertEqual(restored.workflow_id, "wf-001")
        self.assertEqual(restored.status, "running")
        self.assertEqual(restored.tasks, {})
        self.assertEqual(restored.error_log, [])
        self.assertIsNone(restored.end_time)

    def test_roundtrip_with_tasks(self):
        state = WorkflowState(workflow_id="wf-002")
        state.update_task_status(
            "step1", TaskStatus.COMPLETED, result="done",
            execution_time=1.0, data={"x": 1},
        )
        state.update_task_status("step2", TaskStatus.FAILED, result="error msg")
        state.add_error("step2", "backend timeout")
        state.status = "completed"
        state.end_time = time.time()

        data = _roundtrip(state)
        restored = WorkflowState.from_dict(data)

        self.assertEqual(restored.status, "completed")
        self.assertIsNotNone(restored.end_time)
        self.assertEqual(len(restored.error_log), 1)
        self.assertEqual(restored.error_log[0]['task'], "step2")

        step1 = restored.tasks["step1"]
        self.assertEqual(step1['status'], TaskStatus.COMPLETED)
        self.assertEqual(step1['result'], "done")
        self.assertEqual(step1['data'], {"x": 1})

        step2 = restored.tasks["step2"]
        self.assertEqual(step2['status'], TaskStatus.FAILED)

    def test_task_status_survives_json(self):
        state = WorkflowState(workflow_id="wf-003")
        state.update_task_status("t", TaskStatus.IN_PROGRESS)
        restored = WorkflowState.from_dict(_roundtrip(state))
        self.assertEqual(restored.tasks["t"]['status'], TaskStatus.IN_PROGRESS)


if __name__ == '__main__':
    unittest.main()
