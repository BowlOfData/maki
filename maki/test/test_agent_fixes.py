"""
Tests for all issues identified in the critical review of the agents implementation.

Covers:
  1.  use_plugins wired to plugin invocation
  2.  WorkflowTask / WorkflowState updated by run_workflow
  3.  Dependency enforcement (topological sort + cycle detection)
  4.  Stateful conversation history
  5.  decompose_task raises ValueError on invalid JSON
  6.  self_correct supports max_iterations
  7.  History uses deque (O(1) appends, bounded)
  8.  execute_task_with_retry only retries transient errors
  9.  stream_task delegates to backend.stream()
 10.  collaborative_task exposes partial failures via strict=True
 11.  _topological_sort raises on circular deps
"""

import unittest
from collections import deque
from unittest.mock import patch, MagicMock, call

from maki.maki import Maki
from maki.agents import Agent, AgentManager, WorkflowTask, TaskStatus, WorkflowState
from maki.exceptions import MakiNetworkError, MakiTimeoutError, MakiAPIError
from maki.objects import LLMResponse


def _r(content: str) -> LLMResponse:
    return LLMResponse(content=content, model="test", prompt_tokens=0,
                       completion_tokens=0, total_tokens=0, elapsed_seconds=0.0)


class Base(unittest.TestCase):
    def setUp(self):
        self.maki = Maki("localhost", "11434", "llama3", 0.7)
        self.agent = Agent("TestAgent", self.maki, "researcher", "Be helpful")
        self.manager = AgentManager(self.maki)
        self.manager.add_agent("A", "analyst", "Analyse things")


# ---------------------------------------------------------------------------
# 1. use_plugins
# ---------------------------------------------------------------------------
class TestUsePlugins(Base):
    def test_no_plugins_loaded_use_plugins_false(self):
        """execute_task works normally when no plugins are loaded"""
        with patch.object(self.maki, 'request', return_value=_r("ok")) as mock:
            result = self.agent.execute_task("do something", use_plugins=False)
        self.assertEqual(result, "ok")

    def test_use_plugins_adds_plugin_section_to_prompt(self):
        """When use_plugins=True and a plugin is registered, its name appears in the prompt"""
        fake_plugin = MagicMock()
        fake_plugin.read_file = MagicMock(return_value="file content")
        self.agent.plugins['FileReader'] = fake_plugin

        captured_prompts = []

        def fake_request(prompt):
            captured_prompts.append(prompt)
            return _r("final answer")

        with patch.object(self.maki, 'request', side_effect=fake_request):
            self.agent.execute_task("read a file", use_plugins=True)

        self.assertTrue(any("FileReader" in p for p in captured_prompts))
        self.assertTrue(any("TOOL:" in p for p in captured_prompts))

    def test_tool_call_executed_and_result_fed_back(self):
        """TOOL: directive in LLM response triggers plugin call and follow-up request"""
        fake_plugin = MagicMock()
        fake_plugin.read_file = MagicMock(return_value="file content here")
        self.agent.plugins['FileReader'] = fake_plugin

        responses = iter([
            _r('TOOL: {"plugin": "FileReader", "method": "read_file", "args": {"file_path": "/tmp/x.txt"}}\nPartial answer.'),
            _r("Final answer with file content.")
        ])

        with patch.object(self.maki, 'request', side_effect=lambda p: next(responses)):
            result = self.agent.execute_task("read /tmp/x.txt", use_plugins=True)

        fake_plugin.read_file.assert_called_once_with(file_path="/tmp/x.txt")
        self.assertEqual(result, "Final answer with file content.")

    def test_use_plugins_false_no_plugin_call(self):
        """When use_plugins=False plugins are never invoked"""
        fake_plugin = MagicMock()
        self.agent.plugins['FileReader'] = fake_plugin

        with patch.object(self.maki, 'request', return_value=_r("answer")):
            self.agent.execute_task("do something", use_plugins=False)

        fake_plugin.read_file.assert_not_called()

    def test_unknown_plugin_in_tool_call_handled_gracefully(self):
        """An unknown plugin name in a TOOL: call produces an error entry, not an exception"""
        self.agent.plugins['FileReader'] = MagicMock()

        responses = iter([
            _r('TOOL: {"plugin": "NonExistent", "method": "foo", "args": {}}\npartial'),
            _r("final")
        ])
        with patch.object(self.maki, 'request', side_effect=lambda p: next(responses)):
            result = self.agent.execute_task("task", use_plugins=True)

        self.assertEqual(result, "final")


# ---------------------------------------------------------------------------
# 2. WorkflowTask / WorkflowState updated by run_workflow
# ---------------------------------------------------------------------------
class TestWorkflowTaskIntegration(Base):
    def test_run_workflow_tasks_updates_status_and_result(self):
        """After run_workflow(List[WorkflowTask]), task.status == COMPLETED and task.result is set"""
        wt = WorkflowTask(name="t1", agent="A", task="do it")
        self.assertEqual(wt.status, TaskStatus.PENDING)

        with patch.object(self.maki, 'request', return_value=_r("done")):
            self.manager.run_workflow([wt])

        self.assertEqual(wt.status, TaskStatus.COMPLETED)
        self.assertEqual(wt.result, "done")
        self.assertGreater(wt.execution_time, 0.0)

    def test_run_workflow_tasks_returns_results_dict(self):
        wt = WorkflowTask(name="step1", agent="A", task="analyse")
        with patch.object(self.maki, 'request', return_value=_r("analysis")):
            results = self.manager.run_workflow([wt])

        self.assertIn("step1", results)
        self.assertEqual(results["step1"]["result"], "analysis")

    def test_failed_task_marked_failed(self):
        wt = WorkflowTask(name="bad", agent="A", task="fail me", max_retries=1)
        with patch.object(self.maki, 'request', side_effect=MakiNetworkError("oops")):
            results = self.manager.run_workflow([wt])

        self.assertEqual(wt.status, TaskStatus.FAILED)
        self.assertNotIn("bad", results)

    def test_condition_prevents_execution(self):
        """A WorkflowTask whose condition returns False is skipped"""
        wt = WorkflowTask(
            name="skip_me", agent="A", task="skipped",
            conditions=[lambda ctx: False]
        )
        with patch.object(self.maki, 'request', return_value=_r("should not appear")) as mock:
            results = self.manager.run_workflow([wt])

        mock.assert_not_called()
        self.assertNotIn("skip_me", results)
        self.assertEqual(wt.status, TaskStatus.PENDING)


# ---------------------------------------------------------------------------
# 3. Dependency enforcement
# ---------------------------------------------------------------------------
class TestDependencyEnforcement(Base):
    def test_dependencies_respected_in_order(self):
        """task2 depends on task1; both must complete successfully in order"""
        call_order = []

        def fake_request(prompt):
            if "task1_content" in prompt:
                call_order.append("t1")
                return _r("result1")
            if "task2_content" in prompt:
                call_order.append("t2")
                return _r("result2")
            return _r("ok")

        t1 = WorkflowTask(name="t1", agent="A", task="task1_content")
        t2 = WorkflowTask(name="t2", agent="A", task="task2_content", dependencies=["t1"])

        # Pass in reverse order to prove topological sort reorders them
        with patch.object(self.maki, 'request', side_effect=fake_request):
            results = self.manager.run_workflow([t2, t1])

        self.assertEqual(call_order, ["t1", "t2"])
        self.assertIn("t1", results)
        self.assertIn("t2", results)

    def test_circular_dependency_raises(self):
        t1 = WorkflowTask(name="t1", agent="A", task="a", dependencies=["t2"])
        t2 = WorkflowTask(name="t2", agent="A", task="b", dependencies=["t1"])

        with self.assertRaises(ValueError, msg="Circular dependency"):
            self.manager._topological_sort([t1, t2])

    def test_missing_dependency_skipped_gracefully(self):
        """A dependency that is not in the task list is ignored (absent from results)"""
        t1 = WorkflowTask(name="t1", agent="A", task="do it", dependencies=["ghost"])
        with patch.object(self.maki, 'request', return_value=_r("ok")):
            results = self.manager.run_workflow([t1])
        self.assertIn("t1", results)


# ---------------------------------------------------------------------------
# 4. Stateful conversation history
# ---------------------------------------------------------------------------
class TestStatefulAgent(Base):
    def test_stateful_false_no_history_in_prompt(self):
        agent = Agent("Bot", self.maki, "assistant", "", stateful=False)
        captured = []
        with patch.object(self.maki, 'request', side_effect=lambda p: captured.append(p) or _r("r")):
            agent.execute_task("first")
            agent.execute_task("second")

        self.assertNotIn("Prior conversation", captured[1])

    def test_stateful_true_history_in_subsequent_prompt(self):
        agent = Agent("Bot", self.maki, "assistant", "", stateful=True)
        captured = []
        with patch.object(self.maki, 'request', side_effect=lambda p: captured.append(p) or _r("response")):
            agent.execute_task("first task")
            agent.execute_task("second task")

        self.assertIn("Prior conversation", captured[1])
        self.assertIn("first task", captured[1])

    def test_reset_conversation_clears_history(self):
        agent = Agent("Bot", self.maki, "assistant", "", stateful=True)
        with patch.object(self.maki, 'request', return_value=_r("r")):
            agent.execute_task("task one")

        agent.reset_conversation()
        self.assertEqual(len(agent._conversation_history), 0)

        captured = []
        with patch.object(self.maki, 'request', side_effect=lambda p: captured.append(p) or _r("r")):
            agent.execute_task("task two")

        self.assertNotIn("Prior conversation", captured[0])


# ---------------------------------------------------------------------------
# 5. decompose_task raises ValueError on bad JSON
# ---------------------------------------------------------------------------
class TestDecomposeTaskError(Base):
    def test_valid_json_returns_subtasks(self):
        payload = '[{"description": "Step 1", "resources": "none", "expected_outcome": "done"}]'
        with patch.object(self.maki, 'request', return_value=_r(payload)):
            subtasks = self.agent.decompose_task("big task")
        self.assertEqual(len(subtasks), 1)
        self.assertEqual(subtasks[0]["description"], "Step 1")

    def test_invalid_json_raises_value_error(self):
        with patch.object(self.maki, 'request', return_value=_r("This is not JSON at all.")):
            with self.assertRaises(ValueError):
                self.agent.decompose_task("big task")

    def test_non_list_json_raises_value_error(self):
        with patch.object(self.maki, 'request', return_value=_r('{"key": "value"}')):
            with self.assertRaises(ValueError):
                self.agent.decompose_task("big task")


# ---------------------------------------------------------------------------
# 6. self_correct with max_iterations
# ---------------------------------------------------------------------------
class TestSelfCorrect(Base):
    def test_single_iteration_default(self):
        with patch.object(self.maki, 'request', return_value=_r("improved")) as mock:
            result = self.agent.self_correct("original", "be better")
        mock.assert_called_once()
        self.assertEqual(result, "improved")

    def test_multiple_iterations(self):
        call_count = [0]

        def fake(prompt):
            call_count[0] += 1
            return _r(f"v{call_count[0]}")

        # 3 iterations → 3 LLM calls
        with patch.object(self.maki, 'request', side_effect=fake):
            result = self.agent.self_correct("original", "be better", max_iterations=3)

        self.assertEqual(call_count[0], 3)
        self.assertEqual(result, "v3")

    def test_each_iteration_recorded_in_history(self):
        responses = [_r("v1"), _r("v2")]
        with patch.object(self.maki, 'request', side_effect=responses):
            self.agent.self_correct("original", "feedback", max_iterations=2)

        self.assertEqual(len(self.agent.reasoning_history), 2)
        self.assertEqual(self.agent.reasoning_history[0]['iteration'], 1)
        self.assertEqual(self.agent.reasoning_history[1]['iteration'], 2)


# ---------------------------------------------------------------------------
# 7. History uses deque
# ---------------------------------------------------------------------------
class TestHistoryDeque(Base):
    def test_task_history_is_deque(self):
        self.assertIsInstance(self.agent.task_history, deque)

    def test_reasoning_history_is_deque(self):
        self.assertIsInstance(self.agent.reasoning_history, deque)

    def test_deque_maxlen_enforced_without_cleanup_call(self):
        """deque auto-trims; no explicit cleanup needed"""
        self.agent._max_history_entries = 5
        self.agent.task_history = deque(maxlen=5)

        with patch.object(self.maki, 'request', return_value=_r("r")):
            for i in range(10):
                self.agent.execute_task(f"task {i}")

        self.assertEqual(len(self.agent.task_history), 5)
        self.assertEqual(self.agent.task_history[0]['task'], "task 5")
        self.assertEqual(self.agent.task_history[-1]['task'], "task 9")

    def test_set_max_history_entries_recreates_deque(self):
        with patch.object(self.maki, 'request', return_value=_r("r")):
            for i in range(10):
                self.agent.execute_task(f"task {i}")

        self.agent.set_max_history_entries(3)
        self.assertIsInstance(self.agent.task_history, deque)
        self.assertEqual(self.agent.task_history.maxlen, 3)
        self.assertLessEqual(len(self.agent.task_history), 3)

    def test_cleanup_history_is_noop(self):
        """_cleanup_history should be a no-op (deque self-manages)"""
        self.agent._cleanup_history()  # Must not raise


# ---------------------------------------------------------------------------
# 8. execute_task_with_retry only retries transient errors
# ---------------------------------------------------------------------------
class TestRetryBehaviour(Base):
    def test_retries_on_network_error(self):
        side_effects = [MakiNetworkError("conn"), MakiNetworkError("conn"), _r("ok")]
        with patch.object(self.maki, 'request', side_effect=side_effects):
            result = self.agent.execute_task_with_retry("task", retry_delay=0)
        self.assertEqual(result, "ok")

    def test_retries_on_timeout_error(self):
        side_effects = [MakiTimeoutError("timeout"), _r("ok")]
        with patch.object(self.maki, 'request', side_effect=side_effects):
            result = self.agent.execute_task_with_retry("task", retry_delay=0)
        self.assertEqual(result, "ok")

    def test_no_retry_on_api_error(self):
        """MakiAPIError is non-transient and must not be retried"""
        with patch.object(self.maki, 'request', side_effect=MakiAPIError("bad response")) as mock:
            with self.assertRaises(MakiAPIError):
                self.agent.execute_task_with_retry("task", max_retries=3, retry_delay=0)

        # Only one attempt should have been made
        self.assertEqual(mock.call_count, 1)

    def test_no_retry_on_value_error(self):
        """ValueError (bad input) must not be retried"""
        with patch.object(self.maki, 'request', side_effect=ValueError("bad")) as mock:
            with self.assertRaises(Exception):
                self.agent.execute_task_with_retry("task", max_retries=3, retry_delay=0)

        self.assertEqual(mock.call_count, 1)

    def test_raises_after_max_retries_exhausted(self):
        with patch.object(self.maki, 'request', side_effect=MakiNetworkError("always fails")):
            with self.assertRaises(MakiNetworkError):
                self.agent.execute_task_with_retry("task", max_retries=2, retry_delay=0)


# ---------------------------------------------------------------------------
# 9. stream_task
# ---------------------------------------------------------------------------
class TestStreamTask(Base):
    def test_stream_task_raises_if_backend_has_no_stream(self):
        """Base Maki has no stream() method → NotImplementedError"""
        with self.assertRaises(NotImplementedError):
            self.agent.stream_task("tell me a story")

    def test_stream_task_delegates_to_backend_stream(self):
        """If maki.stream() exists, stream_task returns its generator"""
        self.maki.stream = MagicMock(return_value=iter(["chunk1", "chunk2"]))
        gen = self.agent.stream_task("tell me a story")
        chunks = list(gen)
        self.maki.stream.assert_called_once()
        self.assertEqual(chunks, ["chunk1", "chunk2"])

    def test_stream_task_validates_empty_task(self):
        self.maki.stream = MagicMock()
        with self.assertRaises(ValueError):
            self.agent.stream_task("   ")


# ---------------------------------------------------------------------------
# 10. collaborative_task partial failure / strict mode
# ---------------------------------------------------------------------------
class TestCollaborativeTaskStrict(Base):
    def setUp(self):
        super().setUp()
        self.manager.add_agent("B", "writer", "Write things")

    def test_strict_false_proceeds_with_partial_results(self):
        """When strict=False (default) and one agent fails, synthesis still runs"""
        call_count = [0]

        def fake_request(prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                raise MakiNetworkError("agent A down")
            return _r("synthesised")

        with patch.object(self.maki, 'request', side_effect=fake_request):
            result = self.manager.collaborative_task("task", ["A", "B"])

        self.assertEqual(result, "synthesised")

    def test_strict_true_raises_on_any_failure(self):
        """When strict=True any agent failure raises RuntimeError"""
        call_count = [0]

        def fake_request(prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                raise MakiNetworkError("agent A down")
            return _r("ok")

        with patch.object(self.maki, 'request', side_effect=fake_request):
            with self.assertRaises(RuntimeError):
                self.manager.collaborative_task("task", ["A", "B"], strict=True)

    def test_all_agents_fail_raises_runtime_error(self):
        with patch.object(self.maki, 'request', side_effect=MakiNetworkError("down")):
            with self.assertRaises(RuntimeError):
                self.manager.collaborative_task("task", ["A"])


# ---------------------------------------------------------------------------
# 11. Circular dependency detection
# ---------------------------------------------------------------------------
class TestCircularDependency(Base):
    def test_self_referencing_task(self):
        t = WorkflowTask(name="t1", agent="A", task="x", dependencies=["t1"])
        with self.assertRaises(ValueError):
            self.manager._topological_sort([t])

    def test_three_node_cycle(self):
        t1 = WorkflowTask("t1", "A", "a", dependencies=["t3"])
        t2 = WorkflowTask("t2", "A", "b", dependencies=["t1"])
        t3 = WorkflowTask("t3", "A", "c", dependencies=["t2"])
        with self.assertRaises(ValueError):
            self.manager._topological_sort([t1, t2, t3])


# ---------------------------------------------------------------------------
# 12. coordinate_agents / assign_task return-type and failure contracts
# ---------------------------------------------------------------------------
class TestCoordinateAgentsReturnType(Base):
    def test_synthesis_success_includes_final_synthesis_key(self):
        """When coordination_prompt is given and synthesis succeeds, 'final_synthesis' is a string."""
        with patch.object(self.maki, 'request', return_value=_r("ok")):
            result = self.manager.coordinate_agents(
                [{'agent': 'A', 'task': 'do something'}],
                coordination_prompt="Summarise"
            )
        self.assertIn('final_synthesis', result)
        self.assertEqual(result['final_synthesis'], "ok")

    def test_synthesis_failure_sets_final_synthesis_to_none(self):
        """When synthesis fails, 'final_synthesis' is None and individual results survive."""
        call_count = [0]

        def fake_request(prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                return _r("agent result")
            raise MakiNetworkError("synthesis down")

        with patch.object(self.maki, 'request', side_effect=fake_request):
            result = self.manager.coordinate_agents(
                [{'agent': 'A', 'task': 'do something'}],
                coordination_prompt="Summarise"
            )

        self.assertIn('final_synthesis', result)
        self.assertIsNone(result['final_synthesis'])
        # Individual agent results must still be present
        task_keys = [k for k in result if k != 'final_synthesis']
        self.assertEqual(len(task_keys), 1)
        self.assertEqual(result[task_keys[0]], "agent result")

    def test_no_coordination_prompt_omits_final_synthesis_key(self):
        """Without a coordination_prompt, 'final_synthesis' should not appear."""
        with patch.object(self.maki, 'request', return_value=_r("agent result")):
            result = self.manager.coordinate_agents(
                [{'agent': 'A', 'task': 'do something'}]
            )
        self.assertNotIn('final_synthesis', result)

    def test_task_failure_propagates_as_runtime_error(self):
        """A failing agent task raises RuntimeError out of coordinate_agents."""
        with patch.object(self.maki, 'request', side_effect=MakiNetworkError("agent down")):
            with self.assertRaises(RuntimeError):
                self.manager.coordinate_agents(
                    [{'agent': 'A', 'task': 'do something'}]
                )

    def test_assign_task_wraps_exception_with_context(self):
        """assign_task raises RuntimeError chained to the original exception."""
        original = MakiNetworkError("backend down")
        with patch.object(self.maki, 'request', side_effect=original):
            with self.assertRaises(RuntimeError) as ctx:
                self.manager.assign_task('A', 'do something')
        self.assertIs(ctx.exception.__cause__, original)


class TestCollaborativeTaskSynthesisFailure(Base):
    def setUp(self):
        super().setUp()
        self.manager.add_agent("B", "writer", "Write things")

    def test_synthesis_failure_after_agent_success_raises_runtime_error(self):
        """When agents succeed but synthesis fails, RuntimeError is raised."""
        call_count = [0]

        def fake_request(prompt):
            call_count[0] += 1
            if call_count[0] <= 2:          # agent A and B succeed
                return _r(f"result {call_count[0]}")
            raise MakiNetworkError("synthesis down")  # synthesis fails

        with patch.object(self.maki, 'request', side_effect=fake_request):
            with self.assertRaises(RuntimeError) as ctx:
                self.manager.collaborative_task("task", ["A", "B"])

        self.assertIn("synthesis failed", str(ctx.exception).lower())
        self.assertIsInstance(ctx.exception.__cause__, MakiNetworkError)


if __name__ == '__main__':
    unittest.main()
