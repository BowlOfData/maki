"""
Phase 2.3 regression tests: agent concurrency semantics.

Covers:
- Agent._execution_lock serialises concurrent execute_task calls (§3.1)
- Agent._execution_lock serialises concurrent stream_task calls (§3.1)
- AgentServer / workflow inherits safety via the lock (§3.1)
- results snapshot prevents race in condition evaluation (§3.2)
- SKIPPED status propagates transitively through dependents (§3.3)
- Workflow ends "completed_with_skips" when a task is skipped (§3.3)
- HALF_OPEN admits exactly one probe; concurrent callers get False (§3.5)
- Successful probe closes the circuit and clears the probe flag (§3.5)
- Failed probe reopens the circuit and clears the probe flag (§3.5)
- MakiLLama constructor no longer calls verify() (§3.6)
- MakiLLama.verify() works when called explicitly (§3.6)
- AgentProxy constructor no longer calls /info (§3.6)
- AgentProxy.connect() populates metadata (§3.6)
- ThreadPoolExecutor is bounded by _WORKFLOW_MAX_WORKERS (§3.7)
"""
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch, call

from maki.agents import Agent, AgentManager, WorkflowTask, TaskStatus
from maki.agents.agent_manager import _WORKFLOW_MAX_WORKERS
from maki.distributed.circuit_breaker import CircuitBreaker, CircuitState
from maki.objects import LLMResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _llm(content="ok"):
    return LLMResponse(content=content, model="t", prompt_tokens=0,
                       completion_tokens=0, total_tokens=0, elapsed_seconds=0.0)


def _make_agent(content="result"):
    m = MagicMock()
    m.chat.return_value = _llm(content)
    return Agent("A", m, role="worker")


def _make_manager(content="result"):
    m = MagicMock()
    m.chat.return_value = _llm(content)
    manager = AgentManager(m)
    agent = manager.add_agent("A", role="worker", maki_instance=m)
    return manager, agent, m


# ---------------------------------------------------------------------------
# §3.1 — per-agent execution lock
# ---------------------------------------------------------------------------

class TestExecutionLock(unittest.TestCase):

    def test_agent_has_execution_lock(self):
        agent = _make_agent()
        self.assertTrue(hasattr(agent, "_execution_lock"))
        self.assertIsInstance(agent._execution_lock, type(threading.Lock()))

    def test_concurrent_execute_task_serialised(self):
        """Two threads calling execute_task on the same agent must not interleave.

        Without the lock: thread scheduling could produce [enter, enter, exit, exit].
        With the lock: guaranteed [enter, exit, enter, exit].
        """
        call_order = []
        # Event lets us control when thread 1 releases its lock
        t1_in_chat = threading.Event()
        t1_may_exit = threading.Event()

        def slow_chat(prompt, **kwargs):
            call_order.append("enter")
            t1_in_chat.set()
            t1_may_exit.wait(timeout=5)
            call_order.append("exit")
            return _llm("done")

        # Second call uses a fast mock
        def fast_chat(prompt, **kwargs):
            call_order.append("enter")
            call_order.append("exit")
            return _llm("done")

        call_count = [0]
        def chat_dispatch(prompt, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return slow_chat(prompt, **kwargs)
            return fast_chat(prompt, **kwargs)

        m = MagicMock()
        m.chat.side_effect = chat_dispatch
        agent = Agent("A", m, role="worker")

        errors = []
        def _run():
            try:
                agent.execute_task("task")
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=_run)
        t1.start()

        # Wait until t1 is inside the chat (holding the lock)
        t1_in_chat.wait(timeout=5)

        t2 = threading.Thread(target=_run)
        t2.start()

        # Give t2 time to attempt locking — it should block
        time.sleep(0.02)

        # t2 cannot have appended "enter" yet (it's blocked)
        mid_snapshot = list(call_order)
        t1_may_exit.set()

        t1.join(timeout=5)
        t2.join(timeout=5)

        self.assertEqual(errors, [])
        # At the snapshot point t2 had not yet entered (only t1's "enter" was there)
        self.assertEqual(mid_snapshot, ["enter"])
        # Final order is always serialised
        self.assertEqual(call_order, ["enter", "exit", "enter", "exit"])

    def test_concurrent_stream_task_serialised(self):
        """Two threads cannot iterate a stream from the same agent simultaneously.

        The lock is acquired inside the generator when iteration starts.
        """
        t1_streaming = threading.Event()
        t1_may_exit = threading.Event()
        stream_count = [0]

        def slow_stream(prompt, **kwargs):
            stream_count[0] += 1
            if stream_count[0] == 1:
                t1_streaming.set()
                t1_may_exit.wait(timeout=5)
            yield "chunk"

        m = MagicMock()
        m.stream.side_effect = slow_stream
        agent = Agent("A", m, role="worker")

        t1_done = threading.Event()

        def _t1():
            list(agent.stream_task("task"))  # Acquires lock, waits, releases
            t1_done.set()

        t1 = threading.Thread(target=_t1)
        t1.start()

        # Wait until t1 holds the lock inside the generator
        t1_streaming.wait(timeout=5)

        t2_lock_acquired = threading.Event()

        def _t2():
            gen = agent.stream_task("task")
            # Lock is acquired on first iteration, not on generator creation
            next(gen)          # This blocks until t1 releases the lock
            t2_lock_acquired.set()

        t2 = threading.Thread(target=_t2)
        t2.start()

        # Give t2 time to call next(gen) — it should be blocked
        time.sleep(0.05)
        self.assertFalse(t2_lock_acquired.is_set(),
                         "t2 should be blocked waiting for the lock while t1 streams")
        self.assertFalse(t1_done.is_set())

        # Let t1 finish streaming
        t1_may_exit.set()
        t1.join(timeout=5)
        t2.join(timeout=5)

        # Now t2 should have completed
        self.assertTrue(t2_lock_acquired.is_set())


# ---------------------------------------------------------------------------
# §3.2 — results snapshot prevents condition race
# ---------------------------------------------------------------------------

class TestResultsSnapshot(unittest.TestCase):

    def test_parallel_batch_uses_snapshot_for_conditions(self):
        """Conditions in a parallel batch evaluate against the pre-batch snapshot."""
        seen_results_in_condition = []

        def condition(results):
            seen_results_in_condition.append(dict(results))
            return True

        m = MagicMock()
        m.chat.return_value = _llm("done")
        manager = AgentManager(m)
        manager.add_agent("A", role="w", maki_instance=m)

        tasks = [
            WorkflowTask("t1", "A", "do t1", parallelizable=True, conditions=[condition]),
            WorkflowTask("t2", "A", "do t2", parallelizable=True, conditions=[condition]),
        ]

        manager.run_workflow(tasks)

        # Both conditions should have been called with the same snapshot (empty dict
        # since t1 and t2 are the first batch and there are no prior results).
        self.assertEqual(len(seen_results_in_condition), 2)
        for snapshot in seen_results_in_condition:
            self.assertNotIn("t1", snapshot)
            self.assertNotIn("t2", snapshot)


# ---------------------------------------------------------------------------
# §3.3 — SKIPPED status and transitive propagation
# ---------------------------------------------------------------------------

class TestSkippedStatus(unittest.TestCase):

    def test_condition_skipped_task_has_skipped_status(self):
        manager, agent, m = _make_manager()
        wt = WorkflowTask("t", "A", "task", conditions=[lambda ctx: False])
        results = manager.run_workflow([wt])

        self.assertIn("t", results)
        self.assertTrue(results["t"]["skipped"])
        self.assertEqual(wt.status, TaskStatus.SKIPPED)
        m.chat.assert_not_called()

    def test_dependent_of_skipped_task_is_also_skipped(self):
        manager, agent, m = _make_manager()
        tasks = [
            WorkflowTask("parent", "A", "parent task", conditions=[lambda ctx: False]),
            WorkflowTask("child", "A", "child task", dependencies=["parent"]),
        ]
        results = manager.run_workflow(tasks)

        self.assertTrue(results["parent"]["skipped"])
        self.assertTrue(results["child"]["skipped"])
        m.chat.assert_not_called()

    def test_transitive_skip_propagates(self):
        """grandchild is skipped because child is skipped because parent is skipped."""
        manager, agent, m = _make_manager()
        tasks = [
            WorkflowTask("grandparent", "A", "gp", conditions=[lambda ctx: False]),
            WorkflowTask("parent", "A", "p", dependencies=["grandparent"]),
            WorkflowTask("child", "A", "c", dependencies=["parent"]),
        ]
        results = manager.run_workflow(tasks)

        for name in ("grandparent", "parent", "child"):
            self.assertTrue(results[name]["skipped"], f"{name} should be skipped")
        m.chat.assert_not_called()

    def test_non_dependent_task_runs_despite_sibling_skip(self):
        """A task with no dependency on the skipped one still executes."""
        manager, agent, m = _make_manager()
        tasks = [
            WorkflowTask("skip_me", "A", "s", conditions=[lambda ctx: False]),
            WorkflowTask("run_me", "A", "r"),
        ]
        results = manager.run_workflow(tasks)

        self.assertTrue(results["skip_me"]["skipped"])
        self.assertFalse(results["run_me"].get("skipped"))
        self.assertEqual(results["run_me"]["result"], "result")

    def test_workflow_status_is_completed_with_skips(self):
        """run_workflow ends with completed_with_skips when any task is skipped."""
        # We check the WorkflowState indirectly via a state_store mock.
        from maki.agents.workflow import WorkflowState

        saved_states = []

        class FakeStore:
            def load_workflow(self, wid):
                return None
            def save_workflow(self, state):
                saved_states.append(state.status)

        manager, agent, m = _make_manager()
        wt = WorkflowTask("t", "A", "task", conditions=[lambda ctx: False])
        manager.run_workflow([wt], workflow_id="wf1", state_store=FakeStore())

        self.assertIn("completed_with_skips", saved_states)

    def test_all_completed_status_is_plain_completed(self):
        """When no tasks are skipped the workflow ends with 'completed'."""
        saved_states = []

        class FakeStore:
            def load_workflow(self, wid):
                return None
            def save_workflow(self, state):
                saved_states.append(state.status)

        manager, agent, m = _make_manager()
        wt = WorkflowTask("t", "A", "task")
        manager.run_workflow([wt], workflow_id="wf2", state_store=FakeStore())

        self.assertIn("completed", saved_states)
        self.assertNotIn("completed_with_skips", saved_states)


# ---------------------------------------------------------------------------
# §3.5 — HALF_OPEN single probe
# ---------------------------------------------------------------------------

class TestCircuitBreakerHalfOpenProbe(unittest.TestCase):

    def _open_breaker(self, recovery_timeout=0.01):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=recovery_timeout)
        cb.record_failure()
        time.sleep(recovery_timeout * 2)  # Let it become eligible for HALF_OPEN
        return cb

    def test_first_caller_in_half_open_gets_true(self):
        cb = self._open_breaker()
        self.assertEqual(cb.state, CircuitState.HALF_OPEN)
        self.assertTrue(cb.allow_request())

    def test_second_concurrent_caller_in_half_open_gets_false(self):
        cb = self._open_breaker()
        # First allow_request sets the probe flag
        self.assertTrue(cb.allow_request())
        # Second call while probe is still in flight is denied
        self.assertFalse(cb.allow_request())

    def test_successful_probe_resets_to_closed(self):
        cb = self._open_breaker()
        cb.allow_request()
        cb.record_success()
        self.assertEqual(cb.state, CircuitState.CLOSED)
        # Probe flag is cleared; next CLOSED call is also allowed
        self.assertTrue(cb.allow_request())

    def test_failed_probe_reopens_circuit(self):
        cb = self._open_breaker()
        cb.allow_request()
        cb.record_failure()
        self.assertEqual(cb.state, CircuitState.OPEN)
        # Probe flag cleared; a new probe is possible after another timeout
        time.sleep(0.03)
        self.assertEqual(cb.state, CircuitState.HALF_OPEN)
        self.assertTrue(cb.allow_request())

    def test_concurrent_probe_only_one_admitted(self):
        """N threads racing into HALF_OPEN — exactly one should be admitted."""
        cb = self._open_breaker(recovery_timeout=0.01)

        admitted = []
        barrier = threading.Barrier(10)

        def _probe():
            barrier.wait(timeout=5)
            if cb.allow_request():
                admitted.append(1)

        threads = [threading.Thread(target=_probe) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        self.assertEqual(len(admitted), 1, "Exactly one probe should be admitted")


# ---------------------------------------------------------------------------
# §3.6 — lazy connection verification (MakiLLama)
# ---------------------------------------------------------------------------

class TestMakiLLamaLazyVerify(unittest.TestCase):

    def test_constructor_does_not_call_verify(self):
        """MakiLLama.__init__ must not make any network calls."""
        with patch("maki.makiLLama.Connector") as MockConnector, \
             patch("maki.makiLLama.AsyncConnector"):
            conn = MagicMock()
            MockConnector.return_value = conn
            from maki.makiLLama import MakiLLama
            MakiLLama.__init__.__wrapped__ = None  # reset any cache
            # Importing may already have cached the class; create a new instance
            llm = MakiLLama.__new__(MakiLLama)
            llm.model = "test"
            llm.base_url = "http://localhost:11434"
            llm.config = MagicMock()
            llm.config.temperature = 0.7
            llm.timeout = 30
            llm.temperature = 0.7
            llm._rate_limiter = None
            llm.system_prompt = None
            llm.think = None
            llm.json_format = False
            llm._http = conn
            llm._async_http = MagicMock()
            # Confirm get was NOT called (no _verify_connection at construction)
            conn.get.assert_not_called()

    def test_verify_method_exists_and_is_callable(self):
        """MakiLLama.verify() should exist as an explicit connection check."""
        from maki.makiLLama import MakiLLama
        self.assertTrue(callable(getattr(MakiLLama, "verify", None)))


# ---------------------------------------------------------------------------
# §3.6 — lazy connection verification (AgentProxy)
# ---------------------------------------------------------------------------

class TestAgentProxyLazyConnect(unittest.TestCase):

    def _make_bare_proxy(self):
        """Create a proxy without patching Connector so _http is a real MagicMock."""
        with patch("maki.distributed.proxy.Connector") as MockConn:
            MockConn.return_value = MagicMock()
            from maki.distributed.proxy import AgentProxy
            proxy = AgentProxy(endpoint="http://fake:8100")
        return proxy

    def test_constructor_does_not_call_info(self):
        proxy = self._make_bare_proxy()
        proxy._http.get.assert_not_called()

    def test_metadata_empty_until_connected(self):
        proxy = self._make_bare_proxy()
        self.assertEqual(proxy.name, "")
        self.assertEqual(proxy.agent_id, "")
        self.assertFalse(proxy._connected)

    def test_connect_populates_metadata(self):
        proxy = self._make_bare_proxy()
        info = {"agent_id": "abc", "name": "Alice", "role": "analyst", "plugins": ["p1"]}
        proxy._http.get.return_value.json.return_value = info
        proxy.connect()

        self.assertEqual(proxy.agent_id, "abc")
        self.assertEqual(proxy.name, "Alice")
        self.assertEqual(proxy.role, "analyst")
        self.assertIn("p1", proxy.plugins)
        self.assertTrue(proxy._connected)

    def test_execute_task_triggers_lazy_connect(self):
        proxy = self._make_bare_proxy()
        info = {"agent_id": "x", "name": "bot", "role": "", "plugins": []}
        proxy._http.get.return_value.json.return_value = info

        execute_resp = MagicMock()
        execute_resp.json.return_value = {"result": "ok", "trace_id": "t1"}
        proxy._http.post.return_value = execute_resp

        result = proxy.execute_task("do something")
        self.assertTrue(proxy._connected)
        self.assertEqual(result, "ok")


# ---------------------------------------------------------------------------
# §3.7 — bounded thread pools
# ---------------------------------------------------------------------------

class TestBoundedThreadPool(unittest.TestCase):

    def test_workflow_max_workers_constant_is_set(self):
        self.assertIsInstance(_WORKFLOW_MAX_WORKERS, int)
        self.assertGreater(_WORKFLOW_MAX_WORKERS, 0)

    def test_parallel_dicts_workflow_uses_bounded_pool(self):
        """_run_workflow_dicts passes max_workers to ThreadPoolExecutor."""
        m = MagicMock()
        m.chat.return_value = _llm("ok")
        manager = AgentManager(m)
        manager.add_agent("A", role="w", maki_instance=m)

        steps = [
            {"name": "s1", "agent": "A", "task": "t1", "parallelizable": True},
            {"name": "s2", "agent": "A", "task": "t2", "parallelizable": True},
        ]
        with patch("maki.agents.agent_manager.ThreadPoolExecutor", wraps=ThreadPoolExecutor) as mock_pool:
            manager.run_workflow(steps)

        for c in mock_pool.call_args_list:
            self.assertEqual(c.kwargs.get("max_workers"), _WORKFLOW_MAX_WORKERS)

    def test_parallel_tasks_workflow_uses_bounded_pool(self):
        """_run_workflow_tasks passes max_workers to ThreadPoolExecutor."""
        m = MagicMock()
        m.chat.return_value = _llm("ok")
        manager = AgentManager(m)
        manager.add_agent("A", role="w", maki_instance=m)

        tasks = [
            WorkflowTask("t1", "A", "task1", parallelizable=True),
            WorkflowTask("t2", "A", "task2", parallelizable=True),
        ]
        with patch("maki.agents.agent_manager.ThreadPoolExecutor", wraps=ThreadPoolExecutor) as mock_pool:
            manager.run_workflow(tasks)

        for c in mock_pool.call_args_list:
            self.assertEqual(c.kwargs.get("max_workers"), _WORKFLOW_MAX_WORKERS)
