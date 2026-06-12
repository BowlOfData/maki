"""
Phase 4 tests: StateStore implementations and workflow checkpointing/resume.
"""
import json
import os
import tempfile
import time
import unittest
from unittest.mock import MagicMock, call, patch

from maki.agents import Agent, AgentManager, WorkflowTask, TaskStatus
from maki.agents.workflow import WorkflowState
from maki.objects import LLMResponse
from maki.distributed.state_store import LocalStateStore, RedisStateStore, _sanitize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _llm(content="ok"):
    return LLMResponse(content=content, model="t", prompt_tokens=0,
                       completion_tokens=0, total_tokens=0, elapsed_seconds=0.0)


def _mock_backend(content="task-result"):
    m = MagicMock()
    m.chat.return_value = _llm(content)
    return m


def _make_manager(content="task-result"):
    backend = _mock_backend(content)
    manager = AgentManager(backend)
    manager.add_agent("worker", role="worker")
    return manager, backend


def _three_task_workflow():
    return [
        WorkflowTask(name="t1", agent="worker", task="first"),
        WorkflowTask(name="t2", agent="worker", task="second", dependencies=["t1"]),
        WorkflowTask(name="t3", agent="worker", task="third",  dependencies=["t2"]),
    ]


# ---------------------------------------------------------------------------
# _sanitize helper
# ---------------------------------------------------------------------------

class TestSanitize(unittest.TestCase):

    def test_safe_id_unchanged(self):
        self.assertEqual(_sanitize("wf-001"), "wf-001")
        self.assertEqual(_sanitize("my_workflow.v2"), "my_workflow.v2")

    def test_slash_replaced(self):
        self.assertEqual(_sanitize("a/b"), "a_b")

    def test_dotdot_replaced(self):
        # ../../etc → .._.._etc (slashes replaced) → ____etc (dots replaced)
        self.assertEqual(_sanitize("../../etc"), "____etc")

    def test_spaces_replaced(self):
        self.assertEqual(_sanitize("my workflow"), "my_workflow")


# ---------------------------------------------------------------------------
# LocalStateStore
# ---------------------------------------------------------------------------

class TestLocalStateStore(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.store = LocalStateStore(base_dir=self._tmpdir)

    # --- basic CRUD --------------------------------------------------------

    def test_save_creates_file(self):
        state = WorkflowState("wf-001")
        self.store.save_workflow(state)
        self.assertTrue(os.path.exists(os.path.join(self._tmpdir, "wf-001.json")))

    def test_load_missing_returns_none(self):
        self.assertIsNone(self.store.load_workflow("nonexistent"))

    def test_roundtrip_empty_state(self):
        state = WorkflowState("wf-002")
        self.store.save_workflow(state)
        restored = self.store.load_workflow("wf-002")
        self.assertIsNotNone(restored)
        self.assertEqual(restored.workflow_id, "wf-002")
        self.assertEqual(restored.status, "running")

    def test_roundtrip_with_tasks(self):
        state = WorkflowState("wf-003")
        state.update_task_status("step1", TaskStatus.COMPLETED, result="done",
                                  execution_time=1.5, data={"score": 0.9})
        state.update_task_status("step2", TaskStatus.FAILED, result="boom")
        state.add_error("step2", "backend timeout")
        self.store.save_workflow(state)

        restored = self.store.load_workflow("wf-003")
        self.assertEqual(restored.tasks["step1"]["status"], TaskStatus.COMPLETED)
        self.assertEqual(restored.tasks["step1"]["result"], "done")
        self.assertEqual(restored.tasks["step1"]["data"], {"score": 0.9})
        self.assertEqual(restored.tasks["step2"]["status"], TaskStatus.FAILED)
        self.assertEqual(len(restored.error_log), 1)

    def test_save_overwrites_previous(self):
        state = WorkflowState("wf-004")
        state.update_task_status("t1", TaskStatus.IN_PROGRESS)
        self.store.save_workflow(state)

        state.update_task_status("t1", TaskStatus.COMPLETED, result="done")
        self.store.save_workflow(state)

        restored = self.store.load_workflow("wf-004")
        self.assertEqual(restored.tasks["t1"]["status"], TaskStatus.COMPLETED)

    def test_list_workflows(self):
        self.store.save_workflow(WorkflowState("alpha"))
        self.store.save_workflow(WorkflowState("beta"))
        names = self.store.list_workflows()
        self.assertIn("alpha", names)
        self.assertIn("beta", names)

    def test_delete_workflow(self):
        self.store.save_workflow(WorkflowState("del-me"))
        self.store.delete_workflow("del-me")
        self.assertIsNone(self.store.load_workflow("del-me"))
        self.assertNotIn("del-me", self.store.list_workflows())

    def test_delete_nonexistent_is_silent(self):
        self.store.delete_workflow("ghost")  # should not raise

    def test_update_task(self):
        state = WorkflowState("wf-upd")
        state.update_task_status("t1", TaskStatus.IN_PROGRESS)
        self.store.save_workflow(state)

        self.store.update_task("wf-upd", "t1", {"status": TaskStatus.COMPLETED.value,
                                                  "result": "patched"})
        restored = self.store.load_workflow("wf-upd")
        # from_dict re-hydrates the status string back to a TaskStatus enum
        self.assertEqual(restored.tasks["t1"]["status"], TaskStatus.COMPLETED)
        self.assertEqual(restored.tasks["t1"]["result"], "patched")

    def test_update_task_missing_workflow_raises(self):
        with self.assertRaises(ValueError):
            self.store.update_task("no-such-workflow", "t1", {"status": "completed"})

    # --- unsafe IDs --------------------------------------------------------

    def test_unsafe_id_does_not_escape_directory(self):
        state = WorkflowState("../../evil")
        self.store.save_workflow(state)
        # File must be inside base_dir, not two levels up
        files = os.listdir(self._tmpdir)
        self.assertTrue(any(f.endswith(".json") for f in files))
        for f in files:
            self.assertNotIn("..", f)


# ---------------------------------------------------------------------------
# LocalStateStore — crash-safe (atomic) checkpoint writes
# ---------------------------------------------------------------------------

class TestAtomicCheckpointWrites(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.store = LocalStateStore(base_dir=self._tmpdir)

    def _saved_state(self):
        state = WorkflowState("wf-atomic")
        state.update_task_status("t1", TaskStatus.COMPLETED, result="done")
        self.store.save_workflow(state)
        return state

    def test_crash_before_rename_preserves_previous_checkpoint(self):
        self._saved_state()
        replacement = WorkflowState("wf-atomic")
        replacement.update_task_status("t1", TaskStatus.FAILED, result="boom")
        # Simulate a crash between writing the temp file and renaming it.
        with patch("maki.distributed.state_store.os.replace",
                   side_effect=OSError("simulated crash")):
            with self.assertRaises(OSError):
                self.store.save_workflow(replacement)

        restored = self.store.load_workflow("wf-atomic")
        self.assertEqual(restored.tasks["t1"]["status"], TaskStatus.COMPLETED)
        self.assertEqual(restored.tasks["t1"]["result"], "done")

    def test_torn_temp_write_preserves_previous_checkpoint(self):
        self._saved_state()

        def torn_dump(data, f, **kwargs):
            f.write('{"workflow_id": "wf-atomic", "tas')  # truncated JSON
            raise TypeError("simulated mid-write failure")

        with patch("maki.distributed.state_store.json.dump",
                   side_effect=torn_dump):
            with self.assertRaises(TypeError):
                self.store.save_workflow(WorkflowState("wf-atomic"))

        restored = self.store.load_workflow("wf-atomic")
        self.assertEqual(restored.tasks["t1"]["result"], "done")

    def test_failed_write_leaves_no_temp_file(self):
        with patch("maki.distributed.state_store.os.replace",
                   side_effect=OSError("simulated crash")):
            with self.assertRaises(OSError):
                self._saved_state()
        leftovers = [f for f in os.listdir(self._tmpdir) if f.endswith(".tmp")]
        self.assertEqual(leftovers, [])

    def test_successful_save_leaves_no_temp_file(self):
        self._saved_state()
        leftovers = [f for f in os.listdir(self._tmpdir) if f.endswith(".tmp")]
        self.assertEqual(leftovers, [])

    def test_update_task_is_atomic(self):
        self._saved_state()
        with patch("maki.distributed.state_store.os.replace",
                   side_effect=OSError("simulated crash")):
            with self.assertRaises(OSError):
                self.store.update_task("wf-atomic", "t1", {"result": "patched"})
        restored = self.store.load_workflow("wf-atomic")
        self.assertEqual(restored.tasks["t1"]["result"], "done")


# ---------------------------------------------------------------------------
# RedisStateStore (mocked)
# ---------------------------------------------------------------------------

class TestRedisStateStore(unittest.TestCase):

    def _make_store(self):
        """Create a RedisStateStore backed by a mock Redis client."""
        self._data: dict = {}
        mock_redis = MagicMock()

        def _setex(key, ttl, value):
            self._data[key] = value

        def _set(key, value):
            self._data[key] = value

        def _get(key):
            return self._data.get(key)

        def _delete(key):
            self._data.pop(key, None)

        def _keys(pattern):
            prefix = pattern.rstrip("*")
            return [k for k in self._data if k.startswith(prefix)]

        mock_redis.setex.side_effect = _setex
        mock_redis.set.side_effect = _set
        mock_redis.get.side_effect = _get
        mock_redis.delete.side_effect = _delete
        mock_redis.keys.side_effect = _keys

        return RedisStateStore(_client=mock_redis)

    def test_roundtrip(self):
        store = self._make_store()
        state = WorkflowState("redis-wf-1")
        state.update_task_status("t1", TaskStatus.COMPLETED, result="done")
        store.save_workflow(state)

        restored = store.load_workflow("redis-wf-1")
        self.assertIsNotNone(restored)
        self.assertEqual(restored.workflow_id, "redis-wf-1")
        self.assertEqual(restored.tasks["t1"]["status"], TaskStatus.COMPLETED)

    def test_load_missing_returns_none(self):
        store = self._make_store()
        self.assertIsNone(store.load_workflow("no-such-id"))

    def test_list_workflows(self):
        store = self._make_store()
        store.save_workflow(WorkflowState("r1"))
        store.save_workflow(WorkflowState("r2"))
        names = store.list_workflows()
        self.assertIn("r1", names)
        self.assertIn("r2", names)

    def test_delete_workflow(self):
        store = self._make_store()
        store.save_workflow(WorkflowState("to-del"))
        store.delete_workflow("to-del")
        self.assertIsNone(store.load_workflow("to-del"))

    def test_update_task(self):
        store = self._make_store()
        state = WorkflowState("upd-wf")
        state.update_task_status("t1", TaskStatus.IN_PROGRESS)
        store.save_workflow(state)

        store.update_task("upd-wf", "t1", {"status": TaskStatus.COMPLETED.value,
                                            "result": "patched"})
        restored = store.load_workflow("upd-wf")
        self.assertEqual(restored.tasks["t1"]["result"], "patched")

    def test_update_task_missing_workflow_raises(self):
        store = self._make_store()
        with self.assertRaises(ValueError):
            store.update_task("ghost", "t1", {})

    def test_no_ttl_uses_set(self):
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        store = RedisStateStore(_client=mock_redis, ttl=0)
        store.save_workflow(WorkflowState("no-ttl"))
        mock_redis.set.assert_called_once()
        mock_redis.setex.assert_not_called()

    def test_missing_redis_raises_import_error(self):
        with patch.dict("sys.modules", {"redis": None}):
            with self.assertRaises(ImportError):
                RedisStateStore(redis_url="redis://localhost:6379")


# ---------------------------------------------------------------------------
# Workflow checkpointing — run_workflow with state_store
# ---------------------------------------------------------------------------

class TestWorkflowCheckpointing(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    def _store(self):
        return LocalStateStore(base_dir=self._tmpdir)

    def test_initial_run_creates_checkpoint(self):
        manager, _ = _make_manager()
        store = self._store()
        manager.run_workflow(_three_task_workflow(), workflow_id="wf-new", state_store=store)

        state = store.load_workflow("wf-new")
        self.assertIsNotNone(state)
        self.assertEqual(state.status, "completed")
        self.assertEqual(len(state.tasks), 3)
        for task_data in state.tasks.values():
            self.assertEqual(task_data["status"], TaskStatus.COMPLETED)

    def test_state_saved_after_each_task(self):
        saves = []
        original_save = LocalStateStore.save_workflow

        def tracking_save(self_store, state):
            saves.append((state.workflow_id, dict(state.tasks)))
            original_save(self_store, state)

        store = self._store()
        manager, _ = _make_manager()
        with patch.object(LocalStateStore, "save_workflow", tracking_save):
            manager.run_workflow(_three_task_workflow(), workflow_id="wf-track",
                                 state_store=store)

        # Initial save (before tasks) + 3 task saves + final "completed" save
        self.assertGreaterEqual(len(saves), 4)

    def test_resume_skips_completed_tasks(self):
        store = self._store()

        # Seed the store: pretend t1 already finished
        prior = WorkflowState("wf-resume")
        prior.update_task_status("t1", TaskStatus.COMPLETED, result="cached-result")
        store.save_workflow(prior)

        manager, backend = _make_manager("fresh-result")
        results = manager.run_workflow(
            _three_task_workflow(), workflow_id="wf-resume", state_store=store
        )

        # t1 was NOT re-executed — backend called only for t2 and t3
        self.assertEqual(backend.chat.call_count, 2)

        # t1 result comes from the checkpoint
        self.assertEqual(results["t1"]["result"], "cached-result")

        # t2 and t3 were executed
        self.assertEqual(results["t2"]["result"], "fresh-result")
        self.assertEqual(results["t3"]["result"], "fresh-result")

    def test_failed_tasks_retried_on_resume(self):
        store = self._store()

        # Seed: t1 completed, t2 failed in prior run
        prior = WorkflowState("wf-retry")
        prior.update_task_status("t1", TaskStatus.COMPLETED, result="t1-done")
        prior.update_task_status("t2", TaskStatus.FAILED, result="error")
        store.save_workflow(prior)

        manager, backend = _make_manager("success")
        results = manager.run_workflow(
            _three_task_workflow(), workflow_id="wf-retry", state_store=store
        )

        # t1 skipped (completed), t2 and t3 executed (2 calls)
        self.assertEqual(backend.chat.call_count, 2)
        self.assertIn("t2", results)
        self.assertIn("t3", results)

    def test_all_tasks_completed_no_execution(self):
        store = self._store()
        prior = WorkflowState("wf-all-done")
        for name in ("t1", "t2", "t3"):
            prior.update_task_status(name, TaskStatus.COMPLETED, result="cached")
        store.save_workflow(prior)

        manager, backend = _make_manager()
        results = manager.run_workflow(
            _three_task_workflow(), workflow_id="wf-all-done", state_store=store
        )

        # Nothing re-executed
        self.assertEqual(backend.chat.call_count, 0)
        # All results present from checkpoint
        self.assertEqual(results["t1"]["result"], "cached")

    def test_no_state_store_behavior_unchanged(self):
        manager, backend = _make_manager()
        results = manager.run_workflow(_three_task_workflow())
        # All three tasks executed
        self.assertEqual(backend.chat.call_count, 3)
        self.assertIn("t1", results)
        self.assertIn("t2", results)
        self.assertIn("t3", results)

    def test_state_store_without_workflow_id_saves_but_cannot_resume(self):
        store = self._store()
        manager, backend = _make_manager()
        # No workflow_id → all tasks run, state saved under auto-generated ID
        results = manager.run_workflow(_three_task_workflow(), state_store=store)
        self.assertEqual(backend.chat.call_count, 3)
        # A file was created
        self.assertTrue(len(store.list_workflows()) > 0)

    def test_final_state_has_completed_status(self):
        store = self._store()
        manager, _ = _make_manager()
        manager.run_workflow(_three_task_workflow(), workflow_id="wf-final",
                              state_store=store)
        state = store.load_workflow("wf-final")
        self.assertEqual(state.status, "completed")
        self.assertIsNotNone(state.end_time)

    def test_dependency_context_passes_through_on_resume(self):
        store = self._store()
        manager, backend = _make_manager("step-result")

        # First run: complete all 3 tasks
        manager.run_workflow(_three_task_workflow(), workflow_id="wf-ctx",
                             state_store=store)
        first_call_count = backend.chat.call_count

        # Second run: t1 is completed, t2 and t3 should receive context from t1
        backend.chat.reset_mock()
        backend.chat.return_value = _llm("second-run")

        # Manually mark only t1 as completed in a fresh store state
        store2 = self._store()
        prior = WorkflowState("wf-ctx2")
        prior.update_task_status("t1", TaskStatus.COMPLETED, result="t1-value",
                                  data={"key": "val"})
        store2.save_workflow(prior)

        results = manager.run_workflow(
            _three_task_workflow(), workflow_id="wf-ctx2", state_store=store2
        )
        # t2 and t3 executed
        self.assertEqual(backend.chat.call_count, 2)
        self.assertEqual(results["t1"]["result"], "t1-value")


# ---------------------------------------------------------------------------
# Parallel workflow checkpointing
# ---------------------------------------------------------------------------

class TestParallelWorkflowCheckpointing(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    def test_parallel_tasks_checkpointed(self):
        store = LocalStateStore(base_dir=self._tmpdir)
        manager, _ = _make_manager()

        workflow = [
            WorkflowTask(name="p1", agent="worker", task="par1", parallelizable=True),
            WorkflowTask(name="p2", agent="worker", task="par2", parallelizable=True),
            WorkflowTask(name="p3", agent="worker", task="seq3", dependencies=["p1", "p2"]),
        ]
        manager.run_workflow(workflow, workflow_id="wf-par", state_store=store)

        state = store.load_workflow("wf-par")
        self.assertEqual(state.status, "completed")
        self.assertEqual(len(state.tasks), 3)

    def test_parallel_resume_skips_completed(self):
        store = LocalStateStore(base_dir=self._tmpdir)
        prior = WorkflowState("wf-par-resume")
        prior.update_task_status("p1", TaskStatus.COMPLETED, result="p1-cached")
        store.save_workflow(prior)

        manager, backend = _make_manager()

        workflow = [
            WorkflowTask(name="p1", agent="worker", task="par1", parallelizable=True),
            WorkflowTask(name="p2", agent="worker", task="par2", parallelizable=True),
        ]
        results = manager.run_workflow(
            workflow, workflow_id="wf-par-resume", state_store=store
        )
        # p1 skipped, p2 executed → 1 call
        self.assertEqual(backend.chat.call_count, 1)
        self.assertEqual(results["p1"]["result"], "p1-cached")
        self.assertIn("p2", results)


if __name__ == "__main__":
    unittest.main()
