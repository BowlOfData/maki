"""Tests for approval-gated workflow tasks (wait, don't skip) and GateResult reasons."""
import os
import tempfile
import unittest
from unittest.mock import MagicMock

from maki.agents import (
    AgentManager, ApprovalSpec, GateResult, InMemoryApprovalStore, LocalApprovalStore,
    TaskStatus, WorkflowTask,
)
from maki.agents.approvals import ApprovalDecision
from maki.distributed.state_store import LocalStateStore


class StubAgent:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def execute_task_with_retry(self, task, context=None, max_retries=3, retry_delay=1.0):
        self.log.append((self.name, task, dict(context or {})))
        return f"done:{task}"


def make_manager(log):
    mgr = AgentManager(MagicMock())
    mgr.agents["a"] = StubAgent("a", log)
    return mgr


def chain(key="k1", ttl=None):
    return [
        WorkflowTask("plan", "a", "plan"),
        WorkflowTask("launch", "a", "launch", dependencies=["plan"],
                     approval=ApprovalSpec(key_fn=lambda ctx: key, summary_fn=lambda ctx: "launch it",
                                           ttl_seconds=ttl)),
        WorkflowTask("report", "a", "report", dependencies=["launch"]),
        WorkflowTask("independent", "a", "independent"),
    ]


class TestApprovalWorkflow(unittest.TestCase):
    def test_parks_instead_of_skipping_and_blocks_downstream(self):
        log, store = [], InMemoryApprovalStore()
        mgr = make_manager(log)
        res = mgr.run_workflow(chain(), workflow_id="w", approval_store=store)
        ran = [t for _, t, _ in log]
        self.assertEqual(sorted(ran), ["independent", "plan"])  # launch/report did not run
        st = mgr.last_workflow_state
        self.assertEqual(st.status, "awaiting_approval")
        self.assertEqual(st.tasks["launch"]["status"], TaskStatus.AWAITING_APPROVAL)
        self.assertEqual(st.tasks["report"]["status"], TaskStatus.PENDING)  # blocked, not skipped
        self.assertTrue(res["launch"]["awaiting_approval"])
        self.assertTrue(res["report"]["blocked"])
        self.assertEqual([r.task for r in store.list_pending("w")], ["launch"])
        self.assertIsNone(st.end_time)

    def test_resume_after_approval_runs_remaining_only(self):
        log, store = [], InMemoryApprovalStore()
        with tempfile.TemporaryDirectory() as d:
            ss = LocalStateStore(base_dir=d)
            mgr = make_manager(log)
            mgr.run_workflow(chain(), workflow_id="w", state_store=ss, approval_store=store)
            store.approve("w", "launch", key="k1")
            log.clear()
            mgr2 = make_manager(log)
            mgr2.run_workflow(chain(), workflow_id="w", state_store=ss, approval_store=store)
            self.assertEqual([t for _, t, _ in log], ["launch", "report"])  # plan/independent not rerun
            self.assertEqual(mgr2.last_workflow_state.status, "completed")
            self.assertIsNotNone(mgr2.last_workflow_state.end_time)

    def test_changed_content_invalidates_approval(self):
        log, store = [], InMemoryApprovalStore()
        mgr = make_manager(log)
        mgr.run_workflow(chain("k1"), workflow_id="w", approval_store=store)
        store.approve("w", "launch", key="k1")
        log.clear()
        make_manager(log).run_workflow(chain("k2"), workflow_id="w", approval_store=store)  # content changed
        self.assertNotIn("launch", [t for _, t, _ in log])
        self.assertEqual(store.get("w", "launch").status, "pending")
        self.assertEqual(store.get("w", "launch").key, "k2")

    def test_approve_with_stale_key_refused(self):
        store = InMemoryApprovalStore()
        store.check("w", "t", "k2", "s")
        with self.assertRaises(ValueError):
            store.approve("w", "t", key="k1")
        with self.assertRaises(KeyError):
            store.approve("w", "missing")

    def test_rejection_skips_downstream(self):
        log, store = [], InMemoryApprovalStore()
        make_manager(log).run_workflow(chain(), workflow_id="w", approval_store=store)
        store.reject("w", "launch", key="k1", note="no")
        log.clear()
        mgr = make_manager(log)
        mgr.run_workflow(chain(), workflow_id="w", approval_store=store)
        st = mgr.last_workflow_state
        self.assertEqual(st.tasks["launch"]["status"], TaskStatus.SKIPPED)
        self.assertEqual(st.tasks["report"]["status"], TaskStatus.SKIPPED)
        self.assertNotIn("launch", [t for _, t, _ in log])

    def test_expired_approval_requires_new_one(self):
        store = InMemoryApprovalStore()
        self.assertEqual(store.check("w", "t", "k", "s"), ApprovalDecision.PENDING)
        store.approve("w", "t", ttl_seconds=60)
        self.assertEqual(store.check("w", "t", "k"), ApprovalDecision.APPROVED)
        rec = store.get("w", "t")
        rec.expires_at = 1.0
        store._save(rec)
        self.assertEqual(store.check("w", "t", "k", "s"), ApprovalDecision.PENDING)

    def test_requires_store(self):
        with self.assertRaises(ValueError):
            make_manager([]).run_workflow(chain(), workflow_id="w")

    def test_key_fn_error_fails_closed(self):
        def boom(ctx):
            raise RuntimeError("x")
        log = []
        t = [WorkflowTask("t", "a", "t", approval=ApprovalSpec(key_fn=boom))]
        mgr = make_manager(log)
        mgr.run_workflow(t, workflow_id="w", approval_store=InMemoryApprovalStore())
        self.assertEqual(log, [])
        self.assertEqual(mgr.last_workflow_state.tasks["t"]["status"], TaskStatus.FAILED)

    def test_key_receives_dependency_data(self):
        seen = {}
        plan = WorkflowTask("plan", "a", "plan")
        plan.data = {"x": 1}
        # data is attached by the agent in real use; here key_fn just records the context shape
        launch = WorkflowTask("launch", "a", "launch", dependencies=["plan"],
                              approval=ApprovalSpec(key_fn=lambda ctx: seen.setdefault("ctx", sorted(ctx)) and "k"))
        make_manager([]).run_workflow([plan, launch], workflow_id="w", approval_store=InMemoryApprovalStore())
        self.assertIn("plan__result", seen["ctx"])

    def test_local_store_persists_across_instances(self):
        with tempfile.TemporaryDirectory() as d:
            s1 = LocalApprovalStore(d)
            s1.check("wf/1", "t", "k", "summary")
            s2 = LocalApprovalStore(d)
            self.assertEqual(s2.get("wf/1", "t").summary, "summary")
            s2.approve("wf/1", "t", key="k", by="marco")
            self.assertEqual(s1.check("wf/1", "t", "k"), ApprovalDecision.APPROVED)
            self.assertEqual(s1.list_pending(), [])
            self.assertTrue(any(f.endswith(".json") for f in os.listdir(d)))


class TestGateResult(unittest.TestCase):
    def test_reason_recorded_on_skip(self):
        log = []
        mgr = make_manager(log)
        gate = lambda ctx: GateResult(False, "no evidence for claim 3")  # noqa: E731
        mgr.run_workflow([WorkflowTask("t", "a", "t", conditions=[gate])], workflow_id="w")
        entry = mgr.last_workflow_state.tasks["t"]
        self.assertEqual(entry["status"], TaskStatus.SKIPPED)
        self.assertIn("no evidence for claim 3", entry["reason"])
        self.assertEqual(log, [])

    def test_bool_conditions_still_work(self):
        t = WorkflowTask("t", "a", "t", conditions=[lambda c: True, lambda c: False])
        self.assertFalse(t.should_execute({}))
        self.assertIn("returned False", t.gate_reason)
        self.assertTrue(WorkflowTask("u", "a", "u", conditions=[lambda c: GateResult(True)]).should_execute({}))

    def test_exception_reason(self):
        def bad(ctx):
            raise KeyError("k")
        t = WorkflowTask("t", "a", "t", conditions=[bad])
        self.assertFalse(t.should_execute({}))
        self.assertIn("raised KeyError", t.gate_reason)

    def test_reason_roundtrips_in_state(self):
        from maki.agents import WorkflowState
        st = WorkflowState("w")
        st.update_task_status("t", TaskStatus.AWAITING_APPROVAL, None, reason="why")
        back = WorkflowState.from_dict(st.to_dict())
        self.assertEqual(back.tasks["t"]["status"], TaskStatus.AWAITING_APPROVAL)
        self.assertEqual(back.tasks["t"]["reason"], "why")


if __name__ == "__main__":
    unittest.main()


class TestTaskOutput(unittest.TestCase):
    def test_data_flows_to_dependents(self):
        from maki.agents import TaskOutput

        seen = {}

        class Producer:
            def execute_task_with_retry(self, task, context=None, **kw):
                return TaskOutput("text", {"n": 3})

        class Consumer:
            def execute_task_with_retry(self, task, context=None, **kw):
                seen.update(context or {})
                return "ok"

        mgr = AgentManager(MagicMock())
        mgr.agents["p"], mgr.agents["c"] = Producer(), Consumer()
        res = mgr.run_workflow([WorkflowTask("one", "p", "t1"),
                                WorkflowTask("two", "c", "t2", dependencies=["one"])])
        self.assertEqual(res["one"]["result"], "text")
        self.assertEqual(seen["one"], {"n": 3})
