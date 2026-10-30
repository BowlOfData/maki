"""WorkflowTask(loop=...): a task completes only when its loop succeeds."""
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

from maki.agents import AgentManager, LoopTask, Requirement, TaskStatus, VerifySpec, WorkflowTask
from maki.distributed.state_store import LocalStateStore

from .loop_helpers import FakeAgent, verdict

ONE_FAIL = verdict({"r1": True, "r2": False})
ALL_PASS = verdict({"r1": True, "r2": True})
REQS = ["first", "second"]


class Echo:
    def __init__(self):
        self.calls = []

    def execute_task_with_retry(self, task, context=None, max_retries=3, retry_delay=1.0, **kw):
        self.calls.append((task, dict(context or {})))
        return f"done:{task}"


class LoopWorkflowBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def loop(self, **kw):
        kw.setdefault("loop_dir_base", self.tmp.name)
        kw.setdefault("context", "shared")
        return LoopTask(judge="judge", requirements=kw.pop("requirements", REQS), **kw)

    @staticmethod
    def manager(**agents):
        mgr = AgentManager(MagicMock())
        mgr.agents.update(agents)
        return mgr


class CompletionTests(LoopWorkflowBase):
    def test_succeeds_after_revisions_and_stores_the_artifact(self):
        maker, judge = FakeAgent(["v1", "v2"]), FakeAgent([ONE_FAIL, ALL_PASS], stateful=False)
        mgr = self.manager(worker=maker, judge=judge)
        wt = WorkflowTask("t", "worker", "write it", loop=self.loop())
        res = mgr.run_workflow([wt])
        self.assertEqual(res["t"]["result"], "v2")
        entry = mgr.last_workflow_state.tasks["t"]
        self.assertEqual(entry["status"], TaskStatus.COMPLETED)
        self.assertEqual(entry["verification"]["verdict"], "succeeded")
        self.assertEqual(entry["verification"]["revisions"], 1)
        self.assertEqual(wt.attempts, 2)
        self.assertTrue(wt.to_dict()["has_loop"])

    def test_failed_loop_fails_the_task_and_skips_dependents(self):
        maker, judge, echo = FakeAgent(["x"]), FakeAgent([ONE_FAIL], stateful=False), Echo()
        mgr = self.manager(worker=maker, judge=judge, b=echo)
        wts = [WorkflowTask("t", "worker", "do", loop=self.loop(max_iterations=2, stall_limit=9)),
               WorkflowTask("down", "b", "next", dependencies=["t"]),
               WorkflowTask("down2", "b", "later", dependencies=["down"]),
               WorkflowTask("free", "b", "independent")]
        res = mgr.run_workflow(wts)
        st = mgr.last_workflow_state
        self.assertEqual(st.tasks["t"]["status"], TaskStatus.FAILED)
        self.assertIn("loop max_iterations after 2 iteration(s)", st.tasks["t"]["reason"])
        self.assertEqual(st.tasks["down"]["status"], TaskStatus.SKIPPED)
        self.assertEqual(st.tasks["down2"]["status"], TaskStatus.SKIPPED)
        self.assertIn("did not complete its loop", st.tasks["down"]["reason"])
        self.assertEqual(st.tasks["free"]["status"], TaskStatus.COMPLETED)
        self.assertEqual([t for t, _ in echo.calls], ["independent"])
        self.assertTrue(res["t"]["loop_failed"] and res["t"]["verification_failed"])
        self.assertEqual(st.status, "completed_with_skips")

    def test_dependency_outputs_reach_the_maker_in_the_goal(self):
        echo, maker, judge = Echo(), FakeAgent(["v1"]), FakeAgent([ALL_PASS], stateful=False)
        mgr = self.manager(b=echo, worker=maker, judge=judge)
        mgr.run_workflow([WorkflowTask("first", "b", "gather"),
                          WorkflowTask("t", "worker", "write it", dependencies=["first"], loop=self.loop())])
        self.assertIn("write it", maker.prompts[0])
        self.assertIn("done:gather", maker.prompts[0])

    def test_command_requirements_work_in_a_workflow_loop(self):
        reqs = [Requirement("cmd", "exits zero", command=[sys.executable, "-c", "pass"])]
        mgr = self.manager(worker=FakeAgent(["x"]), judge=FakeAgent(["unused"], stateful=False))
        res = mgr.run_workflow([WorkflowTask("t", "worker", "do", loop=self.loop(requirements=reqs))])
        self.assertEqual(res["t"]["result"], "x")

    def test_loops_in_parallel_tasks_use_separate_directories(self):
        mgr = self.manager(w1=FakeAgent(["a"]), w2=FakeAgent(["b"]), judge=FakeAgent([ALL_PASS], stateful=False))
        res = mgr.run_workflow([WorkflowTask("t1", "w1", "do", parallelizable=True, loop=self.loop()),
                                WorkflowTask("t2", "w2", "do", parallelizable=True, loop=self.loop())])
        self.assertEqual((res["t1"]["result"], res["t2"]["result"]), ("a", "b"))


class ValidationTests(LoopWorkflowBase):
    def test_loop_and_verify_are_exclusive(self):
        with self.assertRaises(ValueError):
            WorkflowTask("t", "a", "do", loop=self.loop(), verify=VerifySpec(lambda c: True))

    def test_judge_must_be_registered_and_different(self):
        mgr = self.manager(worker=FakeAgent(["x"]))
        with self.assertRaises(ValueError):
            mgr.run_workflow([WorkflowTask("t", "worker", "do", loop=self.loop())])
        mgr = self.manager(judge=FakeAgent(["x"]))
        with self.assertRaises(ValueError):
            mgr.run_workflow([WorkflowTask("t", "judge", "do", loop=self.loop())])

    def test_remote_proxy_agents_are_refused(self):
        remote = FakeAgent(["x"])
        remote.endpoint = "http://elsewhere:8100"
        mgr = self.manager(worker=remote, judge=FakeAgent(["x"], stateful=False))
        with self.assertRaises(ValueError) as ctx:
            mgr.run_workflow([WorkflowTask("t", "worker", "do", loop=self.loop())])
        self.assertIn("remote proxy", str(ctx.exception))

    def test_loop_task_validates_its_settings(self):
        with self.assertRaises(ValueError):
            LoopTask(judge="", requirements=REQS)
        with self.assertRaises(ValueError):
            LoopTask(judge="j", requirements=[])
        with self.assertRaises(ValueError):
            LoopTask(judge="j", requirements=REQS, max_iterations=0)

    def test_stateless_maker_with_shared_context_is_refused_up_front(self):
        mgr = self.manager(worker=FakeAgent(["x"], stateful=False), judge=FakeAgent([ALL_PASS], stateful=False))
        with self.assertRaises(ValueError):
            mgr.run_workflow([WorkflowTask("t", "worker", "do", loop=self.loop(context="shared"))])

    def test_stateful_judge_is_refused_up_front(self):
        mgr = self.manager(worker=FakeAgent(["x"]), judge=FakeAgent([ALL_PASS], stateful=True))
        with self.assertRaises(ValueError):
            mgr.run_workflow([WorkflowTask("t", "worker", "do", loop=self.loop())])


class ResumeTests(LoopWorkflowBase):
    def test_a_resumed_workflow_resumes_its_loop_from_the_checkpoint(self):
        store = LocalStateStore(os.path.join(self.tmp.name, "wf"))
        mgr = self.manager(worker=FakeAgent(["v1"]), judge=FakeAgent([ONE_FAIL], stateful=False))
        mgr.run_workflow([WorkflowTask("t", "worker", "do", loop=self.loop(max_iterations=1))],
                         workflow_id="wf1", state_store=store)
        self.assertEqual(mgr.last_workflow_state.tasks["t"]["status"], TaskStatus.FAILED)
        maker2 = FakeAgent(["v2"])
        mgr2 = self.manager(worker=maker2, judge=FakeAgent([ALL_PASS], stateful=False))
        wt = WorkflowTask("t", "worker", "do", loop=self.loop(max_iterations=5))
        res = mgr2.run_workflow([wt], workflow_id="wf1", state_store=store)
        self.assertEqual(res["t"]["result"], "v2")
        self.assertIn("v1", maker2.prompts[0])  # the loop continued from iteration 1's artifact
        self.assertEqual(mgr2.last_workflow_state.tasks["t"]["verification"]["revisions"], 1)

    def test_changed_work_starts_a_fresh_loop_instead_of_resuming(self):
        store = LocalStateStore(os.path.join(self.tmp.name, "wf"))
        mgr = self.manager(worker=FakeAgent(["v1"]), judge=FakeAgent([ONE_FAIL], stateful=False))
        mgr.run_workflow([WorkflowTask("t", "worker", "do A", loop=self.loop(max_iterations=1))],
                         workflow_id="wf2", state_store=store)
        maker2 = FakeAgent(["fresh"])
        mgr2 = self.manager(worker=maker2, judge=FakeAgent([ALL_PASS], stateful=False))
        res = mgr2.run_workflow([WorkflowTask("t", "worker", "do B", loop=self.loop(max_iterations=5))],
                                workflow_id="wf2", state_store=store)
        self.assertEqual(res["t"]["result"], "fresh")
        self.assertNotIn("v1", maker2.prompts[0])


if __name__ == "__main__":
    unittest.main()
