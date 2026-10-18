"""Tests for verified workflow tasks: maker/checker revision loop, fail-closed, resume."""
import tempfile
import unittest
from unittest.mock import MagicMock

from maki.agents import (
    AgentManager, TaskOutput, TaskStatus, Verdict, VerifyResult, VerifySpec, WorkflowTask,
    agent_verifier,
)
from maki.distributed.state_store import LocalStateStore


class Maker:
    """Returns a scripted sequence of outputs and records contexts."""

    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []

    def execute_task_with_retry(self, task, context=None, max_retries=3, retry_delay=1.0):
        self.calls.append((task, dict(context or {})))
        out = self.outputs[min(len(self.calls) - 1, len(self.outputs) - 1)]
        if isinstance(out, Exception):
            raise out
        return out


class Echo:
    def __init__(self):
        self.calls = []

    def execute_task_with_retry(self, task, context=None, max_retries=3, retry_delay=1.0):
        self.calls.append((task, dict(context or {})))
        return f"done:{task}"


def manager(**agents):
    mgr = AgentManager(MagicMock())
    mgr.agents.update(agents)
    return mgr


def good_if_ok(ctx):
    if ctx["result"] == "ok":
        return VerifyResult(Verdict.ACCEPT, evidence="saw ok")
    return VerifyResult(Verdict.REVISE, evidence="saw bad", fixes="output ok")


class TestRevisionLoop(unittest.TestCase):
    def test_accept_first_try(self):
        maker = Maker(["ok"])
        mgr = manager(a=maker)
        res = mgr.run_workflow([WorkflowTask("t", "a", "do", verify=VerifySpec(good_if_ok))])
        self.assertEqual(res["t"]["result"], "ok")
        entry = mgr.last_workflow_state.tasks["t"]
        self.assertEqual(entry["status"], TaskStatus.COMPLETED)
        self.assertEqual(entry["verification"]["verdict"], "accept")
        self.assertEqual(entry["verification"]["revisions"], 0)
        self.assertEqual(len(maker.calls), 1)

    def test_revise_then_accept_feeds_back(self):
        maker = Maker(["bad", "ok"])
        mgr = manager(a=maker)
        wt = WorkflowTask("t", "a", "do", verify=VerifySpec(good_if_ok))
        res = mgr.run_workflow([wt])
        self.assertEqual(res["t"]["result"], "ok")
        self.assertEqual(maker.calls[0][1], {})
        self.assertEqual(maker.calls[1][1]["__verification_feedback"], "output ok")
        self.assertEqual(maker.calls[1][1]["__previous_result"], "bad")
        v = mgr.last_workflow_state.tasks["t"]["verification"]
        self.assertEqual(v["revisions"], 1)
        self.assertEqual([h["verdict"] for h in v["history"]], ["revise", "accept"])
        self.assertEqual(wt.attempts, 2)

    def test_exhausted_fails_and_skips_dependents(self):
        maker, echo = Maker(["bad"]), Echo()
        mgr = manager(a=maker, b=echo)
        wts = [
            WorkflowTask("t", "a", "do", verify=VerifySpec(good_if_ok, max_revisions=2)),
            WorkflowTask("down", "b", "next", dependencies=["t"]),
            WorkflowTask("down2", "b", "later", dependencies=["down"]),
            WorkflowTask("free", "b", "independent"),
        ]
        res = mgr.run_workflow(wts)
        st = mgr.last_workflow_state
        self.assertEqual(len(maker.calls), 3)  # 1 + 2 revisions
        self.assertEqual(st.tasks["t"]["status"], TaskStatus.FAILED)
        self.assertIn("verification failed after 2 revision(s)", st.tasks["t"]["reason"])
        self.assertEqual(st.tasks["down"]["status"], TaskStatus.SKIPPED)
        self.assertIn("failed verification", st.tasks["down"]["reason"])
        self.assertEqual(st.tasks["down2"]["status"], TaskStatus.SKIPPED)
        self.assertEqual(st.tasks["free"]["status"], TaskStatus.COMPLETED)
        self.assertEqual([t for t, _ in echo.calls], ["independent"])
        self.assertTrue(res["t"]["verification_failed"])
        self.assertEqual(st.status, "completed_with_skips")
        self.assertTrue(any(e["task"] == "t" for e in st.error_log))

    def test_block_fails_immediately(self):
        maker = Maker(["x"])
        mgr = manager(a=maker)
        spec = VerifySpec(lambda ctx: VerifyResult(Verdict.BLOCK, fixes="unsalvageable"))
        mgr.run_workflow([WorkflowTask("t", "a", "do", verify=spec)])
        self.assertEqual(len(maker.calls), 1)
        entry = mgr.last_workflow_state.tasks["t"]
        self.assertEqual(entry["status"], TaskStatus.FAILED)
        self.assertIn("unsalvageable", entry["reason"])
        self.assertEqual(entry["verification"]["verdict"], "block")

    def test_zero_revisions_verifies_once(self):
        maker = Maker(["bad"])
        mgr = manager(a=maker)
        mgr.run_workflow([WorkflowTask("t", "a", "do", verify=VerifySpec(good_if_ok, max_revisions=0))])
        self.assertEqual(len(maker.calls), 1)
        self.assertEqual(mgr.last_workflow_state.tasks["t"]["status"], TaskStatus.FAILED)

    def test_bool_verifier(self):
        mgr = manager(a=Maker(["ok"]))
        mgr.run_workflow([WorkflowTask("t", "a", "do", verify=VerifySpec(lambda ctx: True))])
        self.assertEqual(mgr.last_workflow_state.tasks["t"]["status"], TaskStatus.COMPLETED)
        mgr = manager(a=Maker(["ok"]))
        mgr.run_workflow([WorkflowTask("t", "a", "do", verify=VerifySpec(lambda ctx: False))])
        self.assertEqual(mgr.last_workflow_state.tasks["t"]["status"], TaskStatus.FAILED)

    def test_verifier_error_and_bad_return_fail_closed(self):
        def boom(ctx):
            raise RuntimeError("kaboom")

        for fn in (boom, lambda ctx: "accept", lambda ctx: None):
            mgr = manager(a=Maker(["ok"]))
            mgr.run_workflow([WorkflowTask("t", "a", "do", verify=VerifySpec(fn))])
            entry = mgr.last_workflow_state.tasks["t"]
            self.assertEqual(entry["status"], TaskStatus.FAILED)
            self.assertEqual(entry["verification"]["verdict"], "block")

    def test_verifier_receives_data_and_context(self):
        seen = {}

        def check(ctx):
            seen.update(ctx)
            return True

        maker = Maker([TaskOutput("text", {"n": 1})])
        mgr = manager(a=Echo(), b=maker)
        wts = [
            WorkflowTask("first", "a", "one"),
            WorkflowTask("t", "b", "two", dependencies=["first"], verify=VerifySpec(check)),
        ]
        res = mgr.run_workflow(wts)
        self.assertEqual(seen["result"], "text")
        self.assertEqual(seen["data"], {"n": 1})
        self.assertEqual(seen["context"]["first__result"], "done:one")
        self.assertEqual(seen["revision"], 0)
        self.assertIs(seen["manager"], mgr)
        self.assertEqual(res["t"]["data"], {"n": 1})

    def test_stale_data_not_carried_across_revisions(self):
        seen = []

        def check(ctx):
            seen.append(ctx["data"])
            return VerifyResult(Verdict.ACCEPT) if len(seen) == 2 else VerifyResult(Verdict.REVISE)

        mgr = manager(a=Maker([TaskOutput("t1", {"n": 1}), "t2"]))
        wt = WorkflowTask("t", "a", "do", verify=VerifySpec(check))
        res = mgr.run_workflow([wt])
        self.assertEqual(seen, [{"n": 1}, None])
        self.assertIsNone(res["t"]["data"])
        self.assertIsNone(wt.data)

    def test_verifier_cannot_mutate_dependency_context(self):
        def check(ctx):
            ctx["context"]["injected"] = True
            return VerifyResult(Verdict.REVISE) if ctx["revision"] == 0 else True

        maker = Maker(["x"])
        mgr = manager(a=Echo(), b=maker)
        mgr.run_workflow([WorkflowTask("first", "a", "one"),
                          WorkflowTask("t", "b", "two", dependencies=["first"],
                                       verify=VerifySpec(check))])
        self.assertNotIn("injected", maker.calls[1][1])

    def test_maker_exception_fails_task(self):
        mgr = manager(a=Maker([ValueError("bad input")]))
        mgr.run_workflow([WorkflowTask("t", "a", "do", verify=VerifySpec(lambda c: True))])
        self.assertEqual(mgr.last_workflow_state.tasks["t"]["status"], TaskStatus.FAILED)

    def test_no_verifier_unchanged(self):
        mgr = manager(a=Echo())
        wt = WorkflowTask("t", "a", "do")
        mgr.run_workflow([wt])
        entry = mgr.last_workflow_state.tasks["t"]
        self.assertNotIn("verification", entry)
        self.assertFalse(wt.to_dict()["has_verifier"])
        self.assertEqual(wt.attempts, 1)

    def test_to_dict_flags_verifier(self):
        wt = WorkflowTask("t", "a", "do", verify=VerifySpec(lambda c: True))
        self.assertTrue(wt.to_dict()["has_verifier"])


class TestAgentVerifierIntegration(unittest.TestCase):
    def test_agent_verifier_revises_then_accepts(self):
        maker = Maker(["bad", "ok"])
        verdicts = Maker([
            '{"verdict": "revise", "evidence": "bad", "fixes": "make it ok"}',
            '{"verdict": "accept", "evidence": "fine"}',
        ])
        mgr = manager(a=maker, v=verdicts)
        spec = VerifySpec(agent_verifier("v", "result must be ok"))
        res = mgr.run_workflow([WorkflowTask("t", "a", "do", verify=spec)])
        self.assertEqual(res["t"]["result"], "ok")
        self.assertEqual(maker.calls[1][1]["__verification_feedback"], "make it ok")
        # the verifier gets criteria + artifact in its prompt and no maker context at all
        self.assertEqual([c for _, c in verdicts.calls], [{}, {}])
        self.assertIn("result must be ok", verdicts.calls[1][0])

    def test_same_agent_as_verifier_rejected(self):
        mgr = manager(a=Maker(["ok"]))
        wt = WorkflowTask("t", "a", "do", verify=VerifySpec(agent_verifier("a", "c")))
        with self.assertRaises(ValueError):
            mgr.run_workflow([wt])

    def test_unregistered_verifier_agent_rejected_before_any_run(self):
        maker = Maker(["ok"])
        mgr = manager(a=maker)
        wt = WorkflowTask("t", "a", "do", verify=VerifySpec(agent_verifier("ghost", "c")))
        with self.assertRaises(ValueError):
            mgr.run_workflow([wt])
        self.assertEqual(maker.calls, [])

    def test_unparsable_agent_verdict_blocks(self):
        mgr = manager(a=Maker(["ok"]), v=Maker(["looks good to me"]))
        mgr.run_workflow([WorkflowTask("t", "a", "do", verify=VerifySpec(agent_verifier("v", "c")))])
        self.assertEqual(mgr.last_workflow_state.tasks["t"]["status"], TaskStatus.FAILED)


class TestDistributedManager(unittest.TestCase):
    def test_proxy_like_agent_can_verify(self):
        from maki.distributed.registry import DistributedAgentManager

        mgr = DistributedAgentManager(MagicMock())
        mgr.agents["a"] = Maker(["ok"])
        mgr.agents["remote_v"] = Maker(['{"verdict": "accept", "evidence": "remote saw ok"}'])
        mgr.run_workflow([WorkflowTask("t", "a", "do",
                                       verify=VerifySpec(agent_verifier("remote_v", "must be ok")))])
        entry = mgr.last_workflow_state.tasks["t"]
        self.assertEqual(entry["status"], TaskStatus.COMPLETED)
        self.assertEqual(entry["verification"]["evidence"], "remote saw ok")


class TestCheckpointResume(unittest.TestCase):
    def test_verdict_round_trips_and_failed_task_reruns(self):
        with tempfile.TemporaryDirectory() as d:
            ss = LocalStateStore(base_dir=d)
            wts = lambda: [  # noqa: E731
                WorkflowTask("t", "a", "do", verify=VerifySpec(good_if_ok, max_revisions=0)),
                WorkflowTask("down", "b", "next", dependencies=["t"]),
            ]
            mgr = manager(a=Maker(["bad"]), b=Echo())
            mgr.run_workflow(wts(), workflow_id="w", state_store=ss)
            saved = ss.load_workflow("w")
            self.assertEqual(saved.tasks["t"]["status"], TaskStatus.FAILED)
            self.assertEqual(saved.tasks["t"]["verification"]["verdict"], "revise")

            echo = Echo()
            mgr2 = manager(a=Maker(["ok"]), b=echo)
            mgr2.run_workflow(wts(), workflow_id="w", state_store=ss)
            st = mgr2.last_workflow_state
            self.assertEqual(st.tasks["t"]["status"], TaskStatus.COMPLETED)
            self.assertEqual(st.tasks["t"]["verification"]["verdict"], "accept")
            self.assertEqual(st.tasks["down"]["status"], TaskStatus.COMPLETED)
            self.assertEqual(st.status, "completed")
            self.assertEqual([t for t, _ in echo.calls], ["next"])

            # a third run skips the already-verified task entirely
            maker3 = Maker(["ok"])
            manager(a=maker3, b=Echo()).run_workflow(wts(), workflow_id="w", state_store=ss)
            self.assertEqual(maker3.calls, [])


if __name__ == "__main__":
    unittest.main()
