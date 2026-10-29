"""Durable per-iteration approval through an ApprovalStore."""
import json
import os
import sys
import tempfile
import unittest

from maki.agents.approvals import InMemoryApprovalStore, LocalApprovalStore
from maki.agents.loop import LoopSpec, Requirement, StopReason, resume_loop, run_loop

from .loop_helpers import Clock, FakeAgent, verdict

AB = [Requirement("a", "first"), Requirement("b", "second")]
ONE_FAIL = verdict({"a": True, "b": False})
ALL_PASS = verdict({"a": True, "b": True})
TASK = "loop-gate"


class ApprovalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = os.path.join(self.tmp.name, "loop")
        self.store = InMemoryApprovalStore()

    def tearDown(self):
        self.tmp.cleanup()

    def spec(self, reqs=AB, **kw):
        return LoopSpec("do the thing", reqs, loop_dir=self.dir, loop_id="ap", **kw)

    def start(self, maker, judge, **kw):
        return run_loop(maker, judge, self.spec(**kw), approvals=self.store, clock=Clock())

    def cont(self, maker, judge, store=None, **kw):
        return resume_loop(maker, judge, self.spec(**kw), approvals=store or self.store, clock=Clock())

    def pending_key(self):
        return self.store.list_pending("ap")[0].key

    def approve(self, note=""):
        self.store.approve("ap", TASK, key=self.pending_key(), note=note)

    def reject(self, note=""):
        self.store.reject("ap", TASK, key=self.pending_key(), note=note)

    def test_suspends_before_the_first_iteration_without_calling_the_agent(self):
        maker = FakeAgent(["x"])
        res = self.start(maker, FakeAgent([ALL_PASS], stateful=False))
        self.assertTrue(res.suspended)
        self.assertEqual((res.iterations, maker.prompts), (0, []))
        self.assertEqual(json.load(open(os.path.join(self.dir, "state.json")))["gate"], 0)

    def test_start_approval_shows_commands_in_full(self):
        reqs = [Requirement("cmd", "tests", command=[sys.executable, "-m", "pytest", "-q"])]
        self.start(FakeAgent(["x"]), FakeAgent(["u"], stateful=False), reqs=reqs)
        self.assertIn("pytest -q", self.store.list_pending("ap")[0].summary)

    def test_approve_then_resume_runs_one_iteration_and_suspends_again(self):
        self.start(FakeAgent(["x"]), FakeAgent([ONE_FAIL], stateful=False))
        self.approve()
        maker = FakeAgent(["draft"])
        res = self.cont(maker, FakeAgent([ONE_FAIL], stateful=False), stall_limit=9)
        self.assertTrue(res.suspended)
        self.assertEqual((res.iterations, len(maker.prompts)), (1, 1))
        self.assertIn("1/2 passing", self.store.list_pending("ap")[0].summary)

    def test_old_approval_does_not_cover_the_next_gate(self):
        self.start(FakeAgent(["x"]), FakeAgent([ONE_FAIL], stateful=False))
        self.approve()
        self.cont(FakeAgent(["d"]), FakeAgent([ONE_FAIL], stateful=False), stall_limit=9)
        maker = FakeAgent(["d2"])
        res = self.cont(maker, FakeAgent([ONE_FAIL], stateful=False), stall_limit=9)
        self.assertTrue(res.suspended)
        self.assertEqual(maker.prompts, [])

    def test_resume_while_still_pending_does_nothing(self):
        self.start(FakeAgent(["x"]), FakeAgent([ALL_PASS], stateful=False))
        maker = FakeAgent(["x"])
        res = self.cont(maker, FakeAgent([ALL_PASS], stateful=False))
        self.assertTrue(res.suspended)
        self.assertEqual(maker.prompts, [])

    def test_full_run_to_success_over_several_gates(self):
        judge = FakeAgent([ONE_FAIL, ALL_PASS], stateful=False)  # one judge across restarts keeps the script
        maker = FakeAgent(["d1", "d2"])
        self.start(maker, judge)
        res = None
        for _ in range(4):
            self.approve()
            res = self.cont(maker, judge, stall_limit=9)
            if not res.suspended:
                break
        self.assertEqual(res.stop_reason, StopReason.SUCCEEDED)

    def test_reject_with_note_continues_with_the_note_as_feedback(self):
        self.start(FakeAgent(["x"]), FakeAgent([ONE_FAIL], stateful=False))
        self.reject("please use snake_case")
        maker = FakeAgent(["draft"])
        res = self.cont(maker, FakeAgent([ONE_FAIL], stateful=False), stall_limit=9)
        self.assertEqual(res.iterations, 1)
        self.assertIn("please use snake_case", maker.prompts[0])

    def test_reject_without_a_note_stops(self):
        self.start(FakeAgent(["x"]), FakeAgent([ONE_FAIL], stateful=False))
        self.reject()
        maker = FakeAgent(["draft"])
        res = self.cont(maker, FakeAgent([ONE_FAIL], stateful=False))
        self.assertEqual(res.stop_reason, StopReason.STOPPED)
        self.assertEqual(maker.prompts, [])

    def test_approve_with_a_note_passes_it_on(self):
        self.start(FakeAgent(["x"]), FakeAgent([ONE_FAIL], stateful=False))
        self.approve("watch the edge cases")
        maker = FakeAgent(["draft"])
        self.cont(maker, FakeAgent([ONE_FAIL], stateful=False), stall_limit=9)
        self.assertIn("watch the edge cases", maker.prompts[0])

    def test_pending_approval_survives_a_process_restart(self):
        base = os.path.join(self.tmp.name, "approvals")
        self.store = LocalApprovalStore(base)
        self.start(FakeAgent(["x"]), FakeAgent([ONE_FAIL], stateful=False))
        self.approve()
        reopened = LocalApprovalStore(base)
        maker = FakeAgent(["draft"])
        res = self.cont(maker, FakeAgent([ONE_FAIL], stateful=False), store=reopened, stall_limit=9)
        self.assertEqual((res.iterations, len(maker.prompts)), (1, 1))

    def test_last_iteration_asks_no_further_approval(self):
        self.start(FakeAgent(["x"]), FakeAgent([ALL_PASS], stateful=False), max_iterations=1)
        self.approve()
        res = self.cont(FakeAgent(["x"]), FakeAgent([ALL_PASS], stateful=False), max_iterations=1)
        self.assertTrue(res.succeeded)
        self.assertEqual(self.store.list_pending("ap"), [])

    def test_resume_without_the_approval_store_is_refused(self):
        self.start(FakeAgent(["x"]), FakeAgent([ONE_FAIL], stateful=False))
        self.approve()
        with self.assertRaises(ValueError):
            resume_loop(FakeAgent(["x"]), FakeAgent([ONE_FAIL], stateful=False), self.spec(), clock=Clock())

    def test_hook_none_is_not_repeated_on_resume(self):
        seen = []
        run_loop(FakeAgent(["x"]), FakeAgent([ONE_FAIL], stateful=False), self.spec(),
                 on_iteration=lambda e: seen.append(e), approvals=self.store, clock=Clock())
        self.approve()
        resume_loop(FakeAgent(["d"]), FakeAgent([ONE_FAIL], stateful=False), self.spec(max_iterations=1),
                    on_iteration=lambda e: seen.append(e), approvals=self.store, clock=Clock())
        self.assertEqual(seen[0], None)
        self.assertNotIn(None, seen[1:])


if __name__ == "__main__":
    unittest.main()
