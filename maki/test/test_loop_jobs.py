"""Background loop jobs for the server (no HTTP, scripted fake agents)."""
import os
import tempfile
import threading
import unittest

from maki.distributed.loop_jobs import (
    LoopCaps, LoopJobBusy, LoopJobNotFound, LoopJobs, NoPendingApproval, parse_requirements,
)

from .loop_helpers import FakeAgent, verdict

ONE_FAIL = verdict({"r1": True, "r2": False})
ALL_PASS = verdict({"r1": True, "r2": True})
REQS = ["first thing", "second thing"]


class GatedAgent(FakeAgent):
    """A maker that blocks until released, so tests can hold a loop mid-run."""

    def __init__(self, replies):
        super().__init__(replies)
        self.entered, self.release = threading.Event(), threading.Event()

    def execute_task_with_retry(self, task, context=None, use_plugins=False, **kw):
        self.entered.set()
        assert self.release.wait(10), "test never released the maker"
        return super().execute_task_with_retry(task, context, use_plugins)


class JobsTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.caps = LoopCaps(max_iterations=5, max_seconds=60, loop_dir_base=self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def jobs(self, maker, judge_replies):
        return LoopJobs(maker, FakeAgent(judge_replies, stateful=False), self.caps)

    @staticmethod
    def wait(job):
        job.thread.join(10)
        assert job.finished, "job did not finish"


class ParseTests(unittest.TestCase):
    def test_strings_and_objects(self):
        out = parse_requirements(["a thing", {"id": "x", "text": " other "}])
        self.assertEqual([(r.id, r.text, r.kind) for r in out], [("r1", "a thing", "agent"), ("x", "other", "agent")])

    def test_rejects_commands_checks_and_unknown_fields(self):
        for bad in ({"text": "x", "command": ["rm", "-rf", "/"]}, {"text": "x", "check": "lambda"},
                    {"text": "x", "cwd": "/"}, {"text": "x", "source": "approved"}):
            with self.assertRaises(ValueError, msg=str(bad)):
                parse_requirements([bad])

    def test_rejects_bad_shapes(self):
        for bad in ([], "text", [""], [3], [{"id": "a"}], ["a" * 2001], ["x"] * 31,
                    [{"id": "d", "text": "a"}, {"id": "d", "text": "b"}],
                    [{"id": "../x", "text": "a"}], [{"id": "a" * 65, "text": "a"}], [{"id": "a b", "text": "a"}]):
            with self.assertRaises(ValueError, msg=str(bad)):
                parse_requirements(bad)


class RunTests(JobsTestBase):
    def test_runs_to_success_with_ordered_events(self):
        jobs = self.jobs(FakeAgent(["a", "b"]), [ONE_FAIL, ALL_PASS])
        job = jobs.start("goal", REQS)
        self.wait(job)
        events, done = job.wait_events(0, timeout=0.1)
        self.assertTrue(done)
        self.assertEqual([e["type"] for e in events], ["started", "iteration", "iteration", "stopped"])
        self.assertEqual(events[-1]["stop_reason"], "succeeded")
        self.assertEqual(events[1]["passing"], 1)
        self.assertEqual(job.snapshot()["status"], "succeeded")
        self.assertTrue(job.id.startswith("loop-"))

    def test_events_can_be_replayed_from_an_index(self):
        jobs = self.jobs(FakeAgent(["a"]), [ALL_PASS])
        job = jobs.start("goal", REQS)
        self.wait(job)
        events, _ = job.wait_events(2, timeout=0.1)
        self.assertEqual([e["type"] for e in events], ["stopped"])

    def test_context_follows_the_agent_statefulness(self):
        maker = FakeAgent(["a"], stateful=False)
        jobs = self.jobs(maker, [ONE_FAIL, ALL_PASS])
        job = jobs.start("goal", REQS)
        self.wait(job)
        self.assertEqual(job.snapshot()["status"], "succeeded")  # shared would have raised ValueError

    def test_caps_are_enforced(self):
        jobs = self.jobs(FakeAgent(["a"]), [ALL_PASS])
        for kw in ({"max_iterations": 6}, {"max_iterations": 0}, {"max_seconds": 61}, {"max_seconds": -1}):
            with self.assertRaises(ValueError, msg=str(kw)):
                jobs.start("goal", REQS, **kw)

    def test_goal_must_be_text(self):
        with self.assertRaises(ValueError):
            self.jobs(FakeAgent(["a"]), [ALL_PASS]).start("  ", REQS)

    def test_shared_context_on_a_stateless_agent_is_refused_up_front(self):
        jobs = LoopJobs(FakeAgent(["a"], stateful=False), FakeAgent(["j"], stateful=False), self.caps)
        with self.assertRaises(ValueError):
            jobs.start("goal", REQS, context="shared")
        self.assertEqual(jobs.list(), [])

    def test_goal_size_and_seconds_are_bounded(self):
        jobs = self.jobs(FakeAgent(["a"]), [ALL_PASS])
        for kw in ({"goal": "g" * 20001}, {"max_seconds": float("inf")}, {"max_seconds": float("nan")}):
            with self.assertRaises(ValueError, msg=str(kw)):
                jobs.start(**{"goal": "goal", "requirements": REQS, **kw})

    def test_unexpected_error_hides_details(self):
        jobs = self.jobs(FakeAgent([ValueError("x")]), [ALL_PASS])
        jobs.judge = None  # judge_requirements will blow up with AttributeError
        job = jobs.start("goal", REQS)
        self.wait(job)
        snap = job.snapshot()
        self.assertEqual(snap["status"], "error")


class ConcurrencyTests(JobsTestBase):
    def test_second_loop_is_refused_while_one_runs(self):
        maker = GatedAgent(["a"])
        jobs = self.jobs(maker, [ALL_PASS])
        job = jobs.start("goal", REQS)
        self.assertTrue(maker.entered.wait(5))
        with self.assertRaises(LoopJobBusy):
            jobs.start("goal two", REQS)
        maker.release.set()
        self.wait(job)
        third = jobs.start("goal three", REQS)  # free again once the first finished
        self.wait(third)

    def test_cancel_stops_after_the_current_iteration(self):
        maker = GatedAgent(["a"])
        jobs = self.jobs(maker, [ONE_FAIL])
        job = jobs.start("goal", REQS, max_iterations=5)
        self.assertTrue(maker.entered.wait(5))
        self.assertTrue(jobs.cancel(job.id))
        maker.release.set()
        self.wait(job)
        self.assertEqual(job.snapshot()["stop_reason"], "stopped")
        self.assertEqual(job.snapshot()["iterations"], 1)

    def test_cancel_unknown_or_finished_is_false(self):
        jobs = self.jobs(FakeAgent(["a"]), [ALL_PASS])
        self.assertFalse(jobs.cancel("nope"))
        job = jobs.start("goal", REQS)
        self.wait(job)
        self.assertFalse(jobs.cancel(job.id))


class RestartTests(JobsTestBase):
    def test_a_new_server_sees_old_loops_from_their_checkpoints(self):
        first = self.jobs(FakeAgent(["a"]), [ONE_FAIL])
        job = first.start("goal", REQS, max_iterations=1)
        self.wait(job)
        second = self.jobs(FakeAgent(["a"]), [ALL_PASS])  # simulates a restarted server
        snap = second.snapshot(job.id)
        self.assertEqual((snap["status"], snap["iterations"], snap["resumable"]), ("max_iterations", 1, True))
        self.assertEqual([r["text"] for r in snap["requirements"]], REQS)
        self.assertIn(job.id, [s["id"] for s in second.list()])

    def test_resume_after_restart_finishes_the_loop(self):
        first = self.jobs(FakeAgent(["a"]), [ONE_FAIL])
        job = first.start("goal", REQS, max_iterations=1)
        self.wait(job)
        second = self.jobs(FakeAgent(["b"]), [ALL_PASS])
        resumed = second.resume(job.id)
        self.wait(resumed)
        self.assertEqual(resumed.snapshot()["status"], "succeeded")
        self.assertEqual(resumed.snapshot()["iterations"], 2)

    def test_resume_refusals(self):
        jobs = self.jobs(FakeAgent(["a"]), [ALL_PASS])
        with self.assertRaises(LoopJobNotFound):
            jobs.resume("loop-missing")
        done = jobs.start("goal", REQS)
        self.wait(done)
        with self.assertRaises(ValueError):
            jobs.resume(done.id)  # already succeeded

    def test_approval_gate_suspends_until_answered_then_resume_continues(self):
        jobs = self.jobs(FakeAgent(["a"]), [ONE_FAIL, ALL_PASS])
        job = jobs.start("goal", REQS, require_approval=True)
        self.wait(job)
        snap = jobs.snapshot(job.id)
        self.assertEqual((snap["status"], snap["iterations"], snap["resumable"]), ("suspended", 0, True))
        self.assertIn("Start loop", snap["pending_approval"]["summary"])
        answered = jobs.approve(job.id, True, "go carefully")
        self.assertIsNone(answered["pending_approval"])
        second = jobs.resume(job.id)
        self.wait(second)
        snap = jobs.snapshot(job.id)
        self.assertEqual((snap["status"], snap["iterations"]), ("suspended", 1))  # gate before iteration 2
        self.assertIn("passing", snap["pending_approval"]["summary"])
        jobs.approve(job.id, True)
        third = jobs.resume(job.id)
        self.wait(third)
        self.assertEqual(jobs.snapshot(job.id)["status"], "succeeded")

    def test_rejection_without_a_note_stops_and_with_a_note_continues(self):
        jobs = self.jobs(FakeAgent(["a"]), [ALL_PASS])
        job = jobs.start("goal", REQS, require_approval=True)
        self.wait(job)
        jobs.approve(job.id, False)
        stopped = jobs.resume(job.id)
        self.wait(stopped)
        self.assertEqual(jobs.snapshot(job.id)["status"], "stopped")

        other = self.jobs(FakeAgent(["a"]), [ALL_PASS])
        job2 = other.start("goal", REQS, require_approval=True)
        self.wait(job2)
        other.approve(job2.id, False, "try the other way")
        resumed = other.resume(job2.id)
        self.wait(resumed)
        self.assertEqual(other.snapshot(job2.id)["status"], "succeeded")

    def test_approve_refusals(self):
        jobs = self.jobs(FakeAgent(["a"]), [ALL_PASS])
        with self.assertRaises(LoopJobNotFound):
            jobs.approve("loop-aaaaaaaaaaaa", True)
        job = jobs.start("goal", REQS)  # no gate
        self.wait(job)
        with self.assertRaises(NoPendingApproval):
            jobs.approve(job.id, True)
        gated = jobs.start("goal", REQS, require_approval=True)
        self.wait(gated)
        for bad in ("x" * 2001, 5):
            with self.assertRaises(ValueError):
                jobs.approve(gated.id, True, bad)
        jobs.approve(gated.id, True)
        with self.assertRaises(NoPendingApproval):
            jobs.approve(gated.id, True)  # already answered

    def test_pending_approval_survives_a_restart(self):
        first = self.jobs(FakeAgent(["a"]), [ALL_PASS])
        job = first.start("goal", REQS, require_approval=True)
        self.wait(job)
        second = self.jobs(FakeAgent(["a"]), [ALL_PASS])
        self.assertIsNotNone(second.snapshot(job.id)["pending_approval"])
        second.approve(job.id, True)
        self.wait(second.resume(job.id))
        self.assertEqual(second.snapshot(job.id)["status"], "succeeded")

    def test_list_ignores_directories_that_are_not_server_loops(self):
        jobs = self.jobs(FakeAgent(["a"]), [ALL_PASS])
        job = jobs.start("goal", REQS)
        self.wait(job)
        os.makedirs(os.path.join(self.tmp.name, ".approvals"), exist_ok=True)
        local = os.path.join(self.tmp.name, "my-local-loop")  # e.g. written by a local run_loop
        os.makedirs(local)
        with open(os.path.join(local, "state.json"), "w") as f:
            f.write('{"loop_id": "my-local-loop", "status": "stalled"}')
        fresh = self.jobs(FakeAgent(["a"]), [ALL_PASS])
        self.assertEqual([s["id"] for s in fresh.list()], [job.id])

    def test_resume_reuses_the_saved_context_and_plugin_flag(self):
        first = self.jobs(FakeAgent(["a"], stateful=True), [ONE_FAIL])
        job = first.start("goal", REQS, max_iterations=1, context="fresh", use_plugins=True)
        self.wait(job)
        second = self.jobs(FakeAgent(["b"], stateful=True), [ALL_PASS])  # stateful: would default to shared
        resumed = second.resume(job.id)
        self.wait(resumed)
        self.assertEqual(resumed.snapshot()["status"], "succeeded")
        with open(os.path.join(self.tmp.name, job.id, "state.json")) as f:
            state = __import__("json").load(f)
        self.assertEqual((state["context"], state["use_plugins"]), ("fresh", True))

    def test_old_server_loops_are_pruned_but_running_and_foreign_ones_are_kept(self):
        import json
        import time
        jobs = self.jobs(FakeAgent(["a"]), [ONE_FAIL])
        old = jobs.start("goal", REQS, max_iterations=1)
        self.wait(old)
        recent = jobs.start("goal", REQS, max_iterations=1)
        self.wait(recent)
        long_ago = time.time() - 40 * 86400
        os.utime(os.path.join(self.tmp.name, old.id, "state.json"), (long_ago, long_ago))
        running = os.path.join(self.tmp.name, "loop-aaaaaaaaaaaa")
        os.makedirs(running)
        with open(os.path.join(running, "state.json"), "w") as f:
            json.dump({"loop_id": "loop-aaaaaaaaaaaa", "status": "running"}, f)
        os.utime(os.path.join(running, "state.json"), (long_ago, long_ago))
        foreign = os.path.join(self.tmp.name, "my-local-loop")
        os.makedirs(foreign)
        os.utime(foreign, (long_ago, long_ago))
        self.assertEqual(jobs.prune(), 1)
        left = set(os.listdir(self.tmp.name))
        self.assertNotIn(old.id, left)
        self.assertTrue({recent.id, "loop-aaaaaaaaaaaa", "my-local-loop"} <= left)

    def test_retention_none_keeps_everything(self):
        self.caps.retention_days = None
        jobs = self.jobs(FakeAgent(["a"]), [ALL_PASS])
        self.assertEqual(jobs.prune(), 0)

    def test_token_budget_is_capped_and_reported(self):
        self.caps.max_tokens = 300
        maker = FakeAgent(["a"], tokens_per_call=100)
        jobs = LoopJobs(maker, FakeAgent([ONE_FAIL], stateful=False, tokens_per_call=20), self.caps)
        for bad in (301, 0, True):
            with self.assertRaises(ValueError, msg=str(bad)):
                jobs.start("goal", REQS, max_tokens=bad)
        job = jobs.start("goal", REQS, stall_limit=5)  # no request value: the cap is the budget
        self.wait(job)
        snap = job.snapshot()
        self.assertEqual((snap["status"], snap["tokens"], snap["iterations"]), ("token_budget", 360, 3))
        self.assertEqual([e["tokens"] for e in job._events if e["type"] == "iteration"], [120, 120, 120])
        fresh = LoopJobs(maker, FakeAgent([ONE_FAIL], stateful=False), self.caps)
        self.assertEqual(fresh.snapshot(job.id)["tokens"], 360)

    def test_resume_keeps_the_saved_budget_and_clamps_it_to_the_cap(self):
        self.caps.max_tokens = 1000
        first = self.jobs(FakeAgent(["a"], tokens_per_call=100), [ONE_FAIL])
        job = first.start("goal", REQS, max_iterations=1, max_tokens=500)
        self.wait(job)
        with open(os.path.join(self.tmp.name, job.id, "state.json")) as f:
            self.assertEqual(__import__("json").load(f)["max_tokens"], 500)
        second = self.jobs(FakeAgent(["a"], tokens_per_call=100), [ONE_FAIL])
        self.caps.max_tokens = 300  # the operator lowered the cap since
        resumed = second.resume(job.id, max_iterations=5)
        self.wait(resumed)
        self.assertEqual(resumed.snapshot()["status"], "token_budget")  # 120 already spent; budget 300 is hit at 360

    def test_a_running_resumed_job_reports_the_total_tokens(self):
        first = self.jobs(FakeAgent(["a"], tokens_per_call=100), [ONE_FAIL])
        job = first.start("goal", REQS, max_iterations=1)
        self.wait(job)
        maker = GatedAgent(["b"])
        second = LoopJobs(maker, FakeAgent([ALL_PASS], stateful=False), self.caps)
        resumed = second.resume(job.id)
        self.assertEqual(resumed.snapshot()["tokens"], 100)  # carried over before any new iteration
        maker.release.set()
        self.wait(resumed)

    def test_pruning_also_forgets_the_loops_approval_record(self):
        import time
        jobs = self.jobs(FakeAgent(["a"]), [ALL_PASS])
        job = jobs.start("goal", REQS, require_approval=True)
        self.wait(job)
        approvals = os.path.join(self.tmp.name, ".approvals")
        self.assertTrue(os.listdir(approvals))
        long_ago = time.time() - 40 * 86400
        os.utime(os.path.join(self.tmp.name, job.id, "state.json"), (long_ago, long_ago))
        self.assertEqual(jobs.prune(), 1)
        self.assertEqual(os.listdir(approvals), [])

    def test_unknown_snapshot(self):
        with self.assertRaises(LoopJobNotFound):
            self.jobs(FakeAgent(["a"]), [ALL_PASS]).snapshot("loop-missing")

    def test_finished_in_memory_job_reports_resumable(self):
        jobs = self.jobs(FakeAgent(["a"]), [ONE_FAIL])
        job = jobs.start("goal", REQS, max_iterations=1)
        self.wait(job)
        self.assertTrue(jobs.snapshot(job.id)["resumable"])

    def test_loops_live_inside_the_configured_base(self):
        jobs = self.jobs(FakeAgent(["a"]), [ALL_PASS])
        job = jobs.start("goal", REQS)
        self.wait(job)
        self.assertTrue(os.path.exists(os.path.join(self.tmp.name, job.id, "state.json")))


if __name__ == "__main__":
    unittest.main()
