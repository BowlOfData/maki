"""Checkpoint resume."""
import json
import os
import sys
import tempfile
import unittest

from maki.agents.loop import Decision, LoopSpec, Requirement, StopReason, resume_loop, run_loop

from .loop_helpers import Clock, FakeAgent, verdict

AB = [Requirement("a", "first"), Requirement("b", "second")]
ONE_FAIL = verdict({"a": True, "b": False})
ALL_PASS = verdict({"a": True, "b": True})


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = os.path.join(self.tmp.name, "loop")

    def tearDown(self):
        self.tmp.cleanup()

    def spec(self, **kw):
        return LoopSpec("do the thing", AB, loop_dir=self.dir, loop_id="r1", **kw)

    def first_run(self, **kw):
        maker, judge = FakeAgent(["draft one\nNOTES: tried A"]), FakeAgent([ONE_FAIL], stateful=False)
        return run_loop(maker, judge, self.spec(**kw), clock=Clock())

    def test_resume_after_a_cap_continues_to_success(self):
        res = self.first_run(max_iterations=1)
        self.assertEqual(res.stop_reason, StopReason.MAX_ITERATIONS)
        maker, judge = FakeAgent(["draft two"]), FakeAgent([ALL_PASS], stateful=False)
        out = resume_loop(maker, judge, self.spec(max_iterations=5), clock=Clock())
        self.assertTrue(out.succeeded)
        self.assertEqual(out.iterations, 2)
        self.assertEqual(len(out.events), 1)  # only the new iteration is in this run's events

    def test_first_resumed_prompt_is_full_even_in_shared_mode(self):
        self.first_run(max_iterations=1)
        maker, judge = FakeAgent(["draft two"]), FakeAgent([ALL_PASS], stateful=False)
        resume_loop(maker, judge, self.spec(max_iterations=5), clock=Clock())
        p = maker.prompts[0]
        for needle in ("do the thing", "draft one", "b (second): fix it", "tried A"):
            self.assertIn(needle, p)

    def test_stall_counter_and_elapsed_survive_a_restart(self):
        clock = Clock()
        maker, judge = FakeAgent(["x"]), FakeAgent([ONE_FAIL], stateful=False)

        def hook(event):
            clock.advance(5)

        run_loop(maker, judge, self.spec(max_iterations=2, stall_limit=3, max_seconds=100), on_iteration=hook, clock=clock)
        state = json.load(open(os.path.join(self.dir, "state.json")))
        self.assertEqual(state["stall"]["streak"], 1)
        self.assertGreaterEqual(state["elapsed"], 10)
        out = resume_loop(FakeAgent(["x"]), FakeAgent([ONE_FAIL], stateful=False),
                          self.spec(max_iterations=10, stall_limit=3, max_seconds=100), clock=Clock())
        self.assertEqual(out.stop_reason, StopReason.STALLED)
        self.assertEqual(out.iterations, 4)  # streak 1 carried over, two more flat iterations

    def test_prior_elapsed_counts_against_the_wall_clock_cap(self):
        self.first_run(max_iterations=1)
        state_path = os.path.join(self.dir, "state.json")
        state = json.load(open(state_path))
        state["elapsed"] = 99
        json.dump(state, open(state_path, "w"))
        clock = Clock()
        out = resume_loop(FakeAgent(["x"]), FakeAgent([ONE_FAIL], stateful=False),
                          self.spec(max_iterations=9, stall_limit=9, max_seconds=100),
                          on_iteration=lambda e: clock.advance(2), clock=clock)
        self.assertEqual(out.stop_reason, StopReason.TIMEOUT)
        self.assertEqual(out.iterations, 2)

    def test_resume_after_agent_error(self):
        maker, judge = FakeAgent([RuntimeError("model down")]), FakeAgent([ALL_PASS], stateful=False)
        res = run_loop(maker, judge, self.spec(), clock=Clock())
        self.assertEqual(res.stop_reason, StopReason.ERROR)
        out = resume_loop(FakeAgent(["ok"]), FakeAgent([ALL_PASS], stateful=False), self.spec(), clock=Clock())
        self.assertTrue(out.succeeded)

    def test_human_scratchpad_edit_made_while_stopped_reaches_the_agent(self):
        self.first_run(max_iterations=1)
        path = os.path.join(self.dir, "scratchpad.md")
        text = open(path).read().replace("b (second): fix it", "b (second): HUMAN SAYS DO X")
        open(path, "w").write(text)
        maker = FakeAgent(["draft two"])
        resume_loop(maker, FakeAgent([ALL_PASS], stateful=False), self.spec(max_iterations=5), clock=Clock())
        self.assertIn("HUMAN SAYS DO X", maker.prompts[0])

    def test_refuses_without_checkpoint(self):
        with self.assertRaises(ValueError):
            resume_loop(FakeAgent(["x"]), FakeAgent(["u"], stateful=False), self.spec())

    def test_refuses_corrupt_checkpoint(self):
        self.first_run(max_iterations=1)
        open(os.path.join(self.dir, "state.json"), "w").write("{not json")
        with self.assertRaises(ValueError):
            resume_loop(FakeAgent(["x"]), FakeAgent(["u"], stateful=False), self.spec(max_iterations=3))

    def test_refuses_checkpoint_with_missing_fields(self):
        self.first_run(max_iterations=1)
        path = os.path.join(self.dir, "state.json")
        state = json.load(open(path))
        del state["stall"]
        json.dump(state, open(path, "w"))
        with self.assertRaises(ValueError):
            resume_loop(FakeAgent(["x"]), FakeAgent(["u"], stateful=False), self.spec(max_iterations=3))

    def test_refuses_mismatched_goal_or_requirements(self):
        self.first_run(max_iterations=1)
        other_goal = LoopSpec("another goal", AB, loop_dir=self.dir, loop_id="r1")
        other_reqs = LoopSpec("do the thing", [Requirement("z", "zz")], loop_dir=self.dir, loop_id="r1")
        for bad in (other_goal, other_reqs):
            with self.assertRaises(ValueError):
                resume_loop(FakeAgent(["x"]), FakeAgent(["u"], stateful=False), bad)

    def test_refuses_a_changed_command_or_text_with_the_same_ids(self):
        reqs = [Requirement("a", "first", command=[sys.executable, "-c", "raise SystemExit(1)"])]
        run_loop(FakeAgent(["x"]), FakeAgent(["u"], stateful=False),
                 LoopSpec("g", reqs, loop_dir=self.dir, loop_id="fp", max_iterations=1), clock=Clock())
        swapped = [Requirement("a", "first", command=[sys.executable, "-c", "pass"])]
        reworded = [Requirement("a", "first, but easier", command=reqs[0].command)]
        for changed in (swapped, reworded):
            with self.assertRaises(ValueError):
                resume_loop(FakeAgent(["x"]), FakeAgent(["u"], stateful=False),
                            LoopSpec("g", changed, loop_dir=self.dir, loop_id="fp", max_iterations=3))

    def test_a_consumed_reviewer_note_is_not_replayed_after_resume(self):
        def hook(event):
            return Decision.proceed("only once") if event is not None else None
        run_loop(FakeAgent(["x"]), FakeAgent([ONE_FAIL], stateful=False),
                 self.spec(max_iterations=2, stall_limit=9), on_iteration=hook, clock=Clock())
        self.assertEqual(json.load(open(os.path.join(self.dir, "state.json")))["note"], "")
        maker = FakeAgent(["x"])
        resume_loop(maker, FakeAgent([ALL_PASS], stateful=False), self.spec(max_iterations=5), clock=Clock())
        self.assertNotIn("only once", maker.prompts[0])

    def test_a_reviewer_note_survives_a_failed_agent_call(self):
        def hook(event):
            return Decision.proceed("do not lose me") if event is not None else None
        maker = FakeAgent(["draft one", RuntimeError("model down")])
        res = run_loop(maker, FakeAgent([ONE_FAIL], stateful=False), self.spec(max_iterations=5, stall_limit=9),
                       on_iteration=hook, clock=Clock())
        self.assertEqual(res.stop_reason, StopReason.ERROR)
        self.assertEqual(json.load(open(os.path.join(self.dir, "state.json")))["note"], "do not lose me")
        retry = FakeAgent(["draft two"])
        resume_loop(retry, FakeAgent([ALL_PASS], stateful=False), self.spec(max_iterations=5), clock=Clock())
        self.assertIn("do not lose me", retry.prompts[0])

    def test_refuses_a_loop_that_already_succeeded(self):
        run_loop(FakeAgent(["x"]), FakeAgent([ALL_PASS], stateful=False), self.spec(), clock=Clock())
        with self.assertRaises(ValueError):
            resume_loop(FakeAgent(["x"]), FakeAgent(["u"], stateful=False), self.spec())

    def test_needs_an_identity(self):
        with self.assertRaises(ValueError):
            resume_loop(FakeAgent(["x"]), FakeAgent(["u"], stateful=False), LoopSpec("g", AB))


if __name__ == "__main__":
    unittest.main()


class CrashRecoveryTests(unittest.TestCase):
    def test_process_killed_mid_iteration_resumes_from_the_last_checkpoint(self):
        import subprocess
        import sys
        script = (
            "import os, sys\n"
            "from maki.agents.loop import LoopSpec, Requirement, run_loop\n"
            "from maki.test.loop_helpers import FakeAgent, verdict\n"
            "class Dies(FakeAgent):\n"
            "    def execute_task_with_retry(self, task, context=None, use_plugins=False, **kw):\n"
            "        if len(self.prompts) == 1:\n"
            "            os._exit(9)\n"
            "        return super().execute_task_with_retry(task, context, use_plugins)\n"
            "reqs = [Requirement('a', 'first'), Requirement('b', 'second')]\n"
            "spec = LoopSpec('do the thing', reqs, loop_dir=sys.argv[1], loop_id='crash', stall_limit=9)\n"
            "run_loop(Dies(['draft one']), FakeAgent([verdict({'a': True, 'b': False})], stateful=False), spec)\n"
        )
        with tempfile.TemporaryDirectory() as d:
            loop_dir = os.path.join(d, "loop")
            proc = subprocess.run([sys.executable, "-c", script, loop_dir], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 9, proc.stderr)
            self.assertEqual(json.load(open(os.path.join(loop_dir, "state.json")))["iteration"], 1)
            spec = LoopSpec("do the thing", AB, loop_dir=loop_dir, loop_id="crash", stall_limit=9)
            maker = FakeAgent(["draft two"])
            out = resume_loop(maker, FakeAgent([ALL_PASS], stateful=False), spec, clock=Clock())
            self.assertTrue(out.succeeded)
            self.assertEqual(out.iterations, 2)
            self.assertIn("draft one", maker.prompts[0])
