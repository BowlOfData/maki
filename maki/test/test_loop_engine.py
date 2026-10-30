"""The loop engine: stopping rules, context modes, hook, scratchpad flow, checkpoints."""
import os
import sys
import tempfile
import unittest

from maki.agents.loop import Decision, LoopSpec, Requirement, StopReason, run_loop
from maki.agents.loop_scratchpad import LoopStore

from .loop_helpers import Clock, FakeAgent, verdict

AB = [Requirement("a", "first"), Requirement("b", "second")]


class EngineBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def spec(self, reqs=AB, **kw):
        kw.setdefault("loop_dir", os.path.join(self.tmp.name, "loop"))
        kw.setdefault("loop_id", "t1")
        return LoopSpec("do the thing", reqs, **kw)

    def run_it(self, maker_replies, judge_replies, reqs=AB, hook=None, maker_kw=None, **spec_kw):
        maker = FakeAgent(maker_replies, **(maker_kw or {}))
        judge = FakeAgent(judge_replies, stateful=False)
        overwrite = spec_kw.pop("overwrite", False)
        result = run_loop(maker, judge, self.spec(reqs, **spec_kw), on_iteration=hook,
                          clock=spec_kw.pop("clock", None) or Clock(), overwrite=overwrite)
        return result, maker, judge


class StoppingTests(EngineBase):
    def test_succeeds_after_revisions(self):
        res, maker, judge = self.run_it(
            ["v1", "v2"], [verdict({"a": True, "b": False}), verdict({"a": True, "b": True})])
        self.assertEqual((res.stop_reason, res.iterations), (StopReason.SUCCEEDED, 2))
        self.assertTrue(res.succeeded)
        self.assertEqual(res.artifact, "v2")
        self.assertIn("b (second): fix it", maker.prompts[1])

    def test_max_iterations(self):
        res, *_ = self.run_it(["x"], [verdict({"a": True, "b": False}), verdict({"a": False, "b": True})],
                              max_iterations=2, stall_limit=9)
        self.assertEqual((res.stop_reason, res.iterations), (StopReason.MAX_ITERATIONS, 2))

    def test_stall_stops_after_limit(self):
        res, *_ = self.run_it(["x"], [verdict({"a": True, "b": False})], max_iterations=10, stall_limit=3)
        self.assertEqual(res.stop_reason, StopReason.STALLED)
        self.assertEqual(res.iterations, 4)  # iteration 1 improves from 0, then 3 flat ones

    def test_recovery_after_regression_resets_stall(self):
        seq = [verdict({"a": True, "b": True, "c": False}), verdict({"a": True, "b": False, "c": False}),
               verdict({"a": True, "b": True, "c": False}), verdict({"a": True, "b": True, "c": True})]
        reqs = AB + [Requirement("c", "third")]
        res, *_ = self.run_it(["x"], seq, reqs=reqs, stall_limit=2)
        self.assertEqual(res.stop_reason, StopReason.SUCCEEDED)
        self.assertEqual(res.events[1].regressions, ["b"])

    def test_judge_error_iteration_does_not_count_toward_stall(self):
        judge = [verdict({"a": True, "b": False}), "junk", "junk", "junk", "junk",
                 verdict({"a": True, "b": True})]
        res, *_ = self.run_it(["x"], judge, stall_limit=2, max_iterations=5)
        self.assertEqual(res.stop_reason, StopReason.SUCCEEDED)
        self.assertEqual(res.iterations, 4)
        self.assertTrue(res.events[1].judge_error)
        self.assertIsNone(res.events[1].passing)

    def test_block_stops(self):
        res, *_ = self.run_it(["x"], [verdict({"a": False, "b": False}, block="goal is impossible")])
        self.assertEqual(res.stop_reason, StopReason.BLOCKED)
        self.assertEqual(res.detail, "goal is impossible")

    def test_wall_clock_cap_checked_between_iterations(self):
        clock = Clock()
        maker, judge = FakeAgent(["x"]), FakeAgent([verdict({"a": True, "b": False})], stateful=False)

        def hook(event):
            clock.advance(10)

        res = run_loop(maker, judge, self.spec(max_seconds=15, stall_limit=9), on_iteration=hook, clock=clock)
        self.assertEqual(res.stop_reason, StopReason.TIMEOUT)
        self.assertEqual(res.iterations, 1)  # the cap counts from loop start, hook time included

    def test_agent_error_stops_cleanly_with_checkpoint(self):
        res, *_ = self.run_it([RuntimeError("model down")], [verdict({"a": True, "b": True})])
        self.assertEqual(res.stop_reason, StopReason.ERROR)
        self.assertIn("model down", res.error)
        self.assertEqual(LoopStore(os.path.join(self.tmp.name, "loop")).load()["status"], "error")


class NotesAndBlockTests(EngineBase):
    def test_a_notes_line_inside_the_work_stays_in_the_artifact(self):
        doc = "Intro\nNOTES: this heading is part of the document\n\nand a second paragraph of the document."
        result, _, _ = self.run_it([doc], [verdict({"a": True, "b": True})])
        self.assertEqual(result.artifact, doc)
        long_tail = "work\nNOTES: " + "n" * 2000
        result, _, _ = self.run_it([long_tail], [verdict({"a": True, "b": True})], overwrite=True)
        self.assertEqual(result.artifact, long_tail)

    def test_a_short_trailing_notes_block_is_still_split_off(self):
        result, _, _ = self.run_it(["the work\nNOTES: tried A"], [verdict({"a": True, "b": True})])
        self.assertEqual(result.artifact, "the work")

    def test_success_beats_a_judge_block(self):
        result, _, _ = self.run_it(["x"], [verdict({"a": True, "b": True}, ) .replace('"block": ""', '"block": "odd"')])
        self.assertEqual(result.stop_reason, StopReason.SUCCEEDED)

    def test_a_block_still_stops_a_failing_loop(self):
        result, _, _ = self.run_it(["x"], [verdict({"a": True, "b": False}).replace('"block": ""', '"block": "impossible"')])
        self.assertEqual((result.stop_reason, result.detail), (StopReason.BLOCKED, "impossible"))


class ArtifactCapTests(EngineBase):
    BIG = "START" + "x" * 5000 + "END"

    def test_judge_and_next_prompt_see_a_clipped_artifact_but_the_result_keeps_the_full_one(self):
        result, maker, judge = self.run_it([self.BIG], [verdict({"a": True, "b": False})], max_artifact_chars=400,
                                           max_iterations=2, stall_limit=5, context="fresh",
                                           maker_kw={"stateful": False})
        self.assertEqual(result.artifact, self.BIG)
        self.assertLess(len(judge.prompts[0]), 2000)
        self.assertIn("characters omitted", judge.prompts[0])
        self.assertIn("START", judge.prompts[0])
        self.assertIn("END", judge.prompts[0])
        self.assertIn("characters omitted", maker.prompts[1])
        self.assertNotIn("x" * 1000, maker.prompts[1])

    def test_callable_checks_get_the_full_artifact(self):
        seen = []
        req = Requirement("c", "full text", check=lambda t: seen.append(len(t)) or True)
        run_loop(FakeAgent([self.BIG]), FakeAgent(["{}"], stateful=False),
                 self.spec([req], max_artifact_chars=400), clock=Clock())
        self.assertEqual(seen, [len(self.BIG)])

    def test_none_disables_the_cap(self):
        result, _, judge = self.run_it([self.BIG], [verdict({"a": True, "b": True})], max_artifact_chars=None)
        self.assertNotIn("omitted", judge.prompts[0])


class OverwriteTests(EngineBase):
    def test_reusing_the_directory_of_an_unfinished_loop_is_refused_unless_overwritten(self):
        first, _, _ = self.run_it(["a"], [verdict({"a": True, "b": False})], max_iterations=1)
        self.assertEqual(first.stop_reason, StopReason.MAX_ITERATIONS)
        with self.assertRaises(ValueError) as cm:
            self.run_it(["a"], [verdict({"a": True, "b": True})])
        self.assertIn("resume_loop", str(cm.exception))
        again, _, _ = self.run_it(["a"], [verdict({"a": True, "b": True})], overwrite=True)
        self.assertEqual(again.stop_reason, StopReason.SUCCEEDED)

    def test_a_finished_loop_can_be_rerun(self):
        self.run_it(["a"], [verdict({"a": True, "b": True})])
        again, _, _ = self.run_it(["a"], [verdict({"a": True, "b": True})])
        self.assertEqual(again.stop_reason, StopReason.SUCCEEDED)


class UnexpectedFailureTests(EngineBase):
    def test_an_exception_while_checking_ends_the_loop_with_error_and_a_checkpoint(self):
        from unittest.mock import patch
        with patch("maki.agents.loop._check_requirements", side_effect=RuntimeError("check blew up")):
            result, _, _ = self.run_it(["draft"], [verdict({"a": True, "b": True})])
        self.assertEqual(result.stop_reason, StopReason.ERROR)
        self.assertIn("check blew up", result.error)
        state = LoopStore(os.path.join(self.tmp.name, "loop")).load()
        self.assertEqual(state["status"], "error")  # resumable, not left as "running"


class CheckTests(EngineBase):
    def test_command_and_callable_requirements_skip_the_judge(self):
        reqs = [Requirement("cmd", "exits zero", command=[sys.executable, "-c", "pass"]),
                Requirement("fn", "has ok", check=lambda t: "ok" in t)]
        res, _, judge = self.run_it(["all ok"], [RuntimeError("judge must not be called")], reqs=reqs)
        self.assertTrue(res.succeeded)
        self.assertEqual(judge.prompts, [])

    def test_failing_command_output_reaches_the_next_prompt(self):
        reqs = [Requirement("cmd", "tests", command=[sys.executable, "-c", "print('E: bad'); raise SystemExit(1)"])]
        res, maker, _ = self.run_it(["x"], ["unused"], reqs=reqs, max_iterations=2, stall_limit=9)
        self.assertEqual(res.stop_reason, StopReason.MAX_ITERATIONS)
        self.assertIn("E: bad", maker.prompts[1])

    def test_command_evidence_reaches_the_judge(self):
        reqs = [Requirement("cmd", "tests", command=[sys.executable, "-c", "print('EVIDENCE-XYZ')"]),
                Requirement("a", "style")]
        _, _, judge = self.run_it(["x"], [verdict({"a": True})], reqs=reqs)
        self.assertIn("EVIDENCE-XYZ", judge.prompts[0])

    def test_callable_exception_fails_closed(self):
        def boom(text):
            raise RuntimeError("bad check")
        res, *_ = self.run_it(["x"], ["u"], reqs=[Requirement("fn", "x", check=boom)], max_iterations=1)
        self.assertFalse(res.final_results["fn"].passed)
        self.assertIn("bad check", res.final_results["fn"].feedback)

    def test_artifact_fn_is_shown_to_the_judge(self):
        _, _, judge = self.run_it(["reply"], [verdict({"a": True, "b": True})], artifact_fn=lambda: "DIFF-123")
        self.assertIn("DIFF-123", judge.prompts[0])
        self.assertIn("reply", judge.prompts[0])


class JudgeErrorFeedbackTests(EngineBase):
    def test_a_judge_failure_is_never_shown_to_the_maker_as_feedback(self):
        _, maker, _ = self.run_it(["x"], ["junk", "junk", verdict({"a": True, "b": True})], stall_limit=9)
        self.assertNotIn("no usable verdict", maker.prompts[1])

    def test_last_real_feedback_is_kept_through_a_judge_error(self):
        judge = [verdict({"a": True, "b": False}, b="fix b like THIS"), "junk", "junk", verdict({"a": True, "b": True})]
        _, maker, _ = self.run_it(["x"], judge, stall_limit=9)
        self.assertIn("fix b like THIS", maker.prompts[2])
        self.assertNotIn("no usable verdict", maker.prompts[2])


class ArtifactTests(EngineBase):
    def test_artifact_fn_output_is_the_tracked_artifact(self):
        seen = []
        reqs = [Requirement("fn", "sees raw artifact", check=lambda t: seen.append(t) or t == "FILE-V1")]
        res, *_ = self.run_it(["I fixed it"], ["u"], reqs=reqs, artifact_fn=lambda: "FILE-V1")
        self.assertEqual(seen, ["FILE-V1"])
        self.assertEqual(res.artifact, "FILE-V1")
        self.assertEqual(res.events[0].reply, "I fixed it")

    def test_fresh_prompt_carries_artifact_fn_output_not_the_reply(self):
        files = iter(["FILE-V1", "FILE-V2", "FILE-V3"])
        _, maker, _ = self.run_it(["I fixed it"], [verdict({"a": True, "b": False}), verdict({"a": True, "b": True})],
                                  context="fresh", maker_kw={"stateful": False}, artifact_fn=lambda: next(files))
        self.assertIn("FILE-V1", maker.prompts[1])
        self.assertNotIn("Current artifact:\nI fixed it", maker.prompts[1])


class ContextTests(EngineBase):
    def test_shared_requires_stateful_agent(self):
        with self.assertRaises(ValueError):
            self.run_it(["x"], [verdict({"a": True, "b": True})], maker_kw={"stateful": False})

    def test_fresh_resets_and_carries_artifact(self):
        res, maker, _ = self.run_it(
            ["draft one", "draft two"], [verdict({"a": True, "b": False}), verdict({"a": True, "b": True})],
            context="fresh", maker_kw={"stateful": False})
        self.assertTrue(res.succeeded)
        self.assertEqual(maker.resets, 2)
        self.assertIn("draft one", maker.prompts[1])
        self.assertIn("do the thing", maker.prompts[1])

    def test_shared_prompt_is_feedback_only(self):
        _, maker, _ = self.run_it(["draft one", "d2"], [verdict({"a": True, "b": False}), verdict({"a": True, "b": True})])
        self.assertNotIn("draft one", maker.prompts[1])
        self.assertEqual(maker.resets, 0)


class HookAndScratchpadTests(EngineBase):
    def test_hook_called_before_start_and_after_each_iteration(self):
        seen = []
        self.run_it(["x"], [verdict({"a": True, "b": False}), verdict({"a": True, "b": True})],
                    hook=lambda e: seen.append(None if e is None else e.iteration))
        self.assertEqual(seen, [None, 1])

    def test_stop_before_first_iteration(self):
        res, maker, _ = self.run_it(["x"], ["u"], hook=lambda e: Decision.STOP)
        self.assertEqual((res.stop_reason, res.iterations), (StopReason.STOPPED, 0))
        self.assertEqual(maker.prompts, [])

    def test_stop_after_an_iteration(self):
        res, *_ = self.run_it(["x"], [verdict({"a": True, "b": False})],
                              hook=lambda e: Decision.STOP if e else None, stall_limit=9)
        self.assertEqual((res.stop_reason, res.iterations), (StopReason.STOPPED, 1))

    def test_reviewer_note_reaches_the_next_prompt(self):
        def hook(e):
            return Decision.proceed("use snake_case") if e else None
        _, maker, _ = self.run_it(["x"], [verdict({"a": True, "b": False}), verdict({"a": True, "b": True})],
                                  hook=hook)
        self.assertIn("use snake_case", maker.prompts[1])

    def test_confirm_requirements_false_cancels(self):
        maker, judge = FakeAgent(["x"]), FakeAgent(["u"], stateful=False)
        res = run_loop(maker, judge, self.spec(), confirm_requirements=lambda reqs: False)
        self.assertEqual(res.stop_reason, StopReason.STOPPED)
        self.assertEqual(maker.prompts, [])

    def test_agent_notes_are_split_off_and_kept(self):
        res, maker, _ = self.run_it(["the result\nNOTES: try approach B next"],
                                    [verdict({"a": True, "b": False}), verdict({"a": True, "b": True})])
        self.assertEqual(res.events[0].reply, "the result")
        with open(res.scratchpad_path) as f:
            self.assertIn("try approach B next", f.read())

    def test_human_feedback_edit_reaches_the_agent(self):
        def hook(e):
            if e is not None:
                with open(e_path[0]) as f:
                    text = f.read()
                with open(e_path[0], "w") as f:
                    f.write(text.replace("- b (second): fix it", "- b (second): HUMAN SAYS DO X"))
        e_path = [os.path.join(self.tmp.name, "loop", "scratchpad.md")]
        _, maker, _ = self.run_it(["x"], [verdict({"a": True, "b": False}), verdict({"a": True, "b": True})], hook=hook)
        self.assertIn("HUMAN SAYS DO X", maker.prompts[1])

    def test_engine_log_tampering_is_reported_and_reverted(self):
        path = os.path.join(self.tmp.name, "loop", "scratchpad.md")

        def hook(e):
            if e is not None:
                with open(path) as f:
                    text = f.read()
                with open(path, "w") as f:
                    f.write(text.replace("1/2 passing", "2/2 passing"))
        res, *_ = self.run_it(["x"], [verdict({"a": True, "b": False}), verdict({"a": True, "b": True})], hook=hook)
        self.assertTrue(res.events[1].scratchpad.engine_tampered)
        with open(path) as f:
            text = f.read()
        self.assertNotIn("iteration 1: 2/2 passing", text)

    def test_checkpoint_has_progress(self):
        self.run_it(["x"], [verdict({"a": True, "b": False}), verdict({"a": True, "b": True})])
        state = LoopStore(os.path.join(self.tmp.name, "loop")).load()
        self.assertEqual((state["status"], state["iteration"]), ("succeeded", 2))
        self.assertEqual(state["requirement_ids"], ["a", "b"])
        self.assertEqual(len(state["history"]), 2)


class AgentMethodTests(EngineBase):
    def test_agent_run_loop_accepts_plain_string_requirements(self):
        from unittest.mock import MagicMock

        from maki.agents import Agent

        backend = MagicMock()
        backend.chat.return_value.content = "the answer"
        agent = Agent("maker", backend, stateful=True)
        judge = FakeAgent([verdict({"r1": True, "r2": True})], stateful=False)
        res = agent.run_loop("goal", ["first", "second"], judge, loop_dir=self.tmp.name, loop_id="am")
        self.assertTrue(res.succeeded)
        self.assertEqual(res.artifact, "the answer")
        self.assertEqual(list(res.final_results), ["r1", "r2"])


class StartValidationTests(EngineBase):
    def test_unapproved_draft_refused(self):
        with self.assertRaises(ValueError):
            self.run_it(["x"], ["u"], reqs=[Requirement("d1", "x", source="draft")])

    def test_dangerous_plugin_requires_opt_in(self):
        class P:
            DANGEROUS_METHODS = ["write_file"]
        maker, judge = FakeAgent(["x"]), FakeAgent(["u"], stateful=False)
        maker.plugins = {"file_writer": P()}
        with self.assertRaises(ValueError):
            run_loop(maker, judge, self.spec(use_plugins=True))
        maker.allow_dangerous_tools = True
        maker.replies, judge.replies = ["x"], [verdict({"a": True, "b": True})]
        self.assertTrue(run_loop(maker, judge, self.spec(use_plugins=True)).succeeded)

    def test_stateful_judge_refused(self):
        with self.assertRaises(ValueError):
            run_loop(FakeAgent(["x"]), FakeAgent(["u"], stateful=True), self.spec())


if __name__ == "__main__":
    unittest.main()
