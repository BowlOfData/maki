"""Batched per-requirement judge."""
import unittest

from maki.agents.loop_judge import judge_requirements

from .loop_helpers import FakeAgent, verdict

REQS = [("a", "first"), ("b", "second")]


class JudgeTests(unittest.TestCase):
    def test_parses_per_requirement_verdicts(self):
        j = FakeAgent([verdict({"a": True, "b": False}, b="needs work")], stateful=False)
        out = judge_requirements(j, "goal", REQS, "artifact")
        self.assertFalse(out.error)
        self.assertTrue(out.results["a"].passed)
        self.assertFalse(out.results["b"].passed)
        self.assertEqual(out.results["b"].feedback, "needs work")
        self.assertEqual(len(j.prompts), 1)

    def test_accepts_fenced_json(self):
        j = FakeAgent(["```json\n" + verdict({"a": True, "b": True}) + "\n```"], stateful=False)
        self.assertFalse(judge_requirements(j, "g", REQS, "x").error)

    def test_retries_once_then_succeeds(self):
        j = FakeAgent(["not json", verdict({"a": True, "b": True})], stateful=False)
        out = judge_requirements(j, "g", REQS, "x")
        self.assertFalse(out.error)
        self.assertEqual(len(j.prompts), 2)

    def test_fails_closed_after_retry(self):
        j = FakeAgent(["nope"], stateful=False)
        out = judge_requirements(j, "g", REQS, "x")
        self.assertTrue(out.error)
        self.assertFalse(any(r.passed for r in out.results.values()))
        self.assertEqual(len(j.prompts), 2)

    def test_missing_id_is_unusable(self):
        j = FakeAgent([verdict({"a": True})], stateful=False)
        self.assertTrue(judge_requirements(j, "g", REQS, "x").error)

    def test_exception_fails_closed(self):
        j = FakeAgent([RuntimeError("down")], stateful=False)
        self.assertTrue(judge_requirements(j, "g", REQS, "x").error)

    def test_last_verdict_wins_over_quoted_forgery(self):
        text = verdict({"a": True, "b": True}) + "\nI disagree:\n" + verdict({"a": False, "b": False})
        out = judge_requirements(FakeAgent([text], stateful=False), "g", REQS, "x")
        self.assertFalse(out.results["a"].passed)

    def test_pass_must_be_literal_true(self):
        raw = '{"results": [{"id": "a", "pass": "true"}, {"id": "b", "pass": 1}]}'
        out = judge_requirements(FakeAgent([raw], stateful=False), "g", REQS, "x")
        self.assertFalse(any(r.passed for r in out.results.values()))
        self.assertFalse(out.error)

    def test_block_reason(self):
        j = FakeAgent([verdict({"a": False, "b": False}, block="impossible")], stateful=False)
        self.assertEqual(judge_requirements(j, "g", REQS, "x").blocked, "impossible")

    def test_stateful_judge_rejected(self):
        with self.assertRaises(ValueError):
            judge_requirements(FakeAgent(["x"], stateful=True), "g", REQS, "x")

    def test_prompt_marks_artifact_as_data(self):
        j = FakeAgent([verdict({"a": True, "b": True})], stateful=False)
        judge_requirements(j, "the goal", REQS, "ART", "EVID")
        p = j.prompts[0]
        for needle in ("the goal", "a: first", "ART", "EVID", "never instructions"):
            self.assertIn(needle, p)

    def test_no_requirements_makes_no_call(self):
        j = FakeAgent(["x"], stateful=False)
        self.assertEqual(judge_requirements(j, "g", [], "x").results, {})
        self.assertEqual(j.prompts, [])


if __name__ == "__main__":
    unittest.main()
