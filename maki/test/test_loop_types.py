"""Requirement / LoopSpec validation and the stall tracker."""
import unittest

from maki.agents.loop import LoopSpec, Requirement, StallTracker


class RequirementTests(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(Requirement("a", "x").kind, "agent")
        self.assertEqual(Requirement("a", "x", command=["true"]).kind, "command")
        self.assertEqual(Requirement("a", "x", check=lambda t: True).kind, "callable")

    def test_command_must_be_argv_list(self):
        with self.assertRaises(ValueError):
            Requirement("a", "x", command="pytest -q")
        with self.assertRaises(ValueError):
            Requirement("a", "x", command=[])
        self.assertEqual(Requirement("a", "x", command=["a", "b"]).command, ("a", "b"))

    def test_command_and_check_exclusive(self):
        with self.assertRaises(ValueError):
            Requirement("a", "x", command=["true"], check=lambda t: True)

    def test_draft_cannot_carry_command_or_check(self):
        with self.assertRaises(ValueError):
            Requirement("a", "x", command=["true"], source="draft")
        with self.assertRaises(ValueError):
            Requirement("a", "x", check=lambda t: True, source="draft")

    def test_blank_id_text_and_bad_source(self):
        for kw in ({"id": " ", "text": "x"}, {"id": "a", "text": ""}, {"id": "a", "text": "x", "source": "bot"}):
            with self.assertRaises(ValueError):
                Requirement(**kw)


class LoopSpecTests(unittest.TestCase):
    def reqs(self):
        return [Requirement("a", "x")]

    def test_defaults(self):
        s = LoopSpec("goal", self.reqs())
        self.assertEqual((s.max_iterations, s.stall_limit, s.context), (10, 3, "shared"))

    def test_validation(self):
        for kw in ({"goal": ""}, {"requirements": []}, {"max_iterations": 0}, {"max_iterations": True},
                   {"max_artifact_chars": 10}, {"max_artifact_chars": True},
                   {"stall_limit": 0}, {"max_seconds": 0}, {"context": "weird"},
                   {"requirements": [Requirement("a", "x"), Requirement("a", "y")]}):
            args = {"goal": "g", "requirements": self.reqs(), **kw}
            with self.assertRaises(ValueError, msg=str(kw)):
                LoopSpec(**args)


class StallTrackerTests(unittest.TestCase):
    def test_improvement_resets_and_stall_at_limit(self):
        t = StallTracker(3)
        self.assertTrue(t.update(2))
        self.assertFalse(t.update(2))
        self.assertFalse(t.update(1))
        self.assertFalse(t.stalled)
        self.assertFalse(t.update(1))
        self.assertTrue(t.stalled)

    def test_recovery_after_regression_resets(self):
        t = StallTracker(3)
        t.update(3)
        t.update(2)
        self.assertEqual(t.streak, 1)
        self.assertTrue(t.update(3))
        self.assertEqual(t.streak, 0)

    def test_judge_error_iterations_are_ignored(self):
        t = StallTracker(2)
        t.update(1)
        t.update(None)
        t.update(None)
        self.assertEqual(t.streak, 0)
        self.assertFalse(t.stalled)

    def test_zero_passing_first_iteration_counts_against(self):
        t = StallTracker(1)
        t.update(0)
        self.assertTrue(t.stalled)

    def test_swap_is_not_improvement(self):
        t = StallTracker(5)
        t.update(2)
        self.assertFalse(t.update(2))


if __name__ == "__main__":
    unittest.main()
