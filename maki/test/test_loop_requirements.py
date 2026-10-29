"""Requirement drafting and approval."""
import json
import unittest

from maki.agents.loop import Requirement, approve_requirements, draft_requirements

from .loop_helpers import FakeAgent


class DraftTests(unittest.TestCase):
    def test_drafts_are_marked_and_get_stable_ids(self):
        reply = json.dumps({"requirements": [{"text": "has a docstring"}, {"text": " no prints "}, {"nope": 1}]})
        out = draft_requirements(FakeAgent([reply], stateful=False), "write code")
        self.assertEqual([(r.id, r.text, r.source, r.kind) for r in out],
                         [("d1", "has a docstring", "draft", "agent"), ("d2", "no prints", "draft", "agent")])

    def test_max_items_caps(self):
        reply = json.dumps({"requirements": [{"text": f"r{i}"} for i in range(20)]})
        self.assertEqual(len(draft_requirements(FakeAgent([reply]), "g", max_items=3)), 3)

    def test_unusable_reply_raises(self):
        for bad in ("nothing", json.dumps({"requirements": []}), json.dumps({"requirements": "x"})):
            with self.assertRaises(ValueError):
                draft_requirements(FakeAgent([bad]), "g")

    def test_drafter_cannot_smuggle_a_command(self):
        reply = json.dumps({"requirements": [{"text": "x", "command": ["rm", "-rf", "/"]}]})
        out = draft_requirements(FakeAgent([reply]), "g")
        self.assertIsNone(out[0].command)


class ApprovalTests(unittest.TestCase):
    def test_approval_freezes_drafts_and_keeps_user_requirements(self):
        reqs = [Requirement("d1", "x", source="draft"), Requirement("u1", "y", command=["true"])]
        out = approve_requirements(reqs, lambda r: True)
        self.assertEqual([r.source for r in out], ["approved", "user"])

    def test_approver_sees_commands_in_full(self):
        seen = []
        approve_requirements([Requirement("u1", "y", command=["pytest", "-q"])], lambda r: seen.append(r) or True)
        self.assertEqual(seen[0][0].command, ("pytest", "-q"))

    def test_decline_raises(self):
        with self.assertRaises(ValueError):
            approve_requirements([Requirement("d1", "x", source="draft")], lambda r: False)


if __name__ == "__main__":
    unittest.main()
