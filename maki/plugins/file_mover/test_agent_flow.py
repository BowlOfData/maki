"""FileMover through the real tool-call path of an Agent (scripted model, no network)."""
import json
import os
import tempfile
import unittest

from maki.agents import Agent
from maki.objects import BackendType, LLMResponse


class ScriptedBackend:
    """Stands in for an LLM: returns the next scripted reply and remembers every prompt it was sent."""
    supports_native_tools = False

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def request(self, prompt):
        return self.chat(prompt)

    def chat(self, prompt, system=None, **kwargs):
        self.prompts.append(prompt)
        return LLMResponse(self.replies.pop(0), "scripted", 1, 1, 2, 0.0, backend=BackendType.OLLAMA)


def tool(method, **args):
    return "TOOL: " + json.dumps({"plugin": "file_mover", "method": method, "args": args})


class AgentFlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = os.path.realpath(self.tmp.name)
        self.work = os.path.join(self.d, "work")
        self.outside = os.path.join(self.d, "outside")
        os.makedirs(self.work)
        os.makedirs(self.outside)
        for name in ("a.txt", "b.txt"):
            with open(os.path.join(self.work, name), "w") as f:
                f.write(name)

    def tearDown(self):
        self.tmp.cleanup()

    def agent(self, replies, dangerous):
        backend = ScriptedBackend(replies)
        agent = Agent("mover", backend, "file organiser", "Organise files.", allow_dangerous_tools=dangerous)
        agent.load_plugin("file_mover", base_dir=self.work, journal_dir=os.path.join(self.d, "journal"))
        return agent, backend

    def w(self, name):
        return os.path.join(self.work, name)

    def test_without_the_dangerous_flag_a_move_is_blocked_but_the_journal_can_be_read(self):
        agent, backend = self.agent([tool("move", src="a.txt", dst="c.txt"), "blocked", tool("list_moves"), "done"],
                                    dangerous=False)
        agent.execute_task("move a.txt to c.txt", use_plugins=True)
        self.assertTrue(os.path.exists(self.w("a.txt")))
        self.assertFalse(os.path.exists(self.w("c.txt")))
        self.assertIn("dangerous", backend.prompts[1])
        agent.execute_task("what moved?", use_plugins=True)
        self.assertIn("'success': True", backend.prompts[3])  # list_moves ran: it is not dangerous

    def test_with_the_flag_a_move_inside_the_scope_happens(self):
        agent, backend = self.agent([tool("move", src="a.txt", dst="renamed/a.txt", create_dirs=True), "moved"],
                                    dangerous=True)
        self.assertEqual(agent.execute_task("organise", use_plugins=True), "moved")
        self.assertTrue(os.path.exists(self.w(os.path.join("renamed", "a.txt"))))
        self.assertFalse(os.path.exists(self.w("a.txt")))
        self.assertIn("'success': True", backend.prompts[1])

    def test_a_move_outside_the_scope_is_refused_even_with_the_flag(self):
        target = os.path.join(self.outside, "a.txt")
        agent, backend = self.agent([tool("move", src="a.txt", dst=target), "refused"], dangerous=True)
        agent.execute_task("move it away", use_plugins=True)
        self.assertTrue(os.path.exists(self.w("a.txt")))
        self.assertFalse(os.path.exists(target))
        self.assertIn("outside the allowed", backend.prompts[1])

    def test_the_model_cannot_pass_scope_arguments(self):
        agent, backend = self.agent([tool("move", src="a.txt", dst="x.txt", allow_any_path=True), "no"],
                                    dangerous=True)
        agent.execute_task("try to widen", use_plugins=True)
        self.assertTrue(os.path.exists(self.w("a.txt")))  # the call failed on the unknown argument
        self.assertIn("allow_any_path", backend.prompts[1])

    def test_dry_run_then_move_then_undo_in_one_conversation(self):
        replies = [tool("move", src="a.txt", dst="c.txt", dry_run=True), "plan ok",
                   tool("move", src="a.txt", dst="c.txt"), "moved",
                   tool("undo_last"), "undone"]
        agent, backend = self.agent(replies, dangerous=True)
        agent.execute_task("preview", use_plugins=True)
        self.assertTrue(os.path.exists(self.w("a.txt")))
        agent.execute_task("do it", use_plugins=True)
        self.assertTrue(os.path.exists(self.w("c.txt")))
        agent.execute_task("undo", use_plugins=True)
        self.assertTrue(os.path.exists(self.w("a.txt")))
        self.assertFalse(os.path.exists(self.w("c.txt")))


if __name__ == "__main__":
    unittest.main()
