"""AgentManager.add_agent forwards the Agent options instead of leaving them to be set afterwards."""
import unittest
from unittest.mock import MagicMock

from maki.agents import AgentManager
from maki.agents.tool_policy import ToolPolicy
from maki.backend import LLMBackend
from maki.callbacks import AgentCallback
from maki.objects import MemoryPolicy


class TestAddAgentOptions(unittest.TestCase):
    def setUp(self):
        self.manager = AgentManager(MagicMock(spec=LLMBackend))

    def test_defaults_match_agent_defaults(self):
        agent = self.manager.add_agent("a")
        self.assertFalse(agent.stateful)
        self.assertFalse(agent.allow_dangerous_tools)
        self.assertEqual(agent._tool_policy(), ToolPolicy())
        self.assertEqual(agent._callbacks(), [])

    def test_options_reach_the_agent(self):
        callback = AgentCallback()
        agent = self.manager.add_agent(
            "a", "role", "instr", stateful=True, allow_dangerous_tools=True,
            memory=MemoryPolicy(token_budget=500, summarize_older=True),
            tool_policy={"retries": 2, "parallel": True}, callbacks=[callback])
        self.assertTrue(agent.stateful)
        self.assertTrue(agent.allow_dangerous_tools)
        self.assertEqual(agent.memory_policy.token_budget, 500)
        self.assertTrue(agent.memory_policy.summarize_older)
        self.assertEqual((agent._tool_policy().retries, agent._tool_policy().parallel), (2, True))
        self.assertEqual(agent._callbacks(), [callback])
        self.assertIs(self.manager.get_agent("a"), agent)

    def test_memory_accepts_a_dict(self):
        agent = self.manager.add_agent("a", stateful=True, memory={"token_budget": 300})
        self.assertEqual(agent.memory_policy.token_budget, 300)

    def test_invalid_options_are_rejected_like_in_agent(self):
        with self.assertRaises((ValueError, TypeError)):
            self.manager.add_agent("a", tool_policy={"nope": 1})
        with self.assertRaises((ValueError, TypeError)):
            self.manager.add_agent("b", memory={"token_budget": -1})
        self.assertNotIn("a", self.manager.agents)

    def test_new_options_are_keyword_only(self):
        with self.assertRaises(TypeError):
            self.manager.add_agent("a", "r", "i", None, False, True)


if __name__ == "__main__":
    unittest.main()
