"""Agent.load_plugin forwards operator options to register_plugin."""
import os
import tempfile
import textwrap
import unittest
from unittest.mock import MagicMock

from maki.agents import Agent
from maki.backend import LLMBackend


def make_agent():
    return Agent("A", MagicMock(spec=LLMBackend), "tester", "t")


class PluginOptionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = os.path.realpath(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_options_reach_the_plugin(self):
        agent = make_agent()
        plugin = agent.load_plugin("file_mover", base_dir=self.d, journal_dir=os.path.join(self.d, "j"),
                                   allow_any_path=True)
        self.assertEqual(plugin.scope.base_dir, self.d)
        self.assertTrue(plugin.scope.allow_any_path)
        self.assertIs(agent.get_plugin("file_mover"), plugin)

    def test_loading_without_options_still_works(self):
        agent = make_agent()
        self.assertIsNotNone(agent.load_plugin("file_reader"))
        self.assertIsNotNone(agent.load_plugin("json_reader"))

    def test_an_existing_plugin_gets_its_option_too(self):
        vault = os.path.join(self.d, "vault")
        os.makedirs(vault)
        plugin = make_agent().load_plugin("obsidian_memory", vault_path=vault)
        self.assertEqual(str(plugin._vault), vault)

    def test_an_option_the_plugin_does_not_know_is_a_clear_error(self):
        agent = make_agent()
        with self.assertRaises(TypeError) as ctx:
            agent.load_plugin("file_mover", not_an_option=True)
        self.assertIn("not_an_option", str(ctx.exception))
        self.assertIsNone(agent.get_plugin("file_mover"))

    def test_options_also_reach_a_plugin_loaded_from_a_path(self):
        folder = os.path.join(self.d, "plugins", "custom")
        os.makedirs(folder)
        with open(os.path.join(folder, "__init__.py"), "w") as f:
            f.write(textwrap.dedent('''
                class Custom:
                    ALLOWED_METHODS = ["hello"]
                    def __init__(self, maki=None, greeting="hi"):
                        self.greeting = greeting
                    def hello(self):
                        return self.greeting

                def register_plugin(maki=None, **options):
                    return Custom(maki, **options)
            '''))
        plugin = make_agent().load_plugin("custom", plugin_path=os.path.join(self.d, "plugins"), greeting="yo")
        self.assertEqual(plugin.hello(), "yo")


if __name__ == "__main__":
    unittest.main()
