"""YAML ``judge:`` and ``loop:`` sections for maki serve."""
import os
import tempfile
import unittest

pytest = __import__("pytest")
pytest.importorskip("yaml", reason="PyYAML not installed")

from maki.distributed.config_loader import load_agent_from_config, load_server_config  # noqa: E402


class ServerConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, body):
        path = os.path.join(self.tmp.name, "agent.yaml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)
        return load_server_config(path)

    def test_no_judge_or_loop_section_means_loops_disabled_with_default_caps(self):
        cfg = self.load("name: a\n")
        self.assertIsNone(cfg.judge)
        self.assertEqual((cfg.loop_caps.max_iterations, cfg.loop_caps.max_seconds), (20, 3600.0))

    def test_judge_is_a_separate_stateless_reviewer_in_json_mode_without_thinking(self):
        cfg = self.load("name: a\nstateful: true\njudge:\n  model: gemma4:26b\n  temperature: 0\n")
        judge = cfg.judge
        self.assertEqual((judge.name, judge.role, judge.stateful), ("a-judge", "reviewer", False))
        self.assertIsNot(judge.maki, cfg.agent.maki)
        self.assertTrue(judge.maki.json_format)
        self.assertIs(judge.maki.think, False)
        self.assertEqual(judge.maki.model, "gemma4:26b")
        self.assertFalse(getattr(cfg.agent.maki, "json_format", False))  # only the judge gets JSON mode
        self.assertFalse(judge.plugins)

    def test_think_flag_reaches_the_ollama_backend_of_the_served_agent(self):
        self.assertIs(self.load("name: a\nthink: false\n").agent.maki.think, False)
        self.assertIsNone(self.load("name: a\n").agent.maki.think)

    def test_judge_with_plugins_or_unknown_keys_is_rejected(self):
        for body in ("name: a\njudge:\n  plugins: [file_writer]\n", "name: a\njudge:\n  allow_dangerous_tools: true\n",
                     "name: a\njudge: nope\n"):
            with self.assertRaises(ValueError, msg=body):
                self.load(body)

    def test_loop_caps(self):
        cfg = self.load("name: a\nloop:\n  max_iterations: 7\n  max_seconds: 90\n  loop_dir_base: ~/x-loops\n")
        self.assertEqual((cfg.loop_caps.max_iterations, cfg.loop_caps.max_seconds), (7, 90.0))
        self.assertEqual(cfg.loop_caps.loop_dir_base, os.path.expanduser("~/x-loops"))

    def test_retention_days(self):
        self.assertEqual(self.load("name: a\n").loop_caps.retention_days, 14.0)
        self.assertEqual(self.load("name: a\nloop:\n  retention_days: 2\n").loop_caps.retention_days, 2.0)
        self.assertIsNone(self.load("name: a\nloop:\n  retention_days: null\n").loop_caps.retention_days)

    def test_loop_cap_validation(self):
        for body in ("name: a\nloop:\n  retention_days: 0\n", "name: a\nloop:\n  max_iterations: 0\n",
                     "name: a\nloop:\n  max_seconds: -5\n", "name: a\nloop:\n  workers: 4\n", "name: a\nloop: 3\n"):
            with self.assertRaises(ValueError, msg=body):
                self.load(body)

    def test_existing_loader_still_ignores_the_new_sections(self):
        path = os.path.join(self.tmp.name, "agent.yaml")
        with open(path, "w", encoding="utf-8") as f:
            f.write("name: a\njudge:\n  model: x\n")
        self.assertEqual(load_agent_from_config(path).name, "a")


if __name__ == "__main__":
    unittest.main()
