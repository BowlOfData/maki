"""
Tests for maki.distributed.config_loader (review §2.6).

The backend builder is patched out so no real LLM/network is required.
Skipped automatically if PyYAML is not installed.
"""
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

pytest = __import__("pytest")
pytest.importorskip("yaml", reason="PyYAML not installed")

from maki.distributed.config_loader import load_agent_from_config


def _write_config(body: str) -> str:
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    f.write(body)
    f.close()
    return f.name


class TestPluginValidation(unittest.TestCase):

    def setUp(self):
        self._paths = []

    def tearDown(self):
        for p in self._paths:
            os.unlink(p)

    def _config(self, body: str) -> str:
        path = _write_config(body)
        self._paths.append(path)
        return path

    def test_unknown_plugin_rejected_before_backend_build(self):
        path = self._config("name: a\nplugins:\n  - no_such_plugin\n")
        with patch(
            "maki.distributed.config_loader._build_backend"
        ) as build:
            with self.assertRaises(ValueError) as ctx:
                load_agent_from_config(path)
        # Fails fast: the backend (which may touch the network) is never built.
        build.assert_not_called()
        msg = str(ctx.exception)
        self.assertIn("no_such_plugin", msg)
        self.assertIn("file_reader", msg)  # lists valid names

    def test_module_traversal_name_rejected(self):
        # §2.6: names like "..something" reached importlib unchecked.
        path = self._config('name: a\nplugins:\n  - "..something"\n')
        with self.assertRaises(ValueError):
            load_agent_from_config(path)

    def test_submodule_name_rejected(self):
        path = self._config('name: a\nplugins:\n  - "ocr.something"\n')
        with self.assertRaises(ValueError):
            load_agent_from_config(path)

    def test_registered_plugin_loads(self):
        path = self._config("name: a\nplugins:\n  - file_reader\n")
        with patch(
            "maki.distributed.config_loader._build_backend",
            return_value=MagicMock(),
        ):
            agent = load_agent_from_config(path)
        self.assertIn("file_reader", agent.plugins)


class TestDangerousToolsFlag(unittest.TestCase):

    def _load(self, body: str):
        path = _write_config(body)
        try:
            with patch(
                "maki.distributed.config_loader._build_backend",
                return_value=MagicMock(),
            ):
                return load_agent_from_config(path)
        finally:
            os.unlink(path)

    def test_defaults_to_false(self):
        agent = self._load("name: a\n")
        self.assertFalse(agent.allow_dangerous_tools)

    def test_explicit_opt_in(self):
        agent = self._load("name: a\nallow_dangerous_tools: true\n")
        self.assertTrue(agent.allow_dangerous_tools)


if __name__ == "__main__":
    unittest.main()
