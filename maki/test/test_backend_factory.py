"""Tests for init_backend("provider:model") and the YAML model shorthand."""
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from maki.backend_factory import BACKENDS, init_backend, split_spec


class Recorder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def request(self, prompt):  # Agent needs a backend-shaped object
        raise NotImplementedError


def _fake_import(requested):
    def fake(name, package=None):
        requested.append(name)
        return SimpleNamespace(**{cls: Recorder for _, cls in BACKENDS.values()})
    return fake


class TestSplitSpec(unittest.TestCase):

    def test_provider_and_model(self):
        self.assertEqual(split_spec("openai:gpt-4o"), ("openai", "gpt-4o"))

    def test_only_first_colon_splits(self):
        self.assertEqual(split_spec("ollama:gemma4:26b"), ("ollama", "gemma4:26b"))

    def test_provider_alone_and_case(self):
        self.assertEqual(split_spec("Anthropic"), ("anthropic", None))
        self.assertEqual(split_spec("ollama:"), ("ollama", None))

    def test_unknown_provider_suggests_ollama_form(self):
        with self.assertRaisesRegex(ValueError, "ollama:gemma4:26b"):
            split_spec("gemma4:26b")

    def test_typo_suggests_provider_and_keeps_ollama_reading(self):
        with self.assertRaisesRegex(ValueError, "Did you mean 'openai'"):
            split_spec("opnai:gpt-4o")

    def test_ollama_tag_that_resembles_a_provider(self):
        with self.assertRaisesRegex(ValueError, "ollama:llama3:8b"):
            split_spec("llama3:8b")

    def test_empty_rejected(self):
        for bad in ("", "  ", None, 3):
            with self.assertRaises(ValueError):
                split_spec(bad)


class TestInitBackend(unittest.TestCase):

    def setUp(self):
        self.requested = []
        patcher = patch("maki.backend_factory.importlib.import_module", _fake_import(self.requested))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_model_and_kwargs_forwarded(self):
        backend = init_backend("openai:gpt-4o", base_url="http://x")
        self.assertEqual(backend.kwargs, {"model": "gpt-4o", "base_url": "http://x"})
        self.assertEqual(self.requested, [".makiOpenAI"])

    def test_bare_provider_leaves_default_model(self):
        self.assertNotIn("model", init_backend("anthropic").kwargs)

    def test_temperature_shorthand_builds_config(self):
        self.assertEqual(init_backend("ollama:m", temperature=0.2).kwargs["config"].temperature, 0.2)
        self.assertEqual(init_backend("ollama:m", temperature=0).kwargs["config"].temperature, 0.0)

    def test_conflicting_arguments_rejected(self):
        with self.assertRaises(ValueError):
            init_backend("openai:gpt-4o", model="other")
        with self.assertRaises(ValueError):
            init_backend("openai:gpt-4o", temperature=0.1, config=object())

    def test_unknown_provider_imports_nothing(self):
        with self.assertRaises(ValueError):
            init_backend("nope:model")
        self.assertEqual(self.requested, [])


class TestPackageExport(unittest.TestCase):

    def test_lazy_export(self):
        import maki
        self.assertIs(maki.init_backend, init_backend)

    def test_real_ollama_backend(self):
        backend = init_backend("ollama:gemma4:26b")
        self.assertEqual(backend.model, "gemma4:26b")


class TestYamlShorthand(unittest.TestCase):

    def _build(self, body):
        pytest = __import__("pytest")
        pytest.importorskip("yaml")
        from maki.distributed import config_loader
        f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
        f.write(body)
        f.close()
        self.addCleanup(os.unlink, f.name)
        seen = []
        with patch.object(config_loader.importlib, "import_module", _fake_import(seen)):
            agent = config_loader.load_agent_from_config(f.name)
        return agent, seen

    def test_provider_prefix_selects_backend(self):
        agent, seen = self._build("name: a\nmodel: 'openai:gpt-4o'\n")
        self.assertEqual(seen, ["..makiOpenAI"])
        self.assertEqual(agent.maki.kwargs["model"], "gpt-4o")

    def test_explicit_backend_keeps_model_verbatim(self):
        agent, seen = self._build("name: a\nbackend: ollama\nmodel: 'gemma4:26b'\n")
        self.assertEqual(seen, ["..makiLLama"])
        self.assertEqual(agent.maki.kwargs["model"], "gemma4:26b")

    def test_ollama_tag_without_backend_is_not_a_provider(self):
        agent, seen = self._build("name: a\nmodel: 'gemma4:26b'\n")
        self.assertEqual(seen, ["..makiLLama"])
        self.assertEqual(agent.maki.kwargs["model"], "gemma4:26b")


if __name__ == "__main__":
    unittest.main()
