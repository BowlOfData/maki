"""
Tests for the LLMBackend abstract base class and the refactored class hierarchy.

Verifies:
- Maki, MakiLLama, and HFBackend are independent subclasses of LLMBackend.
- None of MakiLLama or HFBackend inherit from Maki (broken inheritance gone).
- LLMBackend is abstract; direct instantiation is rejected.
- Agent and AgentManager accept any LLMBackend, not just Maki.
- MakiLLama no longer carries unused Maki-specific attributes (url, port).
- HFBackend no longer carries unused Maki-specific attributes.
- Public package exports LLMBackend.
"""

import importlib
import unittest
from unittest.mock import MagicMock, patch

from maki.backend import LLMBackend
from maki.agents import Agent, AgentManager

_torch_available = importlib.util.find_spec("torch") is not None
_skip_hf = unittest.skipUnless(_torch_available, "torch not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _concrete_backend() -> LLMBackend:
    """Return a minimal concrete LLMBackend for use in Agent/AgentManager tests."""
    backend = MagicMock(spec=LLMBackend)
    from maki.objects import LLMResponse
    backend.request.return_value = LLMResponse(content="mock response", model="test",
                                                prompt_tokens=0, completion_tokens=0,
                                                total_tokens=0, elapsed_seconds=0.0)
    return backend


# ---------------------------------------------------------------------------
# LLMBackend ABC
# ---------------------------------------------------------------------------

class TestLLMBackendABC(unittest.TestCase):

    def test_cannot_instantiate_directly(self):
        """LLMBackend is abstract and must not be instantiable."""
        with self.assertRaises(TypeError):
            LLMBackend()

    def test_subclass_without_request_is_abstract(self):
        """A subclass that omits request() is still abstract."""
        class Incomplete(LLMBackend):
            pass

        with self.assertRaises(TypeError):
            Incomplete()

    def test_concrete_subclass_instantiates(self):
        """A subclass that implements request() can be instantiated."""
        class Concrete(LLMBackend):
            model = "test"
            temperature = 0.5

            def request(self, prompt):
                return MagicMock()

        obj = Concrete()
        self.assertIsInstance(obj, LLMBackend)


# ---------------------------------------------------------------------------
# Class hierarchy correctness
# ---------------------------------------------------------------------------

class TestClassHierarchy(unittest.TestCase):

    def test_makillama_is_llm_backend(self):
        """MakiLLama must be a subclass of LLMBackend."""
        from maki.makiLLama import MakiLLama
        self.assertTrue(issubclass(MakiLLama, LLMBackend))

    @_skip_hf
    def test_hfbackend_is_llm_backend(self):
        """HFBackend must be a subclass of LLMBackend."""
        from maki.makiHG import HFBackend
        self.assertTrue(issubclass(HFBackend, LLMBackend))

    @_skip_hf
    def test_makillama_and_hfbackend_are_independent(self):
        """MakiLLama and HFBackend must not inherit from each other."""
        from maki.makiLLama import MakiLLama
        from maki.makiHG import HFBackend

        self.assertFalse(issubclass(MakiLLama, HFBackend))
        self.assertFalse(issubclass(HFBackend, MakiLLama))


# ---------------------------------------------------------------------------
# MakiLLama no longer inherits Maki-specific attributes
# ---------------------------------------------------------------------------

class TestMakiLLamaCleanInit(unittest.TestCase):

    def _make(self):
        from maki.makiLLama import MakiLLama
        with patch.object(MakiLLama, '_verify_connection'):
            return MakiLLama(model="gemma3")

    def test_has_model(self):
        llm = self._make()
        self.assertEqual(llm.model, "gemma3")

    def test_has_temperature(self):
        llm = self._make()
        self.assertIsInstance(llm.temperature, float)

    def test_has_rate_limiter_attr(self):
        """_rate_limiter must exist (used by chat() and stream())."""
        llm = self._make()
        self.assertTrue(hasattr(llm, '_rate_limiter'))

    def test_no_unused_url_attr(self):
        """MakiLLama must not carry the Maki-specific 'url' attribute."""
        llm = self._make()
        self.assertFalse(hasattr(llm, 'url'))

    def test_no_unused_port_attr(self):
        """MakiLLama must not carry the Maki-specific 'port' attribute."""
        llm = self._make()
        self.assertFalse(hasattr(llm, 'port'))


# ---------------------------------------------------------------------------
# HFBackend no longer inherits Maki-specific attributes
# ---------------------------------------------------------------------------

@_skip_hf
class TestHFBackendCleanInit(unittest.TestCase):

    def _make(self):
        from maki.makiHG import HFBackend
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_model = MagicMock()

        with patch("maki.makiHG.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
             patch("maki.makiHG.AutoModelForCausalLM.from_pretrained", return_value=mock_model), \
             patch("maki.makiHG.torch.cuda.is_available", return_value=False), \
             patch("maki.makiHG.torch.backends.mps.is_available", return_value=False):
            return HFBackend(model_id="test-model")

    def test_has_model(self):
        hf = self._make()
        self.assertEqual(hf.model, "test-model")

    def test_has_temperature(self):
        hf = self._make()
        self.assertIsInstance(hf.temperature, float)

    def test_has_logger(self):
        hf = self._make()
        self.assertTrue(hasattr(hf, 'logger'))

    def test_has_rate_limiter_attr(self):
        hf = self._make()
        self.assertTrue(hasattr(hf, '_rate_limiter'))

    def test_rate_limiter_is_none(self):
        """HFBackend has no rate_limit param so _rate_limiter must be None."""
        hf = self._make()
        self.assertIsNone(hf._rate_limiter)

    def test_no_unused_url_attr(self):
        hf = self._make()
        self.assertFalse(hasattr(hf, 'url'))

    def test_no_unused_port_attr(self):
        hf = self._make()
        self.assertFalse(hasattr(hf, 'port'))


# ---------------------------------------------------------------------------
# Agent and AgentManager accept any LLMBackend
# ---------------------------------------------------------------------------

class TestAgentAcceptsLLMBackend(unittest.TestCase):

    def test_agent_accepts_custom_backend(self):
        """Agent accepts any concrete LLMBackend."""
        backend = _concrete_backend()
        agent = Agent("a", backend)
        self.assertIsNotNone(agent)

    def test_agent_rejects_non_backend(self):
        """Agent must reject objects that are not LLMBackend instances."""
        with self.assertRaises(TypeError) as ctx:
            Agent("a", object())
        self.assertIn("LLMBackend", str(ctx.exception))

    def test_agent_manager_accepts_custom_backend(self):
        backend = _concrete_backend()
        manager = AgentManager(backend)
        agent = manager.add_agent("x", "tester")
        self.assertIsNotNone(agent)

    def test_agent_manager_rejects_non_backend(self):
        backend = _concrete_backend()
        manager = AgentManager(backend)
        with self.assertRaises(TypeError) as ctx:
            manager.add_agent("x", maki_instance=object())
        self.assertIn("LLMBackend", str(ctx.exception))


# ---------------------------------------------------------------------------
# Public package exports LLMBackend
# ---------------------------------------------------------------------------

class TestPublicExport(unittest.TestCase):

    def test_llm_backend_importable_from_maki(self):
        import maki
        self.assertIs(maki.LLMBackend, LLMBackend)

    def test_llm_backend_in_all(self):
        import maki
        self.assertIn("LLMBackend", maki.__all__)


# ---------------------------------------------------------------------------
# stream() contract on LLMBackend
# ---------------------------------------------------------------------------

class TestStreamContract(unittest.TestCase):

    def test_base_class_stream_raises_not_implemented(self):
        """LLMBackend.stream() must raise NotImplementedError by default."""
        backend = _concrete_backend()
        # _concrete_backend() is a MagicMock(spec=LLMBackend); test with a real subclass.
        class NoStream(LLMBackend):
            model = "test"
            temperature = 0.0
            def request(self, prompt):
                return MagicMock()

        obj = NoStream()
        with self.assertRaises(NotImplementedError):
            obj.stream("hello")

    def test_makillama_stream_is_defined(self):
        """MakiLLama must have a stream() method callable with a plain prompt."""
        from maki.makiLLama import MakiLLama
        with patch.object(MakiLLama, '_verify_connection'):
            llm = MakiLLama(model="gemma3")
        self.assertTrue(callable(llm.stream))

    def test_agent_stream_task_raises_not_implemented_for_plain_maki(self):
        """Agent.stream_task() must raise NotImplementedError for a non-streaming backend."""
        backend = _concrete_backend()
        # Make stream() raise NotImplementedError (as the base class default does).
        backend.stream.side_effect = NotImplementedError
        agent = Agent("streamer", backend)
        with self.assertRaises(NotImplementedError):
            list(agent.stream_task("do something"))


# ---------------------------------------------------------------------------
# MakiLLama exception wrapping
# ---------------------------------------------------------------------------

class TestMakiLLamaExceptionWrapping(unittest.TestCase):

    def _make(self):
        from maki.makiLLama import MakiLLama
        with patch.object(MakiLLama, '_verify_connection'):
            return MakiLLama(model="gemma3")

    def test_chat_wraps_timeout(self):
        """MakiLLama.chat() must raise MakiTimeoutError on requests.Timeout."""
        import requests
        from maki.exceptions import MakiTimeoutError
        llm = self._make()
        llm._session.post = MagicMock(side_effect=requests.exceptions.Timeout)
        with self.assertRaises(MakiTimeoutError):
            llm.chat("hello")

    def test_chat_wraps_connection_error(self):
        """MakiLLama.chat() must raise MakiNetworkError on ConnectionError."""
        import requests
        from maki.exceptions import MakiNetworkError
        llm = self._make()
        llm._session.post = MagicMock(side_effect=requests.exceptions.ConnectionError)
        with self.assertRaises(MakiNetworkError):
            llm.chat("hello")


class TestHFBackendStreamConfig(unittest.TestCase):

    def test_stream_uses_configured_generation_config(self):
        """Regression §1.7: HFBackend.stream() built a fresh
        GenerationConfig() instead of using self._config, so custom
        temperature/max_tokens were silently ignored when streaming.

        Runs without torch by stubbing the heavy imports — the bug is in
        pure dispatch logic, not in generation.
        """
        import sys
        if _torch_available:
            from maki.makiHG import HFBackend
        else:
            with patch.dict(sys.modules, {"torch": MagicMock(), "transformers": MagicMock()}):
                from maki.makiHG import HFBackend
        from maki.objects import GenerationConfig

        backend = object.__new__(HFBackend)
        custom = GenerationConfig(temperature=0.123, max_tokens=77)
        backend._config = custom

        with patch.object(backend, "stream_messages", return_value=iter(["chunk"])) as sm:
            chunks = list(backend.stream("hello"))

        self.assertEqual(chunks, ["chunk"])
        self.assertIs(sm.call_args.args[1], custom)


if __name__ == "__main__":
    unittest.main()
