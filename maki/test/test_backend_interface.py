"""
Regression tests for Phase 2.2 — Real backend interface.

Covers:
- LLMBackend.chat() / chat_collect() default implementations
- Agent uses chat() / chat_collect() without hasattr dispatch
- _call_llm available from PluginHandler and ReasoningEngine mixins
- MakiOpenAI converts historical images to image_url content blocks (§1.8)
- MakiAnthropic converts historical images to Anthropic content blocks (§1.8)
- GenerationConfig.to_openai_kwargs(model_family="reasoning") for o3/o1 (§1.9)
- GenerationConfig validates top_p, top_k, max_tokens (§5)
"""

import collections
import unittest
from unittest.mock import MagicMock, patch, call

from maki.backend import LLMBackend
from maki.objects import GenerationConfig, LLMResponse, Message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _llm(content: str = "ok") -> LLMResponse:
    return LLMResponse(content=content, model="test", prompt_tokens=0,
                       completion_tokens=0, total_tokens=0, elapsed_seconds=0.0)


class _MinimalBackend(LLMBackend):
    """Concrete backend that implements only request()."""
    model = "minimal"
    temperature = 0.7

    def request(self, prompt: str) -> LLMResponse:
        return _llm(f"request:{prompt}")


class _ChatBackend(LLMBackend):
    """Backend that overrides chat() (like all real backends do)."""
    model = "chat-only"
    temperature = 0.7

    def request(self, prompt: str) -> LLMResponse:
        return _llm("via-request")

    def chat(self, prompt, history=None, config=None, system=None, images=None):
        return _llm(f"chat:{prompt}")


# ---------------------------------------------------------------------------
# LLMBackend default method behaviour
# ---------------------------------------------------------------------------

class TestLLMBackendDefaults(unittest.TestCase):

    def test_chat_default_delegates_to_request(self):
        b = _MinimalBackend()
        result = b.chat("hello")
        self.assertEqual(result.content, "request:hello")

    def test_chat_collect_default_delegates_to_chat(self):
        b = _ChatBackend()
        result = b.chat_collect("hi")
        self.assertEqual(result.content, "chat:hi")

    def test_chat_collect_on_minimal_falls_back_to_request(self):
        b = _MinimalBackend()
        result = b.chat_collect("test")
        self.assertEqual(result.content, "request:test")

    def test_stream_still_raises_not_implemented(self):
        b = _MinimalBackend()
        with self.assertRaises(NotImplementedError):
            next(b.stream("x"))

    def test_stream_signature_accepts_kwargs(self):
        b = _MinimalBackend()
        with self.assertRaises(NotImplementedError):
            b.stream("x", system="s")


# ---------------------------------------------------------------------------
# Agent dispatch — no hasattr, always uses chat() / chat_collect()
# ---------------------------------------------------------------------------

class TestAgentDispatch(unittest.TestCase):

    def _make_agent(self, use_streaming=False):
        from maki.agents import Agent
        mock = MagicMock()
        mock.chat.return_value = _llm("chat-result")
        mock.chat_collect.return_value = _llm("collect-result")
        agent = Agent("A", mock, use_streaming=use_streaming)
        return agent, mock

    def test_execute_task_calls_chat(self):
        agent, mock = self._make_agent(use_streaming=False)
        result = agent.execute_task("do it")
        self.assertEqual(result, "chat-result")
        mock.chat.assert_called_once()
        mock.request.assert_not_called()

    def test_execute_task_with_streaming_calls_chat_collect(self):
        agent, mock = self._make_agent(use_streaming=True)
        result = agent.execute_task("do it")
        self.assertEqual(result, "collect-result")
        mock.chat_collect.assert_called_once()
        mock.chat.assert_not_called()

    def test_stream_task_passes_system_kwarg(self):
        from maki.agents import Agent
        mock = MagicMock()
        mock.stream.return_value = iter(["tok1", "tok2"])
        agent = Agent("B", mock, role="helper")
        chunks = list(agent.stream_task("task"))
        self.assertEqual(chunks, ["tok1", "tok2"])
        call_kwargs = mock.stream.call_args.kwargs
        self.assertIn("system", call_kwargs)

    def test_no_request_fallback_path_exists(self):
        """_build_prompt / the hasattr-else branch must be gone."""
        from maki.agents import agent as agent_module
        import inspect
        src = inspect.getsource(agent_module.Agent.execute_task)
        self.assertNotIn("hasattr", src)
        self.assertNotIn("_build_prompt", src)


# ---------------------------------------------------------------------------
# _call_llm available from mixins
# ---------------------------------------------------------------------------

class TestCallLlmFromMixins(unittest.TestCase):

    def test_plugin_handler_provides_call_llm(self):
        from maki.agents.plugin_handler import PluginHandler
        mock_maki = MagicMock()
        mock_maki.chat.return_value = _llm("plugin-llm")

        class P(PluginHandler):
            def __init__(self):
                self.name = "p"
                self.maki = mock_maki
                self._init_plugins()

        p = P()
        result = p._call_llm("prompt")
        self.assertEqual(result, "plugin-llm")
        mock_maki.chat.assert_called_once_with("prompt")

    def test_reasoning_engine_provides_call_llm(self):
        from maki.agents.reasoning import ReasoningEngine
        mock_maki = MagicMock()
        mock_maki.chat.return_value = _llm("reasoning-llm")

        class R(ReasoningEngine):
            def __init__(self):
                self.maki = mock_maki
                self.reasoning_history = collections.deque(maxlen=10)
                self._init_reasoning()

        r = R()
        result = r._call_llm("prompt")
        self.assertEqual(result, "reasoning-llm")
        mock_maki.chat.assert_called_once_with("prompt")


# ---------------------------------------------------------------------------
# MakiOpenAI: historical images → image_url content blocks (§1.8)
# ---------------------------------------------------------------------------

class TestMakiOpenAIHistoricalImages(unittest.TestCase):

    def _make_llm(self):
        from maki.makiOpenAI import MakiOpenAI
        llm = MakiOpenAI.__new__(MakiOpenAI)
        llm.model = "gpt-4o-mini"
        llm.config = GenerationConfig()
        llm.temperature = 0.7
        llm.system_prompt = None
        llm.timeout = 120
        llm._rate_limiter = None
        llm._client = MagicMock()
        llm._async_client = MagicMock()
        return llm

    def test_historical_user_message_with_image_converted(self):
        llm = self._make_llm()
        history = [
            Message("user", "What is in this image?", images=["b64data"]),
            Message("assistant", "A cat"),
        ]
        msgs = llm._build_messages("follow-up", history=history)
        user_hist = msgs[0]
        self.assertEqual(user_hist["role"], "user")
        self.assertIsInstance(user_hist["content"], list)
        self.assertEqual(user_hist["content"][0]["type"], "text")
        self.assertEqual(user_hist["content"][1]["type"], "image_url")
        self.assertIn("b64data", user_hist["content"][1]["image_url"]["url"])

    def test_historical_user_message_without_image_is_plain_string(self):
        llm = self._make_llm()
        history = [Message("user", "Hello")]
        msgs = llm._build_messages("follow-up", history=history)
        self.assertEqual(msgs[0]["content"], "Hello")

    def test_historical_assistant_message_has_no_images_key(self):
        llm = self._make_llm()
        history = [Message("assistant", "I see a cat", images=None)]
        msgs = llm._build_messages("ok", history=history)
        # assistant messages have no images; content is a plain string
        self.assertEqual(msgs[0]["content"], "I see a cat")


# ---------------------------------------------------------------------------
# MakiAnthropic: historical images → Anthropic content blocks (§1.8)
# ---------------------------------------------------------------------------

class TestMakiAnthropicHistoricalImages(unittest.TestCase):

    def _make_llm(self):
        from maki.makiAnthropic import MakiAnthropic
        llm = MakiAnthropic.__new__(MakiAnthropic)
        llm.model = "claude-sonnet-4-6"
        llm.config = GenerationConfig()
        llm.temperature = 0.7
        llm.system_prompt = None
        llm.timeout = 120
        llm._rate_limiter = None
        llm._client = MagicMock()
        llm._async_client = MagicMock()
        return llm

    def test_historical_user_image_converted_to_content_block(self):
        llm = self._make_llm()
        history = [
            Message("user", "Describe this", images=["imgdata"]),
            Message("assistant", "A dog"),
        ]
        msgs = llm._build_messages("more", history=history)
        user_hist = msgs[0]
        self.assertEqual(user_hist["role"], "user")
        self.assertIsInstance(user_hist["content"], list)
        self.assertEqual(user_hist["content"][0]["type"], "text")
        self.assertEqual(user_hist["content"][1]["type"], "image")
        self.assertEqual(user_hist["content"][1]["source"]["data"], "imgdata")

    def test_historical_system_message_dropped(self):
        llm = self._make_llm()
        history = [
            Message("system", "You are helpful"),
            Message("user", "Hi"),
        ]
        msgs = llm._build_messages("ok", history=history)
        roles = [m["role"] for m in msgs if isinstance(m.get("content"), str)]
        self.assertNotIn("system", roles)

    def test_historical_user_without_image_is_plain_string(self):
        llm = self._make_llm()
        history = [Message("user", "Plain text")]
        msgs = llm._build_messages("next", history=history)
        self.assertEqual(msgs[0]["content"], "Plain text")


# ---------------------------------------------------------------------------
# GenerationConfig: reasoning model kwargs (§1.9)
# ---------------------------------------------------------------------------

class TestGenerationConfigReasoningKwargs(unittest.TestCase):

    def test_chat_family_returns_standard_kwargs(self):
        cfg = GenerationConfig(temperature=0.5, max_tokens=100)
        kwargs = cfg.to_openai_kwargs(model_family="chat")
        self.assertIn("temperature", kwargs)
        self.assertIn("top_p", kwargs)
        self.assertIn("max_tokens", kwargs)
        self.assertNotIn("max_completion_tokens", kwargs)

    def test_reasoning_family_omits_temperature_and_top_p(self):
        cfg = GenerationConfig(max_tokens=500)
        kwargs = cfg.to_openai_kwargs(model_family="reasoning")
        self.assertNotIn("temperature", kwargs)
        self.assertNotIn("top_p", kwargs)
        self.assertNotIn("max_tokens", kwargs)

    def test_reasoning_family_uses_max_completion_tokens(self):
        cfg = GenerationConfig(max_tokens=1024)
        kwargs = cfg.to_openai_kwargs(model_family="reasoning")
        self.assertEqual(kwargs["max_completion_tokens"], 1024)

    def test_default_is_chat_family(self):
        cfg = GenerationConfig()
        kwargs = cfg.to_openai_kwargs()
        self.assertIn("temperature", kwargs)
        self.assertIn("max_tokens", kwargs)

    def test_o3_model_family_detected(self):
        from maki.makiOpenAI import MakiOpenAI
        llm = MakiOpenAI.__new__(MakiOpenAI)
        llm.model = "o3"
        self.assertEqual(llm._model_family, "reasoning")

    def test_o1_model_family_detected(self):
        from maki.makiOpenAI import MakiOpenAI
        llm = MakiOpenAI.__new__(MakiOpenAI)
        llm.model = "o1-mini"
        self.assertEqual(llm._model_family, "reasoning")

    def test_gpt4o_model_family_is_chat(self):
        from maki.makiOpenAI import MakiOpenAI
        llm = MakiOpenAI.__new__(MakiOpenAI)
        llm.model = "gpt-4o"
        self.assertEqual(llm._model_family, "chat")

    def test_o3_round_trip_does_not_send_temperature(self):
        """Regression §1.9: o3().chat() must not send temperature to the API."""
        from maki.makiOpenAI import MakiOpenAI
        import maki.makiOpenAI as mod
        # Patch the SDK so MakiOpenAI.__new__ + direct attribute injection works.
        original_sdk = mod._openai_sdk
        mock_sdk = MagicMock()
        mock_sdk.APITimeoutError = type("APITimeoutError", (Exception,), {})
        mock_sdk.APIConnectionError = type("APIConnectionError", (Exception,), {})
        mock_sdk.APIStatusError = type("APIStatusError", (Exception,), {})
        mod._openai_sdk = mock_sdk
        self.addCleanup(setattr, mod, "_openai_sdk", original_sdk)

        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "answer"
        resp.model = "o3"
        resp.usage = MagicMock(prompt_tokens=5, completion_tokens=10, total_tokens=15)

        llm = MakiOpenAI.__new__(MakiOpenAI)
        llm.model = "o3"
        llm.config = GenerationConfig()
        llm.temperature = llm.config.temperature
        llm.system_prompt = None
        llm.timeout = 120
        llm._rate_limiter = None
        llm._client = MagicMock()
        llm._async_client = MagicMock()
        llm._client.chat.completions.create.return_value = resp

        llm.chat("solve this")
        call_kwargs = llm._client.chat.completions.create.call_args.kwargs
        self.assertNotIn("temperature", call_kwargs)
        self.assertNotIn("top_p", call_kwargs)
        self.assertIn("max_completion_tokens", call_kwargs)


# ---------------------------------------------------------------------------
# GenerationConfig: parameter validation (§5)
# ---------------------------------------------------------------------------

class TestGenerationConfigValidation(unittest.TestCase):

    def test_valid_defaults_do_not_raise(self):
        cfg = GenerationConfig()
        self.assertIsNotNone(cfg)

    def test_top_p_above_one_raises(self):
        with self.assertRaises(ValueError):
            GenerationConfig(top_p=1.5)

    def test_top_p_below_zero_raises(self):
        with self.assertRaises(ValueError):
            GenerationConfig(top_p=-0.1)

    def test_top_p_zero_is_valid(self):
        cfg = GenerationConfig(top_p=0.0)
        self.assertEqual(cfg.top_p, 0.0)

    def test_top_p_one_is_valid(self):
        cfg = GenerationConfig(top_p=1.0)
        self.assertEqual(cfg.top_p, 1.0)

    def test_top_k_negative_raises(self):
        with self.assertRaises(ValueError):
            GenerationConfig(top_k=-1)

    def test_top_k_zero_is_valid(self):
        cfg = GenerationConfig(top_k=0)
        self.assertEqual(cfg.top_k, 0)

    def test_max_tokens_zero_raises(self):
        with self.assertRaises(ValueError):
            GenerationConfig(max_tokens=0)

    def test_max_tokens_negative_raises(self):
        with self.assertRaises(ValueError):
            GenerationConfig(max_tokens=-100)

    def test_max_tokens_one_is_valid(self):
        cfg = GenerationConfig(max_tokens=1)
        self.assertEqual(cfg.max_tokens, 1)


if __name__ == "__main__":
    unittest.main()
