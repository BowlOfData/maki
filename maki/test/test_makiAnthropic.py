#!/usr/bin/env python3
"""Unit tests for MakiAnthropic — all Anthropic SDK calls are mocked."""

import asyncio
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from maki.objects import BackendType, GenerationConfig, Message
from maki.exceptions import MakiAPIError, MakiNetworkError, MakiTimeoutError


def _make_anthropic_response(text="Test response", model="claude-sonnet-4-6",
                              input_tokens=10, output_tokens=20):
    resp = MagicMock()
    resp.content = [MagicMock()]
    resp.content[0].text = text
    resp.model = model
    resp.usage = MagicMock()
    resp.usage.input_tokens = input_tokens
    resp.usage.output_tokens = output_tokens
    return resp


class TestMakiAnthropic(unittest.TestCase):

    def _make_llm(self, **kwargs):
        """Instantiate MakiAnthropic with the SDK and client pre-mocked."""
        import maki.makiAnthropic as mod
        original_sdk = mod._anthropic_sdk
        mock_sdk = MagicMock()
        mock_sdk.APITimeoutError = type("APITimeoutError", (Exception,), {})
        mock_sdk.APIConnectionError = type("APIConnectionError", (Exception,), {})
        mock_sdk.APIStatusError = type("APIStatusError", (Exception,), {})
        mod._anthropic_sdk = mock_sdk
        self.addCleanup(setattr, mod, "_anthropic_sdk", original_sdk)

        from maki.makiAnthropic import MakiAnthropic
        llm = MakiAnthropic.__new__(MakiAnthropic)
        llm.model = kwargs.get("model", "claude-sonnet-4-6")
        llm.config = kwargs.get("config", GenerationConfig())
        llm.temperature = llm.config.temperature
        llm.system_prompt = kwargs.get("system_prompt")
        llm.timeout = kwargs.get("timeout", 120)
        llm._rate_limiter = None
        llm._client = MagicMock()
        llm._async_client = MagicMock()
        return llm

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    def test_missing_sdk_raises_import_error(self):
        import maki.makiAnthropic as mod
        original = mod._anthropic_sdk
        try:
            mod._anthropic_sdk = None
            from maki.makiAnthropic import MakiAnthropic
            with self.assertRaises(ImportError):
                MakiAnthropic(api_key="sk-ant-test")
        finally:
            mod._anthropic_sdk = original

    def test_missing_api_key_raises_value_error(self):
        import maki.makiAnthropic as mod
        with patch.object(mod, "_anthropic_sdk", MagicMock()):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            from maki.makiAnthropic import MakiAnthropic
            with self.assertRaises(ValueError):
                MakiAnthropic()

    # ------------------------------------------------------------------
    # request() / chat()
    # ------------------------------------------------------------------

    def test_request_routes_to_chat(self):
        llm = self._make_llm()
        llm._client.messages.create.return_value = _make_anthropic_response()
        response = llm.request("Hello")
        self.assertEqual(response.content, "Test response")
        self.assertEqual(response.backend, BackendType.ANTHROPIC)

    def test_request_rejects_empty_prompt(self):
        llm = self._make_llm()
        with self.assertRaises(ValueError):
            llm.request("   ")

    def test_chat_happy_path(self):
        llm = self._make_llm()
        llm._client.messages.create.return_value = _make_anthropic_response(
            text="Paris", input_tokens=5, output_tokens=1
        )
        response = llm.chat("What is the capital of France?")
        self.assertEqual(response.content, "Paris")
        self.assertEqual(response.model, "claude-sonnet-4-6")
        self.assertEqual(response.prompt_tokens, 5)
        self.assertEqual(response.completion_tokens, 1)
        self.assertEqual(response.total_tokens, 6)
        self.assertEqual(response.backend, BackendType.ANTHROPIC)

    def test_chat_system_prompt_is_top_level(self):
        """Anthropic system prompt must NOT appear in the messages list."""
        llm = self._make_llm(system_prompt="Be concise.")
        llm._client.messages.create.return_value = _make_anthropic_response()
        llm.chat("Hello")
        call_kwargs = llm._client.messages.create.call_args.kwargs
        self.assertEqual(call_kwargs["system"], "Be concise.")
        for msg in call_kwargs["messages"]:
            self.assertNotEqual(msg.get("role"), "system")

    def test_chat_overrides_system_prompt_per_call(self):
        llm = self._make_llm(system_prompt="Default system.")
        llm._client.messages.create.return_value = _make_anthropic_response()
        llm.chat("Hello", system="Override system.")
        call_kwargs = llm._client.messages.create.call_args.kwargs
        self.assertEqual(call_kwargs["system"], "Override system.")

    def test_chat_with_history(self):
        llm = self._make_llm()
        llm._client.messages.create.return_value = _make_anthropic_response(text="Sure!")
        history = [Message("user", "Hi"), Message("assistant", "Hello")]
        llm.chat("How are you?", history=history)
        messages = llm._client.messages.create.call_args.kwargs["messages"]
        self.assertEqual(messages[0]["content"], "Hi")
        self.assertEqual(messages[1]["content"], "Hello")
        self.assertEqual(messages[2]["content"], "How are you?")

    def test_chat_history_excludes_system_role(self):
        """System messages in history must be silently dropped (Anthropic constraint)."""
        llm = self._make_llm()
        llm._client.messages.create.return_value = _make_anthropic_response()
        history = [
            Message("system", "Some old system msg"),
            Message("user", "Hi"),
            Message("assistant", "Hello"),
        ]
        llm.chat("Continue", history=history)
        messages = llm._client.messages.create.call_args.kwargs["messages"]
        roles = [m["role"] for m in messages]
        self.assertNotIn("system", roles)

    def test_chat_with_image(self):
        llm = self._make_llm()
        llm._client.messages.create.return_value = _make_anthropic_response(text="A cat")
        llm.chat("What is in the image?", images=["base64encodeddata"])
        messages = llm._client.messages.create.call_args.kwargs["messages"]
        user_content = messages[0]["content"]
        self.assertIsInstance(user_content, list)
        self.assertEqual(user_content[0]["type"], "text")
        self.assertEqual(user_content[1]["type"], "image")
        self.assertEqual(user_content[1]["source"]["type"], "base64")
        self.assertEqual(user_content[1]["source"]["data"], "base64encodeddata")

    # ------------------------------------------------------------------
    # stream()
    # ------------------------------------------------------------------

    def test_stream_yields_chunks(self):
        llm = self._make_llm()
        stream_ctx = MagicMock()
        stream_ctx.__enter__ = MagicMock(return_value=stream_ctx)
        stream_ctx.__exit__ = MagicMock(return_value=False)
        stream_ctx.text_stream = iter(["Hello", " World"])
        llm._client.messages.stream.return_value = stream_ctx
        result = list(llm.stream("Test"))
        self.assertEqual(result, ["Hello", " World"])

    # ------------------------------------------------------------------
    # async_chat()
    # ------------------------------------------------------------------

    def test_async_chat_happy_path(self):
        llm = self._make_llm()

        async def _mock_create(**kwargs):
            return _make_anthropic_response(text="Async response")

        llm._async_client.messages.create = _mock_create
        response = asyncio.run(llm.async_chat("Hello"))
        self.assertEqual(response.content, "Async response")
        self.assertEqual(response.backend, BackendType.ANTHROPIC)

    # ------------------------------------------------------------------
    # session()
    # ------------------------------------------------------------------

    def test_session_creation(self):
        llm = self._make_llm()
        session = llm.session(system="Be concise.")
        self.assertIsNotNone(session)
        self.assertTrue(hasattr(session, "say"))
        self.assertTrue(hasattr(session, "reset"))
        self.assertTrue(hasattr(session, "history"))

    def test_session_accumulates_history(self):
        llm = self._make_llm()
        llm._client.messages.create.return_value = _make_anthropic_response(text="Reply")
        session = llm.session()
        session.say("Turn 1")
        self.assertEqual(len(session.history), 2)
        session.say("Turn 2")
        self.assertEqual(len(session.history), 4)

    # ------------------------------------------------------------------
    # Error mapping
    # ------------------------------------------------------------------

    def test_network_error_maps_to_maki_network_error(self):
        llm = self._make_llm()
        llm._client.messages.create.side_effect = MakiNetworkError("mocked")
        with self.assertRaises(MakiNetworkError):
            llm.chat("test")

    def test_status_error_maps_to_maki_api_error(self):
        llm = self._make_llm()
        llm._client.messages.create.side_effect = MakiAPIError("mocked")
        with self.assertRaises(MakiAPIError):
            llm.chat("test")

    # ------------------------------------------------------------------
    # Rate limiter
    # ------------------------------------------------------------------

    def test_rate_limiter_acquire_called(self):
        llm = self._make_llm()
        mock_limiter = MagicMock()
        llm._rate_limiter = mock_limiter
        llm._client.messages.create.return_value = _make_anthropic_response()
        llm.chat("Hello")
        mock_limiter.acquire.assert_called_once()

    # ------------------------------------------------------------------
    # GenerationConfig serialisation
    # ------------------------------------------------------------------

    def test_generation_config_to_anthropic_kwargs(self):
        cfg = GenerationConfig(temperature=0.3, max_tokens=512)
        kwargs = cfg.to_anthropic_kwargs()
        self.assertEqual(kwargs["temperature"], 0.3)
        self.assertEqual(kwargs["max_tokens"], 512)
        self.assertNotIn("repeat_penalty", kwargs)

    def test_generation_config_stop_sequences(self):
        cfg = GenerationConfig(stop=["END"])
        kwargs = cfg.to_anthropic_kwargs()
        self.assertEqual(kwargs["stop_sequences"], ["END"])

    def test_generation_config_no_stop_when_empty(self):
        cfg = GenerationConfig(stop=[])
        kwargs = cfg.to_anthropic_kwargs()
        self.assertNotIn("stop_sequences", kwargs)

    # ------------------------------------------------------------------
    # __repr__
    # ------------------------------------------------------------------

    def test_repr(self):
        llm = self._make_llm(model="claude-opus-4-8")
        self.assertIn("claude-opus-4-8", repr(llm))


if __name__ == "__main__":
    unittest.main()
