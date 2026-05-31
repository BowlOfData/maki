#!/usr/bin/env python3
"""Unit tests for MakiOpenAI — all OpenAI SDK calls are mocked."""

import asyncio
import sys
import os
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from maki.objects import BackendType, GenerationConfig, Message
from maki.exceptions import MakiAPIError, MakiNetworkError, MakiTimeoutError


def _make_openai_response(content="Test response", model="gpt-4o-mini",
                          prompt_tokens=10, completion_tokens=20):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.model = model
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    resp.usage.total_tokens = prompt_tokens + completion_tokens
    return resp


def _make_stream_chunk(delta_content):
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = delta_content
    return chunk


class TestMakiOpenAI(unittest.TestCase):

    def _make_llm(self, **kwargs):
        """Instantiate MakiOpenAI with the SDK and client pre-mocked."""
        import maki.makiOpenAI as mod
        original_sdk = mod._openai_sdk
        mock_sdk = MagicMock()
        mock_sdk.APITimeoutError = type("APITimeoutError", (Exception,), {})
        mock_sdk.APIConnectionError = type("APIConnectionError", (Exception,), {})
        mock_sdk.APIStatusError = type("APIStatusError", (Exception,), {})
        mod._openai_sdk = mock_sdk
        self.addCleanup(setattr, mod, "_openai_sdk", original_sdk)

        from maki.makiOpenAI import MakiOpenAI
        llm = MakiOpenAI.__new__(MakiOpenAI)
        llm.model = kwargs.get("model", "gpt-4o-mini")
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
        import maki.makiOpenAI as mod
        original = mod._openai_sdk
        try:
            mod._openai_sdk = None
            from maki.makiOpenAI import MakiOpenAI
            with self.assertRaises(ImportError):
                MakiOpenAI(api_key="sk-test")
        finally:
            mod._openai_sdk = original

    def test_missing_api_key_raises_value_error(self):
        import maki.makiOpenAI as mod
        with patch.object(mod, "_openai_sdk", MagicMock()):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("OPENAI_API_KEY", None)
                from maki.makiOpenAI import MakiOpenAI
                with self.assertRaises(ValueError):
                    MakiOpenAI()

    # ------------------------------------------------------------------
    # request() / chat()
    # ------------------------------------------------------------------

    def test_request_routes_to_chat(self):
        llm = self._make_llm()
        llm._client.chat.completions.create.return_value = _make_openai_response()
        response = llm.request("Hello")
        self.assertEqual(response.content, "Test response")
        self.assertEqual(response.backend, BackendType.OPENAI)

    def test_request_rejects_empty_prompt(self):
        llm = self._make_llm()
        with self.assertRaises(ValueError):
            llm.request("   ")

    def test_chat_happy_path(self):
        llm = self._make_llm()
        llm._client.chat.completions.create.return_value = _make_openai_response(
            content="Paris", prompt_tokens=5, completion_tokens=1
        )
        response = llm.chat("What is the capital of France?")
        self.assertEqual(response.content, "Paris")
        self.assertEqual(response.model, "gpt-4o-mini")
        self.assertEqual(response.prompt_tokens, 5)
        self.assertEqual(response.completion_tokens, 1)
        self.assertEqual(response.total_tokens, 6)
        self.assertEqual(response.backend, BackendType.OPENAI)

    def test_chat_with_history(self):
        llm = self._make_llm()
        llm._client.chat.completions.create.return_value = _make_openai_response(content="Sure!")
        history = [Message("user", "Hi"), Message("assistant", "Hello")]
        response = llm.chat("How are you?", history=history)
        call_args = llm._client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        self.assertEqual(messages[0]["content"], "Hi")
        self.assertEqual(messages[1]["content"], "Hello")
        self.assertEqual(messages[2]["content"], "How are you?")

    def test_chat_with_system_prompt(self):
        llm = self._make_llm(system_prompt="You are helpful.")
        llm._client.chat.completions.create.return_value = _make_openai_response()
        llm.chat("Hello")
        messages = llm._client.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "You are helpful.")

    def test_chat_with_image(self):
        llm = self._make_llm()
        llm._client.chat.completions.create.return_value = _make_openai_response(content="A cat")
        response = llm.chat("What is in the image?", images=["base64encodeddata"])
        messages = llm._client.chat.completions.create.call_args.kwargs["messages"]
        user_content = messages[0]["content"]
        self.assertIsInstance(user_content, list)
        self.assertEqual(user_content[0]["type"], "text")
        self.assertEqual(user_content[1]["type"], "image_url")
        self.assertIn("base64encodeddata", user_content[1]["image_url"]["url"])

    # ------------------------------------------------------------------
    # stream()
    # ------------------------------------------------------------------

    def test_stream_yields_chunks(self):
        llm = self._make_llm()
        chunks = [_make_stream_chunk("Hello"), _make_stream_chunk(" World")]
        stream_ctx = MagicMock()
        stream_ctx.__enter__ = MagicMock(return_value=iter(chunks))
        stream_ctx.__exit__ = MagicMock(return_value=False)
        llm._client.chat.completions.create.return_value = stream_ctx
        result = list(llm.stream("Test"))
        self.assertEqual(result, ["Hello", " World"])

    # ------------------------------------------------------------------
    # async_chat()
    # ------------------------------------------------------------------

    def test_async_chat_happy_path(self):
        llm = self._make_llm()

        async def _mock_create(**kwargs):
            return _make_openai_response(content="Async response")

        llm._async_client.chat.completions.create = _mock_create
        response = asyncio.run(llm.async_chat("Hello"))
        self.assertEqual(response.content, "Async response")
        self.assertEqual(response.backend, BackendType.OPENAI)

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
        llm._client.chat.completions.create.return_value = _make_openai_response(content="Reply")
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
        llm._client.chat.completions.create.side_effect = MakiNetworkError("mocked")
        with self.assertRaises(MakiNetworkError):
            llm.chat("test")

    def test_status_error_maps_to_maki_api_error(self):
        llm = self._make_llm()
        llm._client.chat.completions.create.side_effect = MakiAPIError("mocked")
        with self.assertRaises(MakiAPIError):
            llm.chat("test")

    # ------------------------------------------------------------------
    # Rate limiter
    # ------------------------------------------------------------------

    def test_rate_limiter_acquire_called(self):
        llm = self._make_llm()
        mock_limiter = MagicMock()
        llm._rate_limiter = mock_limiter
        llm._client.chat.completions.create.return_value = _make_openai_response()
        llm.chat("Hello")
        mock_limiter.acquire.assert_called_once()

    # ------------------------------------------------------------------
    # GenerationConfig serialisation
    # ------------------------------------------------------------------

    def test_generation_config_to_openai_kwargs(self):
        cfg = GenerationConfig(temperature=0.5, max_tokens=100, seed=42)
        kwargs = cfg.to_openai_kwargs()
        self.assertEqual(kwargs["temperature"], 0.5)
        self.assertEqual(kwargs["max_tokens"], 100)
        self.assertEqual(kwargs["seed"], 42)
        self.assertNotIn("top_k", kwargs)
        self.assertNotIn("repeat_penalty", kwargs)

    def test_generation_config_stop_sequences(self):
        cfg = GenerationConfig(stop=["END", "STOP"])
        kwargs = cfg.to_openai_kwargs()
        self.assertEqual(kwargs["stop"], ["END", "STOP"])

    def test_generation_config_no_seed_when_minus_one(self):
        cfg = GenerationConfig(seed=-1)
        kwargs = cfg.to_openai_kwargs()
        self.assertNotIn("seed", kwargs)

    # ------------------------------------------------------------------
    # __repr__
    # ------------------------------------------------------------------

    def test_repr(self):
        llm = self._make_llm(model="gpt-4o")
        self.assertIn("gpt-4o", repr(llm))


if __name__ == "__main__":
    unittest.main()
