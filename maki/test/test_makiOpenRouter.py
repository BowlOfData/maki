#!/usr/bin/env python3
"""Unit tests for MakiOpenRouter — all OpenAI SDK calls are mocked.

Most behavior (message building, vision, tool-calling, GenerationConfig
serialisation) is inherited unchanged from MakiOpenAI and is already
exhaustively covered by test_makiOpenAI.py. This file focuses on what
MakiOpenRouter actually overrides (client construction/base_url,
attribution headers, OPENROUTER_API_KEY resolution, the vendor-namespaced
model-family check, BackendType stamping) plus enough smoke tests to
confirm inheritance works end to end through the subclass.
"""

import asyncio
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from maki.objects import BackendType, GenerationConfig
from maki.exceptions import MakiAPIError, MakiNetworkError


def _make_openai_response(
    content="Test response", model="openai/gpt-4o-mini",
    prompt_tokens=10, completion_tokens=20,
):
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


class TestMakiOpenRouter(unittest.TestCase):

    def _patch_sdk(self):
        """Patch the shared maki.makiOpenAI._openai_sdk guard, restored on cleanup.

        MakiOpenRouter reads this attribute live off the makiOpenAI module
        rather than importing its own copy, so patching it here is enough
        to cover both the guard check and the OpenAI/AsyncOpenAI client
        construction.
        """
        import maki.makiOpenAI as openai_mod
        original_sdk = openai_mod._openai_sdk
        mock_sdk = MagicMock()
        mock_sdk.APITimeoutError = type("APITimeoutError", (Exception,), {})
        mock_sdk.APIConnectionError = type("APIConnectionError", (Exception,), {})
        mock_sdk.APIStatusError = type("APIStatusError", (Exception,), {})
        openai_mod._openai_sdk = mock_sdk
        self.addCleanup(setattr, openai_mod, "_openai_sdk", original_sdk)
        return mock_sdk

    def _make_llm(self, **kwargs):
        """Instantiate MakiOpenRouter with the SDK and client pre-mocked (bypasses __init__)."""
        self._patch_sdk()
        from maki.makiOpenRouter import MakiOpenRouter
        llm = MakiOpenRouter.__new__(MakiOpenRouter)
        llm.model = kwargs.get("model", "openai/gpt-4o-mini")
        llm.config = kwargs.get("config", GenerationConfig())
        llm.temperature = llm.config.temperature
        llm.system_prompt = kwargs.get("system_prompt")
        llm.timeout = kwargs.get("timeout", 120)
        llm.base_url = kwargs.get("base_url", "https://openrouter.ai/api/v1")
        llm._rate_limiter = None
        llm._client = MagicMock()
        llm._async_client = MagicMock()
        return llm

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    def test_missing_sdk_raises_import_error(self):
        import maki.makiOpenAI as openai_mod
        original = openai_mod._openai_sdk
        try:
            openai_mod._openai_sdk = None
            from maki.makiOpenRouter import MakiOpenRouter
            with self.assertRaises(ImportError):
                MakiOpenRouter(api_key="sk-or-test")
        finally:
            openai_mod._openai_sdk = original

    def test_missing_api_key_raises_value_error(self):
        self._patch_sdk()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENROUTER_API_KEY", None)
            from maki.makiOpenRouter import MakiOpenRouter
            with self.assertRaises(ValueError):
                MakiOpenRouter()

    def test_base_url_default(self):
        mock_sdk = self._patch_sdk()
        from maki.makiOpenRouter import MakiOpenRouter
        MakiOpenRouter(api_key="sk-or-test")
        self.assertEqual(mock_sdk.OpenAI.call_args.kwargs["base_url"], "https://openrouter.ai/api/v1")
        self.assertEqual(mock_sdk.AsyncOpenAI.call_args.kwargs["base_url"], "https://openrouter.ai/api/v1")

    def test_base_url_custom(self):
        mock_sdk = self._patch_sdk()
        from maki.makiOpenRouter import MakiOpenRouter
        MakiOpenRouter(api_key="sk-or-test", base_url="https://custom.example.com/v1")
        self.assertEqual(mock_sdk.OpenAI.call_args.kwargs["base_url"], "https://custom.example.com/v1")

    def test_api_key_resolved_from_env(self):
        mock_sdk = self._patch_sdk()
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-env"}):
            from maki.makiOpenRouter import MakiOpenRouter
            MakiOpenRouter()
            self.assertEqual(mock_sdk.OpenAI.call_args.kwargs["api_key"], "sk-or-env")

    def test_default_headers_from_explicit_params(self):
        mock_sdk = self._patch_sdk()
        from maki.makiOpenRouter import MakiOpenRouter
        MakiOpenRouter(api_key="sk-or-test", site_url="https://example.com", app_name="MyApp")
        self.assertEqual(
            mock_sdk.OpenAI.call_args.kwargs["default_headers"],
            {"HTTP-Referer": "https://example.com", "X-Title": "MyApp"},
        )

    def test_default_headers_from_env_fallback(self):
        mock_sdk = self._patch_sdk()
        env = {"OPENROUTER_SITE_URL": "https://env.example.com", "OPENROUTER_APP_NAME": "EnvApp"}
        with patch.dict(os.environ, env):
            from maki.makiOpenRouter import MakiOpenRouter
            MakiOpenRouter(api_key="sk-or-test")
            self.assertEqual(
                mock_sdk.OpenAI.call_args.kwargs["default_headers"],
                {"HTTP-Referer": "https://env.example.com", "X-Title": "EnvApp"},
            )

    def test_no_default_headers_when_unset(self):
        mock_sdk = self._patch_sdk()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENROUTER_SITE_URL", None)
            os.environ.pop("OPENROUTER_APP_NAME", None)
            from maki.makiOpenRouter import MakiOpenRouter
            MakiOpenRouter(api_key="sk-or-test")
            self.assertNotIn("default_headers", mock_sdk.OpenAI.call_args.kwargs)

    # ------------------------------------------------------------------
    # chat() / BackendType
    # ------------------------------------------------------------------

    def test_chat_happy_path(self):
        llm = self._make_llm()
        llm._client.chat.completions.create.return_value = _make_openai_response(content="Paris")
        response = llm.chat("What is the capital of France?")
        self.assertEqual(response.content, "Paris")
        self.assertEqual(response.backend, BackendType.OPENROUTER)

    def test_request_routes_to_chat(self):
        llm = self._make_llm()
        llm._client.chat.completions.create.return_value = _make_openai_response()
        response = llm.request("Hello")
        self.assertEqual(response.backend, BackendType.OPENROUTER)

    # ------------------------------------------------------------------
    # _model_family (vendor-namespace stripping)
    # ------------------------------------------------------------------

    def test_model_family_strips_vendor_prefix_reasoning(self):
        llm = self._make_llm(model="openai/o1-preview")
        self.assertEqual(llm._model_family, "reasoning")

    def test_model_family_strips_vendor_prefix_chat(self):
        llm = self._make_llm(model="openai/gpt-4o-mini")
        self.assertEqual(llm._model_family, "chat")

    def test_model_family_unnamespaced_model(self):
        llm = self._make_llm(model="o3")
        self.assertEqual(llm._model_family, "reasoning")

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
        self.assertEqual(response.backend, BackendType.OPENROUTER)

    # ------------------------------------------------------------------
    # session()
    # ------------------------------------------------------------------

    def test_session_accumulates_history(self):
        llm = self._make_llm()
        llm._client.chat.completions.create.return_value = _make_openai_response(content="Reply")
        session = llm.session()
        session.say("Turn 1")
        self.assertEqual(len(session.history), 2)
        session.say("Turn 2")
        self.assertEqual(len(session.history), 4)

    # ------------------------------------------------------------------
    # Error mapping (inherited from MakiOpenAI)
    # ------------------------------------------------------------------

    def test_network_error_propagates(self):
        llm = self._make_llm()
        llm._client.chat.completions.create.side_effect = MakiNetworkError("mocked")
        with self.assertRaises(MakiNetworkError):
            llm.chat("test")

    def test_api_error_propagates(self):
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
    # __repr__
    # ------------------------------------------------------------------

    def test_repr(self):
        llm = self._make_llm(model="anthropic/claude-sonnet-4-6")
        r = repr(llm)
        self.assertIn("anthropic/claude-sonnet-4-6", r)
        self.assertIn("openrouter.ai", r)


if __name__ == "__main__":
    unittest.main()
