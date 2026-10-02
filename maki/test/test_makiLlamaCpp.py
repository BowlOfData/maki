#!/usr/bin/env python3
"""Unit tests for MakiLlamaCpp — OpenAI SDK calls and the Connector are mocked.

Message building, vision and tool-calling are inherited from MakiOpenAI and
covered by test_makiOpenAI.py. This file covers what MakiLlamaCpp adds:
client construction (base_url, optional API key), llama.cpp sampling
parameters, reasoning capture, chat_collect streaming, the native
endpoints (health, embeddings, rerank) and the config-loader entry.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from maki.exceptions import MakiAPIError, MakiNetworkError, MakiValidationError
from maki.objects import BackendType, GenerationConfig, LLMResponse


def _make_response(content="Hi", reasoning=None, model="gemma"):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.choices[0].message.reasoning_content = reasoning
    resp.model = model
    resp.usage.prompt_tokens = 5
    resp.usage.completion_tokens = 7
    resp.usage.total_tokens = 12
    return resp


def _make_chunk(content=None, reasoning=None, usage=None, model="gemma", choices=True):
    chunk = MagicMock()
    chunk.model = model
    chunk.usage = usage
    if choices:
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = content
        chunk.choices[0].delta.reasoning_content = reasoning
    else:
        chunk.choices = []
    return chunk


def _http_response(status=200, body=None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = body or {}
    return r


class TestMakiLlamaCpp(unittest.TestCase):

    def setUp(self):
        import maki.makiOpenAI as openai_mod
        original_sdk = openai_mod._openai_sdk
        self.sdk = MagicMock()
        self.sdk.APITimeoutError = type("APITimeoutError", (Exception,), {})
        self.sdk.APIConnectionError = type("APIConnectionError", (Exception,), {})
        self.sdk.APIStatusError = type("APIStatusError", (Exception,), {"status_code": 500})
        openai_mod._openai_sdk = self.sdk
        self.addCleanup(setattr, openai_mod, "_openai_sdk", original_sdk)
        env = patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("LLAMACPP_API_KEY", None)

    def _make_llm(self, **kwargs):
        from maki.makiLlamaCpp import MakiLlamaCpp
        llm = MakiLlamaCpp(**kwargs)
        llm._client = MagicMock()
        llm._http = MagicMock()
        return llm

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def test_missing_sdk_raises_import_error(self):
        import maki.makiOpenAI as openai_mod
        openai_mod._openai_sdk = None
        from maki.makiLlamaCpp import MakiLlamaCpp
        with self.assertRaises(ImportError):
            MakiLlamaCpp()

    def test_defaults_need_no_api_key(self):
        from maki.makiLlamaCpp import MakiLlamaCpp
        llm = MakiLlamaCpp()
        kwargs = self.sdk.OpenAI.call_args.kwargs
        self.assertEqual(kwargs["base_url"], "http://127.0.0.1:8080/v1")
        self.assertEqual(kwargs["api_key"], "no-key")
        self.assertEqual(llm.model, "local-model")
        self.assertEqual(llm._headers(), {})

    def test_api_key_from_env_is_used_everywhere(self):
        os.environ["LLAMACPP_API_KEY"] = "secret"
        from maki.makiLlamaCpp import MakiLlamaCpp
        llm = MakiLlamaCpp()
        self.assertEqual(self.sdk.OpenAI.call_args.kwargs["api_key"], "secret")
        self.assertEqual(self.sdk.AsyncOpenAI.call_args.kwargs["api_key"], "secret")
        self.assertEqual(llm._headers(), {"Authorization": "Bearer secret"})

    def test_custom_base_url_trailing_slash_stripped(self):
        from maki.makiLlamaCpp import MakiLlamaCpp
        llm = MakiLlamaCpp(base_url="http://192.168.1.20:9000/v1/")
        self.assertEqual(llm.base_url, "http://192.168.1.20:9000/v1")
        self.assertEqual(llm._server_root, "http://192.168.1.20:9000")

    def test_invalid_base_url_rejected(self):
        from maki.makiLlamaCpp import MakiLlamaCpp
        with self.assertRaises(MakiValidationError):
            MakiLlamaCpp(base_url="file:///etc/passwd")

    def test_exported_from_package(self):
        import maki
        from maki.makiLlamaCpp import MakiLlamaCpp
        self.assertIs(maki.MakiLlamaCpp, MakiLlamaCpp)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def test_reasoning_style_model_name_keeps_sampling_params(self):
        llm = self._make_llm(model="o3-mini-gguf")
        kwargs = llm._request_kwargs(GenerationConfig(temperature=0.3))
        self.assertEqual(kwargs["temperature"], 0.3)
        self.assertIn("max_tokens", kwargs)

    def test_chat_sends_llamacpp_sampling_params(self):
        llm = self._make_llm()
        llm._client.chat.completions.create.return_value = _make_response()
        llm.chat("hello", config=GenerationConfig(top_k=12, repeat_penalty=1.3))
        kwargs = llm._client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["extra_body"], {"top_k": 12, "repeat_penalty": 1.3})

    def test_chat_stamps_backend_and_reasoning(self):
        llm = self._make_llm()
        llm._client.chat.completions.create.return_value = _make_response(
            content="Paris", reasoning="The capital is Paris."
        )
        result = llm.chat("capital of France?")
        self.assertEqual(result.content, "Paris")
        self.assertEqual(result.reasoning, "The capital is Paris.")
        self.assertEqual(result.backend, BackendType.LLAMACPP)
        self.assertEqual(result.total_tokens, 12)

    def test_chat_without_reasoning_leaves_none(self):
        llm = self._make_llm()
        llm._client.chat.completions.create.return_value = _make_response()
        self.assertIsNone(llm.chat("hi").reasoning)

    def test_tool_calls_send_extra_body(self):
        llm = self._make_llm()
        resp = _make_response(content="done")
        resp.choices[0].message.tool_calls = None
        resp.choices[0].finish_reason = "stop"
        llm._client.chat.completions.create.return_value = resp
        result, calls, _ = llm.chat_with_tools([{"role": "user", "content": "hi"}], [])
        self.assertIsNone(calls)
        self.assertEqual(result.backend, BackendType.LLAMACPP)
        self.assertIn("extra_body", llm._client.chat.completions.create.call_args.kwargs)

    def test_chat_collect_joins_stream(self):
        llm = self._make_llm()
        usage = MagicMock(prompt_tokens=3, completion_tokens=4, total_tokens=7)
        chunks = [
            _make_chunk(reasoning="think "),
            _make_chunk(reasoning="more"),
            _make_chunk(content="Hel"),
            _make_chunk(content="lo"),
            _make_chunk(usage=usage, choices=False),
        ]
        stream = MagicMock()
        stream.__enter__.return_value = iter(chunks)
        llm._client.chat.completions.create.return_value = stream

        result = llm.chat_collect("hi")

        self.assertEqual(result.content, "Hello")
        self.assertEqual(result.reasoning, "think more")
        self.assertEqual(result.total_tokens, 7)
        self.assertEqual(result.model, "gemma")
        kwargs = llm._client.chat.completions.create.call_args.kwargs
        self.assertTrue(kwargs["stream"])
        self.assertEqual(kwargs["stream_options"], {"include_usage": True})

    def test_chat_collect_maps_connection_error(self):
        llm = self._make_llm()
        llm._client.chat.completions.create.side_effect = self.sdk.APIConnectionError("down")
        with self.assertRaises(MakiNetworkError):
            llm.chat_collect("hi")

    # ------------------------------------------------------------------
    # Native endpoints
    # ------------------------------------------------------------------

    def test_health_true_on_200(self):
        llm = self._make_llm()
        llm._http.get.return_value = _http_response(200)
        self.assertTrue(llm.health())
        self.assertEqual(llm._http.get.call_args.args[0], "http://127.0.0.1:8080/health")

    def test_health_false_while_loading(self):
        llm = self._make_llm()
        llm._http.get.return_value = _http_response(503)
        self.assertFalse(llm.health())

    def test_health_false_when_unreachable(self):
        llm = self._make_llm()
        llm._http.get.side_effect = MakiNetworkError("refused")
        self.assertFalse(llm.health())

    def test_embed_batch_orders_by_index(self):
        llm = self._make_llm()
        llm._http.post.return_value = _http_response(body={"data": [
            {"index": 1, "embedding": [0.2]},
            {"index": 0, "embedding": [0.1]},
        ]})
        self.assertEqual(llm.embed_batch(["a", "b"]), [[0.1], [0.2]])
        call = llm._http.post.call_args
        self.assertEqual(call.args[0], "http://127.0.0.1:8080/v1/embeddings")
        self.assertEqual(call.kwargs["json"]["input"], ["a", "b"])

    def test_embed_single_and_empty(self):
        llm = self._make_llm()
        llm._http.post.return_value = _http_response(body={"data": [{"index": 0, "embedding": [1.0, 2.0]}]})
        self.assertEqual(llm.embed("a"), [1.0, 2.0])
        llm._http.post.return_value = _http_response(body={"data": []})
        with self.assertRaises(MakiAPIError):
            llm.embed("a")

    def test_rerank_sorted_with_documents(self):
        llm = self._make_llm()
        docs = ["hi", "it is a bear", "The giant panda is a bear species."]
        llm._http.post.return_value = _http_response(body={"results": [
            {"index": 0, "relevance_score": 0.001},
            {"index": 2, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.05},
        ]})
        ranked = llm.rerank("What is panda?", docs, top_n=2)
        self.assertEqual([r["index"] for r in ranked], [2, 1, 0])
        self.assertEqual(ranked[0]["document"], docs[2])
        call = llm._http.post.call_args
        self.assertEqual(call.args[0], "http://127.0.0.1:8080/v1/rerank")
        self.assertEqual(call.kwargs["json"]["top_n"], 2)


class TestLLMResponseReasoning(unittest.TestCase):

    def test_roundtrip_keeps_reasoning(self):
        resp = LLMResponse(content="x", model="m", prompt_tokens=0, completion_tokens=0,
                           total_tokens=0, elapsed_seconds=0.0,
                           backend=BackendType.LLAMACPP, reasoning="because")
        restored = LLMResponse.from_dict(resp.to_dict())
        self.assertEqual(restored.reasoning, "because")
        self.assertEqual(restored.backend, BackendType.LLAMACPP)

    def test_from_dict_without_reasoning(self):
        data = LLMResponse(content="x", model="m", prompt_tokens=0, completion_tokens=0,
                           total_tokens=0, elapsed_seconds=0.0).to_dict()
        del data["reasoning"]
        self.assertIsNone(LLMResponse.from_dict(data).reasoning)


class TestConfigLoaderBackends(unittest.TestCase):

    def setUp(self):
        __import__("pytest").importorskip("yaml", reason="PyYAML not installed")

    def test_llamacpp_backend_with_base_url(self):
        from maki.distributed.config_loader import _build_backend
        with patch("maki.makiLlamaCpp.MakiLlamaCpp") as cls:
            _build_backend({"backend": "llamacpp", "model": "gemma",
                            "base_url": "http://10.0.0.5:8080/v1"})
        kwargs = cls.call_args.kwargs
        self.assertEqual(kwargs["model"], "gemma")
        self.assertEqual(kwargs["base_url"], "http://10.0.0.5:8080/v1")

    def test_openrouter_backend(self):
        from maki.distributed.config_loader import _build_backend
        with patch("maki.makiOpenRouter.MakiOpenRouter") as cls:
            _build_backend({"backend": "openrouter"})
        cls.assert_called_once()

    def test_unknown_backend_lists_new_values(self):
        from maki.distributed.config_loader import _build_backend
        with self.assertRaisesRegex(ValueError, "llamacpp"):
            _build_backend({"backend": "nope"})


if __name__ == "__main__":
    unittest.main()
