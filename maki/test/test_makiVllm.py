#!/usr/bin/env python3
"""Unit tests for MakiVllm — OpenAI SDK calls and the Connector are mocked.

Message building, vision and tool-calling are inherited from MakiOpenAI and
covered by test_makiOpenAI.py. This file covers what MakiVllm adds:
client construction, vLLM sampling parameters, structured outputs, batch
generation, reasoning capture, the native endpoints and the config-loader entry.
"""

import asyncio
import os
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from maki.exceptions import MakiAPIError, MakiNetworkError, MakiValidationError
from maki.objects import BackendType, GenerationConfig


def _make_response(content="Hi", reasoning=None, model="llama"):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.choices[0].message.reasoning = reasoning
    resp.choices[0].message.reasoning_content = None
    resp.model = model
    resp.usage.prompt_tokens = 5
    resp.usage.completion_tokens = 7
    resp.usage.total_tokens = 12
    return resp


def _make_chunk(content=None, reasoning=None, usage=None, model="llama", choices=True):
    chunk = MagicMock()
    chunk.model = model
    chunk.usage = usage
    if choices:
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = content
        chunk.choices[0].delta.reasoning = reasoning
        chunk.choices[0].delta.reasoning_content = None
    else:
        chunk.choices = []
    return chunk


def _http_response(status=200, body=None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = body or {}
    return r


class TestMakiVllm(unittest.TestCase):

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
        os.environ.pop("VLLM_API_KEY", None)

    def _make_llm(self, **kwargs):
        from maki.makiVllm import MakiVllm
        llm = MakiVllm(**kwargs)
        llm._client = MagicMock()
        llm._async_client = MagicMock()
        llm._http = MagicMock()
        return llm

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def test_missing_sdk_raises_import_error(self):
        import maki.makiOpenAI as openai_mod
        openai_mod._openai_sdk = None
        from maki.makiVllm import MakiVllm
        with self.assertRaises(ImportError):
            MakiVllm()

    def test_defaults_need_no_api_key(self):
        from maki.makiVllm import MakiVllm
        llm = MakiVllm()
        kwargs = self.sdk.OpenAI.call_args.kwargs
        self.assertEqual(kwargs["base_url"], "http://127.0.0.1:8000/v1")
        self.assertEqual(kwargs["api_key"], "no-key")
        self.assertEqual(llm._headers(), {})

    def test_api_key_from_env_is_used_everywhere(self):
        os.environ["VLLM_API_KEY"] = "secret"
        from maki.makiVllm import MakiVllm
        llm = MakiVllm()
        self.assertEqual(self.sdk.OpenAI.call_args.kwargs["api_key"], "secret")
        self.assertEqual(self.sdk.AsyncOpenAI.call_args.kwargs["api_key"], "secret")
        self.assertEqual(llm._headers(), {"Authorization": "Bearer secret"})

    def test_custom_base_url_trailing_slash_stripped(self):
        from maki.makiVllm import MakiVllm
        llm = MakiVllm(base_url="http://192.168.1.20:9000/v1/")
        self.assertEqual(llm.base_url, "http://192.168.1.20:9000/v1")
        self.assertEqual(llm._server_root, "http://192.168.1.20:9000")

    def test_invalid_base_url_rejected(self):
        from maki.makiVllm import MakiVllm
        with self.assertRaises(MakiValidationError):
            MakiVllm(base_url="file:///etc/passwd")

    def test_exported_from_package(self):
        import maki
        from maki.makiVllm import MakiVllm
        self.assertIs(maki.MakiVllm, MakiVllm)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def test_chat_sends_vllm_sampling_params(self):
        llm = self._make_llm()
        llm._client.chat.completions.create.return_value = _make_response()
        llm.chat("hello", config=GenerationConfig(top_k=12, repeat_penalty=1.3))
        kwargs = llm._client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["extra_body"], {"top_k": 12, "repetition_penalty": 1.3})

    def test_chat_stamps_backend_and_reasoning(self):
        llm = self._make_llm(model="llama")
        llm._client.chat.completions.create.return_value = _make_response(
            content="Paris", reasoning="The capital is Paris."
        )
        result = llm.chat("capital of France?")
        self.assertEqual(result.content, "Paris")
        self.assertEqual(result.reasoning, "The capital is Paris.")
        self.assertEqual(result.backend, BackendType.VLLM)

    def test_reasoning_content_fallback(self):
        llm = self._make_llm()
        resp = _make_response()
        resp.choices[0].message.reasoning = None
        resp.choices[0].message.reasoning_content = "older field"
        llm._client.chat.completions.create.return_value = resp
        self.assertEqual(llm.chat("hi").reasoning, "older field")

    def test_chat_without_reasoning_leaves_none(self):
        llm = self._make_llm()
        llm._client.chat.completions.create.return_value = _make_response()
        self.assertIsNone(llm.chat("hi").reasoning)

    def test_chat_collect_joins_stream(self):
        llm = self._make_llm()
        usage = MagicMock(prompt_tokens=3, completion_tokens=4, total_tokens=7)
        usage.prompt_tokens_details.cached_tokens = 2
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
        self.assertEqual(result.cached_tokens, 2)
        self.assertTrue(llm._client.chat.completions.create.call_args.kwargs["stream"])

    def test_chat_collect_maps_connection_error(self):
        llm = self._make_llm()
        llm._client.chat.completions.create.side_effect = self.sdk.APIConnectionError("down")
        with self.assertRaises(MakiNetworkError):
            llm.chat_collect("hi")

    # ------------------------------------------------------------------
    # Structured outputs
    # ------------------------------------------------------------------

    def _extra_body(self, llm):
        return llm._client.chat.completions.create.call_args.kwargs["extra_body"]

    def test_chat_json_sends_schema(self):
        llm = self._make_llm()
        llm._client.chat.completions.create.return_value = _make_response('{"name": "Ada"}')
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = llm.chat_json("who?", schema)
        self.assertEqual(result.content, '{"name": "Ada"}')
        self.assertEqual(self._extra_body(llm)["structured_outputs"], {"json": schema})

    def test_chat_json_accepts_pydantic_model(self):
        class Model:
            @classmethod
            def model_json_schema(cls):
                return {"type": "object"}

        llm = self._make_llm()
        llm._client.chat.completions.create.return_value = _make_response("{}")
        llm.chat_json("x", Model)
        self.assertEqual(self._extra_body(llm)["structured_outputs"], {"json": {"type": "object"}})

    def test_chat_json_rejects_bad_schema(self):
        with self.assertRaises(TypeError):
            self._make_llm().chat_json("x", "not a schema")

    def test_chat_choice_and_regex(self):
        llm = self._make_llm()
        llm._client.chat.completions.create.return_value = _make_response("positive")
        llm.chat_choice("mood?", ["positive", "negative"])
        self.assertEqual(self._extra_body(llm)["structured_outputs"],
                         {"choice": ["positive", "negative"]})
        llm.chat_regex("code?", r"\d{4}")
        self.assertEqual(self._extra_body(llm)["structured_outputs"], {"regex": r"\d{4}"})
        with self.assertRaises(ValueError):
            llm.chat_choice("x", [])

    def test_structured_constraint_does_not_leak_into_next_call(self):
        llm = self._make_llm()
        llm._client.chat.completions.create.return_value = _make_response()
        llm.chat_choice("mood?", ["a", "b"])
        llm.chat("plain")
        self.assertNotIn("structured_outputs", self._extra_body(llm))

    def test_structured_constraint_cleared_after_error(self):
        llm = self._make_llm()
        llm._client.chat.completions.create.side_effect = self.sdk.APIConnectionError("down")
        with self.assertRaises(MakiNetworkError):
            llm.chat_choice("mood?", ["a", "b"])
        llm._client.chat.completions.create.side_effect = None
        llm._client.chat.completions.create.return_value = _make_response()
        llm.chat("plain")
        self.assertNotIn("structured_outputs", self._extra_body(llm))

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    def test_chat_batch_preserves_order(self):
        llm = self._make_llm()

        def fake_create(**kwargs):
            prompt = kwargs["messages"][-1]["content"]
            time.sleep(0.05 if prompt == "first" else 0)
            return _make_response(content=prompt.upper())

        llm._client.chat.completions.create = fake_create
        results = llm.chat_batch(["first", "second", "third"])
        self.assertEqual([r.content for r in results], ["FIRST", "SECOND", "THIRD"])
        self.assertTrue(all(r.backend == BackendType.VLLM for r in results))

    def test_chat_batch_respects_max_concurrency(self):
        llm = self._make_llm()
        lock = threading.Lock()
        state = {"in_flight": 0, "peak": 0}

        def fake_create(**kwargs):
            with lock:
                state["in_flight"] += 1
                state["peak"] = max(state["peak"], state["in_flight"])
            time.sleep(0.02)
            with lock:
                state["in_flight"] -= 1
            return _make_response()

        llm._client.chat.completions.create = fake_create
        llm.chat_batch([str(i) for i in range(8)], max_concurrency=2)
        self.assertEqual(state["peak"], 2)

    def test_chat_batch_raises_first_failure(self):
        llm = self._make_llm()
        llm._client.chat.completions.create.side_effect = self.sdk.APIConnectionError("down")
        with self.assertRaises(MakiNetworkError):
            llm.chat_batch(["a", "b"])

    def test_chat_batch_validates_and_handles_empty(self):
        llm = self._make_llm()
        with self.assertRaises(ValueError):
            llm.chat_batch(["a"], max_concurrency=0)
        self.assertEqual(llm.chat_batch([]), [])

    def test_chat_batch_works_inside_event_loop_and_repeatedly(self):
        llm = self._make_llm()
        llm._client.chat.completions.create.return_value = _make_response()

        async def run():
            return llm.chat_batch(["a", "b"])

        self.assertEqual(len(asyncio.run(run())), 2)
        self.assertEqual(len(llm.chat_batch(["c"])), 1)

    def test_async_chat_batch_preserves_order(self):
        llm = self._make_llm()

        async def fake_create(**kwargs):
            prompt = kwargs["messages"][-1]["content"]
            await asyncio.sleep(0.02 if prompt == "first" else 0)
            return _make_response(content=prompt.upper())

        llm._async_client.chat.completions.create = fake_create
        results = asyncio.run(llm.async_chat_batch(["first", "second"], max_concurrency=2))
        self.assertEqual([r.content for r in results], ["FIRST", "SECOND"])

    # ------------------------------------------------------------------
    # Native endpoints
    # ------------------------------------------------------------------

    def test_health(self):
        llm = self._make_llm()
        llm._http.get.return_value = _http_response(200)
        self.assertTrue(llm.health())
        self.assertEqual(llm._http.get.call_args.args[0], "http://127.0.0.1:8000/health")
        llm._http.get.return_value = _http_response(503)
        self.assertFalse(llm.health())
        llm._http.get.side_effect = MakiNetworkError("refused")
        self.assertFalse(llm.health())

    def test_models_lists_ids(self):
        llm = self._make_llm()
        llm._http.get.return_value = _http_response(body={"data": [{"id": "base"}, {"id": "my-lora"}]})
        self.assertEqual(llm.models(), ["base", "my-lora"])
        self.assertEqual(llm._http.get.call_args.args[0], "http://127.0.0.1:8000/v1/models")

    def test_embed_batch_orders_by_index(self):
        llm = self._make_llm()
        llm._http.post.return_value = _http_response(body={"data": [
            {"index": 1, "embedding": [0.2]},
            {"index": 0, "embedding": [0.1]},
        ]})
        self.assertEqual(llm.embed_batch(["a", "b"]), [[0.1], [0.2]])
        self.assertEqual(llm._http.post.call_args.args[0], "http://127.0.0.1:8000/v1/embeddings")

    def test_embed_empty_raises(self):
        llm = self._make_llm()
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
        self.assertEqual(call.args[0], "http://127.0.0.1:8000/v1/rerank")
        self.assertEqual(call.kwargs["json"]["top_n"], 2)


class TestSharedBase(unittest.TestCase):

    def test_base_is_abstract(self):
        from maki.makiOpenAICompat import OpenAICompatServer
        with self.assertRaises(TypeError):
            OpenAICompatServer("m", "http://127.0.0.1:8000/v1")

    def test_backends_share_the_base_but_not_each_other(self):
        from maki.makiLlamaCpp import MakiLlamaCpp
        from maki.makiOpenAICompat import OpenAICompatServer
        from maki.makiVllm import MakiVllm
        self.assertTrue(issubclass(MakiVllm, OpenAICompatServer))
        self.assertTrue(issubclass(MakiLlamaCpp, OpenAICompatServer))
        self.assertFalse(issubclass(MakiVllm, MakiLlamaCpp))


class TestConfigLoaderVllm(unittest.TestCase):

    def setUp(self):
        __import__("pytest").importorskip("yaml", reason="PyYAML not installed")

    def test_vllm_backend_with_base_url(self):
        from maki.distributed.config_loader import _build_backend
        with patch("maki.makiVllm.MakiVllm") as cls:
            _build_backend({"backend": "vllm", "model": "llama",
                            "base_url": "http://10.0.0.5:8000/v1"})
        kwargs = cls.call_args.kwargs
        self.assertEqual(kwargs["model"], "llama")
        self.assertEqual(kwargs["base_url"], "http://10.0.0.5:8000/v1")

    def test_unknown_backend_lists_vllm(self):
        from maki.distributed.config_loader import _build_backend
        with self.assertRaisesRegex(ValueError, "vllm"):
            _build_backend({"backend": "nope"})


if __name__ == "__main__":
    unittest.main()
