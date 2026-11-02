"""
Runs the shared conformance suite (backend_conformance.py) against every Maki backend.
All transports are faked; nothing touches the network.
"""
import json
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from maki.exceptions import MakiAPIError, MakiNetworkError, MakiTimeoutError
from maki.test.backend_conformance import Adapter, BackendConformance, Sent


def _jsonable(content):
    return content if isinstance(content, str) else json.dumps(content, default=str)


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

class _OllamaResponse:
    """Stands in for ``requests.Response``."""

    def __init__(self, data=None, lines=(), status=200):
        self._data, self.status_code, self.text = data, status, json.dumps(data or {"error": status})
        # A streaming caller (chat_collect) reads the same reply as a single NDJSON line.
        self._lines = list(lines) or ([json.dumps(data).encode()] if data is not None else [])

    def json(self):
        return self._data

    def iter_lines(self, **kwargs):
        return iter(self._lines)

    def close(self):
        pass


class _OllamaSession:
    """Replaces ``Connector._session.post`` so the real Connector's error mapping runs."""

    def __init__(self):
        self.calls, self.queue, self.error = [], [], None

    def post(self, url, json=None, **kwargs):
        self.calls.append(json)
        if self.error:
            raise self.error
        return self.queue.pop(0)


class OllamaAdapter(Adapter):
    """Sync calls run the real Connector over a faked requests session; async calls run the real
    AsyncConnector over an httpx MockTransport, so both error paths are the production ones."""

    def __init__(self, tc):
        import httpx
        self.tc = tc
        self.async_calls, self._async_reply = [], None
        real_client = httpx.AsyncClient
        transport = httpx.MockTransport(self._handle_async)

        def client(*args, **kwargs):
            kwargs["transport"] = transport
            return real_client(*args, **kwargs)

        patcher = patch.object(httpx, "AsyncClient", client)
        patcher.start()
        tc.addCleanup(patcher.stop)

    def _handle_async(self, request):
        import httpx
        self.async_calls.append(json.loads(request.content))
        reply = self._async_reply
        if isinstance(reply, Exception):
            raise reply
        return httpx.Response(reply[0], json=reply[1])

    def make(self, **kwargs):
        from maki.makiLLama import MakiLLama
        backend = MakiLLama(model="m", **kwargs)
        backend._session = _OllamaSession()
        backend._http._session.post = backend._session.post
        return backend

    @staticmethod
    def _data(message, p, c):
        return {"model": "m", "message": message, "done": True, "prompt_eval_count": p, "eval_count": c}

    def script_text(self, backend, text, prompt_tokens=7, completion_tokens=3):
        backend._session.queue.append(_OllamaResponse(self._data(
            {"role": "assistant", "content": text}, prompt_tokens, completion_tokens)))

    def script_tool_call(self, backend, name, args):
        backend._session.queue.append(_OllamaResponse(self._data(
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]},
            5, 2)))

    def script_stream(self, backend, chunks):
        lines = [json.dumps({"message": {"content": c}, "done": False}).encode() for c in chunks]
        lines.append(json.dumps({"message": {"content": ""}, "done": True}).encode())
        backend._session.queue.append(_OllamaResponse(lines=lines))

    def script_error(self, backend, kind):
        import requests
        raised = {"timeout": requests.exceptions.Timeout("t"),
                  "connection": requests.exceptions.ConnectionError("c")}
        if kind in raised:
            backend._session.error = raised[kind]
        else:
            status = {"client_error": 400, "server_error": 500, "gateway_timeout": 504}[kind]
            backend._session.queue.append(_OllamaResponse(status=status))

    def script_async_error(self, backend, kind):
        import httpx
        replies = {"timeout": httpx.ReadTimeout("t"), "connection": httpx.ConnectError("c"),
                   "client_error": (400, {"error": "bad"}), "server_error": (500, {"error": "boom"}),
                   "gateway_timeout": (504, {"error": "slow"})}
        self._async_reply = replies[kind]

    def script_async_text(self, backend, text):
        self._async_reply = (200, self._data({"role": "assistant", "content": text}, 1, 1))

    def sent(self, backend):
        payload = backend._session.calls[-1] if backend._session.calls else self.async_calls[-1]
        messages = [(m["role"], m["content"]) for m in payload["messages"] if m["role"] != "system"]
        system = next((m["content"] for m in payload["messages"] if m["role"] == "system"), None)
        return Sent(system, messages, payload.get("tools"), payload, payload["options"]["temperature"])


# ---------------------------------------------------------------------------
# OpenAI family: OpenAI, OpenRouter, llama.cpp, vLLM
# ---------------------------------------------------------------------------

class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._chunks)


class _OpenAIClient:
    def __init__(self, is_async=False):
        self.calls, self.queue, self.error, self.stream_chunks = [], [], None, []
        create = self._acreate if is_async else self._create
        self.chat = NS(completions=NS(create=create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        if kwargs.get("stream"):
            if self.stream_chunks:
                return _FakeStream(self.stream_chunks)
            return _FakeStream(_chunks_of(self.queue.pop(0)))
        return self.queue.pop(0)

    async def _acreate(self, **kwargs):
        return self._create(**kwargs)


def _chunks_of(response):
    """The stream a server would send for an already-built non-streaming response."""
    message = response.choices[0].message
    return [
        NS(model=response.model, usage=None,
           choices=[NS(delta=NS(content=message.content, reasoning_content=None))]),
        NS(model=response.model, usage=response.usage, choices=[]),
    ]


class OpenAIAdapter(Adapter):
    def __init__(self, tc):
        import maki.makiOpenAI as mod
        self.tc = tc

        class Timeout(Exception):
            pass

        class Connection(Exception):
            pass

        class Status(Exception):
            def __init__(self, status):
                super().__init__(f"HTTP {status}")
                self.status_code = status

        self.errors = {"timeout": Timeout("t"), "connection": Connection("c"),
                       "client_error": Status(400), "server_error": Status(500), "gateway_timeout": Status(504)}
        sdk = NS(OpenAI=lambda **kw: _OpenAIClient(), AsyncOpenAI=lambda **kw: _OpenAIClient(is_async=True),
                 APITimeoutError=Timeout, APIConnectionError=Connection, APIStatusError=Status)
        patcher = patch.object(mod, "_openai_sdk", sdk)
        patcher.start()
        tc.addCleanup(patcher.stop)

    def build(self, **kwargs):
        from maki.makiOpenAI import MakiOpenAI
        return MakiOpenAI(model="m", api_key="k", **kwargs)

    def make(self, **kwargs):
        return self.build(**kwargs)

    @staticmethod
    def _response(message, finish, p, c):
        return NS(model="m", choices=[NS(message=message, finish_reason=finish)],
                  usage=NS(prompt_tokens=p, completion_tokens=c, total_tokens=p + c))

    def script_text(self, backend, text, prompt_tokens=7, completion_tokens=3):
        message = NS(content=text, tool_calls=None, reasoning_content=None)
        backend._client.queue.append(self._response(message, "stop", prompt_tokens, completion_tokens))

    def script_tool_call(self, backend, name, args):
        call = NS(id="call_1", function=NS(name=name, arguments=json.dumps(args)))
        message = NS(content=None, tool_calls=[call], reasoning_content=None)
        backend._client.queue.append(self._response(message, "tool_calls", 5, 2))

    def script_stream(self, backend, chunks):
        backend._client.stream_chunks = [
            NS(model="m", usage=None, choices=[NS(delta=NS(content=c, reasoning_content=None))]) for c in chunks]

    def script_error(self, backend, kind):
        backend._client.error = self.errors[kind]

    def script_async_error(self, backend, kind):
        backend._async_client.error = self.errors[kind]

    def script_async_text(self, backend, text):
        message = NS(content=text, tool_calls=None, reasoning_content=None)
        backend._async_client.queue.append(self._response(message, "stop", 1, 1))

    def sent(self, backend):
        kwargs = backend._client.calls[-1] if backend._client.calls else backend._async_client.calls[-1]
        messages = [(m["role"], m["content"]) for m in kwargs["messages"] if m["role"] != "system"]
        system = next((m["content"] for m in kwargs["messages"] if m["role"] == "system"), None)
        return Sent(system, messages, kwargs.get("tools"), kwargs, kwargs.get("temperature"))


class LlamaCppAdapter(OpenAIAdapter):
    def build(self, **kwargs):
        from maki.makiLlamaCpp import MakiLlamaCpp
        return MakiLlamaCpp(model="m", **kwargs)


class VllmAdapter(OpenAIAdapter):
    def build(self, **kwargs):
        from maki.makiVllm import MakiVllm
        return MakiVllm(model="m", **kwargs)


class OpenRouterAdapter(OpenAIAdapter):
    def build(self, **kwargs):
        from maki.makiOpenRouter import MakiOpenRouter
        return MakiOpenRouter(model="vendor/m", api_key="k", **kwargs)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class _AnthropicStream:
    def __init__(self, texts):
        self.text_stream = iter(texts)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _AnthropicClient:
    def __init__(self, is_async=False):
        self.calls, self.queue, self.error, self.stream_texts = [], [], None, []
        self.messages = NS(create=self._acreate if is_async else self._create, stream=self._stream)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.queue.pop(0)

    async def _acreate(self, **kwargs):
        return self._create(**kwargs)

    def _stream(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return _AnthropicStream(self.stream_texts)


class AnthropicAdapter(Adapter):
    def __init__(self, tc):
        import maki.makiAnthropic as mod

        class Timeout(Exception):
            pass

        class Connection(Exception):
            pass

        class Status(Exception):
            def __init__(self, status):
                super().__init__(f"HTTP {status}")
                self.status_code = status

        self.errors = {"timeout": Timeout("t"), "connection": Connection("c"),
                       "client_error": Status(400), "server_error": Status(500), "gateway_timeout": Status(504)}
        sdk = NS(Anthropic=lambda **kw: _AnthropicClient(), AsyncAnthropic=lambda **kw: _AnthropicClient(True),
                 APITimeoutError=Timeout, APIConnectionError=Connection, APIStatusError=Status)
        patcher = patch.object(mod, "_anthropic_sdk", sdk)
        patcher.start()
        tc.addCleanup(patcher.stop)

    def make(self, **kwargs):
        from maki.makiAnthropic import MakiAnthropic
        return MakiAnthropic(model="m", api_key="k", **kwargs)

    @staticmethod
    def _response(blocks, stop, p, c):
        return NS(model="m", content=blocks, stop_reason=stop, usage=NS(input_tokens=p, output_tokens=c))

    def script_text(self, backend, text, prompt_tokens=7, completion_tokens=3):
        backend._client.queue.append(self._response(
            [NS(type="text", text=text)], "end_turn", prompt_tokens, completion_tokens))

    def script_tool_call(self, backend, name, args):
        backend._client.queue.append(self._response(
            [NS(type="tool_use", id="toolu_1", name=name, input=args)], "tool_use", 5, 2))

    def script_stream(self, backend, chunks):
        backend._client.stream_texts = list(chunks)

    def script_error(self, backend, kind):
        backend._client.error = self.errors[kind]

    def script_async_error(self, backend, kind):
        backend._async_client.error = self.errors[kind]

    def script_async_text(self, backend, text):
        backend._async_client.queue.append(self._response([NS(type="text", text=text)], "end_turn", 1, 1))

    def sent(self, backend):
        kwargs = backend._client.calls[-1] if backend._client.calls else backend._async_client.calls[-1]
        messages = [(m["role"], m["content"]) for m in kwargs["messages"]]
        return Sent(kwargs.get("system"), messages, kwargs.get("tools"), kwargs, kwargs.get("temperature"))


class TestErrorForStatus(unittest.TestCase):

    def test_classification(self):
        from maki.exceptions import error_for_status
        cases = {400: MakiAPIError, 401: MakiAPIError, 404: MakiAPIError, 429: MakiAPIError,
                 408: MakiTimeoutError, 504: MakiTimeoutError, 500: MakiNetworkError, 502: MakiNetworkError,
                 503: MakiNetworkError}
        for status, expected in cases.items():
            error = error_for_status(status, "m")
            self.assertIs(type(error), expected, status)
        self.assertEqual(error_for_status(422, "m").status_code, 422)

    def test_connector_uses_the_same_rule(self):
        from maki.connector import Connector
        for status, expected in ((400, MakiAPIError), (500, MakiNetworkError), (504, MakiTimeoutError)):
            with self.assertRaises(expected):
                Connector.raise_for_response(NS(status_code=status, text="boom"))
        Connector.raise_for_response(NS(status_code=200, text="ok"))  # no error


# ---------------------------------------------------------------------------
# One concrete TestCase per backend
# ---------------------------------------------------------------------------

class TestOllamaConformance(BackendConformance, unittest.TestCase):
    adapter_class = OllamaAdapter


class TestOpenAIConformance(BackendConformance, unittest.TestCase):
    adapter_class = OpenAIAdapter


class TestAnthropicConformance(BackendConformance, unittest.TestCase):
    adapter_class = AnthropicAdapter


class TestOpenRouterConformance(BackendConformance, unittest.TestCase):
    adapter_class = OpenRouterAdapter


class TestLlamaCppConformance(BackendConformance, unittest.TestCase):
    adapter_class = LlamaCppAdapter


class TestVllmConformance(BackendConformance, unittest.TestCase):
    adapter_class = VllmAdapter


if __name__ == "__main__":
    unittest.main()
