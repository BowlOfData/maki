"""
Shared conformance suite for Maki LLM backends.

Every backend promises the same contract (see ``maki/backend.py``): a normalised ``LLMResponse``,
honoured ``system`` and ``history``, rate limiting on every path, the Maki exception tree for
transport failures and, when ``supports_native_tools`` is set, the native tool-calling loop. The
tests live once in :class:`BackendConformance`; a backend joins by supplying an :class:`Adapter`
that fakes its transport and reports what was sent::

    class TestMyBackend(BackendConformance, unittest.TestCase):
        adapter_class = MyAdapter

Nothing here touches the network.
"""

import asyncio
import json
from typing import Any, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

from maki.backend import LLMBackend
from maki.exceptions import MakiAPIError, MakiNetworkError, MakiTimeoutError
from maki.objects import GenerationConfig, LLMResponse, Message, ToolCall
from maki.usage import attach_usage

#: (kind, exception every backend must raise). 5xx and gateway timeouts are transient, hence the
#: network/timeout classes the agent's retry logic keys on; a 4xx is the caller's problem.
ERROR_KINDS = (
    ("timeout", MakiTimeoutError),
    ("connection", MakiNetworkError),
    ("client_error", MakiAPIError),
    ("server_error", MakiNetworkError),
    ("gateway_timeout", MakiTimeoutError),
)

TOOL_SPEC = {
    "name": "files__read",
    "description": "Read a file.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "file to read"},
            "mode": {"type": "string", "enum": ["fast", "deep"]},
        },
        "required": ["path"],
    },
}


class Sent:
    """What a backend sent on its last request, normalised across wire formats."""

    def __init__(self, system: Optional[str], messages: List[Tuple[str, Any]], tools: Any, raw: Any,
                 temperature: Optional[float] = None):
        self.system, self.messages, self.tools, self.raw = system, messages, tools, raw
        self.temperature = temperature

    @property
    def texts(self) -> List[str]:
        return [c if isinstance(c, str) else json.dumps(c, default=str) for _, c in self.messages]


class Adapter:
    """Fakes one backend's transport. Subclasses implement every hook."""

    #: whether stream() is expected to work on this backend
    streams = True
    #: whether async_chat() is expected to work on this backend
    has_async = True
    #: whether the wire format carries tool-call ids (Ollama's does not)
    keeps_call_ids = True

    def make(self, **kwargs) -> LLMBackend:
        raise NotImplementedError

    def script_text(self, backend, text: str, prompt_tokens: int = 7, completion_tokens: int = 3) -> None:
        """Make the next non-streaming call answer with *text*."""
        raise NotImplementedError

    def script_tool_call(self, backend, name: str, args: dict) -> None:
        """Make the next non-streaming call ask for one tool invocation."""
        raise NotImplementedError

    def script_stream(self, backend, chunks: List[str]) -> None:
        raise NotImplementedError

    def script_error(self, backend, kind: str) -> None:
        """Make the transport fail; kind is one of ``ERROR_KINDS`` (HTTP statuses 400, 500, 504)."""
        raise NotImplementedError

    def script_async_text(self, backend, text: str) -> None:
        raise NotImplementedError

    def script_async_error(self, backend, kind: str) -> None:
        raise NotImplementedError

    def sent(self, backend) -> Sent:
        raise NotImplementedError

    def tools_forbidden(self, sent: Sent) -> bool:
        """Whether the last request offered no usable tool (``tool_choice="none"``)."""
        return not sent.tools


class BackendConformance:
    """Mixin of contract tests; combine with ``unittest.TestCase`` and set ``adapter_class``."""

    adapter_class: type = Adapter

    def setUp(self):
        self.adapter = self.adapter_class(self)
        self.backend = self.adapter.make()

    # -- identity -----------------------------------------------------------------

    def test_declares_the_backend_contract(self):
        b = self.backend
        self.assertIsInstance(b, LLMBackend)
        self.assertIsInstance(b.model, str)
        self.assertIsInstance(b.temperature, (int, float))
        self.assertIsInstance(b.supports_native_tools, bool)
        self.assertIsInstance(b.supports_history, bool)

    # -- chat ----------------------------------------------------------------------

    def test_chat_returns_a_normalised_response(self):
        self.adapter.script_text(self.backend, "hello", prompt_tokens=7, completion_tokens=3)
        r = self.backend.chat("hi")
        self.assertIsInstance(r, LLMResponse)
        self.assertEqual(r.content, "hello")
        self.assertEqual((r.prompt_tokens, r.completion_tokens, r.total_tokens), (7, 3, 10))
        self.assertIsInstance(r.model, str)
        self.assertGreaterEqual(r.elapsed_seconds, 0)

    def test_chat_sends_prompt_last_and_honours_system(self):
        self.adapter.script_text(self.backend, "ok")
        self.backend.chat("the-prompt", system="be-brief")
        sent = self.adapter.sent(self.backend)
        self.assertEqual(sent.system, "be-brief")
        self.assertEqual(sent.messages[-1], ("user", "the-prompt"))

    def test_constructor_system_prompt_is_the_default_system(self):
        backend = self.adapter.make(system_prompt="from-constructor")
        self.adapter.script_text(backend, "ok")
        backend.chat("x")
        self.assertEqual(self.adapter.sent(backend).system, "from-constructor")
        self.adapter.script_text(backend, "ok")
        backend.chat("x", system="override")
        self.assertEqual(self.adapter.sent(backend).system, "override")

    def test_history_is_sent_in_order_before_the_prompt_when_supported(self):
        history = [Message("user", "first-q"), Message("assistant", "first-a")]
        self.adapter.script_text(self.backend, "ok")
        self.backend.chat("second-q", history=history)
        sent = self.adapter.sent(self.backend)
        if self.backend.supports_history:
            self.assertEqual(sent.messages, [("user", "first-q"), ("assistant", "first-a"), ("user", "second-q")])
        else:
            self.assertEqual(sent.messages[-1], ("user", "second-q"))

    def test_per_call_config_reaches_the_wire(self):
        self.adapter.script_text(self.backend, "ok")
        self.backend.chat("x", config=GenerationConfig(temperature=0.123))
        self.assertEqual(self.adapter.sent(self.backend).temperature, 0.123)

    def test_request_is_a_plain_chat(self):
        self.adapter.script_text(self.backend, "via request")
        self.assertEqual(self.backend.request("x").content, "via request")

    def test_chat_collect_matches_chat(self):
        self.adapter.script_text(self.backend, "collected")
        r = self.backend.chat_collect("hi", system="s")
        self.assertIsInstance(r, LLMResponse)
        self.assertEqual(r.content, "collected")

    # -- streaming -----------------------------------------------------------------

    def test_stream_yields_text_chunks(self):
        if not self.adapter.streams:
            self.skipTest("backend does not stream")
        self.adapter.script_stream(self.backend, ["Hel", "lo", " world"])
        chunks = list(self.backend.stream("hi", system="s"))
        self.assertEqual("".join(chunks), "Hello world")
        self.assertTrue(all(isinstance(c, str) and c for c in chunks))

    def test_stream_failures_map_onto_the_maki_exception_tree(self):
        if not self.adapter.streams:
            self.skipTest("backend does not stream")
        for kind, expected in ERROR_KINDS:
            with self.subTest(kind=kind):
                backend = self.adapter.make()
                self.adapter.script_error(backend, kind)
                with self.assertRaises(expected):
                    list(backend.stream("x"))

    # -- rate limiting -------------------------------------------------------------

    def test_every_blocking_path_acquires_the_rate_limiter(self):
        limiter = MagicMock()
        self.backend._rate_limiter = limiter
        self.adapter.script_text(self.backend, "ok")
        self.backend.chat("x")
        self.assertEqual(limiter.acquire.call_count, 1)
        self.adapter.script_text(self.backend, "ok")
        self.backend.chat_collect("x")
        self.assertEqual(limiter.acquire.call_count, 2)
        if self.adapter.streams:
            self.adapter.script_stream(self.backend, ["a"])
            before = limiter.acquire.call_count
            list(self.backend.stream("x"))
            self.assertEqual(limiter.acquire.call_count, before + 1)
        if self.backend.supports_native_tools:
            self.adapter.script_text(self.backend, "ok")
            before = limiter.acquire.call_count
            self.backend.chat_with_tools([{"role": "user", "content": "x"}], [])
            self.assertEqual(limiter.acquire.call_count, before + 1)

    def test_async_chat_awaits_the_async_limiter_and_never_blocks_the_loop(self):
        if not self.adapter.has_async:
            self.skipTest("backend has no async_chat")
        limiter = MagicMock()
        limiter.async_acquire = AsyncMock()
        self.backend._rate_limiter = limiter
        self.adapter.script_async_text(self.backend, "async-ok")
        r = asyncio.run(self.backend.async_chat("x"))
        self.assertEqual(r.content, "async-ok")
        limiter.async_acquire.assert_awaited_once()
        limiter.acquire.assert_not_called()  # the blocking acquire would freeze the event loop

    # -- errors --------------------------------------------------------------------

    def test_transport_failures_map_onto_the_maki_exception_tree(self):
        for kind, expected in ERROR_KINDS:
            with self.subTest(kind=kind):
                backend = self.adapter.make()
                self.adapter.script_error(backend, kind)
                with self.assertRaises(expected) as caught:
                    backend.chat("x")
                if kind == "client_error":
                    self.assertEqual(caught.exception.status_code, 400)

    def test_async_failures_map_onto_the_maki_exception_tree(self):
        if not self.adapter.has_async:
            self.skipTest("backend has no async_chat")
        for kind, expected in ERROR_KINDS:
            with self.subTest(kind=kind):
                backend = self.adapter.make()
                self.adapter.script_async_error(backend, kind)
                with self.assertRaises(expected):
                    asyncio.run(backend.async_chat("x"))

    def test_tool_round_failures_map_onto_the_maki_exception_tree(self):
        if not self.backend.supports_native_tools:
            self.skipTest("no native tools")
        for kind, expected in ERROR_KINDS:
            with self.subTest(kind=kind):
                backend = self.adapter.make()
                self.adapter.script_error(backend, kind)
                with self.assertRaises(expected):
                    backend.chat_with_tools([{"role": "user", "content": "x"}], [])

    # -- usage ---------------------------------------------------------------------

    def test_token_usage_is_counted(self):
        usage = attach_usage(self.backend)
        self.adapter.script_text(self.backend, "ok", prompt_tokens=11, completion_tokens=4)
        self.backend.chat("x")
        self.assertEqual((usage.prompt_tokens, usage.completion_tokens, usage.calls), (11, 4, 1))

    # -- native tools --------------------------------------------------------------

    def _needs_tools(self):
        if not self.backend.supports_native_tools:
            self.skipTest("no native tools")

    def test_tool_schemas_keep_name_description_and_typed_parameters(self):
        self._needs_tools()
        dumped = json.dumps(self.backend.to_tool_schemas([TOOL_SPEC]))
        for needle in ("files__read", "Read a file.", "path", "file to read", "enum", "deep"):
            self.assertIn(needle, dumped)

    def test_tool_text_answer_returns_response_and_extends_the_conversation(self):
        self._needs_tools()
        tools = self.backend.to_tool_schemas([TOOL_SPEC])
        self.adapter.script_text(self.backend, "no tool needed")
        messages = [{"role": "user", "content": "q"}]
        response, calls, updated = self.backend.chat_with_tools(messages, tools, system="sys")
        self.assertIsInstance(response, LLMResponse)
        self.assertEqual(response.content, "no tool needed")
        self.assertIsNone(calls)
        self.assertEqual(len(updated), 2)
        self.assertEqual(len(messages), 1, "the caller's list must not be mutated")
        sent = self.adapter.sent(self.backend)
        self.assertEqual(sent.system, "sys")
        self.assertIn("files__read", json.dumps(sent.tools))

    def test_tool_request_returns_tool_calls_not_a_response(self):
        self._needs_tools()
        tools = self.backend.to_tool_schemas([TOOL_SPEC])
        self.adapter.script_tool_call(self.backend, "files__read", {"path": "/tmp/x", "mode": "fast"})
        response, calls, updated = self.backend.chat_with_tools([{"role": "user", "content": "q"}], tools)
        self.assertIsNone(response)
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0], ToolCall)
        self.assertEqual(calls[0].name, "files__read")
        self.assertEqual(calls[0].args, {"path": "/tmp/x", "mode": "fast"})
        self.assertIsInstance(calls[0].id, str)
        self.assertEqual(len(updated), 2)

    def test_empty_tool_list_sends_no_tools(self):
        self._needs_tools()
        self.adapter.script_text(self.backend, "plain")
        self.backend.chat_with_tools([{"role": "user", "content": "q"}], [])
        self.assertFalse(self.adapter.sent(self.backend).tools)

    def test_tool_results_round_trip_into_the_next_request(self):
        self._needs_tools()
        tools = self.backend.to_tool_schemas([TOOL_SPEC])
        self.adapter.script_tool_call(self.backend, "files__read", {"path": "/tmp/x"})
        _, calls, messages = self.backend.chat_with_tools([{"role": "user", "content": "q"}], tools)
        extended = self.backend.append_tool_results(messages, [(calls[0], "FILE-CONTENTS")])
        self.assertGreater(len(extended), len(messages))
        self.assertEqual(len(messages), 2, "append_tool_results must not mutate its input")
        self.adapter.script_text(self.backend, "final answer")
        response, calls2, _ = self.backend.chat_with_tools(extended, tools)
        self.assertEqual(response.content, "final answer")
        self.assertIsNone(calls2)
        self.assertIn("FILE-CONTENTS", json.dumps(self.adapter.sent(self.backend).raw, default=str))

    # -- typed messages --------------------------------------------------------------

    def _needs_typed(self):
        if not self.backend.supports_typed_messages:
            self.skipTest("no typed messages")

    def test_typed_backends_are_native_tool_backends(self):
        if self.backend.supports_typed_messages:
            self.assertTrue(self.backend.supports_native_tools)

    def test_chat_turn_text_answer(self):
        self._needs_typed()
        self.adapter.script_text(self.backend, "typed answer")
        history = [Message("user", "q1"), Message("assistant", "a1"), Message("user", "q2")]
        response = self.backend.chat_turn(history, [TOOL_SPEC], system="sys")
        self.assertIsInstance(response, LLMResponse)
        self.assertEqual(response.content, "typed answer")
        self.assertFalse(response.tool_calls)
        sent = self.adapter.sent(self.backend)
        self.assertEqual(sent.system, "sys")
        self.assertIn("files__read", json.dumps(sent.tools))
        self.assertEqual([t for t in sent.texts if t in ("q1", "a1", "q2")], ["q1", "a1", "q2"])
        self.assertEqual(len(history), 3, "the caller's list must not be mutated")

    def test_chat_turn_tool_request_keeps_the_response_and_gives_every_call_an_id(self):
        self._needs_typed()
        self.adapter.script_tool_call(self.backend, "files__read", {"path": "/tmp/x", "mode": "fast"})
        response = self.backend.chat_turn([Message("user", "q")], [TOOL_SPEC])
        self.assertIsInstance(response, LLMResponse)
        self.assertEqual(len(response.tool_calls), 1)
        call = response.tool_calls[0]
        self.assertEqual((call.name, call.args), ("files__read", {"path": "/tmp/x", "mode": "fast"}))
        self.assertTrue(call.id, "a call without a provider id gets one so the turn can move between backends")
        self.assertGreater(response.total_tokens, 0, "a tool round still reports what it cost")

    def test_chat_turn_tool_rounds_are_counted_in_usage(self):
        self._needs_typed()
        usage = attach_usage(self.backend)
        self.adapter.script_tool_call(self.backend, "files__read", {"path": "/tmp/x"})
        self.backend.chat_turn([Message("user", "q")], [TOOL_SPEC])
        self.assertEqual(usage.calls, 1)
        self.assertGreater(usage.total_tokens, 0)

    def test_chat_turn_replays_a_tool_exchange(self):
        self._needs_typed()
        call = ToolCall(id="call_77", name="files__read", args={"path": "/tmp/x"})
        messages = [Message("user", "q"),
                    Message("assistant", "reading", tool_calls=[call]),
                    Message("tool", "FILE-CONTENTS", tool_call_id="call_77", name="files__read")]
        self.adapter.script_text(self.backend, "final answer")
        response = self.backend.chat_turn(messages, [TOOL_SPEC])
        self.assertEqual(response.content, "final answer")
        self.assertFalse(response.tool_calls)
        raw = json.dumps(self.adapter.sent(self.backend).raw, default=str)
        for needle in ("FILE-CONTENTS", "files__read", "reading", "/tmp/x"):
            self.assertIn(needle, raw)
        if self.adapter.keeps_call_ids:
            self.assertGreaterEqual(raw.count("call_77"), 2, "the call and its result must name the same id")

    def test_chat_turn_with_two_calls_in_one_turn(self):
        self._needs_typed()
        calls = [ToolCall("a1", "files__read", {"path": "/a"}), ToolCall("b2", "files__read", {"path": "/b"})]
        messages = [Message("user", "q"), Message("assistant", "", tool_calls=calls),
                    Message("tool", "RES-A", tool_call_id="a1", name="files__read"),
                    Message("tool", "RES-B", tool_call_id="b2", name="files__read")]
        self.adapter.script_text(self.backend, "done")
        self.assertEqual(self.backend.chat_turn(messages, [TOOL_SPEC]).content, "done")
        raw = json.dumps(self.adapter.sent(self.backend).raw, default=str)
        for needle in ("RES-A", "RES-B", "/a", "/b"):
            self.assertIn(needle, raw)

    def test_chat_turn_can_forbid_tools_after_a_tool_exchange(self):
        self._needs_typed()
        call = ToolCall("call_9", "files__read", {"path": "/tmp/x"})
        messages = [Message("user", "q"), Message("assistant", "", tool_calls=[call]),
                    Message("tool", "RES", tool_call_id="call_9", name="files__read")]
        self.adapter.script_text(self.backend, "wrapped up")
        response = self.backend.chat_turn(messages, [TOOL_SPEC], tool_choice="none")
        self.assertEqual(response.content, "wrapped up")
        self.assertTrue(self.adapter.tools_forbidden(self.adapter.sent(self.backend)))

    def test_chat_turn_without_tools_sends_none(self):
        self._needs_typed()
        self.adapter.script_text(self.backend, "plain")
        self.backend.chat_turn([Message("user", "q")], None)
        self.assertFalse(self.adapter.sent(self.backend).tools)
        self.adapter.script_text(self.backend, "plain")
        self.backend.chat_turn([Message("user", "q")], [])
        self.assertFalse(self.adapter.sent(self.backend).tools)

    def test_chat_turn_lays_a_stored_tool_trace_out_before_the_answer(self):
        self._needs_typed()
        call = ToolCall("t1", "files__read", {"path": "/p"})
        trace = [Message("assistant", "", tool_calls=[call]), Message("tool", "TRACE-RESULT", tool_call_id="t1")]
        history = [Message("user", "earlier q"), Message("assistant", "earlier a", tool_trace=trace),
                   Message("user", "now")]
        self.adapter.script_text(self.backend, "ok")
        self.backend.chat_turn(history, [TOOL_SPEC])
        raw = json.dumps(self.adapter.sent(self.backend).raw, default=str)
        self.assertLess(raw.index("TRACE-RESULT"), raw.index("earlier a"))
        self.assertLess(raw.index("earlier q"), raw.index("TRACE-RESULT"))
        self.assertNotIn("tool_trace", raw)

    def test_chat_turn_errors_map_onto_the_maki_exception_tree(self):
        self._needs_typed()
        for kind, expected in ERROR_KINDS:
            with self.subTest(kind=kind):
                backend = self.adapter.make()
                self.adapter.script_error(backend, kind)
                with self.assertRaises(expected):
                    backend.chat_turn([Message("user", "q")], [TOOL_SPEC])

