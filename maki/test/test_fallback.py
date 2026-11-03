"""FallbackBackend: ordering, which errors move on, streaming, and tool loops that change backend."""
import unittest

from maki import Agent, FallbackBackend
from maki.backend import LLMBackend
from maki.exceptions import MakiAPIError, MakiNetworkError, MakiTimeoutError, MakiValidationError
from maki.objects import LLMResponse, ToolCall


def reply(text, model="m"):
    return LLMResponse(content=text, model=model, prompt_tokens=1, completion_tokens=1, total_tokens=2,
                       elapsed_seconds=0.0)


class Fake(LLMBackend):
    """A backend that fails with ``error`` (until ``fail_times`` runs out) or answers ``label``."""
    supports_history = True

    def __init__(self, label, error=None, fail_times=None, native=False, calls=None):
        self.model, self.temperature, self.label = label, 0.0, label
        self.error, self.fail_times, self.supports_native_tools = error, fail_times, native
        self.calls, self.seen = 0, []
        self.script = list(calls or [])  # tool calls to request, one list per round

    def _maybe_fail(self):
        self.calls += 1
        if self.error and (self.fail_times is None or self.calls <= self.fail_times):
            raise self.error

    def request(self, prompt):
        self._maybe_fail()
        return reply(self.label, self.label)

    def stream(self, prompt, history=None, config=None, system=None):
        self._maybe_fail()
        yield from (self.label, "-2")

    def to_tool_schemas(self, specs):
        return [{"wire": self.label, "name": s["name"]} for s in specs]

    def chat_with_tools(self, messages, tools, *, system=None, config=None):
        self._maybe_fail()
        self.seen.append((list(messages), tools))
        if self.script and tools:
            calls = self.script.pop(0)
            return None, calls, list(messages) + [{"role": "assistant", "from": self.label, "calls": len(calls)}]
        return reply(self.label, self.label), None, list(messages) + [{"role": "assistant", "content": self.label}]

    def append_tool_results(self, messages, results):
        return list(messages) + [{"role": "tool", "from": self.label, "n": len(results)}]


NET = MakiNetworkError("down")


class TestChoosing(unittest.TestCase):
    def test_first_backend_answers(self):
        a, b = Fake("a"), Fake("b")
        fb = FallbackBackend([a, b])
        self.assertEqual(fb.request("hi").content, "a")
        self.assertIs(fb.last_used, a)
        self.assertEqual(b.calls, 0)

    def test_transient_errors_move_on(self):
        for error in (NET, MakiTimeoutError("slow"), MakiAPIError("rate", status_code=429),
                      MakiAPIError("req timeout", status_code=408)):
            with self.subTest(error=error):
                a, b = Fake("a", error=error), Fake("b")
                fb = FallbackBackend([a, b])
                self.assertEqual(fb.chat("hi").content, "b")
                self.assertIs(fb.last_used, b)

    def test_other_errors_are_raised_at_once(self):
        for error in (MakiAPIError("bad request", status_code=400), MakiAPIError("auth", status_code=401),
                      MakiValidationError("nope"), ValueError("x")):
            with self.subTest(error=error):
                a, b = Fake("a", error=error), Fake("b")
                with self.assertRaises(type(error)):
                    FallbackBackend([a, b]).request("hi")
                self.assertEqual(b.calls, 0)

    def test_all_failing_raises_the_last_error_with_its_type(self):
        fb = FallbackBackend([Fake("a", error=NET), Fake("b", error=MakiTimeoutError("slow"))])
        with self.assertRaises(MakiTimeoutError):
            fb.request("hi")

    def test_recovers_for_the_next_call(self):
        a = Fake("a", error=NET, fail_times=1)
        fb = FallbackBackend([a, Fake("b")])
        self.assertEqual(fb.request("1").content, "b")
        self.assertEqual(fb.request("2").content, "a")

    def test_custom_rule(self):
        a, b = Fake("a", error=MakiAPIError("auth", status_code=401)), Fake("b")
        by_class = FallbackBackend([a, b], on=(MakiAPIError,))
        self.assertEqual(by_class.request("x").content, "b")
        by_func = FallbackBackend([Fake("a", error=NET), Fake("b")], on=lambda exc: False)
        with self.assertRaises(MakiNetworkError):
            by_func.request("x")

    def test_rejects_bad_arguments(self):
        for bad in ([], "openai:gpt-4o", [object()]):
            with self.subTest(bad=bad), self.assertRaises((ValueError, TypeError)):
                FallbackBackend(bad)
        with self.assertRaises(TypeError):
            FallbackBackend([Fake("a")], on=42)
        with self.assertRaises(TypeError):
            FallbackBackend([Fake("a")], on=(int,))

    def test_spec_strings_are_built(self):
        from unittest.mock import patch
        built = Fake("built")
        with patch("maki.backend_factory.init_backend", return_value=built) as init:
            fb = FallbackBackend(["ollama:gemma4:26b"])
        init.assert_called_once_with("ollama:gemma4:26b")
        self.assertIs(fb.backends[0], built)

    def test_properties_follow_the_backends(self):
        plain, native = Fake("a"), Fake("b", native=True)
        fb = FallbackBackend([plain, native])
        self.assertEqual((fb.model, fb.temperature), ("a", 0.0))
        self.assertTrue(fb.supports_native_tools)
        self.assertFalse(FallbackBackend([plain]).supports_native_tools)
        self.assertTrue(fb.supports_history)
        no_history = Fake("c")
        no_history.supports_history = False
        self.assertFalse(FallbackBackend([plain, no_history]).supports_history)
        self.assertIn("Fake(a)", repr(fb))


class TestStreaming(unittest.TestCase):
    def test_falls_back_before_the_first_chunk(self):
        fb = FallbackBackend([Fake("a", error=NET), Fake("b")])
        self.assertEqual("".join(fb.stream("hi")), "b-2")
        self.assertEqual(fb.last_used.label, "b")

    def test_a_backend_without_streaming_is_skipped(self):
        class NoStream(Fake):
            stream = LLMBackend.stream
        fb = FallbackBackend([NoStream("a"), Fake("b")])
        self.assertEqual("".join(fb.stream("hi")), "b-2")

    def test_failure_after_a_chunk_is_raised(self):
        class Dies(Fake):
            def stream(self, *a, **k):
                yield "part"
                raise NET
        b = Fake("b")
        fb = FallbackBackend([Dies("a"), b])
        got = []
        with self.assertRaises(MakiNetworkError):
            for chunk in fb.stream("hi"):
                got.append(chunk)
        self.assertEqual(got, ["part"])
        self.assertEqual(b.calls, 0)

    def test_all_failing_raises(self):
        fb = FallbackBackend([Fake("a", error=NET), Fake("b", error=NET)])
        with self.assertRaises(MakiNetworkError):
            list(fb.stream("hi"))


class TestToolLoop(unittest.TestCase):
    SPECS = [{"name": "p__m", "description": "d", "parameters": {"type": "object", "properties": {}}}]
    CALL = ToolCall(id="1", name="p__m", args={"x": 1})

    def test_each_backend_gets_its_own_schema_format(self):
        a, b = Fake("a", error=NET, native=True), Fake("b", native=True)
        fb = FallbackBackend([a, b])
        fb.chat_with_tools([{"role": "user", "content": "go"}], fb.to_tool_schemas(self.SPECS))
        self.assertEqual(b.seen[0][1], [{"wire": "b", "name": "p__m"}])

    def test_non_native_backends_are_skipped_in_a_tool_loop(self):
        plain, native = Fake("plain"), Fake("n", native=True)
        fb = FallbackBackend([plain, native])
        response, _, _ = fb.chat_with_tools([{"role": "user", "content": "go"}], fb.to_tool_schemas(self.SPECS))
        self.assertEqual(response.content, "n")
        self.assertEqual(plain.calls, 0)
        with self.assertRaises(NotImplementedError):
            FallbackBackend([plain]).chat_with_tools([], [])

    def test_continues_natively_on_the_same_backend(self):
        a = Fake("a", native=True, calls=[[self.CALL]])
        fb = FallbackBackend([a, Fake("b", native=True)])
        msgs = [{"role": "user", "content": "go"}]
        _, calls, msgs = fb.chat_with_tools(msgs, fb.to_tool_schemas(self.SPECS))
        msgs = fb.append_tool_results(msgs, [(calls[0], "42")])
        response, _, _ = fb.chat_with_tools(msgs, fb.to_tool_schemas(self.SPECS))
        self.assertEqual(response.content, "a")
        wire = a.seen[1][0]
        self.assertEqual([m.get("from") or m["role"] for m in wire], ["user", "a", "a"])

    def test_switching_mid_loop_folds_the_tool_exchange_into_text(self):
        a = Fake("a", native=True, calls=[[self.CALL]])
        b = Fake("b", native=True)
        fb = FallbackBackend([a, b])
        specs = fb.to_tool_schemas(self.SPECS)
        _, calls, msgs = fb.chat_with_tools([{"role": "user", "content": "go"}], specs)
        msgs = fb.append_tool_results(msgs, [(calls[0], "RESULT-42")])
        a.error, a.fail_times, a.calls = NET, None, 0  # a goes down before the next round
        response, _, _ = fb.chat_with_tools(msgs, specs)
        self.assertEqual(response.content, "b")
        wire = b.seen[0][0]
        self.assertEqual(len(wire), 1)  # no foreign tool/assistant entries
        self.assertEqual(wire[0]["role"], "user")
        self.assertIn("go", wire[0]["content"])
        self.assertIn('p__m({"x": 1})', wire[0]["content"])
        self.assertIn("RESULT-42", wire[0]["content"])

    def test_final_round_without_tools_still_works_after_a_switch(self):
        a = Fake("a", native=True, calls=[[self.CALL]])
        b = Fake("b", native=True)
        fb = FallbackBackend([a, b])
        specs = fb.to_tool_schemas(self.SPECS)
        _, calls, msgs = fb.chat_with_tools([{"role": "user", "content": "go"}], specs)
        msgs = fb.append_tool_results(msgs, [(calls[0], "r")])
        a.error, a.fail_times, a.calls = NET, None, 0
        response, calls, _ = fb.chat_with_tools(msgs, [])
        self.assertEqual((response.content, calls), ("b", None))
        self.assertEqual(b.seen[0][1], [])

    def test_append_needs_a_transcript(self):
        fb = FallbackBackend([Fake("a", native=True)])
        with self.assertRaises(ValueError):
            fb.append_tool_results([{"role": "user", "content": "x"}], [])


class Echo:
    """Plugin whose method the model calls; counts executions."""
    ALLOWED_METHODS = {"say"}
    runs = 0

    def say(self, text: str) -> str:
        """Echo.

        Args:
            text: what to echo
        """
        Echo.runs += 1
        return f"echo:{text}"


class TestWithAgent(unittest.TestCase):
    def test_tool_runs_once_when_the_backend_changes_mid_turn(self):
        Echo.runs = 0
        call = ToolCall(id="1", name="echo__say", args={"text": "hi"})
        a = Fake("a", native=True, calls=[[call]])
        b = Fake("b", native=True)
        fb = FallbackBackend([a, b])
        agent = Agent("t", fb)
        agent.plugins["echo"] = Echo()
        real = a.chat_with_tools

        def flaky(messages, tools, **kw):
            if any(m.get("role") == "tool" for m in messages):
                raise NET  # a dies on the round after the tool ran
            return real(messages, tools, **kw)
        a.chat_with_tools = flaky
        answer = agent.execute_task("say hi", use_plugins=True)
        self.assertEqual(answer, "b")
        self.assertEqual(Echo.runs, 1)
        self.assertIn("echo:hi", b.seen[0][0][0]["content"])

    def test_agent_counts_usage_of_the_serving_backend(self):
        fb = FallbackBackend([Fake("a", error=NET), Fake("b")])
        agent = Agent("t", fb)
        agent.execute_task("hi")
        self.assertEqual(agent.usage.calls, 1)


if __name__ == "__main__":
    unittest.main()
