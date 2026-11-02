"""
Tests for typed tool schemas, argument coercion, ToolPolicy (rounds, retries, parallelism)
and AgentCallback events.
"""

import threading
import time
import unittest
from enum import Enum
from typing import List, Literal, Optional
from unittest.mock import MagicMock

from maki.agents import Agent
from maki.agents.tool_policy import ToolPolicy
from maki.agents.tool_schema import coerce_args, parameters_schema
from maki.callbacks import AgentCallback, fire
from maki.exceptions import MakiNetworkError
from maki.objects import LLMResponse, ToolCall


def _r(content: str) -> LLMResponse:
    return LLMResponse(content=content, model="t", prompt_tokens=0, completion_tokens=0,
                       total_tokens=0, elapsed_seconds=0.0)


class Color(Enum):
    RED = "red"
    BLUE = "blue"


class Plugin:
    ALLOWED_METHODS = ["add", "search", "paint", "untyped", "flaky", "slow", "wipe", "fetch"]
    DANGEROUS_METHODS = ["wipe"]

    def __init__(self):
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def add(self, a: int, b: float = 1.0) -> str:
        """Add numbers.

        Args:
            a: first operand
            b: second operand,
                may be fractional

        Returns:
            the sum
        """
        return str(a + b)

    def search(self, query: str, tags: Optional[List[str]] = None, mode: Literal["fast", "deep"] = "fast"):
        return f"{query}|{tags}|{mode}"

    def paint(self, color: Color, loud: bool = False):
        return f"{color}|{loud}"

    def untyped(self, x, y=None):
        return f"{x}|{y}"

    def flaky(self):
        self.calls += 1
        if self.calls < 3:
            raise MakiNetworkError("down")
        return "up"

    async def fetch(self, n: int) -> str:
        return f"got {n}"

    def wipe(self):
        self.calls += 1
        raise MakiNetworkError("timed out")

    def slow(self):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)
        with self._lock:
            self.active -= 1
        return "done"


class TestSchema(unittest.TestCase):

    def test_types_and_required(self):
        s = parameters_schema(Plugin().add)
        self.assertEqual(s["properties"]["a"]["type"], "integer")
        self.assertEqual(s["properties"]["b"]["type"], "number")
        self.assertEqual(s["required"], ["a"])
        self.assertEqual(s["properties"]["b"]["default"], 1.0)

    def test_docstring_args_with_continuation(self):
        props = parameters_schema(Plugin().add)["properties"]
        self.assertEqual(props["a"]["description"], "first operand")
        self.assertEqual(props["b"]["description"], "second operand, may be fractional")

    def test_optional_list_literal_enum(self):
        props = parameters_schema(Plugin().search)["properties"]
        self.assertEqual(props["tags"], {"type": "array", "items": {"type": "string"}})
        self.assertEqual(props["mode"]["enum"], ["fast", "deep"])
        self.assertEqual(props["mode"]["type"], "string")
        self.assertEqual(parameters_schema(Plugin().paint)["properties"]["color"]["enum"], ["red", "blue"])

    def test_description_is_summary_paragraph(self):
        from maki.agents.tool_schema import method_description

        def f():
            """Search the index
            for matching notes.

            Args:
                x: y
            """
        self.assertEqual(method_description(f, "fb"), "Search the index for matching notes.")
        self.assertEqual(method_description(lambda: 0, "fb"), "fb")

    def test_unannotated_values_pass_through(self):
        self.assertEqual(coerce_args(Plugin().untyped, {"x": 5, "y": [1]})[0], {"x": 5, "y": [1]})

    def test_unannotated_stay_strings(self):
        props = parameters_schema(Plugin().untyped)["properties"]
        self.assertEqual(props["x"], {"type": "string"})
        self.assertEqual(parameters_schema(Plugin().untyped)["required"], ["x"])


class TestCoerce(unittest.TestCase):

    def test_string_numbers_converted(self):
        clean, err = coerce_args(Plugin().add, {"a": "3", "b": "2.5"})
        self.assertIsNone(err)
        self.assertEqual(clean, {"a": 3, "b": 2.5})

    def test_non_finite_numbers_rejected(self):
        for bad in ("nan", "inf", "-inf", float("nan"), float("inf")):
            self.assertIsNotNone(coerce_args(Plugin().add, {"a": 1, "b": bad})[1], bad)
        self.assertEqual(coerce_args(Plugin().add, {"a": 1, "b": "1e3"})[0]["b"], 1000.0)

    def test_unresolvable_annotation_does_not_lose_the_others(self):
        ns = {}
        exec("def f(a: int, b: 'Missing', c: 'int | None' = None): return a", ns)
        self.assertEqual(coerce_args(ns["f"], {"a": "2", "b": object.__name__, "c": "3"})[0],
                         {"a": 2, "b": "object", "c": 3})

    def test_bad_type_reports_argument(self):
        _, err = coerce_args(Plugin().add, {"a": "abc"})
        self.assertIn("'a'", err)
        self.assertIn("integer", err)

    def test_missing_and_unknown(self):
        _, err = coerce_args(Plugin().add, {"zzz": 1})
        self.assertIn("unknown argument 'zzz'", err)
        self.assertIn("missing required argument(s): a", err)

    def test_fractional_float_rejected_for_integer(self):
        self.assertIsNotNone(coerce_args(Plugin().add, {"a": 2.5})[1])
        self.assertEqual(coerce_args(Plugin().add, {"a": 2.0})[0], {"a": 2})

    def test_bool_is_not_an_integer(self):
        self.assertIsNotNone(coerce_args(Plugin().add, {"a": True})[1])

    def test_enum_and_list_item_errors(self):
        self.assertIsNotNone(coerce_args(Plugin().search, {"query": "q", "mode": "slow"})[1])
        _, err = coerce_args(Plugin().search, {"query": "q", "tags": ["a", {"x": 1}]})
        self.assertIn("item 1", err)

    def test_pep604_union_typed(self):
        def f(a: int | None = None, b: str | int = "x"):
            return a
        props = parameters_schema(f)["properties"]
        self.assertEqual(props["a"]["type"], "integer")
        self.assertEqual(props["b"]["anyOf"], [{"type": "string"}, {"type": "integer"}])
        self.assertIsNotNone(coerce_args(f, {"a": "abc"})[1])
        self.assertEqual(coerce_args(f, {"a": "4"})[0], {"a": 4})

    def test_enum_value_becomes_member(self):
        self.assertEqual(coerce_args(Plugin().paint, {"color": "blue"})[0], {"color": Color.BLUE})
        self.assertIsNotNone(coerce_args(Plugin().paint, {"color": "green"})[1])

        def f(c: Optional[Color] = None):
            return c
        self.assertEqual(coerce_args(f, {"c": "red"})[0], {"c": Color.RED})
        self.assertEqual(coerce_args(f, {"c": None})[0], {"c": None})

    def test_enums_nested_in_lists(self):
        def f(colors: List[Color]):
            return colors
        self.assertEqual(coerce_args(f, {"colors": ["red", "blue"]})[0], {"colors": [Color.RED, Color.BLUE]})
        self.assertIn("item 1", coerce_args(f, {"colors": ["red", "pink"]})[1])

    def test_null_means_default_or_error(self):
        clean, err = coerce_args(Plugin().add, {"a": 1, "b": None})
        self.assertEqual((clean, err), ({"a": 1}, None))  # b falls back to its default
        _, err = coerce_args(Plugin().add, {"a": None})
        self.assertIn("'a' must not be null", err)
        self.assertNotIn("missing", err)

    def test_none_accepted_for_optional(self):
        self.assertIsNone(coerce_args(Plugin().search, {"query": "q", "tags": None})[1])

    def test_var_kwargs_accept_extras(self):
        def f(a: int, **kw):
            return a
        clean, err = coerce_args(f, {"a": "1", "other": 5})
        self.assertIsNone(err)
        self.assertEqual(clean, {"a": 1, "other": 5})


def _agent(plugin=None, **kw) -> Agent:
    backend = MagicMock()
    backend.supports_native_tools = True
    agent = Agent("t", backend, **kw)
    agent.plugins["p"] = plugin or Plugin()
    return agent


class TestExecution(unittest.TestCase):

    def test_arguments_coerced_before_call(self):
        out = _agent()._execute_tool_call(ToolCall("", "p__add", {"a": "2", "b": 3}))
        self.assertEqual(out, "5")

    def test_invalid_arguments_returned_to_model_not_called(self):
        class Spy(Plugin):
            ran = False

            def add(self, a: int) -> str:
                Spy.ran = True
                return "x"

        out = _agent(Spy())._execute_tool_call(ToolCall("", "p__add", {"a": "nope"}))
        self.assertTrue(out.startswith("Error: Invalid arguments"))
        self.assertFalse(Spy.ran)

    def test_async_method_is_awaited(self):
        self.assertEqual(_agent()._execute_tool_call(ToolCall("", "p__fetch", {"n": "7"})), "got 7")

    def test_async_method_inside_running_loop(self):
        import asyncio

        async def main():
            return _agent()._execute_tool_call(ToolCall("", "p__fetch", {"n": 1}))
        self.assertEqual(asyncio.run(main()), "got 1")

    def test_helper_thread_tokens_are_credited_to_caller(self):
        from maki.usage import Usage

        class Spender(Plugin):
            ALLOWED_METHODS = ["spend", "aspend"]

            def __init__(self, usage):
                super().__init__()
                self.usage = usage

            def spend(self):
                self.usage.add(5, 5, 10)
                return "ok"

            async def aspend(self):
                self.usage.add(1, 1, 2)
                return "ok"

        agent = _agent(tool_policy=ToolPolicy(parallel=True))
        agent.maki.usage = Usage()
        agent.plugins["p"] = Spender(agent.maki.usage)
        agent._execute_tool_calls([ToolCall("1", "p__spend", {}), ToolCall("2", "p__spend", {})])
        agent._execute_tool_call(ToolCall("3", "p__aspend", {}))
        self.assertEqual(agent.maki.usage.total_tokens, 22)
        self.assertEqual(agent.maki.usage.thread_tokens(), 22)

    def test_retry_transient_error(self):
        plugin = Plugin()
        agent = _agent(plugin, tool_policy=ToolPolicy(retries=2, initial_delay=0, jitter=False))
        self.assertEqual(agent._execute_tool_call(ToolCall("", "p__flaky", {})), "up")
        self.assertEqual(plugin.calls, 3)

    def test_no_retry_by_default(self):
        out = _agent()._execute_tool_call(ToolCall("", "p__flaky", {}))
        self.assertIn("down", out)

    def test_retries_exhausted(self):
        agent = _agent(tool_policy=ToolPolicy(retries=1, initial_delay=0, jitter=False))
        self.assertIn("Error", agent._execute_tool_call(ToolCall("", "p__flaky", {})))

    def test_non_transient_error_not_retried(self):
        plugin = Plugin()
        plugin.flaky = MagicMock(side_effect=ValueError("bad"))
        agent = _agent(plugin, tool_policy=ToolPolicy(retries=3, initial_delay=0))
        agent._execute_tool_call(ToolCall("", "p__flaky", {}))
        self.assertEqual(plugin.flaky.call_count, 1)

    def test_parallel_runs_concurrently_and_keeps_order(self):
        plugin = Plugin()
        agent = _agent(plugin, tool_policy=ToolPolicy(parallel=True, max_workers=3))
        calls = [ToolCall(str(i), "p__slow", {}) for i in range(3)]
        results = agent._execute_tool_calls(calls)
        self.assertEqual([tc.id for tc, _ in results], ["0", "1", "2"])
        self.assertGreater(plugin.max_active, 1)

    def test_dangerous_method_not_retried_by_default(self):
        plugin = Plugin()
        agent = _agent(plugin, allow_dangerous_tools=True,
                       tool_policy=ToolPolicy(retries=3, initial_delay=0, jitter=False))
        agent._execute_tool_call(ToolCall("", "p__wipe", {}))
        self.assertEqual(plugin.calls, 1)

    def test_dangerous_method_retried_when_opted_in(self):
        plugin = Plugin()
        agent = _agent(plugin, allow_dangerous_tools=True,
                       tool_policy=ToolPolicy(retries=2, initial_delay=0, jitter=False, retry_dangerous=True))
        agent._execute_tool_call(ToolCall("", "p__wipe", {}))
        self.assertEqual(plugin.calls, 3)

    def test_turn_with_dangerous_call_runs_sequentially(self):
        plugin = Plugin()
        agent = _agent(plugin, allow_dangerous_tools=True, tool_policy=ToolPolicy(parallel=True))
        agent._execute_tool_calls([ToolCall("1", "p__slow", {}), ToolCall("2", "p__slow", {}),
                                   ToolCall("3", "p__wipe", {})])
        self.assertEqual(plugin.max_active, 1)

    def test_sequential_by_default(self):
        plugin = Plugin()
        _agent(plugin)._execute_tool_calls([ToolCall("", "p__slow", {}) for _ in range(3)])
        self.assertEqual(plugin.max_active, 1)

    def test_max_rounds_respected(self):
        agent = _agent(tool_policy=ToolPolicy(max_rounds=2))
        call = ToolCall("1", "p__add", {"a": 1})
        agent.maki.to_tool_schemas.return_value = []
        agent.maki.chat_with_tools.side_effect = [
            (_r(""), [call], []), (_r(""), [call], []), (_r("final"), None, [])]
        agent.maki.append_tool_results.side_effect = lambda m, r: m
        self.assertEqual(agent.execute_with_native_tools("go", None, "sys"), "final")
        self.assertEqual(agent.maki.chat_with_tools.call_count, 3)
        self.assertEqual(agent.maki.chat_with_tools.call_args[0][1], [])  # forced tool-less answer

    def test_policy_roundtrip_and_validation(self):
        agent = _agent(tool_policy={"retries": 2, "parallel": True})
        restored = Agent.from_dict(agent.to_dict(), agent.maki)
        self.assertEqual(restored.tool_policy.retries, 2)
        self.assertTrue(restored.tool_policy.parallel)
        with self.assertRaises(ValueError):
            ToolPolicy(max_rounds=0)
        with self.assertRaises(TypeError):
            Agent("x", MagicMock(), tool_policy="nope")

    def test_retry_on_normalised_and_validated(self):
        self.assertEqual(ToolPolicy(retry_on=KeyError).retry_on, (KeyError,))
        self.assertEqual(ToolPolicy(retry_on=[KeyError]).retry_on, (KeyError,))
        with self.assertRaises(ValueError):
            ToolPolicy(retry_on=("nope",))

    def test_from_dict_rejects_unknown_field(self):
        with self.assertRaises(ValueError):
            ToolPolicy.from_dict({"retrys": 2})

    def test_backoff_delay_capped(self):
        p = ToolPolicy(initial_delay=1, backoff_factor=2, max_delay=3, jitter=False)
        self.assertEqual([p.delay(i) for i in range(4)], [1, 2, 3, 3])

    def test_legacy_directive_path_validates_and_retries(self):
        plugin = Plugin()
        agent = _agent(plugin, tool_policy=ToolPolicy(retries=2, initial_delay=0, jitter=False))
        ok = agent._run_tool_call(True, "raw", {"plugin": "p", "method": "flaky", "args": {}})
        self.assertEqual(ok["result"], "up")
        bad = agent._run_tool_call(True, "raw", {"plugin": "p", "method": "add", "args": {"a": "x"}})
        self.assertIn("Invalid arguments", bad["error"])


class Recorder(AgentCallback):
    def __init__(self):
        self.events = []

    def on_task_start(self, agent, task):
        self.events.append("task_start")

    def on_task_end(self, agent, task, result, elapsed):
        self.events.append("task_end")

    def on_task_error(self, agent, task, error, elapsed):
        self.events.append("task_error")

    def on_model_start(self, agent, messages):
        self.events.append("model_start")

    def on_model_end(self, agent, response, tool_calls):
        self.events.append("model_end")

    def on_tool_start(self, agent, call):
        self.events.append("tool_start")

    def on_tool_end(self, agent, call, result, elapsed):
        self.events.append("tool_end")

    def on_tool_error(self, agent, call, error, elapsed):
        self.events.append("tool_error")

    def on_tool_retry(self, agent, call, error, attempt, delay):
        self.events.append(f"retry{attempt}")


class TestCallbacks(unittest.TestCase):

    def test_native_run_event_sequence(self):
        rec = Recorder()
        agent = _agent(callbacks=[rec])
        agent.maki.to_tool_schemas.return_value = []
        agent.maki.chat_with_tools.side_effect = [
            (_r(""), [ToolCall("1", "p__add", {"a": 1})], []), (_r("done"), None, [])]
        agent.maki.append_tool_results.side_effect = lambda m, r: m
        agent.execute_task("go", use_plugins=True)
        self.assertEqual(rec.events, [
            "task_start", "model_start", "model_end", "tool_start", "tool_end",
            "model_start", "model_end", "task_end"])

    def test_tool_error_and_retry_events(self):
        rec = Recorder()
        agent = _agent(callbacks=[rec], tool_policy=ToolPolicy(retries=1, initial_delay=0, jitter=False))
        agent._execute_tool_call(ToolCall("", "p__flaky", {}))
        self.assertEqual(rec.events, ["tool_start", "retry1", "tool_error"])

    def test_blocked_call_is_reported(self):
        rec = Recorder()
        agent = _agent(callbacks=[rec])  # allow_dangerous_tools is off, so wipe is blocked
        out = agent._execute_tool_call(ToolCall("", "p__wipe", {}))
        self.assertIn("dangerous", out)
        self.assertEqual(rec.events, ["tool_start", "tool_error"])

    def test_task_error_event(self):
        rec = Recorder()
        agent = _agent(callbacks=[rec])
        agent.maki.chat.side_effect = MakiNetworkError("x")
        with self.assertRaises(MakiNetworkError):
            agent.execute_task("go")
        self.assertEqual(rec.events, ["task_start", "task_error"])

    def test_failing_callback_does_not_break_run(self):
        class Bad(AgentCallback):
            def on_task_start(self, agent, task):
                raise RuntimeError("observer bug")
        agent = _agent(callbacks=[Bad()])
        agent.maki.chat.return_value = _r("fine")
        self.assertEqual(agent.execute_task("go"), "fine")

    def test_stream_task_events(self):
        rec = Recorder()
        agent = _agent(callbacks=[rec])
        agent.maki.stream.return_value = iter(["a", "b"])
        self.assertEqual("".join(agent.stream_task("go")), "ab")
        self.assertEqual(rec.events, ["task_start", "task_end"])

    def test_stream_task_error_event(self):
        def boom():
            yield "a"
            raise MakiNetworkError("cut")
        rec = Recorder()
        agent = _agent(callbacks=[rec])
        agent.maki.stream.return_value = boom()
        with self.assertRaises(MakiNetworkError):
            list(agent.stream_task("go"))
        self.assertEqual(rec.events, ["task_start", "task_error"])

    def test_fire_ignores_missing_handler(self):
        fire([object()], "on_anything")  # no AttributeError


if __name__ == "__main__":
    unittest.main()
