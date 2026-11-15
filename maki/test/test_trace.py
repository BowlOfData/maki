"""Run traces: span tree, ids across threads and processes, file handling, reader and CLI."""
import io
import json
import os
import stat
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from maki.agents import Agent
from maki.agents.tool_policy import ToolPolicy
from maki.exceptions import MakiNetworkError
from maki.objects import LLMResponse, ToolCall
from maki.trace import (TraceCallback, build_spans, current_trace, find_trace, list_traces, new_id, read_trace,
                        render_trace, run_cli, trace_context, valid_id)


def _r(content="", prompt=10, completion=5, cached=0):
    return LLMResponse(content=content, model="t", prompt_tokens=prompt, completion_tokens=completion,
                       total_tokens=prompt + completion, elapsed_seconds=0.0, cached_tokens=cached)


class Plugin:
    ALLOWED_METHODS = ["add", "boom", "flaky", "login"]

    def __init__(self):
        self.calls = 0

    def add(self, a: int) -> str:
        return str(a + 1)

    def boom(self) -> str:
        raise ValueError("bad")

    def flaky(self) -> str:
        self.calls += 1
        if self.calls < 2:
            raise MakiNetworkError("down")
        return "up"

    def login(self, user: str, api_key: str = "") -> str:
        return "ok"


def _agent(path, **kw):
    backend = MagicMock()
    backend.supports_native_tools = True
    backend.to_tool_schemas.return_value = []
    backend.append_tool_results.side_effect = lambda m, r: m
    agent = Agent("t", backend, trace=path, **kw)
    agent.plugins["p"] = Plugin()
    return agent


def _run(agent, calls, final="done"):
    agent.maki.chat_with_tools.side_effect = [(_r(""), calls, []), (_r(final), None, [])]
    return agent.execute_task("go", use_plugins=True)


class Base(unittest.TestCase):
    def setUp(self):
        from maki import trace as trace_module
        trace_module._stack.set(())  # a test that leaves a span open must not leak it into the next one
        self.addCleanup(trace_module._stack.set, ())
        self.addCleanup(trace_module.close_all)
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "runs.jsonl")

    def records(self):
        return read_trace(self.path)


class TestSpanTree(Base):

    def test_native_run_is_one_trace_with_the_right_parents(self):
        agent = _agent(self.path)
        _run(agent, [ToolCall("1", "p__add", {"a": 1})])
        records = self.records()
        self.assertEqual(len({r["trace"] for r in records}), 1)
        roots = build_spans(records)[records[0]["trace"]]
        self.assertEqual(len(roots), 1)
        task = roots[0]
        self.assertEqual((task.kind, task.status), ("task", "ok"))
        self.assertEqual([(c.kind, c.status) for c in task.children], [("model", "ok"), ("tool", "ok"), ("model", "ok")])
        model, tool, _ = task.children
        self.assertEqual((model.end["tool_calls"], model.end["prompt_tokens"], model.end["completion_tokens"]),
                         (["p__add"], 10, 5))
        self.assertEqual((tool.start["tool"], tool.end["result"]), ("p__add", "2"))
        self.assertIsNotNone(task.seconds)
        self.assertIsNone(current_trace(), "no span stays open after the run")

    def test_task_error_marks_the_span_and_closes_unfinished_children(self):
        agent = _agent(self.path)
        agent.maki.chat_with_tools.side_effect = RuntimeError("model down")
        with self.assertRaises(Exception):
            agent.execute_task("go", use_plugins=True)
        records = self.records()
        spans = build_spans(records)[records[0]["trace"]]
        self.assertEqual(spans[0].status, "error")
        self.assertIn("model down", spans[0].end["error"])
        for child in spans[0].children:
            self.assertNotEqual(child.status, "running")
        self.assertIsNone(current_trace())

    def test_failing_tool_and_retries(self):
        agent = _agent(self.path, tool_policy=ToolPolicy(retries=2, initial_delay=0, jitter=False))
        _run(agent, [ToolCall("1", "p__flaky", {}), ToolCall("2", "p__boom", {})])
        tools = [s for s in self._flatten() if s.kind == "tool"]
        self.assertEqual({t.start["tool"]: (t.status, t.retries) for t in tools},
                         {"p__flaky": ("ok", 1), "p__boom": ("error", 0)})

    def _flatten(self):
        out = []

        def walk(span):
            out.append(span)
            for c in span.children:
                walk(c)
        for roots in build_spans(self.records()).values():
            for r in roots:
                walk(r)
        return out

    def test_two_runs_are_two_traces_and_concurrent_runs_do_not_mix(self):
        agents = [_agent(self.path) for _ in range(6)]
        errors = []

        def work(agent):
            try:
                _run(agent, [ToolCall("1", "p__add", {"a": 1})])
            except Exception as exc:  # pragma: no cover
                errors.append(exc)
        threads = [threading.Thread(target=work, args=(a,)) for a in agents]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(errors, [])
        roots = build_spans(self.records())
        self.assertEqual(len(roots), 6)
        for spans in roots.values():
            self.assertEqual(len(spans), 1)
            self.assertEqual([c.kind for c in spans[0].children], ["model", "tool", "model"])

    def test_two_callbacks_on_one_agent_keep_separate_stacks(self):
        other = os.path.join(self.dir.name, "second.jsonl")
        agent = _agent(self.path, callbacks=[TraceCallback(other)])
        _run(agent, [ToolCall("1", "p__add", {"a": 1})])
        for path in (self.path, other):
            records = read_trace(path)
            spans = build_spans(records)[records[0]["trace"]]
            self.assertEqual([c.kind for c in spans[0].children], ["model", "tool", "model"])

    def test_streaming_spans_survive_a_change_of_thread(self):
        agent = _agent(self.path)
        agent.maki.stream.side_effect = lambda *a, **k: iter(["a", "b"])
        chunks = agent.stream_task("hello")
        first = []
        t = threading.Thread(target=lambda: first.append(next(chunks)))
        t.start()
        t.join()
        rest = list(chunks)  # finishes on this thread: the contextvars of the first one are gone
        self.assertEqual(first + rest, ["a", "b"])
        records = self.records()
        task = build_spans(records)[records[0]["trace"]][0]
        self.assertEqual((task.kind, task.status), ("task", "ok"))


class TestContentAndFiles(Base):

    def test_arguments_are_clipped_and_credentials_masked(self):
        agent = _agent(self.path, callbacks=[])
        agent.callbacks[:] = [TraceCallback(self.path, max_chars=40)]
        _run(agent, [ToolCall("1", "p__login", {"user": "ann", "api_key": "sk-verysecret"})])
        text = open(self.path).read()
        self.assertNotIn("sk-verysecret", text)
        self.assertIn("***", text)
        agent2 = _agent(os.path.join(self.dir.name, "c.jsonl"))
        agent2.callbacks[:] = [TraceCallback(os.path.join(self.dir.name, "c.jsonl"), max_chars=10)]
        agent2.maki.chat_with_tools.side_effect = [(_r("x" * 100), None, [])]
        agent2.execute_task("y" * 100, use_plugins=True)
        task = [r for r in read_trace(os.path.join(self.dir.name, "c.jsonl")) if r["kind"] == "task"][0]
        self.assertTrue(task["task"].startswith("y" * 10 + "…[+90"))

    def test_include_content_false_keeps_only_timing_and_names(self):
        agent = _agent(self.path)
        agent.callbacks[:] = [TraceCallback(self.path, include_content=False)]
        _run(agent, [ToolCall("1", "p__add", {"a": 1})], final="the secret answer")
        text = open(self.path).read()
        for leaked in ("secret answer", '\\"a\\": 1', '"go"'):
            self.assertNotIn(leaked, text)
        self.assertIn("p__add", text)

    @unittest.skipIf(os.name == "nt", "POSIX permissions")
    def test_file_is_private(self):
        _run(_agent(self.path), [ToolCall("1", "p__add", {"a": 1})])
        self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o600)

    def test_file_rolls_over_at_max_bytes(self):
        agent = _agent(self.path)
        agent.callbacks[:] = [TraceCallback(self.path, max_bytes=1500)]
        for _ in range(8):
            agent.maki.chat_with_tools.side_effect = [(_r("ok"), None, [])]
            agent.execute_task("go", use_plugins=True)
        self.assertTrue(os.path.exists(self.path + ".1"))
        self.assertLessEqual(os.path.getsize(self.path), 1500)

    def test_unwritable_path_never_breaks_the_run(self):
        agent = _agent(self.dir.name)  # a directory, not a file
        with self.assertLogs("maki.trace", "WARNING") as logs:
            self.assertEqual(_run(agent, [ToolCall("1", "p__add", {"a": 1})]), "done")
        self.assertEqual(len(logs.records), 1, "one warning, not one per event")

    def test_creates_missing_directories(self):
        path = os.path.join(self.dir.name, "a", "b", "t.jsonl")
        _run(_agent(path), [ToolCall("1", "p__add", {"a": 1})])
        self.assertTrue(os.path.exists(path))

    def test_bad_arguments(self):
        for kwargs in ({"max_chars": 0}, {"max_bytes": -1}):
            with self.assertRaises(ValueError):
                TraceCallback(self.path, **kwargs)

    def test_agent_trace_argument_forms(self):
        cb = TraceCallback(self.path)
        self.assertIn(cb, Agent("a", MagicMock(), trace=cb).callbacks)
        self.assertTrue(any(isinstance(c, TraceCallback) for c in Agent("a", MagicMock(), trace=self.path).callbacks))
        self.assertFalse(Agent("a", MagicMock(), trace=None).callbacks)
        self.assertFalse(Agent("a", MagicMock(), trace=False).callbacks)


class TestWriterDetails(Base):

    def test_credentials_are_masked_at_any_depth_and_depth_is_bounded(self):
        from maki.trace import _scrub
        nested = {"outer": [{"password": "p", "keep": 1}, ("tuple", {"Authorization": "Bearer x"})], "ok": "v"}
        self.assertEqual(_scrub(nested), {"outer": [{"password": "***", "keep": 1}, ["tuple", {"Authorization": "***"}]],
                                          "ok": "v"})
        deep = cur = {}
        for _ in range(12):
            cur["x"] = {}
            cur = cur["x"]
        self.assertIn("\\u2026", json.dumps(_scrub(deep)))
        self.assertEqual(_scrub("plain"), "plain")
        self.assertEqual(_scrub(5), 5)

    def test_ids_and_record_shape(self):
        self.assertEqual(len(new_id()), 16)
        self.assertNotEqual(new_id(), new_id())
        cb = TraceCallback(self.path)
        agent = SimpleNamespace(name="a")
        cb.on_task_start(agent, "t")
        cb.on_task_end(agent, "t", "r", 0.0)
        start, end = self.records()
        self.assertEqual(start["v"], 1)
        self.assertNotIn("seconds", start)
        self.assertIn("seconds", end)
        self.assertIsNone(start["parent"])

    def test_unclosed_children_are_closed_as_abandoned_when_the_task_ends(self):
        cb = TraceCallback(self.path)
        agent = SimpleNamespace(name="a")
        call = SimpleNamespace(name="p__add", args={"a": 1})
        cb.on_task_start(agent, "t")
        cb.on_model_start(agent, [1, 2])
        cb.on_tool_start(agent, call)
        cb.on_task_error(agent, "t", RuntimeError("x" * 400), 0.0)
        ends = {r["kind"]: r for r in self.records() if r["ph"] != "start"}
        self.assertEqual(ends["model"]["status"], "abandoned")
        self.assertEqual(ends["tool"]["status"], "abandoned")
        self.assertEqual(ends["task"]["status"], "error")
        self.assertLessEqual(len(ends["task"]["error"]), 300 + len("…[+999 chars]"))
        self.assertIsNone(current_trace())
        self.assertEqual(cb._open_tools, {})
        self.assertEqual(cb._open_tasks, {})

    def test_orphan_cleanup_keeps_other_owners_frames(self):
        other = os.path.join(self.dir.name, "o.jsonl")
        a, b = TraceCallback(self.path), TraceCallback(other)
        agent = SimpleNamespace(name="a")
        a.on_task_start(agent, "t")
        b.on_task_start(agent, "t")
        a.on_task_end(agent, "t", "r", 0.0)
        self.assertIsNotNone(current_trace(), "b's span is still open")
        b.on_task_end(agent, "t", "r", 0.0)
        self.assertIsNone(current_trace())

    def test_token_and_cache_fields_and_tool_errors(self):
        cb = TraceCallback(self.path)
        agent = SimpleNamespace(name="a")
        cb.on_task_start(agent, "t")
        cb.on_model_start(agent, [1])
        cb.on_model_end(agent, _r("x", prompt=7, completion=3, cached=5), None)
        cb.on_model_start(agent, [1])
        cb.on_model_end(agent, _r("x"), [SimpleNamespace(name="p__add")])
        call = SimpleNamespace(name="p__boom", args=None)
        cb.on_tool_start(agent, call)
        cb.on_tool_retry(agent, call, MakiNetworkError("down"), 1, 0.12345)
        cb.on_tool_error(agent, call, ValueError("bad " + "x" * 400), 0.0)
        cb.on_task_end(agent, "t", "r", 0.0)
        by = {}
        for r in self.records():
            by.setdefault((r["kind"], r["ph"]), []).append(r)
        first, second = by[("model", "end")]
        self.assertEqual((first["prompt_tokens"], first["completion_tokens"], first["cached_tokens"]), (7, 3, 5))
        self.assertNotIn("tool_calls", first)
        self.assertNotIn("cached_tokens", second)
        self.assertEqual(second["tool_calls"], ["p__add"])
        retry, = by[("tool", "retry")]
        self.assertEqual((retry["attempt"], retry["delay"]), (1, 0.123))
        self.assertIn("MakiNetworkError", retry["error"])
        self.assertNotIn("seconds", retry)
        err, = by[("tool", "error")]
        self.assertTrue(err["error"].startswith("ValueError: bad"))
        self.assertLessEqual(len(err["error"]), 300 + len("…[+999 chars]"))
        start, = by[("tool", "start")]
        self.assertNotIn("args", start)

    def test_unknown_calls_and_events_are_ignored(self):
        cb = TraceCallback(self.path)
        agent = SimpleNamespace(name="a")
        stray = SimpleNamespace(name="x", args={})
        cb.on_tool_end(agent, stray, "r", 0.0)
        cb.on_tool_error(agent, stray, ValueError("x"), 0.0)
        cb.on_tool_retry(agent, stray, ValueError("x"), 1, 0.0)
        cb.on_model_end(agent, None, None)
        cb.on_task_end(agent, "never started", "r", 0.0)
        cb.on_task_error(agent, "never started", ValueError("x"), 0.0)
        self.assertFalse(os.path.exists(self.path))

    def test_the_same_task_text_twice_is_matched_in_order(self):
        cb = TraceCallback(self.path)
        agent = SimpleNamespace(name="a")
        cb.on_task_start(agent, "same")
        cb.on_task_start(agent, "same")
        cb.on_task_end(agent, "same", "r1", 0.0)
        self.assertEqual(len(cb._open_tasks[(id(agent), "same")]), 1)
        cb.on_task_end(agent, "same", "r2", 0.0)
        self.assertEqual(cb._open_tasks, {})

    def test_close_then_write_reopens(self):
        cb = TraceCallback(self.path)
        agent = SimpleNamespace(name="a")
        cb.on_task_start(agent, "t")
        cb.close()
        cb.close()
        cb.on_task_end(agent, "t", "r", 0.0)
        self.assertEqual(len(self.records()), 2)

    def test_rollover_counts_the_existing_file(self):
        with open(self.path, "w") as f:
            f.write("x" * 900 + "\n")
        cb = TraceCallback(self.path, max_bytes=1000)
        agent = SimpleNamespace(name="a")
        for i in range(6):
            cb.on_task_start(agent, f"t{i}")
        self.assertTrue(os.path.exists(self.path + ".1"))

    def test_status_of_a_span_without_a_status_field(self):
        records = [_rec("s", "start", "task", task="t"), _rec("s", "error", "task", seconds=1.0)]
        self.assertEqual(build_spans(records)["T1"][0].status, "error")
        records = [_rec("s", "start", "task", task="t"), _rec("s", "end", "task", seconds=1.0)]
        self.assertEqual(build_spans(records)["T1"][0].status, "ok")

    def test_list_counts_only_model_ends_for_tokens_and_only_root_tasks_for_seconds(self):
        records = [
            _rec("s", "start", "task"), _rec("s", "end", "task", seconds=2.0, status="ok"),
            _rec("m", "start", "model", "s"), _rec("m", "end", "model", "s", prompt_tokens=4, completion_tokens=6),
            _rec("c", "start", "task", "s"), _rec("c", "end", "task", "s", seconds=1.0, status="ok"),
            _rec("t", "start", "tool", "s", tool="x"), _rec("t", "end", "tool", "s", prompt_tokens=99),
        ]
        (row,) = list_traces(records)
        self.assertEqual((row["tokens"], row["seconds"]), (10, 2.0))

    def test_model_span_text_without_token_counts(self):
        records = [_rec("s", "start", "model", messages=2), _rec("s", "end", "model", status="ok", seconds=0.1)]
        text = render_trace("T1", build_spans(records)["T1"])
        self.assertIn("2 messages  0.10s", text)
        self.assertNotIn("tokens", text)
        records = [_rec("s", "start", "model", messages=2),
                   _rec("s", "end", "model", status="ok", seconds=0.1, completion_tokens=3)]
        self.assertIn("0 in / 3 out tokens", render_trace("T1", build_spans(records)["T1"]))


class TestReviewFixes(Base):

    def test_parallel_tool_calls_stay_in_the_same_trace_under_the_task(self):
        agent = _agent(self.path, tool_policy=ToolPolicy(parallel=True, max_workers=4))
        _run(agent, [ToolCall("1", "p__add", {"a": 1}), ToolCall("2", "p__add", {"a": 2}), ToolCall("3", "p__add", {"a": 3})])
        records = self.records()
        self.assertEqual(len({r["trace"] for r in records}), 1, "pool threads must not start traces of their own")
        (task,) = build_spans(records)[records[0]["trace"]]
        self.assertEqual(sorted(c.kind for c in task.children), ["model", "model", "tool", "tool", "tool"])
        self.assertTrue(all(c.status == "ok" for c in task.children))
        self.assertIsNone(current_trace())

    def test_a_stream_abandoned_before_its_first_chunk_closes_its_span(self):
        agent = _agent(self.path)
        agent.maki.stream.side_effect = lambda *a, **k: iter(["a", "b"])
        chunks = agent.stream_task("hello")
        chunks.close()  # never started: the body has not run, nothing was opened
        self.assertFalse(os.path.exists(self.path))
        chunks = agent.stream_task("hello")
        next(chunks)
        chunks.close()
        self.assertIsNone(current_trace(), "no span is left on the stack to become the next task's parent")
        task = build_spans(self.records())
        (root,) = next(iter(task.values()))
        self.assertEqual(root.status, "ok")
        # An empty stream closes its span too, and the next task is a new trace, not a child of it.
        agent.maki.stream.side_effect = lambda *a, **k: iter([])
        list(agent.stream_task("empty"))
        agent.maki.stream.side_effect = lambda *a, **k: iter(["z"])
        list(agent.stream_task("next"))
        roots = [r for rs in build_spans(self.records()).values() for r in rs]
        self.assertEqual(len(roots), 3)
        self.assertTrue(all(r.status == "ok" and r.parent is None for r in roots))

    def test_urls_with_credentials_are_redacted_in_results_and_errors(self):
        cb = TraceCallback(self.path)
        agent = SimpleNamespace(name="a")
        call = SimpleNamespace(name="web__fetch", args={"url": "https://x.test/a?api_key=SECRET123&q=1"})
        cb.on_task_start(agent, "t")
        cb.on_tool_start(agent, call)
        cb.on_tool_error(agent, call, ValueError("GET https://x.test/a?token=TOPSECRET failed"), 0.0)
        cb.on_task_end(agent, "t", "see https://x.test/b?key=KEY999", 0.0)
        text = open(self.path).read()
        for leaked in ("SECRET123", "TOPSECRET", "KEY999"):
            self.assertNotIn(leaked, text)
        self.assertIn("q=1", text)

    def test_without_content_an_error_is_only_its_type(self):
        cb = TraceCallback(self.path, include_content=False)
        agent = SimpleNamespace(name="a")
        cb.on_task_start(agent, "t")
        cb.on_task_error(agent, "t", ValueError("contains the user's private text"), 0.0)
        text = open(self.path).read()
        self.assertNotIn("private", text)
        self.assertIn('"error": "ValueError"', text)

    def test_two_callbacks_on_one_path_share_the_file_and_the_rollover(self):
        a, b = TraceCallback(self.path, max_bytes=2000), TraceCallback(self.path, max_bytes=2000)
        self.assertIs(a._sink, b._sink)
        self.assertIsNot(a._sink, TraceCallback(self.path, max_bytes=999)._sink)
        agent = SimpleNamespace(name="a")
        for i in range(30):
            cb = a if i % 2 else b
            cb.on_task_start(agent, f"task {i}")
            cb.on_task_end(agent, f"task {i}", "r", 0.0)
        self.assertTrue(os.path.exists(self.path + ".1"))
        self.assertLessEqual(os.path.getsize(self.path), 2000)
        for path in (self.path, self.path + ".1"):
            for line in open(path):
                json.loads(line)  # no torn or interleaved lines

    def test_concurrent_writers_never_interleave_lines(self):
        cbs = [TraceCallback(self.path) for _ in range(4)]
        agent = SimpleNamespace(name="a")

        def work(cb, n):
            for i in range(50):
                cb.on_task_start(agent, f"{n}-{i}-" + "x" * 300)
                cb.on_task_end(agent, f"{n}-{i}-" + "x" * 300, "r", 0.0)
        threads = [threading.Thread(target=work, args=(cb, n)) for n, cb in enumerate(cbs)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(len(self.records()), 4 * 50 * 2)

    def test_trace_context_exited_in_another_context_does_not_raise(self):
        import asyncio

        async def gen():
            with trace_context("agen-trace"):
                yield 1
                yield 2

        async def main():
            g = gen()
            await g.__anext__()
            await g.aclose()  # closed from a different context than it was entered in

        asyncio.run(main())

    def test_shared_state_is_cleared_for_the_next_test(self):
        self.assertIsNone(current_trace())


class TestIds(unittest.TestCase):

    def test_valid_id(self):
        for good in ("abc", "A1._-", "x" * 64, new_id()):
            self.assertEqual(valid_id(good), good)
        for bad in ("", "x" * 65, "has space", "line\nbreak", "../x", None, 5, "a/b", "é", "abc\n", "\nabc", "abc\r"):
            self.assertIsNone(valid_id(bad), bad)

    def test_trace_context_joins_the_trace_and_unsafe_ids_are_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.jsonl")
            agent = _agent(path)
            agent.maki.chat_with_tools.side_effect = [(_r("ok"), None, [])]
            with trace_context("mytrace", "parentspan"):
                agent.execute_task("go", use_plugins=True)
            task = [r for r in read_trace(path) if r["kind"] == "task"][0]
            self.assertEqual((task["trace"], task["parent"]), ("mytrace", "parentspan"))
            agent.maki.chat_with_tools.side_effect = [(_r("ok"), None, [])]
            with trace_context("bad id\n", "x"):
                agent.execute_task("go", use_plugins=True)
            traces = {r["trace"] for r in read_trace(path)}
            self.assertEqual(len(traces), 2)
            self.assertTrue(all(valid_id(t) for t in traces))


class TestReader(Base):

    def test_reader_skips_torn_and_foreign_lines(self):
        _run(_agent(self.path), [ToolCall("1", "p__add", {"a": 1})])
        with open(self.path, "a") as f:
            f.write('{"trace": "x", "span"\n[1, 2]\nnot json\n{"no": "ids"}\n')
        records = self.records()
        self.assertTrue(all("trace" in r and "span" in r for r in records))
        self.assertEqual(len(list_traces(records)), 1)

    def test_list_and_render(self):
        _run(_agent(self.path), [ToolCall("1", "p__add", {"a": 1})])
        (row,) = list_traces(self.records())
        self.assertEqual((row["spans"], row["errors"], row["tokens"], row["agents"]), (4, 0, 30, ["t"]))
        text = render_trace(row["trace"], build_spans(self.records())[row["trace"]])
        for expected in ("task", "model", "tool", "p__add", "└─"):
            self.assertIn(expected, text)

    def test_find_trace_by_prefix(self):
        agent = _agent(self.path)
        for _ in range(2):
            agent.maki.chat_with_tools.side_effect = [(_r("ok"), None, [])]
            agent.execute_task("go", use_plugins=True)
        rows = list_traces(self.records())
        first, last = rows[0]["trace"], rows[-1]["trace"]
        self.assertEqual(find_trace(self.records()), last)
        self.assertEqual(find_trace(self.records(), first), first)
        self.assertEqual(find_trace(self.records(), first[:10]), first)
        self.assertIsNone(find_trace(self.records(), "zzzz"))
        self.assertIsNone(find_trace([]))
        with self.assertRaises(ValueError):
            find_trace(self.records(), "")  # the empty prefix matches both

    def test_a_span_whose_parent_is_in_another_file_is_a_root(self):
        agent = _agent(self.path)
        agent.maki.chat_with_tools.side_effect = [(_r("ok"), None, [])]
        with trace_context("t1", "elsewhere"):
            agent.execute_task("go", use_plugins=True)
        roots = build_spans(self.records())["t1"]
        self.assertEqual(len(roots), 1)
        self.assertIn("another file", render_trace("t1", roots))

    def test_an_unfinished_span_shows_as_running(self):
        cb = TraceCallback(self.path)
        agent = SimpleNamespace(name="a")
        cb.on_task_start(agent, "crashed here")
        (root,) = build_spans(self.records()).popitem()[1]
        self.assertEqual(root.status, "running")
        self.assertIn("running", render_trace(root.trace, [root]))
        cb.on_task_end(agent, "crashed here", "x", 0.0)


def _rec(span, ph, kind, parent=None, ts="2026-01-01T00:00:00.000+00:00", trace="T1", agent="a", **fields):
    return {"v": 1, "ts": ts, "trace": trace, "span": span, "parent": parent, "kind": kind, "ph": ph,
            "agent": agent, **fields}


FIXED = [
    _rec("s1", "start", "task", ts="2026-01-01T00:00:01", task="plan the trip"),
    _rec("s2", "start", "model", "s1", ts="2026-01-01T00:00:02", messages=3),
    _rec("s2", "end", "model", "s1", ts="2026-01-01T00:00:03", status="ok", seconds=1.5, prompt_tokens=100,
         completion_tokens=20, tool_calls=["web__search"]),
    _rec("s3", "start", "tool", "s1", ts="2026-01-01T00:00:04", tool="web__search", args='{"q": "x"}'),
    _rec("s3", "retry", "tool", "s1", ts="2026-01-01T00:00:05", attempt=1),
    _rec("s3", "error", "tool", "s1", ts="2026-01-01T00:00:06", status="error", seconds=0.25,
         tool="web__search", error="MakiNetworkError: down"),
    _rec("s4", "start", "tool", "s1", ts="2026-01-01T00:00:07", tool="web__fetch", args="{}"),
    _rec("s4", "end", "tool", "s1", ts="2026-01-01T00:00:08", status="ok", seconds=0.5, tool="web__fetch",
         result="page"),
    _rec("s1", "end", "task", ts="2026-01-01T00:00:09", status="ok", seconds=8.0, result="done"),
    _rec("o1", "start", "task", ts="2026-01-01T00:01:00", trace="T2", agent="b", task="other"),
    _rec("o1", "error", "task", ts="2026-01-01T00:01:01", trace="T2", agent="b", status="error", seconds=1.0,
         error="ValueError: x"),
]


class TestRenderingExactly(unittest.TestCase):

    def test_list_rows(self):
        rows = list_traces(FIXED)
        self.assertEqual(rows[0], {"trace": "T1", "start": "2026-01-01T00:00:01", "agents": ["a"], "spans": 4,
                                   "errors": 1, "tokens": 120, "seconds": 8.0})
        self.assertEqual((rows[1]["trace"], rows[1]["agents"], rows[1]["errors"], rows[1]["spans"],
                          rows[1]["tokens"], rows[1]["seconds"]), ("T2", ["b"], 1, 1, 0, 1.0))

    def test_tree_text(self):
        roots = build_spans(FIXED)["T1"]
        self.assertEqual(render_trace("T1", roots), "\n".join([
            "trace T1",
            "task   'plan the trip'  8.00s",
            "├─ model  3 messages 100 in / 20 out tokens -> web__search  1.50s",
            "├─ tool   web__search {\"q\": \"x\"}  (1 retry)  error  0.25s  MakiNetworkError: down",
            "└─ tool   web__fetch {}  0.50s  -> 'page'",
        ]))

    def test_span_properties(self):
        s1, = build_spans(FIXED)["T1"]
        self.assertEqual((s1.status, s1.seconds, len(s1.children)), ("ok", 8.0, 3))
        tool = s1.children[1]
        self.assertEqual((tool.status, tool.retries, tool.seconds), ("error", 1, 0.25))
        (o1,) = build_spans(FIXED)["T2"]
        self.assertEqual(o1.status, "error")
        lone = build_spans([FIXED[0]])["T1"][0]
        self.assertEqual((lone.status, lone.seconds), ("running", None))

    def test_retries_are_pluralised(self):
        records = [FIXED[3], FIXED[4], FIXED[4], FIXED[5]]
        records = [dict(r, parent=None) for r in records]
        self.assertIn("(2 retries)", render_trace("T1", build_spans(records)["T1"]))

    def test_width_cuts_long_lines(self):
        text = render_trace("T1", build_spans(FIXED)["T1"], width=40)
        for line in text.splitlines():
            self.assertLessEqual(len(line), 40)
        self.assertIn("…", text)
        self.assertNotIn("…", render_trace("T1", build_spans(FIXED)["T1"], width=400))

    def test_nested_prefixes(self):
        records = [
            _rec("a", "start", "task", task="outer"), _rec("a", "end", "task", status="ok", seconds=1.0),
            _rec("b", "start", "task", "a", task="mid"), _rec("b", "end", "task", "a", status="ok", seconds=1.0),
            _rec("c", "start", "tool", "b", tool="deep", args=""), _rec("c", "end", "tool", "b", status="ok", seconds=0.1),
            _rec("d", "start", "task", "a", task="last"), _rec("d", "end", "task", "a", status="ok", seconds=1.0),
        ]
        lines = render_trace("T1", build_spans(records)["T1"]).splitlines()
        self.assertEqual([line[:5] for line in lines[1:]], ["task ", "├─ ta", "│  └─", "└─ ta"])

    def test_cli_exit_codes_and_output(self):
        import argparse
        from maki.trace import add_cli_arguments
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "f.jsonl")
            with open(path, "w") as f:
                f.write("\n".join(json.dumps(r) for r in FIXED) + "\n")
            parser = argparse.ArgumentParser()
            add_cli_arguments(parser)
            out = io.StringIO()
            self.assertEqual(run_cli(parser.parse_args(["show", path, "--trace", "T1", "--width", "10"]), out=out), 0)
            self.assertTrue(all(len(line) <= 40 for line in out.getvalue().splitlines()), "width is floored at 40")
            out = io.StringIO()
            self.assertEqual(run_cli(parser.parse_args(["list", path]), out=out), 0)
            self.assertIn("4 spans", out.getvalue())
            self.assertIn("1 errors", out.getvalue())
            self.assertIn("120 tokens", out.getvalue())
            out = io.StringIO()
            self.assertEqual(run_cli(parser.parse_args(["show", path]), out=out), 0)
            self.assertIn("trace T2", out.getvalue(), "the latest trace is the default")
            self.assertEqual(run_cli(parser.parse_args(["show", path, "--trace", ""]), out=io.StringIO()), 1)
            out = io.StringIO()
            run_cli(parser.parse_args(["show", path, "--trace", "T1", "--json"]), out=out)
            self.assertEqual(len(out.getvalue().splitlines()), 9)
            empty = os.path.join(d, "empty.jsonl")
            open(empty, "w").close()
            out = io.StringIO()
            run_cli(parser.parse_args(["list", empty]), out=out)
            self.assertIn("no traces", out.getvalue())


class TestCli(Base):

    def _cli(self, *argv):
        import argparse
        from maki.trace import add_cli_arguments
        parser = argparse.ArgumentParser()
        add_cli_arguments(parser)
        out = io.StringIO()
        code = run_cli(parser.parse_args(list(argv)), out=out)
        return code, out.getvalue()

    def test_list_show_and_json(self):
        agent = _agent(self.path)
        for _ in range(2):
            agent.maki.chat_with_tools.side_effect = [(_r(""), [ToolCall("1", "p__add", {"a": 1})], []), (_r("ok"), None, [])]
            agent.execute_task("go", use_plugins=True)
        code, out = self._cli("list", self.path)
        self.assertEqual((code, len(out.strip().splitlines())), (0, 2))
        code, out = self._cli("show", self.path)
        self.assertEqual(code, 0)
        self.assertIn("p__add", out)
        first = list_traces(read_trace(self.path))[0]["trace"]
        code, out = self._cli("show", self.path, "--trace", first[:8], "--json")
        self.assertEqual(code, 0)
        self.assertTrue(all(json.loads(line)["trace"] == first for line in out.splitlines()))

    def test_errors_are_reported_not_raised(self):
        self.assertEqual(self._cli("show", os.path.join(self.dir.name, "missing.jsonl"))[0], 1)
        open(self.path, "w").close()
        self.assertEqual(self._cli("show", self.path)[0], 1)
        self.assertEqual(self._cli("list", self.path)[0], 0)
        _run(_agent(self.path), [ToolCall("1", "p__add", {"a": 1})])
        self.assertEqual(self._cli("show", self.path, "--trace", "nope")[0], 1)

    def test_no_subcommand(self):
        import argparse
        from maki.trace import add_cli_arguments
        parser = argparse.ArgumentParser()
        add_cli_arguments(parser)
        self.assertEqual(run_cli(parser.parse_args([])), 2)

    def test_main_dispatches_trace(self):
        import sys
        from unittest.mock import patch
        from maki import __main__ as cli
        _run(_agent(self.path), [ToolCall("1", "p__add", {"a": 1})])
        with patch.object(sys, "argv", ["maki", "trace", "list", self.path]), \
                patch("sys.stdout", new_callable=io.StringIO) as out, self.assertRaises(SystemExit) as raised:
            cli.main()
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("spans", out.getvalue())


# --- across the HTTP boundary -------------------------------------------------------------------------------

fastapi = pytest.importorskip("fastapi")


class TestServerAndProxy(Base):

    def _client(self):
        from fastapi.testclient import TestClient
        from maki.distributed.server import create_app
        backend = MagicMock()
        backend.chat.return_value = _r("remote answer")
        backend.stream.side_effect = lambda *a, **k: iter(["x", "y"])
        agent = Agent("srv", backend, trace=self.path)
        return TestClient(create_app(agent)), agent

    def test_incoming_ids_become_the_servers_trace(self):
        client, _ = self._client()
        response = client.post("/execute", json={"task": "hi"},
                               headers={"X-Maki-Trace-Id": "client-trace", "X-Maki-Parent-Span": "span-1"})
        self.assertEqual(response.headers["X-Maki-Trace-Id"], "client-trace")
        task = [r for r in self.records() if r["kind"] == "task"][0]
        self.assertEqual((task["trace"], task["parent"]), ("client-trace", "span-1"))

    def test_without_headers_the_servers_own_id_is_used_and_returned(self):
        client, _ = self._client()
        response = client.post("/execute", json={"task": "hi"})
        task = [r for r in self.records() if r["kind"] == "task"][0]
        self.assertEqual(response.headers["X-Maki-Trace-Id"], task["trace"])
        self.assertEqual(response.json()["trace_id"], task["trace"])
        self.assertIsNone(task["parent"])

    def test_hostile_ids_are_replaced_not_echoed(self):
        client, _ = self._client()
        for bad in ("x" * 200, "a b", "../../etc/passwd", "a,b"):
            response = client.post("/execute", json={"task": "hi"}, headers={"X-Maki-Trace-Id": bad})
            echoed = response.headers["X-Maki-Trace-Id"]
            self.assertNotEqual(echoed, bad)
            self.assertTrue(valid_id(echoed) or len(echoed) == 36)
        bad_parent = client.post("/execute", json={"task": "hi"},
                                 headers={"X-Maki-Trace-Id": "ok", "X-Maki-Parent-Span": "bad parent\n"})
        self.assertEqual(bad_parent.status_code, 200)
        self.assertIsNone([r for r in self.records() if r["trace"] == "ok" and r["kind"] == "task"][0]["parent"])

    def test_a_parent_span_without_a_valid_trace_id_is_ignored(self):
        client, _ = self._client()
        for headers in ({"X-Maki-Parent-Span": "orphan"},
                        {"X-Maki-Trace-Id": "bad id", "X-Maki-Parent-Span": "orphan"}):
            client.post("/execute", json={"task": "hi"}, headers=headers)
        tasks = [r for r in self.records() if r["kind"] == "task" and r["ph"] == "start"]
        self.assertEqual(len(tasks), 2)
        self.assertTrue(all(t["parent"] is None for t in tasks))

    def test_streaming_carries_the_ids_too(self):
        client, _ = self._client()
        response = client.post("/stream", json={"task": "hi"},
                               headers={"X-Maki-Trace-Id": "stream-trace", "X-Maki-Parent-Span": "sp"})
        self.assertIn("[DONE]", response.text)
        task = [r for r in self.records() if r["kind"] == "task" and r["ph"] == "start"][0]
        self.assertEqual((task["trace"], task["parent"]), ("stream-trace", "sp"))
        self.assertTrue([r for r in self.records() if r["ph"] == "end"], "the streamed task span was closed")

    def test_proxy_joins_the_callers_trace(self):
        from maki.distributed.proxy import AgentProxy
        proxy = AgentProxy.__new__(AgentProxy)
        proxy._api_key = None
        # Outside a traced run: a fresh id and no parent; an explicit id is used as given.
        self.assertIsNone(proxy._trace_for_request(None)[1])
        self.assertEqual(proxy._trace_for_request("given"), ("given", None))
        # Inside one: the run's trace under the open span.
        probe = TraceCallback(os.path.join(self.dir.name, "p.jsonl"))
        agent = SimpleNamespace(name="a")
        probe.on_task_start(agent, "outer")
        current = current_trace()
        self.assertEqual(proxy._trace_for_request(None), current)
        self.assertEqual(proxy._trace_for_request(current[0]), current)
        self.assertEqual(proxy._trace_for_request("other"), ("other", None))
        headers = proxy._headers(*current)
        self.assertEqual((headers["X-Maki-Trace-Id"], headers["X-Maki-Parent-Span"]), current)
        self.assertNotIn("X-Maki-Parent-Span", proxy._headers("t"))
        probe.on_task_end(agent, "outer", "x", 0.0)

    def test_end_to_end_client_and_server_files_join_into_one_tree(self):
        from fastapi.testclient import TestClient
        from maki.distributed.proxy import AgentProxy
        from maki.distributed.server import create_app
        client_path = os.path.join(self.dir.name, "client.jsonl")
        backend = MagicMock()
        backend.chat.return_value = _r("remote answer")
        test_client = TestClient(create_app(Agent("srv", backend, trace=self.path)))
        proxy = AgentProxy.__new__(AgentProxy)
        proxy._api_key = None
        local = Agent("local", MagicMock(), trace=client_path)
        local.maki.chat.return_value = _r("local")
        cb = [c for c in local.callbacks if isinstance(c, TraceCallback)][0]
        agent = SimpleNamespace(name="local")
        cb.on_task_start(agent, "delegate")
        trace, parent = proxy._trace_for_request(None)
        response = test_client.post("/execute", json={"task": "remote job"}, headers=proxy._headers(trace, parent))
        cb.on_task_end(agent, "delegate", "done", 0.0)
        self.assertEqual(response.status_code, 200)
        joined = read_trace(client_path, self.path)
        roots = build_spans(joined)
        self.assertEqual(list(roots), [trace])
        (outer,) = roots[trace]
        self.assertEqual((outer.agent, [c.agent for c in outer.children]), ("local", ["srv"]))


if __name__ == "__main__":
    unittest.main()
