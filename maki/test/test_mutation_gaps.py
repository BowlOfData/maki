"""Behaviours the mutation check showed untested in context, skills, fallback and eval: exact limits and their
boundaries, budgeting arithmetic, ordering, report contents and defaults."""
import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from maki import FallbackBackend
from maki.context import (NUM_CTX_STEPS, ContextAware, ContextInfo, PromptSizeCheck, choose_num_ctx, estimate_tokens,
                          messages_chars)
from maki.eval import EvalCase, RunResult, _takes_agent, _verdicts, run_eval
from maki.exceptions import MakiAPIError, MakiNetworkError
from maki.skills import (DEFAULT_CATALOG_CHARS, MAX_LOAD_CHARS, MAX_READ_BYTES, MAX_SKILL_FILE_BYTES, SKILL_FILE,
                         SkillRegistry, _simple_yaml, _unquote, _words, load_skill, parse_skill_text)
from maki.test.test_eval import CASES, Scripted
from maki.test.test_fallback import Fake, NET


class TestContextBoundaries(unittest.TestCase):
    def test_every_step_is_reachable_when_the_cap_allows_it(self):
        self.assertEqual(NUM_CTX_STEPS, (8192, 16384, 32768, 65536, 131072))
        for need, expected in ((20_000, 32768), (26_300, 65536), (52_000, 65536), (52_500, 131072), (104_000, 131072)):
            with self.subTest(need=need):
                self.assertEqual(choose_num_ctx(need, cap=1_000_000), expected)

    def test_a_need_beyond_every_step_gets_the_limit(self):
        self.assertEqual(choose_num_ctx(500_000, cap=1_000_000), 1_000_000)
        self.assertEqual(choose_num_ctx(500_000, cap=1_000_000, model_max=300_000), 300_000)
        self.assertEqual(choose_num_ctx(500_000), 32768)

    def test_the_headroom_is_a_quarter(self):
        self.assertEqual(choose_num_ctx(6553), 8192)    # 6553 * 1.25 = 8191.25
        self.assertEqual(choose_num_ctx(6554), 16384)   # 8192.5 does not fit

    def test_default_thresholds_judge_from_512_estimated_tokens(self):
        check = PromptSizeCheck()
        self.assertEqual((check.ratio, check.min_tokens), (0.5, 512))
        for chars, flagged in ((512 * 4 - 4, False), (512 * 4, True)):  # 511 vs 512 estimated tokens
            with self.subTest(chars=chars):
                check.note_request(chars)
                self.assertEqual(check.check(1) is not None, flagged)

    def test_a_report_of_one_token_is_judged_but_zero_is_not(self):
        check = PromptSizeCheck()
        check.note_request(40_000)
        self.assertIsNotNone(check.check(1))
        self.assertIsNone(check.check(0))
        self.assertIsNone(check.check(-3))

    def test_the_ratio_boundary_is_inclusive_of_half(self):
        check = PromptSizeCheck()
        check.note_request(40_000)                  # 10,000 estimated tokens
        self.assertIsNone(check.check(5000))         # exactly half is fine
        self.assertIsNotNone(check.check(4999))

    def test_only_text_blocks_count_even_when_another_block_carries_a_text_field(self):
        content = [{"type": "tool_use", "text": "x" * 500}, {"type": "text", "text": "abc"}, "bare string", None,
                   {"type": "image_url", "image_url": {"url": "A" * 999}}]
        self.assertEqual(messages_chars([{"role": "user", "content": content}]), 3)

    def test_the_default_context_info_knows_nothing(self):
        class Plain(ContextAware):
            pass
        self.assertEqual(Plain().context_info(), ContextInfo())
        self.assertIsNone(Plain()._learn_window())

    def test_context_info_is_immutable_and_effective_prefers_what_is_loaded(self):
        import dataclasses
        info = ContextInfo(max_length=262144, loaded_length=8192)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            info.loaded_length = 1
        self.assertEqual(info.effective, 8192)
        self.assertEqual(ContextInfo(max_length=4096).effective, 4096)
        self.assertIsNone(ContextInfo().effective)

    def test_token_estimates_round_down(self):
        self.assertEqual([estimate_tokens(n) for n in (0, 3, 4, 7, 8)], [0, 0, 1, 1, 2])


def skill_folder(base, name, description="d", body="body", extra_files=None, raw=None):
    folder = Path(base) / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / SKILL_FILE).write_text(raw if raw is not None else f"---\nname: {name}\ndescription: {description}\n---\n{body}")
    for rel, content in (extra_files or {}).items():
        target = folder / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content if isinstance(content, bytes) else content.encode())
    return folder


def plugin_for(base):
    return SkillRegistry.discover(paths=[base], cwd=base, home=Path(base) / "no-home").plugin()


class TestSkillLimits(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()

    def test_instructions_are_returned_whole_up_to_the_limit_and_clipped_with_an_exact_count_beyond_it(self):
        skill_folder(self.root, "exact", body="x" * MAX_LOAD_CHARS)
        skill_folder(self.root, "over", body="x" * (MAX_LOAD_CHARS + 7))
        plugin = plugin_for(self.root)
        exact, over = plugin.load("exact"), plugin.load("over")
        self.assertEqual(len(exact), MAX_LOAD_CHARS)
        self.assertNotIn("omitted", exact)
        self.assertTrue(over.startswith("x" * MAX_LOAD_CHARS))
        self.assertIn("[... 7 more characters omitted ...]", over)

    def test_the_size_limit_of_skill_md_itself(self):
        prefix = "---\nname: big\ndescription: d\n---\n"
        ok = skill_folder(self.root, "big", raw=prefix + "x" * (MAX_SKILL_FILE_BYTES - len(prefix)))
        self.assertEqual((ok / SKILL_FILE).stat().st_size, MAX_SKILL_FILE_BYTES)
        self.assertEqual(load_skill(ok, "user").name, "big")
        too = skill_folder(self.root, "bigger", raw=prefix.replace("big", "bigger") + "x" * MAX_SKILL_FILE_BYTES)
        with self.assertRaisesRegex(ValueError, "larger than 256 KB"):
            load_skill(too, "user")

    def test_bundled_files_have_a_size_limit_with_its_own_message(self):
        skill_folder(self.root, "files", extra_files={"references/ok.txt": b"y" * MAX_READ_BYTES,
                                                      "references/big.txt": b"y" * (MAX_READ_BYTES + 1)})
        plugin = plugin_for(self.root)
        self.assertTrue(plugin.read("files", "references/ok.txt").startswith("yyy"))
        with self.assertRaisesRegex(ValueError, "larger than 1000 KB"):
            plugin.read("files", "references/big.txt")
        self.assertIn("more characters omitted", plugin.read("files", "references/ok.txt"))

    def test_the_file_list_is_capped_and_says_how_many_are_missing(self):
        for count, tail in ((40, None), (41, "... and 1 more"), (45, "... and 5 more")):
            with self.subTest(files=count):
                name = f"many{count}"
                skill_folder(self.root, name, extra_files={f"references/f{i:03d}.txt": "x" for i in range(count)})
                text = plugin_for(self.root).load(name)
                self.assertEqual(text.count("- references/"), min(count, 40))
                if tail:
                    self.assertIn(tail, text)
                else:
                    self.assertNotIn("more", text.split("skills.read")[1])

    def test_every_unsafe_path_shape_is_refused_on_its_own(self):
        skill_folder(self.root, "paths", extra_files={"references/a.txt": "a"})
        plugin = plugin_for(self.root)
        for bad in ("", "/abs/path", os.sep + "etc" + os.sep + "passwd", "..", "a/../b", "references/../../x", "./../x"):
            with self.subTest(path=bad), self.assertRaisesRegex(ValueError, "relative"):
                plugin.read("paths", bad)
        self.assertEqual(plugin.read("paths", "./references/a.txt"), "a")

    def test_the_registry_counts_its_skills(self):
        skill_folder(self.root, "one")
        skill_folder(self.root, "two")
        registry = SkillRegistry.discover(paths=[self.root], cwd=self.root, home=self.root / "none")
        self.assertEqual((len(registry), bool(registry)), (2, True))
        self.assertEqual((len(SkillRegistry()), bool(SkillRegistry())), (0, False))


class TestCatalogArithmetic(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()

    def registry(self, count, description, catalog_chars):
        for i in range(count):
            skill_folder(self.root, f"s{i:02d}", description)
        return SkillRegistry.discover(paths=[self.root], cwd=self.root, home=self.root / "none", catalog_chars=catalog_chars)

    def test_the_budget_is_spent_exactly(self):
        probe = self.registry(4, "d" * 20, 4000)
        header = probe.catalog().split("\n")[0] + "\n"
        line = len("- s00: " + "d" * 20) + 1                    # a line plus its newline
        for fits in (1, 2, 3, 4):
            for delta, expected in ((0, fits), (-1, fits - 1)):
                with self.subTest(fits=fits, delta=delta):
                    registry = SkillRegistry.discover(paths=[self.root], cwd=self.root, home=self.root / "none",
                                                      catalog_chars=max(200, len(header) + fits * line + delta))
                    listed = [x for x in registry.catalog().splitlines() if x.startswith("- ")]
                    self.assertEqual(len(listed), expected if len(header) + fits * line + delta >= 200 else len(listed))

    def test_a_description_is_clipped_exactly_at_the_limit(self):
        exact = self.registry(1, "w" * 300, 4000).catalog()
        self.assertIn("w" * 300, exact)
        self.assertNotIn("…", exact)
        for f in self.root.iterdir():
            for g in f.iterdir():
                g.unlink()
            f.rmdir()
        over = self.registry(1, "w" * 301, 4000).catalog()
        line = [x for x in over.splitlines() if x.startswith("- ")][0]
        self.assertTrue(line.endswith("…"))
        self.assertEqual(len(line), len("- s00: ") + 300)

    def test_the_catalog_budget_has_a_floor_and_a_default(self):
        self.assertEqual(self.registry(1, "d", DEFAULT_CATALOG_CHARS).catalog_chars, 4000)
        SkillRegistry.discover(paths=[self.root], cwd=self.root, home=self.root / "none", catalog_chars=200)
        with self.assertRaises(ValueError):
            SkillRegistry.discover(paths=[self.root], cwd=self.root, home=self.root / "none", catalog_chars=199)


class TestSkillParsingDetails(unittest.TestCase):
    def test_a_first_line_that_only_starts_with_dashes_is_not_front_matter(self):
        text = "--- not a delimiter\nname: x\n---\nbody"
        self.assertEqual(parse_skill_text(text), ({}, text))

    def test_unquote_only_strips_matching_pairs(self):
        for raw, expected in (('"a"', "a"), ("'a'", "a"), ('""', ""), ('"', '"'), ("'", "'"), ('"a\'', '"a\''), ("  'b' ", "b"),
                              ("plain", "plain")):
            with self.subTest(raw=raw):
                self.assertEqual(_unquote(raw), expected)

    def test_simple_yaml_shapes(self):
        data = _simple_yaml("a: 1\nempty:\nlist:\n  - x\n  # comment\n  -\n  - y\nmap:\n  k: v\n  # c\n  k2: 'w'\nfolded: >\n  one\n  two\n"
                            "literal: |\n  l1\n  l2\n#top\nbad line without colon\n  indented: ignored\nlast: end\n")
        self.assertEqual(data["a"], "1")
        self.assertEqual(data["empty"], "")
        self.assertEqual(data["map"], {"k": "v", "k2": "w"})
        self.assertEqual(data["folded"], "one two")
        self.assertEqual(data["literal"], "l1\nl2")
        self.assertEqual(data["last"], "end")
        self.assertNotIn("indented", data)
        self.assertEqual(_simple_yaml("list:\n  - x\n  - 'y z'\n")["list"], ["x", "y z"])

    def test_words(self):
        self.assertEqual(_words(None), [])
        self.assertEqual(_words("a  b\tc"), ["a", "b", "c"])
        self.assertEqual(_words(["a", 1, None]), ["a", "1", "None"])


class TestFallbackOrderingAndMessages(unittest.TestCase):
    def test_a_cooling_backend_in_the_middle_goes_to_the_back_and_the_order_of_the_rest_is_kept(self):
        a, b, c, d = Fake("a"), Fake("b", error=NET), Fake("c"), Fake("d")
        chain = FallbackBackend([a, b, c, d], cooldown=30)
        with patch("maki.fallback.time.monotonic", return_value=0.0):
            chain._failed(b, NET, True)
            self.assertEqual([x.label for x in chain._order(chain.backends)], ["a", "c", "d", "b"])
            chain._failed(a, NET, True)
            self.assertEqual([x.label for x in chain._order(chain.backends)], ["c", "d", "a", "b"])
            for backend in (c, d):
                chain._failed(backend, NET, True)
            self.assertEqual([x.label for x in chain._order(chain.backends)], ["a", "b", "c", "d"])  # all cooling: as given

    def test_the_log_says_whether_another_backend_will_be_tried(self):
        chain = FallbackBackend([Fake("a", error=NET), Fake("b", error=NET)], cooldown=0)
        with self.assertLogs("maki.fallback", "WARNING") as logs, self.assertRaises(MakiNetworkError):
            chain.request("q")
        self.assertIn("trying the next backend", logs.output[0])
        self.assertNotIn("trying the next backend", logs.output[1])

    def test_temperature_and_model_come_from_the_first_backend(self):
        first, second = Fake("one"), Fake("two")
        first.temperature, second.temperature = 0.25, 0.9
        chain = FallbackBackend([first, second])
        self.assertEqual((chain.model, chain.temperature), ("one", 0.25))

    def test_chat_collect_returns_the_answer_and_falls_back(self):
        chain = FallbackBackend([Fake("a", error=NET), Fake("b")], cooldown=0)
        self.assertEqual(chain.chat_collect("hi").content, "b")

    def test_a_stream_that_raises_not_implemented_after_a_chunk_is_not_skipped(self):
        class Breaks(Fake):
            def stream(self, *a, **k):
                yield "part"
                raise NotImplementedError("mid-stream")
        second = Fake("b")
        chain = FallbackBackend([Breaks("a"), second])
        with self.assertRaises(NotImplementedError):
            list(chain.stream("q"))
        self.assertEqual(second.calls, 0)

    def test_async_chat_raises_other_errors_without_touching_the_next_backend(self):
        calls = []

        class Async(Fake):
            def __init__(self, label, error=None):
                super().__init__(label)
                self.err = error

            async def async_chat(self, prompt, **kw):
                calls.append(self.label)
                if self.err:
                    raise self.err
                return MagicMock(content=self.label)
        chain = FallbackBackend([Async("a", MakiAPIError("bad", status_code=400)), Async("b")])
        with self.assertRaises(MakiAPIError):
            asyncio.run(chain.async_chat("q"))
        self.assertEqual(calls, ["a"])

    def test_async_chat_with_every_backend_down_raises_the_last_error(self):
        class Down(Fake):
            async def async_chat(self, prompt, **kw):
                raise NET
        with self.assertRaises(MakiNetworkError):
            asyncio.run(FallbackBackend([Down("a"), Down("b")], cooldown=0).async_chat("q"))

    def test_typed_turns_need_a_typed_backend_and_tools_need_a_native_one(self):
        from maki.objects import Message

        class Typed(Fake):
            supports_typed_messages = True
            supports_native_tools = False

            def chat_turn(self, messages, tool_specs=None, **kw):
                return MagicMock(content=self.label, tool_calls=None)
        plain_chain = FallbackBackend([Fake("a")])
        with self.assertRaises(NotImplementedError):
            plain_chain.chat_turn([Message("user", "q")])
        typed_only = FallbackBackend([Typed("t")])
        self.assertEqual(typed_only.chat_turn([Message("user", "q")]).content, "t")          # no tools: fine
        with self.assertRaises(NotImplementedError):                                        # tools need a native backend
            typed_only.chat_turn([Message("user", "q")], [{"name": "x", "description": "d", "parameters": {}}])

    def test_a_long_tool_result_is_clipped_when_a_conversation_is_handed_to_another_backend(self):
        from maki.fallback import _Transcript, _RESULT_CHARS
        from maki.objects import ToolCall
        call = ToolCall("1", "p__m", {})
        transcript = _Transcript([], None, [{"role": "user", "content": "q"}],
                                 [{"calls": [call], "results": {id(call): "R" * (_RESULT_CHARS + 500)}}])
        text = transcript.flattened()[0]["content"]
        self.assertEqual(text.count("R"), _RESULT_CHARS)
        self.assertEqual(_RESULT_CHARS, 4000)

    def test_the_legacy_transcript_numbers_tool_steps_from_one(self):
        from maki.fallback import _Transcript
        from maki.objects import ToolCall
        call = ToolCall("1", "p__m", {"a": 1})
        transcript = _Transcript([], None, [{"role": "user", "content": "q"}],
                                 [{"calls": [call], "results": {id(call): "R"}}, {"calls": [call], "results": {id(call): "S"}}])
        text = transcript.flattened()[0]["content"]
        self.assertIn("[1] p__m", text)
        self.assertIn("[2] p__m", text)
        self.assertNotIn("[0]", text)


class TestEvalReportDetails(unittest.TestCase):
    def report(self):
        return run_eval(CASES(), {"good": Scripted("a", {"Capital": "Paris", "17": "42"}), "bad": Scripted("b", {"Capital": "Lyon"})},
                        runs=2, warmup=False)

    def test_failures_can_be_filtered_by_model(self):
        report = self.report()
        self.assertEqual(len(report.failures()), 4)             # "good" passes everything; "bad" fails both cases twice
        self.assertTrue(all(r.target == "bad" for r in report.failures("bad")))
        self.assertEqual(report.failures("nobody"), [])
        self.assertEqual(report.failures("good"), [r for r in report.failures() if r.target == "good"])

    def test_pass_rate_by_case_and_overall(self):
        report = self.report()
        self.assertEqual(report.pass_rate("good", "sum"), 1.0)
        self.assertEqual(report.pass_rate("bad", "sum"), 0.0)
        self.assertEqual(report.pass_rate("good", "no-such-case"), 0.0)
        self.assertAlmostEqual(report.pass_rate("good"), report.pass_rate("good", "capital") / 2 + report.pass_rate("good", "sum") / 2)

    def test_the_matrix_cells_count_passes_per_case_and_model(self):
        text = self.report().table()
        row = [line for line in text.splitlines() if line.startswith("sum")][0]
        self.assertRegex(row, r"2/2\s+0/2")
        self.assertEqual(text.count("/2"), 4)

    def test_to_dict_and_save_are_complete_and_readable(self):
        report = self.report()
        data = report.to_dict()
        self.assertEqual(set(data), {"suite", "warmups", "results", "stats"})
        self.assertEqual(set(data["results"][0]), {"target", "case", "run", "passed", "detail", "seconds", "prompt_tokens",
                                                   "completion_tokens", "calls", "error", "output"})
        self.assertEqual({s["target"] for s in data["stats"]}, {"good", "bad"})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "r.json")
            report.save(path)
            raw = open(path).read()
        self.assertIn('\n  "suite"', raw)                       # indented, meant to be read
        self.assertEqual(json.loads(raw)["suite"], "custom")

    def test_run_result_defaults_and_token_sum(self):
        result = RunResult(target="t", case="c", run=1, passed=False)
        self.assertEqual((result.prompt_tokens, result.completion_tokens, result.calls, result.tokens), (0, 0, 0, 0))
        self.assertEqual(EvalCase("n", "t", contains="x").use_plugins, False)
        self.assertEqual(RunResult("t", "c", 1, True, prompt_tokens=3, completion_tokens=4).tokens, 7)

    def test_a_callable_that_cannot_be_inspected_is_not_given_the_agent(self):
        self.assertFalse(_takes_agent(len))
        self.assertFalse(_takes_agent(print))
        report = run_eval([EvalCase("c", "q", check=lambda a: True)], [Scripted("m")], warmup=False)
        self.assertTrue(report.results[0].passed)

    def test_verdict_parsing_needs_braces_or_a_clean_object(self):
        self.assertIsNone(_verdicts("no json here at all"))
        self.assertIsNone(_verdicts(""))
        entries = _verdicts('prose {"results": [{"id": "x", "pass": true}]} trailing prose')
        self.assertEqual(entries, [{"id": "x", "pass": True}])

    def test_a_check_that_raises_says_what_and_why(self):
        def boom(answer):
            raise KeyError("missing-thing")
        result = run_eval([EvalCase("c", "q", check=boom)], [Scripted("m")], warmup=False).results[0]
        self.assertEqual(result.error, "check raised KeyError: 'missing-thing'")

    def test_the_judge_error_message_and_stateful_judges(self):
        class Garbage(Scripted):
            def chat(self, prompt, **kw):
                return MagicMock(content="nope", prompt_tokens=0, completion_tokens=0, total_tokens=0)
        from maki import Agent
        result = run_eval([EvalCase("c", "q", criteria="x")], [Scripted("m")], judge=Agent("j", Garbage("g")),
                          warmup=False).results[0]
        self.assertEqual(result.error, "the judge gave no usable verdict")
        with self.assertRaisesRegex(ValueError, "must not be stateful"):
            run_eval([EvalCase("c", "q", criteria="x")], [Scripted("m")], judge=Agent("j", Garbage("g"), stateful=True),
                     warmup=False)

    def test_timings_are_measured_and_the_warm_up_is_timed_separately(self):
        import time

        class Slow(Scripted):
            def chat(self, prompt, **kw):
                time.sleep(0.02)
                return super().chat(prompt, **kw)
        report = run_eval([EvalCase("c", "q", contains="know")], {"m": Slow("m")})
        self.assertGreaterEqual(report.results[0].seconds, 0.02)
        self.assertGreaterEqual(report.warmups["m"], 0.02)
        self.assertLess(report.results[0].seconds, 1.0)

    def test_the_command_line_warms_up_unless_told_not_to(self):
        import argparse
        import io
        from contextlib import redirect_stdout

        from maki.eval import add_cli_arguments, run_cli
        for flag, warm in ((["--no-warmup"], False), ([], True)):
            backend = Scripted("m", {"PONG": "PONG"})
            parser = argparse.ArgumentParser()
            add_cli_arguments(parser)
            with patch("maki.backend_factory.init_backend", return_value=backend), redirect_stdout(io.StringIO()):
                run_cli(parser.parse_args(["--models", "ollama:x", "--suite", "basic", *flag]))
            self.assertEqual(any("Reply with the single word: OK" in p for p in backend.prompts), warm)

    def test_progress_lines_show_the_error_or_detail_clipped(self):
        import io
        from contextlib import redirect_stdout

        from maki.eval import _print_result
        out = io.StringIO()
        with redirect_stdout(out):
            _print_result(RunResult("m", "c", 1, False, detail="d" * 300, seconds=1.5))
            _print_result(RunResult("m", "c", 2, False, error="boom", detail="ignored", seconds=0.5))
            _print_result(RunResult("m", "c", 3, True, seconds=0.25))
        lines = out.getvalue().splitlines()
        self.assertEqual(lines[0], f"  [m] c #1: FAIL 1.5s - {'d' * 100}")
        self.assertEqual(lines[1], "  [m] c #2: ERROR 0.5s - boom")
        self.assertEqual(lines[2], "  [m] c #3: PASS 0.2s")


if __name__ == "__main__":
    unittest.main()
