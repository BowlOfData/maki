"""maki.eval: scoring, aggregation, error handling, grouping by model, suites and the CLI."""
import io
import json
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from maki import Agent, EvalCase, run_eval
from maki.backend import LLMBackend
from maki.eval import EvalReport, add_cli_arguments, run_cli
from maki.eval_suites import SUITES, get_suite
from maki.exceptions import MakiNetworkError
from maki.objects import LLMResponse


def reply(text, prompt_tokens=10, completion_tokens=5):
    return LLMResponse(content=text, model="m", prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                       total_tokens=prompt_tokens + completion_tokens, elapsed_seconds=0.0)


class Scripted(LLMBackend):
    """Answers by looking the prompt up in ``answers`` (substring match), else ``default``."""
    supports_history = True

    def __init__(self, label, answers=None, default="I do not know", fail_on=None):
        self.model, self.temperature, self.label = label, 0.0, label
        self.answers, self.default, self.fail_on = answers or {}, default, fail_on
        self.prompts = []

    def request(self, prompt):
        return self.chat(prompt)

    def chat(self, prompt, history=None, config=None, system=None, images=None):
        self.prompts.append(prompt)
        if self.fail_on and self.fail_on in prompt:
            raise MakiNetworkError("down")
        for needle, answer in self.answers.items():
            if needle in prompt:
                return reply(answer)
        return reply(self.default)


GOOD = Scripted("good", {"Capital": "Paris", "17": "42"})
CASES = lambda: [EvalCase("capital", "Capital of France?", contains="paris"),  # noqa: E731
                 EvalCase("sum", "What is 17 + 25?", contains=["42"])]


class TestCases(unittest.TestCase):
    def test_validation(self):
        for bad in (dict(name="", task="t", contains="x"), dict(name="n", task="", contains="x"),
                    dict(name="n", task="t"), dict(name="n", task="t", contains=[]),
                    dict(name="n", task="t", contains=[""]), dict(name="n", task="t", check="not callable"),
                    dict(name="n", task="t", criteria="  ")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                EvalCase(**bad)

    def test_a_single_substring_becomes_a_tuple(self):
        self.assertEqual(EvalCase("n", "t", contains="x").contains, ("x",))


class TestRun(unittest.TestCase):
    def test_scores_aggregates_and_keeps_usage(self):
        report = run_eval(CASES(), {"good": Scripted("a", {"Capital": "Paris", "17": "42"}),
                                    "bad": Scripted("b", {"Capital": "Lyon"})}, runs=2, warmup=False)
        self.assertEqual(len(report.results), 8)
        self.assertEqual(report.pass_rate("good"), 1.0)
        self.assertEqual(report.pass_rate("bad"), 0.0)
        self.assertEqual(report.pass_rate("bad", "capital"), 0.0)
        first = report.results[0]
        self.assertEqual((first.prompt_tokens, first.completion_tokens, first.calls), (10, 5, 1))
        self.assertTrue(first.passed and not first.error)
        failed = report.failures("bad")
        self.assertEqual(len(failed), 4)
        self.assertIn("missing", failed[0].detail)
        stats = {s.target: s for s in report.stats()}
        self.assertEqual((stats["good"].passed, stats["good"].runs, stats["bad"].pass_rate), (4, 4, 0.0))
        self.assertEqual(stats["good"].mean_tokens, 15)

    def test_contains_is_case_insensitive_and_needs_every_substring(self):
        backend = Scripted("m", default="The answer is PARIS in france")
        report = run_eval([EvalCase("a", "q", contains=["paris", "FRANCE"]), EvalCase("b", "q2", contains=["paris", "rome"])],
                          [backend], warmup=False)
        self.assertEqual([r.passed for r in report.results], [True, False])

    def test_check_callables_with_and_without_the_agent(self):
        seen = []
        cases = [EvalCase("plain", "q", check=lambda a: a == "I do not know"),
                 EvalCase("tuple", "q", check=lambda a: (False, "my reason")),
                 EvalCase("agent", "q", check=lambda a, agent: seen.append(type(agent).__name__) or True)]
        report = run_eval(cases, [Scripted("m")], warmup=False)
        self.assertEqual([r.passed for r in report.results], [True, False, True])
        self.assertEqual(report.results[1].detail, "my reason")
        self.assertEqual(seen, ["Agent"])

    def test_an_optional_second_parameter_is_not_given_the_agent(self):
        got = []
        cases = [EvalCase("optional", "q", check=lambda a, strict=False: got.append(("optional", strict)) or True),
                 EvalCase("named", "q", check=lambda a, agent=None: got.append(("named", type(agent).__name__)) or True),
                 EvalCase("required", "q", check=lambda a, whatever: got.append(("required", type(whatever).__name__)) or True),
                 EvalCase("varargs", "q", check=lambda a, *rest: got.append(("varargs", rest)) or True)]
        report = run_eval(cases, [Scripted("m")], warmup=False)
        self.assertTrue(all(r.passed for r in report.results), [r.error for r in report.results])
        self.assertEqual(got, [("optional", False), ("named", "Agent"), ("required", "Agent"), ("varargs", ())])

    def test_a_check_that_raises_is_an_error_not_a_model_failure(self):
        def boom(answer):
            raise RuntimeError("bug in the case")
        report = run_eval([EvalCase("c", "q", check=boom)], [Scripted("m")], warmup=False)
        result = report.results[0]
        self.assertFalse(result.passed)
        self.assertIn("check raised RuntimeError", result.error)

    def test_a_failing_model_call_is_recorded_and_the_rest_still_run(self):
        backend = Scripted("m", {"17": "42"}, fail_on="France")
        report = run_eval(CASES(), [backend], runs=2, warmup=False)
        self.assertEqual([bool(r.error) for r in report.results], [True, True, False, False])
        self.assertIn("MakiNetworkError", report.results[0].error)
        self.assertEqual(report.stats()[0].errors, 2)

    def test_each_run_gets_a_fresh_agent(self):
        agents = []

        def make(backend):
            agent = Agent("eval", backend, stateful=True)
            agents.append(agent)
            return agent
        run_eval([EvalCase("c", "q", contains="know")], [Scripted("m")], runs=3, make_agent=make, warmup=False)
        self.assertEqual(len(agents), 3)
        self.assertEqual(len({id(a) for a in agents}), 3)
        self.assertTrue(all(len(a._conversation_memory) == 2 for a in agents))  # one turn each, no carry-over

    def test_setup_turns_run_first_and_only_the_last_answer_is_scored(self):
        backend = Scripted("m", {"what was it": "bluefin", "remember": "noted"})
        case = EvalCase("mem", "what was it", contains="bluefin", setup_turns=["remember bluefin"])
        report = run_eval([case], [backend], warmup=False, make_agent=lambda b: Agent("e", b, stateful=True))
        self.assertTrue(report.results[0].passed)
        self.assertEqual(report.results[0].calls, 2)
        self.assertEqual(len(backend.prompts), 2)

    def test_a_case_can_bring_its_own_agent(self):
        marks = []
        case = EvalCase("c", "q", contains="know", make_agent=lambda b: marks.append(1) or Agent("own", b))
        run_eval([case], [Scripted("m")], warmup=False, make_agent=lambda b: marks.append(2) or Agent("suite", b))
        self.assertEqual(marks, [1])

    def test_models_run_one_after_another_and_warmup_is_separate(self):
        a, b = Scripted("a"), Scripted("b")
        report = run_eval(CASES(), {"a": a, "b": b}, runs=2)
        self.assertEqual([r.target for r in report.results], ["a"] * 4 + ["b"] * 4)
        self.assertIn("Reply with the single word: OK", a.prompts[0])  # warm-up came first, uncounted
        self.assertEqual(sum(r.calls for r in report.results if r.target == "a"), 4)
        self.assertEqual(set(report.warmups), {"a", "b"})
        self.assertNotIn("a", run_eval(CASES(), {"a": Scripted("a")}, warmup=False).warmups)

    def test_a_dead_model_during_warmup_does_not_stop_the_evaluation(self):
        report = run_eval([EvalCase("c", "q", contains="know")], [Scripted("m", fail_on="Reply with")])
        self.assertTrue(report.results[0].passed)
        self.assertEqual(report.warmups, {})

    def test_progress_callback(self):
        seen = []
        run_eval(CASES(), [GOOD], runs=1, warmup=False, on_result=seen.append)
        self.assertEqual([r.case for r in seen], ["capital", "sum"])

    def test_models_can_be_given_as_specs(self):
        from unittest.mock import patch
        with patch("maki.backend_factory.init_backend", return_value=Scripted("m")) as init:
            report = run_eval([EvalCase("c", "q", contains="know")], "ollama:qwen3.5:4b", warmup=False)
        init.assert_called_once_with("ollama:qwen3.5:4b")
        self.assertEqual(report.targets, ["ollama:qwen3.5:4b"])

    def test_argument_errors(self):
        for kwargs in (dict(targets=[]), dict(targets=[GOOD], runs=0), dict(targets=[GOOD], runs=True),
                       dict(targets=[object()]), dict(targets={"x": GOOD, "y": 5})):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                run_eval(CASES(), warmup=False, **kwargs)
        with self.assertRaises(ValueError):
            run_eval([], [GOOD])
        with self.assertRaises(ValueError):
            run_eval([EvalCase("a", "q", contains="x"), EvalCase("a", "q", contains="y")], [GOOD])
        with self.assertRaises(ValueError):  # two backends of the same class and model would collide
            run_eval(CASES(), [Scripted("m"), Scripted("m")])
        with self.assertRaises(ValueError):
            run_eval([EvalCase("c", "q", criteria="be nice")], [GOOD])  # needs a judge
        with self.assertRaises(ValueError):
            run_eval([EvalCase("c", "q", criteria="be nice")], [GOOD],
                     judge=SimpleNamespace(stateful=True))


class TestJudge(unittest.TestCase):
    def _judge(self, *entries, raw=None):
        class JudgeBackend(Scripted):
            def chat(self, prompt, **kw):
                self.prompts.append(prompt)
                return reply(raw if raw is not None else json.dumps({"results": list(entries), "block": ""}))
        backend = JudgeBackend("j")
        agent = Agent("judge", backend)
        agent.backend = backend
        return agent

    def _score(self, judge):
        return run_eval([EvalCase("c", "q", criteria="must be polite and short")], [Scripted("m")], judge=judge,
                        warmup=False).results[0]

    def test_criteria_cases_are_decided_by_the_judge(self):
        ok = self._score(self._judge({"id": "c1", "pass": True, "feedback": ""}))
        no = self._score(self._judge({"id": "c1", "pass": False, "feedback": "too rude"}))
        self.assertTrue(ok.passed)
        self.assertFalse(no.passed)
        self.assertIn("too rude", no.detail)

    def test_the_judge_sees_the_goal_the_criterion_and_the_answer(self):
        judge = self._judge({"id": "c1", "pass": True, "feedback": ""})
        self._score(judge)
        prompt = judge.backend.prompts[0]
        for needle in ("must be polite and short", "I do not know", "c1"):
            self.assertIn(needle, prompt)

    def test_entries_the_judge_made_up_ids_for_still_count(self):
        """Seen live: gemma split 'polite and short' into entries named 'polite' and 'length'."""
        both = self._score(self._judge({"id": "polite", "pass": True, "feedback": ""},
                                       {"id": "length", "pass": True, "feedback": ""}))
        one_fails = self._score(self._judge({"id": "polite", "pass": True, "feedback": ""},
                                            {"id": "length", "pass": False, "feedback": "too long"}))
        self.assertTrue(both.passed)
        self.assertFalse(one_fails.passed)
        self.assertEqual(one_fails.detail, "too long")

    def test_a_reply_wrapped_in_prose_or_fences_is_read(self):
        good = 'Verdict:\n```json\n{"results": [{"id": "c1", "pass": true, "feedback": ""}], "block": ""}\n```'
        self.assertTrue(self._score(self._judge(raw=good)).passed)

    def test_an_unusable_judge_is_an_error_and_is_retried_once(self):
        for raw in ("not json at all", "{}", '{"results": []}', '{"results": [{"id": "c1", "pass": "yes"}]}',
                    '{"results": [{"id": "c1"}]}', "[1, 2]"):
            with self.subTest(raw=raw):
                judge = self._judge(raw=raw)
                result = self._score(judge)
                self.assertIn("judge", result.error)
                self.assertFalse(result.passed)
                self.assertEqual(len(judge.backend.prompts), 2)

    def test_a_judge_that_raises_fails_closed(self):
        class Dead(Scripted):
            def chat(self, prompt, **kw):
                raise MakiNetworkError("down")
        with patch("maki.agents.agent.time.sleep"):  # skip the retry back-off
            result = self._score(Agent("j", Dead("d")))
        self.assertIn("judge", result.error)
        self.assertFalse(result.passed)


class TestReport(unittest.TestCase):
    def _report(self):
        return run_eval(CASES(), {"good": Scripted("a", {"Capital": "Paris", "17": "42"}), "bad": Scripted("b")},
                        runs=2, warmup=False)

    def test_table_has_a_row_per_model_and_a_matrix_per_case(self):
        text = self._report().table()
        self.assertIn("good", text)
        self.assertRegex(text, r"good\s+4/4\s+100%")
        self.assertRegex(text, r"bad\s+0/4\s+0%")
        self.assertRegex(text, r"capital\s+2/2\s+0/2")

    def test_json_round_trip(self):
        report = self._report()
        data = json.loads(json.dumps(report.to_dict()))
        self.assertEqual(len(data["results"]), 8)
        self.assertEqual({s["target"]: s["pass_rate"] for s in data["stats"]}, {"good": 1.0, "bad": 0.0})
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "r.json")
            report.save(path)
            self.assertEqual(json.load(open(path))["suite"], "custom")

    def test_empty_report_is_safe(self):
        report = EvalReport(suite="x")
        self.assertEqual(report.stats(), [])
        self.assertEqual(report.pass_rate("nobody"), 0.0)
        self.assertIn("suite: x", report.table())

    def test_long_outputs_are_truncated(self):
        report = run_eval([EvalCase("c", "q", contains="x")], [Scripted("m", default="x" * 5000)], warmup=False)
        self.assertEqual(len(report.results[0].output), 300)


class TestSuites(unittest.TestCase):
    def test_every_builtin_suite_builds_valid_cases(self):
        for name in SUITES:
            with self.subTest(suite=name):
                suite = get_suite(name)
                self.assertGreaterEqual(len(suite.cases), 2)
                self.assertTrue(suite.description)

    def test_unknown_suite(self):
        with self.assertRaises(KeyError):
            get_suite("nope")

    def test_basic_suite_scores_a_good_and_a_bad_model(self):
        good = Scripted("g", {"PONG": "PONG", "17 + 25": "42", "Order #A-1042": "A-1042", "Eiffel": '{"city": "Paris"}',
                              "primary colors": "red, blue, yellow"})
        report = run_eval(get_suite("basic"), {"good": good, "bad": Scripted("b", default="Sure! Here is a long reply.")},
                          warmup=False)
        self.assertEqual(report.pass_rate("good"), 1.0)
        self.assertEqual(report.pass_rate("bad"), 0.0)

    def test_tool_calling_cases_see_the_plugin_calls(self):
        suite = get_suite("tool_calling")
        agent = suite.make_agent(Scripted("m"))
        self.assertEqual(agent.plugins["calc"].calls, [])
        calc = agent.plugins["calc"]
        calc.add(17, 25)
        calc.multiply(2, 3)
        self.assertEqual(calc.calls, [("add", 17, 25), ("multiply", 2, 3)])
        # a model that never calls tools fails the tool cases but passes the restraint case
        report = run_eval(suite, {"lazy": Scripted("l", {"capital": "Paris", "Capital": "Paris"}, default="42 or 70")}, warmup=False)
        by_case = {r.case: r.passed for r in report.results}
        self.assertEqual(by_case, {"single_call": False, "two_steps": False, "no_tool_needed": True})

    def test_long_context_sizes_the_prompt(self):
        small, large = get_suite("long_context", tokens=500), get_suite("long_context", tokens=4000)
        self.assertGreater(len(large.cases[0].task), 6 * len(small.cases[0].task) // 2)
        self.assertIn("ZEBRA-4471", small.cases[0].task)
        self.assertEqual(small.cases[0].contains, ("ZEBRA-4471",))

    def test_memory_suite_has_a_summary_and_a_no_summary_variant(self):
        suite = get_suite("memory")
        agents = [case.make_agent(Scripted("m")) for case in suite.cases]
        self.assertEqual([a.memory_policy.summarize_older for a in agents], [True, False])
        self.assertEqual(len(suite.cases[0].setup_turns), 9)


class TestCli(unittest.TestCase):
    def _run(self, *argv):
        import argparse
        parser = argparse.ArgumentParser()
        add_cli_arguments(parser)
        out = io.StringIO()
        with redirect_stdout(out):
            code = run_cli(parser.parse_args(list(argv)))
        return code, out.getvalue()

    def test_list(self):
        code, out = self._run("--list")
        self.assertEqual(code, 0)
        self.assertIn("tool_calling", out)

    def test_needs_models_and_a_known_suite(self):
        self.assertEqual(self._run()[0], 2)
        code, out = self._run("--models", "x:y", "--suite", "nope")
        self.assertEqual(code, 2)
        self.assertIn("unknown suite", out)

    def test_runs_a_suite_prints_progress_and_writes_json(self):
        import os
        import tempfile
        from unittest.mock import patch
        backend = Scripted("m", {"PONG": "PONG", "17 + 25": "42"})
        with tempfile.TemporaryDirectory() as tmp, patch("maki.backend_factory.init_backend", return_value=backend):
            path = os.path.join(tmp, "out.json")
            code, out = self._run("--models", "ollama:fake", "--suite", "basic", "--runs", "1", "--no-warmup",
                                  "--json", path)
            data = json.load(open(path))
        self.assertEqual(code, 0)
        self.assertIn("[ollama:fake] arithmetic #1: PASS", out)
        self.assertIn("suite: basic", out)
        self.assertEqual(data["suite"], "basic")

    def test_no_think_builds_ollama_backends_with_thinking_off_and_leaves_others_alone(self):
        from unittest.mock import patch
        built = []

        def fake(spec, **kw):
            built.append((spec, kw))
            return Scripted(spec, {"PONG": "PONG", "17 + 25": "42"})
        with patch("maki.backend_factory.init_backend", side_effect=fake):
            code, out = self._run("--models", "ollama:q", "openai:gpt-4o", "--suite", "basic", "--no-think",
                                  "--no-warmup")
        self.assertIn(("ollama:q", {"think": False}), built)
        self.assertIn(("openai:gpt-4o", {}), built)
        self.assertEqual(code, 0)

    def test_maki_logging_is_quiet_unless_verbose(self):
        import logging
        from unittest.mock import patch
        logger = logging.getLogger("maki")
        previous = logger.level
        self.addCleanup(logger.setLevel, previous)
        with patch("maki.backend_factory.init_backend", return_value=Scripted("m")):
            logger.setLevel(logging.INFO)
            self._run("--models", "ollama:fake", "--suite", "basic", "--no-warmup")
            self.assertEqual(logger.level, logging.WARNING)
            logger.setLevel(logging.INFO)
            self._run("--models", "ollama:fake", "--suite", "basic", "--no-warmup", "--verbose")
            self.assertEqual(logger.level, logging.INFO)

    def test_bad_input_is_an_error_message_not_a_traceback(self):
        for argv, fragment in ((("--models", "ollama:x", "--runs", "0"), "--runs must be at least 1"),
                               (("--models", "nosuchprovider:x"), "Unknown backend"),
                               (("--models", "ollama:x", "--judge", "nosuchprovider:y"), "Unknown backend")):
            with self.subTest(argv=argv):
                code, out = self._run(*argv)
                self.assertEqual(code, 2)
                self.assertIn(fragment, out)

    def test_an_ollama_judge_is_built_in_json_mode_with_thinking_off(self):
        from unittest.mock import patch
        built = []

        def fake(spec, **kw):
            built.append((spec, kw))
            return Scripted(spec)
        with patch("maki.backend_factory.init_backend", side_effect=fake):
            self._run("--models", "ollama:m", "--judge", "ollama:j", "--suite", "basic", "--no-warmup")
            self._run("--models", "ollama:m", "--judge", "openai:gpt-4o", "--suite", "basic", "--no-warmup")
        self.assertIn(("ollama:j", {"think": False, "json_format": True}), built)
        self.assertIn(("openai:gpt-4o", {}), built)

    def test_exit_code_is_nonzero_when_runs_errored(self):
        from unittest.mock import patch
        with patch("maki.backend_factory.init_backend", return_value=Scripted("m", fail_on="e")):
            code, out = self._run("--models", "ollama:fake", "--suite", "basic", "--no-warmup")
        self.assertEqual(code, 1)
        self.assertIn("ERROR", out)


if __name__ == "__main__":
    unittest.main()
