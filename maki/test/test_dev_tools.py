"""The mutation checker and the load check are test tools: these tests make sure the tools themselves are right
(a mutation is one real change; a leak is reported as a leak; a healthy run passes)."""
import argparse
import ast
import io
import os
import unittest
from contextlib import redirect_stdout

from maki.test import load_check, mutation_check
from maki.test.mutation_check import find_sites, mutate

SOURCE = '''
def guard(value, limit):
    if value > limit and not value == 0:
        raise ValueError("too big")
    total = value + 1
    flag = True
    return total
'''


class TestMutationGeneration(unittest.TestCase):
    def test_the_expected_kinds_of_site_are_found(self):
        kinds = sorted(s.kind for s in find_sites(SOURCE))
        self.assertEqual(kinds, ["binop", "boolop", "compare", "compare", "constant", "constant", "constant", "not",
                                 "raise", "return"])  # the constants are 0, 1 and True

    def test_every_mutant_is_valid_python_and_differs_from_the_original_in_one_place(self):
        original = ast.unparse(ast.parse(SOURCE))
        for site in find_sites(SOURCE):
            with self.subTest(kind=site.kind, line=site.line):
                mutated, what = mutate(SOURCE, site.index)
                ast.parse(mutated)  # compiles
                self.assertNotEqual(mutated, original)
                self.assertIn(site.kind, what)
                diff = [(a, b) for a, b in zip(original.splitlines(), mutated.splitlines()) if a != b]
                self.assertLessEqual(len(diff), 1)

    def test_specific_mutations_do_what_they_say(self):
        by_what = {}
        for site in find_sites(SOURCE):
            mutated, what = mutate(SOURCE, site.index)
            by_what.setdefault(what.split(" at")[0], []).append(mutated)
        joined = lambda kind: "\n".join(by_what[kind])  # noqa: E731
        self.assertIn("value > limit or not value == 0", joined("boolop"))
        self.assertIn("value <= limit and", joined("compare"))
        self.assertIn("not value != 0", joined("compare"))
        self.assertIn("and value == 0", joined("not"))
        self.assertNotIn("raise ValueError", joined("raise"))
        self.assertIn("pass", joined("raise"))
        self.assertIn("total = value - 1", joined("binop"))
        self.assertIn("flag = False", joined("constant"))
        self.assertIn("total = value + 2", joined("constant"))
        self.assertIn("return None", joined("return"))

    def test_an_index_past_the_last_site_is_an_error_not_a_silent_no_op(self):
        with self.assertRaises(IndexError):
            mutate(SOURCE, 999)

    def test_scoping_by_function_and_by_lines(self):
        two = SOURCE + "\ndef other(a):\n    return a > 3\n"
        all_sites = find_sites(two)
        only = find_sites(two, functions=["other"])
        self.assertLess(len(only), len(all_sites))
        self.assertTrue(all(s.line >= 9 for s in only))
        window = find_sites(two, lines=(3, 4))
        self.assertTrue(window and all(3 <= s.line <= 4 for s in window))
        self.assertEqual(find_sites(two, functions=["nope"]), [])

    def test_bool_and_integer_constants_change_but_strings_and_none_do_not(self):
        sites = find_sites("x = 'text'\ny = None\nz = 7\nw = False\n")
        self.assertEqual(sorted(s.kind for s in sites), ["constant", "constant"])
        outputs = [mutate("x = 'text'\ny = None\nz = 7\nw = False\n", s.index)[0] for s in sites]
        self.assertTrue(any("z = 8" in o for o in outputs))
        self.assertTrue(any("w = True" in o for o in outputs))

    def test_returning_none_is_not_mutated_to_none(self):
        self.assertEqual(find_sites("def f():\n    return None\n"), [])
        self.assertEqual(find_sites("def f():\n    return\n"), [])


@unittest.skipUnless(os.environ.get("MAKI_SLOW_TESTS"), "set MAKI_SLOW_TESTS=1 to run the end-to-end mutation test (~15 s)")
class TestMutationEndToEnd(unittest.TestCase):
    def test_a_tested_function_is_killed_and_the_report_names_survivors(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = mutation_check.main(["maki/context.py", "--functions", "estimate_tokens", "--tests",
                                        "maki/test/test_context.py::TestEstimation", "--workers", "2", "--limit", "4"])
        self.assertEqual(code, 0)
        self.assertRegex(out.getvalue(), r"killed \d+/\d+")


def samples(rss=None, threads=None, fds=None, count=60):
    return [load_check.Sample(t=float(i), rss_mb=(rss(i) if rss else 80.0), threads=(threads(i) if threads else 30),
                              fds=(fds(i) if fds else 50)) for i in range(count)]


def args(**kw):
    base = dict(max_rss_growth_mb=40.0, max_rss_slope=3.0, max_thread_growth=5, max_fd_growth=10)
    base.update(kw)
    return argparse.Namespace(**base)


class TestLoadCheckVerdicts(unittest.TestCase):
    def test_a_steady_process_passes(self):
        self.assertEqual(load_check.analyse(samples(), args()), [])

    def test_a_warm_up_that_levels_off_passes(self):
        self.assertEqual(load_check.analyse(samples(rss=lambda i: 60 + 25 * (1 - 0.8 ** i)), args()), [])

    def test_steady_leaking_is_reported_as_growth_and_as_a_slope(self):
        problems = load_check.analyse(samples(rss=lambda i: 80 + i * 1.5), args())
        self.assertTrue(any("grew" in p for p in problems), problems)
        self.assertTrue(any("MB/min" in p for p in problems), problems)

    def test_a_slow_but_endless_creep_is_caught_by_the_slope_even_when_total_growth_is_small(self):
        problems = load_check.analyse(samples(rss=lambda i: 80 + i * 0.1, count=120), args())
        self.assertTrue(any("MB/min" in p for p in problems), problems)

    def test_thread_and_descriptor_leaks(self):
        self.assertTrue(any("thread" in p for p in load_check.analyse(samples(threads=lambda i: 30 + i // 5), args())))
        self.assertTrue(any("descriptor" in p for p in load_check.analyse(samples(fds=lambda i: 50 + i), args())))

    def test_too_few_samples_to_judge_say_nothing(self):
        self.assertEqual(load_check.analyse(samples(rss=lambda i: 80 + i * 100, count=5), args()), [])

    def test_limits_are_respected(self):
        leaking = samples(rss=lambda i: 80 + i * 1.5)
        self.assertEqual(load_check.analyse(leaking, args(max_rss_growth_mb=1000, max_rss_slope=1000)), [])

    def test_percentiles(self):
        values = [float(i) for i in range(1, 101)]
        self.assertEqual((load_check.percentile(values, 0.5), load_check.percentile(values, 0.99)), (51.0, 100.0))
        self.assertEqual(load_check.percentile([], 0.5), 0.0)
        self.assertEqual(load_check.percentile([7.0], 0.99), 7.0)


class TestLoadCheckRun(unittest.TestCase):
    def test_a_short_run_against_the_real_server_passes_and_reports_every_kind_of_request(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = load_check.main(["--duration", "3", "--concurrency", "4", "--in-process", "--max-rss-growth-mb", "500",
                                    "--max-thread-growth", "500", "--max-fd-growth", "500", "--max-rss-slope", "1000"])
        text = out.getvalue()
        self.assertEqual(code, 0, text)
        for kind in ("execute", "health", "no_auth", "oversize", "bad_json", "unknown_path", "wrong_method"):
            self.assertIn(kind, text)
        self.assertIn("0 unexpected responses", text)
        self.assertIn("PASS", text)

    def test_the_workload_only_expects_statuses_a_correct_server_produces(self):
        import random
        rng = random.Random(1)
        seen = {}
        for _ in range(3000):
            kind, method, path, kwargs, allowed = load_check.workload(rng, b"x", "k")
            seen.setdefault(kind, allowed)
            self.assertTrue(allowed)
        self.assertGreaterEqual(len(seen), 12)
        self.assertEqual(seen["no_auth"], (401,))
        self.assertEqual(seen["oversize"], (413, "dropped"))

    def test_unexpected_statuses_are_recorded_as_errors(self):
        result = load_check.Result()
        result.record("execute", 200, 0.01, (200,))
        result.record("execute", 500, 0.02, (200,))
        result.record("oversize", "dropped", 0.01, (413, "dropped"))
        self.assertEqual(result.errors, ["execute: unexpected 500"])
        self.assertEqual(result.statuses[("execute", 500)], 1)


if __name__ == "__main__":
    unittest.main()
