"""Command-check runner."""
import os
import sys
import unittest

from maki.agents.loop_judge import run_command_check


class CommandCheckTests(unittest.TestCase):
    def test_exit_zero_passes_with_output(self):
        r = run_command_check([sys.executable, "-c", "print('hello')"])
        self.assertTrue(r.passed)
        self.assertIn("hello", r.feedback)

    def test_nonzero_fails_with_output(self):
        r = run_command_check([sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"])
        self.assertFalse(r.passed)
        self.assertIn("exit code 3", r.feedback)
        self.assertIn("boom", r.feedback)

    def test_string_command_rejected(self):
        for bad in ("pytest -q", [], [1], b"x"):
            with self.assertRaises(ValueError):
                run_command_check(bad)

    def test_timeout_fails_closed(self):
        r = run_command_check([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.5)
        self.assertFalse(r.passed)
        self.assertIn("timed out", r.feedback)

    def test_timeout_kills_grandchildren_that_hold_the_pipe(self):
        import time
        code = ("import subprocess, sys, time; "
                "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); time.sleep(30)")
        start = time.time()
        r = run_command_check([sys.executable, "-c", code], timeout=0.5)
        self.assertFalse(r.passed)
        self.assertLess(time.time() - start, 10)

    def test_missing_program_fails_closed(self):
        r = run_command_check(["definitely-not-a-program-xyz"])
        self.assertFalse(r.passed)
        self.assertIn("could not run", r.feedback)

    def test_no_shell_interpretation(self):
        r = run_command_check([sys.executable, "-c", "import sys; print(sys.argv[1])", "a; echo injected"])
        self.assertIn("a; echo injected", r.feedback)

    def test_minimal_environment(self):
        os.environ["LOOP_TEST_SECRET"] = "s3cret"
        try:
            r = run_command_check([sys.executable, "-c", "import os; print(os.environ.get('LOOP_TEST_SECRET'))"])
        finally:
            del os.environ["LOOP_TEST_SECRET"]
        self.assertIn("None", r.feedback)

    def test_cwd_and_output_cap(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            r = run_command_check([sys.executable, "-c", "import os; print(os.getcwd()); print('x'*500)"],
                                  cwd=d, max_output=100)
        self.assertEqual(len(r.feedback), 100)


if __name__ == "__main__":
    unittest.main()
