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

    def test_output_that_is_not_utf8_does_not_crash_the_check(self):
        code = "import sys; sys.stdout.buffer.write(b'before \\xff\\xfe after'); sys.exit(1)"
        r = run_command_check([sys.executable, "-c", code])
        self.assertFalse(r.passed)
        self.assertIn("before", r.feedback)
        self.assertIn("after", r.feedback)

    def test_a_daemon_that_escapes_the_process_group_cannot_hang_the_check(self):
        import time
        from unittest.mock import patch
        code = ("import subprocess, sys\n"
                "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(4)'], start_new_session=True)\n"
                "import time; time.sleep(30)")
        started = time.monotonic()
        with patch("maki.agents.loop_judge._REAP_SECONDS", 0.3):
            r = run_command_check([sys.executable, "-c", code], timeout=0.5)
        self.assertFalse(r.passed)
        self.assertIn("timed out", r.feedback)
        self.assertLess(time.monotonic() - started, 3)

    def test_only_the_tail_of_huge_output_is_kept(self):
        code = "import sys\nfor i in range(200000): print('line', i)\nsys.exit(2)"
        r = run_command_check([sys.executable, "-c", code], max_output=500)
        self.assertFalse(r.passed)
        self.assertIn("line 199999", r.feedback)
        self.assertNotIn("line 1\n", r.feedback)
        self.assertLess(len(r.feedback), 600)

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
