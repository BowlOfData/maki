"""Run lock: two processes cannot drive the same loop; a dead holder never blocks it."""
import os
import subprocess
import sys
import tempfile
import time
import unittest

from maki.agents.loop import LoopSpec, Requirement, resume_loop, run_loop
from maki.agents.loop_scratchpad import LoopBusyError, LoopLock

from .loop_helpers import Clock, FakeAgent, verdict

AB = [Requirement("a", "first"), Requirement("b", "second")]
ALL_PASS = verdict({"a": True, "b": True})
ONE_FAIL = verdict({"a": True, "b": False})

HOLDER = (
    "import os, sys, time\n"
    "from maki.agents.loop_scratchpad import LoopLock\n"
    "lock = LoopLock(sys.argv[1]); lock.acquire()\n"
    "print('held', flush=True)\n"
    "if sys.argv[2] == 'die': os._exit(0)\n"
    "time.sleep(60)\n"
)


class LockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = os.path.join(self.tmp.name, "loop")

    def tearDown(self):
        self.tmp.cleanup()

    def spec(self, **kw):
        return LoopSpec("do the thing", AB, loop_dir=self.dir, loop_id="lk", **kw)

    def test_second_acquire_is_refused_with_the_holder_pid(self):
        with LoopLock(self.dir):
            with self.assertRaises(LoopBusyError) as ctx:
                LoopLock(self.dir).acquire()
        self.assertIn(str(os.getpid()), str(ctx.exception))

    def test_release_allows_reacquire(self):
        lock = LoopLock(self.dir)
        lock.acquire()
        lock.release()
        lock.release()  # releasing twice is harmless
        with LoopLock(self.dir):
            pass

    def test_run_loop_on_a_held_directory_refuses_before_touching_the_agent(self):
        maker = FakeAgent(["x"])
        with LoopLock(self.dir):
            with self.assertRaises(LoopBusyError):
                run_loop(maker, FakeAgent([ALL_PASS], stateful=False), self.spec(), clock=Clock())
        self.assertEqual(maker.prompts, [])

    def test_resume_loop_on_a_held_directory_refuses(self):
        run_loop(FakeAgent(["x"]), FakeAgent([ONE_FAIL], stateful=False), self.spec(max_iterations=1), clock=Clock())
        with LoopLock(self.dir):
            with self.assertRaises(LoopBusyError):
                resume_loop(FakeAgent(["x"]), FakeAgent([ALL_PASS], stateful=False), self.spec(max_iterations=3))

    def test_lock_is_released_after_a_normal_finish_and_after_an_error(self):
        run_loop(FakeAgent(["x"]), FakeAgent([ONE_FAIL], stateful=False), self.spec(max_iterations=1), clock=Clock())
        with LoopLock(self.dir):
            pass
        with self.assertRaises(ValueError):  # mismatching goal fails inside the lock
            resume_loop(FakeAgent(["x"]), FakeAgent(["u"], stateful=False),
                        LoopSpec("other", AB, loop_dir=self.dir, loop_id="lk"))
        with LoopLock(self.dir):
            pass

    def test_lock_is_released_when_the_agent_raises(self):
        run_loop(FakeAgent([RuntimeError("down")]), FakeAgent([ALL_PASS], stateful=False), self.spec(), clock=Clock())
        with LoopLock(self.dir):
            pass

    def test_another_process_holding_the_lock_blocks_us(self):
        proc = subprocess.Popen([sys.executable, "-c", HOLDER, self.dir, "hold"], stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(proc.stdout.readline().strip(), "held")
            with self.assertRaises(LoopBusyError) as ctx:
                LoopLock(self.dir).acquire()
            self.assertIn(str(proc.pid), str(ctx.exception))
        finally:
            proc.kill()
            proc.wait()

    def test_a_killed_holder_never_leaves_a_stale_lock(self):
        proc = subprocess.Popen([sys.executable, "-c", HOLDER, self.dir, "die"], stdout=subprocess.PIPE, text=True)
        self.assertEqual(proc.stdout.readline().strip(), "held")
        proc.wait()
        deadline = time.time() + 5
        while True:
            try:
                with LoopLock(self.dir):
                    break
            except LoopBusyError:
                if time.time() > deadline:
                    raise
                time.sleep(0.05)


if __name__ == "__main__":
    unittest.main()
