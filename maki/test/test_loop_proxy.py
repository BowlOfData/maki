"""AgentProxy.run_loop against an in-process AgentServer."""
import tempfile
import unittest
from unittest.mock import patch

pytest = __import__("pytest")
pytest.importorskip("fastapi", reason="fastapi not installed")

from maki.agents.loop import Decision, Requirement  # noqa: E402
from maki.distributed.loop_jobs import LoopCaps  # noqa: E402
from maki.distributed.proxy import AgentProxy, RemoteLoopResult  # noqa: E402
from maki.distributed.server import create_app  # noqa: E402
from maki.exceptions import MakiAPIError, MakiNetworkError  # noqa: E402

from .loop_helpers import FakeAgent, verdict  # noqa: E402
from .test_agent_proxy import _make_connector  # noqa: E402
from .test_loop_jobs import GatedAgent  # noqa: E402

ONE_FAIL = verdict({"r1": True, "r2": False})
ALL_PASS = verdict({"r1": True, "r2": True})
REQS = ["first", "second"]


class _Info(FakeAgent):
    """FakeAgent with the attributes /info reads."""
    agent_id, name, role, maki = "id-1", "remote", "worker", object()


class ProxyLoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.caps = LoopCaps(max_iterations=5, max_seconds=60, loop_dir_base=self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def proxy(self, maker=None, judge_replies=(ALL_PASS,), judge=True):
        maker = maker or _Info(["a", "b"])
        judge_agent = FakeAgent(list(judge_replies), stateful=False) if judge else None
        app = create_app(maker, judge=judge_agent, loop_caps=self.caps)
        with patch("maki.distributed.proxy.Connector", return_value=_make_connector(app)):
            proxy = AgentProxy(endpoint="http://fake-host:8100")
        proxy._connected = True
        return proxy, app

    @staticmethod
    def finish(app, loop_id):
        job = app.state.loop_jobs.get(loop_id)
        job.thread.join(10)

    def test_run_loop_follows_to_success_and_calls_the_hook(self):
        proxy, _ = self.proxy(judge_replies=(ONE_FAIL, ALL_PASS))
        seen = []
        result = proxy.run_loop("goal", REQS, on_iteration=lambda e: seen.append(e["iteration"]))
        self.assertIsInstance(result, RemoteLoopResult)
        self.assertTrue(result.succeeded)
        self.assertEqual((result.stop_reason, result.iterations, result.artifact), ("succeeded", 2, "b"))
        self.assertEqual(seen, [1, 2])
        self.assertEqual([e["type"] for e in result.events], ["started", "iteration", "iteration", "stopped"])

    def test_options_reach_the_server(self):
        proxy, app = self.proxy(judge_replies=(ONE_FAIL,))
        result = proxy.run_loop("goal", REQS, max_iterations=1)
        self.assertEqual((result.stop_reason, result.iterations), ("max_iterations", 1))

    def test_requirement_objects_with_checks_are_refused_before_any_request(self):
        proxy, app = self.proxy()
        for bad in (Requirement("c", "tests", command=["pytest"]), Requirement("f", "x", check=lambda t: True)):
            with self.assertRaises(ValueError):
                proxy.run_loop("goal", [bad])
        self.assertEqual(app.state.loop_jobs.list(), [])
        self.assertTrue(proxy.run_loop("goal", [Requirement("r1", "a"), Requirement("r2", "b")]).succeeded)

    def test_unsupported_option_is_refused(self):
        proxy, _ = self.proxy()
        with self.assertRaises(ValueError):
            proxy.run_loop("goal", REQS, cwd="/")

    def test_loops_disabled_on_the_server(self):
        proxy, _ = self.proxy(judge=False)
        with self.assertRaises(MakiNetworkError) as ctx:  # HTTP 503 maps to MakiNetworkError
            proxy.run_loop("goal", REQS)
        self.assertIn("503", str(ctx.exception))

    def test_busy_server_is_a_409_api_error(self):
        maker = GatedAgent(["a"])
        maker.agent_id, maker.name, maker.role, maker.maki = "i", "n", "r", object()
        proxy, app = self.proxy(maker)
        job = app.state.loop_jobs.start("other", REQS)
        self.assertTrue(maker.entered.wait(5))
        with self.assertRaises(MakiAPIError) as ctx:
            proxy.run_loop("goal", REQS)
        self.assertEqual(ctx.exception.status_code, 409)
        maker.release.set()
        job.thread.join(10)

    def test_cancel_and_status(self):
        maker = GatedAgent(["a"])
        maker.agent_id, maker.name, maker.role, maker.maki = "i", "n", "r", object()
        proxy, app = self.proxy(maker, (ONE_FAIL,))
        job = app.state.loop_jobs.start("goal", REQS, max_iterations=5)
        self.assertTrue(maker.entered.wait(5))
        self.assertEqual(proxy.loop_status(job.id)["status"], "running")
        proxy.cancel_loop(job.id)
        maker.release.set()
        job.thread.join(10)
        self.assertEqual(proxy.loop_status(job.id)["stop_reason"], "stopped")
        with self.assertRaises(MakiAPIError):
            proxy.cancel_loop(job.id)  # already stopped -> 409

    def test_a_stop_decision_cancels_and_tolerates_the_loop_having_already_stopped(self):
        proxy, _ = self.proxy(judge_replies=(ONE_FAIL,))
        result = proxy.run_loop("goal", REQS, max_iterations=2, on_iteration=lambda e: Decision.STOP)
        self.assertIn(result.stop_reason, {"stopped", "max_iterations"})

    def test_resume_loop(self):
        proxy, app = self.proxy(judge_replies=(ONE_FAIL, ALL_PASS))
        first = proxy.run_loop("goal", REQS, max_iterations=1)
        self.assertEqual(first.stop_reason, "max_iterations")
        result = proxy.resume_loop(first.loop_id)
        self.assertTrue(result.succeeded)
        self.assertEqual(result.iterations, 2)

    def test_dropped_stream_reconnects_from_the_last_event(self):
        proxy, _ = self.proxy()
        started = {"type": "started"}
        it = {"type": "iteration", "iteration": 1}
        stopped = {"type": "stopped", "stop_reason": "succeeded", "iterations": 1, "artifact": "x"}
        calls = []

        def fake_events(loop_id, after):
            calls.append(after)
            if len(calls) == 1:
                yield 0, started
                yield 1, it  # connection drops here, no stopped event
            else:
                yield 2, stopped

        with patch.object(proxy, "_loop_events", side_effect=fake_events):
            result = proxy._follow_loop("loop-0123456789ab", None)
        self.assertEqual(calls, [-1, 1])
        self.assertEqual((result.stop_reason, result.artifact, len(result.events)), ("succeeded", "x", 3))

    def test_a_stream_that_never_stops_eventually_raises(self):
        proxy, _ = self.proxy()
        with patch.object(proxy, "_loop_events", side_effect=lambda loop_id, after: iter(())), \
                patch("maki.distributed.proxy.time.sleep"):
            with self.assertRaises(MakiNetworkError):
                proxy._follow_loop("loop-0123456789ab", None)

    def test_a_failing_hook_cancels_the_remote_loop_and_propagates(self):
        maker = GatedAgent(["a", "b", "c"])
        maker.release.set()
        proxy, app = self.proxy(maker=maker, judge_replies=(ONE_FAIL, ONE_FAIL, ONE_FAIL))

        def boom(event):
            raise RuntimeError("hook broke")

        with patch.object(proxy, "cancel_loop") as cancel:
            with self.assertRaises(RuntimeError):
                proxy.run_loop("goal", REQS, on_iteration=boom)
        (loop,) = app.state.loop_jobs.list()
        cancel.assert_called_once_with(loop["id"])
        app.state.loop_jobs.get(loop["id"]).thread.join(10)

    def test_loop_ids_are_validated_before_any_request(self):
        proxy, _ = self.proxy()
        for call in (proxy.loop_status, proxy.cancel_loop, proxy.resume_loop):
            with self.assertRaises(ValueError):
                call("../execute")


if __name__ == "__main__":
    unittest.main()
