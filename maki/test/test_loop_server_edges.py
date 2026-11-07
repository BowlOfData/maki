"""Error paths and streaming details of the /loop endpoints that the mutation check showed untested."""
import asyncio
import json
import unittest

pytest = __import__("pytest")
pytest.importorskip("fastapi", reason="fastapi not installed")

from maki.distributed import server  # noqa: E402

from .loop_helpers import FakeAgent, verdict  # noqa: E402
from .test_loop_jobs import GatedAgent  # noqa: E402
from .test_loop_server import BODY, LoopServerBase, sse_events  # noqa: E402

ALL_PASS = verdict({"r1": True, "r2": True})
ONE_FAIL = verdict({"r1": True, "r2": False})


class TestFirstEventIndex(unittest.TestCase):
    def test_every_way_a_client_can_ask_for_a_starting_point(self):
        f = server._first_event_index
        for args, expected in (((None, None), 0), ((-1, None), 0), ((0, None), 1), ((4, None), 5), ((None, "3"), 4),
                               ((None, " 7 "), 8), ((None, "0"), 1), ((None, "-5"), 0), ((None, "abc"), 0),
                               ((None, ""), 0), ((None, "3.5"), 0), ((2, "9"), 3), ((None, "-0"), 1)):
            with self.subTest(args=args):
                self.assertEqual(f(*args), expected)


class TestStoredStream(unittest.TestCase):
    SNAP = {"stop_reason": "", "status": "succeeded", "iterations": 3, "tokens": 120, "detail": "d", "error": None,
            "artifact": "final text"}

    def _collect(self, snap, start):
        return list(server._stored_stream(snap, start))

    def test_from_the_start_it_is_one_stopped_event_then_done(self):
        out = self._collect(self.SNAP, 0)
        self.assertEqual(len(out), 2)
        self.assertTrue(out[0].startswith("id: 0\ndata: "))
        event = json.loads(out[0].split("data: ", 1)[1])
        self.assertEqual((event["type"], event["stop_reason"], event["iterations"], event["tokens"], event["artifact"]),
                         ("stopped", "succeeded", 3, 120, "final text"))
        self.assertEqual(out[1], "data: [DONE]\n\n")

    def test_the_reason_wins_over_the_status_when_there_is_one(self):
        event = json.loads(self._collect({**self.SNAP, "stop_reason": "stalled"}, 0)[0].split("data: ", 1)[1])
        self.assertEqual(event["stop_reason"], "stalled")

    def test_a_client_that_already_saw_it_gets_only_the_end_marker(self):
        self.assertEqual(self._collect(self.SNAP, 1), ["data: [DONE]\n\n"])


class FakeJob:
    """What ``_live_stream`` needs from a job: events appear over time, then it finishes."""

    def __init__(self, script):
        self.script, self.calls = list(script), 0

    def events_from(self, index):
        self.calls += 1
        step = self.script.pop(0) if self.script else ([], True)
        return step


class TestLiveStream(unittest.TestCase):
    def _run(self, job, start=0, heartbeat=0.05):
        async def go():
            return [chunk async for chunk in server._live_stream(job, start, heartbeat)]
        return asyncio.run(asyncio.wait_for(go(), 10))

    def test_events_carry_their_index_and_the_stream_ends_with_done(self):
        out = self._run(FakeJob([([{"type": "started"}, {"type": "iteration"}], False), ([{"type": "stopped"}], True)]), start=5)
        self.assertEqual([c.split("\n")[0] for c in out[:3]], ["id: 5", "id: 6", "id: 7"])
        self.assertEqual(out[-1], "data: [DONE]\n\n")
        self.assertEqual(len(out), 4)

    def test_a_finished_job_with_nothing_left_ends_at_once(self):
        job = FakeJob([([], True)])
        self.assertEqual(self._run(job), ["data: [DONE]\n\n"])
        self.assertEqual(job.calls, 1)

    def test_a_quiet_stretch_produces_keep_alive_comments_not_silence(self):
        job = FakeJob([([], False)] * 8 + [([{"type": "stopped"}], True)])
        out = self._run(job, heartbeat=0.04)
        self.assertIn(": keep-alive\n\n", out)
        self.assertEqual(out[-1], "data: [DONE]\n\n")
        self.assertTrue(any(c.startswith("id: 0") for c in out))

    def test_no_keep_alive_before_the_quiet_period_has_elapsed(self):
        job = FakeJob([([], False)] * 2 + [([], True)])
        out = self._run(job, heartbeat=30)
        self.assertEqual(out, ["data: [DONE]\n\n"])

    def test_a_finished_job_still_delivers_its_last_events(self):
        out = self._run(FakeJob([([{"type": "stopped"}], True)]))
        self.assertEqual(len(out), 2)
        self.assertTrue(out[0].startswith("id: 0"))


class TestUnknownAndInvalidIds(LoopServerBase):
    def test_unknown_but_well_formed_ids_are_404_on_every_endpoint(self):
        client, _ = self.client()
        unknown = "loop-0123456789ab"
        for method, path, body in (("GET", f"/loop/{unknown}", None), ("DELETE", f"/loop/{unknown}", None),
                                   ("POST", f"/loop/{unknown}/resume", {}), ("GET", f"/loop/{unknown}/events", None),
                                   ("POST", f"/loop/{unknown}/approval", {"approve": True})):
            with self.subTest(route=f"{method} {path}"):
                r = client.request(method, path, **({"json": body} if body is not None else {}))
                self.assertEqual((r.status_code, r.json()["detail"]), (404, "Loop not found"))

    def test_ids_that_do_not_look_like_ours_never_reach_the_store(self):
        client, _ = self.client()
        for bad in ("BAD", "loop-xyz", "loop-0123456789AB", "loop-" + "a" * 40, "..", "a b", "loop-0123456789ab%00"):
            for method in ("GET", "DELETE"):
                with self.subTest(id=bad, method=method):
                    r = client.request(method, f"/loop/{bad}")
                    self.assertIn(r.status_code, (404, 405))
                    self.assertNotIn("Traceback", r.text)


class TestStartCancelResume(LoopServerBase):
    def test_a_second_loop_while_one_runs_is_refused_with_the_reason(self):
        maker = GatedAgent(["a"])
        client, app = self.client(maker)
        first = client.post("/loop", json=BODY)
        self.assertEqual(first.status_code, 202)
        maker.entered.wait(5)
        second = client.post("/loop", json=BODY)
        self.assertEqual(second.status_code, 409)
        self.assertTrue(second.json()["detail"])
        maker.release.set()
        self.finish(app, first.json()["id"])

    def test_the_documented_start_body_is_strict(self):
        client, _ = self.client()
        for extra in ({"command": ["rm", "-rf", "/"]}, {"cwd": "/"}, {"loop_dir": "/etc"}, {"unknown": 1}):
            with self.subTest(extra=extra):
                self.assertEqual(client.post("/loop", json={**BODY, **extra}).status_code, 422)
        self.assertEqual(client.post("/loop", json={"goal": "g"}).status_code, 422)
        self.assertEqual(client.post("/loop", json={"requirements": ["x"]}).status_code, 422)

    def test_cancel_reports_what_it_did(self):
        maker = GatedAgent(["a"])
        client, app = self.client(maker)
        loop_id = client.post("/loop", json=BODY).json()["id"]
        maker.entered.wait(5)
        r = client.delete(f"/loop/{loop_id}")
        self.assertEqual((r.status_code, r.json()), (202, {"ok": True, "cancelling": True}))
        maker.release.set()
        self.finish(app, loop_id)
        again = client.delete(f"/loop/{loop_id}")
        self.assertEqual((again.status_code, again.json()["detail"]), (409, "Loop is not running"))

    def test_resume_while_running_is_refused_and_after_a_cap_returns_the_new_state(self):
        maker = GatedAgent(["a", "b", "c"])
        client, app = self.client(maker, (ONE_FAIL, ONE_FAIL, ONE_FAIL, ALL_PASS))
        loop_id = client.post("/loop", json={**BODY, "max_iterations": 1}).json()["id"]
        maker.entered.wait(5)
        busy = client.post(f"/loop/{loop_id}/resume", json={})
        self.assertEqual(busy.status_code, 409)
        self.assertTrue(busy.json()["detail"])
        maker.release.set()
        self.finish(app, loop_id)
        self.assertEqual(client.get(f"/loop/{loop_id}").json()["status"], "max_iterations")
        resumed = client.post(f"/loop/{loop_id}/resume", json={"max_iterations": 3})
        self.assertEqual(resumed.status_code, 202)
        body = resumed.json()
        self.assertEqual(body["id"], loop_id)
        self.assertIn("status", body)
        self.finish(app, loop_id)

    def test_resume_with_nonsense_caps_is_a_400_with_the_reason(self):
        client, app = self.client(FakeAgent(["a"]), (ONE_FAIL,))
        loop_id = client.post("/loop", json={**BODY, "max_iterations": 1}).json()["id"]
        self.finish(app, loop_id)
        for caps in ({"max_iterations": 0}, {"max_iterations": -3}, {"max_seconds": 0}, {"max_tokens": -1}):
            with self.subTest(caps=caps):
                r = client.post(f"/loop/{loop_id}/resume", json=caps)
                self.assertIn(r.status_code, (400, 422))
                self.assertTrue(r.json()["detail"])

    def test_approval_without_a_pending_gate_is_409_and_says_so(self):
        client, app = self.client(FakeAgent(["a"]), (ALL_PASS,))
        loop_id = client.post("/loop", json=BODY).json()["id"]
        self.finish(app, loop_id)
        r = client.post(f"/loop/{loop_id}/approval", json={"approve": True})
        self.assertEqual((r.status_code, r.json()["detail"]), (409, "Loop has no pending approval"))
        self.assertEqual(client.post(f"/loop/{loop_id}/approval", json={"approve": "maybe"}).status_code, 422)
        self.assertEqual(client.post(f"/loop/{loop_id}/approval", json={"approve": True, "extra": 1}).status_code, 422)

    def test_the_list_shows_started_loops(self):
        client, app = self.client(FakeAgent(["a"]), (ALL_PASS,))
        self.assertEqual(client.get("/loop").json(), {"loops": []})
        loop_id = client.post("/loop", json=BODY).json()["id"]
        self.finish(app, loop_id)
        listed = client.get("/loop").json()["loops"]
        self.assertEqual([entry["id"] for entry in listed], [loop_id])

    def test_event_streams_honour_after_and_last_event_id(self):
        client, app = self.client(FakeAgent(["a", "b"]), (ONE_FAIL, ALL_PASS))
        loop_id = client.post("/loop", json=BODY).json()["id"]
        self.finish(app, loop_id)
        everything, _ = sse_events(client.get(f"/loop/{loop_id}/events"))
        self.assertEqual(len(everything), 4)
        after, _ = sse_events(client.get(f"/loop/{loop_id}/events?after=1"))
        self.assertEqual([e["type"] for e in after], [e["type"] for e in everything[2:]])
        resumed, _ = sse_events(client.get(f"/loop/{loop_id}/events", headers={"Last-Event-ID": "2"}))
        self.assertEqual(len(resumed), 1)
        ignored, _ = sse_events(client.get(f"/loop/{loop_id}/events", headers={"Last-Event-ID": "garbage"}))
        self.assertEqual(len(ignored), 4)
        self.assertEqual(client.get(f"/loop/{loop_id}/events?after=-2").status_code, 422)


if __name__ == "__main__":
    unittest.main()
