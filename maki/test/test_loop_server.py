"""The /loop endpoints of the agent server (FastAPI TestClient, scripted fake agents)."""
import json
import tempfile
import threading
import unittest

pytest = __import__("pytest")
pytest.importorskip("fastapi", reason="fastapi not installed")
from fastapi.testclient import TestClient  # noqa: E402

from maki.distributed.loop_jobs import LoopCaps  # noqa: E402
from maki.distributed.server import create_app  # noqa: E402

from .loop_helpers import FakeAgent, verdict  # noqa: E402
from .test_loop_jobs import GatedAgent  # noqa: E402

ONE_FAIL = verdict({"r1": True, "r2": False})
ALL_PASS = verdict({"r1": True, "r2": True})
BODY = {"goal": "do it", "requirements": ["first", "second"]}


def sse_events(response):
    """Parse an SSE body into (events, saw_done)."""
    events, done = [], False
    for block in response.text.split("\n\n"):
        for line in block.splitlines():
            if line == "data: [DONE]":
                done = True
            elif line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events, done


class LoopServerBase(unittest.TestCase):
    API_KEY = None

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.caps = LoopCaps(max_iterations=5, max_seconds=60, loop_dir_base=self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def client(self, maker=None, judge_replies=(ALL_PASS,), judge=True, api_key=None):
        maker = maker or FakeAgent(["draft"])
        judge_agent = FakeAgent(list(judge_replies), stateful=False) if judge else None
        app = create_app(maker, api_key=api_key, judge=judge_agent, loop_caps=self.caps)
        app.state.loop_heartbeat_seconds = 0.2
        return TestClient(app), app

    @staticmethod
    def finish(app, loop_id):
        job = app.state.loop_jobs.get(loop_id)
        job.thread.join(10)
        assert job.finished


class StartAndWatchTests(LoopServerBase):
    def test_start_returns_202_with_a_server_chosen_id_and_runs_to_success(self):
        client, app = self.client(FakeAgent(["a", "b"]), (ONE_FAIL, ALL_PASS))
        r = client.post("/loop", json=BODY)
        self.assertEqual(r.status_code, 202)
        loop_id = r.json()["id"]
        self.assertRegex(loop_id, r"^loop-[0-9a-f]{12}$")
        self.finish(app, loop_id)
        status = client.get(f"/loop/{loop_id}").json()
        self.assertEqual((status["status"], status["iterations"], status["artifact"]), ("succeeded", 2, "b"))

    def test_events_stream_replays_in_order_and_ends_with_done(self):
        client, app = self.client(FakeAgent(["a", "b"]), (ONE_FAIL, ALL_PASS))
        loop_id = client.post("/loop", json=BODY).json()["id"]
        self.finish(app, loop_id)
        events, done = sse_events(client.get(f"/loop/{loop_id}/events"))
        self.assertTrue(done)
        self.assertEqual([e["type"] for e in events], ["started", "iteration", "iteration", "stopped"])
        self.assertEqual(events[-1]["stop_reason"], "succeeded")

    def test_replay_with_after_and_last_event_id(self):
        client, app = self.client(FakeAgent(["a", "b"]), (ONE_FAIL, ALL_PASS))
        loop_id = client.post("/loop", json=BODY).json()["id"]
        self.finish(app, loop_id)
        after, _ = sse_events(client.get(f"/loop/{loop_id}/events?after=2"))
        self.assertEqual([e["type"] for e in after], ["stopped"])
        header, _ = sse_events(client.get(f"/loop/{loop_id}/events", headers={"Last-Event-ID": "1"}))
        self.assertEqual([e["type"] for e in header], ["iteration", "stopped"])
        ids = [line for line in client.get(f"/loop/{loop_id}/events?after=1").text.splitlines() if line.startswith("id:")]
        self.assertEqual(ids, ["id: 2", "id: 3"])

    def test_watching_a_running_loop_streams_until_it_ends(self):
        maker = GatedAgent(["a"])
        client, app = self.client(maker, (ALL_PASS,))
        loop_id = client.post("/loop", json=BODY).json()["id"]
        self.assertTrue(maker.entered.wait(5))
        threading.Timer(0.5, maker.release.set).start()  # the loop ends while we are watching
        events, done = sse_events(client.get(f"/loop/{loop_id}/events"))
        self.assertTrue(done)
        self.assertEqual(events[-1]["type"], "stopped")

    def test_list(self):
        client, app = self.client()
        loop_id = client.post("/loop", json=BODY).json()["id"]
        self.finish(app, loop_id)
        self.assertEqual([s["id"] for s in client.get("/loop").json()["loops"]], [loop_id])


class RejectionTests(LoopServerBase):
    def test_without_a_judge_every_loop_endpoint_is_503(self):
        client, _ = self.client(judge=False)
        for method, path in (("post", "/loop"), ("get", "/loop"), ("get", "/loop/loop-0123456789ab"),
                             ("get", "/loop/loop-0123456789ab/events"), ("delete", "/loop/loop-0123456789ab"),
                             ("post", "/loop/loop-0123456789ab/resume")):
            r = getattr(client, method)(path, **({"json": BODY} if method == "post" and path == "/loop" else {}))
            self.assertEqual(r.status_code, 503, path)

    def test_commands_and_other_privileged_fields_are_rejected(self):
        client, app = self.client()
        for extra in ({"cwd": "/"}, {"loop_dir": "/tmp/x"}, {"command": ["rm", "-rf", "/"]}, {"artifact_fn": "x"},
                      {"loop_id": "loop-aaaaaaaaaaaa"}):
            r = client.post("/loop", json={**BODY, **extra})
            self.assertEqual(r.status_code, 422, extra)
        r = client.post("/loop", json={"goal": "g", "requirements": [{"text": "x", "command": ["id"]}]})
        self.assertEqual(r.status_code, 400)
        self.assertIn("only text requirements", r.json()["detail"])
        self.assertEqual(client.get("/loop").json()["loops"], [])

    def test_bad_input_and_caps_are_400(self):
        client, _ = self.client()
        for body in ({**BODY, "max_iterations": 99}, {**BODY, "max_seconds": 9999}, {"goal": " ", "requirements": ["a"]},
                     {"goal": "g", "requirements": []}):
            self.assertEqual(client.post("/loop", json=body).status_code, 400, body)

    def test_second_loop_while_one_runs_is_409(self):
        maker = GatedAgent(["a"])
        client, app = self.client(maker)
        loop_id = client.post("/loop", json=BODY).json()["id"]
        self.assertTrue(maker.entered.wait(5))
        self.assertEqual(client.post("/loop", json=BODY).status_code, 409)
        maker.release.set()
        self.finish(app, loop_id)

    def test_unknown_or_malformed_ids_are_404(self):
        client, _ = self.client()
        for path in ("/loop/loop-0123456789ab", "/loop/..%2Fetc", "/loop/not-an-id", "/loop/loop-0123456789ab/events"):
            self.assertEqual(client.get(path).status_code, 404, path)
        self.assertEqual(client.delete("/loop/loop-0123456789ab").status_code, 404)
        self.assertEqual(client.post("/loop/loop-0123456789ab/resume").status_code, 404)


class CancelAndResumeTests(LoopServerBase):
    def test_cancel_a_running_loop(self):
        maker = GatedAgent(["a"])
        client, app = self.client(maker, (ONE_FAIL,))
        loop_id = client.post("/loop", json={**BODY, "max_iterations": 5}).json()["id"]
        self.assertTrue(maker.entered.wait(5))
        r = client.delete(f"/loop/{loop_id}")
        self.assertEqual((r.status_code, r.json()["cancelling"]), (202, True))
        maker.release.set()
        self.finish(app, loop_id)
        self.assertEqual(client.get(f"/loop/{loop_id}").json()["stop_reason"], "stopped")

    def test_cancel_a_finished_loop_is_409(self):
        client, app = self.client()
        loop_id = client.post("/loop", json=BODY).json()["id"]
        self.finish(app, loop_id)
        self.assertEqual(client.delete(f"/loop/{loop_id}").status_code, 409)

    def test_resume_after_a_cap_and_on_a_restarted_server(self):
        client, app = self.client(FakeAgent(["a"]), (ONE_FAIL,))
        loop_id = client.post("/loop", json={**BODY, "max_iterations": 1}).json()["id"]
        self.finish(app, loop_id)
        self.assertTrue(client.get(f"/loop/{loop_id}").json()["resumable"])
        client2, app2 = self.client(FakeAgent(["b"]), (ALL_PASS,))  # a restarted server: nothing in memory
        self.assertEqual(client2.get(f"/loop/{loop_id}").json()["status"], "max_iterations")
        events, done = sse_events(client2.get(f"/loop/{loop_id}/events"))  # synthetic stopped event from disk
        self.assertTrue(done)
        self.assertEqual(events[0]["type"], "stopped")
        r = client2.post(f"/loop/{loop_id}/resume")
        self.assertEqual(r.status_code, 202)
        self.finish(app2, loop_id)
        self.assertEqual(client2.get(f"/loop/{loop_id}").json()["status"], "succeeded")

    def test_resume_of_a_succeeded_loop_is_400(self):
        client, app = self.client()
        loop_id = client.post("/loop", json=BODY).json()["id"]
        self.finish(app, loop_id)
        self.assertEqual(client.post(f"/loop/{loop_id}/resume").status_code, 400)


class ApprovalAndBudgetTests(LoopServerBase):
    def test_gated_loop_suspends_is_approved_over_http_and_resumes(self):
        client, app = self.client(FakeAgent(["a"]), (ALL_PASS,))
        loop_id = client.post("/loop", json={**BODY, "require_approval": True}).json()["id"]
        self.finish(app, loop_id)
        status = client.get(f"/loop/{loop_id}").json()
        self.assertEqual(status["status"], "suspended")
        self.assertIn("Start loop", status["pending_approval"]["summary"])
        r = client.post(f"/loop/{loop_id}/approval", json={"approve": True, "note": "careful"})
        self.assertEqual((r.status_code, r.json()["pending_approval"]), (200, None))
        self.assertEqual(client.post(f"/loop/{loop_id}/resume").status_code, 202)
        self.finish(app, loop_id)
        self.assertEqual(client.get(f"/loop/{loop_id}").json()["status"], "succeeded")

    def test_approval_errors(self):
        client, app = self.client()
        self.assertEqual(client.post("/loop/loop-0123456789ab/approval", json={"approve": True}).status_code, 404)
        self.assertEqual(client.post("/loop/not-an-id/approval", json={"approve": True}).status_code, 404)
        loop_id = client.post("/loop", json=BODY).json()["id"]  # no gate
        self.finish(app, loop_id)
        self.assertEqual(client.post(f"/loop/{loop_id}/approval", json={"approve": True}).status_code, 409)
        self.assertEqual(client.post(f"/loop/{loop_id}/approval", json={"approve": True, "x": 1}).status_code, 422)
        self.assertEqual(client.post(f"/loop/{loop_id}/approval", json={}).status_code, 422)

    def test_token_budget_over_http(self):
        self.caps.max_tokens = 500
        client, app = self.client(FakeAgent(["a"], tokens_per_call=100), (ONE_FAIL,))
        self.assertEqual(client.post("/loop", json={**BODY, "max_tokens": 501}).status_code, 400)
        loop_id = client.post("/loop", json={**BODY, "max_tokens": 150, "stall_limit": 5}).json()["id"]
        self.finish(app, loop_id)
        status = client.get(f"/loop/{loop_id}").json()
        self.assertEqual((status["status"], status["iterations"]), ("token_budget", 2))
        events, _ = sse_events(client.get(f"/loop/{loop_id}/events"))
        self.assertEqual(events[-1]["stop_reason"], "token_budget")
        self.assertEqual(events[-1]["tokens"], status["tokens"])


class AuthTests(LoopServerBase):
    def test_approval_needs_the_api_key(self):
        client, _ = self.client(api_key="secret")
        self.assertEqual(client.post("/loop/loop-0123456789ab/approval", json={"approve": True}).status_code, 401)

    def test_loop_endpoints_need_the_api_key(self):
        client, app = self.client(api_key="secret")
        self.assertEqual(client.post("/loop", json=BODY).status_code, 401)
        self.assertEqual(client.get("/loop").status_code, 401)
        self.assertEqual(client.get("/loop/loop-0123456789ab/events").status_code, 401)
        self.assertEqual(client.delete("/loop/loop-0123456789ab").status_code, 401)
        headers = {"Authorization": "Bearer secret"}
        r = client.post("/loop", json=BODY, headers=headers)
        self.assertEqual(r.status_code, 202)
        self.finish(app, r.json()["id"])


if __name__ == "__main__":
    unittest.main()
