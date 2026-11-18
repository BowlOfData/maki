"""Agent checkpoints: state after each task, the unfinished tool loop after each round, resume()."""

import json
import os
import sqlite3
import stat
import tempfile
import unittest
from unittest.mock import MagicMock

from maki.agents import Agent
from maki.agents.tool_policy import ToolPolicy
from maki.checkpoint import SCHEMA_VERSION, CheckpointError, CheckpointStore, redact_value
from maki.objects import LLMResponse, ToolCall


def _r(content, calls=None):
    r = LLMResponse(content=content, model="t", prompt_tokens=0, completion_tokens=0,
                    total_tokens=0, elapsed_seconds=0.0)
    r.tool_calls = calls
    return r


class Plugin:
    ALLOWED_METHODS = ["add", "leak"]

    def __init__(self):
        self.calls = 0

    def add(self, a: int, b: int) -> str:
        self.calls += 1
        return str(a + b)

    def leak(self) -> str:
        return "see https://x.test/?api_key=SECRET123&q=1"


def _backend(script):
    """A typed-message backend whose rounds come from *script* (a response, or an exception to raise)."""
    backend = MagicMock()
    backend.supports_native_tools = True
    backend.supports_typed_messages = True
    backend.seen = []
    steps = list(script)

    def chat_turn(messages, tools, system=None, **kw):
        backend.seen.append([m.to_dict() for m in messages])
        step = steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step
    backend.chat_turn.side_effect = chat_turn
    return backend


def _agent(path, script, plugin=None, **kw):
    agent = Agent("worker", _backend(script), stateful=True, checkpoint=path, **kw)
    agent.plugins["p"] = plugin or Plugin()
    return agent


ADD = ToolCall("c1", "p__add", {"a": 2, "b": 3})
ADD2 = ToolCall("c2", "p__add", {"a": 5, "b": 5})


class Base(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.path = os.path.join(self._dir.name, "agent.db")


class TestCheckpointing(Base):

    def test_state_saved_after_each_task_and_refused_until_settled(self):
        agent = _agent(self.path, [_r("hello")])
        self.assertEqual(agent.execute_task("hi", use_plugins=True), "hello")
        saved = CheckpointStore(self.path).load("worker")
        self.assertIsNone(saved.inflight)
        self.assertEqual([m["content"] for m in saved.state["conversation_memory"]], ["hi", "hello"])

        again = _agent(self.path, [_r("x")])
        with self.assertRaises(CheckpointError):
            again.execute_task("hi", use_plugins=True)
        with self.assertRaises(CheckpointError):
            list(again.stream_task("hi"))

    def test_resume_without_inflight_restores_conversation(self):
        _agent(self.path, [_r("hello")]).execute_task("hi", use_plugins=True)
        again = _agent(self.path, [_r("second")])
        self.assertIsNone(again.resume())
        again.execute_task("more", use_plugins=True)
        first_prompt = again.maki.seen[0]
        self.assertTrue(any(m["content"] == "hello" for m in first_prompt) or "hello" in json.dumps(first_prompt))

    def test_resume_continues_from_last_completed_round_without_rerunning_tools(self):
        plugin = Plugin()
        crashed = _agent(self.path, [_r("", [ADD]), RuntimeError("power cut")], plugin)
        with self.assertRaises(Exception):
            crashed.execute_task("add things", use_plugins=True)
        self.assertEqual(plugin.calls, 1)
        saved = CheckpointStore(self.path).load("worker")
        self.assertEqual(saved.inflight["task"], "add things")
        self.assertEqual([m["role"] for m in saved.inflight["messages"]], ["user", "assistant", "tool"])

        fresh_plugin = Plugin()
        fresh = _agent(self.path, [_r("the sum is 5")], fresh_plugin)
        self.assertEqual(fresh.resume(), "the sum is 5")
        self.assertEqual(fresh_plugin.calls, 0)  # the finished tool was replayed from the checkpoint, not run
        sent = fresh.maki.seen[0]
        self.assertEqual([m["role"] for m in sent], ["user", "assistant", "tool"])
        self.assertEqual(sent[2]["content"], "5")
        done = CheckpointStore(self.path).load("worker")
        self.assertIsNone(done.inflight)
        self.assertEqual(done.state["conversation_memory"][-1]["content"], "the sum is 5")
        self.assertEqual(len(done.state["conversation_memory"]), 2)  # one user turn and one answer, not two tasks

    def test_resumed_rounds_count_against_max_rounds(self):
        policy = ToolPolicy(max_rounds=2)
        crashed = _agent(self.path, [_r("", [ADD]), _r("", [ADD2]), RuntimeError("x")], tool_policy=policy)
        with self.assertRaises(Exception):
            crashed.execute_task("t", use_plugins=True)
        fresh = _agent(self.path, [_r("forced answer")], tool_policy=policy)
        self.assertEqual(fresh.resume(), "forced answer")
        self.assertEqual(fresh.maki.chat_turn.call_count, 1)  # both rounds were used up: straight to the closing call

    def test_task_that_never_started_a_round_leaves_no_inflight(self):
        agent = _agent(self.path, [RuntimeError("boom")])
        with self.assertRaises(Exception):
            agent.execute_task("t", use_plugins=True)
        self.assertIsNone(CheckpointStore(self.path).load("worker"))

    def test_discard_lets_a_fresh_run_start_over(self):
        _agent(self.path, [_r("a")]).execute_task("hi", use_plugins=True)
        again = _agent(self.path, [_r("b")])
        again.discard_checkpoint()
        self.assertEqual(again.execute_task("hi", use_plugins=True), "b")
        self.assertEqual(len(CheckpointStore(self.path).load("worker").state["conversation_memory"]), 2)

    def test_resume_with_no_saved_row_is_a_no_op(self):
        agent = _agent(self.path, [_r("a")])
        self.assertIsNone(agent.resume())
        self.assertEqual(agent.execute_task("hi", use_plugins=True), "a")

    def test_resume_and_discard_need_a_checkpoint(self):
        plain = Agent("p", MagicMock())
        with self.assertRaises(ValueError):
            plain.resume()
        with self.assertRaises(ValueError):
            plain.discard_checkpoint()

    def test_streamed_task_saves_state(self):
        backend = MagicMock()
        backend.stream.return_value = iter(["a", "b"])
        agent = Agent("worker", backend, stateful=True, checkpoint=self.path)
        self.assertEqual("".join(agent.stream_task("t")), "ab")
        self.assertEqual(CheckpointStore(self.path).load("worker").state["conversation_memory"][-1]["content"], "ab")

    def test_two_agents_share_a_file_independently(self):
        a = Agent("alpha", _backend([_r("A")]), checkpoint=self.path)
        b = Agent("beta", _backend([_r("B")]), checkpoint=self.path)
        a.execute_task("t")
        b.execute_task("t")
        self.assertEqual(CheckpointStore(self.path).agents(), ["alpha", "beta"])

    def test_unfinished_task_on_a_backend_without_typed_messages_restarts(self):
        store = CheckpointStore(self.path)
        agent = Agent("worker", MagicMock(), checkpoint=store)
        state = agent.to_dict()
        store.save_inflight("worker", {"task": "t", "context": None, "use_plugins": True,
                                       "messages": [{"role": "user", "content": "t"}]}, state)
        plain = MagicMock()
        plain.chat.return_value = _r("fresh start")
        again = Agent("worker", plain, checkpoint=store)
        self.assertEqual(again.resume(), "fresh start")

    def test_a_failing_disk_does_not_lose_the_answer(self):
        agent = _agent(self.path, [_r("fine")])
        agent._checkpoint = MagicMock()
        agent._checkpoint.save_state.side_effect = OSError("disk full")
        self.assertEqual(agent.execute_task("hi", use_plugins=True), "fine")

    def test_damaged_unfinished_task_is_reported(self):
        store = CheckpointStore(self.path)
        store.save_inflight("worker", {"nothing": "useful"}, {"name": "worker"})
        with self.assertRaises(CheckpointError):
            Agent("worker", MagicMock(), checkpoint=store).resume()

    def test_resume_keeps_this_agents_limits_not_the_saved_ones(self):
        _agent(self.path, [_r("a")], memory={"token_budget": 9000}).execute_task("t", use_plugins=True)
        small = _agent(self.path, [], memory={"token_budget": 500})
        small.resume()
        self.assertEqual(small._conversation_memory.token_budget, 500)
        self.assertEqual([m.content for m in small._conversation_memory._messages], ["t", "a"])

    def test_damaged_row_does_not_block_construction_and_can_be_discarded(self):
        store = CheckpointStore(self.path)
        store.save_state("worker", {"x": 1})
        conn = sqlite3.connect(self.path)
        conn.execute("UPDATE checkpoint SET state = '{broken'")
        conn.commit()
        conn.close()
        agent = Agent("worker", _backend([_r("ok")]), checkpoint=self.path)
        agent.maki.supports_native_tools = False
        agent.maki.chat.return_value = _r("ok")
        with self.assertRaises(CheckpointError):
            agent.execute_task("t")
        with self.assertRaises(CheckpointError):
            agent.resume()
        agent.discard_checkpoint()
        self.assertEqual(agent.execute_task("t"), "ok")

    def test_saved_state_of_the_wrong_shape_is_reported_and_keeps_the_agent_blocked(self):
        store = CheckpointStore(self.path)
        store.save_state("worker", {"conversation_memory": "nonsense", "task_history": 5})
        agent = Agent("worker", _backend([_r("ok")]), checkpoint=store)
        with self.assertRaises(CheckpointError):
            agent.resume()
        with self.assertRaises(CheckpointError):
            agent.execute_task("t")

    def test_rounds_do_not_rebuild_the_state(self):
        agent = _agent(self.path, [_r("", [ADD]), _r("", [ADD2]), _r("done")])
        calls = []
        original = agent.to_dict
        agent.to_dict = lambda: calls.append(1) or original()
        agent.execute_task("t", use_plugins=True)
        # first round creates the row, the finished task saves again; the second round reuses the stored state
        self.assertEqual(len(calls), 2)

    def test_a_second_resume_keeps_unsaved_memory(self):
        plugin = Plugin()
        crashed = _agent(self.path, [_r("", [ADD]), RuntimeError("x")], plugin)
        with self.assertRaises(Exception):
            crashed.execute_task("t", use_plugins=True)
        again = _agent(self.path, [_r("", [ADD2]), RuntimeError("y"), _r("final")])
        with self.assertRaises(Exception):
            again.resume()  # fails again after one more round
        again.remember("note", "made after the last save")
        self.assertEqual(again.resume(), "final")
        self.assertEqual(again.recall("note"), "made after the last save")
        self.assertEqual(CheckpointStore(self.path).load("worker").inflight, None)

    def test_unfinished_task_of_the_wrong_shape_is_reported_before_any_state_is_replaced(self):
        store = CheckpointStore(self.path)
        for bad in ({"task": 5, "messages": []}, {"task": "t", "context": "x"},
                    {"task": "t", "messages": ["nope"]}, {"task": "t", "messages": [{"role": "tool", "content": "x"}]}):
            store.save_inflight("worker", bad, {"name": "worker", "memory": {"k": "v"}})
            conn = sqlite3.connect(self.path)
            conn.execute("UPDATE checkpoint SET inflight = ?", (json.dumps(bad),))
            conn.commit()
            conn.close()
            agent = Agent("worker", MagicMock(), checkpoint=store)
            with self.assertRaises(CheckpointError):
                agent.resume()
            self.assertEqual(agent.memory, {})
            with self.assertRaises(CheckpointError):
                agent.execute_task("t")

    def test_bad_checkpoint_argument(self):
        with self.assertRaises(TypeError):
            Agent("a", MagicMock(), checkpoint=42)
        self.assertIsNone(Agent("a", MagicMock(), checkpoint=False)._checkpoint)


CHILD = r"""
import os, sys
from unittest.mock import MagicMock
from maki.agents import Agent
from maki.objects import LLMResponse, ToolCall

class Plugin:
    ALLOWED_METHODS = ["add"]
    def add(self, a: int, b: int) -> str:
        with open(sys.argv[2], "a") as f:
            f.write("ran\n")
        return str(a + b)

def r(content, calls=None):
    x = LLMResponse(content=content, model="t", prompt_tokens=0, completion_tokens=0, total_tokens=0, elapsed_seconds=0.0)
    x.tool_calls = calls
    return x

backend = MagicMock()
backend.supports_native_tools = True
backend.supports_typed_messages = True
rounds = []

def chat_turn(messages, tools, system=None, **kw):
    rounds.append(1)
    if len(rounds) == 1 and sys.argv[3] != "resume":
        return r("", [ToolCall("c1", "p__add", {"a": 2, "b": 3})])
    if sys.argv[3] == "crash":
        os.kill(os.getpid(), 9)   # no cleanup, no atexit, no finally
    seen = [m.content for m in messages if m.role == "tool"]
    return r("sum=" + ",".join(seen))
backend.chat_turn.side_effect = chat_turn

agent = Agent("worker", backend, stateful=True, checkpoint=sys.argv[1])
agent.plugins["p"] = Plugin()
print(agent.resume() if sys.argv[3] == "resume" else agent.execute_task("add", use_plugins=True))
"""


class TestKilledProcess(Base):

    def _run(self, mode):
        import subprocess
        import sys
        marker = os.path.join(self._dir.name, "tool-runs.txt")
        return subprocess.run([sys.executable, "-c", CHILD, self.path, marker, mode],
                              capture_output=True, text=True, timeout=120), marker

    def test_process_killed_with_sigkill_resumes_without_rerunning_the_tool(self):
        done, marker = self._run("crash")
        self.assertEqual(done.returncode, -9, done.stderr)
        done, _ = self._run("resume")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.strip(), "sum=5")
        with open(marker) as f:
            self.assertEqual(f.read().split(), ["ran"])  # the tool ran once, in the process that was killed


class TestStore(Base):

    def test_file_is_private_and_versioned(self):
        CheckpointStore(self.path)
        self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o600)
        conn = sqlite3.connect(self.path)
        self.addCleanup(conn.close)
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION)

    def test_loose_permissions_of_an_existing_file_are_tightened(self):
        open(self.path, "w").close()
        os.chmod(self.path, 0o644)
        CheckpointStore(self.path)
        self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o600)

    def test_newer_schema_is_refused(self):
        conn = sqlite3.connect(self.path)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
        conn.close()
        with self.assertRaises(CheckpointError):
            CheckpointStore(self.path)

    def test_a_file_that_is_not_a_database_is_refused(self):
        with open(self.path, "w") as f:
            f.write("this is not sqlite" * 50)
        with self.assertRaises(CheckpointError):
            CheckpointStore(self.path)

    def test_damaged_row_is_reported(self):
        store = CheckpointStore(self.path)
        store.save_state("a", {"x": 1})
        conn = sqlite3.connect(self.path)
        conn.execute("UPDATE checkpoint SET state = '{not json'")
        conn.commit()
        conn.close()
        with self.assertRaises(CheckpointError):
            store.load("a")
        conn = sqlite3.connect(self.path)
        conn.execute("UPDATE checkpoint SET state = '[1]'")
        conn.commit()
        conn.close()
        with self.assertRaises(CheckpointError):
            store.load("a")

    def test_secrets_never_reach_the_file(self):
        agent = _agent(self.path, [_r("", [ToolCall("c", "p__leak", {})]), RuntimeError("x")])
        with self.assertRaises(Exception):
            agent.execute_task("go", use_plugins=True)
        with open(self.path, "rb") as f:
            raw = f.read()
        self.assertNotIn(b"SECRET123", raw)
        self.assertIn(b"api_key=***", raw)

    def test_deeply_nested_lists_do_not_recurse_forever(self):
        deep = []
        cur = deep
        for _ in range(5000):
            nxt = []
            cur.append(nxt)
            cur = nxt
        self.assertIn("***", json.dumps(redact_value(deep)))
        self.assertEqual(redact_value([["?token=x"]]), [["?token=***"]])

    def test_schema_version_is_pinned(self):
        self.assertEqual(SCHEMA_VERSION, 1)  # bump deliberately, with a migration

    def test_old_sqlite_is_refused_clearly(self):
        from unittest import mock
        with mock.patch("maki.checkpoint.sqlite3.sqlite_version_info", (3, 7, 17)):
            with self.assertRaises(CheckpointError):
                CheckpointStore(self.path)

    def test_a_path_the_os_refuses_is_a_checkpoint_error(self):
        blocker = os.path.join(self._dir.name, "afile")
        open(blocker, "w").close()
        with self.assertRaises(CheckpointError):
            CheckpointStore(os.path.join(blocker, "inside", "agent.db"))  # a parent that is a file

    def test_lone_surrogates_are_saved_and_restored(self):
        store = CheckpointStore(self.path)
        store.save_state("a", {"text": "cut emoji \ud83d"})
        self.assertEqual(store.load("a").state["text"], "cut emoji \ud83d")

    def test_database_that_breaks_after_opening_raises_checkpoint_error(self):
        store = CheckpointStore(self.path)
        store.save_state("a", {"x": 1})
        with open(self.path, "wb") as f:
            f.write(b"garbage" * 500)
        for call in (lambda: store.load("a"), lambda: store.delete("a"), lambda: store.clear_inflight("a"),
                     store.agents):
            with self.assertRaises(CheckpointError):
                call()
        agent = Agent("a", MagicMock(), checkpoint=store)  # constructing must not crash on it
        with self.assertRaises(CheckpointError):
            agent.execute_task("t")

    def test_unusable_path_is_a_checkpoint_error(self):
        with self.assertRaises(CheckpointError):
            CheckpointStore(self._dir.name)  # a directory

    def test_only_credential_keys_are_masked(self):
        out = redact_value({"keyword": "k", "monkey_name": "m", "tokenizer": "t", "api_key": "1", "apiKey": "2",
                            "access-token": "3", "Authorization": "4", "passwords": "5", "client_secret": "6"})
        self.assertEqual([k for k, v in out.items() if v == "***"],
                         ["api_key", "apiKey", "access-token", "Authorization", "passwords", "client_secret"])

    def test_redact_value(self):
        out = redact_value({"password": "p", "nested": {"api_token": "t", "ok": "?key=1"}, "list": ["?token=a"],
                            "n": 3, "auth": {"user": "u"}})
        self.assertEqual(out["password"], "***")
        self.assertEqual(out["nested"]["api_token"], "***")
        self.assertEqual(out["nested"]["ok"], "?key=***")
        self.assertEqual(out["list"], ["?token=***"])
        self.assertEqual(out["n"], 3)
        deep = cur = {}
        for _ in range(40):
            cur["d"] = {}
            cur = cur["d"]
        self.assertIn("***", json.dumps(redact_value(deep)))

    def test_a_row_created_by_another_process_in_between_is_updated_not_an_error(self):
        store = CheckpointStore(self.path)

        def late_state():
            CheckpointStore(self.path).save_state("a", {"v": "other process"})  # the row appears mid-save
            return {"v": "mine"}
        store.save_inflight("a", {"task": "t"}, late_state)
        loaded = store.load("a")
        self.assertEqual(loaded.inflight, {"task": "t"})

    def test_load_missing_and_delete_and_clear(self):
        store = CheckpointStore(self.path)
        self.assertIsNone(store.load("nobody"))
        store.save_inflight("a", {"task": "t"}, {"name": "a"})
        store.clear_inflight("a")
        self.assertIsNone(store.load("a").inflight)
        store.delete("a")
        self.assertEqual(store.agents(), [])

    def test_save_inflight_keeps_the_last_finished_state(self):
        store = CheckpointStore(self.path)
        store.save_state("a", {"v": "finished"})
        store.save_inflight("a", {"task": "t"}, {"v": "ignored"})
        loaded = store.load("a")
        self.assertEqual(loaded.state, {"v": "finished"})
        self.assertEqual(loaded.inflight, {"task": "t"})
        store.save_state("a", {"v": "next"})
        self.assertIsNone(store.load("a").inflight)


if __name__ == "__main__":
    unittest.main()
