"""MemoryPolicy: opt-in limits on what an agent keeps and sends."""
import unittest
from unittest.mock import MagicMock

from maki.agents import Agent
from maki.objects import LLMResponse, MemoryPolicy, ToolCall, clip_entry, clip_text
from maki.test import memory_bench as mb


def _r(content: str) -> LLMResponse:
    return LLMResponse(content=content, model="t", prompt_tokens=0, completion_tokens=0,
                       total_tokens=0, elapsed_seconds=0.0)


class TestClipText(unittest.TestCase):
    def test_short_text_and_no_limit_are_unchanged(self):
        self.assertEqual(clip_text("abc", 10), "abc")
        self.assertEqual(clip_text("abc" * 100, None), "abc" * 100)

    def test_long_text_keeps_head_and_tail_and_says_what_was_cut(self):
        text = "H" * 50 + "M" * 900 + "T" * 50
        out = clip_text(text, 100)
        self.assertTrue(out.startswith("H" * 50))
        self.assertTrue(out.endswith("T" * 50))
        self.assertIn("[... 900 characters omitted ...]", out)
        self.assertNotIn("M", out)

    def test_clip_entry_clips_long_fields_only(self):
        entry = {"task": "short", "result": "r" * 500, "context": {"k": "v" * 500}, "timestamp": 1.5, "n": None}
        out = clip_entry(entry, 100)
        self.assertEqual(out["task"], "short")
        self.assertIn("omitted", out["result"])
        self.assertIn("omitted", out["context"])
        self.assertEqual((out["timestamp"], out["n"]), (1.5, None))
        self.assertIs(clip_entry(entry, None), entry)


class TestPolicyValidation(unittest.TestCase):
    def test_defaults_reproduce_today(self):
        p = MemoryPolicy()
        self.assertEqual((p.history_mode, p.token_budget, p.tool_result_chars, p.history_entry_chars),
                         ("text", None, None, None))

    def test_rejects_bad_values(self):
        for kwargs in ({"history_mode": "both"}, {"token_budget": 0}, {"tool_result_chars": -1},
                       {"history_entry_chars": True}, {"token_budget": "10"}):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                MemoryPolicy(**kwargs)

    def test_from_dict_round_trips_and_rejects_unknown_keys(self):
        p = MemoryPolicy(history_mode="messages", token_budget=900, tool_result_chars=200)
        self.assertEqual(MemoryPolicy.from_dict(p.to_dict()), p)
        with self.assertRaises(ValueError):
            MemoryPolicy.from_dict({"history_mod": "messages"})
        with self.assertRaises(ValueError):
            MemoryPolicy.from_dict("messages")

    def test_agent_accepts_policy_dict_or_none_and_rejects_other_types(self):
        backend = mb.RecordingBackend()
        self.assertIsNone(Agent("a", backend).memory_policy)
        self.assertEqual(Agent("a", backend, memory={"token_budget": 50}).memory_policy.token_budget, 50)
        with self.assertRaises(TypeError):
            Agent("a", backend, memory="messages")


class TestTokenBudget(unittest.TestCase):
    def test_policy_budget_reaches_the_conversation_memory(self):
        agent = Agent("a", mb.RecordingBackend(), stateful=True, memory=MemoryPolicy(token_budget=500))
        self.assertEqual(agent._conversation_memory.token_budget, 500)

    def test_a_smaller_budget_sends_fewer_tokens(self):
        base = mb.run_stateful(turns=12)
        small = mb.run_stateful(turns=12, memory=MemoryPolicy(token_budget=1000))
        self.assertLess(small[-1], base[-1])


class TestStructuredHistory(unittest.TestCase):
    def _agent(self, backend, **policy):
        return Agent("a", backend, stateful=True, memory=MemoryPolicy(history_mode="messages", **policy))

    def test_supporting_backend_gets_history_and_no_prompt_text(self):
        backend = mb.RecordingBackend(answer_chars=20)
        agent = self._agent(backend)
        agent.execute_task("first")
        agent.execute_task("second")
        first, second = backend.calls
        self.assertEqual(first["history"], [])
        self.assertEqual([m.content for m in second["history"]], ["first", "x" * 20])
        self.assertNotIn("Prior conversation", second["prompt"])
        self.assertEqual(second["prompt"], "second")

    def test_recorded_conversation_is_the_same_in_both_modes(self):
        texts = []
        for mode in ("text", "messages"):
            agent = Agent("a", mb.RecordingBackend(20), stateful=True, memory=MemoryPolicy(history_mode=mode))
            agent.execute_task("one")
            agent.execute_task("two")
            texts.append([m.content for m in agent._conversation_memory.messages()])
        self.assertEqual(texts[0], texts[1])

    def test_backend_without_history_support_falls_back_to_text(self):
        seen = []

        class Plain:
            model, temperature, supports_native_tools = "p", 0.0, False

            def request(self, prompt):
                return self.chat(prompt)

            def chat(self, prompt, system=None):  # no history parameter at all
                seen.append(prompt)
                return _r("ok")

        agent = self._agent(Plain())
        agent.execute_task("first")
        agent.execute_task("second")
        self.assertIn("Prior conversation", seen[1])
        self.assertIn("first", seen[1])

    def test_a_mock_backend_is_not_mistaken_for_one_that_supports_history(self):
        backend = MagicMock()
        backend.chat.return_value = _r("ok")
        agent = self._agent(backend)
        agent.execute_task("first")
        agent.execute_task("second")
        self.assertIn("Prior conversation", backend.chat.call_args_list[1].args[0])
        self.assertNotIn("history", backend.chat.call_args_list[1].kwargs)

    def test_stateless_agent_never_sends_history(self):
        backend = mb.RecordingBackend(20)
        agent = Agent("a", backend, memory=MemoryPolicy(history_mode="messages"))
        agent.execute_task("one")
        agent.execute_task("two")
        self.assertEqual(backend.calls[1]["history"], [])

    def test_stream_task_sends_history_too(self):
        class Streaming(mb.RecordingBackend):
            def stream(self, prompt, history=None, config=None, system=None):
                self.calls.append({"prompt": prompt, "history": list(history or []), "system": system or ""})
                yield "abc"

        backend = Streaming()
        agent = self._agent(backend)
        list(agent.stream_task("one"))
        list(agent.stream_task("two"))
        self.assertEqual([m.content for m in backend.calls[1]["history"]], ["one", "abc"])

    def test_native_tool_path_prefixes_prior_turns_as_messages(self):
        sent = []

        class Native(mb.RecordingBackend):
            supports_native_tools = True

            def to_tool_schemas(self, specs):
                return specs

            def chat_with_tools(self, messages, tools, system=None, config=None):
                sent.append(list(messages))
                return _r("done"), None, messages

        agent = self._agent(Native(20))
        agent.plugins["bulk"] = mb._BulkPlugin(5)
        agent.execute_task("first", use_plugins=True)
        agent.execute_task("second", use_plugins=True)
        self.assertEqual([m["role"] for m in sent[1]], ["user", "assistant", "user"])
        self.assertEqual(sent[1][-1]["content"], "second")

    def test_messages_mode_sends_no_more_tokens_than_text_mode_on_any_turn(self):
        text = mb.run_stateful(turns=20)
        messages = mb.run_stateful(turns=20, memory=MemoryPolicy(history_mode="messages"))
        self.assertTrue(all(m <= t for m, t in zip(messages, text)))  # no "Task:"/"Response:" labels
        self.assertLess(sum(messages), sum(text))


class TestToolResultClipping(unittest.TestCase):
    def test_native_path_clips_results_over_the_cap(self):
        rounds = mb.run_tool_heavy(rounds=4, result_chars=20000, memory=MemoryPolicy(tool_result_chars=400))
        self.assertLess(rounds[-1], 1000)

    def test_native_path_is_untouched_without_a_policy(self):
        self.assertGreater(mb.run_tool_heavy(rounds=2, result_chars=20000)[-1], 10000)

    def test_short_results_and_errors_are_not_clipped(self):
        agent = Agent("a", mb.RecordingBackend(), memory=MemoryPolicy(tool_result_chars=50))
        agent.plugins["bulk"] = mb._BulkPlugin(10)
        self.assertEqual(agent._execute_tool_call(ToolCall("1", "bulk__fetch", {})), "y" * 10)
        error = agent._execute_tool_call(ToolCall("2", "nope__x", {}))
        self.assertTrue(error.startswith("Error:"))

    def test_legacy_tool_directive_path_clips_too(self):
        agent = Agent("a", mb.RecordingBackend(), memory=MemoryPolicy(tool_result_chars=100))
        agent.plugins["bulk"] = mb._BulkPlugin(5000)
        out = agent._run_tool_call(True, "raw", {"plugin": "bulk", "method": "fetch", "args": {}})
        self.assertIn("characters omitted", out["result"])
        self.assertLess(len(out["result"]), 200)


class TestHistoryEntryBound(unittest.TestCase):
    def test_task_history_is_clipped_but_the_conversation_keeps_the_full_answer(self):
        backend = mb.RecordingBackend(answer_chars=5000)
        agent = Agent("a", backend, stateful=True, memory=MemoryPolicy(history_entry_chars=200))
        agent.execute_task("go", context={"big": "c" * 3000})
        entry = agent.task_history[-1]
        self.assertIn("omitted", entry["result"])
        self.assertIn("omitted", entry["context"])
        self.assertEqual(len(agent._conversation_memory.messages()[1].content), 5000)

    def test_footprint_shrinks(self):
        full = mb.history_footprint(turns=10, answer_chars=5000)
        bound = mb.history_footprint(turns=10, answer_chars=5000, memory=MemoryPolicy(history_entry_chars=300))
        self.assertLess(bound, full // 5)

    def test_reasoning_history_is_clipped(self):
        backend = mb.RecordingBackend(answer_chars=5000)
        agent = Agent("a", backend, memory=MemoryPolicy(history_entry_chars=200))
        agent.think_step_by_step("a problem", steps=2)
        self.assertIn("omitted", agent.reasoning_history[-1]["result"])


class TestSerialization(unittest.TestCase):
    def test_policy_round_trips_through_to_dict(self):
        policy = MemoryPolicy(history_mode="messages", token_budget=700, tool_result_chars=300)
        agent = Agent("a", mb.RecordingBackend(), stateful=True, memory=policy)
        restored = Agent.from_dict(agent.to_dict(), mb.RecordingBackend())
        self.assertEqual(restored.memory_policy, policy)

    def test_old_dicts_without_a_policy_still_load(self):
        data = Agent("a", mb.RecordingBackend()).to_dict()
        data.pop("memory_policy")
        self.assertIsNone(Agent.from_dict(data, mb.RecordingBackend()).memory_policy)


class TestBenchmarkWithPolicy(unittest.TestCase):
    """The numbers the plan is judged on (checkpoint C1)."""

    def test_policy_beats_baseline_on_tool_and_footprint_without_losing_recall(self):
        base = mb.run_all()
        tuned = mb.run_all(memory=MemoryPolicy(history_mode="messages", tool_result_chars=2000,
                                               history_entry_chars=2000))
        self.assertLess(tuned["tool_heavy_tokens_last_round"], base["tool_heavy_tokens_last_round"] // 4)
        self.assertLess(tuned["task_history_chars"], base["task_history_chars"] // 4)
        self.assertGreaterEqual(tuned["recall_turn_15"], base["recall_turn_15"])


if __name__ == "__main__":
    unittest.main()
