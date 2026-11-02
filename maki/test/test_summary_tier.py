"""Tests for the summary memory tier: MemoryPolicy(summarize_older=True)."""
import unittest
from unittest.mock import MagicMock

from maki.agents import Agent
from maki.objects import ConversationMemory, LLMResponse, MemoryPolicy, Message


def _r(content):
    return LLMResponse(content=content, model="t", prompt_tokens=1, completion_tokens=1,
                       total_tokens=2, elapsed_seconds=0.0)


def _backend(summary="SUMMARY", fail=False):
    backend = MagicMock()
    backend.supports_native_tools = False
    backend.summary_prompts = []
    backend.task_prompts = []

    def chat(prompt, **kwargs):
        if "running summary" in prompt:
            backend.summary_prompts.append(prompt)
            if fail:
                raise RuntimeError("model down")
            return _r(summary)
        backend.task_prompts.append(prompt)
        return _r("r" * 80)

    backend.chat.side_effect = chat
    return backend


def _agent(backend=None, **policy):
    policy = {"token_budget": 50, "summarize_older": True, **policy}
    return Agent("a", backend or _backend(), stateful=True, memory=MemoryPolicy(**policy))


def _run(agent, n):
    for i in range(n):
        agent.execute_task(f"task {i}")


class TestPolicy(unittest.TestCase):

    def test_defaults_off(self):
        p = MemoryPolicy()
        self.assertFalse(p.summarize_older)
        self.assertIsNone(p.summary_chars)

    def test_validation(self):
        with self.assertRaises(ValueError):
            MemoryPolicy(summary_chars=100)  # needs summarize_older
        with self.assertRaises(ValueError):
            MemoryPolicy(summarize_older="yes")
        with self.assertRaises(ValueError):
            MemoryPolicy(summarize_older=True, summary_chars=0)

    def test_roundtrip_and_yaml_style_dict(self):
        p = MemoryPolicy.from_dict({"summarize_older": True, "summary_chars": 300})
        self.assertEqual(MemoryPolicy.from_dict(p.to_dict()), p)


class TestConversationMemory(unittest.TestCase):

    def _fill(self, memory, n):
        for i in range(n):
            memory.append(Message("user", f"q{i} " + "x" * 60))
            memory.append(Message("assistant", f"a{i} " + "y" * 60))

    def test_nothing_collected_by_default(self):
        m = ConversationMemory(token_budget=40)
        self._fill(m, 6)
        self.assertGreater(m.dropped_pairs, 0)
        self.assertEqual(m.take_pending(), [])

    def test_evicted_pairs_collected_in_order_and_handed_over_once(self):
        m = ConversationMemory(token_budget=40)
        m.collect_evicted = True
        self._fill(m, 6)
        pending = m.take_pending()
        self.assertEqual(len(pending), m.dropped_pairs)
        self.assertTrue(pending[0][0].content.startswith("q0"))
        self.assertEqual(m.take_pending(), [])

    def test_view_shows_summary_then_unsummarized_drops(self):
        m = ConversationMemory(token_budget=40)
        m.collect_evicted = True
        self._fill(m, 4)
        folded = len(m.take_pending())
        m.set_summary("they like tea", folded)
        self.assertTrue(m.view()[0].content.startswith("[Summary of earlier conversation: they like tea] "))
        self.assertNotIn("dropped", m.view()[0].content)
        self._fill(m, 2)  # more evictions the summary does not cover yet
        first = m.view()[0].content
        self.assertIn("they like tea", first)
        self.assertRegex(first, r"\[\d+ earlier turn\(s\) dropped\]")

    def test_summary_counts_against_budget(self):
        m = ConversationMemory(token_budget=60)
        self._fill(m, 3)
        kept = len(m)
        m.set_summary("s" * 200, 0)  # ~50 tokens of the 60
        self.assertLess(len(m), kept)

    def test_pending_is_bounded(self):
        m = ConversationMemory(token_budget=10_000, max_entries=4)
        m.collect_evicted = True
        for i in range(200):
            m.append(Message("user", "u"))
            m.append(Message("assistant", "a"))
        self.assertLessEqual(len(m.take_pending()), ConversationMemory._MAX_PENDING_PAIRS)

    def test_clear_resets_summary_state(self):
        m = ConversationMemory(token_budget=40)
        m.collect_evicted = True
        self._fill(m, 5)
        m.set_summary("x", 1)
        m.clear()
        self.assertEqual((m.summary, m.summarized_pairs, m.take_pending()), ("", 0, []))


class TestAgent(unittest.TestCase):

    def test_summary_is_made_and_sent_with_the_next_task(self):
        backend = _backend()
        agent = _agent(backend)
        _run(agent, 6)
        self.assertTrue(backend.summary_prompts)
        self.assertIn("Summary of earlier conversation: SUMMARY", backend.task_prompts[-1])
        self.assertEqual(agent._conversation_memory.summary, "SUMMARY")

    def test_summary_prompt_carries_previous_summary_and_evicted_turns(self):
        backend = _backend()
        agent = _agent(backend)
        _run(agent, 8)
        self.assertIn("(none yet)", backend.summary_prompts[0])
        self.assertIn("task 0", backend.summary_prompts[0])
        self.assertIn("Current summary:\nSUMMARY", backend.summary_prompts[-1])

    def test_summary_call_is_counted_in_usage_and_oversized_summary_warns(self):
        from maki.backend import LLMBackend

        class Counting(LLMBackend):
            model, temperature = "t", 0.0

            def request(self, prompt):
                return _r("r" * 80)

            def chat(self, prompt, history=None, config=None, system=None, images=None):
                return _r("SUMMARY" if "running summary" in prompt else "r" * 80)

        agent = Agent("a", Counting(), stateful=True,
                      memory=MemoryPolicy(token_budget=50, summarize_older=True))
        _run(agent, 6)
        self.assertGreater(agent._conversation_memory.summarized_pairs, 0)
        calls_per_task = agent.usage.calls
        self.assertGreater(calls_per_task, 6)  # six tasks plus at least one summary call
        with self.assertLogs("maki.agents.agent", level="WARNING"):
            Agent("b", Counting(), stateful=True,
                  memory=MemoryPolicy(token_budget=100, summarize_older=True, summary_chars=2000))

    def test_summary_is_clipped_to_summary_chars(self):
        agent = _agent(_backend("z" * 5000), summary_chars=100)
        _run(agent, 6)
        self.assertLess(len(agent._conversation_memory.summary), 200)  # 100 kept + clip marker

    def test_failure_degrades_to_dropped_note_and_task_still_succeeds(self):
        backend = _backend(fail=True)
        agent = _agent(backend)
        _run(agent, 6)  # must not raise
        self.assertEqual(agent._conversation_memory.summary, "")
        self.assertIn("earlier turn(s) dropped]", backend.task_prompts[-1])

    def test_empty_summary_is_not_stored(self):
        agent = _agent(_backend(summary="  "))
        _run(agent, 6)
        self.assertEqual(agent._conversation_memory.summary, "")

    def test_custom_summarizer_overrides_the_backend_call(self):
        backend = _backend()
        agent = _agent(backend)
        seen = []
        agent.summarizer = lambda previous, messages: (seen.append((previous, len(messages))) or "custom")
        _run(agent, 6)
        self.assertEqual(backend.summary_prompts, [])
        self.assertEqual(agent._conversation_memory.summary, "custom")
        self.assertEqual(seen[0][0], "")
        self.assertEqual(seen[0][1] % 2, 0)

    def test_off_by_default_makes_no_extra_calls(self):
        backend = _backend()
        agent = Agent("a", backend, stateful=True, memory=MemoryPolicy(token_budget=50))
        _run(agent, 6)
        self.assertEqual(backend.summary_prompts, [])
        self.assertEqual(agent._conversation_memory.take_pending(), [])

    def test_stateless_agent_never_summarises(self):
        backend = _backend()
        agent = Agent("a", backend, memory=MemoryPolicy(token_budget=50, summarize_older=True))
        _run(agent, 6)
        self.assertEqual(backend.summary_prompts, [])

    def test_serialisation_roundtrip_keeps_summary(self):
        agent = _agent()
        _run(agent, 6)
        restored = Agent.from_dict(agent.to_dict(), _backend())
        memory = restored._conversation_memory
        self.assertEqual(memory.summary, "SUMMARY")
        self.assertEqual(memory.summarized_pairs, agent._conversation_memory.summarized_pairs)
        self.assertTrue(memory.collect_evicted)

    def test_unfolded_pairs_survive_a_snapshot(self):
        agent = _agent()
        agent._conversation_memory._pending.append((Message("user", "late q"), Message("assistant", "late a")))
        restored = Agent.from_dict(agent.to_dict(), _backend())
        pairs = restored._conversation_memory.take_pending()
        self.assertEqual([(u.content, a.content) for u, a in pairs], [("late q", "late a")])

    def test_summary_prompt_is_bounded_for_a_huge_eviction(self):
        backend = _backend()
        agent = _agent(backend)
        big = [(Message("user", "u" * 5000), Message("assistant", "a" * 5000)) for _ in range(50)]
        agent._conversation_memory._pending.extend(big)
        agent.execute_task("go")
        self.assertLess(len(backend.summary_prompts[0]), 20000)

    def test_text_history_prints_summary_on_its_own_line(self):
        backend = _backend()
        agent = _agent(backend)
        _run(agent, 6)
        block = agent._conversation_memory.format_as_text()
        self.assertIn("\nSummary of earlier conversation: SUMMARY\n", block)
        self.assertNotIn("Task: [Summary", block)

    def test_reset_conversation_forgets_the_summary(self):
        agent = _agent()
        _run(agent, 6)
        agent.reset_conversation()
        self.assertEqual(agent._conversation_memory.summary, "")

    def test_structured_history_mode_carries_the_summary(self):
        backend = _backend()
        backend.supports_history = True
        agent = _agent(backend, history_mode="messages")
        _run(agent, 6)
        history = agent._history_messages()
        self.assertTrue(history[0].content.startswith("[Summary of earlier conversation: SUMMARY]"))

    def test_stream_task_folds_pending_before_streaming(self):
        backend = _backend()
        backend.stream.return_value = iter(["ok"])
        agent = _agent(backend)
        _run(agent, 6)
        before = len(backend.summary_prompts)
        _run(agent, 2)  # leaves evicted turns pending only if the last execute did not fold them
        agent._conversation_memory._pending.append((Message("user", "late"), Message("assistant", "reply")))
        list(agent.stream_task("again"))
        self.assertGreater(len(backend.summary_prompts), before)


if __name__ == "__main__":
    unittest.main()
