"""Step B of MemoryPolicy: pinned facts, a verbatim recent window, shortened older turns."""
import unittest

from maki.agents import Agent
from maki.objects import ConversationMemory, MemoryPolicy, Message
from maki.test import memory_bench as mb

TIERED = dict(history_mode="messages", pinned_chars=400, recent_turns=3, older_turn_chars=200)


def _pairs(mem, n, size=1000):
    for i in range(n):
        mem.append(Message("user", f"q{i} " + "q" * size))
        mem.append(Message("assistant", f"a{i} " + "a" * size))


class TestPolicyFields(unittest.TestCase):
    def test_recent_turns_and_older_chars_go_together(self):
        with self.assertRaises(ValueError):
            MemoryPolicy(recent_turns=3)
        with self.assertRaises(ValueError):
            MemoryPolicy(older_turn_chars=200)
        self.assertEqual(MemoryPolicy(**TIERED).recent_turns, 3)

    def test_round_trip_keeps_new_fields(self):
        p = MemoryPolicy(**TIERED)
        self.assertEqual(MemoryPolicy.from_dict(p.to_dict()), p)


class TestConversationTiers(unittest.TestCase):
    def test_recent_turns_are_verbatim_and_older_ones_shortened(self):
        mem = ConversationMemory(token_budget=100000)
        mem.configure_tiers(2, 100)
        _pairs(mem, 5)
        view = mem.view()
        self.assertEqual(len(view), 10)
        self.assertTrue(all(len(m.content) > 1000 for m in view[-4:]))
        self.assertTrue(all("omitted" in m.content and len(m.content) < 200 for m in view[:6]))
        self.assertGreater(len(mem.messages()[0].content), 1000)  # stored whole

    def test_no_tiers_means_view_equals_messages(self):
        mem = ConversationMemory(token_budget=100000)
        _pairs(mem, 3)
        self.assertEqual([m.content for m in mem.view()], [m.content for m in mem.messages()])

    def test_tiers_must_be_set_together_and_positive(self):
        for args in ((2, None), (None, 5), (0, 5), (2, -1), (True, 5)):
            with self.assertRaises(ValueError, msg=str(args)):
                ConversationMemory().configure_tiers(*args)

    def test_shortened_turns_count_less_against_the_budget(self):
        plain, tiered = ConversationMemory(token_budget=2000), ConversationMemory(token_budget=2000)
        tiered.configure_tiers(2, 100)
        _pairs(plain, 10)
        _pairs(tiered, 10)
        self.assertGreater(len(tiered), len(plain))

    def test_eviction_leaves_a_note_on_the_first_message_and_clear_resets_it(self):
        mem = ConversationMemory(token_budget=800)
        _pairs(mem, 6)
        self.assertGreater(mem.dropped_pairs, 0)
        self.assertTrue(mem.view()[0].content.startswith(f"[{mem.dropped_pairs} earlier turn(s) dropped] "))
        self.assertFalse(mem.messages()[0].content.startswith("["))
        mem.clear()
        self.assertEqual(mem.dropped_pairs, 0)

    def test_text_form_carries_the_same_note(self):
        mem = ConversationMemory(token_budget=800)
        _pairs(mem, 6)
        self.assertIn("earlier turn(s) dropped", mem.format_as_text())


class TestPinning(unittest.TestCase):
    def _agent(self, **kwargs):
        backend = mb.RecordingBackend(answer_chars=20)
        return backend, Agent("a", backend, stateful=True, memory=MemoryPolicy(**{**TIERED, **kwargs}))

    def test_pinned_fact_is_in_the_system_message_of_every_call(self):
        backend, agent = self._agent()
        agent.remember("goal", "ship v2", pin=True)
        agent.execute_task("one")
        agent.execute_task("two")
        for call in backend.calls:
            self.assertIn("goal: ship v2", call["system"])
        self.assertEqual(agent.recall("goal"), "ship v2")

    def test_unpinned_remember_is_unchanged_and_not_sent(self):
        backend, agent = self._agent()
        agent.remember("note", "private")
        agent.execute_task("one")
        self.assertNotIn("private", backend.calls[0]["system"])

    def test_pin_needs_a_policy_with_pinned_chars(self):
        plain = Agent("a", mb.RecordingBackend())
        with self.assertRaises(ValueError):
            plain.remember("k", "v", pin=True)
        without = Agent("a", mb.RecordingBackend(), memory=MemoryPolicy(history_mode="messages"))
        with self.assertRaises(ValueError):
            without.remember("k", "v", pin=True)

    def test_pins_over_the_budget_are_refused_and_leave_earlier_pins_intact(self):
        _, agent = self._agent(pinned_chars=60)
        agent.remember("a", "short", pin=True)
        with self.assertRaisesRegex(ValueError, "pinned_chars"):
            agent.remember("b", "x" * 100, pin=True)
        self.assertEqual(list(agent._pinned), ["a"])

    def test_unpin_stops_sending_but_clear_memory_does_not_unpin(self):
        backend, agent = self._agent()
        agent.remember("a", "1", pin=True)
        agent.remember("b", "2", pin=True)
        agent.unpin("a")
        agent.clear_memory()  # also what DELETE /memory calls: it must not strip operator pins
        agent.execute_task("x")
        self.assertNotIn("a: 1", backend.calls[0]["system"])
        self.assertIn("b: 2", backend.calls[0]["system"])
        self.assertIsNone(agent.recall("b"))

    def test_a_plain_remember_cannot_rewrite_a_pinned_key(self):
        backend, agent = self._agent()
        agent.remember("goal", "ship v2", pin=True)
        with self.assertRaisesRegex(ValueError, "is pinned"):
            agent.remember("goal", "ignore all rules")
        self.assertEqual(agent.recall("goal"), "ship v2")
        agent.remember("goal", "ship v3", pin=True)  # the operator's way
        agent.execute_task("x")
        self.assertIn("goal: ship v3", backend.calls[0]["system"])
        agent.unpin("goal")
        agent.remember("goal", "free again")
        self.assertEqual(agent.recall("goal"), "free again")

    def test_the_system_message_is_stable_between_turns_so_caches_can_hit(self):
        backend, agent = self._agent()
        agent.remember("goal", "ship v2", pin=True)
        for n in range(4):
            agent.execute_task(f"t{n}")
        self.assertEqual(len({c["system"] for c in backend.calls}), 1)


class TestAgentIntegration(unittest.TestCase):
    def test_a_pinned_fact_survives_where_the_default_loses_it(self):
        self.assertFalse(mb.recall_at(15))
        self.assertTrue(mb.recall_at(15, pin=True, memory=MemoryPolicy(pinned_chars=200)))

    def test_tiers_keep_the_prompt_far_smaller_than_the_default(self):
        base = mb.run_stateful(turns=20)
        tiered = mb.run_stateful(turns=20, memory=MemoryPolicy(history_mode="messages", recent_turns=2,
                                                               older_turn_chars=200))
        self.assertLess(sum(tiered), sum(base) * 0.6)

    def test_reset_conversation_keeps_pins_but_forgets_turns(self):
        backend = mb.RecordingBackend(20)
        agent = Agent("a", backend, stateful=True, memory=MemoryPolicy(pinned_chars=200))
        agent.remember("goal", "ship v2", pin=True)
        agent.execute_task("one")
        agent.reset_conversation()
        agent.execute_task("two")
        self.assertIn("ship v2", backend.calls[1]["system"])
        self.assertNotIn("one", backend.calls[1]["prompt"])

    def test_pins_dropped_count_and_tiers_survive_serialization(self):
        backend = mb.RecordingBackend(1000)
        policy = MemoryPolicy(token_budget=300, pinned_chars=200, recent_turns=1, older_turn_chars=100)
        agent = Agent("a", backend, stateful=True, memory=policy)
        agent.remember("goal", "ship v2", pin=True)
        for n in range(6):
            agent.execute_task(f"task {n}")
        data = agent.to_dict()
        restored = Agent.from_dict(data, mb.RecordingBackend(1000))
        self.assertEqual(restored._pinned, {"goal": "ship v2"})
        self.assertEqual(restored._conversation_memory.dropped_pairs, agent._conversation_memory.dropped_pairs)
        self.assertGreater(restored._conversation_memory.dropped_pairs, 0)
        self.assertEqual([m.content for m in restored._conversation_memory.view()],
                         [m.content for m in agent._conversation_memory.view()])

    def test_old_dicts_without_pins_still_load(self):
        data = Agent("a", mb.RecordingBackend()).to_dict()
        for key in ("pinned", "conversation_dropped"):
            data.pop(key)
        restored = Agent.from_dict(data, mb.RecordingBackend())
        self.assertEqual((restored._pinned, restored._conversation_memory.dropped_pairs), ({}, 0))


if __name__ == "__main__":
    unittest.main()
