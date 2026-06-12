"""
Tests for ConversationMemory — token-budgeted conversation history.

Covers: append/eviction, token estimation, format_as_text, serialisation,
ChatSession integration, and Agent stateful-mode integration.
"""

import unittest
from unittest.mock import MagicMock, patch

from maki.objects import ConversationMemory, LLMResponse, Message
from maki.session import ChatSession


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(role, content):
    return Message(role=role, content=content)


def _pair(task, result):
    return _msg("user", task), _msg("assistant", result)


def _mock_llm(response_content="ok"):
    llm = MagicMock()
    llm.chat.return_value = LLMResponse(
        content=response_content, model="test", prompt_tokens=1,
        completion_tokens=1, total_tokens=2, elapsed_seconds=0.1,
    )
    return llm


# ---------------------------------------------------------------------------
# ConversationMemory — construction
# ---------------------------------------------------------------------------

class TestConversationMemoryInit(unittest.TestCase):

    def test_defaults(self):
        mem = ConversationMemory()
        self.assertEqual(mem.token_budget, ConversationMemory.DEFAULT_TOKEN_BUDGET)
        self.assertEqual(mem.max_entries, ConversationMemory.DEFAULT_MAX_ENTRIES)
        self.assertEqual(len(mem), 0)

    def test_custom_budget(self):
        mem = ConversationMemory(token_budget=512, max_entries=20)
        self.assertEqual(mem.token_budget, 512)
        self.assertEqual(mem.max_entries, 20)

    def test_invalid_token_budget(self):
        with self.assertRaises(ValueError):
            ConversationMemory(token_budget=0)
        with self.assertRaises(ValueError):
            ConversationMemory(token_budget=-1)

    def test_invalid_max_entries(self):
        with self.assertRaises(ValueError):
            ConversationMemory(max_entries=1)
        with self.assertRaises(ValueError):
            ConversationMemory(max_entries=0)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

class TestTokenEstimation(unittest.TestCase):

    def test_empty_string_returns_one(self):
        self.assertEqual(ConversationMemory._estimate_tokens(""), 1)

    def test_four_chars_one_token(self):
        self.assertEqual(ConversationMemory._estimate_tokens("abcd"), 1)

    def test_eight_chars_two_tokens(self):
        self.assertEqual(ConversationMemory._estimate_tokens("abcdefgh"), 2)

    def test_total_tokens_sums_all_messages(self):
        mem = ConversationMemory()
        mem._messages.append(_msg("user", "a" * 40))      # 10 tokens
        mem._messages.append(_msg("assistant", "b" * 20)) # 5 tokens
        self.assertEqual(mem._total_tokens(), 15)


# ---------------------------------------------------------------------------
# Append and eviction
# ---------------------------------------------------------------------------

class TestAppendAndEviction(unittest.TestCase):

    def test_simple_append(self):
        mem = ConversationMemory()
        u, a = _pair("hello", "world")
        mem.append(u)
        mem.append(a)
        self.assertEqual(len(mem), 2)

    def test_no_eviction_under_budget(self):
        mem = ConversationMemory(token_budget=1000, max_entries=100)
        for i in range(10):
            u, a = _pair(f"task {i}", f"result {i}")
            mem.append(u)
            mem.append(a)
        self.assertEqual(len(mem), 20)

    def test_max_entries_cap_evicts_oldest_pair(self):
        # max_entries=4 → 2 pairs
        mem = ConversationMemory(token_budget=99999, max_entries=4)
        u1, a1 = _pair("task1", "r1")
        u2, a2 = _pair("task2", "r2")
        u3, a3 = _pair("task3", "r3")
        mem.append(u1); mem.append(a1)
        mem.append(u2); mem.append(a2)
        mem.append(u3); mem.append(a3)
        # After 6 appends with cap=4: oldest pair (task1/r1) evicted
        self.assertEqual(len(mem), 4)
        msgs = mem.messages()
        self.assertEqual(msgs[0].content, "task2")
        self.assertEqual(msgs[2].content, "task3")

    def test_token_budget_evicts_oldest_pair(self):
        # Each message content is 40 chars → 10 tokens.
        # Two pairs = 4 messages × 10 tokens = 40 tokens.
        # budget=25 → first pair (20 tokens) must go after we add the second.
        mem = ConversationMemory(token_budget=25, max_entries=200)
        mem.append(_msg("user",      "a" * 40))   # 10 tok → total 10
        mem.append(_msg("assistant", "b" * 40))   # 10 tok → total 20, ok
        mem.append(_msg("user",      "c" * 40))   # 10 tok → total 30, over; trim → 20
        mem.append(_msg("assistant", "d" * 40))   # 10 tok → total 30, over; trim → 20
        self.assertEqual(len(mem), 2)
        self.assertEqual(mem.messages()[0].content, "c" * 40)

    def test_most_recent_pair_never_evicted(self):
        # Budget so tiny that even a single pair exceeds it.
        # We must still keep the latest pair.
        mem = ConversationMemory(token_budget=1, max_entries=200)
        mem.append(_msg("user",      "x" * 40))
        mem.append(_msg("assistant", "y" * 40))
        self.assertEqual(len(mem), 2)

    def test_clear_empties_memory(self):
        mem = ConversationMemory()
        u, a = _pair("q", "a")
        mem.append(u); mem.append(a)
        mem.clear()
        self.assertEqual(len(mem), 0)

    def test_messages_returns_snapshot(self):
        mem = ConversationMemory()
        u, a = _pair("q", "a")
        mem.append(u); mem.append(a)
        snap = mem.messages()
        mem.clear()
        self.assertEqual(len(snap), 2)  # snapshot unaffected
        self.assertEqual(len(mem), 0)


# ---------------------------------------------------------------------------
# Property setters (resize)
# ---------------------------------------------------------------------------

class TestPropertySetters(unittest.TestCase):

    def test_token_budget_setter_trims(self):
        mem = ConversationMemory(token_budget=99999)
        for i in range(5):
            mem.append(_msg("user",      "a" * 40))
            mem.append(_msg("assistant", "b" * 40))
        self.assertEqual(len(mem), 10)
        # Reduce budget to 25 tokens → should keep only most recent pair
        mem.token_budget = 25
        self.assertEqual(len(mem), 2)

    def test_max_entries_setter_trims(self):
        mem = ConversationMemory(max_entries=200)
        for i in range(5):
            mem.append(_msg("user",      f"q{i}"))
            mem.append(_msg("assistant", f"a{i}"))
        self.assertEqual(len(mem), 10)
        mem.max_entries = 4
        self.assertEqual(len(mem), 4)

    def test_invalid_token_budget_setter(self):
        mem = ConversationMemory()
        with self.assertRaises(ValueError):
            mem.token_budget = 0

    def test_invalid_max_entries_setter(self):
        mem = ConversationMemory()
        with self.assertRaises(ValueError):
            mem.max_entries = 1


# ---------------------------------------------------------------------------
# format_as_text
# ---------------------------------------------------------------------------

class TestFormatAsText(unittest.TestCase):

    def test_empty_returns_empty_string(self):
        mem = ConversationMemory()
        self.assertEqual(mem.format_as_text(), "")

    def test_single_pair(self):
        mem = ConversationMemory()
        mem.append(_msg("user",      "do this"))
        mem.append(_msg("assistant", "done"))
        text = mem.format_as_text()
        self.assertIn("Prior conversation", text)
        self.assertIn("Task: do this", text)
        self.assertIn("Response: done", text)

    def test_multiple_pairs(self):
        mem = ConversationMemory()
        for i in range(3):
            mem.append(_msg("user",      f"task{i}"))
            mem.append(_msg("assistant", f"result{i}"))
        text = mem.format_as_text()
        for i in range(3):
            self.assertIn(f"task{i}", text)
            self.assertIn(f"result{i}", text)

    def test_no_truncation(self):
        long_content = "x" * 500
        mem = ConversationMemory(token_budget=99999)
        mem.append(_msg("user",      "q"))
        mem.append(_msg("assistant", long_content))
        text = mem.format_as_text()
        self.assertIn(long_content, text)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestSerialization(unittest.TestCase):

    def test_empty_to_list(self):
        mem = ConversationMemory()
        self.assertEqual(mem.to_list(), [])

    def test_roundtrip(self):
        mem = ConversationMemory(token_budget=2048, max_entries=50)
        for i in range(3):
            mem.append(_msg("user",      f"task{i}"))
            mem.append(_msg("assistant", f"result{i}"))
        serialized = mem.to_list()
        restored = ConversationMemory.from_list(serialized, token_budget=2048, max_entries=50)
        self.assertEqual(len(restored), len(mem))
        for orig, rest in zip(mem.messages(), restored.messages()):
            self.assertEqual(orig.role, rest.role)
            self.assertEqual(orig.content, rest.content)
        self.assertEqual(restored.token_budget, 2048)
        self.assertEqual(restored.max_entries, 50)

    def test_from_list_defaults(self):
        restored = ConversationMemory.from_list([])
        self.assertEqual(restored.token_budget, ConversationMemory.DEFAULT_TOKEN_BUDGET)
        self.assertEqual(restored.max_entries, ConversationMemory.DEFAULT_MAX_ENTRIES)

    def test_from_list_with_images(self):
        msg = Message(role="user", content="look", images=["base64data"])
        mem = ConversationMemory()
        mem.append(msg)
        restored = ConversationMemory.from_list(mem.to_list())
        self.assertEqual(restored.messages()[0].images, ["base64data"])


# ---------------------------------------------------------------------------
# ChatSession integration
# ---------------------------------------------------------------------------

class TestChatSessionIntegration(unittest.TestCase):

    def _make_session(self, **kw):
        return ChatSession(_mock_llm(), **kw)

    def test_say_appends_pair(self):
        session = self._make_session()
        session.say("hello")
        self.assertEqual(len(session), 2)
        msgs = session.history
        self.assertEqual(msgs[0].role, "user")
        self.assertEqual(msgs[0].content, "hello")
        self.assertEqual(msgs[1].role, "assistant")

    def test_history_passed_to_llm(self):
        llm = _mock_llm()
        session = ChatSession(llm)
        session.say("first")
        session.say("second")
        # Second call should include the first exchange in history
        _, kwargs = llm.chat.call_args_list[1]
        history = kwargs.get("history", [])
        self.assertEqual(len(history), 2)

    def test_token_budget_respected(self):
        # Very small budget → old turns evicted
        llm = _mock_llm(response_content="x" * 40)  # 10 tokens
        session = ChatSession(llm, token_budget=25)
        session.say("a" * 40)   # user: 10 tok, assistant: 10 tok → total 20
        session.say("b" * 40)   # user: 10 tok, assistant: 10 tok → total 40 → trim
        # Only the most recent pair should survive
        self.assertEqual(len(session), 2)

    def test_reset_clears_memory(self):
        session = self._make_session()
        session.say("hi")
        session.reset()
        self.assertEqual(len(session), 0)

    def test_streaming_appends_on_full_consumption(self):
        llm = MagicMock()
        llm.stream.side_effect = lambda *a, **kw: iter(["Hello", " world"])
        session = ChatSession(llm)
        result = "".join(session.say("hi", stream=True))
        self.assertEqual(result, "Hello world")
        self.assertEqual(len(session), 2)

    def test_streaming_appends_on_abandonment(self):
        llm = MagicMock()
        llm.stream.side_effect = lambda *a, **kw: iter(["partial", " more"])
        session = ChatSession(llm)
        gen = session.say("hi", stream=True)
        next(gen)
        gen.close()
        # Partial content recorded
        self.assertEqual(len(session), 2)
        self.assertEqual(session.history[1].content, "partial")


# ---------------------------------------------------------------------------
# Agent stateful-mode integration
# ---------------------------------------------------------------------------

class TestAgentStatefulIntegration(unittest.TestCase):

    def _make_agent(self, **kw):
        from maki.agents import Agent
        llm = _mock_llm()
        return Agent("tester", llm, stateful=True, **kw), llm

    def test_execute_task_populates_memory(self):
        agent, _ = self._make_agent()
        agent.execute_task("do something")
        self.assertEqual(len(agent._conversation_memory), 2)
        msgs = agent._conversation_memory.messages()
        self.assertEqual(msgs[0].role, "user")
        self.assertEqual(msgs[0].content, "do something")
        self.assertEqual(msgs[1].role, "assistant")

    def test_history_injected_into_subsequent_prompt(self):
        agent, llm = self._make_agent()
        captured = []
        llm.chat.side_effect = lambda p, **kw: captured.append(p) or LLMResponse(
            content="ok", model="t", prompt_tokens=1, completion_tokens=1,
            total_tokens=2, elapsed_seconds=0.1,
        )
        agent.execute_task("first task")
        agent.execute_task("second task")
        self.assertIn("Prior conversation", captured[1])
        self.assertIn("first task", captured[1])

    def test_no_300_char_truncation(self):
        long_result = "y" * 500
        agent, llm = self._make_agent()
        llm.chat.return_value = LLMResponse(
            content=long_result, model="t", prompt_tokens=1,
            completion_tokens=1, total_tokens=2, elapsed_seconds=0.1,
        )
        agent.execute_task("a task")
        captured = []
        llm.chat.side_effect = lambda p, **kw: captured.append(p) or LLMResponse(
            content="done", model="t", prompt_tokens=1,
            completion_tokens=1, total_tokens=2, elapsed_seconds=0.1,
        )
        agent.execute_task("follow up")
        self.assertIn(long_result, captured[0])

    def test_reset_conversation_clears_memory(self):
        agent, _ = self._make_agent()
        agent.execute_task("task")
        agent.reset_conversation()
        self.assertEqual(len(agent._conversation_memory), 0)

    def test_non_stateful_agent_does_not_populate_memory(self):
        from maki.agents import Agent
        llm = _mock_llm()
        agent = Agent("non-stateful", llm, stateful=False)
        agent.execute_task("task")
        self.assertEqual(len(agent._conversation_memory), 0)


if __name__ == "__main__":
    unittest.main()
