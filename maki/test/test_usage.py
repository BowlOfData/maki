"""Token usage counting on backends and agents."""
import copy
import pickle
import unittest
from unittest.mock import MagicMock

from maki.agents.agent import Agent
from maki.objects import BackendType, LLMResponse
from maki.usage import Usage, attach_usage


def resp(p, c, content="ok"):
    return LLMResponse(content, "m", p, c, p + c, 0.1, backend=BackendType.OLLAMA)


class PicklableBackend:
    """Module-level so pickle can find it; real bound methods like a genuine backend."""
    supports_native_tools = False

    def chat(self, prompt, system=None):
        return resp(2, 1)


class FakeBackend:
    """Real bound methods, like a genuine backend."""
    supports_native_tools = False

    def __init__(self):
        self.replies = []

    def request(self, prompt):
        return self.replies.pop(0)

    def chat(self, prompt, system=None):
        return self.replies.pop(0)

    def chat_collect(self, prompt, system=None):
        return self.replies.pop(0)

    def chat_with_tools(self, messages, tools, system=None):
        return self.replies.pop(0), None, messages


class DelegatingBackend(FakeBackend):
    """Like the base LLMBackend: chat_collect falls back to chat()."""

    def chat_collect(self, prompt, system=None):
        return self.chat(prompt, system=system)


class UsageTests(unittest.TestCase):
    def test_chat_collect_and_tool_calls_are_counted(self):
        b = FakeBackend()
        usage = attach_usage(b)
        b.replies = [resp(10, 5), resp(3, 2), resp(1, 1)]
        b.chat("x")
        b.chat_collect("x")
        b.chat_with_tools([], [])
        self.assertEqual(usage.to_dict(), {"prompt_tokens": 14, "completion_tokens": 8, "total_tokens": 22, "calls": 3})

    def test_cached_tokens_are_summed_and_shown_only_when_reported(self):
        b = FakeBackend()
        usage = attach_usage(b)
        b.replies = [resp(10, 5)]
        b.chat("x")
        self.assertNotIn("cached_tokens", usage.to_dict())
        cached = resp(10, 5)
        cached.cached_tokens = 8
        b.replies = [cached]
        b.chat("x")
        self.assertEqual(usage.to_dict()["cached_tokens"], 8)
        self.assertEqual((usage.snapshot() - Usage()).cached_tokens, 8)

    def test_a_call_that_delegates_to_another_wrapped_method_is_counted_once(self):
        b = DelegatingBackend()
        usage = attach_usage(b)
        b.replies = [resp(10, 5)]
        b.chat_collect("x")
        self.assertEqual((usage.total_tokens, usage.calls), (15, 1))
        b.replies = [resp(1, 1)]
        b.chat("x")  # a later plain call still counts
        self.assertEqual((usage.total_tokens, usage.calls), (17, 2))

    def test_a_counted_backend_can_still_be_copied_and_pickled(self):
        b = PicklableBackend()
        usage = attach_usage(b)
        b.chat("x")
        for clone in (copy.deepcopy(b), pickle.loads(pickle.dumps(b))):
            self.assertEqual(clone.usage.total_tokens, 3)
            clone.chat("x")  # the clone counts on its own usage
            self.assertEqual((clone.usage.total_tokens, usage.total_tokens), (6, 3))

    def test_thread_tokens_separate_concurrent_users_of_one_backend(self):
        import threading
        b = FakeBackend()
        usage = attach_usage(b)
        done = threading.Event()

        def other():
            b.replies = [resp(100, 100)]
            b.chat("x")
            done.set()

        t = threading.Thread(target=other)
        t.start()
        done.wait(5)
        t.join()
        b.replies = [resp(1, 2)]
        b.chat("x")
        self.assertEqual(usage.total_tokens, 203)
        self.assertEqual(usage.thread_tokens(), 3)  # this thread's calls only

    def test_attach_is_idempotent(self):
        b = FakeBackend()
        first = attach_usage(b)
        self.assertIs(attach_usage(b), first)
        b.replies = [resp(1, 1)]
        b.chat("x")
        self.assertEqual(first.calls, 1)  # not double counted

    def test_snapshot_difference(self):
        b = FakeBackend()
        usage = attach_usage(b)
        b.replies = [resp(1, 1), resp(4, 4)]
        b.chat("x")
        before = usage.snapshot()
        b.chat("x")
        self.assertEqual((usage - before).total_tokens, 8)

    def test_mocks_are_left_alone(self):
        m = MagicMock()
        attach_usage(m)
        m.chat("x")
        m.chat.assert_called_once_with("x")

    def test_agent_usage_counts_its_backend(self):
        b = FakeBackend()
        b.replies = [resp(7, 3, "answer")]
        agent = Agent("a", b)
        self.assertEqual(agent.execute_task("do it"), "answer")
        self.assertEqual((agent.usage.total_tokens, agent.usage.calls), (10, 1))
        self.assertIsInstance(agent.usage, Usage)


if __name__ == "__main__":
    unittest.main()
