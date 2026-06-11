"""
Tests for ChatSession, in particular streaming history bookkeeping.
"""

import unittest
from unittest.mock import MagicMock

from maki.session import ChatSession


def _make_llm(chunks=("Hello", " world")):
    llm = MagicMock()
    llm.stream.side_effect = lambda *a, **kw: iter(chunks)
    return llm


class TestChatSessionStreamHistory(unittest.TestCase):

    def test_full_consumption_records_history_once(self):
        session = ChatSession(_make_llm())
        result = "".join(session.say("hi", stream=True))
        self.assertEqual(result, "Hello world")
        self.assertEqual(len(session.history), 2)
        self.assertEqual(session.history[0].role, "user")
        self.assertEqual(session.history[0].content, "hi")
        self.assertEqual(session.history[1].role, "assistant")
        self.assertEqual(session.history[1].content, "Hello world")

    def test_abandoned_stream_records_partial_history(self):
        """Regression §1.12: history was appended only after the generator
        was fully consumed; a consumer that breaks mid-stream left the
        session missing both turns, silently losing context."""
        session = ChatSession(_make_llm())
        gen = session.say("hi", stream=True)
        self.assertEqual(next(gen), "Hello")
        gen.close()  # consumer abandons the stream

        self.assertEqual(len(session.history), 2)
        self.assertEqual(session.history[0].content, "hi")
        self.assertEqual(session.history[1].content, "Hello")

    def test_error_mid_stream_records_partial_history(self):
        def _failing_stream(*a, **kw):
            yield "partial"
            raise RuntimeError("backend died")

        llm = MagicMock()
        llm.stream.side_effect = _failing_stream
        session = ChatSession(llm)

        with self.assertRaises(RuntimeError):
            list(session.say("hi", stream=True))

        self.assertEqual(len(session.history), 2)
        self.assertEqual(session.history[1].content, "partial")

    def test_stream_abandoned_before_first_chunk_records_nothing(self):
        session = ChatSession(_make_llm())
        gen = session.say("hi", stream=True)
        gen.close()  # never consumed a chunk
        self.assertEqual(len(session.history), 0)


if __name__ == "__main__":
    unittest.main()
