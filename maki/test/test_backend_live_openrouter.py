"""
Live checks of the backend contract against OpenRouter (the OpenAI-compatible SDK path).

Deselected by default (``-m "not network"``); run with a key in the environment:

    OPENROUTER_API_KEY=sk-or-... pytest -m network maki/test/test_backend_live_openrouter.py

MAKI_LIVE_OPENROUTER_MODEL picks the model (default ``openai/gpt-4o-mini``, cheap and tool-capable).
The tests skip without a key or without the ``openai`` SDK. Each call is a few tokens.
"""
import asyncio
import importlib.util
import os
import unittest

import pytest

from maki import FallbackBackend
from maki.agents import Agent
from maki.exceptions import MakiAPIError, MakiNetworkError
from maki.objects import GenerationConfig, LLMResponse, Message
from maki.usage import attach_usage

pytestmark = pytest.mark.network

MODEL = os.environ.get("MAKI_LIVE_OPENROUTER_MODEL", "openai/gpt-4o-mini")
GREEDY = GenerationConfig(temperature=0, max_tokens=60)


class TestLiveOpenRouter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise unittest.SkipTest("OPENROUTER_API_KEY is not set")
        if importlib.util.find_spec("openai") is None:
            raise unittest.SkipTest("the openai SDK is not installed (pip install maki[openrouter])")
        from maki.makiOpenRouter import MakiOpenRouter
        cls.llm = MakiOpenRouter(model=MODEL, timeout=60)
        cls.usage = attach_usage(cls.llm)

    def test_chat_returns_a_normalised_response(self):
        r = self.llm.chat("Reply with exactly the single word: PONG", config=GREEDY)
        self.assertIsInstance(r, LLMResponse)
        self.assertIn("pong", r.content.lower())
        self.assertGreater(r.prompt_tokens, 0)
        self.assertEqual(r.total_tokens, r.prompt_tokens + r.completion_tokens)

    def test_system_and_history_reach_the_model(self):
        history = [Message("user", "My name is Zorblax."), Message("assistant", "Nice to meet you, Zorblax.")]
        r = self.llm.chat("What is my name?", history=history, system="Answer in one short sentence.",
                          config=GREEDY)
        self.assertIn("zorblax", r.content.lower())

    def test_stream_yields_chunks_that_add_up(self):
        chunks = list(self.llm.stream("Count from 1 to 5, digits only, separated by spaces.", config=GREEDY))
        text = "".join(chunks)
        self.assertGreater(len(chunks), 1)
        for digit in "12345":
            self.assertIn(digit, text)

    def test_async_chat(self):
        r = asyncio.run(self.llm.async_chat("Reply with exactly the single word: PONG", config=GREEDY))
        self.assertIn("pong", r.content.lower())

    def test_token_usage_is_counted(self):
        before = self.usage.calls
        self.llm.chat("Say hi.", config=GREEDY)
        self.assertEqual(self.usage.calls, before + 1)

    def test_native_tool_round_trip(self):
        tools = self.llm.to_tool_schemas([{
            "name": "math__add", "description": "Add two integers.",
            "parameters": {"type": "object", "required": ["a", "b"],
                           "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}}])
        messages = [{"role": "user", "content": "Use the tool to add 17 and 25."}]
        response, calls, messages = self.llm.chat_with_tools(
            messages, tools, system="Always use the provided tool for arithmetic.")
        self.assertIsNone(response)
        self.assertEqual((calls[0].name, calls[0].args), ("math__add", {"a": 17, "b": 25}))
        messages = self.llm.append_tool_results(messages, [(calls[0], "42")])
        final, more, _ = self.llm.chat_with_tools(messages, tools)
        self.assertIsNone(more)
        self.assertIn("42", final.content)

    def test_agent_runs_a_typed_plugin_end_to_end(self):
        class Calculator:
            ALLOWED_METHODS = ["add"]
            DANGEROUS_METHODS = None
            calls = []

            def add(self, a: int, b: int) -> str:
                """Add two integers.

                Args:
                    a: first number
                    b: second number
                """
                self.calls.append((a, b))
                return str(a + b)

        agent = Agent("calc", self.llm, role="calculator",
                      instructions="Always use the provided tool for arithmetic.")
        calculator = Calculator()
        agent.plugins["math"] = calculator
        result = agent.execute_task("What is 17 plus 25? Use the tool.", use_plugins=True)
        self.assertEqual(calculator.calls, [(17, 25)], "the model must call the tool with typed ints")
        self.assertIn("42", result)

    def test_unknown_model_is_an_api_error(self):
        from maki.makiOpenRouter import MakiOpenRouter
        with self.assertRaises(MakiAPIError) as caught:
            MakiOpenRouter(model="no-such-vendor/no-such-model", timeout=60).chat("hi")
        self.assertIn(caught.exception.status_code, (400, 404))

    def test_fallback_from_a_dead_backend_to_openrouter(self):
        from maki.makiLLama import MakiLLama
        from unittest.mock import patch
        with patch.object(MakiLLama, "_verify_connection"):
            dead = MakiLLama(model="x", base_url="http://127.0.0.1:9", timeout=2)
        fb = FallbackBackend([dead, self.llm])
        self.assertIn("pong", fb.chat("Reply with exactly the single word: PONG", config=GREEDY).content.lower())
        self.assertIs(fb.last_used, self.llm)
        with self.assertRaises(MakiNetworkError):
            FallbackBackend([dead]).chat("hi")


if __name__ == "__main__":
    unittest.main()
