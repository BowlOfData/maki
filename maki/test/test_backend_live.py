"""
Live checks of the backend contract against a real Ollama server.

The conformance suite proves the wrappers keep the contract against fakes; this file proves the
real wire format still matches. Deselected by default (``-m "not network"`` in pyproject); run with

    pytest -m network maki/test/test_backend_live.py

Configure with MAKI_LIVE_OLLAMA_MODEL (default gemma4:26b) and MAKI_LIVE_OLLAMA_URL (default
http://127.0.0.1:11434). The tests skip when the server or the model is not available.
"""
import asyncio
import json
import os
import unittest
import urllib.request

import pytest

from maki.agents import Agent
from maki.exceptions import MakiAPIError, MakiNetworkError
from maki.makiLLama import MakiLLama
from maki.objects import GenerationConfig, LLMResponse, Message
from maki.usage import attach_usage

pytestmark = pytest.mark.network

MODEL = os.environ.get("MAKI_LIVE_OLLAMA_MODEL", "gemma4:26b")
URL = os.environ.get("MAKI_LIVE_OLLAMA_URL", "http://127.0.0.1:11434")
GREEDY = GenerationConfig(temperature=0, max_tokens=60)


def _available() -> bool:
    try:
        with urllib.request.urlopen(f"{URL}/api/tags", timeout=3) as r:
            names = [m["name"] for m in json.load(r).get("models", [])]
    except Exception:
        return False
    return MODEL in names


class TestLiveOllama(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Checked here, not in a decorator, so merely collecting this module never touches the network.
        if not _available():
            raise unittest.SkipTest(f"Ollama with {MODEL} not reachable at {URL}")
        cls.llm = MakiLLama(model=MODEL, base_url=URL, think=False, timeout=300)
        cls.usage = attach_usage(cls.llm)

    def test_chat_returns_a_normalised_response(self):
        r = self.llm.chat("Reply with exactly the single word: PONG", config=GREEDY)
        self.assertIsInstance(r, LLMResponse)
        self.assertIn("pong", r.content.lower())
        self.assertGreater(r.prompt_tokens, 0)
        self.assertGreater(r.completion_tokens, 0)
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

    def test_chat_collect_matches_chat(self):
        r = self.llm.chat_collect("Reply with exactly the single word: PONG", config=GREEDY)
        self.assertIn("pong", r.content.lower())

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
        with self.assertRaises(MakiAPIError) as caught:
            MakiLLama(model="no-such-model:0b", base_url=URL).chat("hi")
        self.assertEqual(caught.exception.status_code, 404)

    def test_unreachable_server_is_a_network_error(self):
        with self.assertRaises(MakiNetworkError):
            MakiLLama(model=MODEL, base_url="http://127.0.0.1:9", timeout=2).chat("hi")


if __name__ == "__main__":
    unittest.main()
