"""
Offline, deterministic benchmark for agent memory.

A recording backend stands in for the model, so nothing here needs a server or a key. Token counts
use the same ``len // 4`` estimate as ``ConversationMemory``. Each runner takes extra ``Agent``
keyword arguments, so the same scenario can be run with and without a memory policy::

    python -m maki.test.memory_bench
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from maki.agents import Agent
from maki.objects import LLMResponse, ToolCall

FACT = "The project codename is KESTREL-7."
FILLER_TASK = "Describe aspect {n} of the migration in detail."


def estimate(text: str) -> int:
    return max(1, len(text) // 4)


class RecordingBackend:
    """Replies with fixed-size text and records everything it is sent.

    Accepts structured ``history`` so a memory policy that sends it can be measured too.
    """

    model = "bench"
    temperature = 0.0
    supports_native_tools = False
    supports_history = True

    def __init__(self, answer_chars: int = 1200):
        self.answer = "x" * answer_chars
        self.calls: List[Dict[str, Any]] = []

    def request(self, prompt: str) -> LLMResponse:
        return self.chat(prompt)

    def chat(self, prompt, history=None, config=None, system=None, images=None) -> LLMResponse:
        self.calls.append({"prompt": prompt, "history": list(history or []), "system": system or ""})
        tokens = self.call_tokens(self.calls[-1])
        return LLMResponse(self.answer, self.model, tokens, estimate(self.answer),
                           tokens + estimate(self.answer), 0.0)

    chat_collect = chat

    @staticmethod
    def call_tokens(call: Dict[str, Any]) -> int:
        return (estimate(call["system"]) + estimate(call["prompt"])
                + sum(estimate(m.content) for m in call["history"]))

    @staticmethod
    def call_text(call: Dict[str, Any]) -> str:
        return " ".join([call["system"], call["prompt"]] + [m.content for m in call["history"]])


def run_stateful(turns: int = 20, answer_chars: int = 1200, **agent_kwargs) -> List[int]:
    """Prompt tokens sent on each turn of a stateful conversation."""
    backend = RecordingBackend(answer_chars)
    agent = Agent("Bench", backend, "assistant", "Be brief.", stateful=True, **agent_kwargs)
    for n in range(1, turns + 1):
        agent.execute_task(FILLER_TASK.format(n=n))
    return [backend.call_tokens(c) for c in backend.calls]


def recall_at(turn: int = 15, answer_chars: int = 1200, **agent_kwargs) -> bool:
    """True when a fact stated on turn 1 is still visible to the model on ``turn``."""
    backend = RecordingBackend(answer_chars)
    agent = Agent("Bench", backend, "assistant", "Be brief.", stateful=True, **agent_kwargs)
    agent.execute_task(f"Remember this. {FACT}")
    for n in range(2, turn):
        agent.execute_task(FILLER_TASK.format(n=n))
    agent.execute_task("What is the project codename?")
    return FACT in RecordingBackend.call_text(backend.calls[-1])


class _ToolBackend:
    """Native tool-calling stand-in: asks for the tool ``rounds`` times, then answers."""

    model = "bench"
    temperature = 0.0
    supports_native_tools = True
    supports_history = True

    def __init__(self, rounds: int):
        self.rounds = rounds
        self.sent_chars: List[int] = []
        self._asked = 0

    def request(self, prompt):
        raise NotImplementedError

    def to_tool_schemas(self, specs):
        return specs

    def chat_with_tools(self, messages, tools, system=None, config=None):
        self.sent_chars.append(len(json.dumps(messages)))
        if self._asked < self.rounds and tools:
            self._asked += 1
            call = ToolCall(id=str(self._asked), name="bulk__fetch", args={})
            return None, [call], messages
        return LLMResponse("done", self.model, 0, 0, 0, 0.0), None, messages

    def append_tool_results(self, messages, results):
        return list(messages) + [{"role": "tool", "content": out} for _, out in results]


class _BulkPlugin:
    ALLOWED_METHODS = ["fetch"]
    DANGEROUS_METHODS = None

    def __init__(self, chars: int):
        self.chars = chars

    def fetch(self) -> str:
        return "y" * self.chars


def run_tool_heavy(rounds: int = 4, result_chars: int = 20000, **agent_kwargs) -> List[int]:
    """Estimated tokens sent to the model on each round of a task whose tool returns a large result."""
    backend = _ToolBackend(rounds)
    agent = Agent("Bench", backend, "assistant", "Be brief.", **agent_kwargs)
    agent.plugins["bulk"] = _BulkPlugin(result_chars)
    agent.execute_task("Fetch the data.", use_plugins=True)
    return [c // 4 for c in backend.sent_chars]


def history_footprint(turns: int = 50, answer_chars: int = 20000, **agent_kwargs) -> int:
    """Characters held by ``task_history`` after ``turns`` large answers."""
    backend = RecordingBackend(answer_chars)
    agent = Agent("Bench", backend, "assistant", "Be brief.", **agent_kwargs)
    for n in range(1, turns + 1):
        agent.execute_task(FILLER_TASK.format(n=n))
    return sum(len(str(e["result"])) + len(str(e["context"])) for e in agent.task_history)


def run_all(**agent_kwargs) -> Dict[str, Any]:
    stateful = run_stateful(**agent_kwargs)
    tool = run_tool_heavy(**agent_kwargs)
    return {
        "stateful_prompt_tokens_total": sum(stateful),
        "stateful_prompt_tokens_last_turn": stateful[-1],
        "recall_turn_15": recall_at(15, **agent_kwargs),
        "tool_heavy_tokens_last_round": tool[-1],
        "task_history_chars": history_footprint(**agent_kwargs),
    }


if __name__ == "__main__":
    print(json.dumps(run_all(), indent=2))
