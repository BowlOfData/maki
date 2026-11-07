"""
Built-in evaluation suites for :func:`maki.eval.run_eval`.

    basic          instruction following, extraction, JSON output (fast)
    tool_calling   typed tool calls, multi-step use, and not calling a tool that is not needed
    long_context   recall of a fact buried in a long prompt (shows a too-small context window)
    memory         recall after many turns of a small conversation budget, with and without summaries (slow)

Each is a function returning an :class:`~maki.eval.EvalSuite`; ``get_suite(name)`` looks one up.
"""

from __future__ import annotations

import json
from typing import Callable, Dict, List

from .agents.agent import Agent
from .backend import LLMBackend
from .eval import EvalCase, EvalSuite
from .objects import MemoryPolicy


def basic() -> EvalSuite:
    def json_city(answer: str) -> bool:
        try:
            start, end = answer.index("{"), answer.rindex("}") + 1
            return str(json.loads(answer[start:end]).get("city", "")).lower() == "paris"
        except (ValueError, AttributeError):
            return False

    cases = [
        EvalCase("exact_word", "Reply with exactly the single word: PONG",
                 check=lambda a: ("pong" in a.lower(), "") if len(a.split()) <= 3 else (False, "too wordy")),
        EvalCase("arithmetic", "What is 17 + 25? Reply with the number only.", contains="42"),
        EvalCase("extraction", "From the text 'Order #A-1042 was shipped on 3 March to Lena Ortiz.' give the "
                               "order number only.", contains="A-1042"),
        EvalCase("json_output", "Reply with only a JSON object with the key \"city\" set to the city where "
                                "the Eiffel Tower stands.", check=json_city),
        EvalCase("follows_format", "List three primary colors, comma-separated, lowercase, no other text.",
                 check=lambda a: (len([p for p in a.split(",") if p.strip()]) == 3, "expected three items")),
    ]
    return EvalSuite("basic", cases, description="instruction following, extraction and JSON output")


class _Calculator:
    """Typed arithmetic tools that remember how they were called."""
    ALLOWED_METHODS = ["add", "multiply"]
    DANGEROUS_METHODS = None

    def __init__(self) -> None:
        self.calls: List[tuple] = []

    def add(self, a: int, b: int) -> str:
        """Add two integers.

        Args:
            a: first number
            b: second number
        """
        self.calls.append(("add", a, b))
        return str(a + b)

    def multiply(self, a: int, b: int) -> str:
        """Multiply two integers.

        Args:
            a: first number
            b: second number
        """
        self.calls.append(("multiply", a, b))
        return str(a * b)


def tool_calling() -> EvalSuite:
    def make_agent(backend: LLMBackend) -> Agent:
        agent = Agent("eval", backend, instructions="Use the provided tools for any arithmetic.")
        agent.plugins["calc"] = _Calculator()
        return agent

    def calls_of(agent: Agent) -> List[tuple]:
        return agent.plugins["calc"].calls

    cases = [
        EvalCase("single_call", "What is 17 plus 25? Use the tool.", use_plugins=True,
                 check=lambda a, agent: ("42" in a and ("add", 17, 25) in calls_of(agent),
                                         f"answer {a[:40]!r}, calls {calls_of(agent)}")),
        EvalCase("two_steps", "Compute (6 plus 4) times 7 using the tools, one step at a time.", use_plugins=True,
                 check=lambda a, agent: ("70" in a and len(calls_of(agent)) >= 2,
                                         f"answer {a[:40]!r}, calls {calls_of(agent)}")),
        EvalCase("no_tool_needed", "What is the capital of France? One word.", use_plugins=True,
                 check=lambda a, agent: ("paris" in a.lower() and not calls_of(agent),
                                         f"answer {a[:40]!r}, calls {calls_of(agent)}")),
    ]
    return EvalSuite("tool_calling", cases, make_agent, "typed tool calls, multi-step use, restraint")


_FILLER = "The quick brown fox jumps over the lazy dog. "  # about 11 tokens


def long_context(tokens: int = 3000) -> EvalSuite:
    """Recall of a code word placed at the start or the end of a prompt of about *tokens* tokens."""
    filler = _FILLER * max(1, tokens * 4 // len(_FILLER))
    question = "What was the code word? Answer with the code word only."
    cases = [
        EvalCase("word_at_start", f"Remember this code word: ZEBRA-4471.\n\n{filler}\n\n{question}",
                 contains="ZEBRA-4471"),
        EvalCase("word_at_end", f"{filler}\n\nRemember this code word: QUOKKA-9082.\n\n{question}",
                 contains="QUOKKA-9082"),
    ]
    return EvalSuite("long_context", cases, description=f"recall in a prompt of about {tokens} tokens")


_FACTS = ("The project codename is Bluefin.", "The launch deadline is March 14.", "The project owner is Priya Raman.")
_FILLER_TURNS = ("Give me one tip for writing clear emails.", "Name a good habit for staying focused.",
                 "Suggest a healthy breakfast idea.", "What is one way to learn a language faster?",
                 "Give me a short tip for better sleep.", "Suggest a simple weekend hobby.")


def memory(token_budget: int = 300) -> EvalSuite:
    turns = [f"Remember this for later: {fact}" for fact in _FACTS] + list(_FILLER_TURNS)
    question = ("Without asking me anything, list the project codename, the launch deadline and the project "
                "owner, in one line.")

    def make(summarize: bool) -> Callable[[LLMBackend], Agent]:
        def build(backend: LLMBackend) -> Agent:
            policy = MemoryPolicy(history_mode="messages", token_budget=token_budget, summarize_older=summarize,
                                  summary_chars=400 if summarize else None)
            return Agent("eval", backend, stateful=True, memory=policy,
                         instructions="Answer in at most two short sentences.")
        return build

    needles = ("bluefin", "march 14", "priya")
    cases = [
        EvalCase("recall_with_summary", question, setup_turns=turns, make_agent=make(True),
                 check=lambda a: (all(n in a.lower() for n in needles), "lost an early fact")),
        EvalCase("recall_without_summary", question, setup_turns=turns, make_agent=make(False),
                 check=lambda a: (all(n in a.lower() for n in needles), "lost an early fact")),
    ]
    return EvalSuite("memory", cases, description="recall after many turns on a small budget (slow: 10 calls per run)")


SUITES: Dict[str, Callable[[], EvalSuite]] = {
    "basic": basic, "tool_calling": tool_calling, "long_context": long_context, "memory": memory,
}


def get_suite(name: str, **options) -> EvalSuite:
    """The built-in suite called *name* (``options`` go to its builder, e.g. ``tokens=8000``)."""
    try:
        builder = SUITES[name]
    except KeyError:
        raise KeyError(f"unknown suite '{name}'; available: {', '.join(SUITES)}") from None
    return builder(**options)
