"""
Shared stateful multi-turn chat session, usable with any backend that
exposes chat() and stream() with history/config/system kwargs.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, Optional

from .objects import ConversationMemory, GenerationConfig, LLMResponse, Message

log = logging.getLogger(__name__)


class ChatSession:
    """
    Maintains token-budgeted conversation history across turns.

    Usage
    -----
        session = llm.session(system="You are a helpful chef.")
        session.say("How do I make carbonara?")
        session.say("What if I don't have guanciale?")
        session.print_history()
    """

    def __init__(
        self,
        llm: Any,
        system: Optional[str] = None,
        token_budget: int = ConversationMemory.DEFAULT_TOKEN_BUDGET,
    ) -> None:
        self._llm = llm
        self._memory = ConversationMemory(token_budget=token_budget)
        self._system = system

    def say(
        self,
        prompt: str,
        stream: bool = False,
        config: Optional[GenerationConfig] = None,
    ) -> LLMResponse | Iterator[str]:
        """Send a message and automatically extend the conversation history."""
        log.debug("Session saying: %s", prompt[:100] + "..." if len(prompt) > 100 else prompt)
        if stream:
            return self._say_stream(prompt, config)
        history_snapshot = self._memory.messages()
        response = self._llm.chat(prompt, history=history_snapshot, config=config, system=self._system)
        self._memory.append(Message("user", prompt))
        self._memory.append(Message("assistant", response.content))
        return response

    def _say_stream(self, prompt: str, config: Optional[GenerationConfig]) -> Iterator[str]:
        """Generator that streams tokens and appends to history when done."""
        full = ""
        history_snapshot = self._memory.messages()
        try:
            for chunk in self._llm.stream(prompt, history=history_snapshot, config=config, system=self._system):
                full += chunk
                yield chunk
        finally:
            # Always record whatever was produced, even if the consumer
            # abandons the stream mid-way — otherwise both turns vanish
            # from history and later turns silently lose context.
            if full:
                self._memory.append(Message("user", prompt))
                self._memory.append(Message("assistant", full))

    def reset(self) -> None:
        """Clear conversation history."""
        self._memory.clear()
        log.info("Session history cleared.")

    def print_history(self) -> None:
        """Pretty-print the full conversation."""
        for msg in self._memory.messages():
            color = "cyan" if msg.role == "user" else "green"
            print(f"\033[1m{msg.role.upper()}\033[0m")
            print(msg.content)
            print()

    @property
    def history(self) -> list:
        return self._memory.messages()

    def __len__(self) -> int:
        return len(self._memory)
