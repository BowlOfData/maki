"""
Shared stateful multi-turn chat session, usable with any backend that
exposes chat() and stream() with history/config/system kwargs.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, Optional

from .objects import GenerationConfig, LLMResponse, Message

log = logging.getLogger(__name__)


class ChatSession:
    """
    Maintains conversation history across turns.

    Usage
    -----
        session = llm.session(system="You are a helpful chef.")
        session.say("How do I make carbonara?")
        session.say("What if I don't have guanciale?")
        session.print_history()
    """

    def __init__(self, llm: Any, system: Optional[str] = None) -> None:
        self._llm = llm
        self._history: list[Message] = []
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
        response = self._llm.chat(prompt, history=self._history, config=config, system=self._system)
        self._history.append(Message("user", prompt))
        self._history.append(Message("assistant", response.content))
        return response

    def _say_stream(self, prompt: str, config: Optional[GenerationConfig]) -> Iterator[str]:
        """Generator that streams tokens and appends to history when done."""
        full = ""
        for chunk in self._llm.stream(prompt, history=self._history, config=config, system=self._system):
            full += chunk
            yield chunk
        self._history.append(Message("user", prompt))
        self._history.append(Message("assistant", full))

    def reset(self) -> None:
        """Clear conversation history."""
        self._history.clear()
        log.info("Session history cleared.")

    def print_history(self) -> None:
        """Pretty-print the full conversation."""
        for msg in self._history:
            color = "cyan" if msg.role == "user" else "green"
            log.info(f"[bold {color}]{msg.role.upper()}[/bold {color}]")
            log.info(msg.content)
            log.info("")

    @property
    def history(self) -> list[Message]:
        return list(self._history)

    def __len__(self) -> int:
        return len(self._history)
