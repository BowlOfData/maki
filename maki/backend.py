"""
Abstract base class for all Maki LLM backends.

Every backend (Ollama/generate, Ollama/chat, HuggingFace transformers, …)
must inherit from LLMBackend and implement request().  This gives the
agent layer a single stable type to depend on instead of the concrete Maki
class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generator, List, Optional

from .objects import GenerationConfig, LLMResponse, Message


class LLMBackend(ABC):
    """
    Minimal contract shared by all Maki LLM backends.

    Concrete subclasses must:

    * Set ``self.model`` (str) and ``self.temperature`` (float) during
      ``__init__``.
    * Implement ``request(prompt)`` to accept a plain-text prompt and return
      an :class:`~maki.objects.LLMResponse`.
    * Optionally override ``chat()``, ``stream()``, and ``chat_collect()``
      to support multi-turn conversations, token streaming, and
      streaming-with-collect respectively.

    Everything else (URL handling, HTTP sessions, tokenisers, rate limiting,
    …) is backend-specific and lives in the subclass.
    """

    model: str
    temperature: float

    @abstractmethod
    def request(self, prompt: str) -> LLMResponse:
        """Send *prompt* to the model and return a fully resolved response."""
        ...

    def chat(
        self,
        prompt: str,
        history: Optional[List[Message]] = None,
        config: Optional[GenerationConfig] = None,
        system: Optional[str] = None,
        images: Optional[List[str]] = None,
    ) -> LLMResponse:
        """Single-turn (or multi-turn with explicit *history*) generation.

        The default implementation ignores all kwargs and delegates to
        ``request(prompt)``.  Backends with native chat APIs override this.
        """
        return self.request(prompt)

    def stream(
        self,
        prompt: str,
        history: Optional[List[Message]] = None,
        config: Optional[GenerationConfig] = None,
        system: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Stream response tokens for *prompt*.

        Backends that support streaming must override this method.
        The default implementation raises ``NotImplementedError``.

        Yields:
            str: successive text chunks as they are produced by the model.

        Raises:
            NotImplementedError: If this backend does not support streaming.
        """
        raise NotImplementedError(
            f"Backend '{type(self).__name__}' does not support streaming. "
            "Use MakiLLama or another streaming-capable backend instead."
        )

    def chat_collect(
        self,
        prompt: str,
        history: Optional[List[Message]] = None,
        config: Optional[GenerationConfig] = None,
        system: Optional[str] = None,
        images: Optional[List[str]] = None,
    ) -> LLMResponse:
        """Like ``chat()`` but uses HTTP streaming internally when possible.

        The per-chunk timeout applies instead of a single read timeout, so
        long-running generations complete without hitting the global timeout.
        The default implementation falls back to ``chat()``.  Backends with
        native streaming (e.g. MakiLLama) override this.
        """
        return self.chat(prompt, history=history, config=config,
                         system=system, images=images)
