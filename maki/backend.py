"""
Abstract base class for all Maki LLM backends.

Every backend (Ollama/generate, Ollama/chat, HuggingFace transformers, …)
must inherit from LLMBackend and implement request().  This gives the
agent layer a single stable type to depend on instead of the concrete Maki
class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generator

from .objects import LLMResponse


class LLMBackend(ABC):
    """
    Minimal contract shared by all Maki LLM backends.

    Concrete subclasses must:

    * Set ``self.model`` (str) and ``self.temperature`` (float) during
      ``__init__``.
    * Implement ``request(prompt)`` to accept a plain-text prompt and return
      an :class:`~maki.objects.LLMResponse`.
    * Optionally override ``stream(prompt)`` to support token streaming.

    Everything else (URL handling, HTTP sessions, tokenisers, rate limiting,
    …) is backend-specific and lives in the subclass.
    """

    model: str
    temperature: float

    @abstractmethod
    def request(self, prompt: str) -> LLMResponse:
        """Send *prompt* to the model and return a fully resolved response."""
        ...

    def stream(self, prompt: str) -> Generator[str, None, None]:
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
