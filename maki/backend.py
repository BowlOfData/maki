"""
Abstract base class for all Maki LLM backends.

Every backend (Ollama/generate, Ollama/chat, HuggingFace transformers, …)
must inherit from LLMBackend and implement request().  This gives the
agent layer a single stable type to depend on instead of the concrete Maki
class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .objects import LLMResponse


class LLMBackend(ABC):
    """
    Minimal contract shared by all Maki LLM backends.

    Concrete subclasses must:

    * Set ``self.model`` (str) and ``self.temperature`` (float) during
      ``__init__``.
    * Implement ``request(prompt)`` to accept a plain-text prompt and return
      an :class:`~maki.objects.LLMResponse`.

    Everything else (URL handling, HTTP sessions, tokenisers, rate limiting,
    streaming, …) is backend-specific and lives in the subclass.
    """

    model: str
    temperature: float

    @abstractmethod
    def request(self, prompt: str) -> LLMResponse:
        """Send *prompt* to the model and return a fully resolved response."""
        ...
