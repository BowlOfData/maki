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

from typing import Tuple

from .objects import GenerationConfig, LLMResponse, Message, ToolCall


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
    * Set ``supports_native_tools = True`` and override
      ``to_tool_schemas()``, ``chat_with_tools()``, and
      ``append_tool_results()`` to participate in the native tool-calling
      loop driven by :class:`~maki.agents.plugin_handler.PluginHandler`.

    Everything else (URL handling, HTTP sessions, tokenisers, rate limiting,
    …) is backend-specific and lives in the subclass.
    """

    model: str
    temperature: float
    supports_native_tools: bool = False

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

    def to_tool_schemas(self, tool_specs: List[dict]) -> List[dict]:
        """Translate backend-agnostic tool specs to this backend's wire format.

        Each element of *tool_specs* is a dict with keys ``name``,
        ``description``, and ``parameters`` (JSON Schema object).
        Backends that support native tool calling must override this.
        """
        raise NotImplementedError(
            f"Backend '{type(self).__name__}' does not support native tool calling. "
            "Set supports_native_tools = True and override to_tool_schemas(), "
            "chat_with_tools(), and append_tool_results()."
        )

    def chat_with_tools(
        self,
        messages: List[dict],
        tools: List[dict],
        *,
        system: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> Tuple[Optional[LLMResponse], Optional[List[ToolCall]], List[dict]]:
        """One round of tool-aware generation.

        Returns ``(response, None, messages)`` when the model emits a text
        reply with no tool calls, and ``(None, tool_calls, messages)`` when
        it requests one or more tool invocations.  The returned *messages*
        list always includes the assistant turn produced in this call.

        An empty *tools* list forces a plain-text response (useful for the
        final round after the maximum iteration count is reached).
        """
        raise NotImplementedError(
            f"Backend '{type(self).__name__}' does not support native tool calling."
        )

    def append_tool_results(
        self,
        messages: List[dict],
        results: List[Tuple[ToolCall, str]],
    ) -> List[dict]:
        """Append tool results to *messages* in backend-specific format.

        *results* is a list of ``(tool_call, result_string)`` pairs produced
        by :meth:`~maki.agents.plugin_handler.PluginHandler._execute_tool_call`.
        Returns the updated messages list.
        """
        raise NotImplementedError(
            f"Backend '{type(self).__name__}' does not support native tool calling."
        )
