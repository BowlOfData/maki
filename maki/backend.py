"""
Abstract base class for all Maki LLM backends.

Every backend (Ollama/generate, Ollama/chat, HuggingFace transformers, …)
must inherit from LLMBackend and implement request().  This gives the
agent layer a single stable type to depend on instead of the concrete Maki
class.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Generator, List, Optional, Tuple

from .objects import GenerationConfig, LLMResponse, Message, ToolCall, expand_tool_traces


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
    * Optionally set ``supports_typed_messages = True`` and implement
      ``to_wire_messages()`` (plus, for exact token counts on tool rounds,
      ``_chat_with_tools_full()``) to take part in the provider-neutral loop:
      the agent then passes typed :class:`~maki.objects.Message` objects to
      ``chat_turn()`` and never sees the provider's message format.

    Everything else (URL handling, HTTP sessions, tokenisers, rate limiting,
    …) is backend-specific and lives in the subclass.
    """

    model: str
    temperature: float
    supports_native_tools: bool = False
    # True when chat()/stream() honour the ``history`` argument; an agent only sends structured
    # history to a backend that says so (the base chat() ignores it).
    supports_history: bool = False
    # True when to_wire_messages()/chat_turn() work: tool calls and results travel as typed Message
    # objects that this backend converts to and from its own API format.
    supports_typed_messages: bool = False

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

    # ------------------------------------------------------------------
    # Typed (provider-neutral) tool turns
    # ------------------------------------------------------------------

    def to_wire_messages(self, messages: List[Message]) -> List[dict]:
        """Convert typed messages (including tool calls and results) to this backend's wire format."""
        raise NotImplementedError(
            f"Backend '{type(self).__name__}' does not support typed messages. "
            "Set supports_typed_messages = True and implement to_wire_messages()."
        )

    def _chat_with_tools_full(
        self,
        messages: List[dict],
        tools: List[dict],
        *,
        system: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        tool_choice: Optional[str] = None,
    ) -> Tuple[LLMResponse, Optional[List[ToolCall]], List[dict]]:
        """Like :meth:`chat_with_tools` but the response is always present, with its token counts.

        The default wraps ``chat_with_tools`` and invents an empty response for a tool-call round
        (that method returns None there). Backends override it to report real usage on every round.
        ``tool_choice="none"`` means the tools are listed (a provider may insist on them when the
        conversation already holds tool calls) but must not be used; the default simply offers none.
        """
        if tool_choice == "none":
            tools = []
        response, calls, updated = self.chat_with_tools(messages, tools, system=system, config=config)
        if response is None:
            response = LLMResponse(content="", model=self.model, prompt_tokens=0, completion_tokens=0,
                                   total_tokens=0, elapsed_seconds=0.0)
        return response, calls, updated

    def chat_turn(
        self,
        messages: List[Message],
        tool_specs: Optional[List[dict]] = None,
        *,
        system: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        tool_choice: Optional[str] = None,
    ) -> LLMResponse:
        """One model round over typed messages, with optional tools.

        *messages* is the whole conversation so far (``user``/``assistant`` text, ``assistant``
        messages with ``tool_calls`` and ``tool`` messages holding their results). *tool_specs* are
        backend-agnostic specs (``name``, ``description``, ``parameters``); None or empty forces a
        plain-text answer; ``tool_choice="none"`` lists them but forbids using them (the agent's final
        round after the tool-round limit, where Anthropic still wants the tools defined). The returned response carries ``tool_calls`` (each with an id, invented
        when the provider has none) when the model wants tools, and ``content`` otherwise.

        Raises:
            NotImplementedError: If the backend has no typed-message support.
        """
        wire = self.to_wire_messages(expand_tool_traces(messages))
        tools = self.to_tool_schemas(tool_specs) if tool_specs else []
        extra = {"tool_choice": tool_choice} if tool_choice else {}
        response, calls, _ = self._chat_with_tools_full(wire, tools, system=system, config=config, **extra)
        if calls:
            for call in calls:
                if not call.id:
                    call.id = f"call_{uuid.uuid4().hex[:12]}"
            response.tool_calls = list(calls)
        return response
