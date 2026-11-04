"""
MakiAnthropic — Anthropic API backend for the Maki framework.

Requirements:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
"""

from __future__ import annotations

import logging
import os
import time
from typing import Generator, Optional

from .backend import LLMBackend
from .config import (
    ANTHROPIC_API_KEY_ENV,
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_REQUEST_TIMEOUT,
)
from .exceptions import MakiNetworkError, MakiTimeoutError, error_for_status, retry_after_of
from .objects import BackendType, GenerationConfig, LLMResponse, Message, RateLimiter, ToolCall
from .session import ChatSession

try:
    import anthropic as _anthropic_sdk
except ImportError:
    _anthropic_sdk = None  # type: ignore

log = logging.getLogger(__name__)


class MakiAnthropic(LLMBackend):
    """
    Anthropic API backend (Messages API).

    Supports native tool calling via the Anthropic tool-use API.

    Usage
    -----
        llm = MakiAnthropic(model="claude-sonnet-4-6")
        response = llm.chat("What is the capital of France?")
        print(response.content)

        for chunk in llm.stream("Tell me a joke"):
            print(chunk, end="", flush=True)

        session = llm.session(system="You are a senior Python engineer.")
        session.say("Explain list comprehensions.")

    Note: the Anthropic API accepts system prompts as a top-level parameter,
    not as a message in the messages list.
    """
    supports_native_tools: bool = True
    supports_history: bool = True

    def __init__(
        self,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        api_key: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
        timeout: int = DEFAULT_REQUEST_TIMEOUT,
        rate_limit: Optional[int] = None,
    ) -> None:
        if _anthropic_sdk is None:
            raise ImportError("anthropic package is required: pip install anthropic")

        resolved_key = api_key or os.environ.get(ANTHROPIC_API_KEY_ENV)
        if not resolved_key:
            raise ValueError(
                f"Anthropic API key not found. Pass api_key= or set {ANTHROPIC_API_KEY_ENV}."
            )

        self.model = model
        self.config = config or GenerationConfig()
        self.temperature = self.config.temperature
        self.system_prompt = system_prompt
        self.timeout = timeout
        self._rate_limiter = RateLimiter(rate_limit) if rate_limit is not None else None
        self._client = _anthropic_sdk.Anthropic(api_key=resolved_key, timeout=timeout)
        self._async_client = _anthropic_sdk.AsyncAnthropic(api_key=resolved_key, timeout=timeout)
        log.info("MakiAnthropic · model=%s", model)

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        prompt: str,
        history: Optional[list[Message]] = None,
        images: Optional[list[str]] = None,
    ) -> list[dict]:
        """Build messages list. System prompt is handled separately at the call site."""
        msgs: list[dict] = []
        if history:
            for m in history:
                # Anthropic only accepts "user" / "assistant" roles in messages;
                # system messages live in the top-level system parameter.
                if m.role not in ("user", "assistant"):
                    continue
                if m.images and m.role == "user":
                    content: list[dict] = [{"type": "text", "text": m.content}]
                    for b64 in m.images:
                        content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        })
                    msgs.append({"role": m.role, "content": content})
                else:
                    msgs.append({"role": m.role, "content": m.content})
        if images:
            content: list[dict] = [{"type": "text", "text": prompt}]
            for b64 in images:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64,
                    },
                })
            msgs.append({"role": "user", "content": content})
        else:
            msgs.append({"role": "user", "content": prompt})
        return msgs

    def _effective_system(self, system: Optional[str]) -> Optional[str]:
        return system if system is not None else self.system_prompt

    def _parse_response(self, response: object, elapsed: float) -> LLMResponse:
        usage = response.usage  # type: ignore[attr-defined]
        # Text of the text blocks; a tool round may open with a non-text block (tool_use, thinking).
        content = "".join(
            b.text for b in (response.content or [])  # type: ignore[attr-defined]
            if isinstance(getattr(b, "text", None), str)
            and not (isinstance(getattr(b, "type", None), str) and b.type != "text")
        )
        cached = getattr(usage, "cache_read_input_tokens", 0)
        return LLMResponse(
            content=content,
            model=response.model,  # type: ignore[attr-defined]
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
            elapsed_seconds=elapsed,
            done=True,
            backend=BackendType.ANTHROPIC,
            cached_tokens=cached if isinstance(cached, int) else 0,
        )

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def chat(
        self,
        prompt: str,
        history: Optional[list[Message]] = None,
        config: Optional[GenerationConfig] = None,
        system: Optional[str] = None,
        images: Optional[list[str]] = None,
    ) -> LLMResponse:
        """Single-turn (or multi-turn with explicit history) generation."""
        log.debug("chat: %s", prompt[:100])
        if self._rate_limiter:
            self._rate_limiter.acquire()
        cfg = config or self.config
        messages = self._build_messages(prompt, history, images=images)
        kwargs = cfg.to_anthropic_kwargs()
        effective_system = self._effective_system(system)
        if effective_system:
            kwargs["system"] = effective_system
        t0 = time.perf_counter()
        try:
            response = self._client.messages.create(
                model=self.model,
                messages=messages,
                **kwargs,
            )
        except _anthropic_sdk.APITimeoutError as e:
            raise MakiTimeoutError(f"chat() timed out: {e}") from e
        except _anthropic_sdk.APIConnectionError as e:
            raise MakiNetworkError(f"chat() connection failed: {e}") from e
        except _anthropic_sdk.APIStatusError as e:
            raise error_for_status(e.status_code, f"chat() HTTP error {e.status_code}: {e}",
                                   retry_after=retry_after_of(e)) from e
        elapsed = time.perf_counter() - t0
        result = self._parse_response(response, elapsed)
        log.info("chat: %.2fs, %d tokens", elapsed, result.total_tokens)
        return result

    def stream(
        self,
        prompt: str,
        history: Optional[list[Message]] = None,
        config: Optional[GenerationConfig] = None,
        system: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Stream response tokens for *prompt*."""
        log.debug("stream: %s", prompt[:100])
        if self._rate_limiter:
            self._rate_limiter.acquire()
        cfg = config or self.config
        messages = self._build_messages(prompt, history)
        kwargs = cfg.to_anthropic_kwargs()
        effective_system = self._effective_system(system)
        if effective_system:
            kwargs["system"] = effective_system
        try:
            with self._client.messages.stream(
                model=self.model,
                messages=messages,
                **kwargs,
            ) as s:
                for text in s.text_stream:
                    if text:
                        yield text
        except _anthropic_sdk.APITimeoutError as e:
            raise MakiTimeoutError(f"stream() timed out: {e}") from e
        except _anthropic_sdk.APIConnectionError as e:
            raise MakiNetworkError(f"stream() connection failed: {e}") from e
        except _anthropic_sdk.APIStatusError as e:
            raise error_for_status(e.status_code, f"stream() HTTP error {e.status_code}: {e}",
                                   retry_after=retry_after_of(e)) from e

    async def async_chat(
        self,
        prompt: str,
        history: Optional[list[Message]] = None,
        config: Optional[GenerationConfig] = None,
        system: Optional[str] = None,
        images: Optional[list[str]] = None,
    ) -> LLMResponse:
        """Async variant of chat()."""
        log.debug("async_chat: %s", prompt[:100])
        if self._rate_limiter:
            await self._rate_limiter.async_acquire()
        cfg = config or self.config
        messages = self._build_messages(prompt, history, images=images)
        kwargs = cfg.to_anthropic_kwargs()
        effective_system = self._effective_system(system)
        if effective_system:
            kwargs["system"] = effective_system
        t0 = time.perf_counter()
        try:
            response = await self._async_client.messages.create(
                model=self.model,
                messages=messages,
                **kwargs,
            )
        except _anthropic_sdk.APITimeoutError as e:
            raise MakiTimeoutError(f"async_chat() timed out: {e}") from e
        except _anthropic_sdk.APIConnectionError as e:
            raise MakiNetworkError(f"async_chat() connection failed: {e}") from e
        except _anthropic_sdk.APIStatusError as e:
            raise error_for_status(e.status_code, f"async_chat() HTTP error {e.status_code}: {e}",
                                   retry_after=retry_after_of(e)) from e
        elapsed = time.perf_counter() - t0
        result = self._parse_response(response, elapsed)
        log.info("async_chat: %.2fs, %d tokens", elapsed, result.total_tokens)
        return result

    def request(self, prompt: str) -> LLMResponse:
        """Override base request() to route through the chat API."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string")
        return self.chat(prompt)

    def session(self, system: Optional[str] = None) -> ChatSession:
        """Create a stateful multi-turn chat session."""
        return ChatSession(llm=self, system=system)

    def __repr__(self) -> str:
        return f"MakiAnthropic(model={self.model!r})"

    # ------------------------------------------------------------------
    # Native tool-calling (Anthropic tool-use API)
    # ------------------------------------------------------------------

    def to_tool_schemas(self, tool_specs: list) -> list:
        """Translate backend-agnostic specs to Anthropic tool format."""
        schemas = []
        for spec in tool_specs:
            schemas.append({
                "name": spec["name"],
                "description": spec["description"],
                "input_schema": spec["parameters"],
            })
        return schemas

    supports_typed_messages = True

    def to_wire_messages(self, messages: list) -> list:
        """Typed messages as Anthropic messages: ``tool_use`` blocks for calls, and the results of one
        turn batched into a single user message of ``tool_result`` blocks."""
        wire: list = []
        pending: list = []  # tool_result blocks waiting to become one user message

        def flush() -> None:
            if pending:
                wire.append({"role": "user", "content": list(pending)})
                pending.clear()

        for m in messages:
            if m.role == "tool":
                pending.append({"type": "tool_result", "tool_use_id": m.tool_call_id or "", "content": m.content})
                continue
            flush()
            if m.role not in ("user", "assistant"):
                continue  # the system prompt travels in the top-level system parameter
            if m.role == "assistant" and m.tool_calls:
                blocks: list = [{"type": "text", "text": m.content}] if m.content.strip() else []
                blocks.extend({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.args}
                              for tc in m.tool_calls)
                wire.append({"role": "assistant", "content": blocks})
            elif m.images and m.role == "user":
                blocks = [{"type": "text", "text": m.content}]
                blocks.extend({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}
                              for b64 in m.images)
                wire.append({"role": "user", "content": blocks})
            elif m.content.strip():
                wire.append({"role": m.role, "content": m.content})
        flush()
        return wire

    def chat_with_tools(
        self,
        messages: list,
        tools: list,
        *,
        system: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ):
        """One round of Anthropic native tool-calling.

        Returns ``(LLMResponse, None, messages)`` on a text answer, or
        ``(None, [ToolCall, ...], messages)`` when tool calls are requested.
        """
        response, tool_calls, updated = self._chat_with_tools_full(messages, tools, system=system, config=config)
        return (None, tool_calls, updated) if tool_calls else (response, None, updated)

    def _chat_with_tools_full(
        self,
        messages: list,
        tools: list,
        *,
        system: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ):
        """:meth:`chat_with_tools` with a real response (token counts included) on tool rounds too."""
        import time as _t
        if self._rate_limiter:
            self._rate_limiter.acquire()
        cfg = config or self.config
        kwargs = cfg.to_anthropic_kwargs()
        effective_system = self._effective_system(system)
        if effective_system:
            kwargs["system"] = effective_system
        if tools:
            kwargs["tools"] = tools

        t0 = _t.perf_counter()
        try:
            response = self._client.messages.create(
                model=self.model,
                messages=messages,
                **kwargs,
            )
        except _anthropic_sdk.APITimeoutError as e:
            raise MakiTimeoutError(f"chat_with_tools() timed out: {e}") from e
        except _anthropic_sdk.APIConnectionError as e:
            raise MakiNetworkError(f"chat_with_tools() connection failed: {e}") from e
        except _anthropic_sdk.APIStatusError as e:
            raise error_for_status(e.status_code, f"chat_with_tools() HTTP error {e.status_code}: {e}",
                                   retry_after=retry_after_of(e)) from e
        elapsed = _t.perf_counter() - t0

        content_blocks, tool_use_blocks = self._split_content(response)
        updated = list(messages) + [{"role": "assistant", "content": content_blocks}]

        result = self._parse_response(response, elapsed)
        if response.stop_reason == "tool_use" and tool_use_blocks:
            tool_calls = [
                ToolCall(id=b.id, name=b.name, args=b.input)
                for b in tool_use_blocks
            ]
            log.debug("chat_with_tools: %d tool call(s) requested", len(tool_calls))
            return result, tool_calls, updated

        log.info("chat_with_tools (text): %.2fs, %d tokens", elapsed, result.total_tokens)
        return result, None, updated

    @staticmethod
    def _split_content(response) -> tuple:
        """A serializable assistant content block list, and the ``tool_use`` blocks within it."""
        content_blocks = []
        tool_use_blocks = []
        for block in response.content:
            if block.type == "text":
                content_blocks.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content_blocks.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
                tool_use_blocks.append(block)
        return content_blocks, tool_use_blocks

    def append_tool_results(self, messages: list, results: list) -> list:
        """Append Anthropic-format tool results (batched into one user message)."""
        tool_result_blocks = [
            {
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": result_str,
            }
            for tc, result_str in results
        ]
        return list(messages) + [{"role": "user", "content": tool_result_blocks}]


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def claude_sonnet(system: Optional[str] = None, **kwargs) -> MakiAnthropic:
    """Pre-configured wrapper for Claude Sonnet 4.6."""
    return MakiAnthropic(model="claude-sonnet-4-6", system_prompt=system, **kwargs)


def claude_haiku(system: Optional[str] = None, **kwargs) -> MakiAnthropic:
    """Pre-configured wrapper for Claude Haiku 4.5."""
    return MakiAnthropic(model="claude-haiku-4-5-20251001", system_prompt=system, **kwargs)


def claude_opus(system: Optional[str] = None, **kwargs) -> MakiAnthropic:
    """Pre-configured wrapper for Claude Opus 4.8."""
    return MakiAnthropic(model="claude-opus-4-8", system_prompt=system, **kwargs)
