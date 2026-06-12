"""
MakiOpenAI — OpenAI API backend for the Maki framework.

Requirements:
    pip install openai
    export OPENAI_API_KEY=sk-...
"""

from __future__ import annotations

import logging
import os
import time
from typing import Generator, Optional

from .backend import LLMBackend
from .config import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_REQUEST_TIMEOUT,
    OPENAI_API_KEY_ENV,
)
from .exceptions import MakiAPIError, MakiNetworkError, MakiTimeoutError
from .objects import BackendType, GenerationConfig, LLMResponse, Message, RateLimiter, ToolCall
from .session import ChatSession

try:
    import openai as _openai_sdk
except ImportError:
    _openai_sdk = None  # type: ignore

log = logging.getLogger(__name__)


class MakiOpenAI(LLMBackend):
    """
    OpenAI API backend (chat completions).

    Supports native tool calling via the OpenAI function-calling API.

    Usage
    -----
        llm = MakiOpenAI(model="gpt-4o-mini")
        response = llm.chat("What is the capital of France?")
        print(response.content)

        for chunk in llm.stream("Tell me a joke"):
            print(chunk, end="", flush=True)

        session = llm.session(system="You are a senior Python engineer.")
        session.say("Explain list comprehensions.")
    """
    supports_native_tools: bool = True

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        api_key: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
        timeout: int = DEFAULT_REQUEST_TIMEOUT,
        rate_limit: Optional[int] = None,
    ) -> None:
        if _openai_sdk is None:
            raise ImportError("openai package is required: pip install openai")

        resolved_key = api_key or os.environ.get(OPENAI_API_KEY_ENV)
        if not resolved_key:
            raise ValueError(
                f"OpenAI API key not found. Pass api_key= or set {OPENAI_API_KEY_ENV}."
            )

        self.model = model
        self.config = config or GenerationConfig()
        self.temperature = self.config.temperature
        self.system_prompt = system_prompt
        self.timeout = timeout
        self._rate_limiter = RateLimiter(rate_limit) if rate_limit is not None else None
        self._client = _openai_sdk.OpenAI(api_key=resolved_key, timeout=timeout)
        self._async_client = _openai_sdk.AsyncOpenAI(api_key=resolved_key, timeout=timeout)
        log.info("MakiOpenAI · model=%s", model)

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    @property
    def _model_family(self) -> str:
        """Return 'reasoning' for o1/o3/o4 models, 'chat' for everything else."""
        if self.model.startswith(("o1", "o3", "o4")):
            return "reasoning"
        return "chat"

    def _build_messages(
        self,
        prompt: str,
        history: Optional[list[Message]] = None,
        system: Optional[str] = None,
        images: Optional[list[str]] = None,
    ) -> list[dict]:
        msgs: list[dict] = []
        effective_system = system if system is not None else self.system_prompt
        if effective_system:
            msgs.append({"role": "system", "content": effective_system})
        if history:
            for m in history:
                if m.images and m.role == "user":
                    content: list[dict] = [{"type": "text", "text": m.content}]
                    for b64 in m.images:
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        })
                    msgs.append({"role": m.role, "content": content})
                else:
                    msgs.append({"role": m.role, "content": m.content})
        if images:
            content: list[dict] = [{"type": "text", "text": prompt}]
            for b64 in images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
            msgs.append({"role": "user", "content": content})
        else:
            msgs.append({"role": "user", "content": prompt})
        return msgs

    def _parse_response(self, response: object, elapsed: float) -> LLMResponse:
        usage = response.usage  # type: ignore[attr-defined]
        return LLMResponse(
            content=response.choices[0].message.content or "",  # type: ignore[attr-defined]
            model=response.model,  # type: ignore[attr-defined]
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            elapsed_seconds=elapsed,
            done=True,
            backend=BackendType.OPENAI,
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
        messages = self._build_messages(prompt, history, system=system, images=images)
        t0 = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                **cfg.to_openai_kwargs(model_family=self._model_family),
            )
        except _openai_sdk.APITimeoutError as e:
            raise MakiTimeoutError(f"chat() timed out: {e}") from e
        except _openai_sdk.APIConnectionError as e:
            raise MakiNetworkError(f"chat() connection failed: {e}") from e
        except _openai_sdk.APIStatusError as e:
            raise MakiAPIError(f"chat() HTTP error {e.status_code}: {e}") from e
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
        messages = self._build_messages(prompt, history, system=system)
        try:
            with self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                **cfg.to_openai_kwargs(model_family=self._model_family),
            ) as stream:
                for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        yield delta
        except _openai_sdk.APITimeoutError as e:
            raise MakiTimeoutError(f"stream() timed out: {e}") from e
        except _openai_sdk.APIConnectionError as e:
            raise MakiNetworkError(f"stream() connection failed: {e}") from e
        except _openai_sdk.APIStatusError as e:
            raise MakiAPIError(f"stream() HTTP error {e.status_code}: {e}") from e

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
        messages = self._build_messages(prompt, history, system=system, images=images)
        t0 = time.perf_counter()
        try:
            response = await self._async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                **cfg.to_openai_kwargs(model_family=self._model_family),
            )
        except _openai_sdk.APITimeoutError as e:
            raise MakiTimeoutError(f"async_chat() timed out: {e}") from e
        except _openai_sdk.APIConnectionError as e:
            raise MakiNetworkError(f"async_chat() connection failed: {e}") from e
        except _openai_sdk.APIStatusError as e:
            raise MakiAPIError(f"async_chat() HTTP error {e.status_code}: {e}") from e
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
        return f"MakiOpenAI(model={self.model!r})"

    # ------------------------------------------------------------------
    # Native tool-calling (OpenAI function-calling API)
    # ------------------------------------------------------------------

    def to_tool_schemas(self, tool_specs: list) -> list:
        """Translate backend-agnostic specs to OpenAI tool format."""
        schemas = []
        for spec in tool_specs:
            schemas.append({
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": spec["parameters"],
                },
            })
        return schemas

    def chat_with_tools(
        self,
        messages: list,
        tools: list,
        *,
        system: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ):
        """One round of OpenAI native tool-calling.

        Returns ``(LLMResponse, None, messages)`` on a text answer, or
        ``(None, [ToolCall, ...], messages)`` when tool calls are requested.
        """
        import json as _json
        import time as _t
        if self._rate_limiter:
            self._rate_limiter.acquire()
        cfg = config or self.config

        effective_system = system if system is not None else self.system_prompt
        full_messages = []
        if effective_system:
            full_messages.append({"role": "system", "content": effective_system})
        full_messages.extend(messages)

        kwargs = cfg.to_openai_kwargs(model_family=self._model_family)
        if tools:
            kwargs["tools"] = tools

        t0 = _t.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                **kwargs,
            )
        except _openai_sdk.APITimeoutError as e:
            raise MakiTimeoutError(f"chat_with_tools() timed out: {e}") from e
        except _openai_sdk.APIConnectionError as e:
            raise MakiNetworkError(f"chat_with_tools() connection failed: {e}") from e
        except _openai_sdk.APIStatusError as e:
            raise MakiAPIError(f"chat_with_tools() HTTP error {e.status_code}: {e}") from e
        elapsed = _t.perf_counter() - t0

        choice = response.choices[0]
        msg = choice.message

        # Reconstruct a serializable assistant turn.
        assistant_entry: dict = {"role": "assistant", "content": msg.content or ""}
        raw_tool_calls = msg.tool_calls
        if raw_tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in raw_tool_calls
            ]
        updated = list(messages) + [assistant_entry]

        if raw_tool_calls and choice.finish_reason == "tool_calls":
            tool_calls = []
            for tc in raw_tool_calls:
                try:
                    args = _json.loads(tc.function.arguments)
                except _json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, args=args))
            log.debug("chat_with_tools: %d tool call(s) requested", len(tool_calls))
            return None, tool_calls, updated

        result = self._parse_response(response, elapsed)
        log.info("chat_with_tools (text): %.2fs, %d tokens", elapsed, result.total_tokens)
        return result, None, updated

    def append_tool_results(self, messages: list, results: list) -> list:
        """Append OpenAI-format tool result messages (one per call)."""
        updated = list(messages)
        for tc, result_str in results:
            updated.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })
        return updated


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def gpt4o(system: Optional[str] = None, **kwargs) -> MakiOpenAI:
    """Pre-configured wrapper for GPT-4o."""
    return MakiOpenAI(model="gpt-4o", system_prompt=system, **kwargs)


def gpt4o_mini(system: Optional[str] = None, **kwargs) -> MakiOpenAI:
    """Pre-configured wrapper for GPT-4o Mini."""
    return MakiOpenAI(model="gpt-4o-mini", system_prompt=system, **kwargs)


def o3(system: Optional[str] = None, **kwargs) -> MakiOpenAI:
    """Pre-configured wrapper for o3."""
    return MakiOpenAI(model="o3", system_prompt=system, **kwargs)
