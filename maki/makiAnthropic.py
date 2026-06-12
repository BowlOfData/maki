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
from .exceptions import MakiAPIError, MakiNetworkError, MakiTimeoutError
from .objects import BackendType, GenerationConfig, LLMResponse, Message, RateLimiter
from .session import ChatSession

try:
    import anthropic as _anthropic_sdk
except ImportError:
    _anthropic_sdk = None  # type: ignore

log = logging.getLogger(__name__)


class MakiAnthropic(LLMBackend):
    """
    Anthropic API backend (Messages API).

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
        content = response.content[0].text if response.content else ""  # type: ignore[attr-defined]
        return LLMResponse(
            content=content,
            model=response.model,  # type: ignore[attr-defined]
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
            elapsed_seconds=elapsed,
            done=True,
            backend=BackendType.ANTHROPIC,
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
            self._rate_limiter.acquire()
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
        return f"MakiAnthropic(model={self.model!r})"


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
