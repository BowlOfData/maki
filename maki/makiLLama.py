"""
A production-grade Python wrapper for local LLMs served via Ollama.
Supports: Gemma 3, Qwen, Llama, Mistral, Phi, DeepSeek and any other
model available through `ollama pull <model>`.

Quick-start:
    # 1. Install Ollama  →  https://ollama.com
    # 2. Pull a model    →  ollama pull gemma3
    # 3. Use it          →  from maki import MakiLLama
"""

from __future__ import annotations

import json
import time
import logging
from typing import Generator, Optional

from urllib.parse import urlparse

from .backend import LLMBackend
from .config import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_TEMPERATURE,
)
from .connector import AsyncConnector, Connector
from .utils import Utils
from .objects import LLMResponse, Message, GenerationConfig, RateLimiter, BackendType
from .session import ChatSession
from .exceptions import MakiNetworkError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core wrapper
# ---------------------------------------------------------------------------

class MakiLLama(LLMBackend):
    """
    A flexible wrapper around the Ollama HTTP API.

    Usage
    -----
        llm = LocalLLM(model="gemma3")
        response = llm.chat("What is the capital of France?")
        response.print()

        # Streaming
        for chunk in llm.stream("Tell me a joke"):
            print(chunk, end="", flush=True)

        # Multi-turn conversation
        session = llm.session(system="You are a senior Python engineer.")
        session.say("Explain list comprehensions.")
        session.say("Now show me a real-world example.")
    """

    OLLAMA_BASE_URL = DEFAULT_OLLAMA_BASE_URL

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
        timeout: int = DEFAULT_REQUEST_TIMEOUT,
        rate_limit: Optional[int] = None,
        think: Optional[bool] = None,
        json_format: bool = False,
    ) -> None:
        self.model = model
        self.temperature = config.temperature if config else DEFAULT_TEMPERATURE
        self._rate_limiter = RateLimiter(rate_limit) if rate_limit is not None else None
        self.base_url = base_url.rstrip("/")
        self.config = config or GenerationConfig()
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.think = think
        self.json_format = json_format
        # Operator-configured endpoint: private/LAN addresses are legitimate
        # (loopback Ollama is the common case), so allow_private=True.
        self._http = Connector(timeout=timeout, allow_private=True)
        self._async_http = AsyncConnector(timeout=timeout, allow_private=True)

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP session and release connections."""
        self._http.close()

    def __del__(self) -> None:
        try:
            http = getattr(self, "_http", None)
            if http is not None:
                http.close()
        except Exception:
            pass

    def verify(self) -> None:
        """Ping the Ollama daemon and log model availability.

        Call this explicitly after construction if you need to confirm the
        server is reachable before the first inference call.  The constructor
        no longer blocks on this so that ``maki serve`` can start without
        Ollama being up.

        Raises:
            MakiNetworkError: if the Ollama daemon is unreachable or times out.
        """
        try:
            r = self._http.get(f"{self.base_url}/api/tags", timeout=5)
            available = [m["name"] for m in r.json().get("models", [])]
            log.debug("Available models: %s", available)
            if not any(self.model in m for m in available):
                log.warning(
                    "Model '%s' not found locally. Run: ollama pull %s",
                    self.model, self.model,
                )
            else:
                log.info("Connected to Ollama · model=%s", self.model)
        except MakiNetworkError as e:
            log.error("Cannot reach Ollama at %s: %s", self.base_url, e)
            raise

    # Backward-compat alias kept so any callers of the private method still work.
    _verify_connection = verify

    def list_models(self) -> list[str]:
        """Return names of all locally pulled models."""
        r = self._http.get(f"{self.base_url}/api/tags", timeout=10)
        return [m["name"] for m in r.json().get("models", [])]

    def pull(self, model: Optional[str] = None) -> None:
        """Pull a model from the Ollama registry (blocking, shows progress)."""
        target = model or self.model
        log.info("Pulling model '%s' …", target)
        response = None
        try:
            response = self._http.post(
                f"{self.base_url}/api/pull",
                json={"name": target},
                stream=True,
                timeout=600,
            )
            last_pct = -10
            for line in Connector.iter_lines(response):
                if line:
                    data = json.loads(line)
                    status = data.get("status", "")
                    if data.get("total"):
                        pct = int(data.get("completed", 0) / data["total"] * 100)
                        if pct < last_pct:  # a new layer started downloading
                            last_pct = -10
                        if pct >= last_pct + 10:
                            log.info("  %s [%d%%]", status, pct)
                            last_pct = pct
                    else:
                        log.info("  %s", status)
                        last_pct = -10
        except Exception as e:
            log.error("Failed to pull model '%s': %s", target, str(e))
            raise
        finally:
            Utils.cleanup_response(response)
        log.info("Model '%s' ready.", target)

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

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
            msgs.append(Message("system", effective_system).to_dict())
        if history:
            msgs.extend(m.to_dict() for m in history)
        msgs.append(Message("user", prompt, images=images).to_dict())
        return msgs

    def _build_payload(
        self,
        prompt: str,
        history: Optional[list[Message]],
        config: Optional[GenerationConfig],
        *,
        stream: bool,
        system: Optional[str] = None,
        images: Optional[list[str]] = None,
    ) -> dict:
        cfg = config or self.config
        payload: dict = {
            "model": self.model,
            "messages": self._build_messages(prompt, history, system=system, images=images),
            "stream": stream,
            "options": cfg.to_ollama_options(),
        }
        if self.think is not None:
            payload["think"] = self.think
        if self.json_format:
            payload["format"] = "json"
        return payload

    def _parse_response(self, data: dict, elapsed: float) -> LLMResponse:
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        content = data["message"]["content"]
        # Thinking models (gemma4, qwen3, …) may leave content empty when the model
        # exhausted its budget on the reasoning trace. Fall back to the thinking field
        # so callers always receive something extractable.
        if not content.strip():
            thinking = data["message"].get("thinking", "")
            if thinking:
                log.debug("_parse_response: content empty, falling back to thinking field")
                content = thinking
        return LLMResponse(
            content=content,
            model=data.get("model", self.model),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            elapsed_seconds=elapsed,
            done=data.get("done", True),
            backend=BackendType.OLLAMA,
        )

    def chat(
        self,
        prompt: str,
        history: Optional[list[Message]] = None,
        config: Optional[GenerationConfig] = None,
        system: Optional[str] = None,
        images: Optional[list[str]] = None,
    ) -> LLMResponse:
        """
        Single-turn (or multi-turn with explicit history) generation.
        Returns a fully resolved LLMResponse. Pass base64 strings in images for vision models.
        """
        log.debug("chat: %s", prompt[:100])
        if self._rate_limiter:
            self._rate_limiter.acquire()
        payload = self._build_payload(prompt, history, config, stream=False, system=system, images=images)
        t0 = time.perf_counter()
        r = self._http.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        elapsed = time.perf_counter() - t0
        response = self._parse_response(Connector.json_or_raise(r), elapsed)
        log.info("chat: %.2fs, %d tokens", elapsed, response.total_tokens)
        return response

    def stream(
        self,
        prompt: str,
        history: Optional[list[Message]] = None,
        config: Optional[GenerationConfig] = None,
        system: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Streaming generation — yields text chunks as they arrive.

        Example:
            for chunk in llm.stream("Write a haiku about Python"):
                print(chunk, end="", flush=True)
        """
        log.debug("stream: %s", prompt[:100])
        if self._rate_limiter:
            self._rate_limiter.acquire()
        payload = self._build_payload(prompt, history, config, stream=True, system=system)
        response = None
        try:
            response = self._http.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=self.timeout,
            )
            for line in Connector.iter_lines(response):
                if line:
                    data = json.loads(line)
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        yield chunk
                    if data.get("done"):
                        break
        except Exception as e:
            log.error("stream failed: %s", e)
            raise
        finally:
            Utils.cleanup_response(response)

    def chat_collect(
        self,
        prompt: str,
        history: Optional[list[Message]] = None,
        config: Optional[GenerationConfig] = None,
        system: Optional[str] = None,
        images: Optional[list[str]] = None,
    ) -> LLMResponse:
        """
        Like chat() but uses HTTP streaming internally.

        The configured timeout applies per-chunk rather than to the total response,
        so long-running generations (e.g. large ranking tasks) complete without
        hitting the global read timeout.  Raises MakiTimeoutError only if the
        model stops producing output for longer than self.timeout seconds.
        """
        log.debug("chat_collect (stream): %s", prompt[:100])
        if self._rate_limiter:
            self._rate_limiter.acquire()
        payload = self._build_payload(prompt, history, config, stream=True, system=system, images=images)
        t0 = time.perf_counter()
        chunks: list[str] = []
        last_data: dict = {}
        response = None
        try:
            # The configured timeout applies per-chunk on the streaming read,
            # so long generations survive as long as chunks keep arriving.
            response = self._http.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=self.timeout,
            )
            for line in Connector.iter_lines(response):
                if line:
                    data = json.loads(line)
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        chunks.append(chunk)
                    if data.get("done"):
                        last_data = data
                        break
        finally:
            Utils.cleanup_response(response)

        elapsed = time.perf_counter() - t0
        content = "".join(chunks)
        if not content.strip():
            thinking = last_data.get("message", {}).get("thinking", "")
            if thinking:
                log.debug("chat_collect: content empty, falling back to thinking field")
                content = thinking
        prompt_tokens = last_data.get("prompt_eval_count", 0)
        completion_tokens = last_data.get("eval_count", 0)
        log.info("chat_collect: %.2fs, %d tokens", elapsed, prompt_tokens + completion_tokens)
        return LLMResponse(
            content=content,
            model=last_data.get("model", self.model),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            elapsed_seconds=elapsed,
            done=last_data.get("done", True),
        )

    async def async_chat(
        self,
        prompt: str,
        history: Optional[list[Message]] = None,
        config: Optional[GenerationConfig] = None,
        images: Optional[list[str]] = None,
        system: Optional[str] = None,
    ) -> LLMResponse:
        """Async variant of chat() for use inside asyncio event loops. Supports vision via images."""
        log.debug("async_chat: %s", prompt[:100])
        if self._rate_limiter:
            await self._rate_limiter.async_acquire()
        payload = self._build_payload(prompt, history, config, stream=False, images=images, system=system)
        t0 = time.perf_counter()
        r = await self._async_http.post(f"{self.base_url}/api/chat", json=payload)
        elapsed = time.perf_counter() - t0
        response = self._parse_response(Connector.json_or_raise(r), elapsed)
        log.info("async_chat: %.2fs, %d tokens", elapsed, response.total_tokens)
        return response

    def chat_with_image(
        self,
        prompt: str,
        image_b64: str,
        config: Optional[GenerationConfig] = None,
        system: Optional[str] = None,
    ) -> LLMResponse:
        """Single-turn vision chat with a base64-encoded image."""
        return self.chat(prompt, config=config, system=system, images=[image_b64])

    def request(self, prompt: str) -> LLMResponse:
        """Override base request() to route through the chat API."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string")
        return self.chat(prompt)

    def session(self, system: Optional[str] = None) -> "ChatSession":
        """Create a stateful multi-turn chat session."""
        return ChatSession(llm=self, system=system)

    # ------------------------------------------------------------------
    # Convenience shortcuts
    # ------------------------------------------------------------------

    def __call__(self, prompt: str, **kwargs) -> LLMResponse:
        """llm("your prompt") as a shorthand for llm.chat(...)."""
        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string")

        # Pass through every kwarg chat() accepts; warn on anything else.
        allowed_params = {'history', 'config', 'system', 'images'}
        filtered_kwargs = {}

        for key, value in kwargs.items():
            if key in allowed_params:
                filtered_kwargs[key] = value
            else:
                log.warning(f"Disallowed parameter '{key}' ignored in __call__")

        log.debug("Calling LLM with prompt: %s", prompt[:100] + "..." if len(prompt) > 100 else prompt)
        return self.chat(prompt, **filtered_kwargs)

    def __repr__(self) -> str:
        return f"LocalLLM(model={self.model!r}, base_url={self.base_url!r})"


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def gemma3(system: Optional[str] = None, **kwargs) -> MakiLLama:
    """Pre-configured wrapper for Google Gemma 3."""
    return MakiLLama(model="gemma3", system_prompt=system, **kwargs)


def gemma4(variant: str = "gemma4:26b", system: Optional[str] = None, **kwargs) -> MakiLLama:
    """Pre-configured wrapper for Google Gemma 4 (vision-capable)."""
    return MakiLLama(model=variant, system_prompt=system, **kwargs)


def qwen(variant: str = "qwen2.5:7b", system: Optional[str] = None, **kwargs) -> MakiLLama:
    """Pre-configured wrapper for Alibaba Qwen."""
    return MakiLLama(model=variant, system_prompt=system, **kwargs)


def llama(variant: str = "llama3.2", system: Optional[str] = None, **kwargs) -> MakiLLama:
    """Pre-configured wrapper for Meta Llama 3."""
    return MakiLLama(model=variant, system_prompt=system, **kwargs)


def mistral(system: Optional[str] = None, **kwargs) -> MakiLLama:
    """Pre-configured wrapper for Mistral."""
    return MakiLLama(model="mistral", system_prompt=system, **kwargs)

