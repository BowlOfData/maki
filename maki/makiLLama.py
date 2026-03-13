"""
A production-grade Python wrapper for local LLMs served via Ollama.
Supports: Gemma 3, Qwen, Llama, Mistral, Phi, DeepSeek and any other
model available through `ollama pull <model>`.

Requirements:
    pip install requests httpx rich

Quick-start:
    # 1. Install Ollama  →  https://ollama.com
    # 2. Pull a model    →  ollama pull gemma3
    # 3. Run this file   →  python local_llm.py
"""

from __future__ import annotations

import json
import time
import logging
from typing import Generator, Iterator, Optional

from urllib.parse import urlparse

from .maki import Maki
from .utils import Utils
from .objects import LLMResponse, Message, GenerationConfig

import requests
import httpx

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core wrapper
# ---------------------------------------------------------------------------

class MakiLLama(Maki):
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

    OLLAMA_BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        model: str = "gemma3",
        base_url: str = OLLAMA_BASE_URL,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
        timeout: int = 120,
    ) -> None:
        parsed = urlparse(base_url)
        port = str(parsed.port) if parsed.port else "11434"
        super().__init__(url=base_url, port=port, model=model, temperature=config.temperature if config else 0.7)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.config = config or GenerationConfig()
        self.system_prompt = system_prompt
        self.timeout = timeout
        self._verify_connection()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _verify_connection(self) -> None:
        """Ping the Ollama daemon; raise a friendly error if unreachable."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            available = [m["name"] for m in r.json().get("models", [])]
            log.debug("Available models: %s", available)
            if not any(self.model in m for m in available):
                log.warning(
                    "Model '%s' not found locally. Run: ollama pull %s",
                    self.model, self.model,
                )
            else:
                log.info("Connected to Ollama · model=%s", self.model)
        except requests.exceptions.ConnectionError as e:
            log.error("Cannot reach Ollama at %s", self.base_url)
            raise RuntimeError(
                "Cannot reach Ollama at %s.\n"
                "  → Install Ollama: https://ollama.com\n"
                "  → Start the daemon: ollama serve" % self.base_url
            ) from e

    def list_models(self) -> list[str]:
        """Return names of all locally pulled models."""
        r = requests.get(f"{self.base_url}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def pull(self, model: Optional[str] = None) -> None:
        """Pull a model from the Ollama registry (blocking, shows progress)."""
        target = model or self.model
        log.info("Pulling model '%s' …", target)
        response = None
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": target},
                stream=True,
                timeout=600,
            )
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    status = data.get("status", "")
                    if "total" in data and data["total"]:
                        pct = int(data.get("completed", 0) / data["total"] * 100)
                        log.info(f"  {status} [{pct}%]", end="\r")
                    else:
                        log.info(f"  {status}")
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
    ) -> list[dict]:
        msgs: list[dict] = []
        if self.system_prompt:
            msgs.append(Message("system", self.system_prompt).to_dict())
        if history:
            msgs.extend(m.to_dict() for m in history)
        msgs.append(Message("user", prompt).to_dict())
        return msgs

    def _build_payload(
        self,
        prompt: str,
        history: Optional[list[Message]],
        config: Optional[GenerationConfig],
        *,
        stream: bool,
    ) -> dict:
        cfg = config or self.config
        return {
            "model": self.model,
            "messages": self._build_messages(prompt, history),
            "stream": stream,
            "options": cfg.to_ollama_options(),
        }

    def _parse_response(self, data: dict, elapsed: float) -> LLMResponse:
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        return LLMResponse(
            content=data["message"]["content"],
            model=data.get("model", self.model),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            elapsed_seconds=elapsed,
            done=data.get("done", True),
        )

    def chat(
        self,
        prompt: str,
        history: Optional[list[Message]] = None,
        config: Optional[GenerationConfig] = None,
    ) -> LLMResponse:
        """
        Single-turn (or multi-turn with explicit history) generation.
        Returns a fully resolved LLMResponse.
        """
        log.debug("chat: %s", prompt[:100])
        payload = self._build_payload(prompt, history, config, stream=False)
        t0 = time.perf_counter()
        r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        elapsed = time.perf_counter() - t0
        r.raise_for_status()
        response = self._parse_response(r.json(), elapsed)
        log.info("chat: %.2fs, %d tokens", elapsed, response.total_tokens)
        return response

    def stream(
        self,
        prompt: str,
        history: Optional[list[Message]] = None,
        config: Optional[GenerationConfig] = None,
    ) -> Generator[str, None, None]:
        """
        Streaming generation — yields text chunks as they arrive.

        Example:
            for chunk in llm.stream("Write a haiku about Python"):
                print(chunk, end="", flush=True)
        """
        log.debug("stream: %s", prompt[:100])
        payload = self._build_payload(prompt, history, config, stream=True)
        response = None
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=self.timeout,
            )
            response.raise_for_status()
            for line in response.iter_lines():
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

    async def async_chat(
        self,
        prompt: str,
        history: Optional[list[Message]] = None,
        config: Optional[GenerationConfig] = None,
    ) -> LLMResponse:
        """Async variant of chat() for use inside asyncio event loops."""
        log.debug("async_chat: %s", prompt[:100])
        payload = self._build_payload(prompt, history, config, stream=False)
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                r = await client.post(f"{self.base_url}/api/chat", json=payload)
            except httpx.TimeoutException:
                log.error("async_chat timed out after %ds", self.timeout)
                raise
            except httpx.RequestError as e:
                log.error("async_chat network error: %s", e)
                raise
        elapsed = time.perf_counter() - t0
        r.raise_for_status()
        response = self._parse_response(r.json(), elapsed)
        log.info("async_chat: %.2fs, %d tokens", elapsed, response.total_tokens)
        return response

    def session(self, system: Optional[str] = None) -> "ChatSession":
        """Create a stateful multi-turn chat session."""
        return ChatSession(llm=self, system=system)

    # ------------------------------------------------------------------
    # Convenience shortcuts
    # ------------------------------------------------------------------

    def __call__(self, prompt: str, **kwargs) -> LLMResponse:
        """llm("your prompt") as a shorthand for llm.chat(...)."""
        # Validate and sanitize input parameters to prevent code injection
        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string")

        # Only allow specific, safe parameters to be passed through
        allowed_params = {'history', 'config'}
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
# Stateful session
# ---------------------------------------------------------------------------

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

    def __init__(self, llm: MakiLLama, system: Optional[str] = None) -> None:
        self._llm = llm
        self._history: list[Message] = []
        self._system = system

        # Override system prompt for this session only
        if system:
            self._llm_snapshot_system = self._llm.system_prompt
            self._llm.system_prompt = system

    def say(
        self,
        prompt: str,
        stream: bool = False,
        config: Optional[GenerationConfig] = None,
    ) -> LLMResponse | Iterator[str]:
        """Send a message and automatically extend the conversation history."""
        log.debug("Session saying: %s", prompt[:100] + "..." if len(prompt) > 100 else prompt)
        if stream:
            # Collect streamed chunks and store to history after
            full = ""
            for chunk in self._llm.stream(prompt, history=self._history, config=config):
                full += chunk
                yield chunk
            self._history.append(Message("user", prompt))
            self._history.append(Message("assistant", full))
        else:
            response = self._llm.chat(prompt, history=self._history, config=config)
            self._history.append(Message("user", prompt))
            self._history.append(Message("assistant", response.content))
            return response

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


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def gemma3(system: Optional[str] = None, **kwargs) -> MakiLLama:
    """Pre-configured wrapper for Google Gemma 3."""
    return MakiLLama(model="gemma3", system_prompt=system, **kwargs)


def qwen(variant: str = "qwen2.5:7b", system: Optional[str] = None, **kwargs) -> MakiLLama:
    """Pre-configured wrapper for Alibaba Qwen."""
    return MakiLLama(model=variant, system_prompt=system, **kwargs)


def llama(variant: str = "llama3.2", system: Optional[str] = None, **kwargs) -> MakiLLama:
    """Pre-configured wrapper for Meta Llama 3."""
    return MakiLLama(model=variant, system_prompt=system, **kwargs)


def mistral(system: Optional[str] = None, **kwargs) -> MakiLLama:
    """Pre-configured wrapper for Mistral."""
    return MakiLLama(model="mistral", system_prompt=system, **kwargs)


