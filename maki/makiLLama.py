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
from dataclasses import dataclass, field
from typing import Generator, Iterator, Optional

from .maki import Maki
from .utils import Utils

import requests
import httpx
from rich.console import Console
from rich.markdown import Markdown

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger(__name__)

console = Console()

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Message:
    role: str          # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class GenerationConfig:
    """Sampling / generation hyper-parameters forwarded to Ollama."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int = 2048        # maps to num_predict in Ollama
    seed: int = -1                # -1 = random
    stop: list[str] = field(default_factory=list)

    def to_ollama_options(self) -> dict:
        opts: dict = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "num_predict": self.max_tokens,
        }
        if self.seed != -1:
            opts["seed"] = self.seed
        if self.stop:
            opts["stop"] = self.stop
        return opts


@dataclass
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    elapsed_seconds: float
    done: bool = True

    def __str__(self) -> str:
        return self.content

    def print(self, markdown: bool = True) -> None:
        if markdown:
            console.print(Markdown(self.content))
        else:
            console.print(self.content)

    @property
    def tokens_per_second(self) -> float:
        return self.completion_tokens / self.elapsed_seconds if self.elapsed_seconds else 0.0


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
                        console.print(f"  {status} [{pct}%]", end="\r")
                    else:
                        console.print(f"  {status}")
        except Exception as e:
            log.error("Failed to pull model '%s': %s", target, str(e))
            # Ensure we clean up the response if there's an error
            Utils.cleanup_response(response)
            raise e
        finally:
            # Clean up response if needed
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
        cfg = config or self.config
        payload = {
            "model": self.model,
            "messages": self._build_messages(prompt, history),
            "stream": False,
            "options": cfg.to_ollama_options(),
        }
        log.debug("Sending chat request with prompt: %s", prompt[:100] + "..." if len(prompt) > 100 else prompt)
        t0 = time.perf_counter()
        r = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        elapsed = time.perf_counter() - t0
        r.raise_for_status()
        data = r.json()

        response = LLMResponse(
            content=data["message"]["content"],
            model=data.get("model", self.model),
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            elapsed_seconds=elapsed,
            done=data.get("done", True),
        )
        log.info("Chat response generated for prompt (length: %d) in %.2f seconds", len(prompt), elapsed)
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
        cfg = config or self.config
        payload = {
            "model": self.model,
            "messages": self._build_messages(prompt, history),
            "stream": True,
            "options": cfg.to_ollama_options(),
        }
        log.debug("Sending streaming request with prompt: %s", prompt[:100] + "..." if len(prompt) > 100 else prompt)
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
            log.error("Streaming request failed: %s", str(e))
            # Ensure we clean up the response if there's an error
            Utils.cleanup_response(response)
            raise e
        finally:
            # Clean up response if needed
            Utils.cleanup_response(response)

    async def async_chat(
        self,
        prompt: str,
        history: Optional[list[Message]] = None,
        config: Optional[GenerationConfig] = None,
    ) -> LLMResponse:
        """Async variant of chat() for use inside asyncio event loops."""
        cfg = config or self.config
        payload = {
            "model": self.model,
            "messages": self._build_messages(prompt, history),
            "stream": False,
            "options": cfg.to_ollama_options(),
        }
        t0 = time.perf_counter()
        client = None
        try:
            client = httpx.AsyncClient(timeout=self.timeout)
            r = await client.post(f"{self.base_url}/api/chat", json=payload)
            elapsed = time.perf_counter() - t0

            # Check for HTTP errors
            if r.status_code >= 400:
                error_text = await r.aread()
                raise RuntimeError(
                    f"HTTP {r.status_code} when calling Ollama API: {error_text.decode('utf-8', errors='ignore')}"
                )

            # Parse JSON response
            try:
                data = r.json()
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Failed to parse JSON response from Ollama: {e}")

        except httpx.TimeoutException:
            log.error("Async chat request timed out after %d seconds", self.timeout)
            raise RuntimeError(f"Timeout after {self.timeout} seconds when calling Ollama API")
        except httpx.RequestError as e:
            log.error("Network error in async chat request: %s", str(e))
            raise RuntimeError(f"Network error when calling Ollama API: {e}")
        except Exception as e:
            log.error("Error in async chat request: %s", str(e))
            # Re-raise any other exceptions
            raise RuntimeError(f"Error calling Ollama API: {e}")
        finally:
            # Ensure client is closed properly - for async methods we need to handle this differently
            if client is not None:
                try:
                    await client.aclose()
                except:
                    pass

        response = LLMResponse(
            content=data["message"]["content"],
            model=data.get("model", self.model),
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            elapsed_seconds=elapsed,
            done=data.get("done", True),
        )
        log.info("Async chat response generated for prompt (length: %d) in %.2f seconds", len(prompt), elapsed)
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
            console.print(f"[bold {color}]{msg.role.upper()}[/bold {color}]")
            console.print(Markdown(msg.content))
            console.rule()

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


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------

def _demo() -> None:
    console.rule("[bold blue]local_llm.py — Demo[/bold blue]")

    # ── 1. Basic chat ──────────────────────────────────────────────────────
    llm = gemma3(system="You are a concise assistant. Keep answers under 3 sentences.")
    console.print("\n[bold]1. Single-turn chat[/bold]")
    resp = llm("What is recursion in programming?")
    resp.print()
    console.print(
        f"\n[dim]Tokens: {resp.total_tokens} | "
        f"Speed: {resp.tokens_per_second:.1f} tok/s | "
        f"Time: {resp.elapsed_seconds:.2f}s[/dim]"
    )

    # ── 2. Streaming ───────────────────────────────────────────────────────
    console.print("\n[bold]2. Streaming output[/bold]")
    for chunk in llm.stream("Name 3 Python web frameworks in one sentence each."):
        console.print(chunk, end="")
    console.print()

    # ── 3. Multi-turn session ──────────────────────────────────────────────
    console.print("\n[bold]3. Multi-turn session[/bold]")
    session = llm.session(system="You are an expert chef specializing in Italian cuisine.")
    r1 = session.say("What's the secret to perfect pasta carbonara?")
    console.print(r1)
    r2 = session.say("Can I substitute bacon for guanciale?")
    console.print(r2)
    console.print(f"\n[dim]Session turns: {len(session)}[/dim]")

    # ── 4. List local models ────────────────────────────────────────────────
    console.print("\n[bold]4. Available local models[/bold]")
    models = llm.list_models()
    for m in models:
        console.print(f"  • {m}")

    console.rule("[bold green]Done[/bold green]")


if __name__ == "__main__":
    _demo()
