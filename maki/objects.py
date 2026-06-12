import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import threading
import time as _time


class BackendType(str, Enum):
    OLLAMA    = "ollama"
    OPENAI    = "openai"
    ANTHROPIC = "anthropic"
    HF        = "huggingface"


@dataclass
class Message:
    role: str          # "system" | "user" | "assistant"
    content: str
    images: Optional[list[str]] = field(default=None, repr=False)  # base64-encoded images for vision models

    def to_dict(self) -> dict:
        d: dict = {"role": self.role, "content": self.content}
        if self.images:
            d["images"] = self.images
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'Message':
        return cls(role=data["role"], content=data["content"], images=data.get("images"))


@dataclass
class GenerationConfig:
    """Sampling / generation hyper-parameters forwarded to Ollama or HuggingFace."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int = 2048        # maps to num_predict in Ollama, max_new_tokens in HF
    seed: int = -1                # -1 = random
    stop: list[str] = field(default_factory=list)
    do_sample: bool = True        # HuggingFace: enable sampling (set False for greedy)
    num_ctx: Optional[int] = None # Ollama only: total context window (input + output tokens)

    def __post_init__(self):
        if not isinstance(self.temperature, (int, float)):
            raise ValueError("temperature must be a number")
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        if not isinstance(self.top_p, (int, float)) or not (0.0 <= self.top_p <= 1.0):
            raise ValueError("top_p must be a float in [0.0, 1.0]")
        if not isinstance(self.top_k, int) or self.top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        if not isinstance(self.max_tokens, int) or self.max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")

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
        if self.num_ctx is not None:
            opts["num_ctx"] = self.num_ctx
        return opts

    def to_hf_kwargs(self) -> dict:
        return {
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature if self.do_sample else 1.0,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repeat_penalty,
            "do_sample": self.do_sample,
        }

    def to_openai_kwargs(self, model_family: str = "chat") -> dict:
        """Serialise for the OpenAI chat completions API.

        Pass ``model_family="reasoning"`` for o1/o3/o4 models: they require
        ``max_completion_tokens`` and reject ``temperature`` / ``top_p``.
        """
        if model_family == "reasoning":
            kwargs: dict = {"max_completion_tokens": self.max_tokens}
            if self.seed != -1:
                kwargs["seed"] = self.seed
            if self.stop:
                kwargs["stop"] = self.stop
            return kwargs
        kwargs = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.seed != -1:
            kwargs["seed"] = self.seed
        if self.stop:
            kwargs["stop"] = self.stop
        return kwargs

    def to_anthropic_kwargs(self) -> dict:
        kwargs: dict = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.stop:
            kwargs["stop_sequences"] = self.stop
        return kwargs

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "stop": self.stop,
            "do_sample": self.do_sample,
            "num_ctx": self.num_ctx,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'GenerationConfig':
        return cls(
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 0.9),
            top_k=data.get("top_k", 40),
            repeat_penalty=data.get("repeat_penalty", 1.1),
            max_tokens=data.get("max_tokens", 2048),
            seed=data.get("seed", -1),
            stop=data.get("stop", []),
            do_sample=data.get("do_sample", True),
            num_ctx=data.get("num_ctx"),
        )


@dataclass
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    elapsed_seconds: float
    done: bool = True
    backend: BackendType = BackendType.OLLAMA

    def __str__(self) -> str:
        return self.content

    @property
    def tokens_per_second(self) -> float:
        return self.completion_tokens / self.elapsed_seconds if self.elapsed_seconds else 0.0

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "elapsed_seconds": self.elapsed_seconds,
            "done": self.done,
            "backend": self.backend.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'LLMResponse':
        return cls(
            content=data["content"],
            model=data["model"],
            prompt_tokens=data["prompt_tokens"],
            completion_tokens=data["completion_tokens"],
            total_tokens=data["total_tokens"],
            elapsed_seconds=data["elapsed_seconds"],
            done=data.get("done", True),
            backend=BackendType(data["backend"]),
        )


class RateLimiter:
    """Thread-safe token-bucket rate limiter.

    Allows up to `requests_per_minute` calls per minute, supporting
    short bursts up to that capacity and refilling continuously.
    """

    def __init__(self, requests_per_minute: int) -> None:
        if not isinstance(requests_per_minute, int) or requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be a positive integer")
        self._capacity = float(requests_per_minute)
        self._tokens = float(requests_per_minute)
        self._refill_rate = requests_per_minute / 60.0  # tokens per second
        self._lock = threading.Lock()
        self._last_refill = _time.monotonic()

    def acquire(self) -> None:
        """Block until a request token is available, then consume one."""
        while True:
            with self._lock:
                now = _time.monotonic()
                self._tokens = min(
                    self._capacity,
                    self._tokens + (now - self._last_refill) * self._refill_rate,
                )
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._refill_rate
            _time.sleep(wait)

    async def async_acquire(self) -> None:
        """Async variant: yields control to the event loop instead of blocking."""
        while True:
            with self._lock:
                now = _time.monotonic()
                self._tokens = min(
                    self._capacity,
                    self._tokens + (now - self._last_refill) * self._refill_rate,
                )
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._refill_rate
            await asyncio.sleep(wait)


@dataclass
class ToolCall:
    """One native tool-call request emitted by the model.

    ``id`` carries the per-call identifier used by OpenAI and Anthropic to
    correlate results; it is an empty string for Ollama which has none.
    ``name`` uses the ``"plugin__method"`` convention (double-underscore
    separator) so plugin names that themselves contain underscores remain
    unambiguous.
    """
    id: str
    name: str
    args: dict
