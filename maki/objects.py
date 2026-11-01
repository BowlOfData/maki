import asyncio
import json
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Optional
import threading
import time as _time


class BackendType(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HF = "huggingface"
    OPENROUTER = "openrouter"
    LLAMACPP = "llamacpp"
    VLLM = "vllm"


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


class ConversationMemory:
    """
    Token-budgeted conversation history shared by ChatSession and Agent.

    Stores Message objects in user/assistant pairs.  Oldest pairs are
    evicted when the total estimated token count exceeds ``token_budget``
    or the ``max_entries`` hard cap is reached.  Token estimate: len(text)//4
    (no tokenizer dependency).

    Always append messages in user/assistant order; the class does not
    validate role alternation, but mixing roles will produce garbled output
    from ``format_as_text``.
    """

    DEFAULT_TOKEN_BUDGET: int = 4096   # ≈ 16 K chars of conversation
    DEFAULT_MAX_ENTRIES: int = 200     # individual Message objects (= 100 turns)

    def __init__(
        self,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        if not isinstance(token_budget, int) or token_budget < 1:
            raise ValueError("token_budget must be a positive integer")
        if not isinstance(max_entries, int) or max_entries < 2:
            raise ValueError("max_entries must be an integer >= 2")
        self._token_budget = token_budget
        self._max_entries = max_entries
        self._messages: list = []   # list[Message]
        self._recent_turns: Optional[int] = None   # tiers: see configure_tiers()
        self._older_chars: Optional[int] = None
        self.dropped_pairs = 0      # turn pairs evicted since the last clear()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def token_budget(self) -> int:
        return self._token_budget

    @token_budget.setter
    def token_budget(self, value: int) -> None:
        if not isinstance(value, int) or value < 1:
            raise ValueError("token_budget must be a positive integer")
        self._token_budget = value
        self._trim()

    @property
    def max_entries(self) -> int:
        return self._max_entries

    @max_entries.setter
    def max_entries(self, value: int) -> None:
        if not isinstance(value, int) or value < 2:
            raise ValueError("max_entries must be an integer >= 2")
        self._max_entries = value
        self._trim()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def _is_older(self, index: int, count: int) -> bool:
        return self._recent_turns is not None and index < count - 2 * self._recent_turns

    def _total_tokens(self) -> int:
        """Estimated tokens of what the model would see, i.e. with older turns shortened."""
        count = len(self._messages)
        total = 0
        for i, m in enumerate(self._messages):
            length = len(m.content)
            if self._is_older(i, count):
                length = min(length, self._older_chars + self._CLIP_MARKER_CHARS)
            total += max(1, length // 4)
        return total

    def _trim(self) -> None:
        """Evict oldest pairs until both max_entries and token_budget are satisfied.

        We always keep at least the most recent pair (>=4 messages required before
        eviction starts), so a single large exchange is never silently dropped.
        """
        while len(self._messages) > self._max_entries and len(self._messages) >= 4:
            del self._messages[0]
            del self._messages[0]
            self.dropped_pairs += 1
        while self._total_tokens() > self._token_budget and len(self._messages) >= 4:
            del self._messages[0]
            del self._messages[0]
            self.dropped_pairs += 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, message: 'Message') -> None:
        """Append a message and evict oldest pairs if over budget or cap."""
        self._messages.append(message)
        self._trim()

    def messages(self) -> list:
        """Return a snapshot of all stored messages (list[Message])."""
        return list(self._messages)

    # Rough size of the "[... N characters omitted ...]" marker, so the budget counts it.
    _CLIP_MARKER_CHARS = 40

    def configure_tiers(self, recent_turns: Optional[int], older_chars: Optional[int]) -> None:
        """Keep the last ``recent_turns`` turns verbatim and shorten older ones to ``older_chars``.

        Both must be set, or both None (no tiers). Stored messages stay whole; only ``view()``
        and the budget see the shortened text.
        """
        if (recent_turns is None) != (older_chars is None):
            raise ValueError("recent_turns and older_chars must be set together")
        for value in (recent_turns, older_chars):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError("recent_turns and older_chars must be positive integers")
        self._recent_turns, self._older_chars = recent_turns, older_chars
        self._trim()

    def view(self) -> list:
        """The messages as the model should see them: older turns shortened, and a note on the
        first one when turns were dropped, so eviction is never silent."""
        count = len(self._messages)
        out = []
        for i, m in enumerate(self._messages):
            content = clip_text(m.content, self._older_chars) if self._is_older(i, count) else m.content
            out.append(Message(m.role, content, m.images))
        if self.dropped_pairs and out:
            out[0] = Message(out[0].role, f"[{self.dropped_pairs} earlier turn(s) dropped] {out[0].content}",
                             out[0].images)
        return out

    def format_as_text(self) -> str:
        """Format stored pairs as a text block for Agent stateful prompts."""
        messages = self.view()
        if not messages:
            return ""
        lines = []
        i = 0
        while i + 1 < len(messages):
            lines.append(f"Task: {messages[i].content}")
            lines.append(f"Response: {messages[i + 1].content}")
            i += 2
        return "\n\nPrior conversation:\n" + "\n".join(lines) if lines else ""

    def clear(self) -> None:
        self._messages.clear()
        self.dropped_pairs = 0

    def __len__(self) -> int:
        return len(self._messages)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_list(self) -> list:
        """Serialize to a JSON-compatible list of message dicts."""
        return [m.to_dict() for m in self._messages]

    @classmethod
    def from_list(
        cls,
        entries: list,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> 'ConversationMemory':
        """Restore from a list of message dicts produced by ``to_list()``."""
        mem = cls(token_budget=token_budget, max_entries=max_entries)
        for d in entries:
            mem._messages.append(Message.from_dict(d))
        return mem


def clip_text(text: str, limit: Optional[int]) -> str:
    """Keep the head and tail of ``text`` within ``limit`` characters, marking what was cut.

    Text at or under the limit (and any text when ``limit`` is None) is returned unchanged.
    ``limit`` counts the kept text; the marker (about 35 characters) comes on top. The marker states
    how many characters were omitted so the model knows the text is partial.
    """
    if limit is None or len(text) <= limit:
        return text
    omitted = len(text) - limit
    head = limit // 2
    tail = limit - head
    return f"{text[:head]}\n[... {omitted} characters omitted ...]\n{text[len(text) - tail:]}"


def clip_entry(entry: dict, limit: Optional[int]) -> dict:
    """Return ``entry`` with every long text or structured value clipped to ``limit`` characters."""
    if limit is None:
        return entry
    out = {}
    for key, value in entry.items():
        if value is None or isinstance(value, (bool, int, float)):
            out[key] = value
            continue
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        out[key] = value if len(text) <= limit else clip_text(text, limit)
    return out


@dataclass(frozen=True)
class MemoryPolicy:
    """Opt-in limits on what an agent keeps and sends. Every default reproduces today's behaviour.

    Attributes:
        history_mode: ``"text"`` pastes prior turns into the prompt; ``"messages"`` sends them as
            structured chat history when the backend supports it (falls back to text otherwise).
        token_budget: Estimated-token cap on the stateful conversation (None keeps the default).
        tool_result_chars: Clip each plugin result to this many characters, head and tail kept.
        history_entry_chars: Clip the result and context stored in ``task_history`` and
            ``reasoning_history`` to this many characters.
        pinned_chars: Budget for pinned facts (``Agent.remember(..., pin=True)``), which are put
            in the system message and never evicted. Pinning needs this set.
        recent_turns: Keep this many latest turns verbatim; older turns are shortened (needs
            ``older_turn_chars``).
        older_turn_chars: Shorten each older message to this many characters, head and tail kept.
    """
    history_mode: str = "text"
    token_budget: Optional[int] = None
    tool_result_chars: Optional[int] = None
    history_entry_chars: Optional[int] = None
    pinned_chars: Optional[int] = None
    recent_turns: Optional[int] = None
    older_turn_chars: Optional[int] = None

    HISTORY_MODES = ("text", "messages")

    def __post_init__(self) -> None:
        if self.history_mode not in self.HISTORY_MODES:
            raise ValueError(f"history_mode must be one of {self.HISTORY_MODES}, got {self.history_mode!r}")
        for name in ("token_budget", "tool_result_chars", "history_entry_chars", "pinned_chars", "recent_turns",
                     "older_turn_chars"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")
        if (self.recent_turns is None) != (self.older_turn_chars is None):
            raise ValueError("recent_turns and older_turn_chars must be set together")

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict) -> 'MemoryPolicy':
        """Build from a dict (e.g. a YAML section); unknown keys are an error, not ignored."""
        if not isinstance(data, dict):
            raise ValueError("memory policy must be a mapping")
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown memory policy keys: {sorted(unknown)}")
        return cls(**data)


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
    num_ctx: Optional[int] = None  # Ollama only: total context window (input + output tokens)

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
    reasoning: Optional[str] = None  # thinking text, when the backend returns it separately
    cached_tokens: int = 0  # prompt tokens served from the provider's prompt cache, when reported

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
            "reasoning": self.reasoning,
            "cached_tokens": self.cached_tokens,
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
            reasoning=data.get("reasoning"),
            cached_tokens=data.get("cached_tokens", 0),
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
