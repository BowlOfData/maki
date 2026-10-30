"""Token usage counting for LLM backends.

``attach_usage(backend)`` wraps the backend instance's ``chat``, ``chat_collect`` and
``chat_with_tools`` so every :class:`~maki.objects.LLMResponse` they return is added to
``backend.usage``. It is idempotent, and it leaves anything that is not a real bound method
(test doubles, mocks) untouched.
"""

import functools
import inspect
import threading
from dataclasses import dataclass, field
from typing import Any

_WRAPPED = ("chat", "chat_collect", "chat_with_tools")
_nesting = threading.local()  # depth of counted calls on this thread: only the outermost one records


@dataclass
class Usage:
    """Running token totals. ``snapshot()`` + subtraction gives the usage of one stretch of work."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, prompt: int, completion: int, total: int) -> None:
        with self._lock:
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.total_tokens += total or (prompt + completion)
            self.calls += 1

    def snapshot(self) -> "Usage":
        with self._lock:
            return Usage(self.prompt_tokens, self.completion_tokens, self.total_tokens, self.calls)

    def __sub__(self, other: "Usage") -> "Usage":
        return Usage(self.prompt_tokens - other.prompt_tokens, self.completion_tokens - other.completion_tokens,
                     self.total_tokens - other.total_tokens, self.calls - other.calls)

    def to_dict(self) -> dict:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens, "calls": self.calls}


def _response_in(result: Any) -> Any:
    """The LLMResponse-like object in a backend return value (it may be the first of a tuple)."""
    items = result if isinstance(result, tuple) else (result,)
    for item in items:
        if all(isinstance(getattr(item, a, None), int) for a in ("prompt_tokens", "completion_tokens")):
            return item
    return None


def attach_usage(backend: Any) -> Usage:
    """Start counting tokens for ``backend`` (once) and return its :class:`Usage`."""
    existing = getattr(backend, "usage", None)
    if isinstance(existing, Usage):
        return existing
    usage = Usage()
    try:
        backend.usage = usage
    except AttributeError:
        return usage
    for name in _WRAPPED:
        original = getattr(backend, name, None)
        if not inspect.ismethod(original):
            continue

        def make(fn):
            @functools.wraps(fn)
            def counted(*args, **kwargs):
                depth = getattr(_nesting, "depth", 0)
                _nesting.depth = depth + 1
                try:
                    result = fn(*args, **kwargs)
                finally:
                    _nesting.depth = depth
                response = _response_in(result) if depth == 0 else None  # chat_collect() may call chat()
                if response is not None:
                    total = getattr(response, "total_tokens", 0)
                    usage.add(response.prompt_tokens, response.completion_tokens,
                              total if isinstance(total, int) else 0)
                return result
            return counted

        setattr(backend, name, make(original))
    return usage
