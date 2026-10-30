"""Token usage counting for LLM backends.

``attach_usage(backend)`` makes the backend instance's ``chat``, ``chat_collect`` and
``chat_with_tools`` add every :class:`~maki.objects.LLMResponse` they return to
``backend.usage``. It is idempotent, leaves anything that is not a real bound method (test
doubles, mocks) untouched, and keeps the backend copyable and picklable.
"""

import threading
from typing import Any
import inspect

_WRAPPED = ("chat", "chat_collect", "chat_with_tools")
_nesting = threading.local()  # depth of counted calls on this thread: only the outermost one records


class Usage:
    """Running token totals. ``snapshot()`` + subtraction gives the usage of one stretch of work."""

    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0, total_tokens: int = 0, calls: int = 0):
        self.prompt_tokens, self.completion_tokens = prompt_tokens, completion_tokens
        self.total_tokens, self.calls = total_tokens, calls
        self._lock = threading.Lock()
        self._by_thread: dict = {}  # thread id -> tokens, so concurrent users of one backend can be told apart

    def add(self, prompt: int, completion: int, total: int) -> None:
        with self._lock:
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            spent = total or (prompt + completion)
            self.total_tokens += spent
            self.calls += 1
            me = threading.get_ident()
            self._by_thread[me] = self._by_thread.get(me, 0) + spent

    def thread_tokens(self) -> int:
        """Tokens spent by calls made on the current thread (a stretch of work's usage, even when
        other threads share the backend)."""
        with self._lock:
            return self._by_thread.get(threading.get_ident(), 0)

    def snapshot(self) -> "Usage":
        with self._lock:
            return Usage(self.prompt_tokens, self.completion_tokens, self.total_tokens, self.calls)

    def __sub__(self, other: "Usage") -> "Usage":
        return Usage(self.prompt_tokens - other.prompt_tokens, self.completion_tokens - other.completion_tokens,
                     self.total_tokens - other.total_tokens, self.calls - other.calls)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Usage) and self.to_dict() == other.to_dict()

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"Usage({self.to_dict()})"

    def __getstate__(self) -> dict:  # a lock can be neither copied nor pickled
        return self.to_dict()

    def __setstate__(self, state: dict) -> None:
        self.__init__(**state)

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


class _CountedMethod:
    """Stands in for a backend method and records the tokens of its response.

    A module-level class (not a closure) holding the backend and a method name, so a counted
    backend still deep-copies and pickles; the clone then counts on its own ``usage``.
    """

    def __init__(self, backend: Any, name: str):
        self.backend, self.name = backend, name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        depth = getattr(_nesting, "depth", 0)
        _nesting.depth = depth + 1
        try:
            result = getattr(type(self.backend), self.name)(self.backend, *args, **kwargs)
        finally:
            _nesting.depth = depth
        response = _response_in(result) if depth == 0 else None  # chat_collect() may call chat()
        if response is not None:
            total = getattr(response, "total_tokens", 0)
            self.backend.usage.add(response.prompt_tokens, response.completion_tokens,
                                   total if isinstance(total, int) else 0)
        return result


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
        if inspect.ismethod(getattr(backend, name, None)):
            setattr(backend, name, _CountedMethod(backend, name))
    return usage
