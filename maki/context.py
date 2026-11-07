"""
Context-window awareness for local models.

A local server with a small context window does not fail when a prompt is too long: it drops the start of the
prompt and answers anyway (Ollama with ``num_ctx`` 2048 evaluated 1,027 of 7,043 tokens and returned a wrong
answer with ``done_reason: stop`` and no warning). This module gives the backends two things:

* :class:`PromptSizeCheck` compares the prompt tokens a server *reports* with what maki *sent* (estimated at
  four characters per token) and describes the shortfall, so a silent truncation becomes a warning or an error.
* :func:`choose_num_ctx` is the sticky ``num_ctx`` policy behind ``MakiLLama(num_ctx="auto")``: one value per
  model, big enough for the agent's budget, never changed afterwards (changing it makes Ollama reload the model,
  which took 7 to 10 seconds in measurements).

The estimate is rough: it is exact enough for English text and errs towards *not* flagging dense text.
"""

from __future__ import annotations

import contextvars
import json
import logging
import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .exceptions import MakiContextError  # noqa: F401  (re-exported: ``from maki.context import MakiContextError``)

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4
#: Context sizes ``num_ctx="auto"`` picks from, smallest first.
NUM_CTX_STEPS = (8192, 16384, 32768, 65536, 131072)
DEFAULT_NUM_CTX_CAP = 32768
_HEADROOM = 1.25  # the chosen window covers this multiple of the estimated need

TRUNCATION_MODES = ("warn", "error", "ignore")


def estimate_tokens(chars: int) -> int:
    """Rough token count for *chars* characters of text."""
    return max(0, chars) // CHARS_PER_TOKEN


def messages_chars(messages: Iterable[Any]) -> int:
    """Characters in chat messages given as dicts (string or block-list content, tool calls included)."""
    total = 0
    for message in messages:
        if not isinstance(message, dict):
            total += len(str(getattr(message, "content", "")))
            continue
        content = message.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):  # blocks: only text counts, an image's base64 is not prompt text
            total += sum(len(str(b.get("text", ""))) for b in content if isinstance(b, dict) and b.get("type") == "text")
        elif content is not None:
            total += len(json.dumps(content, default=str))
        if message.get("tool_calls"):
            total += len(json.dumps(message["tool_calls"], default=str))
    return total


@dataclass(frozen=True)
class ContextInfo:
    """What a server says about its context window.

    ``max_length`` is the longest context the model supports; ``loaded_length`` is the window the running
    server actually uses (smaller when the operator or a default limits it). Either may be unknown.
    """
    max_length: Optional[int] = None
    loaded_length: Optional[int] = None
    source: str = ""

    @property
    def effective(self) -> Optional[int]:
        """The window prompts really have to fit in."""
        return self.loaded_length or self.max_length


#: (characters sent, context window) of the request the current thread or asyncio task just sent
_last_request: contextvars.ContextVar = contextvars.ContextVar("maki_last_request", default=(0, None))


class PromptSizeCheck:
    """Notes what each request sent and judges the server's reported prompt tokens against it.

    The size of the last request is kept per thread *and per asyncio task* (a context variable), so
    concurrent callers of one backend, threaded or ``async_chat`` on one event loop, never judge a response
    against somebody else's request.
    """

    def __init__(self, ratio: float = 0.5, min_tokens: int = 512) -> None:
        self.ratio, self.min_tokens = ratio, min_tokens

    def note_request(self, chars: int, context_length: Optional[int] = None) -> None:
        """Record the size of the request this task just sent and the window it was sent against."""
        _last_request.set((chars, context_length))

    def last_request_chars(self) -> int:
        return _last_request.get()[0]

    def needs_window(self) -> bool:
        """True when the last request was big enough to judge but its context window is not known."""
        chars, window = _last_request.get()
        return window is None and estimate_tokens(chars) >= self.min_tokens

    def check(self, reported_tokens: int, context_length: Optional[int] = None) -> Optional[str]:
        """A description of the truncation when the report looks too small for the request, else None.

        Flags when the server evaluated less than ``ratio`` of what was sent, or when it evaluated about a
        whole window while more than a window was sent. Prompts under ``min_tokens`` are never flagged (the
        estimate is too coarse to judge them). The reported figure is the whole prompt, tokens served from a
        prompt cache included (Ollama's ``prompt_eval_count`` and OpenAI-style ``usage.prompt_tokens`` both
        work that way), so cached tokens are not added on top.
        """
        context_length = context_length or _last_request.get()[1]
        sent = estimate_tokens(self.last_request_chars())
        seen = max(0, reported_tokens)
        if sent < self.min_tokens or reported_tokens <= 0:
            return None
        cut_to_window = bool(context_length) and sent > context_length and seen >= 0.9 * context_length
        if seen >= self.ratio * sent and not cut_to_window:
            return None
        window = f" (context window {context_length})" if context_length else ""
        return (f"the server evaluated only {seen} prompt tokens but about {sent} were sent{window}; "
                "the start of the prompt was probably dropped silently. Raise the server's context size "
                "(num_ctx for Ollama, --ctx-size for llama.cpp, --max-model-len for vLLM) or shorten the prompt")


def choose_num_ctx(needed_tokens: int, *, model_max: Optional[int] = None, loaded: Optional[int] = None,
                   cap: int = DEFAULT_NUM_CTX_CAP) -> int:
    """The sticky ``num_ctx`` for a model that must hold about *needed_tokens* tokens.

    1. If the model is already loaded with a window that covers the need (with headroom), use it: no reload.
    2. Otherwise the smallest of 8k, 16k, 32k, ... that covers it, capped by *cap* and by the model's own
       maximum. When the need is bigger than that, the cap wins (the caller warns about truncation).
    """
    needed = max(1, math.ceil(needed_tokens * _HEADROOM))
    if loaded and loaded >= needed:
        return loaded
    limit = min(cap, model_max) if model_max else cap
    for step in NUM_CTX_STEPS:
        if step >= needed:
            return min(step, limit)
    return limit


class ContextAware:
    """Mixin for backends that can notice a silently truncated prompt.

    A backend calls :meth:`_init_size_check` once, :meth:`_note_request` with the messages it sends, and
    :meth:`_report_prompt_size` with the prompt tokens the server reports. ``on_truncation`` decides what a
    detected truncation does: ``"warn"`` (log it, the default), ``"error"`` (raise
    :class:`~maki.exceptions.MakiContextError`) or ``"ignore"``. ``last_truncation`` holds the latest finding.
    """

    on_truncation: str = "warn"
    last_truncation: Optional[str] = None
    _size_check_obj: Optional[PromptSizeCheck] = None

    @property
    def _size_check(self) -> PromptSizeCheck:
        # Created on first use so a backend built without __init__ (a test double, a subclass that skips
        # super().__init__) still works.
        if self._size_check_obj is None:
            self._size_check_obj = PromptSizeCheck()
        return self._size_check_obj

    def _init_size_check(self, on_truncation: str) -> None:
        if on_truncation not in TRUNCATION_MODES:
            raise ValueError(f"on_truncation must be one of {TRUNCATION_MODES}, got {on_truncation!r}")
        self.on_truncation = on_truncation
        self.last_truncation = None
        self._size_check_obj = PromptSizeCheck()

    def context_info(self) -> ContextInfo:
        """What the server says about its context window. The default knows nothing."""
        return ContextInfo()

    def _learn_window(self) -> Optional[int]:
        """The window a server is running with, asked for only when a big response could not be judged
        without it (no ``num_ctx`` was sent). Backends that can look it up cheaply override this."""
        return None

    def _note_request(self, messages: Iterable[Any], context_length: Optional[int] = None) -> None:
        if self.on_truncation != "ignore":
            self._size_check.note_request(messages_chars(messages), context_length)

    def _report_prompt_size(self, reported_tokens: int) -> None:
        if self.on_truncation == "ignore":
            return
        finding = self._size_check.check(reported_tokens)
        if finding is None and self._size_check.needs_window():
            # A server left on its default window cuts quietly at that window: 4,096 of 7,000 tokens is
            # above the ratio, so the window itself is needed to see it.
            window = self._learn_window()
            if window:
                finding = self._size_check.check(reported_tokens, window)
        self.last_truncation = finding
        if finding:
            logger.warning("%s: %s", type(self).__name__, finding)
            if self.on_truncation == "error":
                raise MakiContextError(finding)
