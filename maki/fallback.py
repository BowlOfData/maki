"""
Try the next backend when one fails.

    from maki import FallbackBackend, init_backend

    backend = FallbackBackend([
        init_backend("anthropic:claude-sonnet-5-5"),
        "openai:gpt-4o",            # a spec string is built with init_backend
        "ollama:gemma4:26b",        # local last resort
    ])
    agent = Agent("assistant", backend)

A call goes to the first backend; when it fails with a transient error (network, timeout, a 408/429
status) the same call is repeated on the next one. Errors that no other model would fix (a 400, a
401, validation) are raised at once. When every backend failed, the last error is raised with its
original type, so ``ToolPolicy`` retries and callers still classify it correctly.

Native tool loops can switch backends in the middle of a turn. When every backend speaks typed messages
(``supports_typed_messages``, true for all built-in providers with native tools) the loop is already
provider-neutral: the next backend simply receives the same ``Message`` list, tool calls and results
included. For other backends the loop is carried in a neutral transcript instead: the backend that
produced a conversation continues it natively, and any other backend gets the earlier tool calls and
their results folded into the last user message as text. Tools run exactly once either way.
"""

import json
import logging
import threading
import time
from typing import Any, Callable, Generator, List, Optional, Sequence, Tuple, Union

from .backend import LLMBackend
from .exceptions import MakiAPIError, MakiNetworkError, MakiTimeoutError
from .objects import GenerationConfig, LLMResponse, Message, ToolCall
from .session import ChatSession

logger = logging.getLogger(__name__)

_RESULT_CHARS = 4000
_RETRYABLE_STATUS = (408, 429)

FallbackRule = Union[Callable[[BaseException], bool], Tuple[type, ...], type]


def default_fallback_on(exc: BaseException) -> bool:
    """True for failures another backend may not share: unreachable, timed out, rate limited."""
    if isinstance(exc, (MakiNetworkError, MakiTimeoutError)):
        return True
    return isinstance(exc, MakiAPIError) and exc.status_code in _RETRYABLE_STATUS


class _OnClasses:
    """``on=(SomeError, ...)`` as a rule; a class rather than a lambda so the backend still pickles."""

    def __init__(self, classes: Tuple[type, ...]):
        self.classes = classes

    def __call__(self, exc: BaseException) -> bool:
        return isinstance(exc, self.classes)


class _Transcript(list):
    """The wire messages of the backend that produced them, plus a neutral record of the tool exchange."""

    def __init__(self, wire: list, origin: LLMBackend, base: List[dict], steps: List[dict]):
        super().__init__(wire)
        self.origin, self.base, self.steps = origin, base, steps

    def flattened(self) -> List[dict]:
        """The conversation as plain role/content messages, tool exchange folded into the last user turn."""
        messages = [dict(m) for m in self.base]
        lines = ["Tool calls already made for this task (do not repeat them) and their results:"]
        for number, step in enumerate(self.steps, 1):
            for call in step["calls"]:
                result = step.get("results", {}).get(id(call), "(no result)")
                lines.append(f"[{number}] {call.name}({json.dumps(call.args, default=str)})")
                lines.append(f"    result: {str(result)[:_RESULT_CHARS]}")
        note = "\n".join(lines) + "\n\nContinue the task using these results."
        for message in reversed(messages):
            if message.get("role") == "user":
                message["content"] = f"{message.get('content', '')}\n\n{note}"
                break
        else:
            messages.append({"role": "user", "content": note})
        return messages


class FallbackBackend(LLMBackend):
    """An :class:`LLMBackend` that tries *backends* in order until one answers.

    Args:
        backends: Backend instances or ``"provider:model"`` specs, most preferred first.
        on: Which failures move on to the next backend: a tuple of exception classes or a
            ``callable(exc) -> bool``. Default: network errors, timeouts and 408/429 statuses.
        cooldown: Seconds a backend that just failed is tried last, so a dead primary does not cost
            its full timeout on every call (0 turns it off). When every backend is cooling down,
            all are tried in the given order.

    ``last_used`` is the backend that served the most recent call (None before the first).
    """

    def __init__(self, backends: Sequence[Union[LLMBackend, str]], *, on: Optional[FallbackRule] = None,
                 cooldown: float = 30.0):
        if isinstance(backends, (str, LLMBackend)) or not backends:
            raise ValueError("backends must be a non-empty list of backends or 'provider:model' specs")
        built: List[LLMBackend] = []
        for item in backends:
            if isinstance(item, str):
                from .backend_factory import init_backend
                item = init_backend(item)
            if not isinstance(item, LLMBackend):
                raise TypeError(f"Fallback entries must be LLMBackend instances or specs, got {type(item).__name__}")
            built.append(item)
        self.backends: List[LLMBackend] = built
        if isinstance(cooldown, bool) or not isinstance(cooldown, (int, float)) or cooldown < 0:
            raise ValueError("cooldown must be a number of seconds, 0 or more")
        self._on = self._rule(on)
        self.cooldown = float(cooldown)
        self._down_until: dict = {}  # id(backend) -> monotonic time until which it is tried last
        self._lock = threading.Lock()
        self.last_used: Optional[LLMBackend] = None

    def __getstate__(self) -> dict:  # a lock can be neither copied nor pickled; cooldowns key on id()
        state = self.__dict__.copy()
        state.pop("_lock", None)
        state["_down_until"] = {}
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    @staticmethod
    def _rule(on: Optional[FallbackRule]) -> Callable[[BaseException], bool]:
        if on is None:
            return default_fallback_on
        if isinstance(on, type):
            on = (on,)
        if isinstance(on, tuple):
            if not all(isinstance(t, type) and issubclass(t, BaseException) for t in on):
                raise TypeError("on= must hold exception classes")
            return _OnClasses(on)
        if callable(on):
            return on
        raise TypeError("on= must be exception classes or a callable")

    # -- shared attributes -------------------------------------------------------------------

    @property
    def model(self) -> str:  # type: ignore[override]
        return self.backends[0].model

    @property
    def temperature(self) -> float:  # type: ignore[override]
        return self.backends[0].temperature

    @property
    def supports_native_tools(self) -> bool:  # type: ignore[override]
        return any(b.supports_native_tools for b in self.backends)

    @property
    def supports_typed_messages(self) -> bool:  # type: ignore[override]
        # The typed tool loop runs only when every backend that can serve it speaks typed messages;
        # otherwise the agent keeps the transcript path below, which still switches backends.
        native = [b for b in self.backends if b.supports_native_tools]
        return bool(native) and all(b.supports_typed_messages for b in native)

    @property
    def supports_history(self) -> bool:  # type: ignore[override]
        # Structured history goes only to a backend that honours it, so every backend must.
        return all(b.supports_history for b in self.backends)

    def __repr__(self) -> str:
        return f"FallbackBackend([{', '.join(f'{type(b).__name__}({b.model})' for b in self.backends)}])"

    # -- trying backends -----------------------------------------------------------------------

    def _order(self, candidates: Sequence[LLMBackend]) -> List[LLMBackend]:
        """*candidates* with backends that failed within ``cooldown`` seconds moved to the end."""
        now = time.monotonic()
        with self._lock:
            cooling = {id(b) for b in candidates if self._down_until.get(id(b), 0.0) > now}
        if not cooling or len(cooling) == len(candidates):
            return list(candidates)
        return [b for b in candidates if id(b) not in cooling] + [b for b in candidates if id(b) in cooling]

    def _failed(self, backend: LLMBackend, exc: BaseException, more: bool) -> None:
        # A 429 that names a wait keeps that backend out of rotation at least that long.
        asked = getattr(exc, "retry_after", None)
        pause = max(self.cooldown, asked) if self.cooldown and isinstance(asked, (int, float)) else self.cooldown
        with self._lock:
            if pause:
                self._down_until[id(backend)] = time.monotonic() + pause
        logger.warning("%s(%s) failed (%s: %s)%s", type(backend).__name__, backend.model,
                       type(exc).__name__, exc, "; trying the next backend" if more else "")

    def _served(self, backend: LLMBackend) -> None:
        self.last_used = backend
        with self._lock:
            self._down_until.pop(id(backend), None)

    def _try_with(self, candidates: Sequence[LLMBackend], call: Callable[[LLMBackend], Any]) -> Tuple[LLMBackend, Any]:
        """Run *call* on each candidate in turn; returns ``(backend that answered, result)``."""
        ordered = self._order(candidates)
        last: Optional[BaseException] = None
        for index, backend in enumerate(ordered):
            try:
                result = call(backend)
            except Exception as exc:
                if not self._on(exc):
                    raise
                last = exc
                self._failed(backend, exc, index + 1 < len(ordered))
                continue
            self._served(backend)
            return backend, result
        assert last is not None
        raise last

    def _try(self, candidates: Sequence[LLMBackend], call: Callable[[LLMBackend], Any]) -> Any:
        return self._try_with(candidates, call)[1]

    def request(self, prompt: str) -> LLMResponse:
        return self._try(self.backends, lambda b: b.request(prompt))

    def chat(self, prompt: str, history: Optional[List[Message]] = None, config: Optional[GenerationConfig] = None,
             system: Optional[str] = None, images: Optional[List[str]] = None) -> LLMResponse:
        return self._try(self.backends, lambda b: b.chat(prompt, history=history, config=config,
                                                         system=system, images=images))

    def chat_collect(self, prompt: str, history: Optional[List[Message]] = None,
                     config: Optional[GenerationConfig] = None, system: Optional[str] = None,
                     images: Optional[List[str]] = None) -> LLMResponse:
        return self._try(self.backends, lambda b: b.chat_collect(prompt, history=history, config=config,
                                                                 system=system, images=images))

    def stream(self, prompt: str, history: Optional[List[Message]] = None, config: Optional[GenerationConfig] = None,
               system: Optional[str] = None) -> Generator[str, None, None]:
        """Stream from the first backend that starts; a failure after the first chunk is raised, not retried
        (the caller already holds part of the answer)."""
        last: Optional[BaseException] = None
        ordered = self._order(self.backends)
        for index, backend in enumerate(ordered):
            started = False
            try:
                for chunk in backend.stream(prompt, history=history, config=config, system=system):
                    started = True
                    yield chunk
            except NotImplementedError as exc:
                if started:
                    raise
                last = exc
                continue
            except Exception as exc:
                if started or not self._on(exc):
                    raise
                last = exc
                self._failed(backend, exc, index + 1 < len(ordered))
                continue
            self._served(backend)
            return
        assert last is not None
        raise last

    async def async_chat(self, prompt: str, history: Optional[List[Message]] = None,
                         config: Optional[GenerationConfig] = None, system: Optional[str] = None,
                         images: Optional[List[str]] = None) -> LLMResponse:
        """Async ``chat()`` on the first backend that has one and answers."""
        capable = [b for b in self.backends if callable(getattr(b, "async_chat", None))]
        if not capable:
            raise NotImplementedError("None of the fallback backends supports async_chat().")
        ordered = self._order(capable)
        last: Optional[BaseException] = None
        for index, backend in enumerate(ordered):
            try:
                result = await backend.async_chat(prompt,  # type: ignore[attr-defined]  # filtered to async-capable backends above history=history, config=config, system=system,
                                                  images=images)
            except Exception as exc:
                if not self._on(exc):
                    raise
                last = exc
                self._failed(backend, exc, index + 1 < len(ordered))
                continue
            self._served(backend)
            return result
        assert last is not None
        raise last

    def session(self, system: Optional[str] = None) -> ChatSession:
        """A multi-turn chat session over the fallback chain."""
        return ChatSession(llm=self, system=system)

    # -- native tool calling ---------------------------------------------------------------------

    def chat_turn(self, messages: List[Message], tool_specs: Optional[List[dict]] = None, *,
                  system: Optional[str] = None, config: Optional[GenerationConfig] = None,
                  tool_choice: Optional[str] = None) -> LLMResponse:
        """One typed round on the first backend that answers.

        The messages are provider-neutral, so a failure on any round (the first or one after tools ran)
        simply hands the same messages to the next backend: nothing is flattened or replayed.
        """
        typed = [b for b in self.backends if b.supports_typed_messages and (b.supports_native_tools or not tool_specs)]
        if not typed:
            raise NotImplementedError("None of the fallback backends supports typed messages for this turn.")
        return self._try(typed, lambda b: b.chat_turn(messages, tool_specs, system=system, config=config,
                                                      tool_choice=tool_choice))

    def hint_context(self, tokens: int) -> None:
        """Pass an agent's expected prompt size to every backend that sizes its own context window."""
        for backend in self.backends:
            hint = getattr(backend, "hint_context", None)
            if callable(hint):
                hint(tokens)

    def to_tool_schemas(self, tool_specs: List[dict]) -> List[dict]:
        """Keep the neutral specs; each backend translates them when it is called."""
        return list(tool_specs)

    def chat_with_tools(self, messages: List[dict], tools: List[dict], *, system: Optional[str] = None,
                        config: Optional[GenerationConfig] = None):
        if isinstance(messages, _Transcript):
            base, steps = messages.base, messages.steps
        else:
            base, steps = [dict(m) for m in messages], []

        def attempt(backend: LLMBackend):
            if isinstance(messages, _Transcript) and messages.origin is not backend:
                wire = messages.flattened()
            else:
                wire = list(messages)
            specs = backend.to_tool_schemas(tools) if tools else []
            return backend.chat_with_tools(wire, specs, system=system, config=config)

        native = [b for b in self.backends if b.supports_native_tools]
        if not native:
            raise NotImplementedError("None of the fallback backends supports native tool calling.")
        served, (response, calls, updated) = self._try_with(native, attempt)
        steps = list(steps)
        if calls:
            steps.append({"calls": list(calls)})
        return response, calls, _Transcript(updated, served, base, steps)

    def append_tool_results(self, messages: List[dict], results: List[Tuple[ToolCall, str]]) -> List[dict]:
        if not isinstance(messages, _Transcript):
            raise ValueError("append_tool_results() needs the messages returned by this backend's chat_with_tools()")
        updated = messages.origin.append_tool_results(list(messages), results)
        steps = [dict(s) for s in messages.steps]
        if steps:
            steps[-1]["results"] = {id(call): result for call, result in results}
        return _Transcript(updated, messages.origin, messages.base, steps)
