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

Native tool loops can switch backends in the middle of a turn. Tool schemas and tool messages differ
per provider, so the loop is carried in a neutral form: the backend that produced a conversation
continues it natively, and any other backend gets the earlier tool calls and their results folded into
the last user message as text. Tools run exactly once either way.
"""

import json
import logging
from typing import Any, Callable, Generator, List, Optional, Sequence, Tuple, Union

from .backend import LLMBackend
from .exceptions import MakiAPIError, MakiNetworkError, MakiTimeoutError
from .objects import GenerationConfig, LLMResponse, Message, ToolCall

logger = logging.getLogger(__name__)

_RESULT_CHARS = 4000
_RETRYABLE_STATUS = (408, 429)

FallbackRule = Union[Callable[[BaseException], bool], Tuple[type, ...], type]


def default_fallback_on(exc: BaseException) -> bool:
    """True for failures another backend may not share: unreachable, timed out, rate limited."""
    if isinstance(exc, (MakiNetworkError, MakiTimeoutError)):
        return True
    return isinstance(exc, MakiAPIError) and exc.status_code in _RETRYABLE_STATUS


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

    ``last_used`` is the backend that served the most recent call (None before the first).
    """

    def __init__(self, backends: Sequence[Union[LLMBackend, str]], *, on: Optional[FallbackRule] = None):
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
        self._on = self._rule(on)
        self.last_used: Optional[LLMBackend] = None

    @staticmethod
    def _rule(on: Optional[FallbackRule]) -> Callable[[BaseException], bool]:
        if on is None:
            return default_fallback_on
        if isinstance(on, type):
            on = (on,)
        if isinstance(on, tuple):
            if not all(isinstance(t, type) and issubclass(t, BaseException) for t in on):
                raise TypeError("on= must hold exception classes")
            return lambda exc: isinstance(exc, on)
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
    def supports_history(self) -> bool:  # type: ignore[override]
        # Structured history goes only to a backend that honours it, so every backend must.
        return all(b.supports_history for b in self.backends)

    def __repr__(self) -> str:
        return f"FallbackBackend([{', '.join(f'{type(b).__name__}({b.model})' for b in self.backends)}])"

    # -- trying backends -----------------------------------------------------------------------

    def _try(self, candidates: Sequence[LLMBackend], call: Callable[[LLMBackend], Any]) -> Any:
        last: Optional[BaseException] = None
        for index, backend in enumerate(candidates):
            try:
                result = call(backend)
            except Exception as exc:
                if not self._on(exc):
                    raise
                last = exc
                more = index + 1 < len(candidates)
                logger.warning("%s(%s) failed (%s: %s)%s", type(backend).__name__, backend.model,
                               type(exc).__name__, exc, "; trying the next backend" if more else "")
                continue
            self.last_used = backend
            return result
        assert last is not None
        raise last

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
        for backend in self.backends:
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
                logger.warning("%s(%s) failed before streaming (%s: %s); trying the next backend",
                               type(backend).__name__, backend.model, type(exc).__name__, exc)
                continue
            self.last_used = backend
            return
        assert last is not None
        raise last

    # -- native tool calling ---------------------------------------------------------------------

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
        response, calls, updated = self._try(native, attempt)
        steps = list(steps)
        if calls:
            steps.append({"calls": list(calls)})
        return response, calls, _Transcript(updated, self.last_used, base, steps)

    def append_tool_results(self, messages: List[dict], results: List[Tuple[ToolCall, str]]) -> List[dict]:
        if not isinstance(messages, _Transcript):
            raise ValueError("append_tool_results() needs the messages returned by this backend's chat_with_tools()")
        updated = messages.origin.append_tool_results(list(messages), results)
        steps = [dict(s) for s in messages.steps]
        if steps:
            steps[-1]["results"] = {id(call): result for call, result in results}
        return _Transcript(updated, messages.origin, messages.base, steps)
