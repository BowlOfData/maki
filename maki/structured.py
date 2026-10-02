"""
Schema-constrained generation with pydantic validation and error-feedback retries.

The backend must accept ``response_format=<JSON schema dict>`` on ``chat()``
(``MakiLLama`` does, via Ollama structured outputs). Requires ``pydantic>=2``
(extra: ``structured``).

Example:
    from pydantic import BaseModel
    from maki import MakiLLama
    from maki.structured import generate_structured

    class Verdict(BaseModel):
        ok: bool
        reason: str

    result = generate_structured(MakiLLama(model="gemma4:26b"), Verdict, "Is 7 prime?")
    print(result.value.ok, result.attempts)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Generic, Optional, Type, TypeVar

from .exceptions import MakiValidationError
from .objects import GenerationConfig, LLMResponse, Message

log = logging.getLogger(__name__)

T = TypeVar("T")

_RETRY_TEMPLATE = (
    "Your previous reply was rejected because it did not satisfy the required schema.\n"
    "Validation errors:\n{errors}\n"
    "Reply again with ONLY a corrected JSON object that matches the schema."
)


class StructuredOutputError(MakiValidationError):
    """Raised when the model never produced schema-valid output within the retry budget."""

    def __init__(self, message: str, *, attempts: int, errors: list[str], last_raw: str) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.errors = errors
        self.last_raw = last_raw


@dataclass
class StructuredResult(Generic[T]):
    """A validated model instance plus how it was obtained."""

    value: T
    response: LLMResponse
    attempts: int
    errors: list[str] = field(default_factory=list)  # one entry per failed attempt


def _require_pydantic() -> Any:
    try:
        import pydantic  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError("generate_structured requires pydantic; install maki-framework[structured]") from exc
    return pydantic


def _format_errors(exc: Exception) -> str:
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            lines = []
            for e in errors():
                loc = ".".join(str(p) for p in e.get("loc", ())) or "<root>"
                lines.append(f"- {loc}: {e.get('msg', 'invalid')}")
            return "\n".join(lines)
        except Exception:
            pass
    return f"- {exc}"


def generate_structured(
    backend: Any,
    model_cls: Type[T],
    prompt: str,
    *,
    system: Optional[str] = None,
    history: Optional[list[Message]] = None,
    config: Optional[GenerationConfig] = None,
    max_retries: int = 2,
    think: Optional[bool] = False,
) -> StructuredResult[T]:
    """Generate and validate an instance of ``model_cls`` (a pydantic ``BaseModel``).

    On a validation failure the bad reply and the error list are appended to the
    conversation and the model is asked again, up to ``max_retries`` extra times.
    Raises :class:`StructuredOutputError` when the budget is exhausted. Transport
    errors (``MakiNetworkError`` etc.) propagate unchanged.

    ``think`` defaults to ``False``: reasoning traces can consume the whole token
    budget and leave the constrained ``content`` empty. Pass ``None`` to defer to
    the backend's own setting.
    """
    _require_pydantic()
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")

    schema = model_cls.model_json_schema()  # type: ignore[attr-defined]
    convo: list[Message] = list(history or [])
    current_prompt = prompt
    errors: list[str] = []
    raw = ""

    for attempt in range(1, max_retries + 2):
        response = backend.chat(
            current_prompt,
            history=convo or None,
            config=config,
            system=system,
            response_format=schema,
            think=think,
        )
        raw = response.content
        try:
            value = model_cls.model_validate_json(raw)  # type: ignore[attr-defined]
        except Exception as exc:  # pydantic.ValidationError or malformed JSON
            detail = _format_errors(exc)
            errors.append(detail)
            log.warning("structured output attempt %d/%d failed:\n%s", attempt, max_retries + 1, detail)
            convo = convo + [Message("user", current_prompt), Message("assistant", raw)]
            current_prompt = _RETRY_TEMPLATE.format(errors=detail)
            continue
        return StructuredResult(value=value, response=response, attempts=attempt, errors=errors)

    raise StructuredOutputError(
        f"no schema-valid output from {getattr(backend, 'model', '?')} after {max_retries + 1} attempt(s)",
        attempts=max_retries + 1,
        errors=errors,
        last_raw=raw,
    )


def schema_json(model_cls: Type[Any]) -> str:
    """Compact JSON-schema string for a pydantic model (handy for prompt budgeting)."""
    _require_pydantic()
    return json.dumps(model_cls.model_json_schema(), separators=(",", ":"))
