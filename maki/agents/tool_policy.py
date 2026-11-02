"""
How an agent runs the tool calls a model asks for: round limit, retries, parallelism.
"""

import random
from dataclasses import dataclass, field, fields
from typing import Optional, Tuple, Type

from ..exceptions import MakiNetworkError, MakiTimeoutError

# Transient failures worth retrying; a plugin raising ValueError or PermissionError is not.
TRANSIENT_ERRORS: Tuple[Type[BaseException], ...] = (
    MakiNetworkError, MakiTimeoutError, TimeoutError, ConnectionError)


@dataclass
class ToolPolicy:
    """Limits for the native tool-calling loop. Defaults reproduce the previous behaviour.

    Attributes:
        max_rounds: Model/tool round trips before a final tool-less answer is forced
            (``None`` keeps the framework default).
        retries: Extra attempts for a tool that raises one of ``retry_on`` (0 = none).
        retry_on: Exception types treated as transient.
        backoff_factor: Delay multiplier per retry (0 = constant ``initial_delay``).
        initial_delay: Seconds before the first retry.
        max_delay: Cap on the delay between retries.
        jitter: Add +-25% random jitter so concurrent retries do not synchronise.
        retry_dangerous: Also retry methods a plugin lists in ``DANGEROUS_METHODS``. Off by
            default: a write, upload or trade that timed out may already have happened, and
            running it again would repeat it.
        parallel: Run the tool calls of one model turn concurrently. Only enable it for
            plugins that are safe to call from several threads. A turn that includes a
            ``DANGEROUS_METHODS`` call runs sequentially, in the order the model gave.
        max_workers: Thread cap when ``parallel`` is on.
    """
    max_rounds: Optional[int] = None
    retries: int = 0
    retry_on: Tuple[Type[BaseException], ...] = field(default=TRANSIENT_ERRORS, repr=False)
    backoff_factor: float = 2.0
    initial_delay: float = 0.5
    max_delay: float = 10.0
    jitter: bool = True
    retry_dangerous: bool = False
    parallel: bool = False
    max_workers: int = 4

    def __post_init__(self) -> None:
        if isinstance(self.retry_on, type):
            self.retry_on = (self.retry_on,)
        self.retry_on = tuple(self.retry_on)
        if not all(isinstance(t, type) and issubclass(t, BaseException) for t in self.retry_on):
            raise ValueError("retry_on must be exception types")
        if self.max_rounds is not None and self.max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        if self.retries < 0 or self.max_workers < 1:
            raise ValueError("retries must be >= 0 and max_workers >= 1")
        if min(self.backoff_factor, self.initial_delay, self.max_delay) < 0:
            raise ValueError("backoff_factor, initial_delay and max_delay must be >= 0")

    def delay(self, retry_number: int) -> float:
        """Seconds to wait before retry *retry_number* (0-indexed)."""
        if self.backoff_factor == 0:
            delay = self.initial_delay
        else:
            delay = self.initial_delay * (self.backoff_factor ** retry_number)
        delay = min(delay, self.max_delay)
        if self.jitter and delay > 0:
            delay = max(0.0, delay + random.uniform(-0.25 * delay, 0.25 * delay))
        return delay

    def to_dict(self) -> dict:
        """Plain-data form. ``retry_on`` (exception classes) is not included, so a policy
        restored by ``from_dict`` uses the default transient errors."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "retry_on"}

    @classmethod
    def from_dict(cls, data: dict) -> "ToolPolicy":
        known = {f.name for f in fields(cls)} - {"retry_on"}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown tool_policy field(s): {', '.join(unknown)}")
        return cls(**data)
