"""
Observer hooks for agent runs.

Subclass :class:`AgentCallback` and override the events you care about, then pass
``Agent(callbacks=[...])``. Every method is a no-op by default. A callback that raises is
logged and skipped: observers must never break the run they observe.
"""

import logging
from typing import Any, Iterable

logger = logging.getLogger(__name__)


class AgentCallback:
    """Base class for agent observers. All hooks are optional."""

    def on_task_start(self, agent: Any, task: str) -> None: ...

    def on_task_end(self, agent: Any, task: str, result: str, elapsed: float) -> None: ...

    def on_task_error(self, agent: Any, task: str, error: BaseException, elapsed: float) -> None: ...

    def on_model_start(self, agent: Any, messages: list) -> None:
        """A model round is about to run. *messages* is a copy of the conversation: typed ``Message``
        objects on backends with ``supports_typed_messages``, the provider's own dicts otherwise."""

    def on_model_end(self, agent: Any, response: Any, tool_calls: Any) -> None:
        """A model round finished. *tool_calls* lists the tools it asked for (None when it answered).
        *response* carries the token counts; the provider-dict path passes None on a tool round."""

    def on_tool_start(self, agent: Any, call: Any) -> None: ...

    def on_tool_end(self, agent: Any, call: Any, result: str, elapsed: float) -> None: ...

    def on_tool_error(self, agent: Any, call: Any, error: BaseException, elapsed: float) -> None: ...

    def on_tool_retry(self, agent: Any, call: Any, error: BaseException, attempt: int, delay: float) -> None: ...


class LoggingCallback(AgentCallback):
    """Writes one log line per event; a minimal example and a handy debugging aid."""

    def __init__(self, level: int = logging.INFO):
        self.level = level

    def on_task_start(self, agent, task):
        logger.log(self.level, "[%s] task start: %.80s", agent.name, task)

    def on_task_end(self, agent, task, result, elapsed):
        logger.log(self.level, "[%s] task done in %.2fs", agent.name, elapsed)

    def on_task_error(self, agent, task, error, elapsed):
        logger.log(self.level, "[%s] task failed in %.2fs: %s", agent.name, elapsed, error)

    def on_tool_start(self, agent, call):
        logger.log(self.level, "[%s] tool start: %s %s", agent.name, call.name, call.args)

    def on_tool_end(self, agent, call, result, elapsed):
        logger.log(self.level, "[%s] tool %s done in %.2fs", agent.name, call.name, elapsed)

    def on_tool_error(self, agent, call, error, elapsed):
        logger.log(self.level, "[%s] tool %s failed in %.2fs: %s", agent.name, call.name, elapsed, error)

    def on_tool_retry(self, agent, call, error, attempt, delay):
        logger.log(self.level, "[%s] tool %s retry %d in %.2fs: %s", agent.name, call.name, attempt, delay, error)


def fire(callbacks: Iterable[AgentCallback], event: str, *args: Any) -> None:
    """Call ``callback.<event>(*args)`` on each callback, isolating failures."""
    for callback in callbacks or ():
        handler = getattr(callback, event, None)
        if handler is None:
            continue
        try:
            handler(*args)
        except Exception:
            logger.warning("Callback %s.%s raised", type(callback).__name__, event, exc_info=True)
