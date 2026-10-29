"""
Agent loops: assign an agent a goal plus requirements and iterate until a judge confirms
every requirement is met, within hard limits.

Each iteration the agent works, the requirements are checked (user-written commands and
callables, then one batched judge call for the rest), and the loop stops on success, an
iteration or wall-clock cap, a judge BLOCK, a stall, an agent error, or a human STOP.

Progress is the number of requirements passing. An iteration *improves* when more pass than
in the previous iteration; the stall counter resets on any improvement (including a
regressed requirement recovering) and the loop stops after ``stall_limit`` consecutive
iterations without one. Iterations whose judge call failed say nothing about progress and
are skipped by the stall counter. Regressions are logged; the iteration cap is the backstop
against a flapping judge.

Context modes:

* ``"shared"`` (default): one conversation. The agent must be ``stateful=True``.
* ``"fresh"``: every iteration starts clean; the prompt carries the goal, the current
  artifact, the failing feedback and the scratchpad notes.

Command checks are not sandboxed (see :mod:`maki.agents.loop_judge`). Tools that write
files need ``Agent(allow_dangerous_tools=True)``.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .loop_judge import CheckResult, _last_json_object, judge_requirements, run_command_check
from .loop_scratchpad import LoopStore, ReloadReport, Scratchpad, default_loop_dir

logger = logging.getLogger(__name__)

_NOTES_HINT = ("Optionally end your reply with a line `NOTES:` followed by notes to your future self; "
               "everything after that line is kept as notes and is not part of the result.")

CheckFn = Callable[[str], Union[bool, CheckResult, Tuple[bool, str]]]


@dataclass(frozen=True)
class Requirement:
    """One thing the finished work must satisfy.

    Exactly one way of checking applies: ``command`` (argv list, exit code 0 = pass),
    ``check`` (callable taking the artifact text), or neither (judged by the judge agent).
    Commands and callables may only be written by the user, never by a drafting agent.

    source: ``"user"`` (written by the user), ``"draft"`` (proposed by an agent; the loop
            refuses to run it until :func:`approve_requirements` freezes it as
            ``"approved"``).
    """
    id: str
    text: str
    command: Optional[Tuple[str, ...]] = None
    check: Optional[CheckFn] = None
    source: str = "user"

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Requirement.id must be a non-empty string")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Requirement.text must be a non-empty string")
        if self.source not in ("user", "draft", "approved"):
            raise ValueError("Requirement.source must be 'user', 'draft' or 'approved'")
        if self.command is not None and self.check is not None:
            raise ValueError("a Requirement takes a command or a check, not both")
        if self.command is not None:
            if isinstance(self.command, (str, bytes)) or not self.command \
                    or not all(isinstance(a, str) for a in self.command):
                raise ValueError("Requirement.command must be a non-empty list of strings (no shell)")
            object.__setattr__(self, "command", tuple(self.command))
        if self.check is not None and not callable(self.check):
            raise ValueError("Requirement.check must be callable")
        if self.source == "draft" and (self.command is not None or self.check is not None):
            raise ValueError("a drafted requirement cannot carry a command or check")

    @property
    def kind(self) -> str:
        if self.command is not None:
            return "command"
        return "callable" if self.check is not None else "agent"


class StopReason(Enum):
    SUCCEEDED = "succeeded"
    MAX_ITERATIONS = "max_iterations"
    TIMEOUT = "timeout"
    STALLED = "stalled"
    BLOCKED = "blocked"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass(frozen=True)
class Decision:
    """A human's answer from the ``on_iteration`` hook: keep going (optionally with a note
    the agent must address next iteration) or stop."""
    stop: bool = False
    note: str = ""

    @classmethod
    def proceed(cls, note: str = "") -> "Decision":
        return cls(False, note)


Decision.CONTINUE = Decision()
Decision.STOP = Decision(stop=True)


@dataclass
class LoopSpec:
    """Everything that defines a loop run.

    max_seconds:  wall-clock cap, checked between iterations (``None`` = no cap).
    stall_limit:  consecutive non-improving iterations before stopping (default 3).
    context:      ``"shared"`` or ``"fresh"``.
    use_plugins:  let the agent call its plugins (tools) while working.
    artifact_fn:  returns the text the judge should inspect besides the agent's reply
                  (for code: a diff or the file contents).
    cwd:          working directory for command checks (default: the current directory).
    loop_dir:     where the scratchpad and checkpoint live (default ``~/.maki/loops/<id>``).
    """
    goal: str
    requirements: Sequence[Requirement]
    max_iterations: int = 10
    max_seconds: Optional[float] = None
    stall_limit: int = 3
    context: str = "shared"
    use_plugins: bool = False
    artifact_fn: Optional[Callable[[], str]] = None
    cwd: Optional[str] = None
    loop_dir: Optional[str] = None
    loop_id: Optional[str] = None
    command_timeout: float = 120.0

    def __post_init__(self) -> None:
        if not isinstance(self.goal, str) or not self.goal.strip():
            raise ValueError("LoopSpec.goal must be a non-empty string")
        self.requirements = tuple(self.requirements)
        if not self.requirements or not all(isinstance(r, Requirement) for r in self.requirements):
            raise ValueError("LoopSpec.requirements must be a non-empty list of Requirement")
        ids = [r.id for r in self.requirements]
        if len(set(ids)) != len(ids):
            raise ValueError("Requirement ids must be unique")
        for name in ("max_iterations", "stall_limit"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"LoopSpec.{name} must be an integer >= 1")
        if self.max_seconds is not None and not self.max_seconds > 0:
            raise ValueError("LoopSpec.max_seconds must be > 0")
        if self.context not in ("shared", "fresh"):
            raise ValueError("LoopSpec.context must be 'shared' or 'fresh'")
        if self.artifact_fn is not None and not callable(self.artifact_fn):
            raise ValueError("LoopSpec.artifact_fn must be callable")


class StallTracker:
    """Counts consecutive iterations without improvement.

    An iteration improves when its passing count exceeds the previous counted iteration's
    (the start is 0). Iterations with ``passing=None`` (judge error) are ignored.
    """

    def __init__(self, limit: int):
        self.limit = limit
        self.previous = 0
        self.streak = 0

    def update(self, passing: Optional[int]) -> bool:
        """Record an iteration; return True if it improved on the previous one."""
        if passing is None:
            return False
        improved = passing > self.previous
        self.streak = 0 if improved else self.streak + 1
        self.previous = passing
        return improved

    @property
    def stalled(self) -> bool:
        return self.streak >= self.limit


@dataclass
class LoopEvent:
    iteration: int
    results: Dict[str, CheckResult]
    passing: Optional[int]
    total: int
    improved: bool
    stall_streak: int
    regressions: List[str]
    judge_error: bool
    artifact: str
    reply: str
    elapsed_seconds: float
    scratchpad: ReloadReport = field(default_factory=ReloadReport)

    def summary(self) -> str:
        failing = [k for k, v in self.results.items() if not v.passed]
        return (f"iteration {self.iteration}: "
                f"{'judge error' if self.judge_error else f'{self.passing}/{self.total} passing'}; "
                f"improved={self.improved}; stall={self.stall_streak}; failing={failing}")


@dataclass
class LoopResult:
    loop_id: str
    stop_reason: StopReason
    iterations: int
    events: List[LoopEvent]
    artifact: str
    scratchpad_path: str
    final_results: Dict[str, CheckResult] = field(default_factory=dict)
    error: str = ""
    detail: str = ""

    @property
    def succeeded(self) -> bool:
        return self.stop_reason is StopReason.SUCCEEDED


OnIteration = Callable[[Optional[LoopEvent]], Optional[Decision]]


def approve_requirements(requirements: Sequence[Requirement],
                         approver: Callable[[Sequence[Requirement]], bool]) -> Tuple[Requirement, ...]:
    """Show every requirement (commands in full) to ``approver``; on yes, freeze drafts.

    Returns the requirements with drafts re-marked ``"approved"``.

    Raises:
        ValueError: if the approver declines.
    """
    reqs = tuple(requirements)
    if not approver(reqs):
        raise ValueError("requirements were not approved")
    return tuple(replace(r, source="approved") if r.source == "draft" else r for r in reqs)


def draft_requirements(agent: Any, goal: str, max_items: int = 8) -> Tuple[Requirement, ...]:
    """Ask ``agent`` to propose agent-judged requirements for ``goal``.

    The proposals come back as ``source="draft"`` and cannot be run until approved with
    :func:`approve_requirements`. A drafter can never produce commands or callables.

    Raises:
        ValueError: if the reply contains no usable requirements.
    """
    prompt = (f"Goal:\n{goal}\n\nPropose up to {max_items} concrete, independently checkable requirements "
              'that finished work must satisfy. Reply with ONLY JSON: {"requirements": [{"text": "..."}]}')
    raw = agent.execute_task_with_retry(prompt)
    obj = _last_json_object(getattr(raw, "text", raw) if not isinstance(raw, str) else raw, "requirements")
    items = obj.get("requirements") if obj else None
    texts = [i.get("text") for i in items if isinstance(i, dict) and isinstance(i.get("text"), str)
             and i["text"].strip()] if isinstance(items, list) else []
    if not texts:
        raise ValueError("the drafting agent returned no usable requirements")
    return tuple(Requirement(f"d{n}", t.strip(), source="draft") for n, t in enumerate(texts[:max_items], 1))


def _split_notes(reply: str) -> Tuple[str, str]:
    marker = "\nNOTES:"
    text = "\n" + reply
    idx = text.rfind(marker)
    if idx == -1:
        return reply, ""
    return text[:idx].lstrip("\n"), text[idx + len(marker):].strip()


def _run_check(req: Requirement, artifact: str, cwd: str, timeout: float) -> CheckResult:
    if req.command is not None:
        return run_command_check(req.command, cwd=cwd, timeout=timeout)
    try:
        out = req.check(artifact)  # type: ignore[misc]
    except Exception as e:  # noqa: BLE001 - fail closed
        return CheckResult(False, f"check raised {type(e).__name__}: {e}")
    if isinstance(out, CheckResult):
        return out
    if isinstance(out, tuple) and len(out) == 2:
        return CheckResult(out[0] is True, str(out[1]))
    return CheckResult(out is True, "")


def _requirement_listing(reqs: Sequence[Requirement]) -> str:
    return "\n".join(f"- {r.id}: {r.text}" for r in reqs)


def _failing_text(reqs: Sequence[Requirement], results: Dict[str, CheckResult]) -> str:
    return "\n".join(f"- {r.id} ({r.text}): {results[r.id].feedback or 'not met'}"
                     for r in reqs if not results[r.id].passed)


def _build_prompt(spec: LoopSpec, iteration: int, pad: Scratchpad, artifact: str, note: str) -> str:
    if iteration == 1:
        parts = [f"Goal:\n{spec.goal}", f"Requirements (all must be met):\n{_requirement_listing(spec.requirements)}"]
        if note:
            parts.append(f"Reviewer note: {note}")
        parts.append(_NOTES_HINT)
        return "\n\n".join(parts)
    if spec.context == "shared":
        parts = [f"Not all requirements are met yet. Failing requirements and feedback:\n{pad.feedback}"]
    else:
        parts = [f"Goal:\n{spec.goal}",
                 f"Requirements (all must be met):\n{_requirement_listing(spec.requirements)}",
                 f"Current artifact:\n{artifact}",
                 f"Failing requirements and feedback:\n{pad.feedback}",
                 f"Your notes so far:\n{pad.notes or '(none)'}"]
    if note:
        parts.append(f"Reviewer note (address this): {note}")
    parts.append("Produce the complete corrected result.")
    parts.append(_NOTES_HINT)
    return "\n\n".join(parts)


def _validate_start(agent: Any, judge: Any, spec: LoopSpec) -> None:
    for r in spec.requirements:
        if r.source == "draft":
            raise ValueError(f"requirement '{r.id}' is an unapproved draft; use approve_requirements() first")
    if spec.context == "shared" and not getattr(agent, "stateful", False):
        raise ValueError("context='shared' needs Agent(stateful=True); use context='fresh' for a stateless agent")
    if getattr(judge, "stateful", False):
        raise ValueError("the loop judge must not be stateful")
    if spec.use_plugins and not getattr(agent, "allow_dangerous_tools", False):
        for name, plugin in (getattr(agent, "plugins", None) or {}).items():
            if getattr(plugin, "DANGEROUS_METHODS", None):
                raise ValueError(f"plugin '{name}' has dangerous methods; create the agent with "
                                 "allow_dangerous_tools=True to let a loop use them")


def _judged_artifact(spec: LoopSpec, body: str) -> str:
    if spec.artifact_fn is None:
        return body
    try:
        extra = str(spec.artifact_fn())
    except Exception as e:  # noqa: BLE001
        extra = f"(artifact_fn failed: {type(e).__name__}: {e})"
    return f"Agent reply:\n{body}\n\nArtifact:\n{extra}"


def _check_requirements(judge: Any, spec: LoopSpec, artifact: str,
                        cwd: str) -> Tuple[Dict[str, CheckResult], bool, str]:
    """Run command/callable checks, then one batched judge call for the rest.

    Returns ``(results by requirement id, judge_error, blocked reason)``.
    """
    results: Dict[str, CheckResult] = {}
    evidence: List[str] = []
    for r in spec.requirements:
        if r.kind == "agent":
            continue
        results[r.id] = _run_check(r, artifact, cwd, spec.command_timeout)
        if r.command is not None:
            evidence.append(f"{r.id} `{' '.join(r.command)}`: "
                            f"{'passed' if results[r.id].passed else 'FAILED'}\n{results[r.id].feedback}")
    agent_reqs = [r for r in spec.requirements if r.kind == "agent"]
    judge_error, blocked = False, ""
    if agent_reqs:
        outcome = judge_requirements(judge, spec.goal, [(r.id, r.text) for r in agent_reqs], artifact,
                                     "\n\n".join(evidence))
        results.update(outcome.results)
        judge_error, blocked = outcome.error, outcome.blocked
    return {r.id: results[r.id] for r in spec.requirements}, judge_error, blocked


def _after_iteration_stop(spec: LoopSpec, iteration: int, passing: Optional[int], total: int, blocked: str,
                          tracker: StallTracker) -> Optional[Tuple[StopReason, str]]:
    """Return ``(reason, detail)`` if the loop must stop after this iteration, else None."""
    if blocked:
        return StopReason.BLOCKED, blocked
    if passing == total:
        return StopReason.SUCCEEDED, ""
    if tracker.stalled:
        return StopReason.STALLED, f"no improvement for {tracker.streak} iterations"
    if iteration == spec.max_iterations:
        return StopReason.MAX_ITERATIONS, ""
    return None


def run_loop(agent: Any, judge: Any, spec: LoopSpec, on_iteration: Optional[OnIteration] = None,
             confirm_requirements: Optional[Callable[[Sequence[Requirement]], bool]] = None,
             clock: Callable[[], float] = time.monotonic) -> LoopResult:
    """Run ``agent`` on ``spec`` until the requirements are met or a limit is hit.

    Args:
        agent: the maker (``stateful=True`` for ``context="shared"``).
        judge: a separate, stateless agent that judges the requirements without a
            command or callable. Not needed when every requirement has one, but must
            still be passed.
        on_iteration: called with ``None`` before the first iteration and with each
            :class:`LoopEvent` before the next one; return a :class:`Decision`
            (``None`` = continue). A note on a continue decision is shown to the agent.
        confirm_requirements: called once with every requirement (commands in full)
            before anything runs; return False to cancel.

    Raises:
        ValueError: for an invalid setup (unapproved drafts, stateless agent in shared
            mode, dangerous tools not allowed, ...).
    """
    _validate_start(agent, judge, spec)
    loop_id = spec.loop_id or f"loop-{uuid.uuid4().hex[:8]}"
    directory = spec.loop_dir or default_loop_dir(loop_id)
    pad = Scratchpad(os.path.join(directory, "scratchpad.md"), loop_id)
    store = LoopStore(directory)
    cwd = os.path.realpath(spec.cwd or os.getcwd())
    reqs = spec.requirements
    events: List[LoopEvent] = []
    result = LoopResult(loop_id, StopReason.STOPPED, 0, events, "", pad.path)

    def finish(reason: StopReason, detail: str = "", error: str = "") -> LoopResult:
        result.stop_reason, result.detail, result.error = reason, detail, error
        pad.append_engine(f"stopped: {reason.value}{(' - ' + detail) if detail else ''}")
        pad.flush()
        _checkpoint(store, loop_id, spec, result, tracker, reason.value)
        return result

    tracker = StallTracker(spec.stall_limit)
    if confirm_requirements is not None and not confirm_requirements(reqs):
        return finish(StopReason.STOPPED, "requirements not confirmed")
    pad.flush()

    note = ""
    decision = on_iteration(None) if on_iteration else None
    if decision is not None and decision.stop:
        return finish(StopReason.STOPPED, "stopped before the first iteration")
    note = decision.note if decision else ""

    start = clock()
    artifact = ""
    prev_passed: set = set()
    for iteration in range(1, spec.max_iterations + 1):
        if spec.max_seconds is not None and clock() - start >= spec.max_seconds:
            return finish(StopReason.TIMEOUT, f"wall-clock cap of {spec.max_seconds}s reached")
        iter_start = clock()
        report = pad.reload()
        if report.engine_tampered:
            pad.append_engine("warning: the engine section of the scratchpad was edited; it was restored")
        if spec.context == "fresh" and hasattr(agent, "reset_conversation"):
            agent.reset_conversation()
        prompt = _build_prompt(spec, iteration, pad, artifact, note)
        try:
            reply = agent.execute_task_with_retry(prompt, use_plugins=spec.use_plugins)
        except Exception as e:  # noqa: BLE001 - stop cleanly; the checkpoint keeps the state
            result.iterations = iteration
            return finish(StopReason.ERROR, error=f"{type(e).__name__}: {e}")
        reply = getattr(reply, "text", reply)
        body, new_notes = _split_notes(reply)
        pad.append_note(new_notes)
        artifact = body
        judged_artifact = _judged_artifact(spec, body)
        results, judge_error, blocked = _check_requirements(judge, spec, judged_artifact, cwd)

        total = len(reqs)
        passing = None if judge_error else sum(c.passed for c in results.values())
        improved = tracker.update(passing)
        now_passed = {k for k, v in results.items() if v.passed}
        regressions = [] if judge_error else sorted(prev_passed - now_passed)
        if not judge_error:
            prev_passed = now_passed
        event = LoopEvent(iteration, results, passing, total, improved, tracker.streak, regressions,
                          judge_error, artifact, body, clock() - iter_start, report)
        events.append(event)
        result.iterations, result.artifact, result.final_results = iteration, artifact, results
        pad.append_engine(event.summary() + (f"; regressions={regressions}" if regressions else "")
                          + (f"; reviewer note: {note}" if note else ""))
        pad.set_feedback(_failing_text(reqs, results))
        pad.flush()
        _checkpoint(store, loop_id, spec, result, tracker, "running")

        stop = _after_iteration_stop(spec, iteration, passing, total, blocked, tracker)
        if stop is not None:
            return finish(*stop)
        decision = on_iteration(event) if on_iteration else None
        if decision is not None and decision.stop:
            return finish(StopReason.STOPPED, "stopped by the iteration hook")
        note = decision.note if decision else ""
    return finish(StopReason.MAX_ITERATIONS)  # pragma: no cover - the loop returns above


def _checkpoint(store: LoopStore, loop_id: str, spec: LoopSpec, result: LoopResult,
                tracker: StallTracker, status: str) -> None:
    store.save({
        "loop_id": loop_id,
        "status": status,
        "goal": spec.goal,
        "requirement_ids": [r.id for r in spec.requirements],
        "iteration": result.iterations,
        "stall": {"previous": tracker.previous, "streak": tracker.streak, "limit": tracker.limit},
        "history": [e.summary() for e in result.events],
        "detail": result.detail,
        "error": result.error,
    })
