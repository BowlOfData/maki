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

Durability: the state is checkpointed every iteration, so :func:`resume_loop` can continue a
loop after a crash, a stop, or a durable-approval suspension. With an ``ApprovalStore`` the
loop asks for approval before every iteration (gate N = N iterations done) and suspends
(``StopReason.SUSPENDED``) until a human answers: approve = continue, reject with a note =
continue with that note as feedback, reject without a note = stop.

Command checks are not sandboxed (see :mod:`maki.agents.loop_judge`). Tools that write
files need ``Agent(allow_dangerous_tools=True)``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .approvals import ApprovalDecision, ApprovalStore
from .loop_judge import CheckResult, _last_json_object, judge_requirements, run_command_check
from .loop_scratchpad import LoopLock, LoopStore, ReloadReport, Scratchpad, default_loop_dir, safe_id

logger = logging.getLogger(__name__)

_NOTES_HINT = ("Optionally end your reply with a line `NOTES:` followed by notes to your future self; "
               "everything after that line is kept as notes and is not part of the result.")
# Only the last "NOTES:" line splits the reply, so a result that itself contains such a line loses
# the text after it; give the agent a different heading if that matters for your task.

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
    SUSPENDED = "suspended"
    TOKEN_BUDGET = "token_budget"


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
    max_tokens:   token budget for maker + judge together (``None`` = no budget). Checked after
                  each iteration, so a loop can overshoot by one iteration. Counted from the
                  backends' reported usage; an agent that reports none counts as zero.
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
    max_tokens: Optional[int] = None

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
        if self.max_tokens is not None and (isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int)
                                            or self.max_tokens < 1):
            raise ValueError("LoopSpec.max_tokens must be an integer >= 1 (or None)")
        if self.context not in ("shared", "fresh"):
            raise ValueError("LoopSpec.context must be 'shared' or 'fresh'")
        if self.artifact_fn is not None and not callable(self.artifact_fn):
            raise ValueError("LoopSpec.artifact_fn must be callable")


@dataclass
class LoopTask:
    """Attach to a ``WorkflowTask`` (``loop=``) to run its agent in a loop until a judge confirms
    every requirement. The task is ``COMPLETED`` only when the loop succeeds; any other outcome
    fails it and skips its dependents. Mutually exclusive with ``verify=``.

    judge:        name of a registered, *different*, stateless agent that judges the requirements.
    requirements: ``Requirement`` objects or plain strings (agent-judged). Commands and callables
                  are allowed here because the workflow author is the trusted caller.
    The other fields mirror :class:`LoopSpec`. Loop state lives under
    ``<loop_dir_base>/<workflow_id>/<task name>`` so a resumed workflow resumes its loops.
    """
    judge: str
    requirements: Sequence[Union[Requirement, str]]
    max_iterations: int = 10
    max_seconds: Optional[float] = None
    stall_limit: int = 3
    context: str = "fresh"
    use_plugins: bool = False
    artifact_fn: Optional[Callable[[], str]] = None
    cwd: Optional[str] = None
    loop_dir_base: Optional[str] = None
    command_timeout: float = 120.0
    max_tokens: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.judge, str) or not self.judge.strip():
            raise ValueError("LoopTask.judge must be the name of a registered agent")
        self.requirements = tuple(r if isinstance(r, Requirement) else Requirement(f"r{n}", r)
                                  for n, r in enumerate(self.requirements, 1))
        # Reuse LoopSpec's validation of everything else.
        self.to_spec("check", "check goal")

    def to_spec(self, loop_id: str, goal: str) -> LoopSpec:
        base = self.loop_dir_base or os.path.dirname(default_loop_dir("x"))
        return LoopSpec(goal, self.requirements, max_iterations=self.max_iterations, max_seconds=self.max_seconds,
                        stall_limit=self.stall_limit, context=self.context, use_plugins=self.use_plugins,
                        artifact_fn=self.artifact_fn, cwd=self.cwd, loop_id=loop_id,
                        loop_dir=os.path.join(base, *(safe_id(part) for part in loop_id.split("/"))),
                        command_timeout=self.command_timeout, max_tokens=self.max_tokens)


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
    tokens: int = 0  # maker + judge tokens spent on this iteration

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
    tokens: int = 0  # total maker + judge tokens spent, across resumes

    @property
    def succeeded(self) -> bool:
        return self.stop_reason is StopReason.SUCCEEDED

    @property
    def suspended(self) -> bool:
        """True when the loop is waiting for a durable approval; continue with :func:`resume_loop`."""
        return self.stop_reason is StopReason.SUSPENDED


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


def _failing_text(reqs: Sequence[Requirement], results: Dict[str, CheckResult], judge_error: bool = False,
                  last_good: Optional[Dict[str, CheckResult]] = None) -> str:
    """Feedback for the maker. After a judge error the judge's requirements keep the last real
    verdict (or none), so "the judge failed" is never presented as something to fix."""
    lines = []
    for r in reqs:
        res = results[r.id]
        if judge_error and r.kind == "agent":
            res = (last_good or {}).get(r.id)
            if res is None:
                continue
        if not res.passed:
            lines.append(f"- {r.id} ({r.text}): {res.feedback or 'not met'}")
    return "\n".join(lines)


def _build_prompt(spec: LoopSpec, iteration: int, pad: Scratchpad, artifact: str, note: str,
                  full: bool = False) -> str:
    if iteration == 1:
        parts = [f"Goal:\n{spec.goal}", f"Requirements (all must be met):\n{_requirement_listing(spec.requirements)}"]
        if note:
            parts.append(f"Reviewer note: {note}")
        parts.append(_NOTES_HINT)
        return "\n\n".join(parts)
    if spec.context == "shared" and not full:
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


def _artifacts(spec: LoopSpec, body: str) -> Tuple[str, str]:
    """Return ``(current, judged)``: the artifact the loop tracks and the text the judge reads.

    Without ``artifact_fn`` both are the agent's reply. With it, the current artifact is the
    function's output (for code: the files) and the judge also sees the reply.
    """
    if spec.artifact_fn is None:
        return body, body
    try:
        current = str(spec.artifact_fn())
    except Exception as e:  # noqa: BLE001
        current = f"(artifact_fn failed: {type(e).__name__}: {e})"
    return current, f"Agent reply:\n{body}\n\nArtifact:\n{current}"


def _check_requirements(judge: Any, spec: LoopSpec, artifact: str, current: str,
                        cwd: str) -> Tuple[Dict[str, CheckResult], bool, str]:
    """Run command/callable checks, then one batched judge call for the rest.

    Returns ``(results by requirement id, judge_error, blocked reason)``.
    """
    results: Dict[str, CheckResult] = {}
    evidence: List[str] = []
    for r in spec.requirements:
        if r.kind == "agent":
            continue
        results[r.id] = _run_check(r, current, cwd, spec.command_timeout)
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


def _tokens_of(agent: Any) -> int:
    """Total tokens the agent's backend has reported so far (0 if it does not report any)."""
    total = getattr(getattr(agent, "usage", None), "total_tokens", 0)
    return total if isinstance(total, int) else 0


def _after_iteration_stop(spec: LoopSpec, iteration: int, passing: Optional[int], total: int, blocked: str,
                          tracker: StallTracker, tokens_used: int = 0) -> Optional[Tuple[StopReason, str]]:
    """Return ``(reason, detail)`` if the loop must stop after this iteration, else None."""
    if blocked:
        return StopReason.BLOCKED, blocked
    if passing == total:
        return StopReason.SUCCEEDED, ""
    if tracker.stalled:
        return StopReason.STALLED, f"no improvement for {tracker.streak} iterations"
    if spec.max_tokens is not None and tokens_used >= spec.max_tokens:
        return StopReason.TOKEN_BUDGET, f"used {tokens_used} of {spec.max_tokens} tokens"
    if iteration == spec.max_iterations:
        return StopReason.MAX_ITERATIONS, ""
    return None


GATE_TASK = "loop-gate"  # the (workflow, task) pair an approval store keeps a loop's gate under


def _fingerprint(requirements: Sequence[Requirement]) -> str:
    """Hash of what each requirement says and runs (callables count by kind only)."""
    payload = [[r.id, r.text, list(r.command) if r.command else None, r.kind] for r in requirements]
    return hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()


class _LoopRun:
    """One run of a loop: owns the scratchpad, checkpoint and counters, and drives the iterations."""

    def __init__(self, agent: Any, judge: Any, spec: LoopSpec, on_iteration: Optional[OnIteration],
                 approvals: Optional[ApprovalStore], clock: Callable[[], float]):
        _validate_start(agent, judge, spec)
        self.agent, self.judge, self.spec = agent, judge, spec
        self.on_iteration, self.approvals, self.clock = on_iteration, approvals, clock
        self.loop_id = spec.loop_id or f"loop-{uuid.uuid4().hex[:8]}"
        self.directory = spec.loop_dir or default_loop_dir(self.loop_id)
        self.pad = Scratchpad(os.path.join(self.directory, "scratchpad.md"), self.loop_id)
        self.store = LoopStore(self.directory)
        self.lock = LoopLock(self.directory)
        self.cwd = os.path.realpath(spec.cwd or os.getcwd())
        self.tracker = StallTracker(spec.stall_limit)
        self.events: List[LoopEvent] = []
        self.history: List[str] = []
        self.prev_passed: set = set()
        self.last_good: Dict[str, CheckResult] = {}
        self.artifact, self.note = "", ""
        self.iteration_done = 0
        self.base_elapsed = 0.0
        self.tokens_used = 0
        self.rebuild_prompt = False
        self.start = clock()
        self.result = LoopResult(self.loop_id, StopReason.STOPPED, 0, self.events, "", self.pad.path)

    # -- persistence -------------------------------------------------------------------------

    def elapsed(self) -> float:
        return self.base_elapsed + (self.clock() - self.start)

    def checkpoint(self, status: str, gate: Optional[int] = None) -> None:
        self.store.save({
            "loop_id": self.loop_id,
            "status": status,
            "goal": self.spec.goal,
            "requirement_ids": [r.id for r in self.spec.requirements],
            "requirements_fingerprint": _fingerprint(self.spec.requirements),
            "requirements": [{"id": r.id, "text": r.text, "kind": r.kind} for r in self.spec.requirements],
            "iteration": self.iteration_done,
            "stall": {"previous": self.tracker.previous, "streak": self.tracker.streak,
                      "limit": self.tracker.limit},
            "prev_passed": sorted(self.prev_passed),
            "artifact": self.artifact,
            "note": self.note,
            "elapsed": self.elapsed(),
            "engine_log": list(self.pad.engine_log),
            "feedback": self.pad.feedback,
            "notes": self.pad.notes,
            "results": {k: {"passed": v.passed, "feedback": v.feedback}
                        for k, v in self.result.final_results.items()},
            "last_good": {k: {"passed": v.passed, "feedback": v.feedback} for k, v in self.last_good.items()},
            "uses_approvals": self.approvals is not None,
            "tokens": self.tokens_used,
            "max_tokens": self.spec.max_tokens,
            "context": self.spec.context,
            "use_plugins": self.spec.use_plugins,
            "history": list(self.history),
            "gate": gate,
            "detail": self.result.detail,
            "error": self.result.error,
        })

    def restore(self, state: Dict[str, Any]) -> None:
        try:
            self.iteration_done = int(state["iteration"])
            self.tracker.previous = int(state["stall"]["previous"])
            self.tracker.streak = int(state["stall"]["streak"])
            self.prev_passed = set(state["prev_passed"])
            self.artifact, self.note = str(state["artifact"]), str(state["note"])
            self.base_elapsed = float(state["elapsed"])
            self.tokens_used = int(state.get("tokens", 0))
            self.pad.engine_log = [str(x) for x in state["engine_log"]]
            self.pad.feedback, self.pad.notes = str(state["feedback"]), str(state["notes"])
            self.history = [str(x) for x in state["history"]]
            self.result.final_results = {k: CheckResult(bool(v["passed"]), str(v["feedback"]))
                                         for k, v in state["results"].items()}
            self.last_good = {k: CheckResult(bool(v["passed"]), str(v["feedback"]))
                              for k, v in state.get("last_good", {}).items()}
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            raise ValueError(f"loop checkpoint is corrupt ({type(e).__name__}: {e})") from e
        self.result.iterations, self.result.artifact = self.iteration_done, self.artifact
        self.result.tokens = self.tokens_used
        self.rebuild_prompt = True  # a restarted process has lost any shared conversation

    def finish(self, reason: StopReason, detail: str = "", error: str = "") -> LoopResult:
        self.result.stop_reason, self.result.detail, self.result.error = reason, detail, error
        verb = "suspended" if reason is StopReason.SUSPENDED else "stopped"
        self.pad.append_engine(f"{verb}: {reason.value}{(' - ' + detail) if detail else ''}")
        self.pad.flush()
        self.checkpoint(reason.value, gate=self.iteration_done if reason is StopReason.SUSPENDED else None)
        return self.result

    # -- durable approval --------------------------------------------------------------------

    def _gate_summary(self, number: int) -> str:
        if number == 0:
            return (f"Start loop '{self.loop_id}'\nGoal: {self.spec.goal}\n"
                    + "\n".join(f"- {r.id}: {r.text}" + (f"  [command: {' '.join(r.command)}]" if r.command else "")
                                for r in self.spec.requirements))
        return f"{self.history[-1] if self.history else ''}\n{self.pad.feedback}".strip()

    def gate(self, number: int) -> Optional[LoopResult]:
        """Ask for approval before the next iteration; return a result if the loop must not go on."""
        if self.approvals is None:
            return None
        key = hashlib.sha256(f"{self.loop_id}:{number}".encode()).hexdigest()[:16]
        decision = self.approvals.check(self.loop_id, GATE_TASK, key, self._gate_summary(number))
        if decision is ApprovalDecision.PENDING:
            return self.finish(StopReason.SUSPENDED, f"waiting for approval of gate {number}")
        record = self.approvals.get(self.loop_id, GATE_TASK)
        human_note = (record.note if record else "").strip()
        if decision is ApprovalDecision.REJECTED and not human_note:
            return self.finish(StopReason.STOPPED, f"gate {number} rejected")
        if human_note:
            self.note = f"{self.note}\n{human_note}".strip()
        return None

    # -- the iterations ----------------------------------------------------------------------

    def run(self) -> LoopResult:
        spec = self.spec
        for iteration in range(self.iteration_done + 1, spec.max_iterations + 1):
            if spec.max_seconds is not None and self.elapsed() >= spec.max_seconds:
                return self.finish(StopReason.TIMEOUT, f"wall-clock cap of {spec.max_seconds}s reached")
            if spec.max_tokens is not None and self.tokens_used >= spec.max_tokens:
                return self.finish(StopReason.TOKEN_BUDGET, f"used {self.tokens_used} of {spec.max_tokens} tokens")
            blocked = self.gate(iteration - 1)
            if blocked is not None:
                return blocked
            outcome = self._iterate(iteration)
            if outcome is not None:
                return outcome
            decision = self.on_iteration(self.events[-1]) if self.on_iteration else None
            if decision is not None and decision.stop:
                return self.finish(StopReason.STOPPED, "stopped by the iteration hook")
            self.note = decision.note if decision else ""
            if self.note:
                self.checkpoint("running")  # a pending reviewer note must survive a crash
        return self.finish(StopReason.MAX_ITERATIONS)

    def _spend(self, tokens: int) -> None:
        self.tokens_used += max(tokens, 0)
        self.result.tokens = self.tokens_used

    def _iterate(self, iteration: int) -> Optional[LoopResult]:
        """Run one iteration; return a result if the loop ends here, else None."""
        spec, reqs, pad = self.spec, self.spec.requirements, self.pad
        iter_start = self.clock()
        report = pad.reload()
        if report.engine_tampered:
            pad.append_engine("warning: the engine section of the scratchpad was edited; it was restored")
        if spec.context == "fresh" and hasattr(self.agent, "reset_conversation"):
            self.agent.reset_conversation()
        used_note = self.note
        prompt = _build_prompt(spec, iteration, pad, self.artifact, used_note, full=self.rebuild_prompt)
        self.rebuild_prompt = False
        maker_before = _tokens_of(self.agent)
        try:
            reply = self.agent.execute_task_with_retry(prompt, use_plugins=spec.use_plugins)
        except Exception as e:  # noqa: BLE001 - stop cleanly; the checkpoint keeps the state
            self._spend(_tokens_of(self.agent) - maker_before)
            return self.finish(StopReason.ERROR, error=f"{type(e).__name__}: {e}")
        spent = _tokens_of(self.agent) - maker_before
        self.note = ""  # consumed; kept until now so a failed call can be resumed with it
        body, new_notes = _split_notes(getattr(reply, "text", reply))
        pad.append_note(new_notes)
        current, judged = _artifacts(spec, body)
        self.artifact = current
        judge_before = _tokens_of(self.judge)
        results, judge_error, blocked = _check_requirements(self.judge, spec, judged, current, self.cwd)
        spent += _tokens_of(self.judge) - judge_before
        self._spend(spent)

        passing = None if judge_error else sum(c.passed for c in results.values())
        improved = self.tracker.update(passing)
        now_passed = {k for k, v in results.items() if v.passed}
        regressions = [] if judge_error else sorted(self.prev_passed - now_passed)
        if not judge_error:
            self.prev_passed = now_passed
        event = LoopEvent(iteration, results, passing, len(reqs), improved, self.tracker.streak, regressions,
                          judge_error, current, body, self.clock() - iter_start, report, max(spent, 0))
        self.events.append(event)
        self.iteration_done = iteration
        self.result.iterations, self.result.artifact, self.result.final_results = iteration, current, results
        self.history.append(event.summary())
        pad.append_engine(event.summary() + (f"; regressions={regressions}" if regressions else "")
                          + (f"; reviewer note: {used_note}" if used_note else ""))
        if not judge_error:
            self.last_good = dict(results)
        pad.set_feedback(_failing_text(reqs, results, judge_error, self.last_good))
        pad.flush()
        self.checkpoint("running")
        stop = _after_iteration_stop(spec, iteration, passing, len(reqs), blocked, self.tracker, self.tokens_used)
        return self.finish(*stop) if stop is not None else None


def run_loop(agent: Any, judge: Any, spec: LoopSpec, on_iteration: Optional[OnIteration] = None,
             confirm_requirements: Optional[Callable[[Sequence[Requirement]], bool]] = None,
             approvals: Optional[ApprovalStore] = None,
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
        approvals: an :class:`~maki.agents.approvals.ApprovalStore` for durable approval
            before every iteration. A pending approval returns a ``SUSPENDED`` result;
            continue it later with :func:`resume_loop`.

    Raises:
        ValueError: for an invalid setup (unapproved drafts, stateless agent in shared
            mode, dangerous tools not allowed, ...).
        LoopBusyError: another process is already running a loop with the same directory.
    """
    run = _LoopRun(agent, judge, spec, on_iteration, approvals, clock)
    with run.lock:
        if confirm_requirements is not None and not confirm_requirements(spec.requirements):
            return run.finish(StopReason.STOPPED, "requirements not confirmed")
        run.pad.flush()
        decision = on_iteration(None) if on_iteration else None
        if decision is not None and decision.stop:
            return run.finish(StopReason.STOPPED, "stopped before the first iteration")
        run.note = decision.note if decision else ""
        return run.run()


def resume_loop(agent: Any, judge: Any, spec: LoopSpec, on_iteration: Optional[OnIteration] = None,
                approvals: Optional[ApprovalStore] = None,
                clock: Callable[[], float] = time.monotonic) -> LoopResult:
    """Continue a loop from its checkpoint (same ``loop_id``/``loop_dir``, same goal and requirement ids).

    Works after a crash, a stop, an error, a cap or a durable-approval suspension; a loop
    that already succeeded cannot be resumed. The first resumed iteration gets the full
    prompt (goal, artifact, feedback, notes) because a restarted process has lost any
    shared conversation. Raise ``max_iterations`` / ``max_seconds`` in ``spec`` to continue
    past a cap; wall-clock time spent before the restart still counts. The returned
    ``events`` hold only the iterations of this resumed run; ``iterations`` is the total.
    A loop that was started with an ``approvals`` store must be resumed with it.

    Raises:
        ValueError: no usable checkpoint, a corrupt one, or one that does not match ``spec``.
        LoopBusyError: another process is running this loop.
    """
    if not spec.loop_id and not spec.loop_dir:
        raise ValueError("resume_loop needs spec.loop_id or spec.loop_dir")
    run = _LoopRun(agent, judge, spec, on_iteration, approvals, clock)
    with run.lock:
        try:
            state = run.store.load()
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"loop checkpoint is unreadable ({type(e).__name__}: {e})") from e
        if state is None:
            raise ValueError(f"no checkpoint to resume at {run.store.path}")
        if state.get("status") == StopReason.SUCCEEDED.value:
            raise ValueError("this loop already succeeded")
        if state.get("goal") != spec.goal or state.get("requirement_ids") != [r.id for r in spec.requirements]:
            raise ValueError("the checkpoint was written for a different goal or set of requirements")
        if state.get("requirements_fingerprint") != _fingerprint(spec.requirements):
            raise ValueError("a requirement's text or command changed since the checkpoint; "
                             "approvals given for the old ones do not carry over")
        if state.get("uses_approvals") and approvals is None:
            raise ValueError("this loop requires approval before each iteration; pass the same approvals store")
        run.restore(state)
        return run.run()
