"""
Background loop jobs for the agent server (see :mod:`maki.agents.loop`).

``LoopJobs`` runs one agent loop at a time in a background thread and keeps an
append-only event list per job so a client can watch (with replay), cancel, and look the
job up again after the watching connection dropped or the server restarted.

Remote callers are untrusted with respect to what the server executes, so a job accepts
**text requirements only**: no commands, callables, working directories or loop
directories, and the server (not the caller) picks the job id. Caps on iterations and
wall-clock time come from the operator.

Event schema (each event is a JSON-compatible dict with a ``type``):

* ``started``:   ``loop_id``, ``goal``, ``requirements``
* ``iteration``: ``iteration``, ``tokens`` (maker + judge), ``passing`` (None after a judge error), ``total``,
  ``improved``, ``stall_streak``, ``regressions``, ``judge_error``, ``elapsed_seconds``,
  ``results`` (per requirement ``passed`` and ``feedback``)
* ``stopped``:   ``stop_reason`` (``suspended`` = waiting for an approval, see ``LoopJobs.approve``),
  ``iterations``, ``tokens`` (total), ``detail``, ``error``, ``artifact``

Cancel takes effect between iterations; a model call already in flight finishes first.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from ..agents.approvals import ApprovalStore, LocalApprovalStore
from ..agents.loop import (
    GATE_TASK, Decision, LoopEvent, LoopResult, LoopSpec, Requirement, resume_loop, run_loop,
)
from ..agents.loop_scratchpad import LoopBusyError, LoopLock, LoopStore, default_loop_dir, safe_id

logger = logging.getLogger(__name__)

_MAX_FINISHED_IN_MEMORY = 50
_MAX_REQUIREMENTS = 30
_MAX_TEXT_CHARS = 2000
_MAX_GOAL_CHARS = 20000
LOOP_ID_PATTERN = re.compile(r"^loop-[0-9a-f]{12}$")
_REQUIREMENT_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


class LoopJobBusy(RuntimeError):
    """A loop is already running on this server."""


class NoPendingApproval(RuntimeError):
    """The loop has no approval request waiting for an answer."""


class LoopJobNotFound(KeyError):
    """No job with that id, in memory or on disk."""


@dataclass
class LoopCaps:
    """Operator limits for remote loops."""
    max_iterations: int = 20
    max_seconds: Optional[float] = 3600.0
    loop_dir_base: str = field(default_factory=lambda: os.path.dirname(default_loop_dir("x")))
    max_tokens: Optional[int] = None  # cap on a remote loop's token budget; None = callers may leave it unset
    retention_days: Optional[float] = 14.0  # finished server loops older than this are deleted; None = keep

    def __post_init__(self) -> None:
        if isinstance(self.max_iterations, bool) or not isinstance(self.max_iterations, int) \
                or self.max_iterations < 1:
            raise ValueError("LoopCaps.max_iterations must be an integer >= 1")
        if self.max_seconds is not None and not self.max_seconds > 0:
            raise ValueError("LoopCaps.max_seconds must be > 0")
        if self.max_tokens is not None and (isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int)
                                            or self.max_tokens < 1):
            raise ValueError("LoopCaps.max_tokens must be an integer >= 1 (or None)")
        if self.retention_days is not None and not (self.retention_days > 0 and math.isfinite(self.retention_days)):
            raise ValueError("LoopCaps.retention_days must be > 0 (or None to keep everything)")


def _usable_state(state: Any) -> bool:
    """A checkpoint on disk is data a person may have edited: check the shape the server relies on."""
    if not isinstance(state, dict) or not isinstance(state.get("goal"), str):
        return False
    reqs = state.get("requirements")
    if not isinstance(reqs, list) or not all(
            isinstance(r, dict) and isinstance(r.get("id"), str) and isinstance(r.get("text"), str)
            and isinstance(r.get("kind"), str) for r in reqs):
        return False
    stall = state.get("stall")
    limit = stall.get("limit") if isinstance(stall, dict) else None
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        return False
    return all(isinstance(state.get(k, d), t) for k, d, t in (
        ("iteration", 0, int), ("tokens", 0, int), ("status", "", str), ("loop_id", "", str)))


def _parse_requirement(n: int, item: Union[str, Mapping[str, Any]]) -> Requirement:
    if isinstance(item, str):
        rid, text = f"r{n}", item
    elif isinstance(item, Mapping):
        unknown = set(item) - {"id", "text"}
        if unknown:
            raise ValueError(f"requirement {n}: unsupported field(s) {sorted(unknown)}; "
                             "only text requirements are accepted over HTTP")
        rid, text = str(item.get("id") or f"r{n}"), item.get("text")
    else:
        raise ValueError(f"requirement {n} must be a string or an object with 'text'")
    if not _REQUIREMENT_ID.match(rid):
        raise ValueError(f"requirement {n}: id must be 1-64 letters, digits, '_', '.' or '-'")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"requirement {n} needs non-empty text")
    if len(text) > _MAX_TEXT_CHARS:
        raise ValueError(f"requirement {n} is longer than {_MAX_TEXT_CHARS} characters")
    return Requirement(rid, text.strip())


def parse_requirements(raw: Sequence[Union[str, Mapping[str, Any]]]) -> List[Requirement]:
    """Turn caller-supplied requirements into agent-judged ``Requirement`` objects.

    Accepts strings or ``{"id": ..., "text": ...}`` mappings; any other key (a command,
    check, cwd, ...) is rejected rather than ignored.

    Raises:
        ValueError: for an empty or oversized list, bad entries, or unknown keys.
    """
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("requirements must be a non-empty list")
    if len(raw) > _MAX_REQUIREMENTS:
        raise ValueError(f"at most {_MAX_REQUIREMENTS} requirements are allowed")
    out = [_parse_requirement(n, item) for n, item in enumerate(raw, 1)]
    if len({r.id for r in out}) != len(out):
        raise ValueError("requirement ids must be unique")
    return out


def _event_dict(event: LoopEvent) -> Dict[str, Any]:
    return {
        "type": "iteration",
        "iteration": event.iteration,
        "tokens": event.tokens,
        "passing": event.passing,
        "total": event.total,
        "improved": event.improved,
        "stall_streak": event.stall_streak,
        "regressions": list(event.regressions),
        "judge_error": event.judge_error,
        "elapsed_seconds": round(event.elapsed_seconds, 3),
        "results": {k: {"passed": v.passed, "feedback": v.feedback} for k, v in event.results.items()},
    }


def _stopped_event(result: LoopResult) -> Dict[str, Any]:
    return {"type": "stopped", "stop_reason": result.stop_reason.value, "iterations": result.iterations,
            "tokens": result.tokens, "detail": result.detail, "error": result.error, "artifact": result.artifact}


class LoopJob:
    """One loop run: its identity, event log and cancel flag."""

    def __init__(self, loop_id: str, goal: str, requirements: Sequence[Requirement], base_tokens: int = 0):
        self.id = loop_id
        self.base_tokens = base_tokens  # spent before this run (a resumed loop)
        self.goal = goal
        self.requirements = [{"id": r.id, "text": r.text} for r in requirements]
        self.created_at = time.time()
        self.status = "running"  # running | succeeded | stalled | ... (a StopReason value) | error
        self.error = ""
        self.result: Optional[LoopResult] = None
        self.cancel_requested = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self._events: List[Dict[str, Any]] = []
        self._cond = threading.Condition()
        self.finished = False

    def append(self, event: Dict[str, Any]) -> None:
        with self._cond:
            self._events.append(event)
            self._cond.notify_all()

    def finish(self, status: str, error: str = "", result: Optional[LoopResult] = None) -> None:
        with self._cond:
            self.status, self.error, self.finished = status, error, True
            if result is not None:
                self.result = result
            self._cond.notify_all()

    def events_from(self, index: int) -> "tuple[List[Dict[str, Any]], bool]":
        """Events from ``index`` on, without waiting, and whether the job is over."""
        with self._cond:
            return list(self._events[index:]), self.finished

    def wait_events(self, index: int, timeout: float = 15.0) -> "tuple[List[Dict[str, Any]], bool]":
        """Events from ``index`` on (waiting up to ``timeout`` for new ones) and whether the job is over."""
        with self._cond:
            if index >= len(self._events) and not self.finished:
                self._cond.wait(timeout)
            return list(self._events[index:]), self.finished

    def snapshot(self) -> Dict[str, Any]:
        with self._cond:
            res = self.result
            return {
                "id": self.id, "status": self.status, "goal": self.goal, "requirements": self.requirements,
                "iterations": res.iterations if res else (self._last_iteration()),
                "stop_reason": res.stop_reason.value if res else None,
                "detail": res.detail if res else "", "error": self.error or (res.error if res else ""),
                "artifact": res.artifact if res else "", "tokens": res.tokens if res else self._last_tokens(),
                "cancel_requested": self.cancel_requested.is_set(),
                "events": len(self._events), "created_at": self.created_at, "resumable": False,
            }

    def _last_tokens(self) -> int:
        return self.base_tokens + sum(int(e.get("tokens", 0)) for e in self._events if e.get("type") == "iteration")

    def _last_iteration(self) -> int:
        for e in reversed(self._events):
            if e.get("type") == "iteration":
                return int(e["iteration"])
        return 0


class LoopJobs:
    """Runs loops for one served agent, one at a time."""

    def __init__(self, agent: Any, judge: Any, caps: Optional[LoopCaps] = None):
        self.agent, self.judge = agent, judge
        self.caps = caps or LoopCaps()
        self._jobs: Dict[str, LoopJob] = {}
        self._lock = threading.Lock()
        self.approvals: ApprovalStore = LocalApprovalStore(os.path.join(self.caps.loop_dir_base, ".approvals"))
        self.prune()

    # -- housekeeping ------------------------------------------------------------------------

    def prune(self) -> int:
        """Delete server-made loop directories whose checkpoint is older than ``retention_days``.

        Skips anything running in memory, a checkpoint still marked running (it may be a live
        run from another process), and directories that are not server loop ids. Returns the
        number removed.
        """
        days = self.caps.retention_days
        if days is None:
            return 0
        cutoff = time.time() - days * 86400
        removed = 0
        try:
            names = os.listdir(self.caps.loop_dir_base)
        except OSError:
            return 0
        for name in names:
            path = os.path.join(self.caps.loop_dir_base, name)
            if not LOOP_ID_PATTERN.match(name) or os.path.islink(path):
                continue
            with self._lock:  # a start/resume registers its job under this lock, so none can slip in
                if name in self._jobs and not self._jobs[name].finished:
                    continue
                removed += self._prune_one(name, path, cutoff)
        return removed

    def _prune_one(self, name: str, path: str, cutoff: float) -> int:
        lock = LoopLock(path)
        try:
            state_file = os.path.join(path, "state.json")
            age_of = state_file if os.path.exists(state_file) else path  # no checkpoint yet: use the directory
            if os.path.getmtime(age_of) >= cutoff:
                return 0
            try:
                raw = LoopStore(path).load()
            except (OSError, ValueError):
                raw = None  # missing or corrupt and old: it can go
            if isinstance(raw, dict) and raw.get("status") == "running":
                return 0
            lock.acquire()  # held by a loop running in another process: leave it alone
        except LoopBusyError:
            return 0
        except OSError:
            return 0
        try:
            shutil.rmtree(path)
            self.approvals.discard(name)
            return 1
        except OSError:
            logger.warning("could not prune loop directory %s", path, exc_info=True)
            return 0
        finally:
            lock.release()

    # -- starting ----------------------------------------------------------------------------

    def start(self, goal: str, requirements: Sequence[Union[str, Mapping[str, Any]]], *,
              max_iterations: Optional[int] = None, max_seconds: Optional[float] = None,
              stall_limit: Optional[int] = None, context: Optional[str] = None,
              use_plugins: bool = False, max_tokens: Optional[int] = None,
              require_approval: bool = False) -> LoopJob:
        """Start a loop in the background and return its job.

        Raises:
            ValueError: invalid input, or a value above the operator's caps.
            LoopJobBusy: another loop is already running.
        """
        self.prune()
        reqs = parse_requirements(requirements)
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal must be a non-empty string")
        if len(goal) > _MAX_GOAL_CHARS:
            raise ValueError(f"goal is longer than {_MAX_GOAL_CHARS} characters")
        iters = self.caps.max_iterations if max_iterations is None else max_iterations
        if not isinstance(iters, int) or isinstance(iters, bool) or iters < 1 or iters > self.caps.max_iterations:
            raise ValueError(f"max_iterations must be between 1 and {self.caps.max_iterations}")
        seconds = self.caps.max_seconds if max_seconds is None else max_seconds
        if seconds is not None and (not (seconds > 0 and math.isfinite(seconds)) or (
                self.caps.max_seconds is not None and seconds > self.caps.max_seconds)):
            raise ValueError(f"max_seconds must be between 0 and {self.caps.max_seconds}")
        tokens = self._tokens_limit(max_tokens)
        mode = context or ("shared" if getattr(self.agent, "stateful", False) else "fresh")
        if mode == "shared" and not getattr(self.agent, "stateful", False):
            raise ValueError("context 'shared' needs a stateful served agent; use 'fresh'")
        loop_id = f"loop-{uuid.uuid4().hex[:12]}"
        spec = LoopSpec(goal.strip(), reqs, max_iterations=iters, max_seconds=seconds,
                        stall_limit=3 if stall_limit is None else stall_limit, context=mode,
                        use_plugins=bool(use_plugins), loop_id=loop_id, max_tokens=tokens,
                        loop_dir=os.path.join(self.caps.loop_dir_base, safe_id(loop_id)))
        return self._launch(LoopJob(loop_id, spec.goal, reqs), spec, resume=False, gated=bool(require_approval))

    def _tokens_limit(self, requested: Optional[int]) -> Optional[int]:
        cap = self.caps.max_tokens
        value = cap if requested is None else requested
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 1 or (cap is not None and value > cap):
            raise ValueError(f"max_tokens must be between 1 and {cap if cap is not None else 'unbounded'}")
        return value

    def resume(self, loop_id: str, *, max_iterations: Optional[int] = None,
               max_seconds: Optional[float] = None, max_tokens: Optional[int] = None) -> LoopJob:
        """Continue a checkpointed loop (text requirements only) after a stop or a server restart.

        Raises:
            LoopJobNotFound: no such loop.
            ValueError: it cannot be resumed over HTTP (finished successfully, or it had
                commands/callables) or a value is above the caps.
            LoopJobBusy: another loop is already running.
        """
        state = self._load_state(loop_id)
        if state is None:
            if os.path.exists(os.path.join(self.caps.loop_dir_base, safe_id(loop_id), "state.json")):
                raise ValueError("this loop's checkpoint is unreadable or corrupt")
            raise LoopJobNotFound(loop_id)
        if state.get("status") == "succeeded":
            raise ValueError("this loop already succeeded")
        reqs_state = state.get("requirements")
        if not isinstance(reqs_state, list) or not reqs_state \
                or any(not isinstance(r, dict) or r.get("kind") != "agent" for r in reqs_state):
            raise ValueError("this loop cannot be resumed over HTTP")
        reqs = [Requirement(r["id"], r["text"]) for r in reqs_state]
        iters = self.caps.max_iterations if max_iterations is None else max_iterations
        if not isinstance(iters, int) or isinstance(iters, bool) or not 1 <= iters <= self.caps.max_iterations:
            raise ValueError(f"max_iterations must be between 1 and {self.caps.max_iterations}")
        seconds = self.caps.max_seconds if max_seconds is None else max_seconds
        if seconds is not None and (not (seconds > 0 and math.isfinite(seconds)) or (
                self.caps.max_seconds is not None and seconds > self.caps.max_seconds)):
            raise ValueError(f"max_seconds must be between 0 and {self.caps.max_seconds}")
        saved = state.get("max_tokens")  # the original budget stays unless the caller sets a new one
        if max_tokens is None and isinstance(saved, int) and not isinstance(saved, bool) and saved >= 1:
            max_tokens = saved if self.caps.max_tokens is None else min(saved, self.caps.max_tokens)
        tokens = self._tokens_limit(max_tokens)
        mode = state.get("context")  # saved by the original run; older checkpoints lack it
        if mode not in ("shared", "fresh") or (mode == "shared" and not getattr(self.agent, "stateful", False)):
            mode = "shared" if getattr(self.agent, "stateful", False) else "fresh"
        spec = LoopSpec(state["goal"], reqs, max_iterations=iters, max_seconds=seconds, context=mode,
                        use_plugins=state.get("use_plugins") is True, max_tokens=tokens,
                        stall_limit=state["stall"]["limit"], loop_id=loop_id,
                        loop_dir=os.path.join(self.caps.loop_dir_base, safe_id(loop_id)))
        spent = state.get("tokens")
        job = LoopJob(loop_id, spec.goal, reqs, base_tokens=spent if isinstance(spent, int) else 0)
        return self._launch(job, spec, resume=True,
                            gated=bool(state.get("uses_approvals")))

    def _launch(self, job: LoopJob, spec: LoopSpec, resume: bool, gated: bool = False) -> LoopJob:
        with self._lock:
            if any(not j.finished for j in self._jobs.values()):
                raise LoopJobBusy("a loop is already running on this server")
            self._jobs[job.id] = job
            self._evict()
            job.thread = threading.Thread(target=self._run, args=(job, spec, resume, gated),
                                          name=f"maki-{job.id}", daemon=True)
            job.thread.start()
        return job

    def _evict(self) -> None:
        finished = sorted((j for j in self._jobs.values() if j.finished), key=lambda j: j.created_at)
        for j in finished[:max(0, len(finished) - _MAX_FINISHED_IN_MEMORY)]:
            del self._jobs[j.id]

    def _run(self, job: LoopJob, spec: LoopSpec, resume: bool, gated: bool) -> None:
        approvals = self.approvals if gated else None
        emitted = 0

        def hook(event: Optional[LoopEvent]) -> Optional[Decision]:
            nonlocal emitted
            if event is not None:
                job.append(_event_dict(event))
                emitted += 1
            return Decision.STOP if job.cancel_requested.is_set() else None

        job.append({"type": "started", "loop_id": job.id, "goal": job.goal, "requirements": job.requirements})
        try:
            if resume:
                result = resume_loop(self.agent, self.judge, spec, on_iteration=hook, approvals=approvals)
            else:
                result = run_loop(self.agent, self.judge, spec, on_iteration=hook, approvals=approvals)
        except (ValueError, LoopBusyError) as e:
            self._fail(job, str(e))
            return
        except Exception:  # noqa: BLE001 - details stay in the server log
            logger.exception("loop %s failed", job.id)
            self._fail(job, "loop failed; see server logs")
            return
        for event in result.events[emitted:]:  # the engine skips the hook on the iteration that ends the loop
            job.append(_event_dict(event))
        job.append(_stopped_event(result))
        job.finish(result.stop_reason.value, result.error, result)

    @staticmethod
    def _fail(job: LoopJob, message: str) -> None:
        job.append({"type": "stopped", "stop_reason": "error", "iterations": job._last_iteration(),
                    "tokens": job._last_tokens(), "detail": "", "error": message, "artifact": ""})
        job.finish("error", message)

    # -- looking up --------------------------------------------------------------------------

    def get(self, loop_id: str) -> Optional[LoopJob]:
        with self._lock:
            return self._jobs.get(loop_id)

    def cancel(self, loop_id: str) -> bool:
        """Ask a running loop to stop after its current iteration. False if it is not running."""
        job = self.get(loop_id)
        if job is None or job.finished:
            return False
        job.cancel_requested.set()
        return True

    def _load_state(self, loop_id: str) -> Optional[Dict[str, Any]]:
        directory = os.path.join(self.caps.loop_dir_base, safe_id(loop_id))
        try:
            state = LoopStore(directory).load()
        except (OSError, json.JSONDecodeError):
            return None
        return state if _usable_state(state) else None

    def snapshot(self, loop_id: str) -> Dict[str, Any]:
        """A job's status: from memory when known, else rebuilt from its checkpoint on disk."""
        job = self.get(loop_id)
        if job is not None:
            snap = job.snapshot()
            if job.finished and snap["status"] != "succeeded":
                state = self._load_state(loop_id)
                snap["resumable"] = state is not None and self._snapshot_from_state(loop_id, state)["resumable"]
            snap["pending_approval"] = self._pending_approval(loop_id) if snap["status"] == "suspended" else None
            return snap
        state = self._load_state(loop_id)
        if state is None:
            raise LoopJobNotFound(loop_id)
        snap = self._snapshot_from_state(loop_id, state)
        snap["pending_approval"] = self._pending_approval(loop_id) if snap["status"] == "suspended" else None
        return snap

    def _pending_approval(self, loop_id: str) -> Optional[Dict[str, Any]]:
        record = self.approvals.get(loop_id, GATE_TASK)
        if record is None or record.status != "pending":
            return None
        return {"summary": record.summary, "requested_at": record.requested_at}

    def approve(self, loop_id: str, approve: bool, note: str = "", by: str = "http") -> Dict[str, Any]:
        """Answer a suspended loop's pending approval; resume it afterwards to act on the answer.

        Approve (optionally with a note the agent must address) lets the next iteration run;
        reject with a note continues with that note; reject without a note stops the loop.

        Raises:
            LoopJobNotFound: no such loop.
            NoPendingApproval: nothing is waiting for an answer.
            ValueError: the note is not text or is too long.
        """
        if not isinstance(note, str) or len(note) > _MAX_TEXT_CHARS:
            raise ValueError(f"note must be text of at most {_MAX_TEXT_CHARS} characters")
        self.snapshot(loop_id)  # raises LoopJobNotFound
        record = self.approvals.get(loop_id, GATE_TASK)
        if record is None or record.status != "pending":
            raise NoPendingApproval(loop_id)
        if approve:
            self.approvals.approve(loop_id, GATE_TASK, key=record.key, by=by, note=note.strip())
        else:
            self.approvals.reject(loop_id, GATE_TASK, key=record.key, by=by, note=note.strip())
        return self.snapshot(loop_id)

    @staticmethod
    def _snapshot_from_state(loop_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
        status = str(state.get("status", "unknown"))
        reqs = state.get("requirements") if isinstance(state.get("requirements"), list) else []
        return {
            "id": loop_id, "status": "interrupted" if status == "running" else status,
            "goal": state.get("goal", ""),
            "requirements": [{"id": r.get("id"), "text": r.get("text")} for r in reqs if isinstance(r, dict)],
            "iterations": state.get("iteration", 0), "tokens": state.get("tokens", 0),
            "stop_reason": None if status == "running" else status,
            "detail": state.get("detail", ""), "error": state.get("error", ""), "artifact": state.get("artifact", ""),
            "cancel_requested": False, "events": 0, "created_at": None,
            "resumable": status != "succeeded" and bool(reqs)
            and all(isinstance(r, dict) and r.get("kind") == "agent" for r in reqs),
        }

    def list(self) -> List[Dict[str, Any]]:
        """Every known loop: running/finished ones in memory plus checkpoints on disk."""
        with self._lock:
            known = {i: j.snapshot() for i, j in self._jobs.items()}
        base = self.caps.loop_dir_base
        try:
            names = sorted(os.listdir(base))
        except OSError:
            names = []
        for name in names:
            state = self._load_state(name)
            lid = state.get("loop_id") if state else None
            if not isinstance(lid, str) or not LOOP_ID_PATTERN.match(lid) or lid in known:
                continue
            known[lid] = self._snapshot_from_state(lid, state)
        return sorted(known.values(), key=lambda s: (s.get("created_at") or 0, s["id"]), reverse=True)
