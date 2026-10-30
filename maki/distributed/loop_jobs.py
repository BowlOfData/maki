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
* ``iteration``: ``iteration``, ``passing`` (None after a judge error), ``total``,
  ``improved``, ``stall_streak``, ``regressions``, ``judge_error``, ``elapsed_seconds``,
  ``results`` (per requirement ``passed`` and ``feedback``)
* ``stopped``:   ``stop_reason``, ``iterations``, ``detail``, ``error``, ``artifact``

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

from ..agents.loop import Decision, LoopEvent, LoopResult, LoopSpec, Requirement, resume_loop, run_loop
from ..agents.loop_scratchpad import LoopBusyError, LoopStore, default_loop_dir, safe_id

logger = logging.getLogger(__name__)

_MAX_FINISHED_IN_MEMORY = 50
_MAX_REQUIREMENTS = 30
_MAX_TEXT_CHARS = 2000
_MAX_GOAL_CHARS = 20000
LOOP_ID_PATTERN = re.compile(r"^loop-[0-9a-f]{12}$")
_REQUIREMENT_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


class LoopJobBusy(RuntimeError):
    """A loop is already running on this server."""


class LoopJobNotFound(KeyError):
    """No job with that id, in memory or on disk."""


@dataclass
class LoopCaps:
    """Operator limits for remote loops."""
    max_iterations: int = 20
    max_seconds: Optional[float] = 3600.0
    loop_dir_base: str = field(default_factory=lambda: os.path.dirname(default_loop_dir("x")))
    retention_days: Optional[float] = 14.0  # finished server loops older than this are deleted; None = keep

    def __post_init__(self) -> None:
        if isinstance(self.max_iterations, bool) or not isinstance(self.max_iterations, int) \
                or self.max_iterations < 1:
            raise ValueError("LoopCaps.max_iterations must be an integer >= 1")
        if self.max_seconds is not None and not self.max_seconds > 0:
            raise ValueError("LoopCaps.max_seconds must be > 0")
        if self.retention_days is not None and not (self.retention_days > 0 and math.isfinite(self.retention_days)):
            raise ValueError("LoopCaps.retention_days must be > 0 (or None to keep everything)")


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
    out: List[Requirement] = []
    for n, item in enumerate(raw, 1):
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
        out.append(Requirement(rid, text.strip()))
    if len({r.id for r in out}) != len(out):
        raise ValueError("requirement ids must be unique")
    return out


def _event_dict(event: LoopEvent) -> Dict[str, Any]:
    return {
        "type": "iteration",
        "iteration": event.iteration,
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
            "detail": result.detail, "error": result.error, "artifact": result.artifact}


class LoopJob:
    """One loop run: its identity, event log and cancel flag."""

    def __init__(self, loop_id: str, goal: str, requirements: Sequence[Requirement]):
        self.id = loop_id
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
                "artifact": res.artifact if res else "", "cancel_requested": self.cancel_requested.is_set(),
                "events": len(self._events), "created_at": self.created_at, "resumable": False,
            }

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
        with self._lock:
            active = {i for i, j in self._jobs.items() if not j.finished}
        removed = 0
        try:
            names = os.listdir(self.caps.loop_dir_base)
        except OSError:
            return 0
        for name in names:
            path = os.path.join(self.caps.loop_dir_base, name)
            state_file = os.path.join(path, "state.json")
            if not LOOP_ID_PATTERN.match(name) or name in active or os.path.islink(path):
                continue
            try:
                if os.path.getmtime(state_file) >= cutoff:
                    continue
                state = self._load_state(name)
                if state is None or state.get("status") == "running":
                    continue
                shutil.rmtree(path)
                removed += 1
            except OSError:
                logger.warning("could not prune loop directory %s", path, exc_info=True)
        return removed

    # -- starting ----------------------------------------------------------------------------

    def start(self, goal: str, requirements: Sequence[Union[str, Mapping[str, Any]]], *,
              max_iterations: Optional[int] = None, max_seconds: Optional[float] = None,
              stall_limit: Optional[int] = None, context: Optional[str] = None,
              use_plugins: bool = False) -> LoopJob:
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
        mode = context or ("shared" if getattr(self.agent, "stateful", False) else "fresh")
        if mode == "shared" and not getattr(self.agent, "stateful", False):
            raise ValueError("context 'shared' needs a stateful served agent; use 'fresh'")
        loop_id = f"loop-{uuid.uuid4().hex[:12]}"
        spec = LoopSpec(goal.strip(), reqs, max_iterations=iters, max_seconds=seconds,
                        stall_limit=3 if stall_limit is None else stall_limit, context=mode,
                        use_plugins=bool(use_plugins), loop_id=loop_id,
                        loop_dir=os.path.join(self.caps.loop_dir_base, safe_id(loop_id)))
        return self._launch(LoopJob(loop_id, spec.goal, reqs), spec, resume=False)

    def resume(self, loop_id: str, *, max_iterations: Optional[int] = None,
               max_seconds: Optional[float] = None) -> LoopJob:
        """Continue a checkpointed loop (text requirements only) after a stop or a server restart.

        Raises:
            LoopJobNotFound: no such loop.
            ValueError: it cannot be resumed over HTTP (finished successfully, or it had
                commands/callables) or a value is above the caps.
            LoopJobBusy: another loop is already running.
        """
        state = self._load_state(loop_id)
        if state is None:
            raise LoopJobNotFound(loop_id)
        if state.get("status") == "succeeded":
            raise ValueError("this loop already succeeded")
        reqs_state = state.get("requirements")
        if state.get("uses_approvals") or not isinstance(reqs_state, list) or not reqs_state \
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
        mode = state.get("context")  # saved by the original run; older checkpoints lack it
        if mode not in ("shared", "fresh") or (mode == "shared" and not getattr(self.agent, "stateful", False)):
            mode = "shared" if getattr(self.agent, "stateful", False) else "fresh"
        spec = LoopSpec(state["goal"], reqs, max_iterations=iters, max_seconds=seconds, context=mode,
                        use_plugins=state.get("use_plugins") is True,
                        stall_limit=int(state.get("stall", {}).get("limit", 3)), loop_id=loop_id,
                        loop_dir=os.path.join(self.caps.loop_dir_base, safe_id(loop_id)))
        return self._launch(LoopJob(loop_id, spec.goal, reqs), spec, resume=True)

    def _launch(self, job: LoopJob, spec: LoopSpec, resume: bool) -> LoopJob:
        with self._lock:
            if any(not j.finished for j in self._jobs.values()):
                raise LoopJobBusy("a loop is already running on this server")
            self._jobs[job.id] = job
            self._evict()
            job.thread = threading.Thread(target=self._run, args=(job, spec, resume),
                                          name=f"maki-{job.id}", daemon=True)
            job.thread.start()
        return job

    def _evict(self) -> None:
        finished = sorted((j for j in self._jobs.values() if j.finished), key=lambda j: j.created_at)
        for j in finished[:max(0, len(finished) - _MAX_FINISHED_IN_MEMORY)]:
            del self._jobs[j.id]

    def _run(self, job: LoopJob, spec: LoopSpec, resume: bool) -> None:
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
                result = resume_loop(self.agent, self.judge, spec, on_iteration=hook)
            else:
                result = run_loop(self.agent, self.judge, spec, on_iteration=hook)
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
                    "detail": "", "error": message, "artifact": ""})
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
        return state if isinstance(state, dict) else None

    def snapshot(self, loop_id: str) -> Dict[str, Any]:
        """A job's status: from memory when known, else rebuilt from its checkpoint on disk."""
        job = self.get(loop_id)
        if job is not None:
            snap = job.snapshot()
            if job.finished and snap["status"] != "succeeded":
                state = self._load_state(loop_id)
                snap["resumable"] = state is not None and self._snapshot_from_state(loop_id, state)["resumable"]
            return snap
        state = self._load_state(loop_id)
        if state is None:
            raise LoopJobNotFound(loop_id)
        return self._snapshot_from_state(loop_id, state)

    @staticmethod
    def _snapshot_from_state(loop_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
        status = str(state.get("status", "unknown"))
        reqs = state.get("requirements") if isinstance(state.get("requirements"), list) else []
        return {
            "id": loop_id, "status": "interrupted" if status == "running" else status,
            "goal": state.get("goal", ""),
            "requirements": [{"id": r.get("id"), "text": r.get("text")} for r in reqs if isinstance(r, dict)],
            "iterations": state.get("iteration", 0), "stop_reason": None if status == "running" else status,
            "detail": state.get("detail", ""), "error": state.get("error", ""), "artifact": state.get("artifact", ""),
            "cancel_requested": False, "events": 0, "created_at": None,
            "resumable": status != "succeeded" and not state.get("uses_approvals") and bool(reqs)
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
