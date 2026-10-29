"""
Scratchpad and checkpoint store for agent loops (see :mod:`maki.agents.loop`).

The scratchpad is a markdown file with three owned sections:

* **engine log**: written only by the engine (iteration history, pass/fail per
  requirement). Human edits to it are reported and reverted.
* **judge feedback**: replaced by the judge's feedback every iteration. Editable.
* **agent notes**: appended from the agent's ``NOTES:`` trailer. Editable.

A human can edit the feedback and notes sections while the loop runs; the engine
re-reads the file at the start of each iteration. Writes are atomic.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile

try:  # POSIX
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None
    import msvcrt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_ENGINE = "<!-- maki:engine (engine-owned: edits here are reverted) -->"
_FEEDBACK = "<!-- maki:feedback (judge feedback, replaced each round; you may edit) -->"
_NOTES = "<!-- maki:notes (agent notes, appended; you may edit) -->"
_END = "<!-- maki:end -->"
_SAFE = re.compile(r"[^a-zA-Z0-9_\-.]")


def safe_id(value: str) -> str:
    """Make ``value`` usable as a directory name (no separators, no traversal)."""
    cleaned = _SAFE.sub("_", value).replace("..", "_").strip(".") or "loop"
    if cleaned != value:  # distinct ids that sanitise alike (a/b vs a_b) must not share a directory
        cleaned = f"{cleaned}-{hashlib.sha256(value.encode('utf-8')).hexdigest()[:8]}"
    return cleaned


def _neutralise(text: str) -> str:
    """Stop content (command output, notes) from forging a section marker."""
    return text.replace("<!-- maki:", "<!-- maki-")


def atomic_write(path: str, text: str) -> None:
    """Write ``text`` to ``path`` via a temp file and ``os.replace`` (no torn files)."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def default_loop_dir(loop_id: str) -> str:
    return os.path.join(os.path.expanduser("~/.maki/loops"), safe_id(loop_id))


class LoopBusyError(RuntimeError):
    """Another process (or run) already holds this loop."""


class LoopLock:
    """Exclusive, non-blocking lock on a loop directory (``run.lock``).

    An OS file lock, so the operating system releases it when the holder dies; a crashed
    run never leaves a stale lock behind. The holder's pid is written into the file for
    the error message only.
    """

    def __init__(self, directory: str):
        self.path = os.path.join(directory, "run.lock")
        self._fd: Optional[int] = None

    def acquire(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:  # pragma: no cover - Windows
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            holder = os.pread(fd, 32, 0).decode("ascii", "ignore").strip() or "unknown"
            os.close(fd)
            raise LoopBusyError(f"loop is already running (pid {holder}); lock file {self.path}") from None
        os.ftruncate(fd, 0)
        os.pwrite(fd, str(os.getpid()).encode("ascii"), 0)
        self._fd = fd

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            if fcntl is not None:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            else:  # pragma: no cover - Windows
                msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
        finally:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "LoopLock":
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


@dataclass
class ReloadReport:
    """What :meth:`Scratchpad.reload` found in the file."""
    engine_tampered: bool = False
    feedback_edited: bool = False
    notes_edited: bool = False


@dataclass
class Scratchpad:
    """The loop's working memory, kept in memory and mirrored to ``path``."""
    path: str
    loop_id: str
    engine_log: List[str] = field(default_factory=list)
    feedback: str = ""
    notes: str = ""

    def append_engine(self, line: str) -> None:
        self.engine_log.append(" / ".join(_neutralise(line).splitlines()) or "-")

    def set_feedback(self, text: str) -> None:
        self.feedback = _neutralise(text).strip()

    def append_note(self, text: str) -> None:
        text = _neutralise(text).strip()
        if text:
            self.notes = f"{self.notes}\n{text}".strip()

    def render(self) -> str:
        engine = "\n".join(self.engine_log)
        return (f"# Loop scratchpad: {self.loop_id}\n\n{_ENGINE}\n{engine}\n\n{_FEEDBACK}\n{self.feedback}\n\n"
                f"{_NOTES}\n{self.notes}\n\n{_END}\n")

    def flush(self) -> None:
        atomic_write(self.path, self.render())

    def reload(self) -> ReloadReport:
        """Re-read the file: keep human edits to feedback/notes, revert the engine section.

        A missing or mangled file is rewritten from memory and reported as tampering.
        """
        report = ReloadReport()
        try:
            with open(self.path, encoding="utf-8") as f:
                text = f.read()
        except FileNotFoundError:
            self.flush()
            return report
        sections = self._parse(text)
        if sections is None:
            report.engine_tampered = True
            self.flush()
            return report
        engine, feedback, notes = sections
        if engine != "\n".join(self.engine_log).strip():
            report.engine_tampered = True
        if feedback != self.feedback:
            report.feedback_edited = True
            self.feedback = feedback
        if notes != self.notes:
            report.notes_edited = True
            self.notes = notes
        if report.engine_tampered:
            self.flush()
        return report

    @staticmethod
    def _parse(text: str) -> Optional[tuple]:
        pattern = re.compile(
            re.escape(_ENGINE) + r"\n(.*?)\n\n" + re.escape(_FEEDBACK) + r"\n(.*?)\n\n" + re.escape(_NOTES)
            + r"\n(.*?)\n\n" + re.escape(_END), re.S)
        m = pattern.search(text)
        if not m:
            return None
        return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()


class LoopStore:
    """Checkpoints a loop's state as JSON next to its scratchpad (atomic writes)."""

    def __init__(self, directory: str):
        self.directory = directory
        self.path = os.path.join(directory, "state.json")

    def save(self, state: Dict[str, Any]) -> None:
        atomic_write(self.path, json.dumps(state, indent=2, default=str))

    def load(self) -> Optional[Dict[str, Any]]:
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return None
