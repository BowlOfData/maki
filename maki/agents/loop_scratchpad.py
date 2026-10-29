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

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_ENGINE = "<!-- maki:engine (engine-owned: edits here are reverted) -->"
_FEEDBACK = "<!-- maki:feedback (judge feedback, replaced each round; you may edit) -->"
_NOTES = "<!-- maki:notes (agent notes, appended; you may edit) -->"
_END = "<!-- maki:end -->"
_SAFE = re.compile(r"[^a-zA-Z0-9_\-.]")


def safe_id(value: str) -> str:
    """Make ``value`` usable as a directory name (no separators, no traversal)."""
    cleaned = _SAFE.sub("_", value).replace("..", "_").strip(".")
    return cleaned or "loop"


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
        self.engine_log.append(line)

    def set_feedback(self, text: str) -> None:
        self.feedback = text.strip()

    def append_note(self, text: str) -> None:
        text = text.strip()
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
