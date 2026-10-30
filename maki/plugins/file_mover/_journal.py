"""The move journal: one JSON line per move or undo, in a file per month.

Files are named ``YYYY-MM.jsonl`` and live outside the folders being moved (default ``~/.maki/moves``).
A new month starts a new file and, at that moment, files older than ``retention_months`` calendar months
(the current month counts as the first) are deleted; ``retention_months=None`` keeps everything.
"""

import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

DEFAULT_JOURNAL_DIR = "~/.maki/moves"
_FILE = re.compile(r"^(\d{4})-(\d{2})\.jsonl$")


def fingerprint(path: str) -> Dict[str, Any]:
    """A cheap description of what is at ``path``, to tell later whether it changed (no content is read).

    Files: size and ``mtime_ns``. Links: the target text. Folders: the number of entries below and the
    newest ``mtime_ns`` among them and the folder itself.
    """
    info = os.lstat(path)
    if os.path.islink(path):
        return {"kind": "symlink", "target": os.readlink(path)}
    if not os.path.isdir(path):
        return {"kind": "file", "size": info.st_size, "mtime_ns": info.st_mtime_ns}
    count, newest = 0, info.st_mtime_ns
    for current, dirs, files in os.walk(path, followlinks=False):
        for name in dirs + files:
            count += 1
            try:
                newest = max(newest, os.lstat(os.path.join(current, name)).st_mtime_ns)
            except OSError:
                pass
    return {"kind": "dir", "entries": count, "newest_mtime_ns": newest}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Journal:
    """Append-only move log with monthly files and bounded retention."""

    def __init__(self, directory: Optional[str] = None, retention_months: Optional[int] = 12,
                 clock: Callable[[], datetime] = _utcnow):
        if retention_months is not None and (isinstance(retention_months, bool)
                                             or not isinstance(retention_months, int) or retention_months < 1):
            raise ValueError("retention_months must be an integer >= 1 (or None to keep every file)")
        self.directory = os.path.expanduser(directory or DEFAULT_JOURNAL_DIR)
        self.retention_months = retention_months
        self.clock = clock
        self._lock = threading.Lock()

    # -- files -------------------------------------------------------------------------------

    def _month_files(self) -> List[str]:
        try:
            names = os.listdir(self.directory)
        except OSError:
            return []
        return sorted(n for n in names if _FILE.match(n))

    def _path(self, when: datetime) -> str:
        return os.path.join(self.directory, f"{when.year:04d}-{when.month:02d}.jsonl")

    def _rotate(self, now: datetime) -> None:
        if self.retention_months is None:
            return
        current = now.year * 12 + now.month
        for name in self._month_files():
            year, month = (int(g) for g in _FILE.match(name).groups())
            if current - (year * 12 + month) >= self.retention_months:
                try:
                    os.unlink(os.path.join(self.directory, name))
                except OSError:
                    pass

    # -- writing and reading -------------------------------------------------------------------

    def record(self, action: str, **fields: Any) -> Dict[str, Any]:
        """Append one entry (``action`` is ``"move"`` or ``"undo"``) and return it."""
        now = self.clock()
        entry = {"id": uuid.uuid4().hex[:12], "ts": now.isoformat(), "action": action, **fields}
        line = json.dumps(entry, sort_keys=True) + "\n"
        with self._lock:
            os.makedirs(self.directory, exist_ok=True)
            path = self._path(now)
            if not os.path.exists(path):  # first entry of a month
                self._rotate(now)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(line)
        return entry

    def entries(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Entries newest first, skipping any line that is not valid JSON."""
        found: List[Dict[str, Any]] = []
        for name in reversed(self._month_files()):
            try:
                with open(os.path.join(self.directory, name), encoding="utf-8") as handle:
                    lines = handle.readlines()
            except OSError:
                continue
            for line in reversed(lines):
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                if isinstance(item, dict):
                    found.append(item)
                    if limit is not None and len(found) >= limit:
                        return found
        return found

    def last_record(self) -> Optional[Dict[str, Any]]:
        latest = self.entries(limit=1)
        return latest[0] if latest else None
