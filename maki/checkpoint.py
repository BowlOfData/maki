"""Agent checkpoints: a crashed run picks up at its last completed model round.

``Agent(checkpoint="agent.db")`` writes to a stdlib ``sqlite3`` file, one row per agent name:

* after every finished task, the agent's whole state (``Agent.to_dict()``: conversation, summary,
  memory including pinned facts, task history);
* after every completed tool round of a native tool loop, the unfinished task's messages, so
  ``agent.resume()`` can continue without running a finished tool again.

The file holds whole conversations: it is created with ``0600`` permissions, carries a schema
version, and every text is redacted (:func:`redact_value`) before it is written. Redaction means a
secret that appeared in a tool result is not in the file, so a resumed model sees ``***`` there.
"""
import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .connector import redact
from .exceptions import MakiError

SCHEMA_VERSION = 1
# A credential-looking key: ``api_key``, ``access-token``, ``Authorization``, ``apiKey`` ... but not ``keyword``,
# ``monkey_name`` or ``tokenizer``: operator memory and ordinary tool arguments must come back as they went in.
_WORDS = r"(?:api[_-]?key|key|token|secret|passw(?:or)?d|authorization|credentials?|cookie)s?"
_SECRET_KEY = re.compile(rf"(?:^|[^a-z]){_WORDS}(?:$|[^a-z])", re.IGNORECASE)
_CAMEL_SECRET_KEY = re.compile(r"[a-z](?:Key|Token|Secret|Password|Passwd|Cookie)s?$")
_MAX_DEPTH = 12


class CheckpointError(MakiError):
    """The checkpoint file is unusable (newer schema, corrupt row) or the agent has not resumed it yet."""


def _is_secret_key(key: str) -> bool:
    return bool(_SECRET_KEY.search(key) or _CAMEL_SECRET_KEY.search(key))


def redact_value(value: Any, depth: int = 0) -> Any:
    """*value* (JSON-like) with credential-looking query parameters masked in every string and the value
    of every credential-looking dict key replaced by ``***`` when that value is a string."""
    if depth > _MAX_DEPTH:
        return "***"
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        # Only a string under such a key is a secret: counters like ``conversation_token_budget`` stay numbers.
        return {str(k): ("***" if _is_secret_key(str(k)) and isinstance(v, str)
                         else redact_value(v, depth + 1)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_value(v, depth + 1) for v in value]
    return value


@dataclass
class Checkpoint:
    """What a store holds for one agent."""
    state: Dict[str, Any]
    inflight: Optional[Dict[str, Any]]
    saved: float


class CheckpointStore:
    """A sqlite file with one checkpoint per agent name. Safe to share between agents with different names;
    two live agents with the same name overwrite each other (last write wins)."""

    def __init__(self, path: "str | os.PathLike[str]"):
        if sqlite3.sqlite_version_info < (3, 24):  # the saves are upserts
            raise CheckpointError(f"checkpoints need sqlite 3.24 or newer; this Python links {sqlite3.sqlite_version}")
        self.path = os.path.abspath(os.fspath(path))
        self._lock = threading.Lock()
        try:
            directory = os.path.dirname(self.path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            if not os.path.exists(self.path):
                os.close(os.open(self.path, os.O_CREAT | os.O_WRONLY, 0o600))  # private before sqlite writes anything
            elif os.path.isfile(self.path) and os.stat(self.path).st_mode & 0o077:
                os.chmod(self.path, 0o600)  # a file made earlier by another tool holds whole conversations now
        except OSError as exc:
            raise CheckpointError(f"cannot create the checkpoint file {self.path}: {exc}") from exc
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        try:
            conn = sqlite3.connect(self.path, timeout=30)
            conn.execute("PRAGMA busy_timeout = 30000")
        except sqlite3.DatabaseError as exc:
            raise CheckpointError(f"cannot open {self.path}: {exc}") from exc
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                version = conn.execute("PRAGMA user_version").fetchone()[0]
                if version > SCHEMA_VERSION:
                    raise CheckpointError(
                        f"{self.path} was written by a newer maki (checkpoint schema {version}, "
                        f"this one reads {SCHEMA_VERSION}); upgrade maki or use another file")
                with conn:
                    conn.execute("CREATE TABLE IF NOT EXISTS checkpoint ("
                                 "agent TEXT PRIMARY KEY, saved REAL NOT NULL, state TEXT NOT NULL, inflight TEXT)")
                    if version < SCHEMA_VERSION:
                        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            except sqlite3.DatabaseError as exc:
                raise CheckpointError(f"{self.path} is not a usable checkpoint file: {exc}") from exc
            finally:
                conn.close()

    @staticmethod
    def _dump(value: Any) -> str:
        # ASCII-escaped on purpose: sqlite refuses text with a lone surrogate (a cut emoji), and a refused write would
        # mean no checkpoint at all; json.loads brings the character back.
        return json.dumps(redact_value(value), default=str)

    def save_state(self, agent: str, state: Dict[str, Any]) -> None:
        """Store a finished task's state; nothing is in flight any more."""
        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    conn.execute(
                        "INSERT INTO checkpoint(agent, saved, state, inflight) VALUES(?, ?, ?, NULL) "
                        "ON CONFLICT(agent) DO UPDATE SET saved=excluded.saved, state=excluded.state, inflight=NULL",
                        (agent, time.time(), self._dump(state)))
            finally:
                conn.close()

    def save_inflight(self, agent: str, inflight: Dict[str, Any],
                      state: "Dict[str, Any] | Callable[[], Dict[str, Any]]") -> None:
        """Store the unfinished task. *state* (a dict, or a function returning one) is only used when the
        agent has no row yet; otherwise the last finished state stays, and a big state is not rebuilt every round."""
        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    exists = conn.execute("SELECT 1 FROM checkpoint WHERE agent = ?", (agent,)).fetchone()
                    if exists:
                        conn.execute("UPDATE checkpoint SET saved=?, inflight=? WHERE agent = ?",
                                     (time.time(), self._dump(inflight), agent))
                    else:
                        # ON CONFLICT: another process may have created the row since the SELECT
                        conn.execute("INSERT INTO checkpoint(agent, saved, state, inflight) VALUES(?, ?, ?, ?) "
                                     "ON CONFLICT(agent) DO UPDATE SET saved=excluded.saved, inflight=excluded.inflight",
                                     (agent, time.time(), self._dump(state() if callable(state) else state),
                                      self._dump(inflight)))
            finally:
                conn.close()

    def load(self, agent: str) -> Optional[Checkpoint]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT saved, state, inflight FROM checkpoint WHERE agent = ?", (agent,)).fetchone()
            except sqlite3.DatabaseError as exc:
                raise CheckpointError(f"cannot read {self.path}: {exc}") from exc
            finally:
                conn.close()
        if row is None:
            return None
        try:
            state = json.loads(row[1])
            inflight = json.loads(row[2]) if row[2] else None
        except ValueError as exc:
            raise CheckpointError(f"the checkpoint of agent '{agent}' in {self.path} is damaged: {exc}") from exc
        if not isinstance(state, dict) or not (inflight is None or isinstance(inflight, dict)):
            raise CheckpointError(f"the checkpoint of agent '{agent}' in {self.path} is damaged")
        return Checkpoint(state=state, inflight=inflight, saved=float(row[0]))

    def clear_inflight(self, agent: str) -> None:
        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    conn.execute("UPDATE checkpoint SET inflight=NULL WHERE agent = ?", (agent,))
            except sqlite3.DatabaseError as exc:
                raise CheckpointError(f"cannot write {self.path}: {exc}") from exc
            finally:
                conn.close()

    def delete(self, agent: str) -> None:
        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    conn.execute("DELETE FROM checkpoint WHERE agent = ?", (agent,))
            except sqlite3.DatabaseError as exc:
                raise CheckpointError(f"cannot write {self.path}: {exc}") from exc
            finally:
                conn.close()

    def agents(self) -> List[str]:
        with self._lock:
            conn = self._connect()
            try:
                return [r[0] for r in conn.execute("SELECT agent FROM checkpoint ORDER BY agent")]
            except sqlite3.DatabaseError as exc:
                raise CheckpointError(f"cannot read {self.path}: {exc}") from exc
            finally:
                conn.close()
