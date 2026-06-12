"""
StateStore: pluggable persistence layer for WorkflowState.

Implementations
---------------
LocalStateStore   — JSON files in a local directory (default: ~/.maki/workflows/).
                    No extra dependencies.  Good for single-machine dev.

RedisStateStore   — JSON blobs in Redis with an optional TTL.
                    Requires redis-py: pip install "maki[distributed-redis]"
                    Suitable for multi-node deployments where the orchestrator
                    may restart mid-workflow and needs to resume.

Both stores are thread-safe for concurrent task saves within a single workflow.
RedisStateStore's update_task() is not atomic (load→modify→save), so concurrent
callers for the *same* workflow_id should use an external lock.
"""
import hashlib
import json
import logging
import os
import re
import threading
from abc import ABC, abstractmethod
from typing import List, Optional

from ..agents.workflow import TaskStatus, WorkflowState

logger = logging.getLogger(__name__)

# Characters allowed in a workflow ID when used as a filename / Redis key.
_SAFE_ID = re.compile(r"[^a-zA-Z0-9_\-.]")


def _sanitize(workflow_id: str) -> str:
    """Replace unsafe characters so the ID can be used as a filename or key.

    When any character is replaced, a short hash of the original ID is
    appended so distinct IDs that sanitize to the same string (e.g. ``a/b``
    and ``a_b``) cannot collide on one file/key.
    """
    result = _SAFE_ID.sub("_", workflow_id)
    # Replace ".." sequences (path traversal defence) until none remain.
    while ".." in result:
        result = result.replace("..", "_")
    if result != workflow_id:
        digest = hashlib.sha256(workflow_id.encode("utf-8")).hexdigest()[:8]
        result = f"{result}-{digest}"
    return result


def _atomic_write_json(path: str, data: dict) -> None:
    """Write *data* as JSON to *path* atomically (temp file + os.replace).

    A crash mid-write leaves the previous file intact instead of a torn
    checkpoint.
    """
    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


class StateStore(ABC):
    """Abstract persistence contract for WorkflowState."""

    @abstractmethod
    def save_workflow(self, state: WorkflowState) -> None:
        """Persist *state* (full replace)."""
        ...

    @abstractmethod
    def load_workflow(self, workflow_id: str) -> Optional[WorkflowState]:
        """Return the stored state, or None if not found."""
        ...

    @abstractmethod
    def update_task(self, workflow_id: str, task_name: str, update: dict) -> None:
        """Patch specific fields in a single task entry."""
        ...

    @abstractmethod
    def list_workflows(self) -> List[str]:
        """Return all stored workflow IDs."""
        ...

    @abstractmethod
    def delete_workflow(self, workflow_id: str) -> None:
        """Remove a workflow from the store."""
        ...


class LocalStateStore(StateStore):
    """
    Stores each workflow as a JSON file under *base_dir*.

    File name: ``<sanitized_workflow_id>.json``

    Thread-safe: a per-instance lock serialises concurrent saves within
    the same Python process.

    Args:
        base_dir: Directory for workflow files (default: ``~/.maki/workflows``).
    """

    def __init__(self, base_dir: str = "~/.maki/workflows") -> None:
        self.base_dir = os.path.expanduser(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, workflow_id: str) -> str:
        return os.path.join(self.base_dir, f"{_sanitize(workflow_id)}.json")

    def save_workflow(self, state: WorkflowState) -> None:
        path = self._path(state.workflow_id)
        with self._lock:
            _atomic_write_json(path, state.to_dict())
        logger.debug("Saved workflow '%s' to %s", state.workflow_id, path)

    def load_workflow(self, workflow_id: str) -> Optional[WorkflowState]:
        path = self._path(workflow_id)
        if not os.path.exists(path):
            return None
        with self._lock:
            with open(path) as f:
                data = json.load(f)
        return WorkflowState.from_dict(data)

    def update_task(self, workflow_id: str, task_name: str, update: dict) -> None:
        path = self._path(workflow_id)
        with self._lock:
            if not os.path.exists(path):
                raise ValueError(f"Workflow '{workflow_id}' not found in store")
            with open(path) as f:
                data = json.load(f)
            entry = dict(data.get("tasks", {}).get(task_name, {}))
            entry.update(update)
            data.setdefault("tasks", {})[task_name] = entry
            _atomic_write_json(path, data)

    def list_workflows(self) -> List[str]:
        with self._lock:
            names = [
                fname[:-5]
                for fname in os.listdir(self.base_dir)
                if fname.endswith(".json")
            ]
        return names

    def delete_workflow(self, workflow_id: str) -> None:
        path = self._path(workflow_id)
        with self._lock:
            if os.path.exists(path):
                os.remove(path)


class RedisStateStore(StateStore):
    """
    Stores each workflow as a JSON string in Redis with an optional TTL.

    Requires redis-py: ``pip install "maki[distributed-redis]"``

    Args:
        redis_url:  Redis connection string (default: ``redis://localhost:6379``).
        ttl:        Key expiry in seconds (default: 7 days). 0 = no expiry.
        prefix:     Key prefix used for all workflow entries.
        _client:    Inject a pre-built redis client (used in tests; skip URL connect).
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        ttl: int = 86400 * 7,
        prefix: str = "maki:workflow:",
        _client=None,
    ) -> None:
        if _client is not None:
            self._redis = _client
        else:
            try:
                import redis
            except ImportError as e:
                raise ImportError(
                    "RedisStateStore requires redis-py. "
                    'Install it with: pip install "maki[distributed-redis]"'
                ) from e
            self._redis = redis.from_url(redis_url, decode_responses=True)
        self._ttl = ttl
        self._prefix = prefix

    def _key(self, workflow_id: str) -> str:
        return f"{self._prefix}{_sanitize(workflow_id)}"

    def save_workflow(self, state: WorkflowState) -> None:
        serialized = json.dumps(state.to_dict())
        if self._ttl:
            self._redis.setex(self._key(state.workflow_id), self._ttl, serialized)
        else:
            self._redis.set(self._key(state.workflow_id), serialized)

    def load_workflow(self, workflow_id: str) -> Optional[WorkflowState]:
        data = self._redis.get(self._key(workflow_id))
        if data is None:
            return None
        return WorkflowState.from_dict(json.loads(data))

    def update_task(self, workflow_id: str, task_name: str, update: dict) -> None:
        state = self.load_workflow(workflow_id)
        if state is None:
            raise ValueError(f"Workflow '{workflow_id}' not found in store")
        entry = dict(state.tasks.get(task_name, {}))
        entry.update(update)
        state.tasks[task_name] = entry
        self.save_workflow(state)

    def list_workflows(self) -> List[str]:
        plen = len(self._prefix)
        return [k[plen:] for k in self._redis.keys(f"{self._prefix}*")]

    def delete_workflow(self, workflow_id: str) -> None:
        self._redis.delete(self._key(workflow_id))
