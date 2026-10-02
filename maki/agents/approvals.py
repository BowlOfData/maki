"""
Human-approval records for workflow tasks.

A :class:`~maki.agents.workflow.WorkflowTask` created with ``approval=ApprovalSpec(...)``
does not run until a human has approved it. Approval is bound to a *key* (typically a
content hash of whatever is about to be executed), so changing the content invalidates
a previous approval. The workflow parks the task as ``AWAITING_APPROVAL`` and stops
scheduling everything downstream; re-running ``run_workflow`` with the same
``workflow_id`` after :meth:`ApprovalStore.approve` resumes where it stopped.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ApprovalDecision(Enum):
    APPROVED = "approved"
    PENDING = "pending"
    REJECTED = "rejected"


@dataclass
class ApprovalRecord:
    workflow_id: str
    task: str
    key: str
    status: str  # "pending" | "approved" | "rejected"
    summary: str = ""
    requested_at: float = 0.0
    decided_at: Optional[float] = None
    decided_by: Optional[str] = None
    expires_at: Optional[float] = None
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ApprovalSpec:
    """Attach to a WorkflowTask to require human approval before it runs.

    key_fn:     context -> str; the value approval is bound to (e.g. a sha256 of the
                payload the task will execute). Must be deterministic for identical content.
    summary_fn: context -> str; short human-readable description shown when approving.
    ttl_seconds: how long an approval stays valid once granted (None = no expiry).
    """

    key_fn: Callable[[Dict[str, Any]], str]
    summary_fn: Optional[Callable[[Dict[str, Any]], str]] = None
    ttl_seconds: Optional[float] = None


class ApprovalStore(ABC):
    """Persistence + decision API for approvals."""

    @abstractmethod
    def _load(self, workflow_id: str, task: str) -> Optional[ApprovalRecord]: ...

    @abstractmethod
    def _save(self, record: ApprovalRecord) -> None: ...

    @abstractmethod
    def list_pending(self, workflow_id: Optional[str] = None) -> List[ApprovalRecord]: ...

    def get(self, workflow_id: str, task: str) -> Optional[ApprovalRecord]:
        return self._load(workflow_id, task)

    def check(self, workflow_id: str, task: str, key: str, summary: str = "") -> ApprovalDecision:
        """Return the decision for (workflow, task, key), opening a request if needed.

        A record whose key differs from ``key`` is stale (the content changed) and is
        replaced by a fresh pending request, whatever its previous status was.
        An expired approval is likewise replaced.
        """
        rec = self._load(workflow_id, task)
        now = time.time()
        if rec is not None and rec.key == key:
            if rec.status == "rejected":
                return ApprovalDecision.REJECTED
            if rec.status == "approved":
                if rec.expires_at is None or now < rec.expires_at:
                    return ApprovalDecision.APPROVED
                logger.info("Approval for %s/%s expired; re-requesting", workflow_id, task)
            elif rec.status == "pending":
                return ApprovalDecision.PENDING
        self._save(ApprovalRecord(
            workflow_id=workflow_id, task=task, key=key, status="pending",
            summary=summary, requested_at=now,
        ))
        return ApprovalDecision.PENDING

    def approve(self, workflow_id: str, task: str, *, key: Optional[str] = None,
                by: str = "user", ttl_seconds: Optional[float] = None, note: str = "") -> ApprovalRecord:
        """Approve a pending request. If ``key`` is given it must match the pending one."""
        rec = self._require_open(workflow_id, task, key)
        now = time.time()
        rec.status, rec.decided_at, rec.decided_by, rec.note = "approved", now, by, note
        rec.expires_at = now + ttl_seconds if ttl_seconds is not None else None
        self._save(rec)
        return rec

    def reject(self, workflow_id: str, task: str, *, key: Optional[str] = None,
               by: str = "user", note: str = "") -> ApprovalRecord:
        rec = self._require_open(workflow_id, task, key)
        rec.status, rec.decided_at, rec.decided_by, rec.note = "rejected", time.time(), by, note
        self._save(rec)
        return rec

    def _require_open(self, workflow_id: str, task: str, key: Optional[str]) -> ApprovalRecord:
        rec = self._load(workflow_id, task)
        if rec is None:
            raise KeyError(f"no approval request for {workflow_id}/{task}")
        if key is not None and rec.key != key:
            raise ValueError(
                f"approval key mismatch for {workflow_id}/{task}: the content changed since it was shown"
            )
        return rec


class InMemoryApprovalStore(ApprovalStore):
    def __init__(self) -> None:
        self._records: Dict[tuple, ApprovalRecord] = {}
        self._lock = threading.Lock()

    def _load(self, workflow_id: str, task: str) -> Optional[ApprovalRecord]:
        with self._lock:
            rec = self._records.get((workflow_id, task))
            return ApprovalRecord(**rec.to_dict()) if rec else None

    def _save(self, record: ApprovalRecord) -> None:
        with self._lock:
            self._records[(record.workflow_id, record.task)] = ApprovalRecord(**record.to_dict())

    def list_pending(self, workflow_id: Optional[str] = None) -> List[ApprovalRecord]:
        with self._lock:
            return [ApprovalRecord(**r.to_dict()) for (wf, _), r in self._records.items()
                    if r.status == "pending" and (workflow_id is None or wf == workflow_id)]


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


class LocalApprovalStore(ApprovalStore):
    """One JSON file per workflow under ``base_dir`` (default ``~/.maki/approvals``)."""

    def __init__(self, base_dir: str = "~/.maki/approvals") -> None:
        self.base_dir = os.path.expanduser(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, workflow_id: str) -> str:
        return os.path.join(self.base_dir, f"{_safe(workflow_id)}.json")

    def _read(self, workflow_id: str) -> Dict[str, dict]:
        try:
            with open(self._path(workflow_id), encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError:
            return {}

    def _load(self, workflow_id: str, task: str) -> Optional[ApprovalRecord]:
        with self._lock:
            data = self._read(workflow_id).get(task)
        return ApprovalRecord(**data) if data else None

    def _save(self, record: ApprovalRecord) -> None:
        with self._lock:
            data = self._read(record.workflow_id)
            data[record.task] = record.to_dict()
            fd, tmp = tempfile.mkstemp(dir=self.base_dir, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, indent=2)
                os.replace(tmp, self._path(record.workflow_id))
            except BaseException:
                if os.path.exists(tmp):
                    os.unlink(tmp)
                raise

    def list_pending(self, workflow_id: Optional[str] = None) -> List[ApprovalRecord]:
        with self._lock:
            files = ([self._path(workflow_id)] if workflow_id is not None else
                     [os.path.join(self.base_dir, f) for f in sorted(os.listdir(self.base_dir))
                      if f.endswith(".json")])
            out: List[ApprovalRecord] = []
            for path in files:
                try:
                    with open(path, encoding="utf-8") as fh:
                        for rec in json.load(fh).values():
                            if rec.get("status") == "pending":
                                out.append(ApprovalRecord(**rec))
                except FileNotFoundError:
                    continue
            return out
