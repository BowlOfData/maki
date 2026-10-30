"""Undo for the file_mover plugin: reverse the latest journal entry, or refuse and say why."""

import os
from typing import Any, Dict

from ._engine import MoveError, execute_move, plan_move
from ._journal import Journal, fingerprint
from ._scope import Scope, ScopeError


class UndoRefused(Exception):
    """The latest move cannot (or should not) be reversed; the message is the reason."""


def own_entries(journal: Journal, scope: Scope):
    """Journal entries (newest first) whose source and destination are both inside ``scope``.

    The journal directory may be shared by agents with different scopes; each sees and undoes only its own.
    """
    for entry in journal.iter_entries():
        if scope.includes(str(entry.get("src", ""))) and scope.includes(str(entry.get("dst", ""))):
            yield entry


def _undoable_entry(journal: Journal, scope: Scope) -> Dict[str, Any]:
    entry = next(own_entries(journal, scope), None)
    if entry is None:
        raise UndoRefused("There is nothing to undo")
    if entry.get("action") != "move":
        raise UndoRefused("The latest journal entry is an undo; there is no newer move to undo")
    if entry.get("irreversible"):
        raise UndoRefused("The latest move replaced an existing file, which cannot be restored")
    return entry


def _locate(entry: Dict[str, Any], scope: Scope):
    """Where the item is now and where it came from, both checked against the scope."""
    try:
        return scope.require(entry["dst"], follow_final=False), scope.require(entry["src"], follow_final=False)
    except ScopeError as exc:
        raise UndoRefused(f"Undo is outside the allowed directories: {exc}") from exc


def undo_latest(journal: Journal, scope: Scope) -> Dict[str, Any]:
    """Move the item of the latest ``move`` entry back to where it came from.

    Refuses when there is nothing to undo, the latest entry is itself an undo, the move replaced an
    existing file (that file is gone), the item is no longer where the journal left it or changed since,
    something now occupies the original place, or either place is outside the operator's scope.
    """
    entry = _undoable_entry(journal, scope)
    current, original = _locate(entry, scope)
    if not os.path.lexists(current):
        raise UndoRefused(f"The moved item is no longer at {entry['dst']}")
    if fingerprint(current) != entry.get("fingerprint"):
        raise UndoRefused(f"{entry['dst']} has changed since it was moved; refusing to move it back")
    plan = plan_move(current, original, overwrite=False, create_dirs=True)
    if not plan.ok:
        raise UndoRefused(plan.error)
    try:
        result = execute_move(plan)
    except MoveError as exc:
        raise UndoRefused(str(exc)) from exc
    undo_entry = journal.record("undo", src=current, dst=original, kind=plan.kind, method=result["method"],
                                undoes=entry["id"], fingerprint=fingerprint(original))
    return {"entry": undo_entry, "restored": original, "undone": entry["id"], "method": result["method"]}
