"""
File Mover Plugin for Maki Framework

Moves and renames files and directories the same way on Linux, macOS and Windows (pure Python, no shell),
keeps a monthly journal of every move and can undo the latest one.

The operator, not the model, decides where it may work: ``base_dir`` by default, plus any
``allowed_roots``, or everywhere with ``allow_any_path=True``. The tool methods take no scope arguments.
"""

import logging
import os
from typing import Any, Dict, Iterable, Optional

from ._engine import MoveError, execute_move, plan_move
from ._journal import Journal, fingerprint
from ._names import check_path, windows_rules_by_default
from ._scope import Scope, ScopeError
from ._undo import UndoRefused, undo_latest

_MAX_LIST = 100


class FileMover:
    """Move or rename files and directories with a preview, a journal and a one-step undo.

    Args:
        maki_instance: Optional Maki instance (unused; part of the plugin contract).
        base_dir: Root that relative paths start from and that is always allowed (default: the current
            directory).
        allowed_roots: Extra directories that are also allowed.
        allow_any_path: Allow every location (cannot be combined with ``allowed_roots``). A move can
            then touch anything the process can reach; only enable it deliberately.
        journal_dir: Where the monthly journal files live (default ``~/.maki/moves``).
        retention_months: Journal files older than this many calendar months are deleted when a new
            month starts (default 12; ``None`` keeps everything).
        strict_names: Apply the Windows file-name rules on every OS. By default they apply only on a
            Windows host.
    """

    # Methods callable by the LLM via TOOL: directives. Moving and undoing change the file system
    # and require Agent(allow_dangerous_tools=True).
    ALLOWED_METHODS = ["move", "undo_last", "list_moves"]
    DANGEROUS_METHODS = ["move", "undo_last"]

    def __init__(self, maki_instance=None, base_dir: Optional[str] = None,
                 allowed_roots: Optional[Iterable[str]] = None, allow_any_path: bool = False,
                 journal_dir: Optional[str] = None, retention_months: Optional[int] = 12,
                 strict_names: bool = False, clock=None):
        self.maki = maki_instance
        self.logger = logging.getLogger(__name__)
        self.scope = Scope(base_dir=base_dir, allowed_roots=allowed_roots, allow_any_path=allow_any_path)
        journal_args = {} if clock is None else {"clock": clock}
        self.journal = Journal(journal_dir, retention_months=retention_months, **journal_args)
        self.windows_rules = bool(strict_names) or windows_rules_by_default()
        self.logger.info(f"FileMover initialized (base_dir='{self.scope.base_dir}', "
                         f"any_path={self.scope.allow_any_path})")

    @staticmethod
    def _failure(message: str, **extra: Any) -> Dict[str, Any]:
        return {"success": False, "error": message, **extra}

    @staticmethod
    def _as_flag(value: Any) -> Optional[bool]:
        """A real bool, or the strings "true"/"false" (tool schemas describe every argument as a string)."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        return None

    def _argument_problem(self, src: Any, dst: Any, flags: Dict[str, Any]) -> Optional[str]:
        for name, value in (("src", src), ("dst", dst)):
            if not isinstance(value, str) or not value.strip():
                return f"{name} must be a non-empty string"
        for name, value in flags.items():
            if self._as_flag(value) is None:
                return f"{name} must be true or false"
        return None

    def move(self, src: str, dst: str, overwrite: bool = False, dry_run: bool = False,
             create_dirs: bool = False) -> Dict[str, Any]:
        """Move or rename a file or directory: src is the existing path, dst is the full new path (not a folder).

        Args:
            src: The existing file or directory (a symlink is moved as a link).
            dst: The full new path of the item (not a folder to move it into).
            overwrite: Replace an existing destination *file* with a file. This cannot be undone.
            dry_run: Only report what would happen; nothing changes.
            create_dirs: Create missing destination folders.

        The flags may be real booleans or the strings ``"true"`` / ``"false"``: tool schemas describe every
        argument as a string, so a model often sends them that way.

        Returns:
            ``success``, and on success the resolved ``src``/``dst``, ``kind``, ``method`` (``rename`` or
            ``copy`` across volumes), ``undoable`` and the journal id; with ``dry_run`` the full ``plan``.
            On failure ``error`` says why.
        """
        problem = self._argument_problem(src, dst, {"overwrite": overwrite, "dry_run": dry_run,
                                                    "create_dirs": create_dirs})
        if problem:
            return self._failure(problem)
        overwrite, dry_run, create_dirs = (self._as_flag(v) for v in (overwrite, dry_run, create_dirs))
        name_problem = check_path(dst, self.windows_rules)
        if name_problem:
            return self._failure(f"Invalid destination name: {name_problem}")
        try:
            resolved_src = self.scope.require(src, follow_final=False)
            resolved_dst = self.scope.require(dst, follow_final=False)
        except ScopeError as exc:
            self.logger.warning(str(exc))
            return self._failure(str(exc))

        separators = (os.sep, os.altsep) if os.altsep else (os.sep,)
        plan = plan_move(resolved_src, resolved_dst, overwrite=overwrite, create_dirs=create_dirs,
                         dst_trailing_separator=dst.endswith(separators) or dst.endswith("/"))
        if dry_run:
            return {"success": plan.ok, "dry_run": True, "plan": plan.to_dict(), "error": plan.error or None}
        if not plan.ok:
            return self._failure(plan.error, reason=plan.reason)
        try:
            result = execute_move(plan)
        except MoveError as exc:
            self.logger.error(f"Move failed: {exc}")
            return self._failure(str(exc), partial=exc.partial)
        return self._record(plan, result)

    def _record(self, plan, result: Dict[str, Any]) -> Dict[str, Any]:
        response = {"success": True, "src": plan.src, "dst": plan.dst, "kind": plan.kind,
                    "method": result["method"], "undoable": not plan.will_overwrite, "error": None}
        try:
            entry = self.journal.record("move", src=plan.src, dst=plan.dst, kind=plan.kind,
                                        method=result["method"], irreversible=plan.will_overwrite,
                                        fingerprint=fingerprint(plan.dst))
            response["journal_id"] = entry["id"]
        except OSError as exc:  # the move happened; only the undo record is missing
            self.logger.error(f"Could not write the move journal: {exc}")
            response.update(undoable=False, warning=f"The move was done but not journaled ({exc}); it cannot be undone")
        return response

    def undo_last(self) -> Dict[str, Any]:
        """Undo the most recent move by moving the item back (only the latest move; refused if the item changed).

        Refuses when there is nothing to undo, the latest entry is an undo, the move replaced a file,
        the item changed or moved since, or the original place is taken or out of scope.
        """
        try:
            result = undo_latest(self.journal, self.scope)
        except UndoRefused as exc:
            return self._failure(str(exc))
        return {"success": True, "restored": result["restored"], "undone": result["undone"],
                "method": result["method"], "error": None}

    def list_moves(self, limit: int = 10) -> Dict[str, Any]:
        """List the most recent moves and undos from the journal, newest first (limit is how many, 1 to 100)."""
        if isinstance(limit, str) and limit.strip().isdigit():
            limit = int(limit)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_LIST:
            return self._failure(f"limit must be an integer between 1 and {_MAX_LIST}")
        keys = ("id", "ts", "action", "src", "dst", "kind", "method", "irreversible", "undoes")
        moves = [{k: e[k] for k in keys if k in e} for e in self.journal.entries(limit=limit)]
        return {"success": True, "moves": moves, "count": len(moves), "error": None}


def register_plugin(maki_instance=None, **options):
    """Register the FileMover plugin with the Maki framework (``options`` are the constructor options)."""
    return FileMover(maki_instance, **options)
