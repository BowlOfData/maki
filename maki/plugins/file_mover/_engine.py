"""The move engine behind the file_mover plugin: plan a move without touching anything, then execute it.

Only the standard library is used (no shell), and the paths arriving here are already resolved and
checked against the operator's scope. ``dst`` is always the full new path of the item.

* Same volume: an atomic rename (``os.replace`` when overwriting a file).
* Different volume (``EXDEV`` on POSIX, ``ERROR_NOT_SAME_DEVICE`` on Windows): copy next to the
  destination, verify, move into place, and only then remove the source. A failure at any step before
  the final removal leaves the source untouched and the destination as it was.
* Case-only renames (``a.txt`` to ``A.txt`` on a case-insensitive file system) go through a temporary name.
* A symlink is moved as a link; it is never followed.
"""

import errno
import hashlib
import os
import shutil
import stat
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

_WINDOWS_NOT_SAME_DEVICE = 17  # ERROR_NOT_SAME_DEVICE
_CHUNK = 1024 * 1024


class MoveError(Exception):
    """A move failed. ``partial`` means the destination is complete but the source could not be removed."""

    def __init__(self, message: str, partial: bool = False):
        super().__init__(message)
        self.partial = partial


@dataclass
class MovePlan:
    """What a move would do. ``error`` is set (with a ``reason`` code) when it cannot be done."""
    src: str
    dst: str
    kind: str = ""               # "file" | "dir" | "symlink"
    same_volume: bool = True
    case_only: bool = False
    will_overwrite: bool = False
    create_dirs: bool = False
    steps: List[str] = field(default_factory=list)
    reason: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _fail(plan: MovePlan, reason: str, message: str) -> MovePlan:
    plan.reason, plan.error = reason, message
    return plan


def kind_of(path: str) -> str:
    mode = os.lstat(path).st_mode
    if stat.S_ISLNK(mode):
        return "symlink"
    return "dir" if stat.S_ISDIR(mode) else "file"


def _same_object(a: str, b: str) -> bool:
    try:
        return os.path.samestat(os.lstat(a), os.lstat(b))
    except OSError:
        return False


def _is_inside(path: str, folder: str) -> bool:
    """True if ``path`` lies below ``folder``, comparing file system objects (not strings)."""
    current = os.path.dirname(path)
    while True:
        if os.path.lexists(current) and _same_object(current, folder):
            return True
        parent = os.path.dirname(current)
        if parent == current:
            return False
        current = parent


def _nearest_existing(path: str) -> str:
    while not os.path.lexists(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return path


def plan_move(src: str, dst: str, overwrite: bool = False, create_dirs: bool = False,
              dst_trailing_separator: bool = False) -> MovePlan:
    """Work out what moving ``src`` to ``dst`` would do, without changing anything."""
    plan = MovePlan(src=src, dst=dst, create_dirs=create_dirs)
    if dst_trailing_separator:
        return _fail(plan, "dst_is_folder", "dst must be the full new path of the item, not a folder "
                     "(no trailing separator); e.g. 'docs/report.txt'")
    if not os.path.lexists(src):
        return _fail(plan, "missing_source", f"Source does not exist: {src}")
    plan.kind = kind_of(src)

    exists = os.path.lexists(dst)
    if exists and _same_object(src, dst):
        outcome = _same_object_outcome(plan)
        if outcome is not None:
            return outcome
    if plan.kind == "dir" and _is_inside(dst, src):
        return _fail(plan, "into_itself", "Cannot move a directory into itself")

    problem = _check_conflict(plan, overwrite) if exists else _check_parent(plan, create_dirs)
    if problem is not None:
        return problem
    return _describe(plan)


def _same_object_outcome(plan: MovePlan) -> Optional[MovePlan]:
    """The plan when ``src`` and ``dst`` are one object (same path, or a case-only rename); None for a
    hard link, which counts as an existing destination."""
    src_name, dst_name = os.path.basename(plan.src), os.path.basename(plan.dst)
    same_folder = _same_object(os.path.dirname(plan.src), os.path.dirname(plan.dst))
    if src_name == dst_name and same_folder:
        return _fail(plan, "same_path", "Source and destination are the same")
    if same_folder and src_name.casefold() == dst_name.casefold():
        plan.case_only = True
        plan.steps = ["rename through a temporary name (case-only change)"]
        return plan
    return None


def _check_conflict(plan: MovePlan, overwrite: bool) -> Optional[MovePlan]:
    if not overwrite:
        return _fail(plan, "conflict",
                     f"Destination already exists: {plan.dst} (pass overwrite=True to replace a file)")
    if kind_of(plan.dst) == "dir":
        return _fail(plan, "dir_overwrite", "Destination is a directory; overwrite only replaces a file with a file")
    if plan.kind == "dir":
        return _fail(plan, "kind_mismatch", "Cannot replace a file with a directory")
    plan.will_overwrite = True
    return None


def _check_parent(plan: MovePlan, create_dirs: bool) -> Optional[MovePlan]:
    parent = os.path.dirname(plan.dst)
    if os.path.isdir(parent):
        return None
    if not create_dirs:
        return _fail(plan, "missing_parent",
                     f"Destination folder does not exist: {parent} (pass create_dirs=True to create it)")
    if not os.path.isdir(_nearest_existing(parent)):
        return _fail(plan, "missing_parent", f"No existing folder above {parent}")
    return None


def _describe(plan: MovePlan) -> MovePlan:
    parent = os.path.dirname(plan.dst)
    plan.same_volume = os.lstat(plan.src).st_dev == os.stat(_nearest_existing(parent)).st_dev
    if plan.create_dirs and not os.path.isdir(parent):
        plan.steps.append("create the missing destination folders")
    if plan.will_overwrite:
        plan.steps.append("replace the existing destination file (cannot be undone)")
    if plan.same_volume:
        plan.steps.append("rename within the volume")
    else:
        plan.steps.append("copy to a temporary name next to the destination (different volume)")
        plan.steps.append("verify the copy (sizes and SHA-256)")
        plan.steps.append("move the copy into place, then remove the source")
    return plan


def is_cross_device(error: OSError) -> bool:
    return error.errno == errno.EXDEV or getattr(error, "winerror", None) == _WINDOWS_NOT_SAME_DEVICE


def _rename_direct(src: str, dst: str, overwrite: bool) -> None:
    """The first attempt: a plain rename. Raises ``OSError`` (``EXDEV`` when the volumes differ)."""
    if overwrite:
        os.replace(src, dst)
        return
    if os.name != "nt" and os.path.lexists(dst):  # POSIX rename would replace silently
        raise FileExistsError(errno.EEXIST, "Destination appeared during the move", dst)
    os.rename(src, dst)


def _temp_sibling(dst: str) -> str:
    """A hidden, unique temporary name next to ``dst`` (fixed length, so a long file name cannot overflow)."""
    return os.path.join(os.path.dirname(dst), f".maki-moving-{uuid.uuid4().hex[:16]}")


def _case_only_rename(src: str, dst: str) -> None:
    temp = _temp_sibling(src)
    os.rename(src, temp)
    try:
        os.rename(temp, dst)
    except OSError:
        os.rename(temp, src)  # put it back
        raise


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _listing(root: str) -> Dict[str, Any]:
    """Relative path -> ("dir",) | ("link", target) | ("file", size, sha256) for a whole tree."""
    entries: Dict[str, Any] = {}
    for current, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            full = os.path.join(current, name)
            rel = os.path.relpath(full, root)
            mode = os.lstat(full).st_mode
            if stat.S_ISLNK(mode):
                entries[rel] = ("link", os.readlink(full))
            elif stat.S_ISDIR(mode):
                entries[rel] = ("dir",)
            else:
                entries[rel] = ("file", os.path.getsize(full), _sha256(full))
    return entries


def _verify(src: str, copy: str, kind: str) -> None:
    if kind == "file":
        if os.path.getsize(src) != os.path.getsize(copy) or _sha256(src) != _sha256(copy):
            raise MoveError("The copy does not match the source; nothing was removed")
    elif kind == "symlink":
        if os.readlink(src) != os.readlink(copy):
            raise MoveError("The copied link does not match the source; nothing was removed")
    elif _listing(src) != _listing(copy):
        raise MoveError("The copied folder does not match the source; nothing was removed")


def _remove(path: str) -> None:
    if os.path.lexists(path):
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.unlink(path)


def _copy_to_temp(plan: MovePlan, temp: str) -> None:
    if plan.kind == "symlink":
        os.symlink(os.readlink(plan.src), temp)
    elif plan.kind == "dir":
        shutil.copytree(plan.src, temp, symlinks=True, copy_function=shutil.copy2)
    else:
        shutil.copy2(plan.src, temp)


def _copy_then_delete(plan: MovePlan) -> None:
    temp = _temp_sibling(plan.dst)
    try:
        _copy_to_temp(plan, temp)
        _verify(plan.src, temp, plan.kind)
        if plan.will_overwrite:
            os.replace(temp, plan.dst)
        else:
            _rename_direct(temp, plan.dst, False)
    except BaseException:
        _remove(temp)
        raise
    try:
        _remove(plan.src)
    except OSError as exc:
        raise MoveError(f"Copied to {plan.dst}, but could not remove the source: {exc}", partial=True) from exc


def _perform(plan: MovePlan) -> Dict[str, Any]:
    if plan.case_only:
        _case_only_rename(plan.src, plan.dst)
        return {"method": "rename"}
    try:
        _rename_direct(plan.src, plan.dst, plan.will_overwrite)
        return {"method": "rename"}
    except OSError as exc:
        if not is_cross_device(exc):
            raise
    _copy_then_delete(plan)
    return {"method": "copy"}


def execute_move(plan: MovePlan) -> Dict[str, Any]:
    """Carry out a plan. Returns ``{"method": "rename" | "copy"}`` or raises :class:`MoveError`."""
    if not plan.ok:
        raise MoveError(plan.error)
    created = _missing_folders(os.path.dirname(plan.dst)) if plan.create_dirs else []
    try:
        if created:
            os.makedirs(os.path.dirname(plan.dst), exist_ok=True)
        return _perform(plan)
    except MoveError as exc:
        if not exc.partial:
            _remove_empty_folders(created)
        raise
    except OSError as exc:
        _remove_empty_folders(created)
        raise MoveError(f"Could not move: {exc.strerror or exc}") from exc


def _missing_folders(folder: str) -> List[str]:
    """The folders ``os.makedirs(folder)`` would have to create, outermost first."""
    missing: List[str] = []
    while folder and not os.path.lexists(folder):
        missing.append(folder)
        parent = os.path.dirname(folder)
        if parent == folder:
            break
        folder = parent
    return list(reversed(missing))


def _remove_empty_folders(created: List[str]) -> None:
    """Undo the folder creation of a move that failed (innermost first; only if still empty)."""
    for folder in reversed(created):
        try:
            os.rmdir(folder)
        except OSError:
            break
