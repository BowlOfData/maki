"""Where the file_mover plugin may touch files.

The operator decides the scope when the plugin is created; nothing the model passes can widen it.
Containment is decided by walking up from the path and comparing each ancestor with the allowed roots
as *file system objects* (``os.path.samefile``), not as strings, so it is right on case-insensitive
file systems (macOS, Windows) and through symlinks.
"""

import os
from typing import Iterable, List, Optional


class ScopeError(ValueError):
    """A path is outside the directories the operator allowed."""


class Scope:
    """``base_dir`` plus optional extra ``allowed_roots``, or everything with ``allow_any_path``."""

    def __init__(self, base_dir: Optional[str] = None, allowed_roots: Optional[Iterable[str]] = None,
                 allow_any_path: bool = False):
        extra = list(allowed_roots or [])
        if allow_any_path and extra:
            raise ValueError("allow_any_path and allowed_roots cannot be combined")
        self.base_dir = os.path.realpath(base_dir if base_dir is not None else os.getcwd())
        self.allow_any_path = bool(allow_any_path)
        self.roots: List[str] = [self.base_dir] + [os.path.realpath(r) for r in extra]

    def resolve(self, path: str, follow_final: bool = True) -> str:
        """Absolute, symlink-free form of ``path`` (relative paths start at ``base_dir``).

        With ``follow_final=False`` the last component is kept as written, so a symlink is
        located by where the link lives, not where it points.
        """
        joined = os.path.join(self.base_dir, path)
        if follow_final:
            return os.path.realpath(joined)
        joined = os.path.normpath(joined)
        return os.path.join(os.path.realpath(os.path.dirname(joined)), os.path.basename(joined))

    def contains(self, resolved: str) -> bool:
        """True if the already-resolved ``resolved`` is inside (or equal to) an allowed root."""
        if self.allow_any_path:
            return True
        current = resolved
        while True:
            if os.path.lexists(current) and any(_same(current, root) for root in self.roots):
                return True
            parent = os.path.dirname(current)
            if parent == current:
                return False
            current = parent

    def require(self, path: str, follow_final: bool = True) -> str:
        """Resolve ``path`` and return it, or raise :class:`ScopeError` if it is out of scope."""
        resolved = self.resolve(path, follow_final)
        if not self.contains(resolved):
            raise ScopeError(f"Path '{path}' resolves outside the allowed directories")
        return resolved


def _same(a: str, b: str) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False
