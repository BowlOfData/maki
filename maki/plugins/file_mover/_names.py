"""Portable file-name checks for the file_mover plugin.

Some names are legal on one operating system and impossible on another (``CON`` or a trailing dot on
Windows). :func:`check_path` applies the Windows rules when asked to, so the same call gives the same
answer on any host; everything else is refused everywhere.
"""

import os
from pathlib import PurePath
from typing import Optional

MAX_NAME_CHARS = 255

_WINDOWS_RESERVED = frozenset(
    ["CON", "PRN", "AUX", "NUL"]
    + [f"COM{n}" for n in range(1, 10)]
    + [f"LPT{n}" for n in range(1, 10)]
)
_WINDOWS_FORBIDDEN = frozenset('<>:"|?*\\')


def windows_rules_by_default() -> bool:
    """True on a Windows host, where these rules are the file system's own."""
    return os.name == "nt"


def check_name(name: str, windows_rules: bool) -> Optional[str]:
    """Why ``name`` (one path component) cannot be used, or None if it is fine."""
    if "\x00" in name:
        return "a name cannot contain a NUL byte"
    if len(name) > MAX_NAME_CHARS:
        return f"name is longer than {MAX_NAME_CHARS} characters"
    if not windows_rules:
        return None
    bad = sorted({c for c in name if c in _WINDOWS_FORBIDDEN or ord(c) < 32})
    if bad:
        return f"name contains characters Windows does not allow: {''.join(repr(c) for c in bad)}"
    if name.endswith((".", " ")):
        return "Windows does not allow a name ending in a dot or a space"
    if name.split(".", 1)[0].rstrip(" ").upper() in _WINDOWS_RESERVED:
        return f"'{name}' is a reserved device name on Windows"
    return None


def check_path(path: str, windows_rules: bool) -> Optional[str]:
    """Why ``path`` cannot be created, or None. The anchor (``/`` or ``C:\\``) and ``.``/``..`` are skipped."""
    pure = PurePath(path)
    for part in pure.parts[1 if pure.anchor else 0:]:
        if part in (".", ".."):
            continue
        problem = check_name(part, windows_rules)
        if problem:
            return problem
    return None
