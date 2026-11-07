"""
Agent Skills: instructions an agent loads only when a task needs them.

A skill is a folder with a ``SKILL.md`` (YAML front matter with ``name`` and ``description``, then Markdown
instructions) and optional ``scripts/``, ``references/`` and ``assets/``. It is the open format used by Claude
Code, Codex and other agent tools, so a skill written for one works in the others.

The agent sees only each skill's name and description (the *catalog*, a few dozen tokens per skill). When a task
matches, the model calls the ``skills.load`` tool and receives the full instructions; ``skills.read`` fetches a
file bundled with the skill. That keeps small context windows small.

    agent = Agent("helper", backend, skills=True)                       # user skills only
    agent = Agent("helper", backend, skills=["./my-skills"])            # explicit folders
    agent = Agent("helper", backend, skills=SkillRegistry.discover(trust_project=True))
    agent.execute_task("Prepare the release notes", use_plugins=True)

Where skills are looked for, first match wins:

1. folders you pass explicitly (trusted);
2. project folders, from the working directory up to the git root, each checked as ``.maki/skills``,
   ``.agents/skills`` then ``.claude/skills`` (**only with** ``trust_project=True``: a repository you just cloned
   can put instructions in front of your model, so project skills are opt-in);
3. user folders ``~/.maki/skills``, ``~/.agents/skills`` then ``~/.claude/skills``.

Nearer beats farther, project beats user, and a skill hidden by an earlier one of the same name is logged.
``.agents/skills`` is the shared convention (Codex reads it too). ``allowed-tools`` in a skill is ignored:
a skill never grants tools, and skill scripts are never executed (they can be read, as text).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

SKILL_FILE = "SKILL.md"
MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
MAX_SKILL_FILE_BYTES = 256 * 1024     # a bigger SKILL.md is skipped (it would flood any context window)
MAX_LOAD_CHARS = 30_000               # what skills.load / skills.read return at most
MAX_READ_BYTES = 1_000_000
DEFAULT_CATALOG_CHARS = 4000          # about 1,000 tokens
_ENTRY_DESCRIPTION_CHARS = 300        # one catalog line's description is clipped to this
_NAME_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_RESOURCE_DIRS = ("scripts", "references", "assets")
#: folders checked inside each project or user root, in priority order
SKILL_DIRS = (".maki/skills", ".agents/skills", ".claude/skills")


# --------------------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------------------

def _split_front_matter(text: str) -> Tuple[str, str]:
    """``(front matter text, body)``; the front matter is empty when the file has none."""
    text = text.lstrip("\ufeff")  # an editor's byte-order mark
    if not text.startswith("---"):
        return "", text
    lines = text.splitlines()
    if lines[0].strip() != "---":
        return "", text
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return "\n".join(lines[1:index]), "\n".join(lines[index + 1:]).lstrip("\n")
    return "", text


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _simple_yaml(text: str) -> Dict[str, Any]:
    """A forgiving reader for the few shapes skills use: scalars, block scalars, a list, one nested map.

    Real skills often carry values a strict YAML parser rejects (an unquoted colon in a description), so
    this is also the fallback when PyYAML is absent or fails.
    """
    data: Dict[str, Any] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if not line.strip() or line.lstrip().startswith("#") or line[0] in " \t" or ":" not in line:
            continue
        key, _, rest = line.partition(":")
        key, rest = key.strip(), rest.strip()
        block: List[str] = []
        while i < len(lines) and (not lines[i].strip() or lines[i][0] in " \t"):
            block.append(lines[i])
            i += 1
        while block and not block[-1].strip():
            block.pop()
        if rest[:1] in (">", "|"):
            body = [b.strip() for b in block]
            data[key] = (" " if rest[0] == ">" else "\n").join(body).strip()
        elif rest == "" and block:
            items = [b.strip() for b in block if b.strip() and not b.strip().startswith("#")]
            if items and all(item.startswith("- ") for item in items):
                data[key] = [_unquote(item[2:]) for item in items]
            else:
                nested = {}
                for item in items:
                    if ":" in item:
                        k, _, v = item.partition(":")
                        nested[k.strip()] = _unquote(v)
                data[key] = nested
        else:
            data[key] = _unquote(rest)
    return data


def _load_front_matter(text: str) -> Dict[str, Any]:
    try:
        import yaml
        loaded = yaml.safe_load(text)
        if isinstance(loaded, dict):
            return loaded
    except Exception:  # missing PyYAML or invalid YAML: use the forgiving reader
        pass
    return _simple_yaml(text)


def parse_skill_text(text: str) -> Tuple[Dict[str, Any], str]:
    """``(front matter dict, body)`` of a SKILL.md text."""
    front, body = _split_front_matter(text)
    return (_load_front_matter(front) if front.strip() else {}), body


def _words(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return value.split()
    return [str(v) for v in value]


@dataclass(frozen=True)
class Skill:
    """One discovered skill. The instructions are read from disk when asked for, not held in memory."""
    name: str
    description: str
    directory: Path
    scope: str                      # "explicit", "project" or "user"
    license: Optional[str] = None
    compatibility: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)
    allowed_tools: Tuple[str, ...] = ()   # parsed, never acted on

    @property
    def path(self) -> Path:
        return self.directory / SKILL_FILE

    def body(self) -> str:
        """The Markdown instructions (front matter removed)."""
        return parse_skill_text(self.path.read_text(encoding="utf-8", errors="replace"))[1]

    def resources(self) -> List[str]:
        """Files bundled with the skill, as paths relative to its folder (SKILL.md excluded)."""
        found = []
        for root, dirs, files in os.walk(self.directory):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            for name in sorted(files):
                full = Path(root) / name
                relative = full.relative_to(self.directory).as_posix()
                if relative != SKILL_FILE and not name.startswith("."):
                    found.append(relative)
        return found


def load_skill(directory: Path, scope: str) -> Skill:
    """Read one skill folder.

    Raises:
        ValueError: If SKILL.md is missing, too large, or lacks a valid name or description.
    """
    path = directory / SKILL_FILE
    if not path.is_file():
        raise ValueError("no SKILL.md")
    if path.stat().st_size > MAX_SKILL_FILE_BYTES:
        raise ValueError(f"SKILL.md is larger than {MAX_SKILL_FILE_BYTES // 1024} KB")
    front, _ = parse_skill_text(path.read_text(encoding="utf-8", errors="replace"))
    name, description = front.get("name"), front.get("description")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("missing name")
    name = name.strip()
    if len(name) > MAX_NAME_LENGTH or not _NAME_PATTERN.match(name):
        raise ValueError(f"invalid name {name!r} (lowercase letters, digits and single hyphens, at most "
                         f"{MAX_NAME_LENGTH} characters)")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("missing description")
    description = " ".join(description.split())
    if len(description) > MAX_DESCRIPTION_LENGTH:
        raise ValueError(f"description longer than {MAX_DESCRIPTION_LENGTH} characters")
    if directory.name != name:
        logger.warning("skill %r lives in a folder named %r; loading it anyway", name, directory.name)
    meta = front.get("metadata")
    return Skill(
        name=name, description=description, directory=directory, scope=scope,
        license=str(front["license"]) if front.get("license") is not None else None,
        compatibility=str(front["compatibility"]) if front.get("compatibility") is not None else None,
        metadata={str(k): str(v) for k, v in meta.items()} if isinstance(meta, dict) else {},
        allowed_tools=tuple(_words(front.get("allowed-tools", front.get("allowed_tools")))))


# --------------------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------------------

def _git_root(start: Path) -> Optional[Path]:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def project_roots(cwd: Path) -> List[Path]:
    """The folders whose skill directories count as project skills: *cwd* up to the git root, nearest first.

    Without a git repository only *cwd* itself is used, never every parent up to the filesystem root.
    """
    root = _git_root(cwd)
    if root is None:
        return [cwd]
    chain = [cwd]
    for parent in cwd.parents:
        chain.append(parent)
        if parent == root:
            break
    return chain


@dataclass
class SkillRegistry:
    """The skills an agent can use, with what was skipped and why.

    Attributes:
        skills: Name to :class:`Skill`, in priority order.
        problems: ``(path, reason)`` for skill folders that could not be loaded.
        shadowed: ``(name, kept path, hidden path)`` for same-named skills hidden by a higher-priority one.
        untrusted: Project skills that were found but not loaded because ``trust_project`` was off.
        catalog_chars: Size budget of :meth:`catalog`.
    """
    skills: Dict[str, Skill] = field(default_factory=dict)
    problems: List[Tuple[str, str]] = field(default_factory=list)
    shadowed: List[Tuple[str, str, str]] = field(default_factory=list)
    untrusted: List[Skill] = field(default_factory=list)
    catalog_chars: int = DEFAULT_CATALOG_CHARS

    # -- building --------------------------------------------------------------------------------

    @classmethod
    def discover(cls, paths: Optional[Sequence[Union[str, os.PathLike]]] = None, *, cwd: Optional[Union[str, os.PathLike]] = None,
                 home: Optional[Union[str, os.PathLike]] = None, trust_project: bool = False,
                 catalog_chars: int = DEFAULT_CATALOG_CHARS) -> "SkillRegistry":
        """Find skills in explicit folders, then project folders (if trusted), then user folders.

        Args:
            paths: Folders that contain skill folders (each child with a SKILL.md is a skill), or a skill
                folder itself. Trusted.
            cwd: Where project discovery starts (default: the current directory).
            home: The user's home (default: ``~``).
            trust_project: Load skills found in the project tree. Off by default.
            catalog_chars: Size budget of the catalog shown to the model.
        """
        if isinstance(catalog_chars, bool) or not isinstance(catalog_chars, int) or catalog_chars < 200:
            raise ValueError("catalog_chars must be an integer of at least 200")
        registry = cls(catalog_chars=catalog_chars)
        cwd_path = Path(cwd).expanduser().resolve() if cwd else Path.cwd().resolve()
        home_path = Path(home).expanduser().resolve() if home else Path.home().resolve()
        for folder in paths or ():
            registry._scan_explicit(Path(folder).expanduser())
        for root in project_roots(cwd_path):
            if root == home_path:
                continue  # ~/.agents/skills is the user's own folder, not a project's
            for relative in SKILL_DIRS:
                registry._scan(root / relative, "project", trusted=trust_project)
        for relative in SKILL_DIRS:
            registry._scan(home_path / relative, "user", trusted=True)
        if registry.untrusted:
            logger.info("found %d project skill(s) that were not loaded (%s); pass trust_project=True to use them",
                        len(registry.untrusted), ", ".join(s.name for s in registry.untrusted))
        return registry

    def _scan_explicit(self, folder: Path) -> None:
        """An explicit path is a folder of skills, or one skill folder itself (it holds a SKILL.md)."""
        if (folder / SKILL_FILE).is_file():
            try:
                self._add(load_skill(folder, "explicit"), trusted=True)
            except (ValueError, OSError) as exc:
                self.problems.append((str(folder), str(exc)))
                logger.warning("skipping skill %s: %s", folder, exc)
        else:
            self._scan(folder, "explicit", trusted=True)

    def _scan(self, folder: Path, scope: str, trusted: bool) -> None:
        if not folder.is_dir():
            return
        for child in sorted(folder.iterdir()):
            if not child.is_dir() or child.name.startswith(".") or not (child / SKILL_FILE).exists():
                continue
            try:
                skill = load_skill(child, scope)
            except (ValueError, OSError) as exc:
                self.problems.append((str(child), str(exc)))
                logger.warning("skipping skill %s: %s", child, exc)
                continue
            self._add(skill, trusted)

    def _add(self, skill: Skill, trusted: bool) -> None:
        if not trusted:
            self.untrusted.append(skill)
            return
        kept = self.skills.get(skill.name)
        if kept is not None:
            self.shadowed.append((skill.name, str(kept.directory), str(skill.directory)))
            logger.warning("skill %r in %s is hidden by %s", skill.name, skill.directory, kept.directory)
            return
        self.skills[skill.name] = skill

    # -- use -------------------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.skills)

    def __bool__(self) -> bool:
        return bool(self.skills)

    def get(self, name: str) -> Optional[Skill]:
        return self.skills.get(name)

    def catalog(self) -> str:
        """The skill list for the system prompt: one ``name: description`` line each, within the budget.

        When not every skill fits, the rest are named (names are short) so the model still knows they exist.
        """
        if not self.skills:
            return ""
        header = ("Skills you can load. When a task matches a skill's description, call the skills.load tool "
                  "with its name first, then follow the instructions it returns:\n")
        budget = self.catalog_chars - len(header)
        lines, used, skipped = [], 0, []
        for skill in sorted(self.skills.values(), key=lambda s: s.name):
            description = skill.description
            if len(description) > _ENTRY_DESCRIPTION_CHARS:
                description = description[:_ENTRY_DESCRIPTION_CHARS - 1].rstrip() + "…"
            line = f"- {skill.name}: {description}"
            if used + len(line) + 1 > budget:
                skipped.append(skill.name)
                continue
            lines.append(line)
            used += len(line) + 1
        text = header + "\n".join(lines)
        if skipped:
            text += f"\n(more skills, load by name: {', '.join(skipped)})"
        return text

    def plugin(self) -> "SkillsPlugin":
        return SkillsPlugin(self)


# --------------------------------------------------------------------------------------------------
# The tools the model calls
# --------------------------------------------------------------------------------------------------

def _clip(text: str, limit: int = MAX_LOAD_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[... {len(text) - limit} more characters omitted ...]"


class SkillsPlugin:
    """Read-only tools that let a model load a skill's instructions and read its bundled files.

    Registered by ``Agent(skills=...)`` as the plugin ``skills``: the model sees ``skills__load`` and
    ``skills__read``. Nothing here writes, executes or reaches outside a skill's folder.
    """
    ALLOWED_METHODS = ["load", "read"]
    DANGEROUS_METHODS = None

    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry

    def _skill(self, name: str) -> Skill:
        skill = self.registry.get(name)
        if skill is None:
            available = ", ".join(sorted(self.registry.skills)) or "none"
            raise ValueError(f"unknown skill {name!r}; available skills: {available}")
        return skill

    def load(self, name: str) -> str:
        """Load a skill's full instructions.

        Args:
            name: the skill's name, as listed in the skill catalog
        """
        skill = self._skill(name.strip())
        text = _clip(skill.body()) or f"(the skill {skill.name!r} has no written instructions)"
        files = skill.resources()
        if files:
            shown = files[:40]
            text += ("\n\nFiles bundled with this skill (read one with skills.read, giving this skill's name and "
                     "the path):\n" + "\n".join(f"- {f}" for f in shown)
                     + (f"\n- ... and {len(files) - len(shown)} more" if len(files) > len(shown) else ""))
        return text

    def read(self, name: str, path: str) -> str:
        """Read a text file that is bundled with a skill (a reference, template or script).

        Args:
            name: the skill's name
            path: the file's path inside the skill folder, as listed when the skill was loaded
        """
        skill = self._skill(name.strip())
        base = skill.directory.resolve()
        if not path or os.path.isabs(path) or ".." in Path(path).parts:
            raise ValueError("path must be relative and stay inside the skill folder")
        target = (base / path).resolve()
        if os.path.commonpath([str(base), str(target)]) != str(base):
            raise ValueError("path must stay inside the skill folder")
        if target == (base / SKILL_FILE).resolve():
            raise ValueError("use skills.load for the instructions")
        if not target.is_file():
            raise ValueError(f"no file {path!r} in skill {skill.name!r}")
        if target.stat().st_size > MAX_READ_BYTES:
            raise ValueError(f"{path!r} is larger than {MAX_READ_BYTES // 1000} KB")
        try:
            return _clip(target.read_bytes().decode("utf-8"))
        except UnicodeDecodeError:
            raise ValueError(f"{path!r} is not a text file") from None


def coerce_registry(skills: Any, catalog_chars: Optional[int] = None) -> Optional[SkillRegistry]:
    """What ``Agent(skills=...)`` accepts: True, a folder or list of folders, a registry, or a config dict."""
    if skills is None or skills is False:
        return None
    if isinstance(skills, SkillRegistry):
        return skills
    options: Dict[str, Any] = {}
    if skills is True:
        pass
    elif isinstance(skills, (str, os.PathLike)):
        options["paths"] = [skills]
    elif isinstance(skills, dict):
        unknown = set(skills) - {"paths", "trust_project", "catalog_chars"}
        if unknown:
            raise ValueError(f"unknown skills option(s): {sorted(unknown)}")
        options = dict(skills)
    elif isinstance(skills, (list, tuple)):
        options["paths"] = list(skills)
    else:
        raise TypeError("skills must be True, a folder, a list of folders, a SkillRegistry or an options dict")
    if catalog_chars is not None:
        options.setdefault("catalog_chars", catalog_chars)
    return SkillRegistry.discover(**options)
