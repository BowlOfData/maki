"""
Obsidian memory plugin for Maki.

Reads and writes Markdown notes with YAML frontmatter into a local vault.
The vault path is resolved from (in priority order):
  1. TRANDING_VAULT_PATH env var
  2. vault_path constructor arg
  3. ~/ObsidianVaults/tranding
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ALLOWED_METHODS = [
    "append_note",
    "read_note",
    "query_notes",
    "update_frontmatter",
    "list_folder",
    "append_to_jsonl",
    "read_jsonl",
]

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class ObsidianMemory:
    def __init__(self, maki_instance=None, vault_path: Optional[str] = None):
        import yaml as _yaml
        self._yaml = _yaml

        raw = (
            os.environ.get("TRANDING_VAULT_PATH")
            or vault_path
            or "~/ObsidianVaults/tranding"
        )
        self._vault = Path(raw).expanduser().resolve()
        self._vault.mkdir(parents=True, exist_ok=True)
        logger.info(f"ObsidianMemory vault: {self._vault}")

    @property
    def vault_path(self) -> Path:
        return self._vault

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def append_note(
        self,
        folder: str,
        title: str,
        frontmatter: Dict[str, Any],
        body: str,
    ) -> str:
        """Write a new note (or overwrite). Returns the relative path."""
        target_dir = self._vault / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_title = _safe_filename(title)
        path = target_dir / f"{safe_title}.md"
        fm_str = self._yaml.dump(frontmatter, allow_unicode=True, sort_keys=False)
        content = f"---\n{fm_str}---\n\n{body}"
        path.write_text(content, encoding="utf-8")
        rel = str(path.relative_to(self._vault))
        logger.debug(f"Note written: {rel}")
        return rel

    def update_frontmatter(self, rel_path: str, key: str, value: Any) -> bool:
        """Update a single frontmatter key in an existing note. Returns True on success."""
        path = self._vault / rel_path
        if not path.exists():
            logger.warning(f"update_frontmatter: file not found: {rel_path}")
            return False
        content = path.read_text(encoding="utf-8")
        fm, body = _split_frontmatter(content)
        if fm is None:
            return False
        fm[key] = value
        fm_str = self._yaml.dump(fm, allow_unicode=True, sort_keys=False)
        path.write_text(f"---\n{fm_str}---\n\n{body}", encoding="utf-8")
        return True

    def append_to_jsonl(self, rel_path: str, record: Dict[str, Any]) -> str:
        """Append a JSON record to a .jsonl file. Creates the file if missing."""
        path = self._vault / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        return str(path.relative_to(self._vault))

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read_note(self, rel_path: str) -> Dict[str, Any]:
        """Read a note. Returns {frontmatter, body} or {} if not found."""
        path = self._vault / rel_path
        if not path.exists():
            return {}
        content = path.read_text(encoding="utf-8")
        fm, body = _split_frontmatter(content)
        return {"frontmatter": fm or {}, "body": body}

    def query_notes(
        self,
        folder: str,
        since_hours: Optional[int] = None,
        tags: Optional[List[str]] = None,
        frontmatter_filters: Optional[Dict[str, Any]] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Return matching notes from *folder*, newest first."""
        base = self._vault / folder
        if not base.exists():
            return []
        files = sorted(
            (p for p in base.rglob("*.md") if not p.name.startswith("._")),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        results = []
        cutoff = None
        if since_hours:
            from datetime import timedelta
            cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)

        for fp in files:
            try:
                content = fp.read_text(encoding="utf-8")
                fm, body = _split_frontmatter(content)
                if fm is None:
                    fm = {}
                if cutoff:
                    mtime = datetime.fromtimestamp(fp.stat().st_mtime, tz=timezone.utc)
                    if mtime < cutoff:
                        continue
                if tags:
                    note_tags = fm.get("tags") or []
                    if not any(t in note_tags for t in tags):
                        continue
                if frontmatter_filters:
                    if not all(fm.get(k) == v for k, v in frontmatter_filters.items()):
                        continue
                results.append({
                    "path": str(fp.relative_to(self._vault)),
                    "frontmatter": fm,
                    "body": body[:500],
                })
                if len(results) >= limit:
                    break
            except Exception as e:
                logger.warning(f"query_notes: skipping {fp}: {e}")
        return results

    def list_folder(self, folder: str) -> List[str]:
        """List relative paths of all .md files in *folder*."""
        base = self._vault / folder
        if not base.exists():
            return []
        return [str(fp.relative_to(self._vault)) for fp in sorted(base.rglob("*.md")) if not fp.name.startswith("._")]

    def read_jsonl(self, rel_path: str, limit: int = 500) -> List[Dict[str, Any]]:
        """Read up to *limit* records from a .jsonl file."""
        path = self._vault / rel_path
        if not path.exists():
            return []
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records[-limit:]


def register_plugin(maki_instance=None, vault_path: Optional[str] = None):
    return ObsidianMemory(maki_instance, vault_path=vault_path)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _safe_filename(title: str) -> str:
    return re.sub(r'[^\w\-.]', '_', title)[:120]


def _split_frontmatter(content: str):
    import yaml
    m = _FRONTMATTER_RE.match(content)
    if not m:
        return None, content
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except Exception:
        fm = {}
    body = content[m.end():]
    return fm, body
