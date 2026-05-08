"""
JSON Reader Plugin for Maki Framework

Reads a JSON array file and returns a compact, field-filtered representation
suitable for LLM context. Generic — no domain-specific knowledge.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional


class JsonReader:
    """Plugin that reads a JSON array file and returns selected fields as compact text."""

    def __init__(self, maki_instance=None, base_dir: str = None):
        self.maki = maki_instance
        self.logger = logging.getLogger(__name__)
        self.base_dir = os.path.realpath(base_dir if base_dir is not None else os.getcwd())
        self.logger.info(f"JsonReader plugin initialized (base_dir='{self.base_dir}')")

    ALLOWED_METHODS = ["read_json_fields"]

    def _safe_path(self, path: str) -> str:
        resolved = os.path.realpath(os.path.join(self.base_dir, path))
        if resolved != self.base_dir and not resolved.startswith(self.base_dir + os.sep):
            raise ValueError(
                f"Path '{path}' resolves outside the allowed directory '{self.base_dir}'"
            )
        return resolved

    def read_json_fields(
        self,
        file_path: str,
        fields: List[str],
        max_items: int = 0,
    ) -> Dict[str, Any]:
        """
        Read a JSON array file and return each item formatted with only the requested fields.

        Args:
            file_path: Path to the JSON file (relative to base_dir).
            fields: List of field names to include for each item.
            max_items: Maximum number of items to return. 0 means no limit.

        Returns:
            Dict with keys:
              - success (bool)
              - content (str): numbered list, one item per line
              - count (int): number of items returned
              - error (str | None)
        """
        result: Dict[str, Any] = {
            "success": False,
            "content": "",
            "count": 0,
            "error": None,
        }

        if not isinstance(file_path, str) or not file_path.strip():
            result["error"] = "file_path must be a non-empty string"
            return result
        if not fields:
            result["error"] = "fields must be a non-empty list"
            return result

        try:
            safe = self._safe_path(file_path)
        except ValueError as exc:
            result["error"] = str(exc)
            return result

        if not os.path.isfile(safe):
            result["error"] = f"File not found: {file_path}"
            return result

        try:
            with open(safe, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            result["error"] = f"Could not read JSON: {exc}"
            return result

        if not isinstance(data, list):
            result["error"] = "JSON file must contain a top-level array"
            return result

        items = data if max_items == 0 else data[:max_items]
        lines: List[str] = []
        for idx, item in enumerate(items, start=1):
            parts = []
            for field in fields:
                value = item.get(field, "")
                if isinstance(value, list):
                    value = ", ".join(str(v) for v in value)
                parts.append(f"{field}: {value!r}")
            lines.append(f"[{idx}] " + " | ".join(parts))

        result["success"] = True
        result["content"] = "\n".join(lines)
        result["count"] = len(lines)
        self.logger.info(
            "JsonReader: returned %d items from '%s' (fields=%s)",
            len(lines),
            file_path,
            fields,
        )
        return result


def register_plugin(maki_instance=None, base_dir: str = None):
    return JsonReader(maki_instance, base_dir=base_dir)
