"""
Tests for the JsonReader plugin.
"""

import json
import os
import tempfile

from maki.plugins.json_reader.json_reader import JsonReader


def test_json_reader_reads_selected_fields():
    base_dir = tempfile.mkdtemp()
    file_path = os.path.join(base_dir, "items.json")
    data = [
        {"title": "One", "tags": ["a", "b"], "author": "Ada"},
        {"title": "Two", "tags": ["c"], "author": "Lin"},
    ]

    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle)

    reader = JsonReader(base_dir=base_dir)
    result = reader.read_json_fields("items.json", fields=["title", "tags"], max_items=1)

    assert result["success"] is True
    assert result["count"] == 1
    assert "title: 'One'" in result["content"]
    assert "tags: 'a, b'" in result["content"]


def test_json_reader_blocks_path_escape():
    reader = JsonReader(base_dir=tempfile.gettempdir())
    result = reader.read_json_fields("../../etc/passwd", fields=["title"])

    assert result["success"] is False
    assert "outside the allowed directory" in result["error"]
