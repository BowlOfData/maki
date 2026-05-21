import json
import tempfile
import pytest
from pathlib import Path
from maki.plugins.obsidian_memory.obsidian_memory import ObsidianMemory, _safe_filename, _split_frontmatter


@pytest.fixture()
def vault(tmp_path):
    return ObsidianMemory(vault_path=str(tmp_path))


def test_append_and_read_note(vault):
    path = vault.append_note(
        folder="notes",
        title="Test Note",
        frontmatter={"tags": ["test"], "status": "open"},
        body="Hello world",
    )
    note = vault.read_note(path)
    assert note["frontmatter"]["status"] == "open"
    assert "Hello world" in note["body"]


def test_read_note_missing(vault):
    assert vault.read_note("nonexistent/path.md") == {}


def test_update_frontmatter(vault):
    path = vault.append_note("notes", "update test", {"status": "open"}, "body")
    ok = vault.update_frontmatter(path, "status", "closed")
    assert ok is True
    note = vault.read_note(path)
    assert note["frontmatter"]["status"] == "closed"


def test_update_frontmatter_missing_file(vault):
    assert vault.update_frontmatter("ghost.md", "x", 1) is False


def test_list_folder(vault):
    vault.append_note("folder", "note1", {}, "a")
    vault.append_note("folder", "note2", {}, "b")
    files = vault.list_folder("folder")
    assert len(files) == 2


def test_list_folder_empty(vault):
    assert vault.list_folder("missing") == []


def test_query_notes_by_tag(vault):
    vault.append_note("q", "btc note", {"tags": ["btc"]}, "btc body")
    vault.append_note("q", "eth note", {"tags": ["eth"]}, "eth body")
    results = vault.query_notes("q", tags=["btc"])
    assert len(results) == 1
    assert results[0]["frontmatter"]["tags"] == ["btc"]


def test_append_and_read_jsonl(vault):
    vault.append_to_jsonl("logs/events.jsonl", {"event": "filled", "price": 100})
    vault.append_to_jsonl("logs/events.jsonl", {"event": "cancelled"})
    records = vault.read_jsonl("logs/events.jsonl")
    assert len(records) == 2
    assert records[0]["event"] == "filled"


def test_read_jsonl_missing(vault):
    assert vault.read_jsonl("no_file.jsonl") == []


def test_safe_filename():
    assert _safe_filename("Hello World!") == "Hello_World_"
    assert len(_safe_filename("x" * 200)) == 120


def test_split_frontmatter_no_fm():
    fm, body = _split_frontmatter("No frontmatter here")
    assert fm is None
    assert "No frontmatter" in body
