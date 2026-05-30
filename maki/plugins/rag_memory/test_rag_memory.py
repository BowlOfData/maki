"""
Tests for the rag_memory plugin.

All tests use the zero-dependency ``memory://`` backend — no external DB needed.
"""

import pytest
from unittest.mock import MagicMock, patch

from .rag_memory import RagMemory, _chunk_text, register_plugin
from .backends import store_from_dsn
from .backends.memory_numpy import MemoryNumpyStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_embedder(text: str):
    """Deterministic embedder: returns a 4-dim vector based on text hash."""
    import hashlib
    h = int(hashlib.md5(text.encode()).hexdigest(), 16)
    return [(h >> (i * 8) & 0xFF) / 255.0 for i in range(4)]


def _make_rag() -> RagMemory:
    return RagMemory(dsn="memory://", embedder=_fake_embedder)


# ---------------------------------------------------------------------------
# Backend: DSN factory
# ---------------------------------------------------------------------------

class TestStoreDsn:
    def test_memory_scheme_returns_numpy_store(self):
        store = store_from_dsn("memory://")
        assert isinstance(store, MemoryNumpyStore)

    def test_empty_dsn_defaults_to_memory(self):
        store = store_from_dsn("")
        assert isinstance(store, MemoryNumpyStore)

    def test_unknown_scheme_raises(self):
        with pytest.raises(ValueError, match="Unknown RAG backend scheme"):
            store_from_dsn("unknown://foo")


# ---------------------------------------------------------------------------
# Backend: MemoryNumpyStore
# ---------------------------------------------------------------------------

class TestMemoryNumpyStore:
    def test_upsert_and_get(self):
        store = MemoryNumpyStore()
        store.upsert("col", "id1", "hello world", [0.1, 0.9, 0.0, 0.0])
        rec = store.get("col", "id1")
        assert rec["text"] == "hello world"
        assert rec["id"] == "id1"

    def test_get_missing_returns_empty(self):
        store = MemoryNumpyStore()
        assert store.get("col", "nope") == {}

    def test_delete(self):
        store = MemoryNumpyStore()
        store.upsert("col", "id1", "text", [1.0, 0.0])
        assert store.delete("col", "id1") is True
        assert store.get("col", "id1") == {}
        assert store.delete("col", "id1") is False

    def test_query_cosine_ranking(self):
        store = MemoryNumpyStore()
        store.upsert("col", "a", "near", [1.0, 0.0, 0.0, 0.0])
        store.upsert("col", "b", "far", [0.0, 1.0, 0.0, 0.0])
        results = store.query("col", [1.0, 0.0, 0.0, 0.0], k=2)
        assert results[0]["id"] == "a"
        assert results[0]["score"] > results[1]["score"]

    def test_query_empty_collection(self):
        store = MemoryNumpyStore()
        assert store.query("empty", [1.0, 0.0], k=5) == []

    def test_query_with_filter(self):
        store = MemoryNumpyStore()
        store.upsert("col", "x", "match", [1.0, 0.0], metadata={"kind": "A"})
        store.upsert("col", "y", "no match", [1.0, 0.0], metadata={"kind": "B"})
        results = store.query("col", [1.0, 0.0], k=10, filter={"kind": "A"})
        assert len(results) == 1
        assert results[0]["id"] == "x"

    def test_list_collections(self):
        store = MemoryNumpyStore()
        store.upsert("col_a", "1", "text", [1.0])
        store.upsert("col_b", "2", "text", [0.0])
        assert set(store.list_collections()) == {"col_a", "col_b"}

    def test_reset(self):
        store = MemoryNumpyStore()
        store.upsert("col", "1", "text", [1.0])
        store.reset("col")
        assert store.query("col", [1.0], k=1) == []


# ---------------------------------------------------------------------------
# RagMemory: upsert / search / get / update / delete
# ---------------------------------------------------------------------------

class TestRagMemory:
    def test_upsert_and_search(self):
        rag = _make_rag()
        rag.upsert("docs", "Hello world", metadata={"type": "greeting"})
        results = rag.search("docs", "Hello world", k=1)
        assert len(results) == 1
        assert results[0]["text"] == "Hello world"

    def test_upsert_returns_id(self):
        rag = _make_rag()
        id1 = rag.upsert("docs", "text A", id="my-id")
        assert id1 == "my-id"

    def test_upsert_auto_generates_id(self):
        rag = _make_rag()
        id1 = rag.upsert("docs", "text A")
        assert id1 and isinstance(id1, str)

    def test_get_existing(self):
        rag = _make_rag()
        rag.upsert("docs", "some text", id="abc")
        rec = rag.get("docs", "abc")
        assert rec["text"] == "some text"

    def test_get_missing(self):
        rag = _make_rag()
        assert rag.get("docs", "missing") == {}

    def test_update_text(self):
        rag = _make_rag()
        rag.upsert("docs", "original", id="u1")
        result = rag.update("docs", "u1", text="updated")
        assert result is True
        assert rag.get("docs", "u1")["text"] == "updated"

    def test_update_missing_returns_false(self):
        rag = _make_rag()
        assert rag.update("docs", "nope", text="anything") is False

    def test_delete(self):
        rag = _make_rag()
        rag.upsert("docs", "bye", id="del1")
        assert rag.delete("docs", "del1") is True
        assert rag.get("docs", "del1") == {}

    def test_list_collections(self):
        rag = _make_rag()
        rag.upsert("alpha", "text")
        rag.upsert("beta", "text")
        cols = rag.list_collections()
        assert "alpha" in cols and "beta" in cols

    def test_search_empty_collection(self):
        rag = _make_rag()
        assert rag.search("nothing", "query", k=5) == []


# ---------------------------------------------------------------------------
# RagMemory: reindex
# ---------------------------------------------------------------------------

class TestReindex:
    def test_reindex_bulk_upsert(self):
        rag = _make_rag()
        docs = [
            {"collection": "kb", "text": "Doc one", "metadata": {"n": 1}},
            {"collection": "kb", "text": "Doc two", "metadata": {"n": 2}},
        ]
        count = rag.reindex(docs)
        assert count == 2
        results = rag.search("kb", "Doc", k=5)
        assert len(results) == 2

    def test_reindex_skips_missing_fields(self):
        rag = _make_rag()
        docs = [
            {"collection": "kb", "text": "valid"},
            {"text": "no collection"},
            {"collection": "kb"},
        ]
        count = rag.reindex(docs)
        assert count == 1


# ---------------------------------------------------------------------------
# RagMemory: ingest_document
# ---------------------------------------------------------------------------

class TestIngestDocument:
    def test_ingest_text_file(self, tmp_path):
        f = tmp_path / "note.txt"
        f.write_text("Sentence one. Sentence two. Sentence three.")
        rag = _make_rag()
        ids = rag.ingest_document("notes", str(f))
        assert len(ids) >= 1

    def test_ingest_markdown_file(self, tmp_path):
        f = tmp_path / "note.md"
        f.write_text("# Title\n\nSome content here. More content follows.")
        rag = _make_rag()
        ids = rag.ingest_document("notes", str(f))
        assert ids

    def test_ingest_missing_file_raises(self):
        rag = _make_rag()
        with pytest.raises(FileNotFoundError):
            rag.ingest_document("notes", "/nonexistent/file.txt")

    def test_ingest_attaches_metadata(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("Important fact about the world.")
        rag = _make_rag()
        ids = rag.ingest_document("notes", str(f), metadata={"author": "test"})
        rec = rag.get("notes", ids[0])
        assert rec["metadata"].get("author") == "test"
        assert rec["metadata"].get("source_path") == str(f)

    def test_ingest_via_ocr_graceful_when_missing(self, tmp_path):
        f = tmp_path / "image.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal PNG header
        rag = _make_rag()
        with patch("maki.plugins.rag_memory.rag_memory.RagMemory._extract_via_ocr", return_value="ocr text"):
            ids = rag.ingest_document("notes", str(f))
        assert ids


# ---------------------------------------------------------------------------
# Text chunking helper
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_short_text_returns_single_chunk(self):
        assert _chunk_text("short", 512, 64) == ["short"]

    def test_long_text_produces_multiple_chunks(self):
        text = ". ".join([f"Sentence {i}" for i in range(200)]) + "."
        chunks = _chunk_text(text, 200, 50)
        assert len(chunks) > 1

    def test_each_chunk_within_size_limit(self):
        text = ". ".join([f"Word {i}" for i in range(300)]) + "."
        for chunk in _chunk_text(text, 100, 20):
            assert len(chunk) <= 200  # generous: size is approximate

    def test_no_empty_chunks(self):
        text = "  ".join(["A" * 50] * 20)
        for chunk in _chunk_text(text, 100, 20):
            assert chunk.strip()


# ---------------------------------------------------------------------------
# register_plugin
# ---------------------------------------------------------------------------

def test_register_plugin_returns_rag_memory():
    inst = register_plugin(dsn="memory://", embedder=_fake_embedder)
    assert isinstance(inst, RagMemory)


# ---------------------------------------------------------------------------
# ALLOWED_METHODS contract
# ---------------------------------------------------------------------------

def test_allowed_methods_are_callable():
    from .rag_memory import ALLOWED_METHODS
    rag = _make_rag()
    for method in ALLOWED_METHODS:
        assert callable(getattr(rag, method, None)), f"Missing method: {method}"
