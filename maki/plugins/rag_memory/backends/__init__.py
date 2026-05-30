"""
Backend registry and DSN factory for rag_memory.

Usage::

    from maki.plugins.rag_memory.backends import store_from_dsn
    store = store_from_dsn("memory://")           # in-process numpy
    store = store_from_dsn("chroma:///tmp/mydb")  # chromadb
    store = store_from_dsn("qdrant://localhost:6333")
    store = store_from_dsn("postgresql://user:pw@host/db")
    store = store_from_dsn("faiss:///tmp/indexes")

Adding a new backend: create a module in this package, implement ``VectorStore``,
then add a key in ``_SCHEME_MAP`` below.  No other changes needed.
"""

from __future__ import annotations

from urllib.parse import urlparse

from .base import VectorStore

_SCHEME_MAP = {
    "memory": "memory_numpy.MemoryNumpyStore",
    "chroma": "chroma.ChromaStore",
    "qdrant": "qdrant.QdrantStore",
    "postgresql": "pgvector.PgVectorStore",
    "postgres": "pgvector.PgVectorStore",
    "faiss": "faiss_local.FaissLocalStore",
}


def store_from_dsn(dsn: str) -> VectorStore:
    """
    Parse *dsn* and return the matching ``VectorStore`` instance.

    Raises ``ValueError`` for unknown schemes and re-raises ``ImportError``
    with a helpful message when the backend's driver is not installed.
    """
    if not dsn:
        dsn = "memory://"
    scheme = urlparse(dsn).scheme.lower()
    cls_path = _SCHEME_MAP.get(scheme)
    if cls_path is None:
        known = ", ".join(f'"{s}://"' for s in _SCHEME_MAP)
        raise ValueError(
            f"Unknown RAG backend scheme '{scheme}://'. "
            f"Supported: {known}"
        )
    module_name, class_name = cls_path.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(f".{module_name}", package=__package__)
    cls = getattr(mod, class_name)
    return cls(dsn)


__all__ = ["VectorStore", "store_from_dsn"]
