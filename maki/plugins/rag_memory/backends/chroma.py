"""
Chromadb backend for rag_memory  (DSN: ``chroma:///path/to/dir``).

Optional dependency: ``chromadb>=0.5``
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from .base import VectorStore

try:
    import chromadb
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False


class ChromaStore(VectorStore):
    """Persistent chromadb vector store."""

    def __init__(self, dsn: str) -> None:
        parsed = urlparse(dsn)
        path = parsed.path or "."
        if not _CHROMA_AVAILABLE:
            raise ImportError(
                "chromadb is not installed. Run: pip install chromadb>=0.5"
            )
        self._client = chromadb.PersistentClient(path=path)

    def is_available(self) -> bool:
        return _CHROMA_AVAILABLE

    def _col(self, collection: str):
        return self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, collection, id, text, embedding, metadata=None) -> str:
        if not id:
            id = str(uuid.uuid4())
        self._col(collection).upsert(
            ids=[id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata or {}],
        )
        return id

    def query(self, collection, embedding, k=5, filter=None) -> List[Dict]:
        col = self._col(collection)
        kwargs: Dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": min(k, max(col.count(), 1)),
            "include": ["documents", "metadatas", "distances"],
        }
        if filter:
            kwargs["where"] = filter
        res = col.query(**kwargs)
        results = []
        for i, id in enumerate(res["ids"][0]):
            score = 1.0 - res["distances"][0][i]
            results.append({
                "id": id,
                "text": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "score": score,
            })
        return results

    def get(self, collection, id) -> Dict:
        res = self._col(collection).get(ids=[id], include=["documents", "metadatas"])
        if not res["ids"]:
            return {}
        return {"id": id, "text": res["documents"][0], "metadata": res["metadatas"][0]}

    def delete(self, collection, id) -> bool:
        try:
            self._col(collection).delete(ids=[id])
            return True
        except Exception:
            return False

    def list_collections(self) -> List[str]:
        return [c.name for c in self._client.list_collections()]

    def reset(self, collection) -> None:
        try:
            self._client.delete_collection(collection)
        except Exception:
            pass
