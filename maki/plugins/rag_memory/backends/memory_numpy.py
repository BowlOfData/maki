"""
In-process numpy vector store backend for rag_memory.

Zero external dependencies beyond numpy (already a common transitive dep).
Used as the default backend (DSN: ``memory://``) and for tests — no DB setup needed.
Not persisted to disk: data lives only for the lifetime of the process.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from .base import VectorStore

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False


class MemoryNumpyStore(VectorStore):
    """
    Ephemeral, in-process vector store backed by a plain Python dict + numpy
    cosine similarity.  Suitable for tests, small projects, and as a fallback.
    """

    def __init__(self, dsn: str = "memory://") -> None:
        # collections: name → {id: {text, metadata, embedding}}
        self._data: Dict[str, Dict[str, Dict]] = {}

    def is_available(self) -> bool:
        return _NUMPY_AVAILABLE

    def _col(self, collection: str) -> Dict[str, Dict]:
        return self._data.setdefault(collection, {})

    def upsert(
        self,
        collection: str,
        id: str,
        text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not id:
            id = str(uuid.uuid4())
        self._col(collection)[id] = {
            "text": text,
            "metadata": metadata or {},
            "embedding": embedding,
        }
        return id

    def query(
        self,
        collection: str,
        embedding: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not _NUMPY_AVAILABLE:
            raise RuntimeError("numpy is required for the memory:// backend")
        col = self._col(collection)
        if not col:
            return []

        q = np.array(embedding, dtype=float)
        q_norm = np.linalg.norm(q)

        results = []
        for id, rec in col.items():
            if filter and not all(rec["metadata"].get(k) == v for k, v in filter.items()):
                continue
            vec = np.array(rec["embedding"], dtype=float)
            norm = np.linalg.norm(vec)
            score = float(np.dot(q, vec) / (q_norm * norm)) if q_norm and norm else 0.0
            results.append({"id": id, "text": rec["text"], "metadata": rec["metadata"], "score": score})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    def get(self, collection: str, id: str) -> Dict[str, Any]:
        rec = self._col(collection).get(id)
        if rec is None:
            return {}
        return {"id": id, "text": rec["text"], "metadata": rec["metadata"]}

    def delete(self, collection: str, id: str) -> bool:
        col = self._col(collection)
        if id in col:
            del col[id]
            return True
        return False

    def list_collections(self) -> List[str]:
        return list(self._data.keys())

    def reset(self, collection: str) -> None:
        self._data.pop(collection, None)
