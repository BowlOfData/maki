"""
FAISS on-disk backend for rag_memory  (DSN: ``faiss:///path/to/dir``).

Optional dependency: ``faiss-cpu>=1.7`` (or ``faiss-gpu``).
Metadata and text are stored as a companion JSON file alongside the FAISS index.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from .base import VectorStore

try:
    import faiss
    import numpy as np
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


class FaissLocalStore(VectorStore):
    """FAISS flat-cosine index persisted to a directory.  One index per collection."""

    def __init__(self, dsn: str) -> None:
        if not _FAISS_AVAILABLE:
            raise ImportError(
                "faiss-cpu is not installed. Run: pip install faiss-cpu"
            )
        parsed = urlparse(dsn)
        self._base = Path(parsed.path or ".").expanduser().resolve()
        self._base.mkdir(parents=True, exist_ok=True)
        self._indexes: Dict[str, Any] = {}
        self._docs: Dict[str, Dict[str, Dict]] = {}

    def is_available(self) -> bool:
        return _FAISS_AVAILABLE

    def _load(self, collection: str) -> tuple:
        idx_path = self._base / f"{collection}.faiss"
        meta_path = self._base / f"{collection}.json"
        if collection not in self._indexes:
            if idx_path.exists():
                self._indexes[collection] = faiss.read_index(str(idx_path))
                self._docs[collection] = json.loads(meta_path.read_text())
            else:
                self._indexes[collection] = None
                self._docs[collection] = {}
        return self._indexes[collection], self._docs[collection]

    def _save(self, collection: str) -> None:
        idx_path = self._base / f"{collection}.faiss"
        meta_path = self._base / f"{collection}.json"
        if self._indexes.get(collection) is not None:
            faiss.write_index(self._indexes[collection], str(idx_path))
        meta_path.write_text(json.dumps(self._docs.get(collection, {})))

    def upsert(self, collection, id, text, embedding, metadata=None) -> str:
        if not id:
            id = str(uuid.uuid4())
        idx, docs = self._load(collection)
        vec = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(vec)
        if idx is None:
            dim = len(embedding)
            idx = faiss.IndexFlatIP(dim)
            self._indexes[collection] = idx
        if id in docs:
            faiss_id = docs[id]["faiss_id"]
        else:
            faiss_id = idx.ntotal
        idx.add(vec)
        docs[id] = {"text": text, "metadata": metadata or {}, "faiss_id": faiss_id}
        self._save(collection)
        return id

    def query(self, collection, embedding, k=5, filter=None) -> List[Dict]:
        idx, docs = self._load(collection)
        if idx is None or idx.ntotal == 0:
            return []
        vec = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(vec)
        k = min(k, idx.ntotal)
        scores, indices = idx.search(vec, k)
        id_by_faiss: Dict[int, str] = {v["faiss_id"]: k for k, v in docs.items()}
        results = []
        for score, faiss_id in zip(scores[0], indices[0]):
            doc_id = id_by_faiss.get(int(faiss_id))
            if doc_id is None:
                continue
            rec = docs[doc_id]
            if filter and not all(rec["metadata"].get(fk) == fv for fk, fv in filter.items()):
                continue
            results.append({"id": doc_id, "text": rec["text"], "metadata": rec["metadata"], "score": float(score)})
        return results

    def get(self, collection, id) -> Dict:
        _, docs = self._load(collection)
        rec = docs.get(id)
        if not rec:
            return {}
        return {"id": id, "text": rec["text"], "metadata": rec["metadata"]}

    def delete(self, collection, id) -> bool:
        idx, docs = self._load(collection)
        if id not in docs:
            return False
        del docs[id]
        self._save(collection)
        return True

    def list_collections(self) -> List[str]:
        return [p.stem for p in self._base.glob("*.faiss")]

    def reset(self, collection) -> None:
        (self._base / f"{collection}.faiss").unlink(missing_ok=True)
        (self._base / f"{collection}.json").unlink(missing_ok=True)
        self._indexes.pop(collection, None)
        self._docs.pop(collection, None)
