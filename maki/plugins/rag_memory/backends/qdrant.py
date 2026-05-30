"""
Qdrant backend for rag_memory  (DSN: ``qdrant://host:port``).

Optional dependency: ``qdrant-client>=1.7``
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from .base import VectorStore

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, PointStruct, VectorParams,
        Filter, FieldCondition, MatchValue,
    )
    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False


class QdrantStore(VectorStore):
    """Remote or local Qdrant vector store."""

    def __init__(self, dsn: str) -> None:
        if not _QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client is not installed. Run: pip install qdrant-client>=1.7"
            )
        parsed = urlparse(dsn)
        host = parsed.hostname or "localhost"
        port = parsed.port or 6333
        self._client = QdrantClient(host=host, port=port)
        self._dim: Dict[str, int] = {}

    def is_available(self) -> bool:
        return _QDRANT_AVAILABLE

    def _ensure_collection(self, collection: str, dim: int) -> None:
        existing = [c.name for c in self._client.get_collections().collections]
        if collection not in existing:
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        self._dim[collection] = dim

    def upsert(self, collection, id, text, embedding, metadata=None) -> str:
        if not id:
            id = str(uuid.uuid4())
        self._ensure_collection(collection, len(embedding))
        payload = {"text": text, **(metadata or {})}
        self._client.upsert(
            collection_name=collection,
            points=[PointStruct(id=id, vector=embedding, payload=payload)],
        )
        return id

    def query(self, collection, embedding, k=5, filter=None) -> List[Dict]:
        qdrant_filter = None
        if filter:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter.items()
            ]
            qdrant_filter = Filter(must=conditions)
        hits = self._client.search(
            collection_name=collection,
            query_vector=embedding,
            limit=k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        results = []
        for h in hits:
            payload = dict(h.payload or {})
            text = payload.pop("text", "")
            results.append({"id": str(h.id), "text": text, "metadata": payload, "score": h.score})
        return results

    def get(self, collection, id) -> Dict:
        res = self._client.retrieve(collection_name=collection, ids=[id], with_payload=True)
        if not res:
            return {}
        payload = dict(res[0].payload or {})
        text = payload.pop("text", "")
        return {"id": id, "text": text, "metadata": payload}

    def delete(self, collection, id) -> bool:
        try:
            from qdrant_client.models import PointIdsList
            self._client.delete(collection_name=collection, points_selector=PointIdsList(points=[id]))
            return True
        except Exception:
            return False

    def list_collections(self) -> List[str]:
        return [c.name for c in self._client.get_collections().collections]

    def reset(self, collection) -> None:
        try:
            self._client.delete_collection(collection)
        except Exception:
            pass
