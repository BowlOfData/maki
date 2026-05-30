"""
PostgreSQL + pgvector backend for rag_memory.

DSN:  ``postgresql://user:password@host/dbname``
      Optionally append ``?table=rag`` to choose a custom table prefix
      (default: ``rag_chunks``).

Optional dependencies: ``psycopg[binary]>=3``  and  ``pgvector``
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs

from .base import VectorStore

try:
    import psycopg
    from pgvector.psycopg import register_vector
    _PG_AVAILABLE = True
except ImportError:
    _PG_AVAILABLE = False


class PgVectorStore(VectorStore):
    """PostgreSQL + pgvector vector store (one table per collection, lazy-created)."""

    def __init__(self, dsn: str) -> None:
        if not _PG_AVAILABLE:
            raise ImportError(
                "psycopg and pgvector are not installed. "
                "Run: pip install 'psycopg[binary]' pgvector"
            )
        parsed = urlparse(dsn)
        qs = parse_qs(parsed.query)
        self._table_prefix = (qs.get("table", ["rag_chunks"])[0])
        clean = dsn.split("?")[0]
        self._conn = psycopg.connect(clean, autocommit=True)
        register_vector(self._conn)
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self._created: set = set()

    def is_available(self) -> bool:
        return _PG_AVAILABLE

    def _table(self, collection: str) -> str:
        safe = collection.replace("-", "_").replace(".", "_")
        return f"{self._table_prefix}_{safe}"

    def _ensure_table(self, table: str, dim: int) -> None:
        if table in self._created:
            return
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                metadata JSONB,
                embedding vector({dim})
            )
        """)
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS {table}_idx ON {table} "
            "USING ivfflat (embedding vector_cosine_ops)"
        )
        self._created.add(table)

    def upsert(self, collection, id, text, embedding, metadata=None) -> str:
        if not id:
            id = str(uuid.uuid4())
        table = self._table(collection)
        self._ensure_table(table, len(embedding))
        self._conn.execute(
            f"INSERT INTO {table}(id, text, metadata, embedding) VALUES(%s,%s,%s,%s) "
            "ON CONFLICT(id) DO UPDATE SET text=EXCLUDED.text, metadata=EXCLUDED.metadata, "
            "embedding=EXCLUDED.embedding",
            (id, text, json.dumps(metadata or {}), embedding),
        )
        return id

    def query(self, collection, embedding, k=5, filter=None) -> List[Dict]:
        table = self._table(collection)
        if table not in self._created:
            return []
        where = ""
        params: list = [embedding, k]
        if filter:
            clauses = []
            for key, val in filter.items():
                clauses.append(f"metadata->>'{key}' = %s")
                params.insert(-1, str(val))
            where = "WHERE " + " AND ".join(clauses)
        rows = self._conn.execute(
            f"SELECT id, text, metadata, 1 - (embedding <=> %s) AS score "
            f"FROM {table} {where} ORDER BY score DESC LIMIT %s",
            params,
        ).fetchall()
        return [
            {"id": r[0], "text": r[1], "metadata": r[2] or {}, "score": float(r[3])}
            for r in rows
        ]

    def get(self, collection, id) -> Dict:
        table = self._table(collection)
        if table not in self._created:
            return {}
        row = self._conn.execute(
            f"SELECT id, text, metadata FROM {table} WHERE id=%s", (id,)
        ).fetchone()
        if not row:
            return {}
        return {"id": row[0], "text": row[1], "metadata": row[2] or {}}

    def delete(self, collection, id) -> bool:
        table = self._table(collection)
        if table not in self._created:
            return False
        cur = self._conn.execute(f"DELETE FROM {table} WHERE id=%s", (id,))
        return cur.rowcount > 0

    def list_collections(self) -> List[str]:
        prefix = self._table_prefix + "_"
        rows = self._conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname='public' AND tablename LIKE %s",
            (prefix + "%",),
        ).fetchall()
        return [r[0][len(prefix):] for r in rows]

    def reset(self, collection) -> None:
        table = self._table(collection)
        try:
            self._conn.execute(f"DROP TABLE IF EXISTS {table}")
            self._created.discard(table)
        except Exception:
            pass
