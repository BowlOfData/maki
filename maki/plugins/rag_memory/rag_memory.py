"""
rag_memory — generic, reusable RAG plugin for the Maki framework.

Provides read + write access to a vector store backed by an embedding model.
Completely domain-neutral: callers choose collection names and metadata keys.
The active store is selected by a DSN (connection string), making it trivial
to swap databases without changing application code.

Supported DSN schemes (backend docs in backends/):
  memory://                     — in-process numpy (default, zero setup)
  chroma:///path/to/dir         — chromadb
  qdrant://host:port            — Qdrant
  postgresql://user:pw@host/db  — Postgres + pgvector
  faiss:///path/to/dir          — FAISS on-disk

Embedding: Ollama embeddings endpoint via the existing Connector/MakiLLama
HTTP layer (SSRF protection preserved). Any callable can be injected instead.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

ALLOWED_METHODS = [
    "upsert",
    "search",
    "get",
    "update",
    "delete",
    "list_collections",
    "reindex",
    "ingest_document",
]

_DEFAULT_CHUNK_SIZE = 512
_DEFAULT_CHUNK_OVERLAP = 64
_DEFAULT_EMBED_MODEL = "nomic-embed-text"


class RagMemory:
    """
    Generic RAG layer for Maki agents.

    Args:
        maki_instance: A MakiLLama (or compatible) instance used to call the
                       Ollama embeddings endpoint.  Required unless a custom
                       *embedder* callable is supplied.
        dsn:           Connection string selecting the vector store backend.
                       Defaults to env ``RAG_DSN`` or ``"memory://"``.
        embedder:      Optional callable ``(text: str) -> List[float]``.
                       When provided, *maki_instance* and *embed_model* are
                       ignored for embedding generation.
        embed_model:   Ollama model used for embeddings (default: ``nomic-embed-text``
                       or env ``RAG_EMBED_MODEL``).
    """

    # Mirror the module-level whitelist on the class: tool-call validation
    # reads ALLOWED_METHODS from the plugin instance, not the module.
    ALLOWED_METHODS = ALLOWED_METHODS

    def __init__(
        self,
        maki_instance=None,
        dsn: Optional[str] = None,
        embedder: Optional[Callable[[str], List[float]]] = None,
        embed_model: Optional[str] = None,
    ) -> None:
        from .backends import store_from_dsn

        resolved_dsn = dsn or os.environ.get("RAG_DSN", "memory://")
        self._store = store_from_dsn(resolved_dsn)
        self.maki = maki_instance
        self._embedder = embedder
        self._embed_model = (
            embed_model
            or os.environ.get("RAG_EMBED_MODEL", _DEFAULT_EMBED_MODEL)
        )
        logger.info("RagMemory initialised with DSN=%s embed_model=%s", resolved_dsn, self._embed_model)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> List[float]:
        """Return the embedding vector for *text*."""
        if self._embedder:
            return self._embedder(text)
        if self.maki is None:
            raise RuntimeError(
                "RagMemory requires either a maki_instance or a custom embedder callable."
            )
        return self._ollama_embed(text)

    def _ollama_embed(self, text: str) -> List[float]:
        """Call the Ollama /api/embed endpoint via the existing maki HTTP layer."""
        # MakiLLama exposes its base URL; we reuse the same Connector for SSRF safety.
        base_url = getattr(self.maki, "base_url", None) or getattr(self.maki, "_base_url", None)
        if base_url is None:
            # Fall back: try to import and use the configured default
            from maki.config import DEFAULT_OLLAMA_BASE_URL
            base_url = DEFAULT_OLLAMA_BASE_URL

        from maki.connector import Connector
        connector = Connector()
        url = base_url.rstrip("/") + "/api/embed"
        resp = connector.post(url, json={"model": self._embed_model, "input": text})
        data = resp.json()
        # Ollama returns either {"embeddings": [[...]]} or {"embedding": [...]}
        embeddings = data.get("embeddings") or data.get("embedding")
        if not embeddings:
            raise RuntimeError(f"Unexpected embed response: {data}")
        if isinstance(embeddings[0], list):
            return embeddings[0]
        return embeddings

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(
        self,
        collection: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
    ) -> str:
        """
        Embed *text* and insert/update it in *collection*.

        Returns the id that was stored (auto-generated UUID if not supplied).
        """
        if not id:
            id = str(uuid.uuid4())
        embedding = self._embed(text)
        return self._store.upsert(collection, id, text, embedding, metadata)

    def update(
        self,
        collection: str,
        id: str,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update an existing record.  If *text* changes, the embedding is recomputed.
        Returns True when the record existed and was updated, False otherwise.
        """
        existing = self._store.get(collection, id)
        if not existing:
            return False
        new_text = text if text is not None else existing["text"]
        new_meta = metadata if metadata is not None else existing["metadata"]
        embedding = self._embed(new_text)
        self._store.upsert(collection, id, new_text, embedding, new_meta)
        return True

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        collection: str,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Embed *query* and return the *k* most semantically similar records.

        Returns a list of ``{id, text, metadata, score}`` dicts, ordered by
        descending relevance score.  An empty list means nothing was found.
        """
        embedding = self._embed(query)
        return self._store.query(collection, embedding, k=k, filter=filter)

    def get(self, collection: str, id: str) -> Dict[str, Any]:
        """Return ``{id, text, metadata}`` for an exact id, or ``{}`` if not found."""
        return self._store.get(collection, id)

    def list_collections(self) -> List[str]:
        """Return the names of all collections currently stored."""
        return self._store.list_collections()

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def delete(self, collection: str, id: str) -> bool:
        """Delete the record with *id* from *collection*.  Returns True on success."""
        return self._store.delete(collection, id)

    def reindex(self, documents: List[Dict[str, Any]]) -> int:
        """
        Re-embed and upsert a list of documents in bulk.

        Each document must be a dict with at least ``collection`` and ``text`` keys.
        Optional: ``id`` and ``metadata``.  Useful for rebuilding an index from a
        canonical source (e.g. an Obsidian vault) after manual edits.

        Returns the count of documents processed.
        """
        count = 0
        for doc in documents:
            collection = doc.get("collection")
            text = doc.get("text")
            if not collection or not text:
                logger.warning("reindex: skipping document missing 'collection' or 'text': %s", doc)
                continue
            self.upsert(
                collection=collection,
                text=text,
                metadata=doc.get("metadata"),
                id=doc.get("id"),
            )
            count += 1
        logger.info("reindex: processed %d documents", count)
        return count

    # ------------------------------------------------------------------
    # Document ingestion (composes the OCR plugin)
    # ------------------------------------------------------------------

    def ingest_document(
        self,
        collection: str,
        path: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
    ) -> List[str]:
        """
        Extract text from *path*, chunk it, embed each chunk, and upsert all chunks
        into *collection*.

        Supported file types (resolved by extension):
          - Plain text / Markdown  — read directly
          - PDF (digital)          — text layer extracted; pages with no text fall back to OCR
          - Images / scanned docs  — OCR via the existing ``ocr`` plugin

        The ``ocr`` plugin must be importable for image/PDF fallback to work;
        it is an optional dependency — plain text and digital-PDF paths work without it.

        Args:
            collection:    Destination collection name.
            path:          Absolute or CWD-relative path to the source file.
            metadata:      Extra metadata attached to every stored chunk (e.g. source, author).
            chunk_size:    Approximate character count per chunk (default 512).
            chunk_overlap: Characters of overlap between consecutive chunks (default 64).

        Returns:
            List of stored chunk ids.
        """
        import os as _os
        path = _os.path.realpath(path)
        if not _os.path.isfile(path):
            raise FileNotFoundError(f"ingest_document: file not found: {path}")

        text = self._extract_text(path)
        if not text or not text.strip():
            logger.warning("ingest_document: no text extracted from %s", path)
            return []

        chunks = _chunk_text(text, chunk_size, chunk_overlap)
        base_meta = {"source_path": path, **(metadata or {})}

        ids = []
        for i, chunk in enumerate(chunks):
            chunk_meta = {**base_meta, "chunk_index": i, "chunk_total": len(chunks)}
            id = self.upsert(collection=collection, text=chunk, metadata=chunk_meta)
            ids.append(id)

        logger.info(
            "ingest_document: %s → collection='%s', %d chunks stored",
            path, collection, len(ids),
        )
        return ids

    def _extract_text(self, path: str) -> str:
        """Return plain text from *path*, dispatching by file type."""
        ext = os.path.splitext(path)[1].lower()

        if ext in (".txt", ".md", ".rst", ".csv"):
            return open(path, encoding="utf-8", errors="replace").read()

        if ext == ".pdf":
            return self._extract_pdf(path)

        # Images and other binary formats → OCR plugin
        return self._extract_via_ocr(path)

    def _extract_pdf(self, path: str) -> str:
        """Extract text from a digital PDF; fall back to OCR for image-only pages."""
        try:
            import pdfplumber
            pages = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    if text.strip():
                        pages.append(text)
                    else:
                        pages.append(self._extract_via_ocr(path, page_hint=page.page_number))
            return "\n\n".join(pages)
        except ImportError:
            return self._extract_via_ocr(path)

    def _extract_via_ocr(self, path: str, page_hint: Optional[int] = None) -> str:
        """Delegate to the existing OCR plugin and return the extracted Markdown text."""
        try:
            from maki.plugins.ocr.ocr import OCR
            ocr = OCR(maki_instance=self.maki, base_dir=os.path.dirname(path))
            result = ocr.extract(path)
            if result.get("success"):
                return result.get("markdown", "")
            logger.warning("ingest_document: OCR failed for %s: %s", path, result.get("error"))
            return ""
        except ImportError:
            logger.warning(
                "ingest_document: OCR plugin not available for %s. "
                "Install maki OCR dependencies or use a text/Markdown file.", path
            )
            return ""


# ------------------------------------------------------------------
# Plugin entry point
# ------------------------------------------------------------------

def register_plugin(
    maki_instance=None,
    dsn: Optional[str] = None,
    embedder: Optional[Callable] = None,
    embed_model: Optional[str] = None,
) -> RagMemory:
    return RagMemory(
        maki_instance=maki_instance,
        dsn=dsn,
        embedder=embedder,
        embed_model=embed_model,
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """Split *text* into overlapping chunks of approximately *size* characters."""
    if size <= 0:
        return [text]
    text = text.strip()
    if len(text) <= size:
        return [text]

    # Split on sentence boundaries where possible
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) > size and current:
            chunk = " ".join(current)
            chunks.append(chunk)
            # Keep overlap: take words from end of current chunk
            overlap_text = chunk[-overlap:] if overlap else ""
            current = [overlap_text] if overlap_text else []
            current_len = len(overlap_text)
        current.append(sentence)
        current_len += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c.strip()]
