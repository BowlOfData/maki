"""
Abstract base class for all rag_memory vector store backends.

Every backend must implement this interface.  Domain knowledge lives elsewhere —
this contract is deliberately generic: collections, text chunks, metadata dicts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class VectorStore(ABC):
    """
    Protocol that every rag_memory storage backend must satisfy.

    A backend receives pre-computed embedding vectors from RagMemory and
    stores/retrieves (id, text, metadata, vector) tuples.  It never calls
    an LLM or an embeddings model — that is the responsibility of the caller.
    """

    @abstractmethod
    def upsert(
        self,
        collection: str,
        id: str,
        text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Insert or update a document chunk.

        Returns the id that was stored.
        """

    @abstractmethod
    def query(
        self,
        collection: str,
        embedding: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return the *k* nearest neighbours as a list of
        ``{id, text, metadata, score}`` dicts, ordered by descending score.
        An empty list is a valid result when nothing is stored.
        """

    @abstractmethod
    def get(self, collection: str, id: str) -> Dict[str, Any]:
        """Return ``{id, text, metadata}`` or ``{}`` if not found."""

    @abstractmethod
    def delete(self, collection: str, id: str) -> bool:
        """Delete the record with *id* from *collection*.  Returns True on success."""

    @abstractmethod
    def list_collections(self) -> List[str]:
        """Return the names of all collections currently stored."""

    @abstractmethod
    def reset(self, collection: str) -> None:
        """Delete all records in *collection*."""

    def is_available(self) -> bool:
        """Return True if this backend's dependencies are installed."""
        return True
