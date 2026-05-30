"""
rag_memory plugin — usage examples.

Run with:  python -m maki.plugins.rag_memory.example_usage
"""

from maki.plugins.rag_memory import RagMemory


def _fake_embedder(text):
    """Deterministic 4-dim embedder for demo (no Ollama required)."""
    import hashlib
    h = int(hashlib.md5(text.encode()).hexdigest(), 16)
    return [(h >> (i * 8) & 0xFF) / 255.0 for i in range(4)]


def main():
    # --- 1. In-process store (no DB setup needed) ---
    rag = RagMemory(dsn="memory://", embedder=_fake_embedder)

    # Write
    rag.upsert("docs", "Python is a high-level programming language.", metadata={"lang": "en"})
    rag.upsert("docs", "Maki is a multi-agent LLM framework.", metadata={"lang": "en"})
    rag.upsert("docs", "ChromaDB stores vector embeddings.", metadata={"lang": "en"})

    # Read
    results = rag.search("docs", "LLM framework", k=2)
    print("Search results for 'LLM framework':")
    for r in results:
        print(f"  [{r['score']:.3f}] {r['text']}")

    # Get by id
    id1 = rag.upsert("kb", "Exact lookup test", id="item-1")
    print("\nGet by id:", rag.get("kb", id1))

    # Update
    rag.update("kb", "item-1", text="Updated lookup test")
    print("After update:", rag.get("kb", "item-1")["text"])

    # Delete
    rag.delete("kb", "item-1")
    print("After delete:", rag.get("kb", "item-1"))

    # --- 2. Bulk reindex ---
    docs = [
        {"collection": "instructions", "text": "The hero must reach the lighthouse.", "id": "inst-1"},
        {"collection": "instructions", "text": "The villain controls the tides.", "id": "inst-2"},
    ]
    count = rag.reindex(docs)
    print(f"\nReindexed {count} documents")
    print("Recall:", rag.search("instructions", "lighthouse", k=1)[0]["text"])

    # --- 3. Ingest a text file ---
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("The sky is blue. The sea is calm. Stars guide sailors home.")
        tmp_path = f.name
    ids = rag.ingest_document("notes", tmp_path, metadata={"source": "demo"})
    print(f"\nIngested {len(ids)} chunks from file")
    os.unlink(tmp_path)

    # --- 4. Switching backends (chroma example — requires chromadb) ---
    # rag_chroma = RagMemory(dsn="chroma:///tmp/my_rag_db", embedder=_fake_embedder)
    # rag_chroma.upsert("docs", "Persistent storage example.")
    # Swap to qdrant: RagMemory(dsn="qdrant://localhost:6333", embedder=_fake_embedder)

    print("\nDone.")


if __name__ == "__main__":
    main()
