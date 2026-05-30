# rag_memory Plugin

Generic, reusable RAG (Retrieval-Augmented Generation) plugin for the Maki framework.

Provides **read and write** access to a vector store via a simple API. Completely
domain-neutral — callers choose collection names and metadata keys. Suitable for any
maki project (newsletter, trading, story generation, etc.).

## Features

- Write: `upsert`, `update`, bulk `reindex`, `ingest_document`
- Read: `search` (semantic), `get` (exact by id)
- Storage: pluggable via a **DSN** — swap DBs with a config change, no code change
- Document ingestion: composes the existing `ocr` plugin for PDFs and images
- Embeddings: Ollama embeddings endpoint (SSRF-safe) or any injected callable

## DSN / Backends

| DSN | Backend | Extra deps |
|---|---|---|
| `memory://` | In-process numpy (default) | `numpy` |
| `chroma:///path/to/dir` | chromadb | `chromadb>=0.5` |
| `qdrant://host:6333` | Qdrant | `qdrant-client>=1.7` |
| `postgresql://user:pw@host/db` | Postgres + pgvector | `psycopg[binary]`, `pgvector` |
| `faiss:///path/to/dir` | FAISS on-disk | `faiss-cpu>=1.7` |

## Usage

```python
from maki.plugins.rag_memory import RagMemory

rag = RagMemory(dsn="memory://", embed_model="nomic-embed-text")

# Write
rag.upsert("docs", "Python is a high-level language.", metadata={"lang": "en"})

# Read
results = rag.search("docs", "programming language", k=3)
for r in results:
    print(r["score"], r["text"])

# Exact lookup
record = rag.get("docs", "some-id")

# Update
rag.update("docs", "some-id", text="Updated text.")

# Delete
rag.delete("docs", "some-id")

# Bulk reindex from canonical source
rag.reindex([
    {"collection": "kb", "id": "1", "text": "Fact one.", "metadata": {"kind": "world"}},
    {"collection": "kb", "id": "2", "text": "Fact two.", "metadata": {"kind": "character"}},
])

# Ingest a document (PDF / image / text / Markdown)
ids = rag.ingest_document("kb", "reference.pdf", metadata={"author": "me"})
```

## Agent usage

```python
agent.load_plugin("rag_memory")
result = agent.execute_task("Find relevant context about the hero.", use_plugins=True)
```

## ALLOWED_METHODS

`upsert`, `search`, `get`, `update`, `delete`, `list_collections`, `reindex`, `ingest_document`

## Adding a new backend

1. Create `backends/my_db.py` implementing `VectorStore` from `backends/base.py`.
2. Add `"myscheme": "my_db.MyDbStore"` to `_SCHEME_MAP` in `backends/__init__.py`.
3. Done — all other code picks it up automatically via the DSN.
