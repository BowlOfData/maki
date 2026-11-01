# Memory: Obsidian and RAG

Maki ships two memory plugins that work with any backend:

| Plugin | Stores | Best for |
|---|---|---|
| `obsidian_memory` | Markdown notes with YAML frontmatter in a vault folder | Memory you want to read, edit and search by hand in [Obsidian](https://obsidian.md): decisions, logs, task history |
| `rag_memory` | Text chunks plus embedding vectors in a vector store | Finding relevant passages by meaning in a large pile of text: docs, manuals, notes |

They combine well. Obsidian is the readable source of truth, and RAG is the index that finds the right note.

The examples below use [vLLM](backends.md#vllm) for the model and the embeddings, but any backend works for chat. For embeddings, use any callable that turns text into a vector (`MakiVllm.embed`, `MakiLlamaCpp.embed`, or your own).

```bash
pip install "maki-framework[vllm]"
vllm serve Qwen/Qwen2.5-1.5B-Instruct                     # chat model on :8000
vllm serve BAAI/bge-m3 --port 8001                        # embedding model on :8001 (add the pooling/embedding option your vLLM version requires)
```

```python
from maki import MakiVllm

llm = MakiVllm(model="Qwen/Qwen2.5-1.5B-Instruct")
embedder = MakiVllm(model="BAAI/bge-m3", base_url="http://127.0.0.1:8001/v1")
```

## Keeping an agent's own memory small

By default a stateful agent pastes its whole recent conversation into every prompt, plugin results go to the model in full, and `task_history` keeps every answer in full. A `MemoryPolicy` limits each of those. Nothing changes unless you pass one:

```python
from maki.agents import Agent
from maki.objects import MemoryPolicy

agent = Agent(
    name="Researcher",
    maki_instance=llm,
    stateful=True,
    memory=MemoryPolicy(
        history_mode="messages",     # prior turns as chat messages, not pasted text
        token_budget=2048,           # estimated tokens of conversation to keep
        tool_result_chars=2000,      # clip each plugin result (head and tail kept)
        history_entry_chars=2000,    # clip what task_history / reasoning_history store
    ),
)
```

| Field | Default | Effect |
|---|---|---|
| `history_mode` | `"text"` | `"messages"` sends prior turns as structured chat history when the backend declares `supports_history` (Ollama, OpenAI, Anthropic and their subclasses). Other backends keep the text form and log it once, so memory is never silently lost. |
| `token_budget` | 4096 | Estimated tokens (`len // 4`) of conversation kept. The oldest turn pairs are dropped first. |
| `tool_result_chars` | none | A plugin result longer than this is cut to its first and last half with a `[... N characters omitted ...]` marker. Errors and short results are untouched. |
| `history_entry_chars` | none | The same clipping for each field stored in `task_history` and `reasoning_history`. The conversation memory keeps full answers. |

`maki serve` takes the same fields under a `memory:` key in the agent YAML; unknown keys are an error.

What to expect, measured by the offline benchmark in `maki/test/memory_bench.py` (`python -m maki.test.memory_bench`; token counts are the `len // 4` estimate):

- **Tool results and stored history:** clipping to 2000 characters cut a four-round, 20,000-character tool run from about 20,000 to 2,100 tokens in the last round, and 50 stored 20,000-character answers from 1.0 MB to 0.1 MB.
- **Conversation tokens:** `history_mode="messages"` alone saves about 2% (it drops the `Task:`/`Response:` labels). The reason to use it is a stable prefix that provider prompt caches can reuse: `Usage.cached_tokens` reports what Anthropic and OpenAI served from cache. A smaller `token_budget` saves much more (about 40% at 2048 on the 20-turn run) but drops older turns sooner.
- **Older facts are still dropped silently once they fall out of the budget.** A fact stated on turn 1 is gone by turn 15 with every setting above. Keep anything that must survive in the task or in `context`.

## Obsidian memory

`ObsidianMemory` reads and writes notes in a vault folder. The vault is any directory; open it in Obsidian to browse, edit and link the notes. It is resolved from the `vault_path` argument, then the `MAKI_VAULT_PATH` environment variable, then `~/ObsidianVaults/maki`.

```python
from maki.plugins.obsidian_memory import ObsidianMemory

vault = ObsidianMemory(vault_path="~/ObsidianVaults/research")

path = vault.append_note(
    folder="decisions",
    title="Use vLLM for batch jobs",
    frontmatter={"tags": ["infra", "llm"], "status": "open"},
    body="vLLM batches concurrent requests on the GPU, so nightly summaries finish faster.",
)
print(path)                                    # decisions/Use_vLLM_for_batch_jobs.md

note = vault.read_note(path)                   # {"frontmatter": {...}, "body": "..."}
vault.update_frontmatter(path, "status", "accepted")

accepted_infra = vault.query_notes("decisions", tags=["infra"], frontmatter_filters={"status": "accepted"})
recent = vault.query_notes("decisions", since_hours=24, limit=5)
```

Notes are plain `.md` files, so they show up in Obsidian immediately. Writing a note with an existing title overwrites it.

### Example: an agent that remembers across runs

Read what the vault already holds, hand it to the agent as context, and save the result as a new note. The next run starts with that history.

```python
from datetime import date
from maki.agents import Agent

agent = Agent(name="Researcher", maki_instance=llm, role="research assistant")

def run_with_memory(task: str) -> str:
    past = vault.query_notes("research-log", limit=5)
    memory = "\n\n".join(f"{n['frontmatter'].get('title', n['path'])}: {n['body']}" for n in past)

    answer = agent.execute_task(task, context={"previous notes": memory} if memory else None)

    vault.append_note(
        folder="research-log",
        title=f"{date.today()} {task[:40]}",
        frontmatter={"title": task[:60], "tags": ["research"], "date": str(date.today())},
        body=answer,
    )
    return answer

print(run_with_memory("Compare vLLM and llama.cpp for serving a 7B model."))
print(run_with_memory("Which of those two is better for a laptop?"))   # sees the first answer
```

`query_notes` returns the newest notes first and trims each body to 500 characters, which keeps the context small.

### Example: a structured log

For machine-readable records, append JSON lines instead of notes:

```python
vault.append_to_jsonl("logs/runs.jsonl", {"task": "summarise", "tokens": 812, "ok": True})
for record in vault.read_jsonl("logs/runs.jsonl", limit=10):
    print(record)
```

### Letting the model call it

Load the plugin by name and set `use_plugins=True`; the model can then read and write notes itself. The tool-call arguments are passed through as the model produces them, so prefer simple calls (`read_note`, `list_folder`, `query_notes`) and do structured writes in your own code, as above.

```python
agent.load_plugin("obsidian_memory")
print(agent.execute_task("List the notes in the 'decisions' folder.", use_plugins=True))
```

## RAG memory

`RagMemory` embeds text and finds the closest chunks to a question. The store is chosen by a DSN, so you can start in memory and move to a database without touching your code:

| DSN | Store |
|---|---|
| `memory://` | In-process, nothing to install (default) |
| `chroma:///path/to/dir` | ChromaDB |
| `qdrant://host:6333` | Qdrant |
| `postgresql://user:pw@host/db` | Postgres with pgvector |
| `faiss:///path/to/dir` | FAISS on disk |

The DSN falls back to the `RAG_DSN` environment variable. Pass the embedding function as `embedder`:

```python
from maki.plugins.rag_memory import RagMemory

rag = RagMemory(dsn="memory://", embedder=embedder.embed)

rag.upsert("docs", "vLLM batches concurrent requests on the GPU.", metadata={"topic": "vllm"})
rag.upsert("docs", "llama.cpp runs GGUF models on CPU or GPU.", metadata={"topic": "llamacpp"})

for hit in rag.search("docs", "Which server is best for many parallel users?", k=2):
    print(f"{hit['score']:.2f}  {hit['text']}  {hit['metadata']}")

rag.search("docs", "GGUF", k=3, filter={"topic": "llamacpp"})    # restrict by metadata
```

Each hit is `{id, text, metadata, score}`, best first. `get`, `update`, `delete`, `list_collections` and `reindex` manage the records.

### Example: answer questions from your own documents

Ingest files once, then retrieve the best chunks and put them in the prompt:

```python
rag.ingest_document("manuals", "docs/handbook.md", metadata={"source": "handbook"})
rag.ingest_document("manuals", "docs/policy.pdf", metadata={"source": "policy"})   # digital PDFs work as-is

def ask(question: str) -> str:
    hits = rag.search("manuals", question, k=4)
    context = "\n\n".join(f"[{h['metadata'].get('source')}] {h['text']}" for h in hits)
    prompt = (
        "Answer using only the context below. If the answer is not there, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    return llm.chat(prompt).content

print(ask("How many vacation days do new hires get?"))
```

`ingest_document` splits the text into overlapping chunks (512 characters with 64 overlap by default; change them with `chunk_size` and `chunk_overlap`) and stores the file path, chunk index and any metadata you pass with every chunk. Plain text, Markdown and digital PDFs work out of the box. Scanned PDFs and images need the optional `ocr` plugin.

### Example: rerank for better answers

Vector search is fast but approximate. With a reranker model served by vLLM, fetch more candidates and keep the best few:

```python
reranker = MakiVllm(model="BAAI/bge-reranker-v2-m3", base_url="http://127.0.0.1:8002/v1")

def retrieve(question: str, k: int = 4) -> list[str]:
    candidates = [h["text"] for h in rag.search("manuals", question, k=20)]
    return [r["document"] for r in reranker.rerank(question, candidates, top_n=k)]
```

### Example: let an agent search the knowledge base

`load_plugin("rag_memory")` builds the plugin with the Ollama embeddings endpoint. To use your own embedder, create the instance yourself and register it on the agent:

```python
agent = Agent(name="Support", maki_instance=llm, role="support engineer")
agent.plugins["rag_memory"] = rag

print(agent.execute_task("Search the 'manuals' collection for the refund policy and summarise it.",
                         use_plugins=True))
```

## Obsidian and RAG together

Keep the vault as the source of truth and index it for semantic search. This turns a folder of notes into something you can ask questions of:

```python
def index_vault(folder: str = "") -> int:
    """Embed every note in the vault folder; safe to re-run (same path = same record)."""
    count = 0
    for rel_path in vault.list_folder(folder):
        note = vault.read_note(rel_path)
        if not note["body"].strip():
            continue
        rag.upsert(
            "vault",
            note["body"],
            metadata={"path": rel_path, **{k: str(v) for k, v in note["frontmatter"].items()}},
            id=rel_path,
        )
        count += 1
    return count

print(index_vault(), "notes indexed")

for hit in rag.search("vault", "why did we pick vLLM?", k=3):
    print(hit["metadata"]["path"], f"{hit['score']:.2f}")
```

Using the note path as the record id makes re-indexing an update instead of a duplicate. Run `index_vault()` again after you edit notes, or call it at the start of each run.

!!! note
    The built-in `memory://` store lives in the process and is empty on the next start. Use `chroma://`, `qdrant://`, `postgresql://` or `faiss://` to keep the index between runs. The vault itself is always on disk.

## Choosing between them

- **Few notes, you want to read and edit them:** Obsidian alone. Filter with `query_notes`.
- **Many documents, you need the relevant passage:** RAG alone.
- **A growing knowledge base you also curate by hand:** both, with the vault as the source and RAG as the index.
