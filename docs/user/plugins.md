# Plugins

Plugins give agents tools. Load one by name:

```python
from maki import MakiLLama
from maki.agents import Agent

agent = Agent(name="ToolUser", maki_instance=MakiLLama(model="llama3.2"), role="assistant")
agent.load_plugin("file_reader")

result = agent.execute_task(
    "Read the first lines of README.md and summarize them.",
    use_plugins=True,
)
print(result)
```

When `use_plugins=True` (or the backend supports native tool calling), the plugin's methods are advertised to the model and executed automatically. You can also load your own plugin by passing `plugin_path`.

## Built-in plugins

| Plugin | Description | Extra |
|---|---|---|
| `directory_reader` | List and inspect directory contents | none |
| `file_reader` | Read files from disk | none |
| `file_writer` | Write files to disk | none |
| `file_mover` | Move and rename files and directories the same way on every OS, with a preview, a journal and a one-step undo | none |
| `json_reader` | Parse and query JSON files | none |
| `image_classifier` | Classify images via a local model | none |
| `ocr` | Extract text from images (not in the default registry; load via `plugin_path`) | none |
| `web_search` | RSS, HackerNews, Reddit, GitHub Trending, Lobste.rs | `web` |
| `web_to_md` | Fetch a URL and convert to Markdown | `web` |
| `provider_updates` | Fetch LLM provider release notes | `web` |
| `trend_search` | Google Trends queries | `trends` |
| `ftp_client` | FTP/SFTP file transfers | `ftp` |
| `alpaca_data` | Crypto, equity and forex bars and quotes | `alpaca` |
| `alpaca_news` | Financial news from Alpaca and RSS | `alpaca` |
| `alpaca_trading` | Submit and manage Alpaca trades | `alpaca` |
| `alpaca_stream` | Live crypto data stream | `alpaca` |
| `obsidian_memory` | Persistent note-based memory (Obsidian vault) | none |
| `rag_memory` | Retrieval-augmented memory with pluggable vector backends | none |
| `postgres_store` | PostgreSQL client with pooled connections, verified TLS and a migration runner; the model only gets a `ping` health check | `postgres` |

See [Memory: Obsidian and RAG](memory.md) for worked examples of the two memory plugins.

## Safety model

Tool calls are **fail-closed**:

- A plugin exposes only the methods listed in its class-level `ALLOWED_METHODS`. No list means nothing is exposed.
- Destructive methods (file writes and moves, FTP transfers and deletes, trades) are listed in `DANGEROUS_METHODS` and additionally require `Agent(allow_dangerous_tools=True)`.
- The trading plugin runs in paper mode unless you opt in to live trading.
- Web plugins fetch through Maki's hardened connector, which blocks private and reserved address ranges for content-derived URLs.

```python
# Needed before the model may write files:
agent = Agent(name="Writer", maki_instance=llm, role="writer", allow_dangerous_tools=True)
agent.load_plugin("file_writer")
```

## Plugin options

`load_plugin` accepts keyword options that are passed to the plugin's `register_plugin`. They are set by you,
never by the model. For example, `file_mover` works inside `base_dir` unless you widen it:

```python
agent = Agent(name="Organiser", maki_instance=llm, role="file organiser", allow_dangerous_tools=True)
agent.load_plugin("file_mover", base_dir="/path/to/project", allowed_roots=["/path/to/archive"])
```

`allow_any_path=True` lets the plugin touch anything the process can reach; use it only for agents that
do not read untrusted content. See the plugin's README for the options, the undo rules and the per-OS notes.
