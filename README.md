<p align="center">
  <img src="img/logo.png" alt="Maki Logo" width="200" />
</p>

# Maki

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-888%20passing-brightgreen?logo=pytest)](https://pytest.org/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama%20local-lightgrey?logo=ollama)](https://ollama.ai/)
[![HuggingFace](https://img.shields.io/badge/LLM-HuggingFace-yellow?logo=huggingface)](https://huggingface.co/)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)](https://github.com/)

**Maki** is a Python framework for building multi-agent LLM applications. It supports multiple LLM backends (Ollama, OpenAI, Anthropic, HuggingFace), a plugin system with 17 built-in tools, a workflow engine with dependency resolution and parallel execution, and a distributed layer for serving agents over HTTP.

---

## Architecture

<p align="center">
  <img src="img/diagram.png" alt="Maki Architecture Diagram" width="900" />
</p>

The framework is organized into four layers:

- **Public API** — `maki/__init__.py` lazy-loads all exports on first access
- **LLM Backends** — `MakiLLama`, `MakiOpenAI`, `MakiAnthropic`, and `HFBackend` all implement the abstract `LLMBackend` contract
- **Agent System** — `Agent` composes `PluginHandler` and `ReasoningEngine` mixins; `AgentManager` orchestrates agents via `WorkflowTask` and `WorkflowState`
- **Distributed Layer** — `AgentServer` (FastAPI) exposes agents over HTTP; `AgentProxy` provides a remote-agent client with circuit-breaking; `DistributedAgentManager` mixes local and remote agents
- **Infrastructure** — `Connector` (SSRF-protected HTTP with connect-time IP pinning), shared data classes, typed exceptions, runtime config, and structured logging

The Plugin System sits alongside the Agent layer: plugins are loaded on demand and invoked automatically when the LLM emits a `TOOL:` directive or via native tool-calling APIs (Ollama, OpenAI, Anthropic).

---

## Features

- `MakiLLama` — Ollama chat API with synchronous, streaming, async, and vision-capable workflows
- `MakiOpenAI` — OpenAI chat completions, including reasoning models (o3/o4)
- `MakiAnthropic` — Anthropic messages API (Claude Sonnet, Haiku, Opus)
- `HFBackend` — direct HuggingFace Transformers integration with quantization and device selection
- `Agent` — role-based agents with task execution, memory, reasoning, and plugin support; per-agent execution lock for concurrent safety
- `AgentManager` — multi-agent orchestration: sequential pipelines, collaborative tasks, and dependency-aware workflows with parallel batching and checkpoint/resume
- `ConversationMemory` — token-budgeted, pair-based conversation history shared by `Agent` (stateful mode) and `ChatSession`
- Native tool-calling for all backends (Ollama `tools=`, OpenAI, Anthropic tool use) with multi-round execution and self-correction
- 17 built-in plugins covering files, web content, search, trading, memory, and media
- Distributed agent serving: `maki serve` exposes any agent over HTTP; `AgentProxy` consumes remote agents transparently
- SSRF-protected HTTP connector with DNS pinning, error classification, and configurable timeouts
- Fail-closed plugin security: every plugin declares `ALLOWED_METHODS`; destructive methods require explicit opt-in
- Explicit logging setup and a typed exception hierarchy

---

## Installation

```bash
pip install -e .
```

For development tools:

```bash
pip install -e ".[dev]"
```

Some built-in plugins and backends rely on optional extras (defined in [pyproject.toml](pyproject.toml)):

- `maki[web]` — `feedparser`, `readability-lxml`, `html2text` (web search / web-to-Markdown)
- `maki[trends]` — `pytrends` (Google Trends)
- `maki[alpaca]` — `alpaca-py` (market data, news, trading, streaming)
- `maki[ftp]` — `paramiko` (FTP/SFTP)
- `maki[gui]` — `PySide6` (desktop GUI)
- `maki[openai]` — `openai` (OpenAI backend)
- `maki[anthropic]` — `anthropic` (Anthropic backend)
- `maki[distributed]` — `fastapi`, `uvicorn`, `pyyaml` (agent server and proxies)
- `maki[distributed-redis]` — `redis` (Redis workflow checkpoints)

Install everything with `pip install -e ".[all]"`.

---

## Configuration

Shared runtime defaults live in [maki/config.py](maki/config.py). All values are overridable via environment variables or a `.env` file (`python-dotenv` is supported).

| Variable | Description |
|---|---|
| `MAKI_OLLAMA_BASE_URL` | Full Ollama base URL |
| `MAKI_OLLAMA_HOST` | Ollama hostname |
| `MAKI_OLLAMA_PORT` | Ollama port |
| `MAKI_DEFAULT_MODEL` | Default model name |
| `MAKI_DEFAULT_TEMPERATURE` | Sampling temperature |
| `MAKI_REQUEST_TIMEOUT` | Per-request timeout (seconds) |
| `MAKI_HTTP_TIMEOUT` | Low-level HTTP timeout |
| `MAKI_LOG_LEVEL` | Logging level |
| `MAKI_WEB_USER_AGENT` | User-agent string for web plugins |

---

## Quick Start

### Basic request

```python
from maki import MakiLLama

llm = MakiLLama(model="gemma4:26b")
response = llm.chat("Explain recursion in one sentence.")
print(response.content)
```

### Chat, streaming, and async

```python
import asyncio
from maki import MakiLLama
from maki.objects import GenerationConfig

config = GenerationConfig(temperature=0.7, max_tokens=512)
llm = MakiLLama(model="gemma4:26b", config=config)

reply = llm.chat("Give me three project naming ideas.")
print(reply.content)

for chunk in llm.stream("Write a short haiku about testing"):
    print(chunk, end="", flush=True)

async def main():
    response = await llm.async_chat("Summarize the benefits of type hints.")
    print(response.content)

asyncio.run(main())
```

### Stateful session

```python
from maki import MakiLLama

llm = MakiLLama(model="gemma4:26b")
session = llm.session(system="You are a concise engineering assistant.")

session.say("We are building a release checklist.")
response = session.say("What should we verify before publishing a Python package?")
print(response.content)
```

### Hosted backends

```python
from maki import MakiOpenAI, MakiAnthropic

# OpenAI
llm = MakiOpenAI(model="gpt-4o")
response = llm.chat("What is the capital of France?")

# Anthropic
llm = MakiAnthropic(model="claude-sonnet-4-5")
response = llm.chat("Summarize this code in one sentence.")
```

---

## Agents

### Basic agent

```python
from maki import MakiLLama
from maki.agents import Agent

llm = MakiLLama(model="gemma4:26b")
agent = Agent(
    name="Reviewer",
    maki_instance=llm,
    role="code reviewer",
    instructions="Focus on bugs, regressions, and missing validation.",
    stateful=True,
)

result = agent.execute_task("Review this design: a plugin system with file access.")
print(result)
```

### Memory and reasoning helpers

```python
agent.remember("repo", "maki")
print(agent.recall("repo"))

steps = agent.think_step_by_step("How should we structure plugin validation?")
subtasks = agent.decompose_task("Prepare this repository for a public release")
```

### Streaming task execution

```python
for chunk in agent.stream_task("Draft a short changelog entry."):
    print(chunk, end="", flush=True)
```

### Long-running tasks with `use_streaming`

By default, `execute_task` sends one blocking HTTP request. For tasks that exceed the configured timeout (default 120 s), set `use_streaming=True` — the timeout then applies per chunk rather than to the whole response.

```python
agent = Agent(
    name="Ranker",
    maki_instance=llm,
    role="content ranker",
    use_streaming=True,
)

result = agent.execute_task("Rank these 50 articles by relevance: ...")
print(result)
```

---

## Agent Manager and Workflows

`AgentManager` coordinates multiple agents and can run collaborative or dependency-aware workflows.

```python
from maki import MakiLLama
from maki.agents import AgentManager, WorkflowTask

llm = MakiLLama(model="gemma4:26b")
manager = AgentManager(llm)

manager.add_agent("Researcher", role="researcher")
manager.add_agent("Writer", role="writer")

workflow = [
    WorkflowTask(
        name="research",
        agent="Researcher",
        task="Find the main public-release risks for this repository.",
    ),
    WorkflowTask(
        name="summary",
        agent="Writer",
        task="Summarize the research into a release checklist.",
        dependencies=["research"],
    ),
]

results = manager.run_workflow(workflow)
print(results["summary"]["result"])
```

Supported manager patterns:

| Method | Behaviour |
|---|---|
| `assign_task()` | Route a single task to one named agent |
| `coordinate_agents()` | Sequential multi-agent pipeline with optional synthesis step |
| `collaborative_task()` | All agents work on the same task independently |
| `run_workflow()` | Dependency-aware execution with retries and optional parallel batches |

---

## Distributed Layer

Serve any agent over HTTP with `maki serve`:

```bash
maki serve --config agent.yaml --host 127.0.0.1 --port 8100
```

```yaml
# agent.yaml
name: MyAgent
model: gemma4:27b
role: assistant
plugins:
  - web_search
  - file_reader
```

Connect to a remote agent from another process:

```python
from maki.distributed.proxy import AgentProxy

agent = AgentProxy(name="MyAgent", base_url="http://127.0.0.1:8100")
result = agent.execute_task("Summarize the latest AI news.")
```

`DistributedAgentManager` lets you mix local and remote agents in the same workflow.

---

## Plugins

Built-in plugins are registered in [maki/plugins/\_\_init\_\_.py](maki/plugins/__init__.py):

| Plugin | Description | Extra |
|---|---|---|
| `directory_reader` | List and inspect directory contents | — |
| `file_reader` | Read files from disk | — |
| `file_writer` | Write files to disk | — |
| `json_reader` | Parse and query JSON files | — |
| `image_classifier` | Classify images via a local model | — |
| `ocr` | Extract text from images | — |
| `web_search` | RSS, HackerNews, Reddit, GitHub Trending, Lobste.rs | `web` |
| `web_to_md` | Fetch a URL and convert to Markdown | `web` |
| `provider_updates` | Fetch LLM provider release notes | `web` |
| `media_search` | Search Pexels for images | `web` |
| `trend_search` | Google Trends queries | `trends` |
| `ftp_client` | FTP/SFTP file transfers | `ftp` |
| `alpaca_data` | Crypto bar and quote data | `alpaca` |
| `alpaca_news` | Financial news from Alpaca and RSS | `alpaca` |
| `alpaca_trading` | Submit and manage Alpaca trades | `alpaca` |
| `alpaca_stream` | Live crypto data stream | `alpaca` |
| `obsidian_memory` | Persistent note-based memory (Obsidian vault) | — |
| `rag_memory` | Retrieval-augmented memory with pluggable vector backends | — |

### Loading a plugin in an agent

```python
from maki import MakiLLama
from maki.agents import Agent

llm = MakiLLama(model="gemma4:26b")
agent = Agent(name="ToolUser", maki_instance=llm, role="assistant")
agent.load_plugin("file_reader")

result = agent.execute_task(
    "Read the first lines of README.md and summarize them.",
    use_plugins=True,
)
print(result)
```

When `use_plugins=True` (or the backend supports native tool-calling), available plugin methods are advertised to the model and executed automatically. Destructive methods (file writes, trades, FTP deletes) require `Agent(allow_dangerous_tools=True)`.

---

## HuggingFace Backend

`HFBackend` runs models directly via HuggingFace Transformers — no Ollama required.

```python
from maki import HFBackend

llm = HFBackend(model="mistralai/Mistral-7B-Instruct-v0.2", device="cuda")
response = llm.chat("Explain attention mechanisms.")
print(response.content)
```

Supports quantization and device selection (`cpu`, `cuda`, `mps`).

---

## Public API

Top-level imports exposed by `maki`:

- `MakiLLama`, `MakiOpenAI`, `MakiAnthropic`, `HFBackend`
- `LLMBackend`, `BackendType`
- `Agent`, `AgentManager`
- `GenerationConfig`, `LLMResponse`, `Message`, `ToolCall`
- `ConversationMemory`, `RateLimiter`
- `Connector`, `Utils`
- `config`

All exports are lazy-loaded on first access.

---

## Desktop App

The repository includes a PySide6/QML desktop shell (requires `maki[gui]`):

```bash
maki-gui
```

---

## Testing

```bash
pytest
```

888 tests covering backends, agents, workflows, plugins, connectors, distributed layer, and security-related behaviour. Tests marked `@pytest.mark.network` (requiring live external services) are excluded by default; run them explicitly with `pytest -m network`.

---

## Contributing

Contributions are welcome: bug fixes, documentation improvements, new plugins, and feature suggestions all help move the project forward. Open an issue or submit a pull request on GitHub.

If you are interested in this line of research, consider joining [Bowl of Data](https://bowlofdata.netlify.app/), an open-source AI research community.

---

## License

[MIT](LICENSE)
