<p align="center">
  <img src="img/logo.png" alt="Maki Logo" width="200" />
</p>

# Maki

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![CI](https://github.com/BowlOfData/maki/actions/workflows/python-package.yml/badge.svg)](https://github.com/BowlOfData/maki/actions/workflows/python-package.yml)
[![Ollama](https://img.shields.io/badge/LLM-Ollama%20local-lightgrey?logo=ollama)](https://ollama.ai/)
[![HuggingFace](https://img.shields.io/badge/LLM-HuggingFace-yellow?logo=huggingface)](https://huggingface.co/)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)](https://github.com/)

**Maki is a Python framework for multi-agent LLM applications that run wherever your models do: on your own hardware (through Ollama or llama.cpp), on a hosted API (OpenAI, Anthropic, OpenRouter), or a mix of both, with the same agent code.**

It is for developers who want to build and run tool-using agents on local models first, keep the option of moving any task to a hosted model by swapping one object, and have guardrails switched on by default while those agents touch files, the web, or a broker account.

---

## Why Maki

- **Local and hosted models are equals.** Ollama and llama.cpp are built-in backends, not add-ons, and sit next to OpenAI, Anthropic, and OpenRouter behind the same `LLMBackend` contract. Agents, workflows, and plugins never know which one they are talking to. An in-process HuggingFace Transformers backend is also included (manual install, see [below](#huggingface-backend)).
- **Guardrails on by default.** Requests Maki makes itself (Ollama, remote agents, web plugins) go through a hardened connector. URLs that come from content, such as pages and feeds, are checked against private and reserved address ranges at connect time (redirect hops included); operator-configured endpoints like a LAN Ollama host are allowed. Plugins are fail-closed: a model can only call methods a plugin explicitly lists, and destructive ones (file writes, FTP transfers, trades) stay disabled until you pass `Agent(allow_dangerous_tools=True)`. The trading plugin runs in paper mode unless you opt in to live.
- **Small core.** The base install has three dependencies (`requests`, `httpx`, `python-dotenv`). Everything else is an opt-in extra.
- **Agents as services.** `maki serve` exposes any agent over HTTP, and `AgentProxy` lets another process use it as if it were local, with a circuit breaker and optional bearer-token auth.
- **Batteries included.** 16 built-in plugins (files, web, market data, memory) and a workflow engine with dependency resolution, retries, parallel execution, and checkpoint/resume.

### Who it is not for

Maki is young (0.x) and deliberately small. If you need a large catalog of third-party integrations, hosted tracing and observability, or a big community and ecosystem around your framework, one of the larger agent frameworks will serve you better.

---

## Backends

| Backend | Where inference runs | Class | Requires |
|---|---|---|---|
| Ollama | Your machine or LAN | `MakiLLama` | a running Ollama server |
| llama.cpp | Your machine or LAN | `MakiLlamaCpp` | `maki-framework[llamacpp]` and a running `llama serve` |
| HuggingFace Transformers | In-process, on your CPU/GPU | `HFBackend` | `torch`, `transformers`, `accelerate` (manual install, see [HuggingFace Backend](#huggingface-backend)) |
| OpenAI | Hosted | `MakiOpenAI` | `maki-framework[openai]` |
| Anthropic | Hosted | `MakiAnthropic` | `maki-framework[anthropic]` |
| OpenRouter | Hosted, any vendor-namespaced model | `MakiOpenRouter` | `maki-framework[openrouter]` |

The same agent runs on any of them; only the backend object changes:

```python
from maki import MakiLLama, MakiLlamaCpp, MakiOpenAI, MakiAnthropic
from maki.agents import Agent

def reviewer(llm):
    return Agent(
        name="Reviewer",
        maki_instance=llm,
        role="code reviewer",
        instructions="Focus on bugs, regressions, and missing validation.",
    )

agent = reviewer(MakiLLama(model="gemma4:26b"))              # local, via Ollama
# agent = reviewer(MakiLlamaCpp())                           # local, via llama.cpp
# agent = reviewer(MakiOpenAI(model="gpt-4o"))               # hosted
# agent = reviewer(MakiAnthropic(model="claude-sonnet-4-5")) # hosted

print(agent.execute_task("Review this design: a plugin system with file access."))
```

---

## Features

- `MakiLLama` — Ollama chat API with synchronous, streaming, async, and vision-capable workflows
- `MakiLlamaCpp` — llama.cpp `llama serve` (GGUF models) with streaming, async, vision, tool calling, separate reasoning output, embeddings, and reranking
- `MakiOpenAI` — OpenAI chat completions, including reasoning models (o3/o4)
- `MakiAnthropic` — Anthropic messages API (Claude Sonnet, Haiku, Opus)
- `HFBackend` — direct HuggingFace Transformers integration with quantization and device selection
- `Agent` — role-based agents with task execution, memory, reasoning, and plugin support; per-agent execution lock for concurrent safety
- `AgentManager` — multi-agent orchestration: sequential pipelines, collaborative tasks, and dependency-aware workflows with parallel batching and checkpoint/resume
- `ConversationMemory` — token-budgeted, pair-based conversation history shared by `Agent` (stateful mode) and `ChatSession`
- Native tool-calling for all backends (Ollama `tools=`, llama.cpp, OpenAI, Anthropic tool use) with multi-round execution and self-correction
- 16 built-in plugins covering files, web content, search, trading, and memory
- Distributed agent serving: `maki serve` exposes any agent over HTTP; `AgentProxy` consumes remote agents transparently
- Hardened HTTP connector: URL validation, private-address blocking, DNS pinning on the default path, typed error classification, and configurable timeouts (hosted-model traffic goes through the vendors' own SDKs)
- Fail-closed plugin security: every plugin declares `ALLOWED_METHODS`; destructive methods require explicit opt-in
- Explicit logging setup and a typed exception hierarchy

---

## Installation

```bash
curl -fsSL https://raw.githubusercontent.com/BowlOfData/maki/main/install.sh | bash
```

This clones the repo into `~/.maki`, installs it into an isolated virtual environment, and symlinks the `maki` command into `~/.local/bin` (override with the `MAKI_INSTALL_DIR` / `MAKI_BIN_DIR` env vars). Re-run it any time to update to the latest `main`.

### From source

```bash
pip install -e .
```

For development tools:

```bash
pip install -e ".[dev]"
```

Some built-in plugins and backends rely on optional extras (defined in [pyproject.toml](pyproject.toml)):

- `maki-framework[web]` — `feedparser`, `readability-lxml`, `html2text` (web search / web-to-Markdown)
- `maki-framework[trends]` — `pytrends` (Google Trends)
- `maki-framework[alpaca]` — `alpaca-py` (market data, news, trading, streaming)
- `maki-framework[ftp]` — `paramiko` (FTP/SFTP)
- `maki-framework[gui]` — `PySide6` (desktop GUI)
- `maki-framework[openai]` — `openai` (OpenAI backend)
- `maki-framework[anthropic]` — `anthropic` (Anthropic backend)
- `maki-framework[openrouter]` — `openai` (OpenRouter backend, via its OpenAI-compatible API)
- `maki-framework[llamacpp]` — `openai` (llama.cpp backend, via the `llama serve` OpenAI-compatible API)
- `maki-framework[distributed]` — `fastapi`, `uvicorn`, `pyyaml` (agent server and proxies)
- `maki-framework[distributed-redis]` — `redis` (Redis workflow checkpoints)

Install everything with `pip install -e ".[all]"`.

---

## Configuration

Shared runtime defaults live in [maki/config.py](maki/config.py). All values are overridable via environment variables or a `.env` file (`python-dotenv` is supported).

| Variable | Description |
|---|---|
| `MAKI_OLLAMA_BASE_URL` | Full Ollama base URL |
| `MAKI_OLLAMA_HOST` | Ollama hostname |
| `MAKI_OLLAMA_PORT` | Ollama port |
| `MAKI_LLAMACPP_BASE_URL` | llama.cpp server URL (default `http://127.0.0.1:8080/v1`) |
| `MAKI_LLAMACPP_MODEL` | llama.cpp model name (default `local-model`; set it in router mode) |
| `LLAMACPP_API_KEY` | llama.cpp API key, only if the server was started with `--api-key` |
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
model: gemma4:26b
role: assistant
# backend: ollama | llamacpp | openai | anthropic | openrouter (default: ollama)
# base_url: http://127.0.0.1:8080/v1   # llamacpp only
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
| `ocr` | Extract text from images (not in the default registry; load via `plugin_path`) | — |
| `web_search` | RSS, HackerNews, Reddit, GitHub Trending, Lobste.rs | `web` |
| `web_to_md` | Fetch a URL and convert to Markdown | `web` |
| `provider_updates` | Fetch LLM provider release notes | `web` |
| `trend_search` | Google Trends queries | `trends` |
| `ftp_client` | FTP/SFTP file transfers | `ftp` |
| `alpaca_data` | Crypto, equity, and forex bar and quote data | `alpaca` |
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

## llama.cpp Backend

`MakiLlamaCpp` talks to a [llama.cpp](https://llama.app/docs/introduction) server, which runs GGUF models from Hugging Face or a local file. Install the extra and start a server:

```bash
pip install "maki-framework[llamacpp]"
llama serve -hf ggml-org/gemma-4-e4b-it-GGUF:Q4_0    # listens on http://127.0.0.1:8080
```

```python
from maki import MakiLlamaCpp
from maki.objects import GenerationConfig

llm = MakiLlamaCpp(config=GenerationConfig(temperature=0.7, top_k=40))
print(llm.health())                         # False while the model is still loading

response = llm.chat("What is the capital of France?")
print(response.content)
print(response.reasoning)                   # thinking text from reasoning models, else None

for chunk in llm.stream("Tell me a joke"):
    print(chunk, end="", flush=True)
```

- **Everything `MakiOpenAI` does** works unchanged: chat, streaming, async, vision (images as base64), sessions, and native tool calling with agents and plugins.
- **llama.cpp sampling settings** `top_k` and `repeat_penalty` from `GenerationConfig` are sent to the server.
- **Long generations:** `chat_collect()` (used by `Agent(use_streaming=True)`) streams internally, so the timeout applies per chunk instead of to the whole answer.
- **Router mode:** start `llama serve` without a model and pass the model name, e.g. `MakiLlamaCpp(model="ggml-org/gemma-3-4b-it-qat-GGUF:Q4_0")`. The server loads it on demand.
- **Other servers:** pass `base_url="http://192.168.1.20:8080/v1"` for a LAN host, and `api_key=` (or `LLAMACPP_API_KEY`) if it was started with `--api-key`.

Embeddings and reranking need a server started with `--embedding` or `--rerank`:

```python
from maki.plugins.rag_memory import RagMemory

embedder = MakiLlamaCpp()                   # llama serve -hf unsloth/embeddinggemma-300m-GGUF --embedding
rag = RagMemory(dsn="memory://", embedder=embedder.embed)

reranker = MakiLlamaCpp()                   # llama serve -m reranker.gguf --rerank
for hit in reranker.rerank("What is a panda?", ["hi", "The giant panda is a bear."], top_n=1):
    print(hit["relevance_score"], hit["document"])
```

Chat requests go through the `openai` SDK; `health()`, `embed()` and `rerank()` go through Maki's connector, which allows loopback and LAN addresses for this operator-configured endpoint.

---

## HuggingFace Backend

`HFBackend` runs models directly via HuggingFace Transformers — no Ollama required.

`torch`, `transformers`, and `accelerate` are not installed with Maki (they are large and platform-specific), so install them first:

```bash
pip install torch transformers accelerate   # add bitsandbytes for 4/8-bit quantization
```

```python
from maki import HFBackend

llm = HFBackend(model_id="mistralai/Mistral-7B-Instruct-v0.3", device="cuda")
response = llm.chat("Explain attention mechanisms.")
print(response.content)
```

Supports quantization (`load_in_4bit`, `load_in_8bit`) and device selection (`cpu`, `cuda`, `mps`; auto-detected when omitted). Remote model code is not executed unless you pass `trust_remote_code=True`.

---

## Architecture

<p align="center">
  <img src="img/diagram.png" alt="Maki Architecture Diagram" width="900" />
</p>

The framework is organized into four layers on top of a shared infrastructure layer:

- **Public API** — `maki/__init__.py` lazy-loads all exports on first access
- **LLM Backends** — `MakiLLama`, `MakiOpenAI`, `MakiAnthropic`, `MakiOpenRouter`, `MakiLlamaCpp`, and `HFBackend` all implement the abstract `LLMBackend` contract
- **Agent System** — `Agent` composes `PluginHandler` and `ReasoningEngine` mixins; `AgentManager` orchestrates agents via `WorkflowTask` and `WorkflowState`
- **Distributed Layer** — `AgentServer` (FastAPI) exposes agents over HTTP; `AgentProxy` provides a remote-agent client with circuit-breaking; `DistributedAgentManager` mixes local and remote agents
- **Infrastructure** — `Connector` (hardened HTTP: URL validation, connect-time IP pinning by default), shared data classes, typed exceptions, runtime config, and structured logging

The Plugin System sits alongside the Agent layer: plugins are loaded on demand and invoked automatically when the LLM emits a `TOOL:` directive or via native tool-calling APIs (Ollama, OpenAI, Anthropic).

---

## Public API

Top-level imports exposed by `maki`:

- `MakiLLama`, `MakiOpenAI`, `MakiAnthropic`, `MakiOpenRouter`, `MakiLlamaCpp`, `HFBackend`
- `LLMBackend`, `BackendType`
- `Agent`, `AgentManager`
- `GenerationConfig`, `LLMResponse`, `Message`, `ToolCall`
- `ConversationMemory`, `RateLimiter`
- `Connector`, `Utils`
- `config`

All exports are lazy-loaded on first access.

---

## Desktop App

The repository includes a PySide6/QML desktop shell (requires `maki-framework[gui]`):

```bash
maki-gui
```

---

## Testing

```bash
pytest
```

900+ tests covering backends, agents, workflows, plugins, connectors, distributed layer, and security-related behaviour. Tests marked `@pytest.mark.network` (requiring live external services) are excluded by default; run them explicitly with `pytest -m network`.

---

## Contributing

Contributions are welcome: bug fixes, documentation improvements, new plugins, and feature suggestions all help move the project forward. Open an issue or submit a pull request on GitHub.

If you are interested in this line of research, consider joining [Bowl of Data](https://bowlofdata.net/), an open-source AI research community.

---

## License

[MIT](LICENSE)
