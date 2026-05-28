<p align="center">
  <img src="img/logo.png" alt="Maki Logo" width="200" />
</p>

# Maki

[![Python 3.7+](https://img.shields.io/badge/python-3.7%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-405%20passing-brightgreen?logo=pytest)](https://pytest.org/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama%20local-lightgrey?logo=ollama)](https://ollama.ai/)
[![HuggingFace](https://img.shields.io/badge/LLM-HuggingFace-yellow?logo=huggingface)](https://huggingface.co/)
[![PySide6](https://img.shields.io/badge/GUI-PySide6-41CD52?logo=qt&logoColor=white)](https://doc.qt.io/qtforpython/)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)](https://github.com/)

**Maki** is a Python framework for building multi-agent LLM applications. It supports multiple LLM backends (Ollama and HuggingFace), a plugin system with 12 built-in tools, and a workflow engine with dependency resolution and parallel execution.

---

## Architecture

<p align="center">
  <img src="img/diagram.png" alt="Maki Architecture Diagram" width="900" />
</p>

The framework is organized into four layers:

- **Public API** — `maki/__init__.py` lazy-loads all exports on first access
- **LLM Backends** — `MakiLLama` (Ollama) and `HFBackend` (HuggingFace) both implement the abstract `LLMBackend` contract
- **Agent System** — `Agent` composes `PluginHandler` and `ReasoningEngine` mixins; `AgentManager` orchestrates agents via `WorkflowTask` and `WorkflowState`
- **Infrastructure** — `Connector` (SSRF-protected HTTP), shared data classes, typed exceptions, runtime config, and logging

The Plugin System sits alongside the Agent layer: plugins are loaded on demand and invoked automatically when the LLM emits a `TOOL:` directive.

---

## Features

- `MakiLLama` — Ollama chat API with synchronous, streaming, async, and vision-capable workflows
- `HFBackend` — direct HuggingFace Transformers integration with quantization and device selection
- `Agent` — role-based agents with task execution, memory, reasoning, and plugin support
- `AgentManager` — multi-agent orchestration: sequential pipelines, collaborative tasks, and dependency-aware workflows
- Stateful chat sessions and bounded conversation history
- 12 built-in plugins covering files, directories, JSON, web content, FTP/SFTP, search, image classification, OCR, and media
- SSRF-protected HTTP connector with error classification and timeout handling
- Explicit logging setup and a typed exception hierarchy

---

## Installation

```bash
pip install .
```

For development tools:

```bash
pip install ".[dev]"
```

Some built-in plugins rely on extra packages listed in [requirements.txt](requirements.txt):

- `feedparser` for RSS parsing
- `pytrends` for Google Trends
- `readability-lxml` and `html2text` for web-to-Markdown conversion

Install from `requirements.txt` to make all plugin dependencies available.

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

### Vision call

```python
import base64
from maki import MakiLLama

llm = MakiLLama(model="gemma4:26b")
with open("image.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

response = llm.chat_with_image("Describe this image.", image_b64=image_b64)
print(response.content)
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

Use this whenever the task involves a large prompt or a long expected output — ranking, summarisation of many items, code generation, etc.

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

## Plugins

Built-in plugins are registered in [maki/plugins/\_\_init\_\_.py](maki/plugins/__init__.py):

| Plugin | Description |
|---|---|
| `directory_reader` | List and inspect directory contents |
| `file_reader` | Read files from disk |
| `file_writer` | Write files to disk |
| `ftp_client` | FTP/SFTP file transfers |
| `image_classifier` | Classify images via a local model |
| `json_reader` | Parse and query JSON files |
| `media_search` | Search media sources |
| `ocr` | Extract text from images |
| `provider_updates` | Fetch provider news and updates |
| `trend_search` | Google Trends queries via `pytrends` |
| `web_search` | Web search integration |
| `web_to_md` | Fetch a URL and convert to Markdown |

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

When `use_plugins=True`, the agent prompt exposes available plugin methods and the model can emit `TOOL:` directives that Maki validates and executes automatically.

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

- `MakiLLama`
- `HFBackend`
- `Agent`
- `AgentManager`
- `LLMBackend`
- `GenerationConfig`
- `LLMResponse`
- `Message`
- `RateLimiter`
- `config`

All exports are lazy-loaded on first access.

---

## Desktop App

The repository includes a PySide6/QML desktop shell:

```bash
maki-gui
```

---

## Testing

```bash
pytest
```

405 tests covering backends, agents, workflows, plugins, connectors, and security-related behaviour.

---

## Contributing

Contributions are welcome: bug fixes, documentation improvements, new plugins, and feature suggestions all help move the project forward. Open an issue or submit a pull request on GitHub.

If you are interested in this line of research, consider joining [Bowl of Data](https://bowlofdata.netlify.app/), an open-source AI research community.

---

## License

[MIT](LICENSE)
