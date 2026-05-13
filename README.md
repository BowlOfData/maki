![Maki Logo](img/logo.png#center)

# Maki

Maki is a Python framework for building LLM applications with local backends, multi-agent coordination, plugin-based tool use, and workflow orchestration.

It currently ships with:

- `MakiLLama` for Ollama's chat API, streaming, async calls, and sessions
- `Agent` and `AgentManager` for role-based multi-agent execution
- Built-in plugins for files, directories, JSON, web content, FTP/SFTP, search, and image classification
- A desktop GUI shell exposed through `maki-gui`

## Features

- `MakiLLama` Ollama backend built on the chat API with a shared `LLMBackend` contract
- Synchronous, streaming, async, and vision-capable Ollama workflows
- Stateful chat sessions and stateful agents
- Agent memory, retry logic, and reasoning helpers
- Plugin loading plus structured `TOOL:` calls from agents
- Workflow execution with dependencies, retries, conditions, and parallel batches
- Explicit logging setup and custom exception types

## Installation

Install the package:

```bash
pip install .
```

For development tools:

```bash
pip install ".[dev]"
```

Some built-in plugins rely on extra packages listed in [requirements.txt](requirements.txt), especially:

- `feedparser` for RSS parsing
- `pytrends` for Google Trends
- `readability-lxml` and `html2text` for web-to-Markdown conversion

If you want all plugin dependencies available, install from `requirements.txt` as well.

## Configuration

Shared runtime defaults live in [maki/config.py](maki/config.py).

Supported environment variables include:

- `MAKI_OLLAMA_BASE_URL`
- `MAKI_OLLAMA_HOST`
- `MAKI_OLLAMA_PORT`
- `MAKI_DEFAULT_MODEL`
- `MAKI_DEFAULT_TEMPERATURE`
- `MAKI_REQUEST_TIMEOUT`
- `MAKI_HTTP_TIMEOUT`
- `MAKI_LOG_LEVEL`
- `MAKI_WEB_USER_AGENT`

`python-dotenv` is supported, so `.env` values can be loaded automatically.

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

## Agents

### Basic agent usage

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
print(steps)

subtasks = agent.decompose_task("Prepare this repository for a public release")
print(subtasks)
```

### Streaming task execution

```python
from maki import MakiLLama
from maki.agents import Agent

llm = MakiLLama(model="gemma4:26b")
agent = Agent(name="Writer", maki_instance=llm, role="writer")

for chunk in agent.stream_task("Draft a short changelog entry."):
    print(chunk, end="", flush=True)
```

### Long-running tasks with `use_streaming`

By default, `execute_task` sends one blocking HTTP request. If the model takes longer than the configured timeout (default 120 s) to finish, it raises `MakiTimeoutError`.

Set `use_streaming=True` on the agent to avoid this. Internally, Maki switches to a streaming connection and collects the full response — the timeout then applies *per chunk* rather than to the whole reply, so generation can take as long as it needs.

```python
from maki import MakiLLama
from maki.agents import Agent

llm = MakiLLama(model="gemma4:26b")
agent = Agent(
    name="Ranker",
    maki_instance=llm,
    role="content ranker",
    use_streaming=True,   # no global timeout on the response
)

result = agent.execute_task("Rank these 50 articles by relevance: ...")
print(result)
```

Use this whenever the task involves a large prompt or a long expected output — ranking, summarisation of many items, code generation, etc.

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

Supported manager patterns include:

- `assign_task()` for direct task routing
- `coordinate_agents()` for sequential multi-agent execution with optional synthesis
- `collaborative_task()` for many agents working on one task
- `run_workflow()` for dict-based or `WorkflowTask`-based orchestration

## Plugins

Built-in plugins are registered in [maki/plugins/__init__.py](maki/plugins/__init__.py):

- `directory_reader`
- `file_reader`
- `file_writer`
- `ftp_client`
- `image_classifier`
- `json_reader`
- `media_search`
- `provider_updates`
- `trend_search`
- `web_search`
- `web_to_md`

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

When `use_plugins=True`, the agent prompt exposes available plugin methods and the model can emit `TOOL:` directives that Maki validates and executes.

## Desktop App

The repository includes a desktop GUI shell launched with:

```bash
maki-gui
```

The current implementation is a PySide6/QML bootstrap with an application shell, not a full production desktop client yet.

## Public API

Top-level imports currently exposed by `maki` include:

- `MakiLLama`
- `Agent`
- `AgentManager`
- `LLMBackend`
- `GenerationConfig`
- `LLMResponse`
- `Message`
- `RateLimiter`
- `config`

## Testing

Run the test suite with:

```bash
python3 -m pytest
```

The repository currently has broad automated coverage across backends, agents, workflows, plugins, and security-related behavior.

Test