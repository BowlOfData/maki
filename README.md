![Maki Logo](img/logo.png#center)

# Maki

Maki is a Python framework for building multi-agent LLM applications. It supports multiple backends (Ollama, HuggingFace Transformers), a plugin system for extending agent capabilities, and a workflow engine for orchestrating complex multi-agent tasks.

## Features

- **Multiple LLM backends**: Ollama (via `Maki` and `MakiLLama`) and HuggingFace Transformers (via `HFBackend`)
- **Multi-agent system**: Create and coordinate agents with roles, instructions, and memory
- **Stateful agents**: Maintain conversation history across task executions
- **Streaming support**: Token-by-token output via `MakiLLama.stream()`
- **Async support**: `async_chat()` for asyncio-based applications
- **Multi-turn sessions**: `ChatSession` for persistent conversations
- **Plugin system**: Extend agents with built-in plugins (`file_reader`, `file_writer`, `directory_reader`, `web_to_md`, `ftp_client`)
- **Tool calling in agents**: Agents can invoke plugins via structured `TOOL:` directives
- **Workflow engine**: Dependency-based task orchestration with retries and parallel execution
- **Image input support**: Send image files alongside prompts
- **Configurable logging**: Explicit logging setup with no side effects on import
- **Custom exceptions**: `MakiNetworkError`, `MakiTimeoutError`, `MakiAPIError`

## Installation

### Option 1: Install with pip (recommended)
```bash
pip install .
```

### Option 2: Copy folder to your project
Copy the `maki` folder into your project and install dependencies:
```bash
pip install requests httpx
```

## Quick Start

```python
from maki import Maki, MakiLLama

# Ollama – simple generate API
maki = Maki(url="http://localhost", port="11434", model="llama3", temperature=0.7)
result = maki.request("What is the capital of France?")
print(result)

# Ollama – richer chat API with streaming and sessions
llm = MakiLLama(model="gemma3")
response = llm.chat("Tell me a joke")
print(response.content)
print(f"Tokens: {response.total_tokens} | Speed: {response.tokens_per_second:.1f} tok/s")
```

## Backends

### `Maki` – Ollama generate API

The base class. Use it when you need a minimal, synchronous interface to Ollama.

```python
from maki import Maki

maki = Maki(url="http://localhost", port="11434", model="llama3", temperature=0.7)

# Text request
result = maki.request("Explain quantum computing in one sentence.")

# Image request
result = maki.request_with_images("What's in this image?", img="photo.jpg")
```

### `MakiLLama` – Ollama chat API

A full-featured Ollama wrapper with streaming, async, and session support.

```python
from maki import MakiLLama
from maki.objects import GenerationConfig

config = GenerationConfig(temperature=0.8, max_tokens=1024)
llm = MakiLLama(model="gemma3", system_prompt="You are a concise assistant.", config=config)

# Single-turn chat
response = llm.chat("What is recursion?")
response_text = response.content

# Streaming
for chunk in llm.stream("Write a haiku about Python"):
    print(chunk, end="", flush=True)

# Async
import asyncio
response = asyncio.run(llm.async_chat("Explain async/await in Python"))

# List available local models
models = llm.list_models()

# Pull a model
llm.pull("mistral")
```

#### Multi-turn sessions

```python
session = llm.session(system="You are an expert chef specializing in Italian cuisine.")
r1 = session.say("What's the secret to perfect pasta carbonara?")
r2 = session.say("Can I substitute bacon for guanciale?")
session.print_history()
session.reset()
```

#### Factory functions

```python
from maki.makiLLama import gemma3, qwen, llama, mistral

llm = gemma3(system="You are a helpful assistant.")
llm = qwen(variant="qwen2.5:7b")
llm = llama(variant="llama3.2")
llm = mistral()
```

### `HFBackend` – HuggingFace Transformers

Runs models directly from HuggingFace without Ollama. Supports CUDA, MPS (Apple Silicon), and CPU.

```python
from maki import HFBackend
from maki.objects import GenerationConfig

# Load a model (downloads from HuggingFace Hub on first run)
llm = HFBackend(
    model_id="google/gemma-3-1b-it",
    load_in_4bit=True,   # optional quantization
)

# Simple request
result = llm.request("What is the capital of France?")

# Full generation with config
from maki.objects import GenerationConfig
config = GenerationConfig(max_tokens=512, temperature=0.7)
response = llm.generate([{"role": "user", "content": "Tell me a joke"}], config)
print(response.content)

# Streaming
for chunk in llm.stream([{"role": "user", "content": "Write a poem"}], config):
    print(chunk, end="", flush=True)

# Release GPU memory when done
llm.unload()
```

Supported model families: Gemma 3, Qwen 2.5, Llama 3, Mistral, Phi-3, and any HuggingFace chat model.

## Multi-Agent System

### Basic usage

```python
from maki import Maki
from maki.agents import AgentManager

maki = Maki(url="http://localhost", port="11434", model="llama3", temperature=0.7)
manager = AgentManager(maki)

researcher = manager.add_agent(
    name="Researcher",
    role="research analyst",
    instructions="You are an expert researcher who finds and analyzes information.",
)

writer = manager.add_agent(
    name="Writer",
    role="technical writer",
    instructions="You write clear, concise summaries from research findings.",
)

result = manager.assign_task("Researcher", "Research the benefits of renewable energy")
print(result)
```

### Stateful agents

Stateful agents include prior task results in subsequent prompts, enabling multi-turn reasoning.

```python
from maki import Maki
from maki.agents import Agent

maki = Maki(url="http://localhost", port="11434", model="llama3", temperature=0.7)
agent = Agent(name="Analyst", maki_instance=maki, role="data analyst", stateful=True)

agent.execute_task("Summarize the Q1 sales data: revenue $1.2M, units 4500.")
agent.execute_task("Based on Q1, what should our Q2 targets be?")  # sees Q1 result

agent.reset_conversation()  # clear history
```

### Agent memory

```python
agent.remember("client_name", "Acme Corp")
client = agent.recall("client_name")
agent.clear_memory()
```

### Streaming tasks

```python
from maki import MakiLLama
from maki.agents import Agent

llm = MakiLLama(model="gemma3")
agent = Agent(name="Writer", maki_instance=llm, role="writer")

for chunk in agent.stream_task("Write a short poem about the ocean"):
    print(chunk, end="", flush=True)
```

### Task retry

```python
result = agent.execute_task_with_retry(
    "Summarize the latest AI news",
    max_retries=3,
    retry_delay=2.0,
)
```

### Enhanced reasoning

```python
# Step-by-step reasoning
result = agent.think_step_by_step("How do we reduce customer churn?", steps=4)

# Self-correction
initial = agent.execute_task("Draft a product description for item X")
improved = agent.self_correct(initial, feedback="Make it shorter and more engaging", max_iterations=2)

# Task decomposition – returns a list of subtask dicts
subtasks = agent.decompose_task("Launch a new mobile app", max_subtasks=5)
for subtask in subtasks:
    print(subtask["description"], "→", subtask["expected_outcome"])
```

### Per-agent LLM instances

Each agent can use a different model or temperature:

```python
fast_llm = Maki(url="http://localhost", port="11434", model="mistral", temperature=0.3)
creative_llm = Maki(url="http://localhost", port="11434", model="llama3", temperature=0.9)

manager.add_agent("Classifier", role="classifier", maki_instance=fast_llm)
manager.add_agent("Storyteller", role="storyteller", maki_instance=creative_llm)
```

## Multi-Agent Coordination

### Coordinate agents sequentially

```python
tasks = [
    {"agent": "Researcher", "task": "Research renewable energy trends"},
    {"agent": "Writer", "task": "Summarize the research", "context": {"audience": "executives"}},
]

results = manager.coordinate_agents(
    tasks,
    coordination_prompt="Synthesize these findings into an executive briefing."
)
print(results.get("final_synthesis"))
```

### Collaborative task (all agents work on the same task)

```python
result = manager.collaborative_task(
    task="What are the key risks in our Q3 strategy?",
    agents=["Researcher", "Analyst", "Writer"],
    strict=False,   # True = fail if any agent errors
)
```

## Plugin System

Plugins extend agent capabilities with specialized tools.

```python
# Load a plugin
file_reader = agent.load_plugin("file_reader")

# Use directly
content = file_reader.read_file("report.txt")

# Or retrieve a loaded plugin later
plugin = agent.get_plugin("file_reader")

# Unload
agent.unload_plugin("file_reader")
```

### Built-in plugins

| Plugin | Description |
|---|---|
| `file_reader` | Read files from the filesystem |
| `file_writer` | Write files to the filesystem |
| `directory_reader` | List and explore directory contents |
| `web_to_md` | Fetch a URL and convert to Markdown |
| `ftp_client` | Connect to and transfer files via FTP |

### Tool calling in agents

When `use_plugins=True`, the agent can issue `TOOL:` calls that are automatically executed:

```python
agent.load_plugin("file_reader")
agent.load_plugin("web_to_md")

result = agent.execute_task(
    "Read the file config.json and summarize its contents",
    use_plugins=True,
)
```

### Custom plugins

Load a plugin from a custom path:

```python
plugin = agent.load_plugin("my_plugin", plugin_path="/path/to/plugins")
```

A plugin module must expose either a `register_plugin(maki_instance)` function or a class with the same name as the plugin.

## Workflow Engine

The workflow engine orchestrates multi-agent tasks with dependencies, retries, conditions, and parallel execution.

```python
from maki.agents import WorkflowTask, TaskStatus, WorkflowState

tasks = [
    WorkflowTask(
        name="research",
        agent="Researcher",
        task="Research the latest developments in AI safety",
        dependencies=[],
        max_retries=2,
        retry_delay=1.0,
    ),
    WorkflowTask(
        name="write_report",
        agent="Writer",
        task="Write a report based on the AI safety research",
        dependencies=["research"],   # runs after "research"
        max_retries=1,
    ),
    WorkflowTask(
        name="review",
        agent="Reviewer",
        task="Review and critique the report",
        dependencies=["write_report"],
        parallelizable=False,
    ),
]

results = manager.run_workflow(tasks)
for name, data in results.items():
    print(f"{name}: {data['result'][:100]}")
```

### Dict-based workflow (simple)

```python
workflow = [
    {"name": "step1", "agent": "Researcher", "task": "Gather data"},
    {"name": "step2", "agent": "Analyst",    "task": "Analyze data", "parallelizable": True},
    {"name": "step3", "agent": "Analyst",    "task": "Verify data",  "parallelizable": True},
]
results = manager.run_workflow(workflow)
```

Workflow features:
- **Dependency enforcement** via topological sort
- **Retry logic** with configurable delays per task
- **Parallel execution** of tasks marked `parallelizable=True`
- **Conditional execution** via task conditions evaluated at runtime
- **Status tracking**: `PENDING`, `IN_PROGRESS`, `COMPLETED`, `FAILED`

## Logging

Maki does not configure logging automatically. Call `configure_logging()` explicitly in your application:

```python
import logging
from maki.logging_config import configure_logging

# Console only (default)
configure_logging()

# Console + file with custom level
configure_logging(log_level=logging.DEBUG, log_file_path="maki.log")
```

Log format:
```
2026-03-15 14:30:45,123 - module_name - LEVEL - Message
```

## Error Handling

```python
from maki.exceptions import MakiNetworkError, MakiTimeoutError, MakiAPIError

try:
    result = maki.request("Hello")
except MakiTimeoutError:
    print("Request timed out")
except MakiNetworkError:
    print("Network error")
except MakiAPIError:
    print("API returned an error")
```

Only `MakiNetworkError` and `MakiTimeoutError` are retried automatically by `execute_task_with_retry`. `MakiAPIError` and input validation errors (`ValueError`, `TypeError`) fail immediately.

## Examples

```bash
# Run the built-in demo
python -m maki

# MakiLLama demo
python -m maki.makiLLama
```

## License

MIT

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Create a new Pull Request

## Support

For support, please open an issue on the GitHub repository.
