<p align="center">
  <img src="https://raw.githubusercontent.com/BowlOfData/maki/main/img/logo.png" alt="Maki Logo" width="200" />
</p>

# Maki

[![PyPI](https://img.shields.io/pypi/v/maki-framework?logo=pypi&logoColor=white)](https://pypi.org/project/maki-framework/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/BowlOfData/maki/blob/main/LICENSE)
[![CI](https://github.com/BowlOfData/maki/actions/workflows/python-package.yml/badge.svg)](https://github.com/BowlOfData/maki/actions/workflows/python-package.yml)
[![Docs](https://img.shields.io/badge/docs-bowlofdata.github.io-informational)](https://bowlofdata.github.io/maki/docs/)

**Maki is a Python framework for multi-agent LLM applications that run wherever your models do: on your own hardware (through Ollama, llama.cpp or vLLM), on a hosted API (OpenAI, Anthropic, OpenRouter), or a mix of both, with the same agent code.**

It is for developers who want to build and run tool-using agents on local models first, keep the option of moving any task to a hosted model by swapping one object, and have guardrails switched on by default while those agents touch files, the web, or a broker account.

**Documentation: <https://bowlofdata.github.io/maki/docs/>**

## Quickstart

Install [Ollama](https://ollama.com/download), pull a small model, then install Maki:

```bash
ollama pull llama3.2
pip install maki-framework
```

The current release is [0.4.1](https://pypi.org/project/maki-framework/0.4.1/); to pin it, use `pip install maki-framework==0.4.1`.

```python
from maki import MakiLLama
from maki.agents import Agent

agent = Agent(
    name="Reviewer",
    maki_instance=MakiLLama(model="llama3.2"),
    role="code reviewer",
    instructions="Focus on bugs, regressions, and missing validation.",
)

print(agent.execute_task("Review this design: a plugin system with file access."))
```

The package is `maki-framework`; the import is `maki`. To run the same agent on a hosted model, swap the backend object:

```python
from maki import MakiAnthropic   # pip install "maki-framework[anthropic]"

agent = Agent(name="Reviewer", maki_instance=MakiAnthropic(model="claude-sonnet-4-5"), role="code reviewer")
```

Pick a backend from one string, and chain several so a transient failure (network error, timeout, rate limit) moves to the next one:

```python
from maki import FallbackBackend, init_backend

backend = FallbackBackend([
    init_backend("anthropic:claude-sonnet-4-5"),
    "openai:gpt-4o",          # spec strings are built with init_backend
    "ollama:gemma4:26b",      # local last resort
])
agent = Agent(name="Reviewer", maki_instance=backend, role="code reviewer")
```

## Why Maki

- **Local and hosted models are equals.** Ollama, llama.cpp, vLLM, OpenAI, Anthropic and OpenRouter sit behind the same `LLMBackend` contract, and an in-process HuggingFace backend is included. `init_backend("provider:model")` builds any of them from a string, `FallbackBackend` fails over between them (even in the middle of a tool loop), and one conformance suite pins identical behaviour and error mapping across all of them.
- **Guardrails on by default.** Content-derived URLs are checked against loopback, private and reserved address ranges at connect time. Plugins are fail-closed, and destructive methods (file writes, FTP transfers, trades) stay disabled until you pass `Agent(allow_dangerous_tools=True)`.
- **Small core.** Three libraries (`requests`, `httpx`, `python-dotenv`, plus `urllib3`, which `requests` already brings); everything else is an opt-in extra. Their minimum versions are the oldest with no published security advisory.
- **Agents as services.** `maki serve` exposes any agent over HTTP; `AgentProxy` consumes it remotely.
- **One tool loop for every provider.** Tool calls and results are typed `Message` objects that each backend converts to its own API, so a `FallbackBackend` can move a half-finished tool loop between providers, and with `MemoryPolicy(tool_history=True)` an agent remembers which tools it ran.
- **Know your model.** `python -m maki eval` runs your tasks through the whole agent on several local or hosted models and reports pass rate, time and tokens. maki also warns when a local server silently drops the start of a prompt that does not fit its context window, and `MakiLLama(num_ctx="auto")` sizes the window once instead of guessing.
- **Skills, loaded on demand.** Agents read Agent Skills (`SKILL.md` folders, the format Claude Code and Codex use) only when a task needs them, from `.agents/skills` and friends, with project skills opt-in.
- **Async-friendly.** `await agent.aexecute_task(...)`, `async for chunk in agent.astream_task(...)` and `await manager.agather_tasks([...])` run agents concurrently without blocking your event loop.
- **Typed, observable tools.** Native tool schemas come from type hints and docstrings, arguments are validated before a method runs, and `ToolPolicy` adds retries with backoff and optional parallel calls (destructive methods are never retried). `AgentCallback` hooks report task, model and tool events.
- **Batteries included.** 18 built-in plugins and a workflow engine with dependency resolution, retries, parallel execution, checkpoint/resume and maker/checker verification.
- **Agents that iterate.** `agent.run_loop(goal, requirements, judge)` keeps an agent working until a separate judge, your tests or your own checks confirm every requirement, with stall detection, token and time budgets, resume and optional human approval; it runs locally, over HTTP and inside workflows.
- **Memory that stays useful.** An opt-in `MemoryPolicy` shortens old turns, pins the facts that must never be lost and clips oversized tool output; an optional summary tier folds evicted turns into a running summary instead of dropping them (in the same live test it recalled 4/4 early facts where dropping recalled 0/4). In a live test on a local `gemma4:26b`, the default setup forgot a fact stated at the start of a 13-message chat while the policy setting answered correctly, with about 45% less text sent.


## Documentation

| | |
|---|---|
| [Quickstart](https://bowlofdata.github.io/maki/docs/quickstart/) | Install to first agent in 5 minutes |
| [Installation and extras](https://bowlofdata.github.io/maki/docs/installation/) | Optional dependencies per backend and plugin |
| [Backends](https://bowlofdata.github.io/maki/docs/backends/) | Ollama, llama.cpp, vLLM, OpenAI, Anthropic, OpenRouter, HuggingFace |
| [Agents](https://bowlofdata.github.io/maki/docs/agents/) | Memory, reasoning, streaming, tools |
| [Workflows](https://bowlofdata.github.io/maki/docs/workflows/) | Multi-agent pipelines and verified tasks |
| [Agent memory](https://bowlofdata.github.io/maki/docs/agent-memory/) | Keep long conversations correct and prompts small |
| [Evaluating models](https://bowlofdata.github.io/maki/docs/eval/) | Compare models on your own tasks with `maki eval` |
| [Skills](https://bowlofdata.github.io/maki/docs/skills/) | Load Agent Skills on demand (shared with Claude Code and Codex) |
| [Agent loops](https://bowlofdata.github.io/maki/docs/loops/) | Iterate an agent until its requirements are met |
| [Plugins](https://bowlofdata.github.io/maki/docs/plugins/) | Built-in plugins and the safety model |
| [Serving agents](https://bowlofdata.github.io/maki/docs/serving/) | `maki serve` and `AgentProxy` |
| [Configuration](https://bowlofdata.github.io/maki/docs/configuration/) | Environment variables |

## Contributing

Bug fixes, documentation, new plugins and suggestions are welcome. Open an issue or pull request on [GitHub](https://github.com/BowlOfData/maki). Maki is part of [Bowl of Data](https://bowlofdata.net/), an open-source AI research community.

## License

[MIT](https://github.com/BowlOfData/maki/blob/main/LICENSE)
