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

## Why Maki

- **Local and hosted models are equals.** Ollama, llama.cpp, vLLM, OpenAI, Anthropic and OpenRouter sit behind the same `LLMBackend` contract, and an in-process HuggingFace backend is included.
- **Guardrails on by default.** Content-derived URLs are checked against private and reserved address ranges at connect time. Plugins are fail-closed, and destructive methods (file writes, FTP transfers, trades) stay disabled until you pass `Agent(allow_dangerous_tools=True)`.
- **Small core.** Three dependencies (`requests`, `httpx`, `python-dotenv`); everything else is an opt-in extra.
- **Agents as services.** `maki serve` exposes any agent over HTTP; `AgentProxy` consumes it remotely.
- **Batteries included.** 18 built-in plugins and a workflow engine with dependency resolution, retries, parallel execution, checkpoint/resume and maker/checker verification.
- **Agents that iterate.** `agent.run_loop(goal, requirements, judge)` keeps an agent working until a separate judge, your tests or your own checks confirm every requirement, with stall detection, token and time budgets, resume and optional human approval; it runs locally, over HTTP and inside workflows.
- **Memory that stays useful.** An opt-in `MemoryPolicy` shortens old turns, pins the facts that must never be lost and clips oversized tool output. In a live test on a local `gemma4:26b`, the default setup forgot a fact stated at the start of a 13-message chat while the policy setting answered correctly, with about 45% less text sent.


## Documentation

| | |
|---|---|
| [Quickstart](https://bowlofdata.github.io/maki/docs/quickstart/) | Install to first agent in 5 minutes |
| [Installation and extras](https://bowlofdata.github.io/maki/docs/installation/) | Optional dependencies per backend and plugin |
| [Backends](https://bowlofdata.github.io/maki/docs/backends/) | Ollama, llama.cpp, vLLM, OpenAI, Anthropic, OpenRouter, HuggingFace |
| [Agents](https://bowlofdata.github.io/maki/docs/agents/) | Memory, reasoning, streaming, tools |
| [Workflows](https://bowlofdata.github.io/maki/docs/workflows/) | Multi-agent pipelines and verified tasks |
| [Agent memory](https://bowlofdata.github.io/maki/docs/agent-memory/) | Keep long conversations correct and prompts small |
| [Agent loops](https://bowlofdata.github.io/maki/docs/loops/) | Iterate an agent until its requirements are met |
| [Plugins](https://bowlofdata.github.io/maki/docs/plugins/) | Built-in plugins and the safety model |
| [Serving agents](https://bowlofdata.github.io/maki/docs/serving/) | `maki serve` and `AgentProxy` |
| [Configuration](https://bowlofdata.github.io/maki/docs/configuration/) | Environment variables |

## Contributing

Bug fixes, documentation, new plugins and suggestions are welcome. Open an issue or pull request on [GitHub](https://github.com/BowlOfData/maki). Maki is part of [Bowl of Data](https://bowlofdata.net/), an open-source AI research community.

## License

[MIT](https://github.com/BowlOfData/maki/blob/main/LICENSE)
