# Maki

**Maki is a Python framework for multi-agent LLM applications that run wherever your models do:** on your own hardware (through Ollama, llama.cpp or vLLM), on a hosted API (OpenAI, Anthropic, OpenRouter), or a mix of both, with the same agent code.

It is for developers who want to build tool-using agents on local models first, keep the option of moving any task to a hosted model by swapping one object, and have guardrails on by default while those agents touch files, the web, or a broker account.

```bash
pip install maki-framework
```

[Get started in 5 minutes :material-arrow-right:](quickstart.md){ .md-button .md-button--primary }

## Why Maki

- **Local and hosted models are equals.** Ollama, llama.cpp and vLLM sit next to OpenAI, Anthropic and OpenRouter behind the same `LLMBackend` contract. Agents, workflows and plugins never know which one they are talking to.
- **Guardrails on by default.** Outbound requests go through a hardened connector that blocks private and reserved address ranges for content-derived URLs. Plugins are fail-closed: a model can only call methods a plugin explicitly lists, and destructive ones stay disabled until you pass `Agent(allow_dangerous_tools=True)`.
- **Small core.** The base install has three dependencies (`requests`, `httpx`, `python-dotenv`). Everything else is an opt-in extra.
- **Agents as services.** `maki serve` exposes any agent over HTTP, and `AgentProxy` lets another process use it as if it were local.
- **Batteries included.** 18 built-in plugins and a workflow engine with dependency resolution, retries, parallel execution and checkpoint/resume.
- **Agents that iterate.** Give an agent a goal and requirements and let it loop until a judge or your own checks are satisfied; see [Agent loops](loops.md).
- **Memory that stays useful.** An opt-in `MemoryPolicy` shortens old turns, pins the facts that must never be lost and clips oversized tool output, so long conversations keep giving correct answers; see [Agent memory](agent-memory.md).

!!! note "Who it is not for"
    Maki is young (0.x) and deliberately small. If you need a large catalog of third-party integrations, hosted tracing, or a big ecosystem, one of the larger agent frameworks will serve you better.

## Where to go next

| I want to... | Read |
|---|---|
| Run my first agent | [Quickstart](quickstart.md) |
| Use OpenAI, Anthropic, llama.cpp, vLLM or HuggingFace | [Backends](backends.md) |
| Give an agent memory, streaming or tools | [Agents](agents.md) |
| Chain several agents together | [Workflows](workflows.md) |
| Let an agent iterate until requirements are met | [Agent loops](loops.md) |
| Let an agent read files or search the web | [Plugins](plugins.md) |
| Keep a long-running agent's answers correct and its prompts small | [Agent memory](agent-memory.md) |
| Give an agent a notes vault or a searchable knowledge base | [Memory: Obsidian and RAG](memory.md) |
| Expose an agent over HTTP | [Serving agents](serving.md) |
