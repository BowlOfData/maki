# Quickstart

Five minutes from `pip install` to a working agent running on a local model.

!!! info "Package name vs import name"
    You install `maki-framework` but import `maki`.

## 1. Install Ollama and pull a small model

Install [Ollama](https://ollama.com/download) and start it (the desktop app starts the server for you; otherwise run `ollama serve`). Then pull a small model:

```bash
ollama pull llama3.2
```

Maki talks to Ollama at `http://localhost:11434` by default. See [Configuration](configuration.md) to point it elsewhere.

## 2. Install Maki

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install maki-framework
```

Requires Python 3.10 or newer.

## 3. Ask the model a question

```python
from maki import MakiLLama

llm = MakiLLama(model="llama3.2")
response = llm.chat("Explain recursion in one sentence.")
print(response.content)
```

## 4. Run your first agent

```python
from maki import MakiLLama
from maki.agents import Agent

llm = MakiLLama(model="llama3.2")
agent = Agent(
    name="Reviewer",
    maki_instance=llm,
    role="code reviewer",
    instructions="Focus on bugs, regressions, and missing validation.",
)

print(agent.execute_task("Review this design: a plugin system with file access."))
```

## 5. Swap the backend

The same agent runs on a hosted model by changing one object:

```python
from maki import MakiAnthropic   # pip install "maki-framework[anthropic]"

llm = MakiAnthropic(model="claude-sonnet-4-5")
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| Connection error to `localhost:11434` | Ollama is not running. Start the app or run `ollama serve`. |
| `model not found` | Run `ollama pull <model>` for the exact name you passed. |
| Request times out on a long answer | Use `Agent(use_streaming=True)`; the timeout then applies per chunk. See [Agents](agents.md). |
| `ImportError` mentioning `maki[...]` | Install the extra it names, e.g. `pip install "maki-framework[openai]"`. |

## Next steps

- [Backends](backends.md): llama.cpp, vLLM, OpenAI, Anthropic, OpenRouter, HuggingFace
- [Agents](agents.md): memory, reasoning, streaming
- [Workflows](workflows.md): multi-agent pipelines
