# Installation

```bash
pip install maki-framework
```

The core install has three dependencies: `requests`, `httpx` and `python-dotenv`. Python 3.10 or newer is required.

## Extras

Backends and plugins with heavier dependencies are opt-in:

| Extra | Installs | Enables |
|---|---|---|
| `openai` | `openai` | OpenAI backend |
| `anthropic` | `anthropic` | Anthropic backend |
| `openrouter` | `openai` | OpenRouter backend |
| `llamacpp` | `openai` | llama.cpp backend |
| `vllm` | `openai` | vLLM backend |
| `distributed` | `fastapi`, `uvicorn`, `pyyaml` | `maki serve`, `AgentProxy` |
| `distributed-redis` | `redis` | Redis workflow checkpoints |
| `web` | `feedparser`, `readability-lxml`, `html2text` | `web_search`, `web_to_md`, `provider_updates` |
| `trends` | `pytrends` | `trend_search` |
| `alpaca` | `alpaca-py`, `feedparser`, `yfinance` | Alpaca plugins |
| `ftp` | `paramiko` | `ftp_client` |
| `structured` | `pydantic` | Structured output |
| `postgres` | `psycopg` | Postgres-backed stores |
| `all` | everything above | |

Combine extras with commas:

```bash
pip install "maki-framework[openai,distributed]"
```

## HuggingFace backend

`torch`, `transformers` and `accelerate` are not installed by any extra (they are large and platform-specific). See [Backends](backends.md#huggingface-transformers).

## From source

```bash
git clone https://github.com/BowlOfData/maki.git
cd maki
pip install -e ".[dev]"
pytest
```

## One-line installer

```bash
curl -fsSL https://raw.githubusercontent.com/BowlOfData/maki/main/install.sh | bash
```

This clones the repo into `~/.maki`, installs it into an isolated virtual environment, and symlinks the `maki` command into `~/.local/bin` (override with `MAKI_INSTALL_DIR` and `MAKI_BIN_DIR`). Re-run it to update to the latest `main`.
