# Configuration

Shared runtime defaults live in `maki/config.py`. Every value can be overridden with an environment variable or a `.env` file (`python-dotenv` is supported).

!!! note "Where a `.env` file is found"
    The `.env` file is searched **upward from where the `maki` package is installed**, not from the directory you run your program in, and real environment variables always win over it. It works as expected in a source checkout (`pip install -e .`); for an installed or deployed instance set real environment variables instead. Agent YAML files have no key fields: secrets are read from the environment only.

| Variable | Description |
|---|---|
| `MAKI_OLLAMA_BASE_URL` | Full Ollama base URL |
| `MAKI_OLLAMA_HOST` | Ollama hostname |
| `MAKI_OLLAMA_PORT` | Ollama port |
| `MAKI_LLAMACPP_BASE_URL` | llama.cpp server URL (default `http://127.0.0.1:8080/v1`) |
| `MAKI_LLAMACPP_MODEL` | llama.cpp model name (default `local-model`; set it in router mode) |
| `MAKI_VLLM_BASE_URL` | vLLM server URL (default `http://127.0.0.1:8000/v1`) |
| `MAKI_VLLM_MODEL` | vLLM model name (default `Qwen/Qwen2.5-1.5B-Instruct`, the vLLM quickstart model); must match the served name |
| `MAKI_API_KEY` | Bearer token **your `maki serve` instance requires from its callers**. You invent it; it is not a model-provider key (see [Running in production](production.md#api-keys-which-one-is-which)) |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY` | Provider keys for the hosted backends; create them in the provider's dashboard |
| `VLLM_API_KEY` | vLLM API key, only if the server was started with `--api-key` |
| `LLAMACPP_API_KEY` | llama.cpp API key, only if the server was started with `--api-key` |
| `MAKI_DEFAULT_MODEL` | Default model name |
| `MAKI_RATE_LIMIT_RETRIES` | Times an HTTP backend (Ollama, llama.cpp, vLLM) retries a 429 response (default `2`, `0` disables) |
| `MAKI_RATE_LIMIT_MAX_WAIT` | Longest `Retry-After` honoured for a 429, in seconds (default `30`); a longer request is raised at once |
| `MAKI_DEFAULT_TEMPERATURE` | Sampling temperature |
| `MAKI_REQUEST_TIMEOUT` | Per-request timeout (seconds) |
| `MAKI_HTTP_TIMEOUT` | Low-level HTTP timeout |
| `MAKI_LOG_LEVEL` | Logging level |
| `MAKI_WEB_USER_AGENT` | User-agent string for web plugins |

Example `.env` for an Ollama server on another machine on your LAN:

```bash
MAKI_OLLAMA_BASE_URL=http://192.168.1.20:11434
MAKI_REQUEST_TIMEOUT=300
```

## Logging

Maki does not configure logging on import. Call it explicitly:

```python
from maki.logging_config import configure_logging

configure_logging()
```
