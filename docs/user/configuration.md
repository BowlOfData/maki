# Configuration

Shared runtime defaults live in `maki/config.py`. Every value can be overridden with an environment variable or a `.env` file (`python-dotenv` is supported).

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
