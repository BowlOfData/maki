# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- RAG support and initial refactoring toward a provider-agnostic architecture
- Equity market data support
- Forex market data support
- Optional-dependency extras: `gui`, `ftp`, `web`, `trends`, `alpaca`, `distributed-redis`, `all`

### Changed
- **Security**: `Connector` reworked into the single hardened HTTP layer for the whole framework — it owns URL validation, DNS resolution with resolved-IP validation and connect-time pinning (a hostname whose records include any private/reserved address is rejected, and the socket connects to the exact validated IP, closing the DNS-rebinding window; TLS still verifies against the original hostname), timeouts from `maki.config` (the hardcoded 180 s is gone), and the mapping of transport/status failures onto the Maki exception tree (transport timeout & HTTP 408/504 → `MakiTimeoutError`, connection failures & 5xx → `MakiNetworkError`, other 4xx → `MakiAPIError`). A new `AsyncConnector` (httpx) covers the async paths with pre-flight validation
- All outbound HTTP now routes through the hardened layer: `MakiLLama`, `AgentProxy`, and the `web_to_md`, `web_search`, `provider_updates`, `media_search`, and `rag_memory` plugins no longer make raw `requests`/`httpx` calls. Operator-configured endpoints (Ollama base URL, registered remote agents, the RAG embedder) use `allow_private=True` so LAN/loopback deployments keep working; content-derived URLs (web plugins) get the full SSRF treatment
- Connection pinning falls back through the validated address list, so dual-stack hosts whose service listens on one address family only (e.g. `localhost` → `::1` first, server on IPv4) still connect — every attempted address has passed validation
- `MakiLLama` HTTP error classes changed with the unified mapping: 5xx responses now raise the retryable `MakiNetworkError` (previously `MakiAPIError`), and `stream()` failures are classified instead of escaping as raw `requests` exceptions
- `AgentProxy` no longer requires `httpx` (the distributed client now rides on the core `requests`-based layer)
- `Utils.PRIVATE_IP_RANGES`: removed `128.0.0.0/16` (allocated public space, wrongly blocked for decades' worth of hosts) and corrected the TEST-NET labels (`198.18.0.0/15` is the RFC 2544 benchmarking range, not TEST-NET)
- `web_to_md` retries on exception *types* (`MakiNetworkError`) instead of matching exception-name substrings
- **Breaking**: `AgentServer` `/stream` is now `POST` with the same body as `/execute`, so task text stays out of access logs and URL length limits no longer apply; `AgentProxy` updated to match, and its `context` argument — previously dropped by the GET query string — is now forwarded
- **Security**: `AgentServer` API-key comparison is constant-time (`secrets.compare_digest`)
- **Security**: `/execute` no longer echoes internal error details to remote callers — validation errors (`MakiValidationError`/`ValueError`) return 400 with the validation message, backend timeouts return 504 (which `AgentProxy` maps back to `MakiTimeoutError`), network failures 502, and everything else a generic 400/500 body with the full exception logged server-side only; the `/stream` SSE error event is sanitized the same way
- `GET /health` no longer requires the API key, so load-balancer/k8s probes work against authenticated servers
- YAML agent configs: plugin names are validated against `maki.plugins.PLUGIN_REGISTRY` (previously any string reached `importlib`), and served agents keep dangerous tool methods disabled unless the config sets the new `allow_dangerous_tools: true` field
- **Breaking / security**: plugin tool calls are now **fail-closed** — a plugin exposes only the methods named in its class-level `ALLOWED_METHODS`; a plugin without one exposes nothing to `TOOL:` directives (a warning is logged at load time). Destructive methods listed in a plugin's `DANGEROUS_METHODS` (file writes, FTP uploads/downloads/deletes, trade submission/cancellation) additionally require `Agent(allow_dangerous_tools=True)`
- `file_reader`, `file_writer`, `directory_reader`, `ftp_client`, and `web_to_md` now declare `ALLOWED_METHODS`; `file_writer`, `ftp_client`, and `alpaca_trading` mark their destructive methods in `DANGEROUS_METHODS`
- **Packaging**: `pyproject.toml` is the single manifest — `setup.py` and `requirements.txt` removed; version bumped to 0.2.0
- **Breaking**: `PySide6` and `paramiko` are no longer core dependencies — install `maki[gui]` / `maki[ftp]`; core install is just `requests`, `httpx`, `python-dotenv`
- `requires-python` raised to `>=3.10` (the code already used 3.9+/3.10+ syntax); CI matrix now 3.10–3.13
- Plan documents moved from the repo root to `docs/`

### Fixed
- `MakiLLama.chat()` computed `elapsed` before checking the response status, and a 200 response with a non-JSON body leaked a raw `json.JSONDecodeError`; both now go through the hardened layer (`MakiAPIError` for invalid JSON)
- `rag_memory._ollama_embed` called `Connector.post()`, a method that did not exist (the old `Connector` only had `simple`/`version`) — embedding via Ollama would have crashed with `AttributeError`; the reworked instance API makes the call real
- `LocalStateStore` checkpoint writes are now atomic (temp file + `os.replace`) — a crash mid-save no longer leaves a corrupt checkpoint, which is the scenario checkpointing exists for; applies to both `save_workflow` and `update_task`
- Workflow IDs that sanitize to the same filename/Redis key (e.g. `a/b` and `a_b`) no longer overwrite each other's state — sanitized IDs get a short hash suffix (IDs that need no sanitization are unchanged)
- `MakiLLama.pull()` crashed with `TypeError` on the first progress chunk (`log.info(..., end="\r")`); progress is now logged at ~10% intervals
- `AgentProxy` streaming raised `httpx.ResponseNotRead` on non-2xx responses instead of the mapped Maki error, and the circuit breaker never recorded the failure
- `Utils.convert64` rejected any path under a symlinked directory (e.g. `/tmp` on macOS); now resolves symlinks and optionally enforces containment via a new `allowed_dirs` parameter. It also returns a base64 `str` (as documented) instead of `bytes`
- `MakiLLama.__call__` silently dropped the `system` and `images` kwargs
- `HFBackend.stream()` ignored the configured `GenerationConfig` and streamed with defaults
- `Utils.cleanup_response` kept no reference to the scheduled `aclose()` task, so the cleanup could be garbage-collected before running
- `PluginHandler.load_plugin` fell through to calling a module object (always `TypeError`); now raises a clear `MakiValidationError` naming the plugin
- `ChatSession` streaming lost both turns of history when the consumer abandoned the stream mid-way
- Ten plugins (`trend_search`, `web_search`, `provider_updates`, `media_search`, `rag_memory`, `obsidian_memory`, and the four `alpaca_*` plugins) declared `ALLOWED_METHODS` only at module level, where tool-call validation (which reads the plugin instance) never saw it — their whitelists were silently ineffective. The lists are now mirrored onto the plugin classes

### Removed
- Stale root-level files: `local_llm.py`, `local_llm_v2.py`, `orchestrator.py`, `review.md`, `examples/demo_implementation.py` (all referenced the deleted `Maki` class or were pre-package copies)
- `Utils.compose_url` (110 lines of URL sanitisation serving no remaining caller), `Utils._validate_port`, and `maki/urls.py` (`Actions` enum, `GENERIC_LLAMA_URL`) — dead since the `Maki` class was removed; the hardened `Connector` is the one place URLs are validated now
- `Connector.simple()` / `Connector.version()` static methods (unused; replaced by the instance `get`/`post`/`delete` API)

---

## [0.1.0] — 2025

### Added
- **Agent system** — `Agent` class with task execution, memory (remember/recall/clear), conversation history, plugin support, and step-by-step reasoning via `PluginHandler` and `ReasoningEngine` mixins
- **AgentManager** — orchestrates multiple agents with `assign_task`, `coordinate_agents`, `collaborative_task`, and `run_workflow`
- **Workflow engine** — `WorkflowTask` / `WorkflowState` with topological sort, retry logic, and optional parallel execution
- **Ollama backends** — `Maki` (single-turn generate API) and `MakiLLama` (full chat API with streaming, async, multi-turn sessions); convenience factory functions `gemma3()`, `qwen()`, `llama()`, `mistral()`
- **HuggingFace backend** — `HFBackend` for direct Transformers integration with quantization and device selection (no Ollama required)
- **Plugin system** — plugins triggered by `TOOL:` directives in LLM output; built-in plugins: `file_reader`, `file_writer`, `directory_reader`, `web_to_md`, `ftp_client`
- **Alpaca trading support** — integration for brokerage operations
- **OCR plugin** — image-to-text via external OCR tooling
- **Infrastructure layer** — `Connector` HTTP client with SSRF protection, error classification, timeout handling, and rate limiting
- **Shared data classes** — `Message`, `GenerationConfig`, `LLMResponse` in `objects.py`
- **Custom exception hierarchy** — `MakiNetworkError`, `MakiTimeoutError`, `MakiAPIError`, `MakiValidationError`
- **Lazy loading** — `__init__.py` defers module imports until first access
- **Logging** — `configure_logging()` in `logging_config.py`; not auto-configured on import

### Security
- SSRF protection on all outbound URLs (loopback allowed for local Ollama)
- FTP/SFTP remote path traversal fixes
- Plugin path injection hardening
- Robust LLM output parsing to prevent injection via model responses
- Connection pooling and rate limiting added to HTTP layer
- Bounded conversation history using `deque(maxlen=1000)`

### Fixed
- Backend return type consistency across all LLM backends
- Mixin contracts enforced between `Agent`, `PluginHandler`, and `ReasoningEngine`
- Temperature validation and payload construction
- Workflow dependency resolution and parallel execution race conditions
- Undefined logger reference in `utils.py`
- ChatSession system prompt mutation bug
- FTP absolute path injection
